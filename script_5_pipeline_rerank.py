"""
=====================================================================
 SCRIPT 5: PIPELINE RAG CON RE-RANKING (FASE 2)
=====================================================================

Aplica un cross-encoder re-ranker sobre las TOP-3 combinaciones del
ranking principal del experimento base (script 1):

    1. B_markdown + semantica_experto
    2. A_fixed    + semantica_experto
    3. C_semantica + semantica_generalista

Flujo del pipeline modificado:
    1. Recuperación: el retriever base devuelve N=20 candidatos (en vez de 5)
    2. Re-ranking: el cross-encoder BAAI/bge-reranker-v2-m3 puntúa cada
       (pregunta, chunk) y reordena
    3. Top-K: nos quedamos con los top 5 finales tras el reordenamiento
    4. Generación: idéntica al pipeline base con esos top 5

Para cada ejecución guardamos:
    - chunks recuperados originalmente (top-N=20 con sus scores)
    - chunks finales tras re-ranker (top-5 con sus scores nuevos)
    - respuesta generada
    - latencias separadas: retrieval, rerank, generación

Diseñado para ser idempotente: checkpoints incrementales con misma lógica
de clave que script 1, ampliada con 'rerank'.

Input:
    - dataset_submuestreado.json (160 preguntas)
    - Elasticsearch con los 4 índices ya poblados

Output:
    - resultados_rerank.json (incremental)

Coste estimado: ~1.5-2.5h en GPU 4090 (480 ejecuciones × ~15-20s/ejec)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════

# Archivos
ARCHIVO_DATASET = 'dataset_submuestreado.json'
ARCHIVO_RESULTADOS = 'resultados_rerank.json'

# Elasticsearch
ES_HOST = 'http://localhost:9201'

# Parámetros de recuperación con re-ranking
TOP_N_CANDIDATOS = 20  # cuántos chunks recupera el retriever base
TOP_K_FINAL = 5        # cuántos chunks pasamos al LLM tras re-ranking
NUM_CANDIDATES_KNN = 100  # parámetro interno de Elasticsearch para KNN

# Modelos
MODELO_GENERALISTA = 'intfloat/multilingual-e5-base'
MODELO_EXPERTO_PATH = './modelo_especializado_e5'
MODELO_LLM = 'Qwen/Qwen2.5-7B-Instruct'
MODELO_RERANKER = 'BAAI/bge-reranker-v2-m3'

# Mapa chunking → índice en ES
INDICES_POR_CHUNKING = {
    'A_fixed': 'indice_estrategia_a_fixed',
    'B_markdown': 'indice_estrategia_b_markdown',
    'C_semantica': 'indice_estrategia_c_semantica',
}

# ─── COMBINACIONES A EVALUAR (top 3 del ranking principal) ───
COMBINACIONES_TOP3 = [
    ('B_markdown',  'semantica_experto'),
    ('A_fixed',     'semantica_experto'),
    ('C_semantica', 'semantica_generalista'),
]

# Generación LLM (idénticos al script 1)
LLM_MAX_CHARS_CONTEXTO = 150000
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURA = 0.1

# ═══════════════════════════════════════════════════════════════════
# CARGA DE MODELOS (una sola vez)
# ═══════════════════════════════════════════════════════════════════

print("═" * 70)
print(" PIPELINE FASE 2: RE-RANKING SOBRE TOP-3 COMBINACIONES")
print("═" * 70)

# Elasticsearch
es = Elasticsearch(ES_HOST, request_timeout=60, max_retries=3, retry_on_timeout=True)
assert es.ping(), 'ERROR: Elasticsearch no disponible'
print(f" ✓ Conectado a Elasticsearch en {ES_HOST}")

# Embedders (solo los que usan las top-3: generalista y experto)
print(" - Cargando embedder generalista (E5)...")
modelo_generalista = SentenceTransformer(MODELO_GENERALISTA, device='cuda')

print(" - Cargando embedder experto (E5 fine-tuneado)...")
modelo_experto = SentenceTransformer(MODELO_EXPERTO_PATH, device='cuda')

# Re-ranker (cross-encoder)
print(f" - Cargando re-ranker ({MODELO_RERANKER})...")
reranker = CrossEncoder(MODELO_RERANKER, device='cuda', max_length=512)
print(f"   Re-ranker listo en GPU. max_length=512")

# LLM generador
print(f" - Cargando LLM generador ({MODELO_LLM})...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM)
modelo_llm = AutoModelForCausalLM.from_pretrained(
    MODELO_LLM,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
)
pipe_llm = pipeline(
    'text-generation',
    model=modelo_llm,
    tokenizer=tokenizer,
    return_full_text=False,
)
print(" ✓ Todos los modelos cargados")
print()


# ═══════════════════════════════════════════════════════════════════
# FUNCIONES DE RETRIEVAL (idénticas al script 1, ampliadas a top_n)
# ═══════════════════════════════════════════════════════════════════

def _extraer_info_hits(hits, es_jerarquico, nombre_indice):
    """Extrae chunk_id, texto, metadata y rank de los hits de ES."""
    if not es_jerarquico:
        return [
            {
                'chunk_id': h['_source'].get('chunk_id'),
                'text': h['_source'].get('text', ''),
                'metadata': h['_source'].get('metadata', {}),
                'rank': rank + 1,
                'score_retriever': h.get('_score', None),
            }
            for rank, h in enumerate(hits)
        ]
    # En jerárquico no hay rerank: solo se aplica a A/B/C, así que esta rama
    # no se ejecuta en este script. Se deja por consistencia.
    return [
        {
            'chunk_id': h['_source'].get('chunk_id'),
            'text': h['_source'].get('text', ''),
            'metadata': h['_source'].get('metadata', {}),
            'rank': rank + 1,
            'score_retriever': h.get('_score', None),
        }
        for rank, h in enumerate(hits)
    ]


def retrieve_semantica(pregunta, nombre_indice, tipo_modelo, top_n):
    """Igual que en script 1, pero parametrizado por top_n (no fijo a TOP_K=5)."""
    if tipo_modelo == 'generalista':
        vector = modelo_generalista.encode(f'query: {pregunta}').tolist()
        campo = 'vector_generalista'
    elif tipo_modelo == 'experto':
        vector = modelo_experto.encode(f'query: {pregunta}').tolist()
        campo = 'vector_experto'
    else:
        raise ValueError(f'tipo_modelo no soportado en este script: {tipo_modelo}')

    body = {
        'knn': {
            'field': campo,
            'query_vector': vector,
            'k': top_n,
            'num_candidates': NUM_CANDIDATES_KNN,
        },
        '_source': ['text', 'chunk_id', 'metadata'],
    }
    resp = es.search(index=nombre_indice, body=body)
    return _extraer_info_hits(resp['hits']['hits'], False, nombre_indice)


def ejecutar_retrieval_top_n(pregunta, chunking, retrieval, top_n):
    """Despacha al retrieval correspondiente. Solo soporta semánticas porque
    las top-3 combinaciones todas usan semantica_experto o semantica_generalista."""
    indice = INDICES_POR_CHUNKING[chunking]
    if retrieval.startswith('semantica_'):
        tipo = retrieval.replace('semantica_', '')
        return retrieve_semantica(pregunta, indice, tipo, top_n)
    raise ValueError(f"retrieval no soportado: {retrieval}. "
                     f"Este script solo procesa las top-3: {COMBINACIONES_TOP3}")


# ═══════════════════════════════════════════════════════════════════
# RE-RANKING
# ═══════════════════════════════════════════════════════════════════

def aplicar_reranker(pregunta, chunks_top_n, top_k_final):
    """
    Reordena los chunks usando un cross-encoder y devuelve los top_k_final.

    El cross-encoder recibe pares (pregunta, texto_chunk) y devuelve un score
    de relevancia. Reordenamos según ese score y nos quedamos con los k mejores.

    Devuelve:
        chunks_finales: lista con los top_k_final, anotados con score_rerank
                        y nuevo rank (1 a top_k_final).
    """
    if not chunks_top_n:
        return []

    pares = [(pregunta, ch['text']) for ch in chunks_top_n]
    scores = reranker.predict(pares, batch_size=16, show_progress_bar=False)

    # Anotar cada chunk con su score de rerank (sin perder el del retriever)
    for ch, sc in zip(chunks_top_n, scores):
        ch['score_rerank'] = float(sc)

    # Ordenar por score_rerank descendente
    reordenados = sorted(chunks_top_n, key=lambda x: x['score_rerank'], reverse=True)

    # Quedarnos con los top_k_final y reasignar rank (1..k)
    finales = reordenados[:top_k_final]
    for nuevo_rank, ch in enumerate(finales, 1):
        ch['rank_rerank'] = nuevo_rank

    return finales


# ═══════════════════════════════════════════════════════════════════
# GENERACIÓN (idéntica al script 1)
# ═══════════════════════════════════════════════════════════════════

INSTRUCCIONES_SISTEMA = (
    'Eres un asistente médico experto y preciso. Tu tarea es responder a la '
    'pregunta del usuario utilizando ÚNICAMENTE la información proporcionada '
    'en el contexto. Si el contexto no contiene información suficiente para '
    'responder a la pregunta, debes indicar claramente que no tienes '
    'suficientes datos para responder. No inventes información.'
)


def generar_respuesta(pregunta, chunks_recuperados):
    """Construye prompt y llama al LLM. Mismo comportamiento que script 1."""
    contextos_usados = []
    chars = 0
    truncado = False
    for ch in chunks_recuperados:
        texto = ch['text']
        if chars + len(texto) <= LLM_MAX_CHARS_CONTEXTO:
            contextos_usados.append(texto)
            chars += len(texto)
        else:
            truncado = True
            break

    if truncado:
        n_omitidos = len(chunks_recuperados) - len(contextos_usados)
        print(f'   [!! WARNING] Truncado de contexto activado: {n_omitidos} chunk(s) omitidos.')

    contexto_unido = '\n\n---\n\n'.join(contextos_usados)
    mensajes = [
        {'role': 'system', 'content': INSTRUCCIONES_SISTEMA},
        {'role': 'user',
         'content': f'Contexto recuperado:\n{contexto_unido}\n\nPregunta: {pregunta}'},
    ]

    resultados = pipe_llm(
        mensajes,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        max_length=None,
        temperature=LLM_TEMPERATURA,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if isinstance(resultados[0]['generated_text'], list):
        respuesta = resultados[0]['generated_text'][-1]['content']
    else:
        respuesta = resultados[0]['generated_text']
    return respuesta, contextos_usados


# ═══════════════════════════════════════════════════════════════════
# CHECKPOINTS (misma lógica que script 1)
# ═══════════════════════════════════════════════════════════════════

def cargar_resultados_previos():
    if not os.path.exists(ARCHIVO_RESULTADOS):
        return []
    with open(ARCHIVO_RESULTADOS, 'r', encoding='utf-8') as f:
        return json.load(f)


def guardar_resultados(resultados):
    with open(ARCHIVO_RESULTADOS, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)


def clave_ejecucion(pregunta_idx, chunking, retrieval):
    """Misma clave que script 1. No incluye 'rerank' porque este JSON es
    independiente: si existe un registro aquí, ya es con rerank."""
    return f'{pregunta_idx}__{chunking}__{retrieval}'


# ═══════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main():
    # Cargar dataset
    with open(ARCHIVO_DATASET, 'r', encoding='utf-8') as f:
        preguntas = json.load(f)

    # Cargar checkpoint
    resultados = cargar_resultados_previos()
    hechos = {
        clave_ejecucion(r['pregunta_idx'], r['chunking'], r['retrieval'])
        for r in resultados
    }

    total = len(preguntas) * len(COMBINACIONES_TOP3)
    print(f" • Preguntas:                 {len(preguntas)}")
    print(f" • Combinaciones top-3:       {len(COMBINACIONES_TOP3)}")
    print(f" • Total ejecuciones:         {total}")
    print(f" • Ya hechas (checkpoint):    {len(hechos)}")
    print(f" • Pendientes:                {total - len(hechos)}")
    print(f" • N candidatos → top K:      {TOP_N_CANDIDATOS} → {TOP_K_FINAL}")
    print(f" • Re-ranker:                 {MODELO_RERANKER}")
    print()

    if len(hechos) >= total:
        print(" ✓ Todo ya estaba hecho. Nada que procesar.")
        return

    t_inicio = time.time()
    contador = len(hechos)

    for p_idx, pregunta in enumerate(preguntas):
        for chunking, retrieval in COMBINACIONES_TOP3:
            clave = clave_ejecucion(p_idx, chunking, retrieval)
            if clave in hechos:
                continue

            elapsed = (time.time() - t_inicio) / 60
            velocidad = (contador - len(hechos) + 1) / max(elapsed, 0.01)
            restantes = total - contador
            eta_min = restantes / max(velocidad, 0.1)
            print(f' [{contador+1}/{total} | {elapsed:.1f}min | '
                  f'{velocidad:.1f}/min | ETA {eta_min:.0f}min] '
                  f'P{p_idx} | {chunking} | {retrieval}')

            try:
                # FASE 1: retrieval con N candidatos
                t1 = time.perf_counter()
                chunks_top_n = ejecutar_retrieval_top_n(
                    pregunta['pregunta'], chunking, retrieval, TOP_N_CANDIDATOS
                )
                lat_retrieval_ms = (time.perf_counter() - t1) * 1000

                # FASE 2: re-ranking con cross-encoder
                t2 = time.perf_counter()
                chunks_finales = aplicar_reranker(
                    pregunta['pregunta'], chunks_top_n, TOP_K_FINAL
                )
                lat_rerank_ms = (time.perf_counter() - t2) * 1000

                # FASE 3: generación con los top-K re-rankeados
                t3 = time.perf_counter()
                respuesta, contextos_usados = generar_respuesta(
                    pregunta['pregunta'], chunks_finales
                )
                lat_gen_ms = (time.perf_counter() - t3) * 1000

                chars_contexto = sum(len(t) for t in contextos_usados)
                tokens_contexto_aprox = len(tokenizer.encode(
                    '\n\n---\n\n'.join(contextos_usados)
                ))

                # Guardar resultado completo (con trazabilidad de ambos rankings)
                resultado = {
                    'pregunta_idx': p_idx,
                    'pregunta': pregunta['pregunta'],
                    'tipo': pregunta['tipo'],
                    'source': pregunta['source'],
                    'macro_id_origen': pregunta['macro_id_origen'],
                    'respuesta_ideal': pregunta['respuesta_ideal'],
                    'cita_literal': pregunta['cita_literal'],
                    'chunking': chunking,
                    'retrieval': retrieval,
                    'rerank': True,
                    'top_n_candidatos': TOP_N_CANDIDATOS,
                    'top_k_final': TOP_K_FINAL,
                    # Top-N original del retriever (con scores) para auditoría
                    'chunks_top_n_inicial': chunks_top_n,
                    # Top-K final tras rerank (lo que ve el LLM y se evalúa con IR/RAGAS)
                    'chunks_recuperados': chunks_finales,
                    'respuesta_generada': respuesta,
                    'n_contextos_usados': len(contextos_usados),
                    # Eficiencia con desglose por fase
                    'latencia_retrieval_ms': round(lat_retrieval_ms, 2),
                    'latencia_rerank_ms': round(lat_rerank_ms, 2),
                    'latencia_generacion_ms': round(lat_gen_ms, 2),
                    'latencia_total_ms': round(
                        lat_retrieval_ms + lat_rerank_ms + lat_gen_ms, 2),
                    'chars_contexto': chars_contexto,
                    'tokens_contexto_aprox': tokens_contexto_aprox,
                }
                resultados.append(resultado)
                hechos.add(clave)
                contador += 1

                # Guardado incremental cada 10 ejecuciones (conservador)
                if contador % 10 == 0:
                    guardar_resultados(resultados)
                    print(f'   ✓ Checkpoint guardado ({contador}/{total})')

            except Exception as e:
                print(f'   [!! ERROR] {type(e).__name__}: {e}')
                continue

    # Guardado final
    guardar_resultados(resultados)
    elapsed_total = (time.time() - t_inicio) / 60
    print()
    print("═" * 70)
    print(" PIPELINE FASE 2 COMPLETADO")
    print("═" * 70)
    print(f" • Tiempo total:        {elapsed_total:.1f} min")
    print(f" • Ejecuciones nuevas:  {contador - (total - (total - len(hechos)))}")
    print(f" • Total guardado:      {len(resultados)}")
    print(f" • Archivo de salida:   {ARCHIVO_RESULTADOS}")
    print()


if __name__ == '__main__':
    main()
