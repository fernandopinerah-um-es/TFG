"""
=====================================================================
 SCRIPT 1: PIPELINE DE RETRIEVAL + GENERACIÓN PARA EVALUACIÓN RAG
=====================================================================

Ejecuta el pipeline RAG completo sobre el dataset submuestreado, para
todas las combinaciones experimentales:
    4 estrategias de chunking × 7 estrategias de retrieval = 28 combinaciones
    × 160 preguntas = 4480 ejecuciones

Para cada ejecución guarda:
    - chunks recuperados (chunk_id, texto, rank)
    - respuesta generada por el LLM

Los resultados se guardan incrementalmente en JSON con checkpoints,
de modo que si se interrumpe la ejecución se pueda reanudar sin repetir
lo ya hecho.

Input:
    - dataset_submuestreado.json (160 preguntas)
    - Elasticsearch con los 4 índices de chunking ya poblados

Output:
    - resultados_retrieval_generacion.json (incremental)
"""

# ════════ SILENCIAR WARNINGS (debe ir antes de cualquier otro import) ════════
import warnings
import os
import logging

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')

for nombre_logger in [
    'transformers',
    'transformers.generation',
    'transformers.generation.utils',
    'transformers.generation.configuration_utils',
    'elastic_transport',
    'elasticsearch',
    'urllib3',
    'sentence_transformers',
    'torch',
]:
    logging.getLogger(nombre_logger).setLevel(logging.ERROR)

try:
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
except ImportError:
    pass
# ═════════════════════════════════════════════════════════════════════════

# Ahora sí los imports normales
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════

# Archivos
ARCHIVO_DATASET = 'dataset_submuestreado.json'
ARCHIVO_RESULTADOS = 'resultados_retrieval_generacion.json'

# Elasticsearch
ES_HOST = 'http://localhost:9201'

# Parámetros de recuperación
TOP_K = 5  # retrievamos y reportamos a K=5 (después calcularemos métricas a K=3 y K=5)
NUM_CANDIDATES = 100  # para KNN

# Modelos
MODELO_GENERALISTA = 'intfloat/multilingual-e5-base'
MODELO_MEDICO = 'cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR'
MODELO_EXPERTO_PATH = './modelo_especializado_e5'
MODELO_LLM = 'Qwen/Qwen2.5-7B-Instruct'

# Mapa chunking → índice en ES
INDICES_POR_CHUNKING = {
    'A_fixed': 'indice_estrategia_a_fixed',
    'B_markdown': 'indice_estrategia_b_markdown',
    'C_semantica': 'indice_estrategia_c_semantica',
    'D_jerarquico': 'indice_estrategia_d_jerarquico',
}

# Estrategias de retrieval
ESTRATEGIAS_RETRIEVAL = [
    'bm25',
    'semantica_generalista',
    'semantica_medico',
    'semantica_experto',
    'hibrida_generalista',
    'hibrida_medico',
    'hibrida_experto',
]

# Generación LLM
# Este límite es defensivo: con K=5 y chunks de ~600 chars nunca se alcanza
# en condiciones normales (máximo teórico ~20k chars con la estrategia
# jerárquica). Si se activara indicaría un caso anómalo que hay que investigar
# porque introduciría inconsistencia entre métricas IR y RAGAS (el LLM vería
# menos contextos de los medidos).
LLM_MAX_CHARS_CONTEXTO = 150000
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURA = 0.1

# ═══════════════════════════════════════════════════════════════════
# CARGA DE MODELOS (una sola vez)
# ═══════════════════════════════════════════════════════════════════

print("═" * 70)
print(" INICIALIZANDO PIPELINE DE EVALUACIÓN")
print("═" * 70)

# Elasticsearch
es = Elasticsearch(ES_HOST, request_timeout=60, max_retries=3, retry_on_timeout=True)
assert es.ping(), 'ERROR: Elasticsearch no disponible'
print(f" Conectado a Elasticsearch en {ES_HOST}")

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f" Dispositivo: {device.upper()}")

# Embeddings
print(" \nCargando modelo generalista (E5-base)...")
modelo_generalista = SentenceTransformer(MODELO_GENERALISTA, device='cpu')

print(" \nCargando modelo médico (SapBERT con CLS pooling)...")
word_model = models.Transformer(MODELO_MEDICO)
pool_model = models.Pooling(
    word_model.get_word_embedding_dimension(),
    pooling_mode_cls_token=True,
    pooling_mode_mean_tokens=False,
)
modelo_medico = SentenceTransformer(modules=[word_model, pool_model], device='cpu')

print(" \nCargando modelo experto (E5 finetuneado)...")
modelo_experto = SentenceTransformer(MODELO_EXPERTO_PATH, device=device)

# LLM
print(f" \nCargando LLM ({MODELO_LLM})...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM)
model_llm = AutoModelForCausalLM.from_pretrained(
    MODELO_LLM,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
pipe_llm = pipeline('text-generation', model=model_llm, tokenizer=tokenizer)

print("¡Todos los modelos cargados con éxito!\n")
print()


# ═══════════════════════════════════════════════════════════════════
# FUNCIONES DE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

def _extraer_info_hits(hits, es_jerarquico, nombre_indice):
    """
    Procesa los resultados crudos de ES y devuelve lista de dicts con
    chunk_id, text, rank. Si es jerárquico, sustituye cada hijo por su padre
    (deduplicando padres).
    """
    if not es_jerarquico:
        return [
            {
                'chunk_id': hit['_source']['chunk_id'],
                'text': hit['_source']['text'],
                'rank': rank + 1,
            }
            for rank, hit in enumerate(hits)
        ]

    # Jerárquico: recuperar padres por parent_id preservando orden
    parent_ids_ordenados = []
    vistos = set()
    for hit in hits:
        parent_id = hit['_source']['metadata']['parent_id']
        if parent_id not in vistos:
            parent_ids_ordenados.append(parent_id)
            vistos.add(parent_id)

    if not parent_ids_ordenados:
        return []

    # Recuperar padres (con una sola consulta)
    resp = es.search(
        index=nombre_indice,
        query={'terms': {'_id': parent_ids_ordenados}},
        size=len(parent_ids_ordenados),
    )
    # Construir mapa id → texto
    padres_map = {h['_id']: h['_source']['text'] for h in resp['hits']['hits']}

    return [
        {
            'chunk_id': pid,
            'text': padres_map[pid],
            'rank': rank + 1,
        }
        for rank, pid in enumerate(parent_ids_ordenados)
        if pid in padres_map
    ]


def retrieve_bm25(pregunta, nombre_indice, top_k=TOP_K):
    es_jerarquico = 'jerarquico' in nombre_indice.lower()
    query = {'bool': {'must': [{'match': {'text': pregunta}}]}}
    if es_jerarquico:
        query['bool']['filter'] = [{'term': {'metadata.doc_type': 'hijo'}}]
    resp = es.search(
        index=nombre_indice,
        query=query,
        size=top_k,
        _source=['text', 'chunk_id', 'metadata'],
    )
    return _extraer_info_hits(resp['hits']['hits'], es_jerarquico, nombre_indice)


def retrieve_semantica(pregunta, nombre_indice, tipo_modelo, top_k=TOP_K):
    es_jerarquico = 'jerarquico' in nombre_indice.lower()

    if tipo_modelo == 'generalista':
        vector = modelo_generalista.encode(f'query: {pregunta}').tolist()
        campo = 'vector_generalista'
    elif tipo_modelo == 'medico':
        vector = modelo_medico.encode(pregunta).tolist()
        campo = 'vector_medico'
    elif tipo_modelo == 'experto':
        vector = modelo_experto.encode(f'query: {pregunta}').tolist()
        campo = 'vector_experto'
    else:
        raise ValueError(f'tipo_modelo desconocido: {tipo_modelo}')

    body = {
        'knn': {
            'field': campo,
            'query_vector': vector,
            'k': top_k,
            'num_candidates': NUM_CANDIDATES,
        },
        '_source': ['text', 'chunk_id', 'metadata'],
    }
    if es_jerarquico:
        body['knn']['filter'] = {'term': {'metadata.doc_type': 'hijo'}}

    resp = es.search(index=nombre_indice, body=body, size=top_k)
    return _extraer_info_hits(resp['hits']['hits'], es_jerarquico, nombre_indice)


def retrieve_hibrida(pregunta, nombre_indice, tipo_modelo, top_k=TOP_K, k_rrf=60):
    es_jerarquico = 'jerarquico' in nombre_indice.lower()

    if tipo_modelo == 'generalista':
        vector = modelo_generalista.encode(f'query: {pregunta}').tolist()
        campo = 'vector_generalista'
    elif tipo_modelo == 'medico':
        vector = modelo_medico.encode(pregunta).tolist()
        campo = 'vector_medico'
    elif tipo_modelo == 'experto':
        vector = modelo_experto.encode(f'query: {pregunta}').tolist()
        campo = 'vector_experto'
    else:
        raise ValueError(f'tipo_modelo desconocido: {tipo_modelo}')

    source = ['text', 'chunk_id', 'metadata']
    filtro_hijo = {'term': {'metadata.doc_type': 'hijo'}}

    # ── Búsqueda BM25 ──────────────────────────────────────────────────────────
    body_bm25 = {
        'size': top_k,
        'query': {'bool': {'must': [{'match': {'text': {'query': pregunta}}}]}},
        '_source': source,
    }

    # ── Búsqueda kNN ───────────────────────────────────────────────────────────
    body_knn = {
        'size': top_k,
        'knn': {
            'field': campo,
            'query_vector': vector,
            'k': top_k,
            'num_candidates': NUM_CANDIDATES,
        },
        '_source': source,
    }

    if es_jerarquico:
        body_bm25['query']['bool']['filter'] = [filtro_hijo]
        body_knn['knn']['filter'] = filtro_hijo

    hits_bm25 = es.search(index=nombre_indice, body=body_bm25)['hits']['hits']
    hits_knn  = es.search(index=nombre_indice, body=body_knn)['hits']['hits']

    # ── RRF manual: score(d) = Σ 1/(k_rrf + rank + 1) ────────────────────────
    scores, store = {}, {}
    for lista in (hits_bm25, hits_knn):
        for rank, hit in enumerate(lista):
            doc_id = hit['_id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            store[doc_id]  = hit                           # conserva el hit más reciente

    hits_fusionados = sorted(
        store.values(),
        key=lambda h: scores[h['_id']],
        reverse=True,
    )[:top_k]

    return _extraer_info_hits(hits_fusionados, es_jerarquico, nombre_indice)

def ejecutar_retrieval(pregunta, estrategia_chunking, estrategia_retrieval):
    """Despacha al método de retrieval correspondiente."""
    indice = INDICES_POR_CHUNKING[estrategia_chunking]

    if estrategia_retrieval == 'bm25':
        return retrieve_bm25(pregunta, indice)
    elif estrategia_retrieval.startswith('semantica_'):
        tipo = estrategia_retrieval.replace('semantica_', '')
        return retrieve_semantica(pregunta, indice, tipo)
    elif estrategia_retrieval.startswith('hibrida_'):
        tipo = estrategia_retrieval.replace('hibrida_', '')
        return retrieve_hibrida(pregunta, indice, tipo)
    else:
        raise ValueError(f'Estrategia desconocida: {estrategia_retrieval}')


# ═══════════════════════════════════════════════════════════════════
# GENERACIÓN CON LLM
# ═══════════════════════════════════════════════════════════════════

INSTRUCCIONES_SISTEMA = (
    'Eres un asistente médico experto y preciso. Tu tarea es responder a la '
    'pregunta del usuario utilizando ÚNICAMENTE la información proporcionada '
    'en el contexto. Si el contexto no contiene información suficiente para '
    'responder a la pregunta, debes indicar claramente que no tienes '
    'suficientes datos para responder. No inventes información.'
)


def generar_respuesta(pregunta, chunks_recuperados):
    """
    Construye el prompt con los textos recuperados y genera respuesta.
    Devuelve (respuesta, textos_usados) para poder trazar qué vio el LLM.
    Emite warning si el truncado defensivo se activa, ya que introduciría
    inconsistencia entre las métricas IR (sobre todos los chunks recuperados)
    y las métricas RAGAS (sobre los contextos que realmente vio el LLM).
    """
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
        print(f'   [!! WARNING] Truncado de contexto activado: {n_omitidos} chunk(s) omitidos. '
              f'Total chars: {chars}. Esto genera inconsistencia entre métricas IR y RAGAS.')

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
    respuesta = resultados[0]['generated_text'][-1]['content']

    return respuesta, contextos_usados


# ═══════════════════════════════════════════════════════════════════
# GESTIÓN DE CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════

def cargar_resultados_previos():
    """Carga resultados parciales si existen para reanudar."""
    if not os.path.exists(ARCHIVO_RESULTADOS):
        return []
    with open(ARCHIVO_RESULTADOS, 'r', encoding='utf-8') as f:
        return json.load(f)


def guardar_resultados(resultados):
    """Guarda la lista completa de resultados al disco."""
    with open(ARCHIVO_RESULTADOS, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)


def clave_ejecucion(pregunta_idx, chunking, retrieval):
    """Clave única para identificar una ejecución ya hecha."""
    return f'{pregunta_idx}__{chunking}__{retrieval}'


# ═══════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main():
    # Cargar dataset de preguntas
    with open(ARCHIVO_DATASET, 'r', encoding='utf-8') as f:
        preguntas = json.load(f)

    # Cargar resultados previos (checkpoint)
    resultados = cargar_resultados_previos()
    hechos = {
        clave_ejecucion(r['pregunta_idx'], r['chunking'], r['retrieval'])
        for r in resultados
    }

    total_combinaciones = len(preguntas) * len(INDICES_POR_CHUNKING) * len(ESTRATEGIAS_RETRIEVAL)
    hechos_inicial = len(hechos)

    print(f" - Preguntas en dataset:       {len(preguntas)}")
    print(f" - Estrategias chunking:       {len(INDICES_POR_CHUNKING)}")
    print(f" - Estrategias retrieval:      {len(ESTRATEGIAS_RETRIEVAL)}")
    print(f" - Total ejecuciones:          {total_combinaciones}")
    print(f" - Ya completadas (checkpoint): {hechos_inicial}")
    print(f" - Pendientes:                 {total_combinaciones - hechos_inicial}")
    print()

    if hechos_inicial >= total_combinaciones:
        print(" !!! Todo ya estaba hecho. Nada que procesar.")
        return

    t_inicio = time.time()
    contador = hechos_inicial

    # Iteramos: pregunta → chunking → retrieval
    # Este orden minimiza cambios de índice en ES
    for p_idx, pregunta in enumerate(preguntas):
        for chunking in INDICES_POR_CHUNKING:
            for retrieval in ESTRATEGIAS_RETRIEVAL:
                clave = clave_ejecucion(p_idx, chunking, retrieval)
                if clave in hechos:
                    continue

                contador += 1
                progreso = contador / total_combinaciones * 100
                t_trans = time.time() - t_inicio
                velocidad = (contador - hechos_inicial) / max(t_trans, 1) * 60
                eta_min = (total_combinaciones - contador) / max(velocidad / 60, 0.001) / 60

                print(f'[{contador}/{total_combinaciones} | {progreso:.1f}% | '
                      f'{velocidad:.1f}/min | ETA {eta_min:.0f} min] '
                      f'P{p_idx} | {chunking} | {retrieval}')

                try:
                    # FASE 1: Retrieval (con medición de latencia)
                    t_retrieval_ini = time.perf_counter()
                    chunks = ejecutar_retrieval(
                        pregunta['pregunta'], chunking, retrieval
                    )
                    latencia_retrieval_ms = (time.perf_counter() - t_retrieval_ini) * 1000

                    # FASE 2: Generación (con medición de latencia)
                    t_gen_ini = time.perf_counter()
                    respuesta, contextos_usados = generar_respuesta(
                        pregunta['pregunta'], chunks
                    )
                    latencia_generacion_ms = (time.perf_counter() - t_gen_ini) * 1000

                    # Tamaño del contexto realmente entregado al LLM
                    chars_contexto = sum(len(t) for t in contextos_usados)
                    tokens_contexto_aprox = len(tokenizer.encode(
                        '\n\n---\n\n'.join(contextos_usados)
                    ))

                    # Guardar resultado
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
                        'chunks_recuperados': chunks,
                        'respuesta_generada': respuesta,
                        'n_contextos_usados': len(contextos_usados),
                        # Métricas de eficiencia computacional
                        'latencia_retrieval_ms': round(latencia_retrieval_ms, 2),
                        'latencia_generacion_ms': round(latencia_generacion_ms, 2),
                        'latencia_total_ms': round(latencia_retrieval_ms + latencia_generacion_ms, 2),
                        'chars_contexto': chars_contexto,
                        'tokens_contexto_aprox': tokens_contexto_aprox,
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                    }
                    resultados.append(resultado)
                    hechos.add(clave)

                    # Checkpoint cada 10 ejecuciones
                    if contador % 10 == 0:
                        guardar_resultados(resultados)

                except Exception as e:
                    print(f'   [ERROR] {type(e).__name__}: {e}')
                    # Guardar antes de continuar para no perder progreso
                    guardar_resultados(resultados)
                    # Seguimos con la siguiente combinación
                    continue

    # Guardado final
    guardar_resultados(resultados)

    t_total = time.time() - t_inicio
    print()
    print("═" * 70)
    print(f' !!! Pipeline completado')
    print(f' - Ejecuciones nuevas:    {contador - hechos_inicial}')
    print(f' - Total en archivo:      {len(resultados)}')
    print(f' - Tiempo transcurrido:   {t_total/60:.1f} min')
    print(f' - Archivo:               {ARCHIVO_RESULTADOS}')
    print("═" * 70)


if __name__ == '__main__':
    main()
