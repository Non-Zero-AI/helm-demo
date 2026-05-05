"""
HELM Demo API — FastAPI Backend
Serves HELM retrieval + EMBER reconstruction with live metrics.
"""

import os
import time
import json
import gc
import re
import numpy as np
import faiss
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import threading
from collections import Counter

# ─── Config ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
PARQUET_PATH = os.path.join(DATA_DIR, 'helm_kb_hrr.parquet')
SEMANTIC_INDEX_PATH = os.path.join(DATA_DIR, 'minilm_384d.faiss')
HRR_INDEX_PATH = os.path.join(DATA_DIR, 'hrr_1024d.faiss')

# ─── Global state ──────────────────────────────────────────────────────────
kb = {
    'units': [],           # list of dicts: domain, topic, content, confidence
    'semantic_index': None,
    'hrr_index': None,
    'domains': set(),
    'loaded': False,
    'load_time': 0,
    'total_units': 0,
    'total_domains': 0,
    'corpus_size_bytes': 0,
    'compressed_size_bytes': 0,
    'dim_semantic': 0,
    'dim_hrr': 0,
}

# ─── Startup ───────────────────────────────────────────────────────────────
def load_knowledge_base():
    """Load pre-built FAISS indexes and metadata from disk."""
    import pyarrow.parquet as pq

    start = time.time()
    print("[BOOT] Loading knowledge base...")

    if not os.path.exists(PARQUET_PATH):
        print(f"[BOOT] ERROR: Parquet file not found at {PARQUET_PATH}")
        return

    # 1. Load metadata only (no vectors — saves ~1.3GB RAM)
    print("[BOOT] Loading metadata...")
    table = pq.read_table(PARQUET_PATH, columns=['domain', 'topic', 'content', 'confidence'])
    domains = set()
    corpus_size = 0
    units = []
    for i in range(table.num_rows):
        domain = str(table.column('domain')[i].as_py())
        topic = str(table.column('topic')[i].as_py())
        content = str(table.column('content')[i].as_py())
        confidence = float(table.column('confidence')[i].as_py())
        units.append({
            'domain': domain,
            'topic': topic,
            'content': content,
            'confidence': confidence,
        })
        corpus_size += len(content.encode('utf-8'))
        domains.add(domain)
    del table
    gc.collect()
    print(f"[BOOT] {len(units)} units, {len(domains)} domains, {corpus_size/1024/1024:.0f} MB corpus")

    # 2. Load FAISS indexes
    semantic_index = None
    hrr_index = None
    dim_semantic = 0
    dim_hrr = 0

    if os.path.exists(SEMANTIC_INDEX_PATH):
        print("[BOOT] Loading semantic FAISS index...")
        semantic_index = faiss.read_index(SEMANTIC_INDEX_PATH)
        dim_semantic = semantic_index.d
        print(f"  → {semantic_index.ntotal} vectors × {dim_semantic}D")
    else:
        print(f"[BOOT] WARNING: Semantic index not found at {SEMANTIC_INDEX_PATH}")

    if os.path.exists(HRR_INDEX_PATH):
        print("[BOOT] Loading HRR FAISS index...")
        hrr_index = faiss.read_index(HRR_INDEX_PATH)
        dim_hrr = hrr_index.d
        print(f"  → {hrr_index.ntotal} vectors × {dim_hrr}D")
    else:
        print(f"[BOOT] WARNING: HRR index not found at {HRR_INDEX_PATH}")

    # 3. Compute compressed size (1024D × float16 × 328K)
    compressed_size = len(units) * dim_hrr * 2 if dim_hrr > 0 else 0

    # Store everything
    kb['units'] = units
    kb['semantic_index'] = semantic_index
    kb['hrr_index'] = hrr_index
    kb['domains'] = domains
    kb['loaded'] = True
    kb['load_time'] = time.time() - start
    kb['total_units'] = len(units)
    kb['total_domains'] = len(domains)
    kb['corpus_size_bytes'] = corpus_size
    kb['compressed_size_bytes'] = compressed_size
    kb['dim_semantic'] = dim_semantic
    kb['dim_hrr'] = dim_hrr

    print(f"[BOOT] Ready! {len(units)} units, {len(domains)} domains, "
          f"loaded in {kb['load_time']:.1f}s")


# ─── Cached model ──────────────────────────────────────────────────────────
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


# ─── Retrieval pipeline ───────────────────────────────────────────────────
STOPWORDS = set('the a an is are was were be been being have has had do does did will would shall should may might can could of in to for on with at by from as into through during before after above below between out off over under again further then once here there when where why how all each every both few more most other some such no nor not only own same so than too very'.split())

def tokenize(text):
    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def retrieve(query: str, top_k: int = 5):
    """HELM retrieval: FAISS semantic search → HRR re-ranking → results."""
    metrics = {}

    # Step 1: Encode query
    t0 = time.time()
    model = get_embedding_model()
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    metrics['encode_ms'] = round((time.time() - t0) * 1000, 1)

    if kb['semantic_index'] is None:
        return [], metrics

    # Step 2: FAISS semantic search
    t1 = time.time()
    k_search = min(100, kb['total_units'])
    scores, indices = kb['semantic_index'].search(query_emb, k_search)
    metrics['faiss_ms'] = round((time.time() - t1) * 1000, 1)

    # Step 3: HRR re-ranking
    t2 = time.time()
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= kb['total_units']:
            continue
        candidates.append({
            'index': int(idx),
            'semantic_score': float(score),
            'unit': kb['units'][idx],
        })

    if kb['hrr_index'] is not None and len(candidates) > 5:
        # HRR centroid re-ranking on top-20
        top_n = min(20, len(candidates))
        candidate_indices = [c['index'] for c in candidates[:top_n]]
        # Retrieve HRR vectors from FAISS index
        hrr_vectors = np.vstack([
            kb['hrr_index'].reconstruct(int(idx)).reshape(1, -1)
            for idx in candidate_indices
        ])
        centroid = np.mean(hrr_vectors, axis=0, keepdims=True)
        hrr_scores = np.dot(hrr_vectors, centroid.T).flatten()
        for i, c in enumerate(candidates[:top_n]):
            c['hrr_score'] = float(hrr_scores[i])
            c['combined_score'] = c['semantic_score'] * 0.6 + c['hrr_score'] * 0.4
        candidates[:top_n] = sorted(candidates[:top_n], key=lambda x: x.get('combined_score', 0), reverse=True)

    metrics['hrr_rerank_ms'] = round((time.time() - t2) * 1000, 1)
    metrics['total_ms'] = round(sum(v for k, v in metrics.items() if k.endswith('_ms')), 1)

    # Build results
    results = []
    for c in candidates[:top_k]:
        results.append({
            'domain': c['unit']['domain'],
            'topic': c['unit']['topic'],
            'content': c['unit']['content'],
            'confidence': c['unit']['confidence'],
            'semantic_score': round(c['semantic_score'], 4),
            'hrr_score': round(c.get('hrr_score', 0), 4),
            'combined_score': round(c.get('combined_score', c['semantic_score']), 4),
        })

    return results, metrics


# ─── LLM Reconstruction ──────────────────────────────────────────────────
def reconstruct_with_llm(results, query: str):
    """Use an LLM to reconstruct a coherent answer from retrieved knowledge."""
    import openai

    t0 = time.time()

    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Source {i+1}] Domain: {r['domain']} | Topic: {r['topic']} | "
            f"Confidence: {r['confidence']}\n{r['content']}"
        )
    context = "\n\n".join(context_parts)
    context_tokens = len(context) // 4
    query_tokens = len(query) // 4
    total_input_tokens = context_tokens + query_tokens

    api_key = os.environ.get('OPENROUTER_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return {
            'answer': '[LLM reconstruction requires an API key]',
            'context': context,
            'tokens': {'input': total_input_tokens, 'output': 0, 'total': total_input_tokens},
            'reconstruction_ms': 0,
            'cost_comparison': compute_cost_comparison(total_input_tokens, 0),
        }

    # Use OpenAI endpoint if key starts with sk-proj or sk-, otherwise OpenRouter
    if api_key.startswith('sk-proj-') or api_key.startswith('sk-'):
        client = openai.OpenAI(api_key=api_key, timeout=10.0)
        model_name = "gpt-4o-mini"
    else:
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=10.0)
        model_name = "qwen/qwen3-8b"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a knowledge retrieval system. Answer the user's question using ONLY the provided context. Be concise and cite sources by number [1], [2], etc."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=500,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        output_tokens = len(answer) // 4
    except Exception as e:
        answer = f"[LLM unavailable: {str(e)[:100]}]\n\nRetrieved {len(results)} knowledge sources — see context below."
        output_tokens = 0

    return {
        'answer': answer,
        'context': context,
        'tokens': {'input': total_input_tokens, 'output': output_tokens, 'total': total_input_tokens + output_tokens},
        'reconstruction_ms': round((time.time() - t0) * 1000, 1),
        'cost_comparison': compute_cost_comparison(total_input_tokens, output_tokens),
    }


def compute_cost_comparison(input_tokens, output_tokens):
    """Compare cost: HELM (8B) vs traditional (70B+)."""
    # gpt-4o-mini pricing
    helm_cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
    # gpt-4o pricing
    trad_cost = (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000
    savings_pct = ((trad_cost - helm_cost) / trad_cost * 100) if trad_cost > 0 else 0
    return {
        'helm_cost_usd': round(helm_cost, 6),
        'traditional_cost_usd': round(trad_cost, 6),
        'savings_percent': round(savings_pct, 1),
        'cost_ratio': round(trad_cost / helm_cost, 1) if helm_cost > 0 else 0,
    }


# ─── App lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_knowledge_base, daemon=True)
    thread.start()
    yield

app = FastAPI(title="HELM Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok" if kb['loaded'] else "loading",
        "units": kb['total_units'],
        "domains": kb['total_domains'],
    }


@app.get("/api/stats")
async def stats():
    if not kb['loaded']:
        return JSONResponse({"error": "Knowledge base still loading"}, status_code=503)

    corpus_mb = kb['corpus_size_bytes'] / (1024 * 1024)
    compressed_mb = kb['compressed_size_bytes'] / (1024 * 1024)
    ratio = corpus_mb / compressed_mb if compressed_mb > 0 else 0

    return {
        "total_units": kb['total_units'],
        "total_domains": kb['total_domains'],
        "corpus_size_mb": round(corpus_mb, 1),
        "corpus_size_gb": round(corpus_mb / 1024, 2),
        "compressed_size_mb": round(compressed_mb, 2),
        "compression_ratio": round(ratio, 1),
        "hrr_dimensions": kb['dim_hrr'],
        "semantic_dimensions": kb['dim_semantic'],
        "load_time_seconds": round(kb['load_time'], 1),
        "sample_domains": sorted(list(kb['domains']))[:50],
    }


@app.get("/api/query")
async def query(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
    reconstruct: bool = Query(True, description="Run LLM reconstruction"),
):
    if not kb['loaded']:
        return JSONResponse({"error": "Knowledge base still loading"}, status_code=503)

    results, retrieval_metrics = retrieve(q, top_k=top_k)

    reconstruction = None
    if reconstruct:
        reconstruction = reconstruct_with_llm(results, q)

    response = {
        "query": q,
        "results": results,
        "retrieval_metrics": retrieval_metrics,
    }

    if reconstruction:
        response["reconstruction"] = reconstruction
        response["total_ms"] = round(retrieval_metrics['total_ms'] + reconstruction['reconstruction_ms'], 1)
    else:
        response["total_ms"] = retrieval_metrics['total_ms']

    context_size = sum(len(r['content'].encode('utf-8')) for r in results)
    response["query_stats"] = {
        "results_returned": len(results),
        "context_size_bytes": context_size,
        "context_size_kb": round(context_size / 1024, 1),
        "knowledge_base_units": kb['total_units'],
        "knowledge_base_domains": kb['total_domains'],
    }

    return response


@app.get("/api/domains")
async def domains():
    if not kb['loaded']:
        return JSONResponse({"error": "Knowledge base still loading"}, status_code=503)
    return {"domains": sorted(list(kb['domains'])), "count": len(kb['domains'])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
