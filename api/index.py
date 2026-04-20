"""
HELM Demo API — Vercel Serverless Function
Runs the HELM pipeline: embed → search → context → generate
"""

import json
import os
import math
import time

# Load knowledge base at cold start (cached across invocations)
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge-light.json')
knowledge = None

def load_knowledge():
    global knowledge
    if knowledge is None:
        with open(KNOWLEDGE_PATH) as f:
            knowledge = json.load(f)
    return knowledge

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a * norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def tokenize(text):
    import re
    return [t for t in re.sub(r'[^\w\s]', ' ', text.lower()).split() if len(t) > 2]

def text_similarity(query, content):
    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(content))
    if not query_tokens:
        return 0
    matches = len(query_tokens & content_tokens)
    return matches / len(query_tokens)

def search_knowledge(query_text, top_k=5):
    kb = load_knowledge()
    scored = []
    for unit in kb['units']:
        # Hybrid score: semantic + text
        vec_score = 0
        txt_score = text_similarity(query_text, unit['content'])
        score = max(vec_score, txt_score)
        scored.append({
            'content': unit['content'],
            'type': unit['type'],
            'confidence': unit['confidence'],
            'source': unit['source'].split(':')[0] if ':' in unit['source'] else unit['source'],
            'score': round(score, 4),
        })
    scored.sort(key=lambda x: x['score'], reverse=True)
    return [s for s in scored[:top_k] if s['score'] > 0]

def build_prompt(query, context_results):
    context_block = ""
    for i, r in enumerate(context_results, 1):
        context_block += "[" + str(i) + "] (" + r['type'] + ", relevance: " + str(round(r['score'], 2)) + ")\n"
        context_block += r['content'] + "\n\n"
    
    return """You are a helpful assistant. Use the following knowledge to answer the question. If the knowledge doesn't contain relevant information, say so.

Knowledge:
""" + context_block + """
Question: """ + query + """

Answer:"""

def handler(request):
    """Vercel serverless handler"""
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': ''
        }
    
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        body = json.loads(request.body)
        query = body.get('query', '')
        top_k = body.get('top_k', 3)
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Missing query'})
            }
        
        start = time.time()
        
        # Search (text-based since we can't run MiniLM in serverless)
        results = search_knowledge(query, top_k)
        
        # Build prompt
        prompt = build_prompt(query, results) if results else query
        
        elapsed = round(time.time() - start, 3)
        kb = load_knowledge()
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'query': query,
                'context': results,
                'prompt': prompt,
                'stats': {
                    'total_units': kb['total_units'],
                    'search_time_ms': round(elapsed * 1000),
                    'results_found': len(results),
                }
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }
