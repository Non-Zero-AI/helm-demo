"""
HELM Demo API — Vercel Serverless Function
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import math
import re
import time

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge-light.json')
knowledge = None

def load_knowledge():
    global knowledge
    if knowledge is None:
        with open(KNOWLEDGE_PATH) as f:
            knowledge = json.load(f)
    return knowledge

def tokenize(text):
    return [t for t in re.sub(r'[^\w\s]', ' ', text.lower()).split() if len(t) > 2]

def text_similarity(query, content):
    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(content))
    if not query_tokens:
        return 0
    return len(query_tokens & content_tokens) / len(query_tokens)

def search_knowledge(query_text, top_k=5):
    kb = load_knowledge()
    scored = []
    for unit in kb['units']:
        score = text_similarity(query_text, unit['content'])
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
    
    return ("You are a helpful assistant. Use the following knowledge to answer the question.\n\n"
            "Knowledge:\n" + context_block + "\n"
            "Question: " + query + "\n\n"
            "Answer:")

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length))
            query = body.get('query', '')
            top_k = body.get('top_k', 3)

            if not query:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Missing query'}).encode())
                return

            start = time.time()
            results = search_knowledge(query, top_k)
            prompt = build_prompt(query, results) if results else query
            elapsed = round(time.time() - start, 3)
            kb = load_knowledge()

            response = {
                'query': query,
                'context': results,
                'prompt': prompt,
                'stats': {
                    'total_units': kb['total_units'],
                    'search_time_ms': round(elapsed * 1000),
                    'results_found': len(results),
                }
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_GET(self):
        self.send_response(405)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'error': 'Use POST'}).encode())
