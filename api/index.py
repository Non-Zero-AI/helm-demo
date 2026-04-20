"""
HELM Demo API — Vercel Serverless Function
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import re
import time

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge-light.json')
SYNONYMS_PATH = os.path.join(os.path.dirname(__file__), 'synonyms.json')
knowledge = None
synonyms = None

STOP_WORDS = set('the a an is are was were be been being have has had do does did will would shall should may might can could of in to for on with at by from as into through during before after above below between out off over under again further then once here there when where why how all each every both few more most other some such no nor not only own same so than too very s t d ll ve re m'.split())

def load_knowledge():
    global knowledge
    if knowledge is None:
        with open(KNOWLEDGE_PATH) as f:
            knowledge = json.load(f)
    return knowledge

def load_synonyms():
    global synonyms
    if synonyms is None:
        with open(SYNONYMS_PATH) as f:
            synonyms = json.load(f)
    return synonyms

def expand_query(query):
    """Add related terms to the query based on synonym dictionary."""
    syns = load_synonyms()
    query_lower = query.lower()
    expanded = set(tokenize(query))
    
    # Check each synonym key against the query
    for key, related in syns.items():
        if key in query_lower:
            expanded.update(related)
    
    return expanded

def tokenize(text):
    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    return set(w for w in words if len(w) > 2 and w not in STOP_WORDS)

def text_similarity(query_tokens, content):
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0
    
    intersection = query_tokens & content_tokens
    if not intersection:
        return 0
    
    recall = len(intersection) / len(query_tokens)
    precision = len(intersection) / len(content_tokens)
    
    return (recall * 0.7) + (precision * 0.3)

def search_knowledge(query_text, top_k=5):
    kb = load_knowledge()
    expanded_tokens = expand_query(query_text)
    
    scored = []
    for unit in kb['units']:
        score = text_similarity(expanded_tokens, unit['content'])
        if score > 0.08:
            scored.append({
                'content': unit['content'],
                'type': unit['type'],
                'confidence': unit['confidence'],
                'source': unit['source'].split(':')[0] if ':' in unit['source'] else unit['source'],
                'score': round(score, 4),
            })
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_k]

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
