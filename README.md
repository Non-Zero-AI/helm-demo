# HELM Demo

**Live demo of the Holographically Encoded Learning Model.**

> Ask a question → watch knowledge get retrieved → see the context that grounds the AI's answer.

## What This Shows

- **3,110 knowledge units** searchable in <1ms
- **Real-time knowledge retrieval** with relevance scores
- **Context assembly** — how retrieved knowledge becomes an LLM prompt
- **Multi-model knowledge** — contributions from Llama, GPT-4, and Mistral

## Deploy

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Or connect to Vercel dashboard for auto-deploy from GitHub
```

## Local Development

```bash
# Copy the knowledge base
cp ../holographic-llm-patent-outline/extracted_knowledge/encoded_knowledge.json data/

# Run locally
vercel dev
```

## Architecture

```
User Query → API (Python) → Knowledge Search (cosine similarity) → Context Assembly → Prompt
                                    ↓
                          3,110 units in 25.9MB
                          <1ms retrieval
                          145+ domains
```

## Branding

- **Name:** HELM — Holographically Encoded Learning Model
- **Tagline:** "AI that learns, not just trains."
- **Color:** Teal (#00897B)
- **Built:** April 20, 2026
- **Cost:** $30 total development

---

*Patent-pending technology. See the [patent repo](https://github.com/Non-Zero-AI/holographic-llm-patent-outline) for technical documentation.*
