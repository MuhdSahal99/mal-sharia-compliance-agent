# Mal Sharia Compliance Agent

> RAG-powered AI agent for assessing whether financial products and transactions comply with Sharia law, based on AAOIFI standards.

Built for Mal's internal compliance team. Submit a plain-English query about any financial product — get a structured verdict (`COMPLIANT`, `NON_COMPLIANT`, or `NEEDS_REVIEW`) with cited reasoning from AAOIFI Sharia standards.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│  POST /assess ──────────────────────────── GET /health          │
│       │                                         │               │
│  TraceIDMiddleware ──── ULID generation ──── ContextVars        │
└───────┬─────────────────────────────────────────┬───────────────┘
        │                                         │
        ▼                                         ▼
┌───────────────────────────────────────────────────────────────┐
│                    Compliance Agent Pipeline                   │
│                                                               │
│  Phase 1: RETRIEVE ──► Phase 2: REASON ──► Phase 3: EXTRACT  │
│      │                      │                    │            │
│      ▼                      ▼                    ▼            │
│  ┌──────────┐      ┌──────────────┐     ┌──────────────┐     │
│  │ Retriever│      │  Gemini API  │     │   Verdict    │     │
│  │  top-k   │      │  gpt-4o-mini │     │  Extractor   │     │
│  │ + filter │      │  temp=0.1    │     │  JSON→Struct │     │
│  └────┬─────┘      │  json_mode   │     └──────────────┘     │
│       │            └──────────────┘                           │
│       ▼                                                       │
│  ┌──────────┐                                                │
│  │ Embedder │ ◄── gemini-embedding-001                      │
│  └────┬─────┘                                                │
│       ▼                                                       │
│  ┌──────────────────┐                                        │
│  │   Pinecone        │ ◄── Managed, cosine similarity       │
│  │   Vector Store    │     ~100 chunks from 5 AAOIFI docs   │
│  └──────────────────┘                                        │
└───────────────────────────────────────────────────────────────┘
        ▲
        │ Ingestion Pipeline (runs once at first startup)
        │
┌───────┴───────────────────────────────────────────────────────┐
│  Markdown files ──► Section-aware chunking ──► Batch embed    │
│  src/documents/     (header + paragraph)        (Gemini API)  │
│  5 AAOIFI std docs  ~400 tokens/chunk                         │
└───────────────────────────────────────────────────────────────┘

Observability: Every phase logs with trace_id via structlog + contextvars
```

## Quick Start

### Prerequisites

- Python 3.11+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- A Pinecone API key from [Pinecone](https://app.pinecone.io/)

### 1. Clone and install

```bash
git clone https://github.com/your-username/mal-sharia-compliance-agent.git
cd mal-sharia-compliance-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY and PINECONE_API_KEY
```

Default configuration in `.env.example` uses Gemini and Pinecone:
```env
GEMINI_API_KEY=your-gemini-api-key-here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=2048
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=mal-sharia-standards
PINECONE_DIMENSION=2048
```

### 3. Run the server

```bash
# Development (with auto-reload)
uvicorn src.main:app --reload --port 8000

# Or use Make
make dev
```

On first startup, the server automatically creates the Pinecone index if needed and ingests the 5 AAOIFI standard documents. This takes ~10 seconds.

### 4. Test it

```bash
# Health check
curl http://localhost:8000/health

# Submit a compliance query
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"query": "Can Mal offer a fixed-return savings account?"}'
```

## Example Request & Response

### Request

```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can Mal offer a fixed-return savings account that guarantees 5% annual return on deposits?"
  }'
```

### Response

```json
{
  "trace_id": "01J8KXYZ1234ABCD5678EFGH",
  "query": "Can Mal offer a fixed-return savings account that guarantees 5% annual return on deposits?",
  "verdict": "NON_COMPLIANT",
  "reasoning": "A savings account that guarantees a fixed 5% annual return on the deposited principal constitutes Riba al-Nasiah under AAOIFI Sharia Standard No. 3 (Section 2.1). The guarantee of any return above principal — whether expressed as a fixed percentage or variable rate — is categorically prohibited. Ruling R-3.01 explicitly classifies fixed interest on deposits as NON_COMPLIANT. Permissible alternatives include: (1) Mudaraba-based savings where returns are based on actual profit-sharing ratios (Ruling R-3.02), or (2) Wadiah-based savings where the bank may provide a non-contractual gift (hibah) at its discretion.",
  "conditions": [],
  "sources": [
    {
      "document": "aaoifi_riba_prohibition",
      "section": "Fixed-Return Savings Accounts",
      "relevance_score": 0.94
    },
    {
      "document": "aaoifi_riba_prohibition",
      "section": "Riba al-Nasiah (Deferment Interest)",
      "relevance_score": 0.87
    }
  ],
  "confidence": 0.96
}
```

## API Documentation

Interactive API docs are available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

| Method | Path      | Description                                |
|--------|-----------|--------------------------------------------|
| POST   | `/assess` | Submit a compliance query                  |
| GET    | `/health` | Service health and vector store status     |
| GET    | `/docs`   | Interactive Swagger documentation          |

```

## Known Limitations

1. **Small knowledge base**: Only 5 synthesized AAOIFI standard documents. Production would need the full AAOIFI standard library, CBUAE regulations, and institution-specific fatwa.

2. **Managed vector store dependency**: Pinecone adds network latency and vendor-specific operational limits.

3. **No caching**: Every query triggers an embedding + LLM call. Identical queries pay full latency and cost each time.

4. **No authentication**: The API is open. Production requires API key auth, rate limiting, and audit logging.

5. **English only**: Queries and responses are in English. Many AAOIFI standards are originally in Arabic; nuances may be lost.

6. **No human-in-the-loop**: NEEDS_REVIEW verdicts are returned to the caller but not routed to an actual Sharia scholar for review.

7. **LLM dependency**: Gemini API outages or quota limits would make the service unavailable. No fallback LLM configured.

8. **Synchronous LLM calls**: The LLM call blocks the request. Under high load, this becomes a bottleneck.

## License

MIT
