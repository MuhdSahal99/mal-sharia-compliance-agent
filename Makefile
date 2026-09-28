.PHONY: install dev test lint ingest run docker-build docker-run clean

# ── Development ──────────────────────────────────────────────

install:
	pip install -r requirements.txt

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/main.py
	python -m py_compile src/config.py
	python -m py_compile src/agent/compliance_agent.py
	python -m py_compile src/rag/chunker.py
	python -m py_compile src/rag/embedder.py
	python -m py_compile src/rag/retriever.py
	python -m py_compile src/rag/ingestion.py

# ── Data Pipeline ────────────────────────────────────────────

ingest:
	python scripts/ingest.py

# ── Docker ───────────────────────────────────────────────────

docker-build:
	docker build -t mal-sharia-agent .

docker-run:
	docker run -p 8000:8000 --env-file .env mal-sharia-agent

# ── Cleanup ──────────────────────────────────────────────────

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
