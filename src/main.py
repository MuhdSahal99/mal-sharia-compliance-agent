"""
Application entry point and FastAPI application factory.

Responsible for:
1. Creating the FastAPI app instance
2. Wiring up middleware
3. Initializing the RAG pipeline components (embedder, store, retriever)
4. Running document ingestion on first startup
5. Injecting dependencies into the route layer

Design decision — startup initialization vs. lazy loading:
We initialize everything at startup (in the lifespan handler) rather than
lazily on the first request because:
- The first request would be extremely slow (embedding model download, ingestion)
- Health checks should reflect actual readiness, not "ready to initialize"
- Failures during init should prevent the app from accepting traffic
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.config import settings
from src.observability.logger import get_logger
from src.api.middleware import TraceIDMiddleware
from src.api.routes import router, configure_routes
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.ingestion import IngestionPipeline
from src.vectorstore.pinecone_store import PineconeStore
from src.agent.compliance_agent import ComplianceAgent

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler — runs at startup and shutdown.

    Startup:
    1. Initialize the embedding model and vector store
    2. Check if documents need ingestion (first run)
    3. Build the retriever and compliance agent
    4. Wire everything into the route layer

    Shutdown:
    - Log clean shutdown (resources are GC'd automatically)
    """
    logger.info("application_starting", version=__version__)

    # ── Initialize infrastructure ────────────────────────────
    embedder = Embedder()
    store = PineconeStore()

    # ── Ingest documents if the store is empty ───────────────
    if store.count() == 0:
        if settings.effective_embedding_api_key:
            logger.info("vector_store_empty_running_ingestion")
            pipeline = IngestionPipeline(embedder=embedder, store=store)
            total = pipeline.ingest_directory()
            logger.info("initial_ingestion_complete", chunks=total)
        else:
            logger.warning(
                "api_key_not_configured",
                message=(
                    "Neither NVIDIA_API_KEY nor OPENAI_API_KEY is configured. "
                    "Skipping automatic ingestion. Set the key in .env and run 'python -m scripts.ingest'."
                ),
            )
    else:
        logger.info(
            "vector_store_populated",
            document_count=store.count(),
        )

    # ── Build the agent pipeline ─────────────────────────────
    retriever = Retriever(embedder=embedder, store=store)
    agent = ComplianceAgent(retriever=retriever)

    # ── Inject into routes ───────────────────────────────────
    configure_routes(agent=agent, store=store)

    logger.info(
        "application_ready",
        model=settings.llm_model,
        llm_base_url=settings.effective_llm_base_url,
        embedding_model=settings.effective_embedding_model,
        embedding_base_url=settings.effective_embedding_base_url,
        vector_store_count=store.count(),
    )

    yield  # ← Application serves requests here

    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """
    Application factory.

    Returns a fully configured FastAPI instance with middleware,
    routes, and lifespan management.
    """
    app = FastAPI(
        title="Mal Sharia Compliance Agent",
        description=(
            "RAG-powered AI agent for assessing whether financial products "
            "and transactions comply with Sharia law, based on AAOIFI standards."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order matters — first added = outermost) ──
    app.add_middleware(TraceIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────
    app.include_router(router)

    return app


# Create the app instance — uvicorn expects this
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Auto-reload in development
        log_level=settings.log_level.lower(),
    )
