"""
Request middleware for trace ID injection.

Every inbound request gets a unique trace ID assigned before any
business logic runs. This trace ID is:
1. Set in contextvars (auto-propagates to all log lines)
2. Returned in the X-Trace-ID response header
3. Included in the response body for client-side correlation

Design decision: Middleware vs. dependency injection
We use middleware rather than FastAPI's Depends() because:
- Middleware runs BEFORE route matching (catches 404s, validation errors too)
- The trace ID is set in contextvars, not passed as a parameter
- It applies universally — no risk of forgetting to inject it in a new route
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.observability.tracer import generate_trace_id
from src.observability.logger import get_logger

logger = get_logger(__name__)


class TraceIDMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique trace ID into every request lifecycle.

    The trace ID is:
    - Generated as a ULID (time-sortable, unique)
    - Bound to structlog's contextvars for automatic log enrichment
    - Added to the response as X-Trace-ID header
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        trace_id = generate_trace_id()

        logger.info(
            "request_received",
            method=request.method,
            path=str(request.url.path),
            client=request.client.host if request.client else "unknown",
        )

        response: Response = await call_next(request)

        # Include trace ID in response header for client-side correlation
        response.headers["X-Trace-ID"] = trace_id

        logger.info(
            "request_completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
        )

        return response
