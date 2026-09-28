"""
Request-scoped trace ID management.

Every inbound API request gets a unique trace ID (ULID) that propagates
through the entire request lifecycle — from ingestion through retrieval,
LLM call, and response. This enables:

1. Correlating all log lines for a single request
2. Debugging specific compliance verdicts end-to-end
3. Auditing which documents were consulted for each verdict

Design decision: ULID over UUID because ULIDs are:
- Lexicographically sortable by time (useful for log queries)
- Monotonically increasing within the same millisecond
- URL-safe without encoding

We use Python contextvars + structlog's merge_contextvars to propagate
the trace_id automatically. No explicit passing through function args.
"""

import structlog
from contextvars import ContextVar
from ulid import ULID


# Context variable that holds the current request's trace ID.
# Each async task / thread gets its own copy automatically.
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="no-trace")


def generate_trace_id() -> str:
    """Generate a new ULID-based trace ID and bind it to the current context."""
    trace_id = str(ULID())
    _trace_id_var.set(trace_id)

    # Bind to structlog's contextvars so every subsequent log line
    # in this request includes the trace_id without explicit passing.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    return trace_id


def get_current_trace_id() -> str:
    """Retrieve the trace ID for the current request context."""
    return _trace_id_var.get()
