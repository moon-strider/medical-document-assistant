import logging
import os
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from medical_assistant.retrieval import Retrieval, RetrievalError
from medical_assistant.settings import get_settings


class StrictResult(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(StrictResult):
    chunk_id: str
    source_id: str
    source_title: str
    source_hash: str
    span_ids: list[str]
    snippet: str
    score: float
    branches: list[str]
    rerank_score: float | None = None


class QueryStageMs(StrictResult):
    embedding: float
    dense: float
    bm25: float
    literal: float
    phrase: float
    fusion: float


class RetrievalStats(StrictResult):
    dense_candidates: int
    bm25_candidates: int
    literal_candidates: int
    phrase_candidates: int
    rrf_pool_size: int
    query_stage_ms: QueryStageMs
    index_scope: Literal["collection_partition"]
    search_ms: float


class RerankStats(StrictResult):
    status: Literal["ok", "skipped_empty"]
    model: str
    revision: str
    input_count: int
    scored_pairs: int
    truncated_pairs: int
    elapsed_ms: float


class SearchResult(StrictResult):
    scope_applied: str
    query: str
    variant: str
    candidates: list[Candidate]
    no_candidates: bool
    retrieval: RetrievalStats
    rerank: RerankStats | None = None


class SpanRecord(StrictResult):
    id: str
    source_id: str
    page: int | None
    line_start: int | None
    line_end: int | None
    text: str
    bbox: list[float] | None
    page_width: float | None
    page_height: float | None
    sha256: str
    section: str | None
    source_title: str
    source_hash: str
    fragment_index: int | None = None
    fragment_count: int | None = None


class ReadResult(StrictResult):
    scope_applied: str
    spans: list[SpanRecord]
    delivered_source_ids: list[str]
    text_bytes: int
    truncated: bool
    inventory_complete: bool
    inventory_count: int
    unavailable_count: int
    span_inventory_count: int | None
    span_inventory_sha256: str | None
    snapshot_hash: str


class CollectResult(StrictResult):
    scope_applied: str
    spans: list[SpanRecord]
    cursor: str
    next_cursor: str
    inventory_count: int
    delivered_source_ids: list[str]
    unavailable_count: int
    inventory_complete: bool
    truncated: bool
    snapshot_hash: str
    text_bytes: int


_server = MCPServer(name="medical-document-evidence", version="0.1.0")
_LOGGER = logging.getLogger(__name__)
_retrieval: Retrieval | None = None
_READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False, destructiveHint=False)


def _service() -> Retrieval:
    global _retrieval
    if _retrieval is None:
        settings = get_settings()
        cursor_key = os.environ.get("PFL_MCP_CURSOR_KEY")
        _retrieval = Retrieval(
            settings.read_database_url,
            cursor_key=bytes.fromhex(cursor_key) if cursor_key is not None else None,
        )
    return _retrieval


def _invoke(method: str, **arguments: Any) -> dict:
    try:
        return getattr(_service(), method)(**arguments)
    except RetrievalError as exc:
        raise ToolError(exc.code) from exc
    except Exception as exc:
        _LOGGER.error("mcp_tool_failed tool=%s cause=%s", method, type(exc).__name__)
        raise ToolError("retrieval_failed") from exc


@_server.tool(
    description="Find ranked evidence candidates within an immutable run scope.",
    annotations=_READ_ONLY,
    structured_output=True,
)
def search_evidence(
    run_scope_id: str, query: str, variant: str = "V3", limit: int = 12
) -> SearchResult:
    return SearchResult.model_validate(
        _invoke(
            "search_evidence",
            run_scope_id=run_scope_id,
            query=query,
            variant=variant,
            limit=limit,
        )
    )


@_server.tool(
    description="Read original source spans by IDs after run scope and revision checks.",
    annotations=_READ_ONLY,
    structured_output=True,
)
def read_evidence(run_scope_id: str, span_ids: list[str], budget_bytes: int = 16384) -> ReadResult:
    return ReadResult.model_validate(
        _invoke(
            "read_evidence",
            run_scope_id=run_scope_id,
            span_ids=span_ids,
            budget_bytes=budget_bytes,
        )
    )


@_server.tool(
    description="Page through every raw span in a frozen run scope, up to 8192 UTF-8 text bytes per page.",
    annotations=_READ_ONLY,
    structured_output=True,
)
def collect_scope(run_scope_id: str, cursor: str = "", page_bytes: int = 8192) -> CollectResult:
    return CollectResult.model_validate(
        _invoke("collect_scope", run_scope_id=run_scope_id, cursor=cursor, page_bytes=page_bytes)
    )


def main() -> None:
    _service()
    _server.run(transport="stdio")


if __name__ == "__main__":
    main()
