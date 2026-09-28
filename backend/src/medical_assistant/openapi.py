from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.schemas import AnswerStatus, Claim, ModelName, ToolRequest, Variant
from medical_assistant.security import COOKIE_NAME

RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]


class ErrorDetail(BaseModel):
    code: str = Field(
        description="Stable application error code; use this rather than message text."
    )
    message: str = Field(
        description="Human-readable explanation; private provider audits are omitted."
    )


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class MultipartErrorResponse(BaseModel):
    detail: str = Field(
        description="Multipart parser error, such as a missing boundary, malformed part, missing field name or parser limit violation."
    )


class RunFailure(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    message: str | None = Field(
        default=None, description="Some terminal failures/cancellation store only code."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    provider: str
    model: str
    langfuse_url: str | None = Field(
        default=None, description="Omitted on ownership-loss responses."
    )
    version: str = Field(default="0.1.0", description="Omitted on ownership-loss responses.")


class ReadinessResponse(BaseModel):
    provider: str
    ready: bool = Field(description="Configuration/connection readiness, not successful inference.")
    connection: str = Field(description="Provider-specific connection check state.")
    configuration: str = Field(description="Provider-specific configuration check state.")
    inference: Literal["unverified"]
    reason: str = Field(description="Provider-specific readiness or failure reason.")


class SessionResponse(BaseModel):
    authenticated: bool
    csrf_token: str | None = Field(
        description="Send as X-CSRF-Token for cookie-authenticated mutations."
    )


class CollectionResponse(BaseModel):
    id: UUID
    title: str
    description: str
    revision: int = Field(
        description="Changes with source inventory or searchable evidence publication."
    )
    source_count: int = Field(
        description="Non-deleted sources, including not-yet-searchable sources."
    )
    ready_count: int = Field(description="Sources whose evidence is available to retrieval.")
    unavailable_count: int = Field(
        description="Pending, processing or failed sources; source_count minus ready_count."
    )
    has_pending_sources: bool
    created_at: datetime


class CollectionsResponse(BaseModel):
    items: list[CollectionResponse]


class SourceResponse(BaseModel):
    id: UUID
    collection_id: UUID
    request_id: str = Field(description="Upload idempotency key scoped to this collection.")
    title: str
    filename: str
    media_type: Literal["application/pdf", "text/plain"]
    document_class: Literal["D1", "D2", "D3", "D4", "D5", "other"]
    status: Literal["pending", "processing", "ready", "failed", "deleted"]
    sha256: str = Field(description="SHA-256 of immutable original bytes, not extracted text.")
    byte_count: int = Field(description="Original file size in bytes.")
    page_count: int | None = Field(
        description="Extraction page count; null before successful indexing."
    )
    error: str | None = Field(description="Source ingestion error, if present.")
    file_key: str = Field(
        description="Internal private-file key; fetch originals through the file endpoint."
    )
    cleanup_status: Literal["queued", "retrying", "complete"] | None
    cleanup_error: str | None
    created_at: datetime
    deleted_at: datetime | None


class SourcesResponse(BaseModel):
    items: list[SourceResponse]
    next_cursor: str | None = Field(
        description="Opaque source-list cursor; null when no older page was observed."
    )
    collection: CollectionResponse


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: UUID
    source_id: UUID
    page: int | None = Field(description="One-based PDF page; null for TXT.")
    line_start: int = Field(
        description="One-based extracted line number; PDF numbering is page-local."
    )
    line_end: int = Field(description="Inclusive ending extracted line number.")
    text: str
    bbox: tuple[float, float, float, float] | None = Field(
        description="PDF (x0, top, x1, bottom) in page points; null for TXT."
    )
    page_width: float | None = Field(description="PDF page width in points; null for TXT.")
    page_height: float | None = Field(description="PDF page height in points; null for TXT.")
    sha256: str = Field(
        description="SHA-256 of original source bytes, copied into the extracted span."
    )
    section: str


class StoredSpan(EvidenceSpan):
    collection_id: UUID
    ordinal: int = Field(description="One-based source extraction order, used by span pagination.")
    created_at: datetime


class CitationSpan(EvidenceSpan):
    source_title: str
    source_hash: str = Field(description="SHA-256 of original source bytes.")


class SpanResponse(StoredSpan):
    source_title: str
    source_hash: str
    source_filename: str
    source_media_type: Literal["application/pdf", "text/plain"]


class SpansResponse(BaseModel):
    items: list[StoredSpan]
    next_cursor: UUID | None = Field(
        description="Last returned span UUID, used as the exclusive cursor; null at the end."
    )


class ConversationResponse(BaseModel):
    id: UUID
    collection_id: UUID
    title: str
    revision: int
    created_at: datetime


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    run_id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class RunScope(BaseModel):
    id: UUID = Field(description="The run ID used to bind MCP reads to this scope.")
    collection_id: UUID
    revision: int
    source_count: int
    ready_count: int
    unavailable_count: int
    snapshot_hash: str = Field(
        description="SHA-256 of collection ID and counters; not an archive content digest."
    )


class CoverageReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")

    complete: bool = Field(
        description="Whether bounded evidence delivery certifies complete frozen-scope coverage."
    )
    inventory_count: int
    delivered_source_ids: list[UUID]
    unavailable_count: int
    truncated: bool
    reason: str
    snapshot_hash: str
    provider_input_packet_sha256: str = Field(
        description="Digest of the actual provider evidence packet."
    )
    provider_invocation_id: str | None = Field(
        default=None, description="Present after a final-answer provider invocation."
    )
    retrieval_truncated: bool | None = Field(
        default=None, description="Additional retrieved spans did not fit in the packet."
    )


class AnswerResponse(BaseModel):
    kind: Literal["answer"]
    status: AnswerStatus = Field(
        description="Evidence/answer status, independent of run execution status."
    )
    answer: str = Field(
        description="Saved final prose; claim text replaces provider prose when claims exist."
    )
    claims: list[Claim]
    limitations: list[str]
    coverage_requirement: Literal["claims", "complete_scope"]
    requests: list[ToolRequest]
    citations: list[CitationSpan] = Field(
        description="Resolved cited source spans, ordered by span ID."
    )
    coverage: CoverageReceipt | None = Field(
        default=None, description="Absent for some bounded-limit answers."
    )


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None


class RunMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    end_to_end_ms: float | None = Field(
        default=None,
        description="Dialogue execution milliseconds; excludes queue and trace delivery.",
    )
    model_calls: int | None = None
    tool_calls: int | None = None
    provider_ms: float | None = None
    tool_ms: float | None = None
    evidence_bytes: int | None = Field(
        default=None, description="Sum of UTF-8 evidence bytes delivered across answer calls."
    )
    usage: list[ModelUsage] | None = Field(
        default=None,
        description="Provider-normalized usage per model call; fields may be unavailable.",
    )
    coverage_policy: Literal["claims", "complete_scope"] | None = None
    coverage_complete: bool | None = None
    initial_candidate_chunk_ids: list[UUID] | None = None
    initial_candidate_block_span_ids: list[UUID] | None = None
    initial_provider_span_ids: list[UUID] | None = None
    final_provider_span_ids: list[UUID] | None = None
    initial_receipt_complete: bool | None = None
    final_receipt_complete: bool | None = None
    initial_method: str | None = None
    second_batch_kind: str | None = None
    guard_action: str | None = None
    retrieved_span_ids: list[UUID] | None = None
    delivered_span_ids: list[UUID] | None = None
    retrieval_stages: list[dict[str, object]] | None = Field(
        default=None,
        description="Per-search query and diagnostic retrieval observations; shape follows search method.",
    )


class RunResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    request_id: str
    question: str
    model: ModelName
    variant: Variant
    status: RunStatus
    scope: RunScope
    answer: AnswerResponse | None
    error: RunFailure | None
    trace_id: str | None
    trace_url: str | None
    trace_status: str | None = Field(
        description="Trace delivery state, independent of answer availability."
    )
    metrics: RunMetrics = Field(
        description="Empty before successful execution; diagnostics vary by workflow path."
    )
    created_at: datetime
    finished_at: datetime | None
    retry_of_run_id: UUID | None
    trace_reconcile_after: datetime | None
    trace_reconcile_attempts: int


class RunsResponse(BaseModel):
    items: list[RunResponse]
    next_cursor: str | None


class ConversationsResponse(BaseModel):
    items: list[ConversationResponse]
    next_cursor: str | None


class ConversationDetail(ConversationResponse):
    messages: list[MessageResponse] = Field(
        description="Newest first, ordered by (created_at, id) descending."
    )
    runs: list[RunResponse] = Field(
        description="Newest first, independently paginated from messages."
    )
    messages_next_cursor: str | None
    runs_next_cursor: str | None


class RunEventPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    scope: RunScope | None = None
    answer: AnswerResponse | None = None
    stage: str | None = None
    tool: str | None = None
    elapsed_ms: float | None = None
    count: int | None = None
    source_count: int | None = None
    attempt: int | None = None
    phase: str | None = None
    model: ModelName | None = None
    code: str | None = None
    message: str | None = None


class RunEvent(BaseModel):
    run_id: UUID
    seq: int = Field(description="Monotonically increasing sequence within this run.")
    type: str = Field(
        description="run.created, stage.started/completed, tool.started/completed, answer.ready or a terminal run event."
    )
    at: datetime
    payload: RunEventPayload = Field(
        description="Event-specific fields; omitted fields are not emitted as null."
    )


class StatsResponse(BaseModel):
    total_runs: int
    latency_n: int
    latency_p50_ms: float | None
    runtime_status_counts: dict[str, int]
    answer_status_counts: dict[str, int]
    trace_status_counts: dict[str, int]
    terminal_trace_status_counts: dict[str, int]
    latency_measure: str
    observed_at: datetime = Field(description="Repeatable-read snapshot time for this response.")


class Ratio(BaseModel):
    numerator: int
    denominator: int
    value: float | None = Field(
        description="Numerator / denominator; null when denominator is zero."
    )


class LatencyDistribution(BaseModel):
    n: int
    p50_ms: float | None
    p90_ms: float | None


class EvidenceAvailability(BaseModel):
    claims: Ratio
    cases: Ratio
    unknown_positive_cases: int
    no_positive_set_cases: int


class DiagnosticReceipt(BaseModel):
    complete: Ratio
    unknown_cases: int


class DiagnosticEvidence(BaseModel):
    indexed: dict[str, Ratio]
    initial_candidate_block: EvidenceAvailability
    initial_packet: EvidenceAvailability
    final_packet: EvidenceAvailability


class CostGroup(BaseModel):
    call_count: int
    estimated_call_count: int
    unestimated_call_count: int
    known_subtotal_api_usd: str = Field(
        description="Decimal USD string for the known API-equivalent subtotal."
    )
    estimated_total_api_usd: str | None = Field(
        description="Null if there are no calls or any unestimated call."
    )
    estimate_limitation_counts: dict[str, int]
    actual_billed_usd: None = Field(description="Actual subscription/provider billing is unknown.")
    billing_mode_counts: dict[str, int]


class CostSummary(BaseModel):
    generation: CostGroup
    judge: CostGroup
    price_snapshot_id: str | None


class LatestCorrection(BaseModel):
    attempted_rows: int
    completed: int
    state_counts: dict[str, int]
    confirmed_grounded_success: Ratio


class ExperimentMetrics(BaseModel):
    planned: int
    execution_state_counts: dict[str, int]
    confirmed_grounded_success: Ratio
    confirmed_grounded_success_on_assessable: Ratio
    technical_completion: Ratio
    judge_assessable: Ratio
    judge_not_assessable: Ratio
    judge_na_reason_counts: dict[str, int]
    first_turn_failure: Ratio
    answer_status_exact: Ratio
    status_confusion: dict[str, dict[str, int]]
    status_missing_answer_by_gold: dict[str, int]
    required_claim_recall_conditional: Ratio
    required_claim_present_conservative: Ratio
    required_claim_na_count: int
    own_claim_support: Ratio
    own_claim_na_count: int
    unlisted_prose_claim_count: int
    citation_edge_precision: Ratio
    citation_edge_na_count: int
    evidence_claim_availability: Ratio
    evidence_case_availability: Ratio
    evidence_unknown_positive_cases: int
    indexed_evidence_claim_availability: Ratio
    indexed_evidence_case_availability: Ratio
    diagnostic_evidence: DiagnosticEvidence
    diagnostic_policy_counts: dict[str, int]
    diagnostic_initial_method_counts: dict[str, int]
    diagnostic_second_batch_counts: dict[str, int]
    diagnostic_guard_counts: dict[str, int]
    diagnostic_initial_receipt: DiagnosticReceipt
    diagnostic_final_receipt: DiagnosticReceipt
    diagnostic_missing_cases: int
    evidence_claim_no_set_count: int
    severity_counts_assessable: dict[str, int]
    trace_status_counts: dict[str, int]
    trace_original_completed: Ratio
    trace_delivered_completed: Ratio
    latency_total_completed: LatencyDistribution
    latency_target_completed: LatencyDistribution
    latency_missing_completed_count: int
    api_equivalent_cost: CostSummary
    latest_correction: LatestCorrection


class ExperimentResponse(BaseModel):
    id: str = Field(
        description="Composite key: campaign UUID~variant~model; only V0 and V3 are exposed."
    )
    campaign_id: UUID
    corpus_id: str
    partition: str
    name: str
    status: str
    planned: int
    completed: int
    model: ModelName
    variant: Literal["V0", "V3"]
    metrics: ExperimentMetrics | None = Field(
        description="Frozen primary-attempt summary; null for an empty cell."
    )
    latest_correction: LatestCorrection = Field(
        description="Latest retry/correction summary, separate from primary comparison."
    )
    created_at: datetime
    observed_at: datetime


class ExperimentsResponse(BaseModel):
    items: list[ExperimentResponse] = Field(
        description="Four configuration cells per campaign: V0/V3 × Sol/Luna."
    )
    next_cursor: str | None = Field(description="Campaign cursor, not an individual cell cursor.")


class PrimaryAttempt(BaseModel):
    attempt_id: str | None
    run_ids: list[UUID]
    judge_request_id: str | None
    judge_verdict: str | None
    judge_operation_id: str | None
    judge_trace_id: str | None
    judge_delivery_status: str | None


class LatestAttempt(PrimaryAttempt):
    state: str
    judge_status: str | None
    retry_reason: str | None


class ExperimentCase(BaseModel):
    case_id: str
    family_id: str
    status: str = Field(description="Primary execution state used by the frozen comparison.")
    first_run_id: UUID | None
    target_run_id: UUID | None
    judge_status: str | None
    judge_verdict: str | None
    primary: PrimaryAttempt
    latest: LatestAttempt
    metrics: dict[str, object] = Field(
        description="Persisted case diagnostics/cost metadata excluding internal projection and primary records."
    )


class BreakdownMetrics(BaseModel):
    planned: int
    confirmed_grounded_success: Ratio
    technical_completion: Ratio
    judge_assessable: Ratio
    evidence_case_availability: Ratio
    evidence_unknown_positive_cases: int


class ExperimentDetail(ExperimentResponse):
    cases: list[ExperimentCase] = Field(
        description="Case-ID order; primary and latest attempts are shown separately."
    )
    breakdown: dict[str, dict[str, BreakdownMetrics]] = Field(
        description="Axis → group → metrics. Required-document-class groups can overlap."
    )


TAGS = [
    {
        "name": "System",
        "description": "Local process health and provider readiness without inference.",
    },
    {"name": "Session", "description": "Browser launch-token exchange and session/CSRF bootstrap."},
    {"name": "Collections", "description": "Shared research libraries and source inventory."},
    {"name": "Sources", "description": "Original file lifecycle and extracted citation evidence."},
    {"name": "Conversations", "description": "Collection-bound conversation history."},
    {
        "name": "Runs",
        "description": "Asynchronous questions, committed progress, saved answers and export.",
    },
    {
        "name": "Evaluation",
        "description": "Read-only saved experiment results; these endpoints do not launch evaluation.",
    },
]

DESCRIPTION = """Local research API for English native-text PDF/TXT documents and evidence-backed questions.

Workflow: exchange the launch token at `POST /api/session`; create a collection; upload sources and wait for `ready`; create a conversation; submit a run (202); follow committed SSE stages or poll the run; inspect citations or export a succeeded answer. A succeeded run can still contain a partial, conflicting, not-documented or clarification answer. Evidence is bounded; retrieval is not a validated patient cohort or exhaustive counting interface.

Browser reads require the signed `pfl_session` cookie. Browser mutations additionally require `X-CSRF-Token` from the session response. A configured service token in `Authorization: Bearer …` is an alternative for protected reads and mutations and bypasses the cookie/CSRF pair. An Authorization header takes precedence over a cookie: an invalid bearer header fails even with a valid session. Session bootstrap, health and readiness do not require either scheme. Host must match the configured application origin; if Origin is present it must also match, and cross-site/same-site Fetch Metadata requests are denied, including service requests. The interactive docs can reuse a browser session; cookie credentials are managed by the browser, and mutations need the CSRF value in Authorize. Bearer authorization is useful for a configured service client.

JSON bodies reject unknown fields. Mutation bodies are limited to 64 KiB except uploads (configured original-byte maximum plus 1 MiB multipart allowance). Session creation and upload require Content-Length. Application-generated errors use `{\"detail\": {\"code\": \"…\", \"message\": \"…\"}}`, including 422 validation errors; they do not expose Pydantic field-by-field errors. Upload multipart-parser failures can instead return HTTP 400 with `{\"detail\": \"…\"}` (a string), for example a missing boundary, malformed multipart data, a missing part field name, or parser file/field/part limits. Background failures appear in the saved run/source rather than turning the earlier accepted request into an HTTP failure.

Collection/source revisions bind each run's retrieval and publication. Changes can expire a run and withhold its answer. Run status, answer status and trace delivery are independent. Lists with opaque cursors use `(created_at, id)` descending except collections (ascending); reuse cursors only with their original filter/scope. These are keyset pages over current state, not a frozen multi-request snapshot. `next_cursor=null` means that no additional page was observed in that request.
"""

_READ_SECURITY = [{"SessionCookie": []}, {"ServiceBearer": []}]
_WRITE_SECURITY = [{"SessionCookie": [], "CSRFToken": []}, {"ServiceBearer": []}]

_OPERATIONS = {
    "health": (
        "System",
        "Check API and database health",
        "Checks API ownership and database reachability. No provider inference or telemetry delivery check. A 503 body is a health result, not an error envelope.",
        HealthResponse,
        200,
    ),
    "provider_readiness": (
        "System",
        "Inspect provider readiness without inference",
        "Reports configured provider connection/configuration state. HTTP 200 means ready; HTTP 503 means unavailable or unready. Inference remains unverified even when ready is true.",
        ReadinessResponse,
        200,
    ),
    "get_session": (
        "Session",
        "Read browser session and CSRF state",
        "Checks only the session cookie; a service bearer does not create a browser session. Returns authenticated=false and csrf_token=null for a missing/expired cookie.",
        SessionResponse,
        200,
    ),
    "start_session": (
        "Session",
        "Exchange launch token for browser session",
        "Validates the configured launch token and sets a 12-hour HttpOnly SameSite=Strict cookie at /. Secure is enabled for HTTPS origins. Return csrf_token must accompany subsequent cookie-authenticated mutations. Requires application/json and Content-Length.",
        SessionResponse,
        200,
    ),
    "list_collections": (
        "Collections",
        "List all research collections",
        "Returns all collections ordered by creation time and ID ascending; no pagination. Counters include non-deleted sources regardless of ingestion state.",
        CollectionsResponse,
        200,
    ),
    "create_collection": (
        "Collections",
        "Create a research collection",
        "Creates a collection and its search partition. Title is stripped and whitespace-only titles fail with 400. This operation has no idempotency key.",
        CollectionResponse,
        200,
    ),
    "get_collection": (
        "Collections",
        "Read collection inventory and revision",
        "Returns current counters and whether ingestion is pending or processing. Inventory pagination never restricts the collection searched by a run.",
        CollectionResponse,
        200,
    ),
    "list_sources": (
        "Sources",
        "Page through non-deleted source inventory",
        "Returns newest sources first with current collection counters. Sources in pending/processing/failed states are included; deleted sources are excluded. limit=1..100, default 100. Use the returned opaque cursor only for this collection.",
        SourcesResponse,
        200,
    ),
    "upload_source": (
        "Sources",
        "Accept and queue a PDF or TXT source",
        "Persists immutable original bytes and queues ingestion; 202 is acceptance, not search readiness. Filename extension, declared media type and PDF signature must agree; TXT must be UTF-8 (optional BOM) without NUL bytes. Native-text PDF/TXT only; no OCR. Requires Content-Length. request_id (1..100 characters) is scoped to the collection: identical bytes and metadata return the existing source, even if already deleted; changed content/metadata returns 409. document_class is D1 clinical notes, D2 laboratory/microbiology, D3 medication records, D4 discharge, D5 imaging or other; it is metadata, not a retrieval filter.",
        SourceResponse,
        202,
    ),
    "get_source": (
        "Sources",
        "Read source ingestion and cleanup state",
        "Returns the source record including a tombstone after deletion. Only ready sources are searchable. cleanup_status separately describes deferred deletion of private bytes.",
        SourceResponse,
        200,
    ),
    "delete_source": (
        "Sources",
        "Remove source evidence and queue private-file cleanup",
        "Marks a source deleted, removes indexed spans/chunks and updates collection revision/counters transactionally. Returns the source record; file cleanup is asynchronous. Repeated deletion can requeue incomplete cleanup. Text already sent to a provider cannot be retracted.",
        SourceResponse,
        200,
    ),
    "source_file": (
        "Sources",
        "Read integrity-checked original source bytes",
        "Returns the immutable original PDF or TXT inline after byte-count and SHA-256 checks; ingestion need not be ready. Deleted, missing or modified bytes return 410. The UTF-8 filename is supplied in Content-Disposition; private responses are not cached.",
        None,
        200,
    ),
    "source_spans": (
        "Sources",
        "Page source spans or open a citation anchor",
        "Only ready, non-deleted evidence is returned in extraction ordinal ascending; an unready source returns an empty page. limit=1..100, default 100. cursor is an exclusive span UUID; anchor is an inclusive span UUID so the first item is the cited span. They are mutually exclusive and must refer to a span of this source. This uses span UUIDs, not opaque list cursors.",
        SpansResponse,
        200,
    ),
    "get_span": (
        "Sources",
        "Resolve a citation to current extracted evidence",
        "Returns a stored span plus original source metadata. A span is visible only while its source is ready and non-deleted; unavailable/deleted evidence returns 404. PDF coordinates support opening/highlighting the original page.",
        SpanResponse,
        200,
    ),
    "list_conversations": (
        "Conversations",
        "Page conversations across libraries or one collection",
        "Returns newest conversations first. Optional collection_id selects one collection; a valid but unknown collection yields an empty list. limit=1..100, default 100. Cursor must match the collection filter.",
        ConversationsResponse,
        200,
    ),
    "create_conversation": (
        "Conversations",
        "Create a collection-bound conversation",
        "Creates an empty conversation in an existing collection. New questions search that collection's entire ready library. This operation has no idempotency key.",
        ConversationResponse,
        200,
    ),
    "get_conversation": (
        "Conversations",
        "Read independently paginated messages and runs",
        "Returns current conversation metadata plus newest messages/runs first. messages_limit defaults to 100 and runs_limit to 50; both allow 1..100. Each list has its own opaque cursor and next cursor. Their reads are independent and do not promise one shared database snapshot.",
        ConversationDetail,
        200,
    ),
    "create_run": (
        "Runs",
        "Accept an asynchronous question or explicit retry",
        "Freezes collection revision/counters, saves the user message and queues background dialogue. request_id is scoped to the conversation: identical content returns the existing run without another message; reuse with different question/model/variant/retry gives 409. Only one queued/running run per conversation is allowed. retry_of_run_id must be terminal and in this conversation (terminal includes failed/cancelled/interrupted); retry uses the retry ancestor's prior history boundary. API-wide capacity rejection is 429 with Retry-After: 1. Collection changes or cancellation can withhold the answer after acceptance; restart interrupts unfinished runs rather than automatically retrying.",
        RunResponse,
        202,
    ),
    "list_runs": (
        "Runs",
        "Page saved runs by execution status",
        "Returns newest saved runs first. limit=1..100, default 100; status is all, queued, running, succeeded, failed, cancelled or interrupted. Cursor is bound to the status filter. Execution status is separate from answer quality and trace delivery.",
        RunsResponse,
        200,
    ),
    "get_run": (
        "Runs",
        "Read saved question, execution state and answer",
        "Poll this record after acceptance, or fetch it after an SSE terminal event. answer is null until publication; failed/cancelled/interrupted runs retain state without a successful answer. metrics are empty before successful completion. Trace reconciliation can update trace fields later.",
        RunResponse,
        200,
    ),
    "stream_events": (
        "Runs",
        "Replay and follow committed run events (SSE)",
        "Returns text/event-stream in increasing seq order. Each frame uses id: <run UUID>:<seq>, event: <type>, and data: JSON RunEvent. after defaults to 0 and must be nonnegative; Last-Event-ID must match this run and resumes after max(after, header sequence). Events are committed progress and a complete answer.ready payload, not token streaming. Keepalive comments are emitted after about 15 seconds without frames. The stream drains stored events then closes once the run is terminal; reconnect can replay missed events. A disconnected client does not cancel the run.",
        None,
        200,
    ),
    "cancel_run": (
        "Runs",
        "Cancel an active run and return current state",
        "Marks queued/running work cancelled, requests provider cancellation and stops the local task. A terminal run is returned unchanged. Provider cancellation failure is privately audited and does not turn successful cancellation into an HTTP error; cancellation does not retract provider inputs or finished work.",
        RunResponse,
        200,
    ),
    "export_run": (
        "Runs",
        "Download a succeeded answer as Markdown",
        "Returns UTF-8 Markdown with the question, saved answer, source/span identifiers and limitations. Only execution status succeeded is required; an answer can be partial or need clarification. Other run states return 409. Content-Disposition supplies run-<UUID>.md and the response is private/no-store.",
        None,
        200,
    ),
    "stats": (
        "Evaluation",
        "Read global run, answer and trace statistics",
        "Uses one read-only repeatable-read snapshot. Latency includes only succeeded runs with numeric end_to_end_ms; p50 is null when none exist. Measures dialogue execution, excluding queue/trace delivery. Counts are global to this local database; runtime, answer and trace denominators differ.",
        StatsResponse,
        200,
    ),
    "experiments": (
        "Evaluation",
        "Page saved campaign configuration summaries",
        "limit counts campaigns (1..25, default 10), not cells: each campaign contributes four V0/V3 × Sol/Luna items. Campaigns are newest first; cursor identifies the last campaign. Summaries use frozen primary attempts, while latest_correction shows retry outcomes separately. Invalid stored campaign/projection data returns 409; no evaluation or model call is started.",
        ExperimentsResponse,
        200,
    ),
    "experiment": (
        "Evaluation",
        "Inspect one experiment cell, cases and metric slices",
        "experiment_id is campaign UUID~V0-or-V3~gpt-6-sol-or-gpt-6-luna. Returns cases in case-ID order, primary and latest attempt references, and breakdowns from the frozen primary comparison. Required-document-class slices overlap. Ratios expose numerator/denominator and null for zero denominator; latency is in milliseconds and cost is API-equivalent USD, not actual subscription billing. Invalid stored data returns 409.",
        ExperimentDetail,
        200,
    ),
}

_ERRORS = {
    400: "Invalid request values, metadata, cursor, page size or retry reference; code identifies the cause.",
    401: "Authentication required, invalid bearer token, or invalid launch token on session creation.",
    403: "Host/Origin/Fetch Metadata denied, or invalid/missing CSRF for a cookie-authenticated mutation.",
    404: "Resource absent, malformed UUID, or citation evidence not currently available.",
    409: "Domain conflict, request ID reused with different content, incomplete run export, or invalid saved experiment data.",
    410: "Source deleted, original bytes unavailable, or original integrity check failed.",
    411: "Content-Length is required for session creation and upload.",
    413: "Configured file limit or request body limit exceeded.",
    415: "Unsupported original format/encoding, or non-JSON session body.",
    422: "FastAPI request parsing/validation failed; returns the application's invalid_request envelope.",
    429: "Active run capacity exceeded. Retry-After is 1 second.",
    503: "API ownership lost; protected mutation cannot proceed.",
}

_SPECIFIC_ERRORS = {
    "health": (),
    "provider_readiness": (),
    "get_session": (),
    "start_session": (400, 401, 411, 413, 415, 422),
    "list_collections": (),
    "create_collection": (400,),
    "get_collection": (404,),
    "list_sources": (400, 404),
    "upload_source": (400, 404, 409, 411, 415),
    "get_source": (404,),
    "delete_source": (404,),
    "source_file": (404, 410),
    "source_spans": (400, 404, 410),
    "get_span": (404,),
    "list_conversations": (400, 404),
    "create_conversation": (404,),
    "get_conversation": (400, 404),
    "create_run": (400, 404, 409, 429),
    "list_runs": (400,),
    "get_run": (404,),
    "stream_events": (400, 404),
    "cancel_run": (404,),
    "export_run": (404, 409),
    "stats": (),
    "experiments": (400, 409),
    "experiment": (404, 409),
}

_PUBLIC = {"health", "provider_readiness", "get_session", "start_session"}
_MUTATIONS = {
    "start_session",
    "create_collection",
    "upload_source",
    "delete_source",
    "create_conversation",
    "create_run",
    "cancel_run",
}


def operation(name):
    tag, summary, description, model, status = _OPERATIONS[name]
    errors = {403, *_SPECIFIC_ERRORS[name]}
    if name not in _PUBLIC:
        errors.add(401)
    if name in _MUTATIONS and name != "start_session":
        errors.update((400, 413, 503))
    if name in {"create_collection", "create_conversation", "create_run", "upload_source"}:
        errors.add(422)
    responses = {
        code: {"model": ErrorResponse, "description": _ERRORS[code]} for code in sorted(errors)
    }
    if name == "upload_source":
        responses[400] = {
            "model": ErrorResponse | MultipartErrorResponse,
            "description": "Application validation errors use detail:{code,message}; multipart-parser errors use detail:string (missing boundary, malformed multipart data, missing field name or parser limits).",
        }
    if model is not None:
        responses[status] = {"model": model, "description": summary}
    if name in {"health", "provider_readiness"}:
        responses[503] = {
            "model": model,
            "description": "Degraded/unready result; same shape as success, with readiness false or status degraded.",
        }
    if name == "create_run":
        responses[429]["headers"] = {
            "Retry-After": {"schema": {"type": "string"}, "description": "1 (seconds)."}
        }
    if name == "start_session":
        responses[200]["headers"] = {
            "Set-Cookie": {
                "schema": {"type": "string"},
                "description": "Signed pfl_session cookie; HttpOnly, SameSite=Strict, Path=/, Max-Age=43200; Secure on HTTPS.",
            }
        }
    security = [] if name in _PUBLIC else _WRITE_SECURITY if name in _MUTATIONS else _READ_SECURITY
    return {
        "tags": [tag],
        "summary": summary,
        "description": description,
        "responses": responses,
        "openapi_extra": {"security": security},
    }


def configure_openapi(app):
    def schema():
        if app.openapi_schema is not None:
            return app.openapi_schema
        document = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            openapi_version=app.openapi_version,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        components = document.setdefault("components", {})
        components["securitySchemes"] = {
            "SessionCookie": {
                "type": "apiKey",
                "in": "cookie",
                "name": COOKIE_NAME,
                "description": "12-hour signed browser session from POST /api/session. Browser manages this HttpOnly cookie.",
            },
            "CSRFToken": {
                "type": "apiKey",
                "in": "header",
                "name": "X-CSRF-Token",
                "description": "Session-specific csrf_token from GET/POST /api/session; required with cookie authentication for mutations.",
            },
            "ServiceBearer": {
                "type": "http",
                "scheme": "bearer",
                "description": "Configured PFL_SERVICE_TOKEN. Alternative to browser cookie/CSRF; Host/Origin guards still apply.",
            },
        }
        event_schema = RunEvent.model_json_schema(ref_template="#/components/schemas/{model}")
        components["schemas"].update(event_schema.pop("$defs", {}))
        components["schemas"]["RunEvent"] = event_schema
        upload = document["paths"]["/api/collections/{collection_id}/sources"]["post"]
        body_schema = upload["requestBody"]["content"]["multipart/form-data"]["schema"]
        upload_fields = components["schemas"][body_schema["$ref"].rsplit("/", 1)[1]]["properties"]
        upload_fields["file"]["description"] = (
            "Required original native-text PDF or UTF-8 TXT. Extension and declared type must agree; "
            "original byte count must not exceed the configured upload limit."
        )
        upload_fields["document_class"]["description"] = (
            "D1 clinical notes, D2 laboratory/microbiology, D3 medication records, "
            "D4 discharge, D5 imaging or other (default). Metadata only; not a retrieval filter."
        )
        upload_fields["request_id"]["description"] = (
            "Required 1..100-character upload idempotency key, scoped to collection. "
            "Identical bytes/metadata reuse the source; changed content/metadata returns 409."
        )
        for path, methods in document["paths"].items():
            for method, entry in methods.items():
                if method not in {"get", "post", "delete", "put", "patch"}:
                    continue
                responses = entry["responses"]
                if "422" in responses:
                    responses["422"] = {
                        "description": _ERRORS[422],
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    }
                for parameter in entry.get("parameters", []):
                    name = parameter["name"]
                    if parameter["in"] == "path":
                        parameter["description"] = (
                            "Composite campaign UUID~variant~model key."
                            if name == "experiment_id"
                            else "Resource UUID; malformed IDs return 404, not a UUID validation error."
                        )
                    elif name in {"limit", "messages_limit", "runs_limit"}:
                        maximum = 25 if path == "/api/experiments" else 100
                        parameter["description"] = (
                            f"Page size from 1 to {maximum}; out-of-range values return 400, non-integers return 422."
                        )
                    elif name == "after":
                        parameter["description"] = (
                            "Exclusive nonnegative event sequence; combined with Last-Event-ID using the larger value."
                        )
                    elif name == "status":
                        parameter["description"] = (
                            "all, queued, running, succeeded, failed, cancelled or interrupted; invalid strings return 400."
                        )
                    elif name == "collection_id":
                        parameter["description"] = (
                            "Optional collection UUID filter; cursor is tied to this filter."
                        )
                    elif "cursor" in name:
                        parameter["description"] = (
                            "Exclusive span UUID in this source; mutually exclusive with anchor."
                            if path.endswith("/spans")
                            else "Opaque cursor from the matching next_cursor field; empty starts at newest page, reuse only in its original scope/filter."
                        )
                    elif name == "anchor":
                        parameter["description"] = (
                            "Inclusive span UUID in this source; mutually exclusive with cursor."
                        )
                if path.endswith("/file"):
                    responses["200"] = {
                        "description": "Integrity-checked original bytes.",
                        "content": {
                            "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                            "text/plain": {"schema": {"type": "string", "format": "binary"}},
                        },
                        "headers": {
                            "Content-Disposition": {
                                "schema": {"type": "string"},
                                "description": "inline; filename*=UTF-8''<percent-encoded filename>",
                            }
                        },
                    }
                elif path.endswith("/export"):
                    responses["200"] = {
                        "description": "UTF-8 Markdown answer attachment.",
                        "content": {"text/markdown": {"schema": {"type": "string"}}},
                        "headers": {
                            "Content-Disposition": {
                                "schema": {"type": "string"},
                                "description": "attachment; filename=run-<UUID>.md",
                            }
                        },
                    }
                elif path.endswith("/events"):
                    responses["200"] = {
                        "description": "SSE frames; data is a JSON RunEvent. See x-event-schema for the decoded contract.",
                        "content": {
                            "text/event-stream": {
                                "schema": {"type": "string"},
                                "example": 'id: 00000000-0000-0000-0000-000000000001:1\nevent: run.created\ndata: {"run_id":"00000000-0000-0000-0000-000000000001","seq":1,"type":"run.created","at":"2026-09-28T00:00:00Z","payload":{"scope":{"id":"00000000-0000-0000-0000-000000000001","collection_id":"00000000-0000-0000-0000-000000000002","revision":0,"source_count":0,"ready_count":0,"unavailable_count":0,"snapshot_hash":"<sha256>"}}}\n\n',
                            }
                        },
                        "x-event-schema": {"$ref": "#/components/schemas/RunEvent"},
                    }
                    entry.setdefault("parameters", []).append(
                        {
                            "name": "Last-Event-ID",
                            "in": "header",
                            "required": False,
                            "description": "Resume ID in <this run UUID>:<nonnegative seq> form. Malformed/mismatched ID returns 400 (malformed UUID can return 404).",
                            "schema": {"type": "string"},
                        }
                    )
        components["schemas"].pop("HTTPValidationError", None)
        components["schemas"].pop("ValidationError", None)
        app.openapi_schema = document
        return document

    app.openapi = schema
