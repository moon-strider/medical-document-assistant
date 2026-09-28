from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["gpt-6-sol", "gpt-6-luna"]
Variant = Literal["V0", "V1", "V2", "V3"]
AnswerStatus = Literal[
    "supported", "partial", "conflicting", "not_documented", "needs_clarification"
]
MAX_SEARCH_QUERY_LENGTH = 512


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CollectionInput(StrictModel):
    title: str = Field(
        min_length=1,
        max_length=160,
        description="Library title; storage strips surrounding whitespace and rejects whitespace-only titles.",
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Optional researcher-facing description; not a retrieval filter.",
    )


class ConversationInput(StrictModel):
    collection_id: str = Field(
        min_length=1,
        max_length=100,
        description="Existing collection UUID. All ready sources in this library are eligible for each question.",
    )
    title: str = Field(
        default="New conversation", max_length=160, description="Conversation label; may be empty."
    )


class RunInput(StrictModel):
    request_id: str = Field(
        min_length=8,
        max_length=100,
        description="Idempotency key scoped to this conversation. Identical payload returns the existing run; changed payload returns 409.",
    )
    question: str = Field(
        min_length=1,
        max_length=8000,
        description="English document question or follow-up. Whitespace-only input is rejected by storage. The supported language is a workflow constraint, not auto-detection.",
    )
    model: ModelName = Field(default="gpt-6-sol", description="Generator model for this run.")
    variant: Variant = Field(
        default="V3",
        description="V0: dense fixed chunks; V1: hybrid structural chunks; V2: hybrid plus one tool round; V3: hybrid, reranking and one tool round.",
    )
    retry_of_run_id: str | None = Field(
        default=None,
        description="Explicit retry of a terminal run in this conversation; null for a normal new question. A retry uses its ancestor's prior-message boundary.",
    )


class SessionInput(StrictModel):
    token: str = Field(
        min_length=16,
        max_length=256,
        description="Configured PFL_LAUNCH_TOKEN, exchanged for an HttpOnly browser cookie and session-specific CSRF token. This is not the service bearer token.",
    )


class Claim(StrictModel):
    text: str = Field(max_length=8000)
    evidence_ids: list[str] = Field(max_length=40)


class ToolRequest(StrictModel):
    tool: Literal["search_evidence", "read_evidence", "collect_scope"]
    query: str = Field(
        max_length=MAX_SEARCH_QUERY_LENGTH,
        json_schema_extra={"pattern": rf"^[\s\S]{{0,{MAX_SEARCH_QUERY_LENGTH}}}$"},
    )
    span_ids: list[str] = Field(max_length=48)


class ModelAction(StrictModel):
    kind: Literal["answer", "request_tools"]
    status: AnswerStatus
    answer: str = Field(max_length=32000)
    claims: list[Claim] = Field(max_length=80)
    limitations: list[str] = Field(max_length=30)
    coverage_requirement: Literal["claims", "complete_scope"]
    requests: list[ToolRequest] = Field(max_length=2)
