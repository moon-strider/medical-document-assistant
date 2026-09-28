import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.schemas import ModelAction


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AnswerOutput(ModelAction):
    model_config = ConfigDict(extra="forbid", strict=True)


class JudgeScores(StrictModel):
    claim_support: int = Field(ge=0, le=4)
    completeness: int = Field(ge=0, le=4)
    contradictions: int = Field(ge=0, le=4)
    temporal_subject_units: int = Field(ge=0, le=4)
    coverage: int = Field(ge=0, le=4)


class ClaimReview(StrictModel):
    claim_index: int = Field(ge=-1)
    claim_text: str
    evidence_ids: list[str]
    support_label: Literal["supported", "partial", "unsupported", "not_assessable"]
    reason: str


class RequiredClaimReview(StrictModel):
    gold_claim_id: str
    label: Literal["present", "partial", "missing", "not_assessable"]
    evidence_ids: list[str]
    reason: str


class CitationReview(StrictModel):
    claim_index: int = Field(ge=-1)
    evidence_id: str
    label: Literal["useful", "redundant", "irrelevant", "not_assessable"]
    reason: str


class JudgeOutput(StrictModel):
    verdict: Literal["pass", "fail", "not_assessable"]
    severity: Literal["none", "S0", "S1", "S2"]
    status_correct: bool
    coverage_valid: bool
    scores: JudgeScores
    claim_reviews: list[ClaimReview]
    required_claim_reviews: list[RequiredClaimReview]
    citation_reviews: list[CitationReview]
    reason: str


SCHEMA_MODELS = {"answer": AnswerOutput, "judge": JudgeOutput}


def validate_output(role: str, output: dict, final_only: bool = False) -> dict:
    """Validate a provider object before the graph or evaluator consumes it.

    Apply the role schema, then reject answer/tool combinations that would make
    control flow ambiguous. Evidence support and task-specific tool arguments
    remain the graph's responsibility; judging semantics remain the evaluator's.
    Answer status describes evidence support rather than provider execution.
    Tool requests instruct the graph; this validation does not execute them.

    Args:
        role (str): "answer" or "judge"; query planning uses "answer".
        output (dict): Required answer fields are kind ("answer" or
            "request_tools"), status (supported, partial, conflicting,
            not_documented or needs_clarification), answer (str), claims
            (list of {text: str, evidence_ids: list[str]}), limitations (list[str]),
            coverage_requirement ("claims" or "complete_scope") and requests
            (list of {tool: search_evidence/read_evidence/collect_scope, query: str,
            span_ids: list[str]}). Required judge fields are verdict
            (pass/fail/not_assessable), severity (none/S0/S1/S2), status_correct
            and coverage_valid (bool), reason (str), scores (mapping with integer
            0–4 claim_support/completeness/contradictions/temporal_subject_units/
            coverage), and three review lists. claim_reviews contain claim_index
            (int >= -1), claim_text (str), evidence_ids (list[str]), support_label
            (supported/partial/unsupported/not_assessable) and reason (str).
            required_claim_reviews contain gold_claim_id (str), evidence_ids
            (list[str]), label (present/partial/missing/not_assessable) and reason
            (str). citation_reviews contain claim_index (int >= -1), evidence_id
            (str), label (useful/redundant/irrelevant/not_assessable) and reason
            (str). Shape validation does not resolve identifiers against evidence,
            establish complete coverage or prove a judge's conclusions.
        final_only (bool): For the answer role, require kind="answer". This flag
            adds no constraint to the judge schema.

    Returns:
        dict: A validated model dump without modifying the supplied mapping.
        An answer has no requests; a request_tools action has at least one.

    Raises:
        pydantic.ValidationError: The object violates the selected role schema.
        ValueError: The role is unsupported, final-only inference requested tools,
            a final answer includes requests, or a tool action has no requests.
    """

    if role not in SCHEMA_MODELS:
        raise ValueError("unsupported role")
    validated = SCHEMA_MODELS[role].model_validate(output)
    if role == "answer":
        if final_only and validated.kind != "answer":
            raise ValueError("final-only inference requested tools")
        if validated.kind == "answer" and validated.requests:
            raise ValueError("final answer included tool requests")
        if validated.kind == "request_tools" and not validated.requests:
            raise ValueError("tool request contains no requests")
    return validated.model_dump()


def schema_for(role: str) -> dict:
    if role not in SCHEMA_MODELS:
        raise ValueError("unsupported role")
    schema = SCHEMA_MODELS[role].model_json_schema()
    if role == "answer":
        pending = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                node.pop("maxLength", None)
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)
    return schema


def schema_path(role: str) -> Path:
    root = Path(__file__).resolve().parents[4]
    return root / "prompts" / "schemas" / f"{role}.json"


def load_schema(role: str) -> dict:
    path = schema_path(role)
    schema = json.loads(path.read_text(encoding="utf-8"))
    if schema != schema_for(role):
        raise ValueError("provider output schema does not match code")
    return schema
