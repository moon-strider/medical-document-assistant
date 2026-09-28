import asyncio
import json
import os
import time

from openai import APITimeoutError, AsyncOpenAI
from pydantic import ValidationError

from .common import ProviderError, check_request, instruction_hash, instruction_paths, prompt_for
from .contracts import load_schema, validate_output


def normalize_response_usage(raw_usage):
    """Expose API token components while preserving unknown counts as None.

    Args:
        raw_usage (dict | object): Responses usage mapping with input_tokens,
            output_tokens and optional input_tokens_details/output_tokens_details.

    Returns:
        dict: input_tokens, cached_input_tokens, cache_write_input_tokens,
        output_tokens and reasoning_output_tokens, or {} for a non-mapping input.
        Absent components are None rather than zero. Cached tokens are a component
        of input and reasoning tokens a component of output, so consumers must not
        add them to the totals a second time. No monetary cost is computed here.
    """

    if not isinstance(raw_usage, dict):
        return {}
    input_details = raw_usage.get("input_tokens_details") or {}
    output_details = raw_usage.get("output_tokens_details") or {}
    return {
        "input_tokens": raw_usage.get("input_tokens"),
        "cached_input_tokens": input_details.get("cached_tokens"),
        "cache_write_input_tokens": input_details.get("cache_write_tokens"),
        "output_tokens": raw_usage.get("output_tokens"),
        "reasoning_output_tokens": output_details.get("reasoning_tokens"),
    }


class APIProvider:
    """Run explicit API-billed inference with task-local cancellation tracking."""

    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client
        self.active: dict[str, asyncio.Task] = {}

    async def readiness(self) -> dict:
        """Describe API credential presence without contacting the API.

        Returns:
            dict: provider="api", ready (bool), connection="not_checked",
            configuration ("present" or "missing"), inference="unverified" and
            reason ("configured" or "api_key_missing"). ready only reflects a
            nonempty OPENAI_API_KEY, even when an injected client is available;
            it proves neither credential validity nor model access.
        """

        configured = bool(os.environ.get("OPENAI_API_KEY"))
        return {
            "provider": "api",
            "ready": configured,
            "connection": "not_checked",
            "configuration": "present" if configured else "missing",
            "inference": "unverified",
            "reason": "configured" if configured else "api_key_missing",
        }

    async def generate(
        self,
        payload: dict,
        role: str = "answer",
        model: str = "gpt-6-sol",
        request_id: str = "",
        final_only: bool = False,
    ) -> dict:
        """Request one structured answer, query plan or judge verdict via Responses.

        Validate admission and the installed role schema, send role instructions
        plus the untrusted task packet with tools disabled, and accept only a
        completed, schema-valid response. Lazily create and retain the API client.
        The default client disables retries; an injected client retains its own
        policy. Reusing a finished request ID starts another potentially billable
        call: active-ID rejection is not durable deduplication or idempotency.

        Args:
            payload (dict): JSON task packet admitted by check_request. Answer
                packets carry question, history and evidence.spans; planning uses
                task_mode="query_planning"; judge packets carry candidate, gold,
                source excerpts and rubric_version. These input fields are not
                schema-validated by the adapter.
            role (str): "answer" for answers/planning or "judge" for evaluation.
            model (str): gpt-6-sol, or gpt-6-luna for the answer role only.
            request_id (str): Valid nonempty identifier, unique among active calls
                on this instance. Used for local cancellation, not API idempotency.
            final_only (bool): Reject an answer-role tool request when True.

        Returns:
            dict: output (validated role mapping), usage (token components with
            None for missing counts), raw_provider_usage (API mapping), elapsed_ms
            (integer milliseconds), provider="api", model, role, instruction_hash,
            billing_mode="api", cost_state="unknown", raw_visible_events=[] and
            input_packet (the original payload). A completed transport response
            does not establish the answer's factual status or citation support.
            cost_state describes missing monetary cost, not missing token usage.

        Raises:
            ProviderError: Admission fails, credentials are missing, an ID is
                active, inference times out, the API response is non-completed,
                refuses, is empty or violates the role contract. Available usage
                and response status/reason survive response-validation failures.
            asyncio.CancelledError: The active task is cancelled, including by
                cancel(). Cleanup removes its ID; remote work or billing may
                already have occurred and no remote cancellation is requested.
            openai.APIError: Other API client failures propagate unchanged.
            OSError: A role instruction or schema file cannot be read.
            ValueError: The installed output schema differs from the code schema.
        """

        check_request(payload, role, model, request_id)
        schema = load_schema(role)
        if self.client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ProviderError("api_key_missing", "API provider requires an explicit API key")
            self.client = AsyncOpenAI(max_retries=0, timeout=self.settings.provider_timeout_seconds)
        agents, base = instruction_paths(role)
        prompt = prompt_for(payload, role, final_only)
        started = time.monotonic()
        task = asyncio.current_task()
        if request_id in self.active:
            raise ProviderError("duplicate_request_id", "request_id is already active")
        self.active[request_id] = task
        try:
            try:
                async with asyncio.timeout(self.settings.provider_timeout_seconds):
                    response = await self.client.responses.create(
                        model=model,
                        reasoning={"effort": "high"},
                        instructions=base.read_text(encoding="utf-8")
                        + "\n"
                        + agents.read_text(encoding="utf-8"),
                        input=prompt,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": f"pfl_{role}",
                                "schema": schema,
                                "strict": True,
                            }
                        },
                        tools=[],
                        store=False,
                        timeout=self.settings.provider_timeout_seconds,
                    )
            except (TimeoutError, APITimeoutError) as exc:
                raise ProviderError("timeout", "API inference timed out") from exc
            raw_usage = response.usage.model_dump() if response.usage else {}
            usage = normalize_response_usage(raw_usage)
            status = getattr(response, "status", None)
            error = getattr(response, "error", None)
            incomplete = getattr(response, "incomplete_details", None)
            reason = getattr(error, "code", None) or getattr(incomplete, "reason", None)
            if status not in {
                "completed",
                "failed",
                "incomplete",
                "cancelled",
                "queued",
                "in_progress",
            }:
                status = "unknown"
            if (
                not isinstance(reason, str)
                or len(reason) > 64
                or not reason.replace("_", "").isalnum()
            ):
                reason = "unknown"
            if status != "completed":
                raise ProviderError(
                    "response_incomplete",
                    "API response did not complete",
                    usage=usage,
                    raw_provider_usage=raw_usage,
                    response_status=status,
                    response_reason=reason,
                )
            if not response.output_text:
                refused = any(
                    getattr(content, "type", None) == "refusal"
                    for item in response.output
                    for content in getattr(item, "content", ())
                )
                raise ProviderError(
                    "response_refusal" if refused else "invalid_output",
                    "API returned no structured output",
                    usage=usage,
                    raw_provider_usage=raw_usage,
                    response_status=status,
                    response_reason="refusal" if refused else "empty_output",
                )
            try:
                output = validate_output(
                    role, json.loads(response.output_text), final_only=final_only
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                raise ProviderError(
                    "invalid_output",
                    "API output violates role schema",
                    usage=usage,
                    raw_provider_usage=raw_usage,
                    response_status=status,
                    response_reason="schema_violation",
                ) from exc
            return {
                "output": output,
                "usage": usage,
                "raw_provider_usage": raw_usage,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "provider": "api",
                "model": model,
                "role": role,
                "instruction_hash": instruction_hash(role),
                "billing_mode": "api",
                "cost_state": "unknown",
                "raw_visible_events": [],
                "input_packet": payload,
            }
        finally:
            self.active.pop(request_id, None)

    async def cancel(self, request_id: str) -> None:
        """Signal cancellation of this instance's active inference task.

        Args:
            request_id (str): Identifier previously passed to generate().

        Returns:
            None: The task receives cancel() if active; an unknown or completed ID
            is a no-op. This does not wait for cleanup, reserve cancellation for a
            future request or cancel a remote API response. Repeated calls may
            signal the same task again while it remains active.
        """

        task = self.active.get(request_id)
        if task is not None:
            task.cancel()
