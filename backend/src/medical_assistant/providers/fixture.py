import time

from pydantic import ValidationError

from .common import ProviderError, check_request, instruction_hash
from .contracts import validate_output


class FixtureProvider:
    """Validate explicitly supplied outputs for deterministic local verification.

    This adapter never invokes a model and is never selected as a live-provider
    fallback. Callers must include fixture_output in each task packet.
    """

    def __init__(self, settings):
        self.settings = settings

    async def readiness(self) -> dict:
        """Report that fixtures cannot establish live inference readiness.

        Returns:
            dict: provider="fixture", ready=False, connection="not_checked",
            configuration="present", inference="unverified", reason="fixture_only".
        """

        return {
            "provider": "fixture",
            "ready": False,
            "connection": "not_checked",
            "configuration": "present",
            "inference": "unverified",
            "reason": "fixture_only",
        }

    async def generate(
        self,
        payload: dict,
        role: str = "answer",
        model: str = "gpt-6-sol",
        request_id: str = "",
        final_only: bool = False,
    ) -> dict:
        """Validate a caller-supplied role output without inference or billing.

        Args:
            payload (dict): JSON task packet containing required fixture_output,
                an AnswerOutput or JudgeOutput mapping. Other task fields are
                retained in input_packet but do not affect the fixture result.
            role (str): "answer" or "judge"; planning uses the answer role.
            model (str): gpt-6-sol, or gpt-6-luna for answer only; validated and
                reported as metadata, never invoked.
            request_id (str): Valid nonempty cancellation-style identifier. This
                adapter does not track IDs or reject concurrent reuse.
            final_only (bool): Reject answer-role tool requests when True.

        Returns:
            dict: Validated output, usage={}, elapsed_ms (integer milliseconds),
            provider="fixture", model, role, instruction_hash,
            billing_mode="fixture", cost_state="not_applicable",
            raw_visible_events=[] and input_packet (the original payload).

        Raises:
            ProviderError: Admission fails, fixture_output is absent, or it violates
                the selected role contract. No model usage is incurred.
            OSError: Role instruction files needed for their hash cannot be read.
        """

        check_request(payload, role, model, request_id)
        started = time.monotonic()
        if "fixture_output" not in payload:
            raise ProviderError("fixture_missing", "fixture_output must be supplied explicitly")
        try:
            output = validate_output(role, payload["fixture_output"], final_only=final_only)
        except (ValidationError, ValueError) as exc:
            raise ProviderError("invalid_fixture", "fixture_output violates role schema") from exc
        return {
            "output": output,
            "usage": {},
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "provider": "fixture",
            "model": model,
            "role": role,
            "instruction_hash": instruction_hash(role),
            "billing_mode": "fixture",
            "cost_state": "not_applicable",
            "raw_visible_events": [],
            "input_packet": payload,
        }

    async def cancel(self, request_id: str) -> None:
        """Accept the shared cancellation interface as an idempotent no-op.

        Args:
            request_id (str): Ignored identifier; no task or future reservation
                is associated with it.

        Returns:
            None: Fixtures perform no asynchronous inference to interrupt.
        """

        return None
