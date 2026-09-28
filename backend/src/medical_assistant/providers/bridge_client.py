import asyncio
import json

import httpx
from pydantic import ValidationError

from .common import ProviderError, check_request
from .contracts import validate_output

_BRIDGE_ERROR_MESSAGES = {
    "role_model_rejected": "bridge rejected the role/model combination",
    "invalid_request_id": "bridge rejected the request id",
    "invalid_payload": "bridge rejected the payload",
    "payload_too_large": "bridge payload exceeded the limit",
    "binary_missing": "Codex binary is unavailable on the bridge host",
    "version_mismatch": "Codex version differs from the pinned version",
    "configuration_missing": "Codex configuration is unavailable on the bridge host",
    "configuration_drift": "Codex configuration changed on the bridge host",
    "profile_drift": "Codex role profile changed on the bridge host",
    "authentication_missing": "Codex login is unavailable on the bridge host",
    "stderr_too_large": "Codex stderr exceeded the limit",
    "events_too_large": "Codex event stream exceeded the limit",
    "invalid_event": "Codex emitted an invalid event",
    "unexpected_tool_use": "Codex attempted an unexpected tool use",
    "multiple_final_messages": "Codex emitted multiple final messages",
    "invalid_output": "Codex output is invalid",
    "multiple_turn_completions": "Codex completed more than once",
    "turn_failed": "Codex turn failed",
    "unexpected_event": "Codex emitted an unexpected event",
    "incomplete_turn": "Codex did not return a final output",
    "duplicate_request_id": "bridge request id is already active",
    "cancelled": "bridge request was cancelled",
    "cancellation_capacity": "bridge cannot reserve another cancellation",
    "timeout": "Codex inference timed out on the bridge host",
    "process_failed": "Codex process failed on the bridge host",
    "response_refusal": "API response was refused",
}


def _bridge_error(response: httpx.Response) -> ProviderError:
    status = response.status_code
    if status == 401:
        return ProviderError("bridge_unauthorized", "bridge authentication failed")
    if status in {500, 502, 503, 504}:
        return ProviderError("bridge_unavailable", f"bridge returned HTTP {status}")
    if status == 422 and len(response.content) <= 2048:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and set(body) == {"detail"}:
            detail = body["detail"]
            if isinstance(detail, dict) and set(detail) in (
                {"code", "message"},
                {"code", "message", "usage"},
            ):
                code = detail["code"]
                message = detail["message"]
                usage = detail.get("usage", {})
                if (
                    isinstance(code, str)
                    and type(message) is str
                    and 1 <= len(message) <= 512
                    and code in _BRIDGE_ERROR_MESSAGES
                    and isinstance(usage, dict)
                    and set(usage)
                    <= {
                        "input_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                    }
                    and all(type(value) is int and value >= 0 for value in usage.values())
                ):
                    return ProviderError(code, _BRIDGE_ERROR_MESSAGES[code], usage=usage)
    return ProviderError("bridge_failed", f"bridge returned HTTP {status}")


class BridgeProvider:
    """Forward the subscription inference contract to the authenticated host CLI.

    No API-billed or fixture fallback is attempted when the bridge is unavailable.
    Active work and pre-start cancellation reservations belong to the host's
    CodexProvider rather than this client instance.
    """

    def __init__(self, settings):
        self.settings = settings

    async def readiness(self) -> dict:
        """Probe authenticated bridge reachability and host configuration.

        Bound the probe to two seconds and validate the small host health response;
        make no model request. Missing credentials, transport failure, malformed
        replies and host configuration failures become readiness data.

        Returns:
            dict: provider="codex", ready (bool), connection ("reachable" or
            "unreachable"), configuration ("present", "missing" or "unknown"),
            inference="unverified" and reason. ready=True means the bridge reports
            configured local CLI prerequisites, not successful inference, valid
            service login, checked CLI version or available model quota.
        """

        status = {
            "provider": "codex",
            "ready": False,
            "connection": "unreachable",
            "configuration": "unknown",
            "inference": "unverified",
            "reason": "bridge_unavailable",
        }
        if not self.settings.bridge_token:
            return {**status, "reason": "bridge_token_missing"}
        try:
            async with asyncio.timeout(2.0):
                async with httpx.AsyncClient(timeout=2.0) as client:
                    async with client.stream(
                        "GET",
                        self.settings.bridge_url.rstrip("/") + "/health",
                        headers={"Authorization": f"Bearer {self.settings.bridge_token}"},
                    ) as response:
                        if response.status_code == 401:
                            return {
                                **status,
                                "connection": "reachable",
                                "reason": "bridge_unauthorized",
                            }
                        if response.status_code != 200:
                            return {**status, "connection": "reachable", "reason": "bridge_failed"}
                        content = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=1025):
                            if len(content) + len(chunk) > 1024:
                                return {
                                    **status,
                                    "connection": "reachable",
                                    "reason": "bridge_failed",
                                }
                            content.extend(chunk)
        except (httpx.HTTPError, ValueError, TimeoutError):
            return status
        try:
            body = json.loads(content)
        except ValueError:
            return {**status, "connection": "reachable", "reason": "bridge_failed"}
        reasons = {
            "configured",
            "binary_missing",
            "configuration_missing",
            "configuration_drift",
            "profile_drift",
            "authentication_missing",
            "configuration_unavailable",
        }
        if (
            not isinstance(body, dict)
            or set(body)
            != {"provider", "ready", "connection", "configuration", "inference", "reason"}
            or body["provider"] != "codex"
            or type(body["ready"]) is not bool
            or body["connection"] != "local"
            or type(body["configuration"]) is not str
            or body["configuration"] not in {"present", "missing"}
            or body["inference"] != "unverified"
            or type(body["reason"]) is not str
            or body["reason"] not in reasons
            or body["ready"] != (body["reason"] == "configured")
        ):
            return {**status, "connection": "reachable", "reason": "bridge_failed"}
        return {
            **status,
            "ready": body["ready"],
            "connection": "reachable",
            "configuration": body["configuration"],
            "reason": body["reason"],
        }

    async def generate(
        self,
        payload: dict,
        role: str = "answer",
        model: str = "gpt-6-sol",
        request_id: str = "",
        final_only: bool = False,
    ) -> dict:
        """Ask the host CLI for one inference result and revalidate its role output.

        Apply local admission, forward the packet with bridge authentication, and
        check the response envelope and role action before handing it to the graph
        or evaluator. The transport timeout adds 15 seconds to the configured
        inference timeout. This adapter performs no retry or deduplication; the
        host rejects active IDs and pending cancellations. A transport failure
        does not establish whether host inference ran or consumed tokens.

        Args:
            payload (dict): Untrusted JSON task packet: answer/planning question,
                history and evidence context, or judge candidate/gold/source/rubric
                context. Only JSON transport shape and size are checked locally.
            role (str): "answer" for answers/planning or "judge" for evaluation.
            model (str): gpt-6-sol, or gpt-6-luna for answer only.
            request_id (str): Valid nonempty host inference/cancellation ID.
                Reusing a completed ID can launch another subscription call.
            final_only (bool): Reject answer-role request_tools output when True.

        Returns:
            dict: Host result with revalidated output, usage (reported token
            mapping) and elapsed_ms (integer host milliseconds). The host normally
            adds provider="codex", model, role, instruction_hash,
            billing_mode="subscription", cost_state="unknown", raw_visible_events,
            input_packet, stderr and effective_capabilities; these additional
            fields are passed through rather than validated locally. An answer's
            support status is independent of successful bridge/CLI execution.

        Raises:
            ProviderError: Admission or bridge credentials fail, transport/HTTP
                work fails, the host rejects inference, or its response/output is
                invalid. Accepted host failure usage and usage accompanying an
                invalid response are retained when available; absence is not zero.
            asyncio.CancelledError: The caller's task is cancelled. Ending this
                client request alone does not explicitly cancel host inference;
                cancel() must be invoked separately for that ID.
        """

        check_request(payload, role, model, request_id)
        if not self.settings.bridge_token:
            raise ProviderError("bridge_token_missing", "bridge token is required")
        body = {
            "payload": payload,
            "role": role,
            "model": model,
            "request_id": request_id,
            "final_only": final_only,
        }
        timeout = self.settings.provider_timeout_seconds + 15
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    self.settings.bridge_url.rstrip("/") + "/infer",
                    json=body,
                    headers={"Authorization": f"Bearer {self.settings.bridge_token}"},
                )
            except httpx.TransportError as exc:
                raise ProviderError(
                    "bridge_transport_unavailable", "bridge transport failed"
                ) from exc
        if response.status_code != 200:
            raise _bridge_error(response)
        try:
            result = response.json()
        except ValueError as exc:
            raise ProviderError("bridge_failed", "bridge returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ProviderError("bridge_failed", "bridge response is invalid")
        usage = result.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        if (
            not isinstance(result.get("output"), dict)
            or not isinstance(result.get("usage"), dict)
            or type(result.get("elapsed_ms")) is not int
        ):
            raise ProviderError("bridge_failed", "bridge response is invalid", usage=usage)
        try:
            result["output"] = validate_output(role, result["output"], final_only=final_only)
        except (ValidationError, ValueError, TypeError) as exc:
            raise ProviderError("invalid_output", "bridge output is invalid", usage=usage) from exc
        return result

    async def cancel(self, request_id: str) -> None:
        """Request host-side active cancellation or a pre-start reservation.

        Args:
            request_id (str): Host inference ID. The host validates its format.

        Returns:
            None: The bridge confirms {"cancelled": True} after its cancellation
            operation. The host can reserve cancellation for an unknown ID, so
            confirmation does not prove that a running process existed. Active
            cancellation waits for host task cleanup; transmitted data and token
            consumption cannot be retracted. No retry is attempted by this client.

        Raises:
            ProviderError: Credentials are missing, cancellation transport/HTTP
                fails, the host rejects the ID/reservation, or confirmation is
                invalid. A lost response can follow successful host cancellation.
        """

        if not self.settings.bridge_token:
            raise ProviderError("bridge_token_missing", "bridge token is required")
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    self.settings.bridge_url.rstrip("/") + "/cancel/" + request_id,
                    headers={"Authorization": f"Bearer {self.settings.bridge_token}"},
                )
            except httpx.HTTPError as exc:
                raise ProviderError(
                    "bridge_transport_unavailable", "bridge cancellation transport failed"
                ) from exc
        if response.status_code != 200:
            raise _bridge_error(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("bridge_failed", "bridge cancellation response is invalid") from exc
        if body != {"cancelled": True}:
            raise ProviderError("bridge_failed", "bridge did not confirm cancellation")
