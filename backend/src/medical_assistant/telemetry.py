import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from langfuse import Langfuse

_LOG = logging.getLogger(__name__)
_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_RECOVERY_WAIT_SECONDS = 45
_RECOVERY_RETRY_SECONDS = 60
_MAX_RECOVERY_BYTES = 1048576
_PRICE_SNAPSHOT_PATH = Path(__file__).with_name("pricing_snapshot.json")
_SCOPE_FIELDS = (
    "id",
    "collection_id",
    "revision",
    "source_count",
    "ready_count",
    "unavailable_count",
    "snapshot_hash",
)
_SEARCH_FIELDS = (
    "dense_candidates",
    "bm25_candidates",
    "literal_candidates",
    "phrase_candidates",
    "rrf_pool_size",
    "search_ms",
    "index_scope",
)
_QUERY_STAGE_FIELDS = ("embedding", "dense", "bm25", "literal", "phrase", "fusion")
_RERANK_FIELDS = (
    "status",
    "model",
    "revision",
    "input_count",
    "scored_pairs",
    "truncated_pairs",
    "elapsed_ms",
)


def compact_scope(scope):
    """Project the frozen evidence scope into trace-safe summary fields.

    Args:
        scope (dict[str, object]): Required ``id``, ``collection_id``,
            ``revision``, ``source_count``, ``ready_count``,
            ``unavailable_count``, and ``snapshot_hash`` from the run snapshot.

    Returns:
        dict[str, object]: Those seven fields only, without the per-source
        ID/hash inventory retained in PostgreSQL. The source mapping is unchanged."""
    return {key: scope[key] for key in _SCOPE_FIELDS}


def search_observation_metadata(result):
    """Retain bounded retrieval diagnostics without duplicating evidence text.

    Project lane candidate counts, RRF pool size, total/stage milliseconds and
    index scope, plus reranker status/model/revision and pair counts/timing.
    This projection supports both search traces and run metrics; it does not
    copy candidate lists, snippets, or chunk text, or alter generation usage.

    Args:
        result (object): Search result, normally a dict with optional
            ``retrieval`` and ``rerank`` mappings. ``retrieval.query_stage_ms``
            may map embedding/dense/BM25/literal/phrase/fusion to milliseconds.

    Returns:
        dict[str, dict[str, object]]: Whitelisted retrieval/rerank mappings.
        Nonmapping input produces an empty dict; missing fields stay absent."""
    if not isinstance(result, dict):
        return {}
    projection = {}
    retrieval = result.get("retrieval")
    if isinstance(retrieval, dict):
        summary = {key: retrieval[key] for key in _SEARCH_FIELDS if key in retrieval}
        stages = retrieval.get("query_stage_ms")
        if isinstance(stages, dict):
            summary["query_stage_ms"] = {
                key: stages[key] for key in _QUERY_STAGE_FIELDS if key in stages
            }
        projection["retrieval"] = summary
    rerank = result.get("rerank")
    if isinstance(rerank, dict):
        projection["rerank"] = {key: rerank[key] for key in _RERANK_FIELDS if key in rerank}
    return projection


def _same_output(actual, expected):
    if actual == expected:
        return True
    if isinstance(actual, str):
        try:
            return json.loads(actual) == expected
        except (TypeError, ValueError):
            return False
    return False


def _json_object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _safe_error(exc):
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    code = getattr(exc, "code", None)
    message = str(exc)
    return {
        "type": type(exc).__name__,
        "status_code": status if isinstance(status, int) else None,
        "code": code[:64] if isinstance(code, str) and code.isascii() and code.isalnum() else None,
        "message_sha256": hashlib.sha256(message.encode()).hexdigest() if message else None,
    }


def estimate_api_equivalent_cost(
    model, usage, price_snapshot="openai_standard_short_2026-09-25", billing_mode="subscription"
):
    """Estimate hypothetical API spend from reported usage and frozen rates.

    Cached reads are a subset of total input and are charged once; reported
    output already includes reasoning and is not increased by reasoning tokens.
    Apply the snapshot's long-context rate multiplier only above its input-token
    threshold. Missing/invalid buckets, unsupported models, invalid partitions,
    and nonzero cache-write input leave the estimate unknown. This reads the
    versioned local snapshot and makes no inference or billing-service request.

    Args:
        model (str): Actual reported model key in the pricing snapshot.
        usage (dict[str, int] | None): Required nonnegative integer
            ``input_tokens``, ``cached_input_tokens``,
            ``cache_write_input_tokens``, and ``output_tokens`` buckets;
            ``reasoning_output_tokens`` may also be present but is not added.
        price_snapshot (str): Exact supported versioned snapshot ID.
        billing_mode (str): Provenance label, normally ``subscription``; it
            does not convert an estimate into a measured charge.

    Returns:
        dict[str, object]: Model/billing/usage provenance, the full price snapshot,
        ``actual_billed_usd`` always None, and ``estimated_api_cost_usd`` as an
        unrounded decimal USD string or None with ``estimate_limitation``.
        Known estimates also contain decimal-string ``cost_components_usd``
        and ``price_context``. Zero reported usage can yield a known zero cost;
        unknown usage never becomes zero. Subscription per-call billing remains
        unknown even when the hypothetical estimate is known.

    Raises:
        ValueError: The requested price snapshot is unsupported."""
    snapshot = json.loads(_PRICE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if price_snapshot != snapshot["id"]:
        raise ValueError("unsupported price snapshot")
    result = {
        "model": model,
        "billing_mode": billing_mode,
        "actual_billed_usd": None,
        "reported_usage": usage,
        "price_snapshot": snapshot,
        "estimated_api_cost_usd": None,
        "provenance": "estimated_from_reported_usage_and_versioned_price",
        "estimate_limitation": None,
    }
    if model not in snapshot["models"]:
        result["estimate_limitation"] = "unsupported_model"
        return result
    keys = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0 for key in keys
    ):
        result["estimate_limitation"] = "missing_or_invalid_reported_token_bucket"
        return result
    input_tokens = usage["input_tokens"]
    cached_tokens = usage["cached_input_tokens"]
    cache_write_tokens = usage["cache_write_input_tokens"]
    output_tokens = usage["output_tokens"]
    if cached_tokens + cache_write_tokens > input_tokens:
        result["estimate_limitation"] = "input_token_partitions_exceed_total"
        return result
    if cache_write_tokens:
        result["estimate_limitation"] = "codex_cli_cache_write_partition_semantics_unverified"
        return result
    rates = snapshot["models"][model]
    multiplier = input_tokens > snapshot["short_context_input_limit_tokens"]
    input_rate = Decimal(rates["input"]) * (2 if multiplier else 1)
    cached_rate = Decimal(rates["cached_input"]) * (2 if multiplier else 1)
    output_rate = Decimal(rates["output"]) * (Decimal("1.5") if multiplier else 1)
    million = Decimal(1000000)
    input_cost = Decimal(input_tokens - cached_tokens) * input_rate / million
    cached_cost = Decimal(cached_tokens) * cached_rate / million
    output_cost = Decimal(output_tokens) * output_rate / million
    result["estimated_api_cost_usd"] = str(input_cost + cached_cost + output_cost)
    result["cost_components_usd"] = {
        "uncached_input": str(input_cost),
        "cached_input": str(cached_cost),
        "output_including_reasoning": str(output_cost),
    }
    result["price_context"] = "long" if multiplier else "short"
    result["token_accounting"] = (
        "cached_input_tokens is treated as part of input_tokens; "
        "output_tokens already includes reasoning_output_tokens"
    )
    return result


class Observation:
    """Wrap one optional Langfuse observation in an asynchronous context.

    Enter/exit the SDK's synchronous context in the calling task and persist
    delivery failures asynchronously. Without a client, or after a start failure,
    the wrapper remains a no-op for updates. SDK delivery exceptions are logged
    and audited rather than replacing the application result. The wrapper does
    not make an inference call or confirm downstream materialization. Supply
    final output/usage before closing: accepted Langfuse observations are
    immutable, and replaying an observation ID can duplicate rows.

    Args:
        client (Langfuse | None): Optional trace SDK client.
        name (str): Business operation name, such as search_evidence or
            answer.generate.
        kind (str): Langfuse observation type.
        input_value (object): Private tool/provider request payload.
        model (str | None): Actual model for generation pricing metadata.
        metadata (dict[str, object] | None): Operation metadata; stable
            ``operation_id`` links the observation to durable audit.
        trace_id (str | None): Explicit trace context when opening a root.
        store (Store | None): Destination for delivery-failure audit events.
        run_id (str | None): Run owning those failure events.
        operation_id (str | None): Durable business-operation identity.
        persist (Callable[..., Awaitable[object]] | None): Optional async
            persistence adapter; otherwise store calls run in a worker thread."""

    def __init__(
        self,
        client,
        name,
        kind,
        input_value,
        model=None,
        metadata=None,
        trace_id=None,
        store=None,
        run_id=None,
        operation_id=None,
        persist=None,
    ):
        self.client = client
        self.name = name
        self.kind = kind
        self.input_value = input_value
        self.model = model
        self.metadata = metadata
        self.trace_id = trace_id
        self.store = store
        self.run_id = run_id
        self.operation_id = operation_id
        self.persist = persist
        self.failures = []
        self.manager = None
        self.inner = None

    def __bool__(self):
        return self.inner is not None

    async def _persist(self, method, *args):
        if self.persist is None:
            return await asyncio.to_thread(method, *args)
        return await self.persist(method, *args)

    async def _persist_failures(self):
        """Drain buffered SDK failures into the owning run's durable audit.

        Returns:
            None: The buffer is cleared even without a store/run or when an
            individual audit write fails. Persistence exceptions are logged;
            failed audit writes are not requeued by this wrapper."""
        if self.store is None or self.run_id is None:
            self.failures.clear()
            return
        failures, self.failures = self.failures, []
        for failure in failures:
            try:
                await self._persist(
                    self.store.save_audit,
                    self.run_id,
                    "trace.observation_delivery_failed",
                    failure,
                )
            except Exception:
                _LOG.exception(
                    "Failed to save observation delivery failure for run %s", self.run_id
                )

    def _record_failure(self, stage, exc):
        self.failures.append(
            {
                "operation_id": self.operation_id,
                "name": self.name,
                "stage": stage,
                "reason": _safe_error(exc),
            }
        )

    async def __aenter__(self):
        """Start the optional SDK context and persist any start-delivery failure.

        Returns:
            Observation: This wrapper, including when tracing is disabled or
            the SDK start failed. A returned wrapper need not have a live span.
            No provider work is started. Store failures during failure auditing
            are logged rather than propagated."""
        if self.client is None:
            return self
        arguments = {
            "name": self.name,
            "as_type": self.kind,
            "input": self.input_value,
            "metadata": self.metadata,
            "model": self.model,
        }
        if self.trace_id:
            arguments["trace_context"] = {"trace_id": self.trace_id}
        try:
            self.manager = self.client.start_as_current_observation(**arguments)
            self.inner = self.manager.__enter__()
        except Exception as exc:
            _LOG.warning("Langfuse observation start failed: %s", type(exc).__name__)
            self._record_failure("start", exc)
            self.manager = None
            self.inner = None
        await self._persist_failures()
        return self

    def update(self, **values):
        """Send final observation fields while its SDK context is still open.

        Search output additionally contributes the bounded retrieval/rerank
        projection to metadata. SDK exceptions buffer a failure for persistence
        at context exit; an absent live observation makes the call a no-op.
        Updating does not prove delivery and cannot repair an already accepted
        immutable observation through later replay.

        Args:
            **values (object): Langfuse update fields, usually ``output``,
                ``metadata``, ``usage_details``, and ``cost_details``; metadata
                is a mapping and output retains the actual private result.

        Returns:
            None: Any SDK failure is buffered, not raised."""
        if self.inner is None:
            return
        try:
            if self.name == "search_evidence" and "output" in values:
                projection = search_observation_metadata(values["output"])
                if projection:
                    metadata = values.get("metadata")
                    values["metadata"] = {
                        **(self.metadata or {}),
                        **(metadata if isinstance(metadata, dict) else {}),
                        **projection,
                    }
            self.inner.update(**values)
        except Exception as exc:
            _LOG.warning(
                "Langfuse observation update failed for %s: %s", self.name, _safe_error(exc)
            )
            self._record_failure("update", exc)

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Close the SDK context and audit buffered delivery failures.

        Args:
            exc_type (type[BaseException] | None): Exception from the body.
            exc_value (BaseException | None): Body exception instance.
            traceback (TracebackType | None): Body traceback.

        Returns:
            bool: The SDK manager's exit result, or False without a manager or
            after an SDK close failure. SDK close failures are logged/audited;
            body exception suppression follows that manager's return value."""
        result = False
        if self.manager is not None:
            try:
                result = self.manager.__exit__(exc_type, exc_value, traceback)
            except Exception as exc:
                _LOG.warning(
                    "Langfuse observation close failed for %s: %s", self.name, _safe_error(exc)
                )
                self._record_failure("close", exc)
        await self._persist_failures()
        return result


class Trace:
    """Own the durable trace plan and optional SDK root for one application run.

    The run audit remains authoritative when Langfuse is unavailable. Entering
    plans a deterministic root and marks trace delivery pending before opening
    the SDK context; exiting records trace.closed without claiming delivery.

    Args:
        telemetry (Telemetry): Optional SDK client plus authoritative Store.
        run (dict[str, object]): Required ``id``, ``status``, ``conversation_id``,
            ``question``, ``model``, ``variant``, and frozen ``scope`` with the
            seven fields required by ``compact_scope``.
        persist (Callable[..., Awaitable[object]] | None): Async persistence
            adapter, or None to run synchronous store calls in worker threads."""

    def __init__(self, telemetry, run, persist=None):
        self.telemetry = telemetry
        self.run = run
        self.persist = persist
        self.trace_id = Langfuse.create_trace_id(seed=f"pfl-run:{run['id']}")
        self.trace_url = None
        self.observation = None

    async def _persist(self, method, *args):
        if self.persist is None:
            return await asyncio.to_thread(method, *args)
        return await self.persist(method, *args)

    async def __aenter__(self):
        """Persist the original trace identity and open its asynchronous root.

        Save trace.planned with the deterministic run/root operation ID, then
        store the trace ID and pending status. Root metadata includes the compact
        frozen scope and application backlink, while input contains question,
        model, and variant. SDK start failure can leave these durable writes
        committed and the root inactive; storage failures propagate.

        Returns:
            Trace: This context with a deterministic trace_id. trace_url remains
            None until separate reconciliation verifies materialization."""
        run = self.run
        operation_id = f"{run['id']}:root"
        scope = compact_scope(run["scope"])
        await self._persist(
            self.telemetry.store.save_audit,
            run["id"],
            "trace.planned",
            {"trace_id": self.trace_id, "run_status": run["status"], "operation_id": operation_id},
        )
        await self._persist(
            self.telemetry.store.update_trace, run["id"], self.trace_id, None, "pending"
        )
        metadata = {
            "run_id": run["id"],
            "conversation_id": run["conversation_id"],
            "variant": run["variant"],
            "model": run["model"],
            "scope": scope,
            "app_url": self.telemetry.settings.app_origin.rstrip("/") + "/runs/" + run["id"],
            "delivery_kind": "original",
            "operation_id": operation_id,
        }
        self.observation = Observation(
            self.telemetry.client,
            "run",
            "chain",
            {"question": run["question"], "model": run["model"], "variant": run["variant"]},
            metadata=metadata,
            trace_id=self.trace_id,
            store=self.telemetry.store,
            run_id=run["id"],
            operation_id=operation_id,
            persist=self.persist,
        )
        await self.observation.__aenter__()
        return self

    def update(self, **values):
        if self.observation is not None:
            self.observation.update(**values)

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Close the optional root and append trace.closed to durable audit.

        Args:
            exc_type (type[BaseException] | None): Body exception type.
            exc_value (BaseException | None): Body exception instance.
            traceback (TracebackType | None): Body traceback.

        Returns:
            bool: Always False, preserving body exceptions and cancellation.
            The closing audit records whether the body raised; it is not a
            delivery receipt. Storage failures can propagate after SDK closure."""
        if self.observation is not None:
            await self.observation.__aexit__(exc_type, exc_value, traceback)
        await self._persist(
            self.telemetry.store.save_audit,
            self.run["id"],
            "trace.closed",
            {"trace_id": self.trace_id, "raised": exc_type is not None},
        )
        return False


class Telemetry:
    """Publish an optional trace view while PostgreSQL owns run outcomes.

    SDK initialization or delivery failure does not redefine execution success,
    evidence coverage, or evaluation denominators. Original observations and
    scores need readback before delivery is confirmed. Recovery exports recorded
    facts to separate deterministic traces; it never reruns inference or invents
    missing outputs, usage, or original timings.

    Args:
        settings (Settings): Langfuse public/secret keys and base/public URLs,
            plus app_origin for trace backlinks. Missing either key disables
            the optional client; initialization exceptions also leave it absent.
        store (Store): Durable run, audit, score-plan, and trace-status access."""

    def __init__(self, settings, store):
        self.settings = settings
        self.store = store
        self.client = None
        self._judge_recovery_attempts = {}
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    base_url=settings.langfuse_base_url,
                    timeout=5,
                    sample_rate=1.0,
                    flush_at=15,
                    flush_interval=2.0,
                )
            except Exception as exc:
                _LOG.warning("Langfuse client initialization failed: %s", type(exc).__name__)

    def trace(self, run, persist=None):
        """Create the async root context that durably plans a run's original trace.

        Args:
            run (dict[str, object]): Run with id, status, conversation_id,
                question, model, variant, and compact_scope-compatible scope.
            persist (Callable[..., Awaitable[object]] | None): Async store-call
                adapter; None uses worker-thread persistence.

        Returns:
            Trace: Unentered async context. Construction alone writes no audit;
            entering it records the trace plan even when the SDK is disabled."""
        return Trace(self, run, persist=persist)

    def observation(self, name, kind="span", input=None, model=None, metadata=None, persist=None):
        """Create a child context for a tool, generation, or evaluation operation.

        Unsupported kinds fall back to span. A model adds actual_model and
        hypothetical pricing metadata, never a subscription-charge assertion.
        operation_id's prefix identifies the run for delivery-failure auditing;
        judge.generate is handled through the evaluation ledger instead.

        Args:
            name (str): Business operation name, including answer.plan_query,
                answer.generate, an evidence tool, or judge.generate.
            kind (str): SDK observation type, default span.
            input (object): Actual private operation request payload.
            model (str | None): Actual reported generation model.
            metadata (dict[str, object] | None): Context/linkage fields; a stable
                ``operation_id`` enables per-run failure auditing.
            persist (Callable[..., Awaitable[object]] | None): Async persistence
                adapter for failure audit, or None to use worker threads.

        Returns:
            Observation: Unentered async context, possibly backed by no client.
            Construction starts neither SDK delivery nor provider inference."""
        allowed = {
            "generation",
            "embedding",
            "span",
            "agent",
            "tool",
            "chain",
            "retriever",
            "evaluator",
            "guardrail",
        }
        observation_type = kind if kind in allowed else "span"
        if model is not None:
            metadata = {**(metadata or {}), "actual_model": model}
            metadata["pricing_status"] = "hypothetical_api_estimate"
        operation_id = (metadata or {}).get("operation_id")
        run_id = operation_id.split(":", 1)[0] if isinstance(operation_id, str) else None
        return Observation(
            self.client,
            name,
            observation_type,
            input,
            model=model,
            metadata=metadata,
            store=self.store if run_id and name != "judge.generate" else None,
            run_id=run_id if name != "judge.generate" else None,
            operation_id=operation_id,
            persist=persist,
        )

    def score(
        self, run_id, name, value, data_type="NUMERIC", comment=None, metadata=None, score_id=None
    ):
        """Durably plan an evaluation score before attempting SDK delivery.

        Save trace.score_planned, mark the run's trace pending, and reset durable
        reconciliation backoff so a late score is checked again. The default
        score ID is stable per run/name, but repeated calls append audit plans
        and enqueue again; this method does not promise exactly-once delivery.
        SDK enqueue failure leaves the durable plan available for replay.

        Args:
            run_id (str): Existing application run ID.
            name (str): Score name defining its default deterministic identity.
            value (int | float | str | bool): Value passed through for the chosen
                SDK data_type; no local coercion or value validation is applied.
            data_type (str): NUMERIC, CATEGORICAL, BOOLEAN, TEXT, or CORRECTION.
            comment (str | None): Optional private score explanation.
            metadata (dict[str, object] | None): Optional score provenance/linkage.
            score_id (str | None): Explicit identity, or None for run/name UUID.

        Returns:
            str: Planned score ID, even when no SDK/client trace is available or
            enqueue fails. Database writes may already be committed.

        Raises:
            ValueError: data_type is unsupported.
            KeyError: The run does not exist."""
        if data_type not in {"NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT", "CORRECTION"}:
            raise ValueError("unsupported score data type")
        if score_id is None:
            score_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pfl-score:{run_id}:{name}"))
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        trace_id = run.get("trace_id") or self._original_trace_id(self.store.audit(run_id))
        payload = {
            "score_id": score_id,
            "name": name,
            "value": value,
            "data_type": data_type,
            "comment": comment,
            "metadata": metadata,
            "trace_id": trace_id,
        }
        self.store.save_audit(run_id, "trace.score_planned", payload)
        self.store.update_trace(run_id, trace_id, run.get("trace_url"), "pending")
        self.store.reset_trace_reconcile(run_id)
        if self.client is not None and trace_id:
            try:
                self.client.create_score(**payload)
            except Exception as exc:
                _LOG.warning("Langfuse score enqueue failed: %s", type(exc).__name__)
        return score_id

    def _original_trace_id(self, audit):
        for row in audit:
            if row["stage"] == "trace.planned":
                return row["payload"]["trace_id"]
        return None

    def _recovery_attempts(self, audit):
        return [row for row in audit if row["stage"] == "trace.recovery_planned"]

    def _recovery_fingerprint(self, run, audit):
        stages = {
            "trace.planned",
            "trace.closed",
            "run.exception",
            "validation.completed",
            "tool.started",
            "tool.completed",
            "tool.rejected",
            "generation.started",
            "generation.completed",
            "generation.failed",
            "provider.cancel_failed",
        }
        content = {
            "run_id": run["id"],
            "status": run["status"],
            "scope": compact_scope(run["scope"]),
            "answer": run.get("answer"),
            "error": run.get("error"),
            "finished_at": str(run.get("finished_at")),
            "audit": [
                {"stage": row["stage"], "payload": row["payload"]}
                for row in audit
                if row["stage"] in stages
            ],
        }
        return hashlib.sha256(
            json.dumps(content, default=str, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def _answer_sha256(self, answer):
        return hashlib.sha256(
            json.dumps(
                answer, default=str, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def _recovery_content(self, run, audit, original_id, emit=None):
        recovered_id = Langfuse.create_trace_id(seed=f"pfl-recovered:{run['id']}")
        emits = [
            row["payload"]
            for row in audit
            if row["stage"] == "trace.recovery_emit"
            and row["payload"].get("trace_id") == recovered_id
        ]
        if emit is not None:
            emits.append(emit)
        return {
            ("recovered_run", f"{run['id']}:recovery"): {
                "recovery": {
                    "run_id": run["id"],
                    "status": run["status"],
                    "answer": run.get("answer"),
                    "error": run.get("error"),
                    "scope": compact_scope(run["scope"]),
                    "original_trace_id": original_id,
                    "semantic_fingerprint": self._recovery_fingerprint(run, audit),
                    "answer_sha256": self._answer_sha256(run.get("answer")),
                    "emits": emits,
                }
            }
        }

    def _expected_operations(self, audit, run_id):
        """Derive the original span inventory from ordered durable operation starts.

        Check at most one root plan, deriving its identity from the run when other
        trace evidence exists without a plan. Check sequential generation/tool
        operation IDs, generation ordinals and known generation phases. Each
        root/tool/generation identity must be unique. No result is inferred
        from a start event.

        Args:
            audit (list[dict[str, object]]): Ordered audit events with ``stage``
                and ``payload``. Starts require operation_id, generation phase
                and ordinal, or tool name; trace.planned requires the root ID.
            run_id (str): Run prefix required by every original operation ID.

        Returns:
            dict[str, set[str]] | None: Expected observation IDs grouped by
            business name, including the root; None for missing trace evidence
            or inconsistent identities. A queued failed run's empty-original
            exception is handled by reconcile_run, not by this helper."""
        planned = [row for row in audit if row["stage"] == "trace.planned"]
        if len(planned) > 1 or (
            planned and planned[0]["payload"].get("operation_id") != f"{run_id}:root"
        ):
            return None
        if not planned and not any(
            row["stage"]
            in {"trace.closed", "generation.started", "tool.started", "validation.completed"}
            for row in audit
        ):
            return None
        operations = defaultdict(set, {"run": {f"{run_id}:root"}})
        generation_index = 0
        tool_index = 0
        for row in audit:
            stage = row["stage"]
            if stage == "generation.started":
                phase = row["payload"].get("phase")
                if phase not in {"query_planning", "answer"}:
                    return None
                name = "answer.plan_query" if phase == "query_planning" else "answer.generate"
                generation_index += 1
                if row["payload"].get("generation") != generation_index:
                    return None
                expected_id = f"{run_id}:generation:{generation_index}"
            elif stage == "tool.started":
                name = row["payload"].get("name")
                tool_index += 1
                expected_id = f"{run_id}:tool:{tool_index}"
            else:
                continue
            operation_id = row["payload"].get("operation_id")
            if (
                not name
                or operation_id != expected_id
                or any(operation_id in values for values in operations.values())
            ):
                return None
            operations[name].add(operation_id)
        return operations

    def _planned_scores(self, audit, trace_id=None):
        return [
            row["payload"]
            for row in audit
            if row["stage"] == "trace.score_planned"
            and (trace_id is None or row["payload"]["trace_id"] == trace_id)
        ]

    def _read_pages(self, fetch, **arguments):
        rows = []
        cursor = None
        seen = set()
        for _ in range(10):
            page = fetch(**arguments, **({"cursor": cursor} if cursor else {}))
            rows.extend(page.data)
            cursor = getattr(getattr(page, "meta", None), "cursor", None)
            if not cursor:
                return rows
            if cursor in seen:
                raise ValueError("Langfuse pagination cursor repeated")
            seen.add(cursor)
        raise ValueError("Langfuse pagination exceeds verification limit")

    def _expected_content(self, audit, run):
        """Build original-span readback expectations from saved requests and results.

        Args:
            audit (list[dict[str, object]]): Ordered events with stage/payload;
                starts carry operation_id and payload/arguments, completions
                carry output/result and optional usage/api_equivalent_estimate.
            run (dict[str, object]): Required id/question/model/variant and
                optional answer, which is checked as the published root output.

        Returns:
            dict[tuple[str, str], dict[str, object]]: Content by business-name/
            operation-ID pair. Started operations get input expectations;
            completed operations additionally get their actual output and
            reported accounting. Open operations gain no fabricated completion."""
        content = {
            ("run", f"{run['id']}:root"): {
                "input": {
                    "question": run["question"],
                    "model": run["model"],
                    "variant": run["variant"],
                },
                "answer": run.get("answer"),
            }
        }
        generation_names = {
            row["payload"].get("operation_id"): (
                "answer.plan_query"
                if row["payload"].get("phase") == "query_planning"
                else "answer.generate"
            )
            for row in audit
            if row["stage"] == "generation.started"
        }
        for row in audit:
            payload = row["payload"]
            operation_id = payload.get("operation_id")
            if row["stage"] == "tool.completed" and operation_id:
                content[(payload["name"], operation_id)] = {"output": payload.get("result")}
            elif row["stage"] == "generation.completed" and operation_id:
                name = generation_names.get(operation_id)
                if name is None:
                    continue
                content[(name, operation_id)] = {
                    "output": payload.get("output"),
                    "usage": payload.get("usage"),
                    "estimate": payload.get("api_equivalent_estimate"),
                }
        for row in audit:
            payload = row["payload"]
            operation_id = payload.get("operation_id")
            if row["stage"] == "tool.started" and operation_id:
                content.setdefault((payload["name"], operation_id), {})["input"] = payload.get(
                    "arguments"
                )
            elif row["stage"] == "generation.started" and operation_id:
                name = generation_names[operation_id]
                content.setdefault((name, operation_id), {})["input"] = payload.get("payload")
        return content

    def _observation_issues(self, item, name, expected, root_id, native_usage=True):
        """Check materialized content, linkage, timing, and recorded accounting.

        Original generations must retain reported usage and native cost details
        when known; recovered judge spans use metadata accounting instead.
        Recovery checks require terminal status/scope, original provenance,
        matching semantic and audit fingerprints, a recognized full/compact
        payload, and flags denying original timing/tree completeness. None
        remains unknown; a known nonzero estimate cannot read back as zero.

        Args:
            item (object): Langfuse readback record with timing, parent/root,
                input/output, metadata, and optional usage/cost detail attributes.
            name (str): Expected business observation name.
            expected (dict[str, object]): Applicable ``input``, ``output``,
                ``answer``, ``usage``, ``estimate``, or ``recovery`` contract.
                Recovery includes run_id/status/answer/error/scope,
                original_trace_id, semantic_fingerprint, answer_sha256 and emits
                containing audit_sha256/recovery_payload_kind.
            root_id (str | None): Parent ID required for original child spans.
            native_usage (bool): Require SDK-native usage/cost fields when True.

        Returns:
            list[str]: Readback defect labels, empty only if all applicable
            checks pass. No observation or application state is modified."""
        issues = []
        if getattr(item, "end_time", None) is None:
            issues.append("missing_end_time")
        start = getattr(item, "start_time", None)
        end = getattr(item, "end_time", None)
        if start is None or (end is not None and end < start):
            issues.append("invalid_timing")
        if name in {"run", "recovered_run", "judge.generate", "recovered_judge"}:
            if getattr(item, "is_root_observation", None) is False:
                issues.append("invalid_root")
        elif root_id is not None and getattr(item, "parent_observation_id", None) != root_id:
            issues.append("invalid_parent")
        output = getattr(item, "output", None)
        if "input" in expected and not _same_output(
            getattr(item, "input", None), expected["input"]
        ):
            issues.append("input_mismatch")
        if output is None:
            issues.append("missing_output")
        elif "output" in expected and not _same_output(output, expected["output"]):
            issues.append("output_mismatch")
        elif "answer" in expected:
            decoded = output
            if isinstance(decoded, str):
                try:
                    decoded = json.loads(decoded)
                except (TypeError, ValueError):
                    decoded = None
            if (
                not isinstance(decoded, dict)
                or decoded.get("published") is not True
                or not _same_output(decoded.get("answer"), expected["answer"])
            ):
                issues.append("answer_mismatch")
        metadata = getattr(item, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        if "recovery" in expected:
            recovery = expected["recovery"]
            input_value = _json_object(getattr(item, "input", None))
            output_value = _json_object(output)
            kind = metadata.get("recovery_payload_kind")
            audit_hash = metadata.get("audit_sha256")
            valid_emits = {
                (emit.get("audit_sha256"), emit.get("recovery_payload_kind"))
                for emit in recovery["emits"]
            }
            if (
                metadata.get("delivery_kind") != "recovered_from_audit"
                or metadata.get("original_trace_id") != recovery["original_trace_id"]
                or metadata.get("terminal_status") != recovery["status"]
                or metadata.get("semantic_fingerprint") != recovery["semantic_fingerprint"]
                or metadata.get("original_tree_complete") is not False
                or metadata.get("original_latency_measurement") is not False
                or (audit_hash, kind) not in valid_emits
            ):
                issues.append("recovery_metadata_mismatch")
            if not isinstance(input_value, dict) or any(
                input_value.get(key) != value
                for key, value in {
                    "run_id": recovery["run_id"],
                    "original_trace_id": recovery["original_trace_id"],
                    "terminal_status": recovery["status"],
                    "scope": recovery["scope"],
                    "terminal_error": recovery["error"],
                    "semantic_fingerprint": recovery["semantic_fingerprint"],
                    "audit_sha256": audit_hash,
                    "answer_sha256": recovery["answer_sha256"],
                    "original_tree_complete": False,
                    "recovery_payload_kind": kind,
                }.items()
            ):
                issues.append("recovery_input_mismatch")
            if kind == "full_audit" and isinstance(input_value, dict):
                observed_audit = input_value.get("audit")
                observed_hash = (
                    hashlib.sha256(
                        json.dumps(
                            observed_audit,
                            default=str,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    if isinstance(observed_audit, list)
                    else None
                )
                if observed_hash != audit_hash:
                    issues.append("recovery_audit_mismatch")
            expected_output = (
                {
                    "status": recovery["status"],
                    "answer": recovery["answer"],
                    "error": recovery["error"],
                }
                if kind == "full_audit"
                else {
                    "status": recovery["status"],
                    "error": recovery["error"],
                    "answer_sha256": recovery["answer_sha256"],
                    "answer_stored_in_postgres": recovery["answer"] is not None,
                }
                if kind == "compact_audit_summary"
                else None
            )
            if expected_output is None or output_value != expected_output:
                issues.append("recovery_output_mismatch")
        if name == "search_evidence" and isinstance(expected.get("output"), dict):
            projection = search_observation_metadata(expected["output"])
            if any(metadata.get(key) != value for key, value in projection.items()):
                issues.append("retrieval_metadata_mismatch")
        usage = expected.get("usage")
        if isinstance(usage, dict) and usage:
            if metadata.get("usage") != usage:
                issues.append("usage_metadata_mismatch")
            reported = {
                key: usage[source]
                for key, source in (("input", "input_tokens"), ("output", "output_tokens"))
                if type(usage.get(source)) is int
            }
            details = getattr(item, "usage_details", None)
            if (
                native_usage
                and reported
                and (
                    not isinstance(details, dict)
                    or any(details.get(key) != value for key, value in reported.items())
                )
            ):
                issues.append("usage_details_mismatch")
        estimate = expected.get("estimate")
        if isinstance(estimate, dict):
            if metadata.get("api_equivalent_estimate") != estimate:
                issues.append("estimate_metadata_mismatch")
            if native_usage and estimate.get("estimated_api_cost_usd") is not None:
                details = getattr(item, "cost_details", None)
                components = estimate.get("cost_components_usd") or {}
                if not all(
                    key in components
                    for key in ("uncached_input", "cached_input", "output_including_reasoning")
                ):
                    issues.append("estimate_components_missing")
                else:
                    expected_costs = {
                        "input": Decimal(components["uncached_input"])
                        + Decimal(components["cached_input"]),
                        "output": Decimal(components["output_including_reasoning"]),
                    }
                    for key, expected_cost in expected_costs.items():
                        actual = details.get(key) if isinstance(details, dict) else None
                        if type(actual) not in {int, float}:
                            issues.append(f"missing_{key}_cost")
                            continue
                        actual_cost = Decimal(str(actual))
                        tolerance = max(Decimal("1e-12"), abs(expected_cost) * Decimal("1e-6"))
                        if (
                            not actual_cost.is_finite()
                            or (expected_cost != 0 and actual_cost == 0)
                            or abs(actual_cost - expected_cost) > tolerance
                        ):
                            issues.append(f"{key}_cost_mismatch")
                    total = getattr(item, "total_cost", None)
                    if total is not None:
                        expected_total = sum(expected_costs.values())
                        if type(total) not in {int, float}:
                            issues.append("total_cost_mismatch")
                        else:
                            actual_total = Decimal(str(total))
                            tolerance = max(Decimal("1e-12"), abs(expected_total) * Decimal("1e-6"))
                            if (
                                not actual_total.is_finite()
                                or (expected_total != 0 and actual_total == 0)
                                or abs(actual_total - expected_total) > tolerance
                            ):
                                issues.append("total_cost_mismatch")
        return issues

    def _inspect(self, trace_id, expected_operations, planned_scores, expected_content=None):
        """Verify expected operation identities and scores through Langfuse readback.

        Every original operation must have exactly one matching name/operation-ID
        observation with a distinct SDK ID, correct trace/root linkage and
        complete expected content. A recovered_run instead accepts any complete,
        matching recovery observation among repeated emissions to its fixed
        trace. This allows later recovery without mutating accepted immutable
        spans. Score delivery is checked by planned score IDs, not score count
        or flush success; unrelated observations do not satisfy expected pairs.

        Args:
            trace_id (str): Original or recovery trace to inspect.
            expected_operations (dict[str, set[str]]): Business names mapped to
                all expected stable operation IDs.
            planned_scores (list[dict[str, object]]): Score payloads with
                ``score_id`` plus replay fields name/value/data_type/trace_id.
            expected_content (dict[tuple[str, str], dict[str, object]] | None):
                Readback contracts from _expected_content/_recovery_content.

        Returns:
            tuple[bool, bool, list[dict[str, object]], list[str]]: Whether any
            observation exists, whether the complete expected tree is verified,
            missing planned score payloads, and observation issue labels.

        Raises:
            ValueError: Readback pagination repeats or exceeds the ten-page
                verification limit."""
        observations = self._read_pages(
            self.client.api.observations.get_many,
            trace_id=trace_id,
            fields="core,basic,io,metadata,usage",
            expand_metadata="usage,api_equivalent_estimate,retrieval,rerank",
            limit=100,
        )
        operation_items = defaultdict(list)
        for item in observations:
            observation_id = getattr(item, "id", None)
            name = getattr(item, "name", None)
            if not observation_id or not name:
                continue
            metadata = getattr(item, "metadata", None)
            if isinstance(metadata, dict) and isinstance(metadata.get("operation_id"), str):
                operation_items[(name, metadata["operation_id"])].append(item)
        scores = self._read_pages(
            self.client.api.scores_v3.get_many_v3, trace_id=trace_id, limit=100
        )
        present_score_ids = {item.id for item in scores}
        expected_pairs = {
            (name, operation_id)
            for name, values in expected_operations.items()
            for operation_id in values
        }
        issues = []
        root_pair = next(
            (pair for pair in expected_pairs if pair[0] in {"run", "recovered_run"}), None
        )
        root_items = operation_items[root_pair] if root_pair else []
        root_id = root_items[0].id if len(root_items) == 1 else None
        ids = set()
        for pair in expected_pairs:
            items = operation_items[pair]
            if pair[0] == "recovered_run" and items:
                matching = [
                    item
                    for item in items
                    if getattr(item, "trace_id", None) == trace_id
                    and not self._observation_issues(
                        item, pair[0], (expected_content or {}).get(pair, {}), None
                    )
                ]
                if matching:
                    root_id = matching[0].id
                    ids.add(matching[0].id)
                    continue
                issues.append(f"{pair[0]}:{pair[1]}:no_matching_recovery")
                continue
            if len(items) != 1:
                issues.append(f"{pair[0]}:{pair[1]}:count_{len(items)}")
                continue
            item = items[0]
            if item.id in ids:
                issues.append(f"{pair[0]}:{pair[1]}:duplicate_id")
            ids.add(item.id)
            if getattr(item, "trace_id", None) != trace_id:
                issues.append(f"{pair[0]}:{pair[1]}:trace_mismatch")
            for reason in self._observation_issues(
                item, pair[0], (expected_content or {}).get(pair, {}), root_id
            ):
                issues.append(f"{pair[0]}:{pair[1]}:{reason}")
        complete_observations = not issues and root_id is not None
        missing_scores = [
            score for score in planned_scores if score["score_id"] not in present_score_ids
        ]
        return bool(observations), complete_observations, missing_scores, issues

    def _trace_url(self, trace_id):
        url = self.client.get_trace_url(trace_id=trace_id)
        base = self.settings.langfuse_base_url.rstrip("/")
        if url and url.startswith(base + "/"):
            return self.settings.langfuse_public_url.rstrip("/") + url[len(base) :]
        return url

    def _judge_observation_state(
        self,
        trace_id,
        name,
        operation_id,
        payload,
        output,
        usage,
        estimate,
        metadata,
        recovered=False,
    ):
        """Classify one judge observation using the persisted ledger contract.

        Duplicate original judge generations are incomplete even if one matches;
        a recovery trace accepts any complete matching recovered span. Recovery
        usage/cost is verified in metadata and is not presented as a new model
        generation. Existing unrelated observations yield incomplete rather than
        proving this operation's delivery.

        Args:
            trace_id (str): Trace containing the judge operation.
            name (str): judge.generate or recovered_judge.
            operation_id (str): Stable evaluation-attempt identity.
            payload (dict[str, object]): Persisted judge request.
            output (dict[str, object]): Persisted judge response.
            usage (dict[str, int] | None): Reported token buckets, or unknown.
            estimate (dict[str, object] | None): Recorded hypothetical cost.
            metadata (dict[str, object]): Expected evaluation/run/attempt linkage.
            recovered (bool): Permit any complete recovery and metadata-only
                accounting when True.

        Returns:
            tuple[str, datetime | None, int]: complete/incomplete/pending, latest
            matching start time for retry scheduling when incomplete, and count
            of matching observations. Readback errors propagate to the caller."""
        observations = self._read_pages(
            self.client.api.observations.get_many,
            trace_id=trace_id,
            fields="core,basic,time,io,metadata,usage",
            expand_metadata="usage,api_equivalent_estimate",
            limit=100,
        )
        matches = [
            item
            for item in observations
            if getattr(item, "name", None) == name
            and getattr(item, "trace_id", None) == trace_id
            and isinstance(getattr(item, "metadata", None), dict)
            and item.metadata.get("operation_id") == operation_id
        ]
        if not recovered and len(matches) > 1:
            latest = max(
                (item.start_time for item in matches if getattr(item, "start_time", None)),
                default=None,
            )
            return "incomplete", latest, len(matches)
        expected = {"input": payload, "output": output, "usage": usage, "estimate": estimate}
        for item in matches:
            if not getattr(item, "id", None):
                continue
            issues = self._observation_issues(
                item, name, expected, None, native_usage=not recovered
            )
            if any(
                item.metadata.get(key) != value
                for key, value in metadata.items()
                if value is not None
            ):
                issues.append("link_metadata_mismatch")
            if not issues:
                return "complete", None, len(matches)
        latest = max(
            (item.start_time for item in matches if getattr(item, "start_time", None)),
            default=None,
        )
        return ("incomplete" if observations else "pending"), latest, len(matches)

    def reconcile_judge_observation(
        self,
        *,
        trace_id,
        operation_id,
        payload,
        output,
        usage,
        estimate,
        billing_mode,
        metadata,
        finished_at=None,
        force_recovery=False,
    ):
        """Confirm a judge trace or export the existing ledger result for recovery.

        Flush and verify the original generation first, then check a separate
        deterministic recovery trace. A missing response cannot be recovered.
        Normally wait 45 seconds after completion before emission; subsequent
        attempts back off from 60 seconds up to one hour using existing recovery
        count/timing plus an in-memory guard. force_recovery bypasses those waits.
        Emit a span containing the saved request/result, usage, hypothetical
        cost, and provenance without inference or an immutable-span update.
        Ledger status persistence is the caller's responsibility.

        Args:
            trace_id (str | None): Original judge trace ID.
            operation_id (str | None): Stable judge-attempt identity.
            payload (dict[str, object]): Actual judge input stored in the ledger.
            output (dict[str, object] | None): Actual ledger response, or None.
            usage (dict[str, int] | None): Reported usage; absence stays unknown.
            estimate (dict[str, object] | None): Recorded API-equivalent estimate.
            billing_mode (str): Provider billing provenance, typically subscription.
            metadata (dict[str, object]): Evaluation/run/attempt linkage copied
                into recovery alongside operation and delivery provenance.
            finished_at (datetime | str | None): Original completion timestamp;
                ISO strings are accepted and naive times are interpreted as UTC.
                None has age zero and delays automatic first recovery.
            force_recovery (bool): Bypass completion and recovery retry waits.

        Returns:
            str: original_materialized, recovered_from_ledger, pending, or
            incomplete. Missing client/IDs and caught delivery/readback errors
            return pending; missing output returns incomplete after client/ID
            checks. Pending may follow a successful recovery enqueue or a flush."""
        if self.client is None or not trace_id or not operation_id:
            return "pending"
        if output is None:
            return "incomplete"
        if not hasattr(self, "_judge_recovery_attempts"):
            self._judge_recovery_attempts = {}
        try:
            self.client.flush()
            original, _, _ = self._judge_observation_state(
                trace_id, "judge.generate", operation_id, payload, output, usage, estimate, metadata
            )
            if original == "complete":
                return "original_materialized"
            recovered_id = Langfuse.create_trace_id(seed=f"pfl-recovered-judge:{operation_id}")
            recovery_metadata = {
                **metadata,
                "operation_id": operation_id,
                "delivery_kind": "recovered_from_ledger",
                "original_trace_id": trace_id,
                "original_tree_complete": False,
                "original_latency_measurement": False,
            }
            recovered, last_recovery, recovery_count = self._judge_observation_state(
                recovered_id,
                "recovered_judge",
                operation_id,
                payload,
                output,
                usage,
                estimate,
                recovery_metadata,
                recovered=True,
            )
            if recovered == "complete":
                self._judge_recovery_attempts.pop(operation_id, None)
                return "recovered_from_ledger"
            if not force_recovery and self._age_seconds(finished_at) < _RECOVERY_WAIT_SECONDS:
                return original
            retry_delay = min(3600, 60 * (2 ** min(max(recovery_count - 1, 0), 6)))
            if (
                last_recovery
                and not force_recovery
                and self._age_seconds(last_recovery) < retry_delay
            ):
                return "pending"
            recent_attempt = self._judge_recovery_attempts.get(operation_id)
            if (
                recent_attempt
                and not force_recovery
                and self._age_seconds(recent_attempt) < retry_delay
            ):
                return "pending"
            recovery_metadata.update(
                {
                    "usage": usage,
                    "billing_mode": billing_mode,
                    "api_equivalent_estimate": estimate,
                    "pricing_status": "hypothetical_api_estimate",
                    "actual_billed_cost": "unknown",
                    "source_model": "gpt-6-sol",
                }
            )
            with self.client.start_as_current_observation(
                name="recovered_judge",
                as_type="span",
                input=payload,
                metadata=recovery_metadata,
                trace_context={"trace_id": recovered_id},
            ) as observation:
                observation.update(output=output)
            self._judge_recovery_attempts[operation_id] = datetime.now(UTC)
            if len(self._judge_recovery_attempts) > 1000:
                self._judge_recovery_attempts = {
                    key: value
                    for key, value in self._judge_recovery_attempts.items()
                    if self._age_seconds(value) < 60
                }
            self.client.flush()
            recovered, _, _ = self._judge_observation_state(
                recovered_id,
                "recovered_judge",
                operation_id,
                payload,
                output,
                usage,
                estimate,
                recovery_metadata,
                recovered=True,
            )
            return "recovered_from_ledger" if recovered == "complete" else "pending"
        except Exception as exc:
            _LOG.warning(
                "Langfuse judge reconciliation deferred for %s: %s", operation_id, _safe_error(exc)
            )
            return "pending"

    def _age_seconds(self, value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value is None:
            return 0
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return (datetime.now(UTC) - value).total_seconds()

    def _audit_complete(self, run, audit):
        """Require a closed, validated successful audit before run recovery.

        Args:
            run (dict[str, object]): Run with status.
            audit (list[dict[str, object]]): Ordered stage/payload audit events.

        Returns:
            bool: True only for succeeded runs with exactly one validation,
            at least one generation start and trace close, and a valid operation
            inventory without an open operation. Failed/cancelled/interrupted
            recovery uses its known terminal facts under a separate policy."""
        if run["status"] != "succeeded":
            return False
        stages = Counter(row["stage"] for row in audit)
        if (
            stages["validation.completed"] != 1
            or stages["generation.started"] == 0
            or stages["trace.closed"] == 0
        ):
            return False
        inventory = self._operation_inventory(audit)
        return inventory is not None and not inventory["open_operations"]

    def _operation_inventory(self, audit):
        """Separate recorded operation completions from unresolved starts.

        Enforce serial starts, unique nonempty identities, matching generation
        or tool completion identity, and matching tool names. A generation.failed
        closes that known operation as failed; an unmatched final start remains
        unknown. Rejected tools or inconsistent event order invalidate inventory.

        Args:
            audit (list[dict[str, object]]): Ordered events with stage/payload;
                operation starts/completions require operation_id and tool name
                or generation phase, with optional generation failure code.

        Returns:
            dict[str, list[dict[str, object]]] | None: completed_operations and
            open_operations with kind/name/operation_id/outcome, or None for an
            invalid history. An empty valid history produces empty lists, never
            inferred provider results or usage."""
        active = None
        seen_operations = set()
        completed = []
        for row in audit:
            stage = row["stage"]
            payload = row["payload"]
            operation_id = payload.get("operation_id")
            if stage == "tool.rejected":
                return None
            if stage in {"generation.started", "tool.started"}:
                if (
                    active is not None
                    or not isinstance(operation_id, str)
                    or not operation_id
                    or operation_id in seen_operations
                ):
                    return None
                if stage == "tool.started" and not payload.get("name"):
                    return None
                active = {
                    "kind": "generation" if stage == "generation.started" else "tool",
                    "name": payload.get("phase")
                    if stage == "generation.started"
                    else payload["name"],
                    "operation_id": operation_id,
                }
                seen_operations.add(operation_id)
            elif stage in {"generation.completed", "generation.failed", "tool.completed"}:
                kind = "generation" if stage.startswith("generation.") else "tool"
                if (
                    active is None
                    or active["kind"] != kind
                    or operation_id != active["operation_id"]
                ):
                    return None
                if kind == "tool" and payload.get("name") != active["name"]:
                    return None
                completed.append(
                    {
                        **active,
                        "outcome": "failed" if stage == "generation.failed" else "completed",
                        **(
                            {"failure_code": payload.get("code")}
                            if stage == "generation.failed"
                            else {}
                        ),
                    }
                )
                active = None
        return {
            "completed_operations": completed,
            "open_operations": [{**active, "outcome": "unknown"}] if active else [],
        }

    def _replay_scores(self, trace_id, scores):
        """Reenqueue durable score plans against the verified or recovery trace.

        Args:
            trace_id (str): Destination trace; overrides each stored trace_id.
            scores (list[dict[str, object]]): Existing score payloads containing
                score_id/name/value/data_type and optional comment/metadata.

        Returns:
            None: SDK failures are logged individually and do not stop later
            scores. Reuses score identities without claiming readback delivery."""
        for score in scores:
            replay = {**score, "trace_id": trace_id}
            try:
                self.client.create_score(**replay)
            except Exception as exc:
                _LOG.warning("Langfuse score replay failed: %s", type(exc).__name__)

    def _mark(self, run_id, trace_id, trace_url, status, attempt_id):
        """Persist a reconciliation classification and its audit receipt.

        Args:
            run_id (str): Application run to update.
            trace_id (str | None): Chosen original/recovery trace identity.
            trace_url (str | None): Verified display URL, absent for failure states.
            status (str): Delivery classification, independent of execution status.
            attempt_id (str): Identity of the classification attempt.

        Returns:
            dict[str, str | None]: trace_id, trace_url, and trace_status after
            storing trace state and a trace.delivery event. Store errors propagate;
            the first write may already have committed."""
        self.store.update_trace(run_id, trace_id, trace_url, status)
        self.store.save_audit(
            run_id,
            "trace.delivery",
            {"attempt_id": attempt_id, "trace_id": trace_id, "status": status},
        )
        return {"trace_id": trace_id, "trace_url": trace_url, "trace_status": status}

    def reconcile_run(self, run_id, force_recovery=False):
        """Verify a terminal run's trace, or recover only its durable recorded facts.

        Validate original operation IDs and serial audit inventory before any
        readback. Invalid audit becomes unrecoverable; a nonsuccess terminal run
        that never started original operations can have a recovery record.
        A succeeded run needs complete original readback and scores for
        original_materialized, or a complete audit before recovery. Failed,
        cancelled, and interrupted runs are represented by verified recovery
        rather than a claim of original-tree completeness. Flush is not proof.

        Reuse a deterministic separate recovery trace, replay missing scores,
        and confirm content through readback. Normally allow 45 seconds after
        finish before first recovery and 60 seconds between run recovery emits.
        force_recovery bypasses the first wait only. Repeated emits may add spans;
        no exactly-once replay or accepted-observation mutation is promised.
        This performs no inference and changes neither the answer nor execution
        status. It writes attempt/diagnostic/delivery audit and trace state;
        persistent queue scheduling belongs to run_once/Store.

        Args:
            run_id (str): Existing authoritative application run ID.
            force_recovery (bool): Permit immediate first recovery despite the
                completion grace period; it does not bypass final-status exits,
                audit validity, or the wait after a previous recovery emission.

        Returns:
            dict[str, str | None]: trace_id, trace_url, and trace_status. Active
            runs return pending without writes. Already materialized/recovered/
            unrecoverable runs return their stored state. Valid audit without a
            client returns pending after recording an attempt; incomplete
            successful audit can return incomplete. Pending can follow committed
            attempts, score replay, recovery emission, or deferred SDK failure.
            Known delivery requires complete readback, including planned scores.

        Raises:
            KeyError: The application run is absent."""
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] not in _TERMINAL:
            return {
                "trace_id": run.get("trace_id"),
                "trace_url": run.get("trace_url"),
                "trace_status": "pending",
            }
        if run.get("trace_status") in {
            "original_materialized",
            "recovered_from_audit",
            "unrecoverable",
        }:
            return {
                "trace_id": run.get("trace_id"),
                "trace_url": run.get("trace_url"),
                "trace_status": run["trace_status"],
            }
        audit = self.store.audit(run_id)
        recovery_plans = self._recovery_attempts(audit)
        original_id = (
            recovery_plans[0]["payload"].get("original_trace_id")
            if recovery_plans
            else self._original_trace_id(audit) or run.get("trace_id")
        )
        if (
            original_id is None
            and not recovery_plans
            and any(
                row["stage"]
                in {"trace.closed", "generation.started", "tool.started", "validation.completed"}
                for row in audit
            )
        ):
            original_id = Langfuse.create_trace_id(seed=f"pfl-run:{run_id}")
        expected_operations = self._expected_operations(audit, run_id)
        inventory = self._operation_inventory(audit)
        no_original_started = not any(
            row["stage"]
            in {
                "trace.planned",
                "trace.closed",
                "generation.started",
                "tool.started",
                "validation.completed",
            }
            for row in audit
        )
        if expected_operations is None and no_original_started and run["status"] != "succeeded":
            expected_operations = {}
        if (
            inventory is None
            or expected_operations is None
            or (run["status"] == "succeeded" and not expected_operations.get("answer.generate"))
        ):
            self.store.save_audit(
                run_id,
                "trace.reconcile_invalid_audit",
                {"reason": "missing_or_invalid_operation_id", "method_version": "operation_id_v1"},
            )
            return self._mark(run_id, original_id, None, "unrecoverable", str(uuid.uuid4()))
        attempt_id = str(uuid.uuid4())
        self.store.save_audit(
            run_id,
            "trace.reconcile_attempt",
            {"attempt_id": attempt_id, "original_trace_id": original_id},
        )
        if self.client is None:
            return {"trace_id": original_id, "trace_url": None, "trace_status": "pending"}
        try:
            self.client.flush()
            recovered_id = Langfuse.create_trace_id(seed=f"pfl-recovered:{run_id}")
            recovered_operations = {"recovered_run": {f"{run_id}:recovery"}}
            all_scores = self._planned_scores(audit)
            if run.get("trace_id") == recovered_id and self._recovery_attempts(audit):
                recovered, complete, missing, _ = self._inspect(
                    recovered_id,
                    recovered_operations,
                    all_scores,
                    self._recovery_content(run, audit, original_id),
                )
                if recovered and complete:
                    if missing:
                        self._replay_scores(recovered_id, missing)
                        return {
                            "trace_id": recovered_id,
                            "trace_url": None,
                            "trace_status": "pending",
                        }
                    return self._mark(
                        run_id,
                        recovered_id,
                        self._trace_url(recovered_id),
                        "recovered_from_audit",
                        attempt_id,
                    )
            original_scores = self._planned_scores(audit, original_id)
            original, observations_complete, missing_scores, inspection_issues = (
                self._inspect(
                    original_id,
                    expected_operations,
                    original_scores,
                    self._expected_content(audit, run),
                )
                if original_id is not None
                else (False, False, [], ["missing_original_trace_id"])
            )
            if inspection_issues:
                self.store.save_audit(
                    run_id,
                    "trace.reconcile_incomplete",
                    {"attempt_id": attempt_id, "issues": inspection_issues[:100]},
                )
            if run["status"] == "succeeded" and original and observations_complete:
                if missing_scores:
                    self._replay_scores(original_id, missing_scores)
                    return {"trace_id": original_id, "trace_url": None, "trace_status": "pending"}
                return self._mark(
                    run_id,
                    original_id,
                    self._trace_url(original_id),
                    "original_materialized",
                    attempt_id,
                )
            if run["status"] == "succeeded" and not self._audit_complete(run, audit):
                return self._mark(run_id, original_id, None, "incomplete", attempt_id)
            recovered, complete, missing, _ = self._inspect(
                recovered_id,
                recovered_operations,
                all_scores,
                self._recovery_content(run, audit, original_id),
            )
            if recovered and complete:
                if missing:
                    self._replay_scores(recovered_id, missing)
                    return {"trace_id": recovered_id, "trace_url": None, "trace_status": "pending"}
                return self._mark(
                    run_id,
                    recovered_id,
                    self._trace_url(recovered_id),
                    "recovered_from_audit",
                    attempt_id,
                )
            if (
                not force_recovery
                and self._age_seconds(run.get("finished_at")) < _RECOVERY_WAIT_SECONDS
            ):
                return {"trace_id": original_id, "trace_url": None, "trace_status": "pending"}
            prior_emits = [row for row in audit if row["stage"] == "trace.recovery_emit"]
            if prior_emits and self._age_seconds(prior_emits[-1]["at"]) < _RECOVERY_RETRY_SECONDS:
                return {"trace_id": recovered_id, "trace_url": None, "trace_status": "pending"}
            return self._recover(
                run, audit, original_id, original_scores, attempt_id, recovered_id, inventory
            )
        except Exception as exc:
            detail = _safe_error(exc)
            _LOG.warning("Langfuse reconciliation deferred for run %s: %s", run_id, detail)
            self.store.save_audit(
                run_id,
                "trace.reconcile_deferred",
                {"attempt_id": attempt_id, "reason": detail},
            )
            return {"trace_id": original_id, "trace_url": None, "trace_status": "pending"}

    def _recover(self, run, audit, original_id, scores, attempt_id, recovered_id, inventory):
        """Export a terminal run's audit without recreating its original span tree.

        Preserve terminal status/error, frozen scope, original timestamp/latency
        provenance, known/open operation inventory, and only reported usage.
        Payloads exceeding one MiB become explicitly compact summaries with
        audit/answer hashes; full audit and answer remain in PostgreSQL. Plan and
        audit the emit before sending one recovered_run root on the deterministic
        recovery trace, replay scores, and verify the record through readback.
        The new timing is recovery timing and flags deny original-tree/latency
        completeness. No inference, missing result, or unknown cost is fabricated.

        Args:
            run (dict[str, object]): Terminal run with id/status/scope and optional
                answer/error/finished_at/metrics.end_to_end_ms.
            audit (list[dict[str, object]]): Ordered stage/payload/at audit; recorded
                generation results may include usage/api_equivalent_estimate.
            original_id (str | None): Original trace provenance when known.
            scores (list[dict[str, object]]): Planned score payloads for replay.
            attempt_id (str): Current reconciliation attempt identity.
            recovered_id (str): Deterministic separate recovery trace ID.
            inventory (dict[str, list[dict[str, object]]]): Valid completed/open
                operations from _operation_inventory.

        Returns:
            dict[str, str | None]: Verified recovered trace URL/status, or pending
            with the original ID when readback is still incomplete. Pending can
            follow committed recovery-plan/emission audit and SDK delivery.
            Delivery or storage exceptions propagate to reconcile_run's handling."""
        audit_bytes = json.dumps(
            audit, default=str, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
        semantic_fingerprint = self._recovery_fingerprint(run, audit)
        answer_sha256 = self._answer_sha256(run.get("answer"))
        scope = compact_scope(run["scope"])
        provider_usage = [
            {
                "stage": row["stage"],
                "operation_id": row["payload"].get("operation_id"),
                "usage": row["payload"].get("usage"),
                "usage_state": (
                    "reported"
                    if isinstance(row["payload"].get("usage"), dict) and row["payload"]["usage"]
                    else "unknown"
                ),
                "api_equivalent_estimate": row["payload"].get("api_equivalent_estimate"),
            }
            for row in audit
            if row["stage"] in {"generation.completed", "generation.failed"}
        ]
        payload = {
            "run_id": run["id"],
            "original_trace_id": original_id,
            "original_finished_at": str(run.get("finished_at")),
            "original_latency_ms": (run.get("metrics") or {}).get("end_to_end_ms"),
            "terminal_status": run["status"],
            "terminal_error": run.get("error"),
            "scope": scope,
            "operation_inventory": inventory,
            "known_operation_count": len(inventory["completed_operations"])
            + len(inventory["open_operations"]),
            "provider_usage": provider_usage or None,
            "original_tree_complete": False,
            "audit": audit,
            "recovery_payload_kind": "full_audit",
            "audit_sha256": audit_sha256,
            "semantic_fingerprint": semantic_fingerprint,
            "answer_sha256": answer_sha256,
        }
        compact = (
            len(json.dumps(payload, default=str, ensure_ascii=False).encode()) > _MAX_RECOVERY_BYTES
        )
        if compact:
            payload = {
                "run_id": run["id"],
                "original_trace_id": original_id,
                "original_finished_at": str(run.get("finished_at")),
                "original_latency_ms": (run.get("metrics") or {}).get("end_to_end_ms"),
                "terminal_status": run["status"],
                "terminal_error": run.get("error"),
                "scope": scope,
                "operation_inventory": inventory,
                "known_operation_count": len(inventory["completed_operations"])
                + len(inventory["open_operations"]),
                "provider_usage": provider_usage or None,
                "original_tree_complete": False,
                "recovery_payload_kind": "compact_audit_summary",
                "audit_sha256": audit_sha256,
                "semantic_fingerprint": semantic_fingerprint,
                "audit_event_count": len(audit),
                "audit_stage_counts": dict(Counter(row["stage"] for row in audit)),
                "answer_sha256": answer_sha256,
            }
        if not self._recovery_attempts(audit):
            self.store.save_audit(
                run["id"],
                "trace.recovery_planned",
                {
                    "attempt_id": attempt_id,
                    "trace_id": recovered_id,
                    "original_trace_id": original_id,
                },
            )
        emit = {
            "attempt_id": attempt_id,
            "trace_id": recovered_id,
            "recovery_payload_kind": payload["recovery_payload_kind"],
            "audit_sha256": audit_sha256,
            "semantic_fingerprint": semantic_fingerprint,
        }
        self.store.save_audit(
            run["id"],
            "trace.recovery_emit",
            emit,
        )
        with self.client.start_as_current_observation(
            name="recovered_run",
            as_type="chain",
            input=payload,
            metadata={
                "delivery_kind": "recovered_from_audit",
                "original_trace_id": original_id,
                "original_latency_measurement": False,
                "original_tree_complete": False,
                "terminal_status": run["status"],
                "scope": scope,
                "known_operation_count": payload["known_operation_count"],
                "open_operations": inventory["open_operations"],
                "app_url": self.settings.app_origin.rstrip("/") + "/runs/" + run["id"],
                "recovery_payload_kind": payload["recovery_payload_kind"],
                "audit_sha256": audit_sha256,
                "semantic_fingerprint": semantic_fingerprint,
                "operation_id": f"{run['id']}:recovery",
            },
            trace_context={"trace_id": recovered_id},
        ) as observation:
            observation.update(
                output={
                    "status": run["status"],
                    "answer": run.get("answer"),
                    "error": run.get("error"),
                }
                if not compact
                else {
                    "status": run["status"],
                    "error": run.get("error"),
                    "answer_sha256": payload["answer_sha256"],
                    "answer_stored_in_postgres": run.get("answer") is not None,
                }
            )
        self._replay_scores(recovered_id, scores)
        self.client.flush()
        recovered, complete, missing, _ = self._inspect(
            recovered_id,
            {"recovered_run": {f"{run['id']}:recovery"}},
            scores,
            self._recovery_content(run, audit, original_id, emit),
        )
        if not recovered or not complete or missing:
            return {"trace_id": original_id, "trace_url": None, "trace_status": "pending"}
        return self._mark(
            run["id"],
            recovered_id,
            self._trace_url(recovered_id),
            "recovered_from_audit",
            attempt_id,
        )
