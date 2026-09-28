import asyncio
import hashlib
import json
import re
import time
from decimal import Decimal
from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medical_assistant.providers.common import MAX_PAYLOAD_BYTES, ProviderError
from medical_assistant.schemas import ModelAction
from medical_assistant.telemetry import estimate_api_equivalent_cost, search_observation_metadata

_TOOL_CONTRACT = {
    "search_evidence": "Supply one or two distinct queries in this tool round; the application merges their spans with the current packet and reads one bounded final packet.",
    "read_evidence": "Supply span_ids; the application merges them with the current packet and reads one bounded final packet from the frozen scope.",
    "collect_scope": "Request the whole frozen source scope, bounded to four pages and 32 KiB text; no arguments.",
}


class RunState(TypedDict, total=False):
    """Carry one run's evidence, decisions and audit measurements between nodes.

    ``run`` is the stored run dict: string ``id``, ``conversation_id``,
    ``question``, ``model`` and ``variant``, optional ``retry_of_run_id``, and
    frozen ``scope`` counts and ``snapshot_hash``. ``evidence`` holds delivered
    span dicts and a coverage receipt; ``action`` is a validated ModelAction
    dict, while ``answer`` adds resolved citations and coverage. ``policy`` is
    ``claims`` or ``complete_scope``. History contains role/content dicts used
    for navigation, never as source evidence. Counters bound generation and
    tool escalation; mutable ``metrics`` records calls, milliseconds, UTF-8
    evidence bytes, usage, delivered identifiers and coverage guards.

    Fields are populated progressively, so a node requires only the fields
    established by its predecessors. This state is not a durable checkpoint
    or an authorization to publish the resulting answer.
    """

    run: dict
    evidence: dict
    action: dict
    answer: dict
    policy: str
    generations: int
    escalations: int
    metrics: dict
    history: list[dict]
    history_older_messages_present: bool
    answer_generations: int
    search_query: str


def packet_hash(spans: list[dict]) -> str:
    value = [{"id": s["id"], "text": s["text"], "source_hash": s.get("source_hash")} for s in spans]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fit_provider_evidence(payload: dict, evidence: dict) -> dict:
    """Fit whole evidence spans within the provider's serialized request limit.

    Count the complete JSON request, including history, metadata and escaping.
    If it is too large, retain an ordered prefix of whole spans and mark its
    receipt incomplete. Reserve the original delivered-source list while
    packing, then replace it with the retained sources and recompute the hash.
    Packing is linear in evidence size and never cuts text inside a span.

    Args:
        payload (dict): Answer request fields before attaching evidence.
        evidence (dict): Spans and their verified coverage receipt.

    Returns:
        dict: Original evidence when it fits; otherwise retained spans and an
        updated receipt. Inputs are unchanged. The provider still validates
        the request, including any oversized non-evidence fields.
    """

    def encoded_size(value):
        return len(
            json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        )

    if encoded_size({**payload, "evidence": evidence}) <= MAX_PAYLOAD_BYTES:
        return evidence
    compact = {"spans": evidence["spans"], "receipt": evidence["receipt"]}
    if encoded_size({**payload, "evidence": compact}) <= MAX_PAYLOAD_BYTES:
        return compact
    receipt = {
        **evidence["receipt"],
        "complete": False,
        "truncated": True,
        "retrieval_truncated": True,
        "reason": "provider_payload_budget",
    }
    bounded = {"spans": [], "receipt": receipt}
    used = encoded_size({**payload, "evidence": bounded})
    for span in evidence["spans"]:
        additional = encoded_size(span) + bool(bounded["spans"])
        if used + additional > MAX_PAYLOAD_BYTES:
            break
        bounded["spans"].append(span)
        used += additional
    receipt["delivered_source_ids"] = sorted({span["source_id"] for span in bounded["spans"]})
    receipt["provider_input_packet_sha256"] = packet_hash(bounded["spans"])
    return bounded


def coverage_policy(question: str) -> str:
    """Choose whether a question needs evidence of whole-library coverage.

    Case-insensitive English patterns recognize exhaustive listings and
    questions about whether something was documented anywhere. This is a
    conservative routing heuristic, not a semantic classifier or a proof of
    coverage; the final action and absence wording can also require coverage.

    Args:
        question (str): The user's document question, before query planning.

    Returns:
        str: ``complete_scope`` when a pattern matches, otherwise ``claims``.
    """
    q = question.casefold()
    exhaustive = (
        r"\b(?:everywhere|anywhere|nowhere)\b"
        r"|\b(?:all|every|each|entire|whole)\s+(?:the\s+)?"
        r"(?:archive|collection|library|documents|records|sources|entries|mentions)\b"
        r"|\b(?:across|throughout)\s+(?:the\s+)?(?:whole|entire|all)?\s*"
        r"(?:archive|collection|library|documents|records|sources)\b"
        r"|\b(?:list|show|give|identify|name|enumerate)\s+(?:me\s+)?all\s+(?:the\s+)?"
        r"(?:[\w-]+\s+){0,3}"
        r"(?:values|findings|events|observations|results|mentions|records|entries)\b"
        r"|\bnone\s+of\s+(?:the\s+)?(?:documents|records|sources|entries)\b"
        r"|\b(?:not|never)\s+(?:documented|recorded|mentioned)\s+"
        r"(?:in|anywhere\s+in)\s+(?:the\s+)?(?:archive|collection|library|documents|records|sources)\b"
    )
    if re.search(exhaustive, q):
        return "complete_scope"
    if re.search(
        r"\bany\b.{0,100}\b(?:records?|documents?|sources?|collection|library|archive)\b", q
    ):
        return "complete_scope"
    if re.search(r"^(?:was|were|is|are|has|have|did|do|does)\b", q) and re.search(
        r"\b(?:documented|recorded|mentioned)\b", q
    ):
        return "complete_scope"
    return "claims"


def _coverage_limit_text():
    return (
        "Only retrieved excerpts from this collection were inspected. "
        "A complete review of this library is unavailable within the current limit. "
        "Ask a narrower question or upload a smaller set to a new collection."
    )


def limited_answer(reason: str) -> dict:
    """Build a partial answer when the workflow cannot safely supply claims.

    Args:
        reason (str): User-visible explanation of the exhausted workflow limit.

    Returns:
        dict: Answer action with ``kind=answer``, ``status=partial``, the reason
        as ``answer`` and the sole ``limitations`` entry, ``coverage_requirement``
        set to ``claims``, and empty claims, requests and citations. It contains
        no coverage receipt and does not mutate or persist run state.
    """
    return {
        "kind": "answer",
        "status": "partial",
        "answer": reason,
        "claims": [],
        "limitations": [reason],
        "coverage_requirement": "claims",
        "requests": [],
        "citations": [],
    }


def limited_evidence_answer(action, spans, reason, preserve_claims=True):
    """Reduce an unsupported coverage conclusion to a partial evidence answer.

    Preserve existing claims only for an already partial action that explains
    its limitations and when preservation is allowed. Otherwise, render up to
    three distinct cited spans as excerpts, with newlines flattened and each
    excerpt capped at 400 characters, to avoid repeating an unverified global
    conclusion. Preserve prior limitations only for a partial action.

    Args:
        action (dict): ModelAction fields, including ``status``, ``claims``
            (``text`` and ``evidence_ids``) and string ``limitations``.
        spans (dict[str, dict]): Delivered spans keyed by citation identifier;
            each referenced span must contain string ``text``.
        reason (str): Coverage or missing-evidence limitation to add.
        preserve_claims (bool): Whether eligible partial claims may survive.

    Returns:
        dict: Partial answer action with claim text as ``answer`` (or reason
        when no claims remain), unique limitations, ``claims`` coverage and
        no tool requests. Citations and coverage are attached by validation.
        The input is not modified; preserved claims reuse its claim list.
    """
    if preserve_claims and action["status"] == "partial" and action["limitations"]:
        claims = action["claims"]
    else:
        ids = list(
            dict.fromkeys(sid for claim in action["claims"] for sid in claim["evidence_ids"])
        )[:3]
        claims = [
            {
                "text": f"Source excerpt: {spans[sid]['text'].replace(chr(10), ' ')[:400]}",
                "evidence_ids": [sid],
            }
            for sid in ids
        ]
    return {
        "kind": "answer",
        "status": "partial",
        "answer": "\n".join(claim["text"] for claim in claims) or reason,
        "claims": claims,
        "limitations": list(dict.fromkeys([*action["limitations"], reason]))
        if action["status"] == "partial"
        else [reason],
        "coverage_requirement": "claims",
        "requests": [],
    }


def bounded_history(messages, older_messages_present=False, budget_bytes=16384):
    """Fit recent conversation hints into a bounded provider history packet.

    Consider at most the last four messages, prioritizing newest entries, and
    return them in their original order. Truncate UTF-8 prefixes without
    splitting a decoded character, mark truncation, and account for serialized
    role/content entries before accepting them. This bounds history context;
    it neither verifies earlier assistant claims nor retrieves their sources.

    Args:
        messages (list[dict]): Chronological messages with string ``role``
            and ``content``; other stored message fields are omitted.
        older_messages_present (bool): Whether the storage query omitted
            earlier history before this list.
        budget_bytes (int): Total serialized UTF-8 entry budget; defaults to
            16 KiB. The per-message target is one quarter minus 128 bytes.

    Returns:
        tuple[list[dict], bool]: Selected role/content dicts and a flag that
        earlier messages, content truncation or skipped entries exist. Inputs
        are not mutated; the flag may be true even with an empty result.
    """
    selected = []
    used = 0
    omitted = False
    for message in reversed(messages[-4:]):
        content = message["content"]
        raw = content.encode("utf-8")
        per_message = max(0, budget_bytes // 4 - 128)
        marker = " [message truncated]"
        if len(raw) > per_message:
            prefix = raw[: max(0, per_message - len(marker.encode()))].decode("utf-8", "ignore")
            content = prefix + marker
            omitted = True
        entry = {"role": message["role"], "content": content}
        size = len(json.dumps(entry, ensure_ascii=False).encode())
        prefix_bytes = min(len(raw), per_message)
        while size > per_message and prefix_bytes:
            prefix_bytes //= 2
            content = raw[:prefix_bytes].decode("utf-8", "ignore") + marker
            entry = {"role": message["role"], "content": content}
            size = len(json.dumps(entry, ensure_ascii=False).encode())
            omitted = True
        if used + size > budget_bytes:
            omitted = True
            continue
        selected.append(entry)
        used += size
    return list(reversed(selected)), older_messages_present or omitted or len(messages) > len(
        selected
    )


class Dialogue:
    """Produce a cited document answer within one frozen collection scope.

    The compiled graph loads bounded history and evidence, generates an
    action, permits one extra evidence round for V2/V3, and validates citation
    membership and coverage before returning an answer. History can resolve
    a search target but cannot substantiate a claim. Validation checks packet
    identity and delivery, not whether source text semantically proves a claim
    or whether the answer is clinically correct.

    Dependencies provide durable audit/events, scoped MCP reads, inference,
    telemetry, configuration and a database executor. Execution writes audit
    and progress records but does not publish an answer; the caller must use
    the store's atomic active-run/scope check. Calls are not idempotent and a
    failure can leave committed audit records without a completed answer.
    """

    def __init__(self, store, mcp, provider, settings, telemetry, db_executor):
        self.store = store
        self.mcp = mcp
        self.provider = provider
        self.settings = settings
        self.telemetry = telemetry
        self.db_executor = db_executor
        graph = StateGraph(RunState)
        graph.add_node("evidence", self.initial_evidence)
        graph.add_node("generate", self.generate)
        graph.add_node("tools", self.tools)
        graph.add_node("validate", self.validate)
        graph.add_edge(START, "evidence")
        graph.add_conditional_edges(
            "evidence", self.after_evidence, {"generate": "generate", "done": END}
        )
        graph.add_conditional_edges(
            "generate", self.next_node, {"tools": "tools", "validate": "validate"}
        )
        graph.add_edge("tools", "generate")
        graph.add_edge("validate", END)
        self.graph = graph.compile()

    async def db(self, method, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.db_executor, partial(method, *args))

    def after_evidence(self, state):
        """Route to completion when planning already produced clarification.

        Args:
            state (RunState): Evidence-node output, optionally with ``answer``.

        Returns:
            str: ``done`` when the answer key exists, otherwise ``generate``.
        """
        return "done" if "answer" in state else "generate"

    async def check_running(self, run_id):
        """Reject work after cancellation or a change to the frozen collection.

        Args:
            run_id (str): Stored run identifier whose status/scope to read.

        Raises:
            asyncio.CancelledError: The run is absent or no longer running.
            ValueError: ``scope_expired`` when current collection state differs
                from the frozen scope. This check does not lock later work or
                replace the publication-time scope check.
        """
        state = await self.db(self.store.run_scope_state, run_id)
        if state == "stopped":
            raise asyncio.CancelledError()
        if state == "scope_expired":
            raise ValueError("scope_expired")

    async def call_tool(self, state, name, arguments):
        """Execute and audit a read tool bound to the run's frozen scope.

        Check run validity before and after MCP execution, inject the run's
        scope identifier, and reject a returned scope or supplied snapshot
        hash that differs. Persist start/completion audit and progress events,
        observe telemetry, then accumulate successful call counts, elapsed
        milliseconds and search-stage metadata. A failed call can retain its
        start record; completing the remote read does not guarantee success.
        MCP and storage errors propagate; telemetry delivery failures are
        recorded separately.

        Args:
            state (RunState): ``run`` with ``id`` and scope ``snapshot_hash``;
                mutable ``metrics`` with tool count, timing and retrieval list.
            name (str): MCP tool name, normally search/read/collect evidence.
            arguments (dict): Tool-specific query/variant/limit, span_ids/budget,
                or cursor/page size. ``run_scope_id`` is overwritten locally.

        Returns:
            dict: MCP result containing matching ``scope_applied`` and
            tool-specific candidates, spans and inventory metadata.

        Raises:
            asyncio.CancelledError: The run stops before or after the call.
            ValueError: ``scope_expired`` or ``tool_scope_mismatch`` for a
                changed run/returned scope.
        """
        run = state["run"]
        await self.check_running(run["id"])
        operation_id = f"{run['id']}:tool:{state['metrics']['tool_calls'] + 1}"
        arguments = {**arguments, "run_scope_id": run["id"]}
        await self.db(
            self.store.save_audit,
            run["id"],
            "tool.started",
            {"name": name, "arguments": arguments, "operation_id": operation_id},
        )
        await self.db(self.store.append_event, run["id"], "tool.started", {"tool": name})
        start = time.perf_counter()
        async with self.telemetry.observation(
            name, "tool", arguments, metadata={"operation_id": operation_id}, persist=self.db
        ) as observation:
            result = await self.mcp.call(name, arguments)
            await self.check_running(run["id"])
            if result.get("scope_applied") != run["id"] or (
                "snapshot_hash" in result
                and result["snapshot_hash"] != run["scope"]["snapshot_hash"]
            ):
                raise ValueError("tool_scope_mismatch")
            if observation:
                observation.update(output=result)
        elapsed = (time.perf_counter() - start) * 1000
        await self.db(
            self.store.save_audit,
            run["id"],
            "tool.completed",
            {"name": name, "result": result, "elapsed_ms": elapsed, "operation_id": operation_id},
        )
        await self.db(
            self.store.append_event,
            run["id"],
            "tool.completed",
            {
                "tool": name,
                "elapsed_ms": elapsed,
                "count": len(result.get("spans", result.get("candidates", []))),
            },
        )
        state["metrics"]["tool_calls"] += 1
        state["metrics"]["tool_ms"] += elapsed
        if name == "search_evidence":
            state["metrics"]["retrieval_stages"].append(
                {"query": arguments["query"][:512], **search_observation_metadata(result)}
            )
        return result

    async def collect(self, state):
        """Attempt whole-scope evidence delivery within four bounded pages.

        Fetch up to four 8 KiB pages, join indexed fragments for each span,
        and verify each page's frozen inventory and delivered-source metadata.
        A complete receipt requires exhausted pagination, consistent complete
        fragments, no unavailable sources, a complete inventory marker and
        delivery from every ready source. Hitting a bound yields incomplete
        evidence rather than an assertion of archive-wide absence.

        Args:
            state (RunState): Active ``run`` with scope ``source_count``,
                ``ready_count``, ``unavailable_count`` and ``snapshot_hash``;
                tool metrics are mutated by each page read.

        Returns:
            dict: ``spans`` with id, source_id, text and source metadata, plus
            ``receipt`` with completeness, inventory/delivered sources,
            unavailable count, truncation/reason, snapshot hash and packet hash.
            ``complete`` may be false despite successfully delivered excerpts.

        Raises:
            ValueError: ``collect_scope_mismatch`` for inconsistent inventory
                metadata or ``evidence_budget_exceeded`` above 32 KiB of
                delivered UTF-8 text. Scoped tool failures also propagate.
        """
        pages = []
        cursor = ""
        for _ in range(4):
            page = await self.call_tool(
                state, "collect_scope", {"cursor": cursor, "page_bytes": 8192}
            )
            pages.append(page)
            cursor = page.get("next_cursor") or ""
            if not cursor:
                break
        fragments: dict[str, list[dict]] = {}
        for page in pages:
            for span in page.get("spans", []):
                fragments.setdefault(span["id"], []).append(span)
        spans = []
        incomplete_fragments = False
        for parts in fragments.values():
            ordered = {p.get("fragment_index", 0): p for p in parts}
            first = next(iter(ordered.values()))
            expected = first.get("fragment_count", 1)
            incomplete_fragments |= (
                not isinstance(expected, int)
                or expected < 1
                or len(ordered) != len(parts)
                or set(ordered) != set(range(expected))
                or any(part.get("fragment_count", 1) != expected for part in parts)
            )
            spans.append({**first, "text": "".join(ordered[k]["text"] for k in sorted(ordered))})
        final = pages[-1]
        scope = state["run"]["scope"]
        if any(
            page["inventory_count"] != scope["source_count"]
            or page["unavailable_count"] != scope["unavailable_count"]
            or page["snapshot_hash"] != scope["snapshot_hash"]
            or set(page["delivered_source_ids"]) != {span["source_id"] for span in page["spans"]}
            for page in pages
        ):
            raise ValueError("collect_scope_mismatch")
        complete = (
            not cursor
            and scope["unavailable_count"] == 0
            and not incomplete_fragments
            and bool(final.get("inventory_complete"))
            and len({span["source_id"] for span in spans}) == scope["ready_count"]
        )
        if sum(len(s["text"].encode()) for s in spans) > 32768:
            raise ValueError("evidence_budget_exceeded")
        return {
            "spans": spans,
            "receipt": {
                "complete": complete,
                "inventory_count": scope["source_count"],
                "delivered_source_ids": sorted({s["source_id"] for s in spans}),
                "unavailable_count": scope["unavailable_count"],
                "truncated": bool(cursor) or incomplete_fragments,
                "reason": "complete_scope" if complete else "scope_incomplete",
                "snapshot_hash": state["run"]["scope"]["snapshot_hash"],
                "provider_input_packet_sha256": packet_hash(spans),
            },
        }

    async def read(self, state, ids):
        """Read selected source spans and certify what actually reached context.

        Request the first 64 identifiers within 16 KiB, verify inventory,
        source membership and UTF-8 byte totals, then attach a packet receipt.
        Complete coverage additionally requires matching unique span inventory
        and hash, all ready sources and no unavailable or truncated content.
        Selected retrieval normally gives a subset, not exhaustive coverage.

        Args:
            state (RunState): Active run's frozen scope and mutable metrics.
            ids (list[str]): Ordered requested span identifiers; entries after
                64 are not requested and mark the receipt truncated.

        Returns:
            dict: MCP evidence with ``spans``, inventory metadata and added
            ``receipt`` describing complete/truncated coverage, delivered
            source IDs, reason, snapshot and provider packet hashes. Mutates
            ``metrics.retrieved_span_ids`` to actual delivered IDs.

        Raises:
            ValueError: ``read_scope_mismatch`` for inventory, delivered-source
                or byte-count inconsistency. Scoped tool failures propagate.
        """
        evidence = await self.call_tool(
            state, "read_evidence", {"span_ids": ids[:64], "budget_bytes": 16384}
        )
        scope = state["run"]["scope"]
        if (
            evidence["inventory_count"] != scope["source_count"]
            or evidence["unavailable_count"] != scope["unavailable_count"]
            or set(evidence["delivered_source_ids"])
            != {span["source_id"] for span in evidence["spans"]}
            or evidence["text_bytes"]
            != sum(len(span["text"].encode()) for span in evidence["spans"])
        ):
            raise ValueError("read_scope_mismatch")
        truncated = bool(evidence.get("truncated")) or len(ids) > 64
        spans = evidence["spans"]
        span_ids = [span["id"] for span in spans]
        inventory_hash = hashlib.sha256(
            json.dumps(sorted(span_ids), separators=(",", ":")).encode()
        ).hexdigest()
        complete = (
            bool(evidence["inventory_complete"])
            and not truncated
            and scope["unavailable_count"] == 0
            and evidence["span_inventory_count"] == len(span_ids)
            and len(set(span_ids)) == len(span_ids)
            and evidence["span_inventory_sha256"] == inventory_hash
            and len(evidence["delivered_source_ids"]) == scope["ready_count"]
        )
        evidence["receipt"] = {
            "complete": complete,
            "inventory_count": evidence["inventory_count"],
            "delivered_source_ids": evidence["delivered_source_ids"],
            "unavailable_count": scope["unavailable_count"],
            "truncated": truncated,
            "reason": "complete_scope"
            if complete
            else "retrieved_subset_truncated"
            if truncated
            else "retrieved_subset",
            "snapshot_hash": scope["snapshot_hash"],
            "provider_input_packet_sha256": packet_hash(spans),
        }
        state["metrics"]["retrieved_span_ids"] = [s["id"] for s in evidence["spans"]]
        return evidence

    async def search_read(self, state, query):
        """Turn ranked candidates into the initial bounded source-text packet.

        Search with the run's retrieval variant and a twelve-candidate limit,
        deduplicate candidate span IDs in first-seen order, and read those
        spans. Candidate metadata is navigation, not citable source evidence.

        Args:
            state (RunState): Active ``run`` with ``variant`` and scope, plus
                mutable ``metrics`` for candidate and delivered identifiers.
            query (str): Standalone search target from the question or planner.

        Returns:
            dict: Evidence/receipt from ``read``; may have no spans or partial
            coverage. Records initial candidate chunk and block-span IDs in
            metrics. Search/read audit side effects and failures propagate.
        """
        search = await self.call_tool(
            state,
            "search_evidence",
            {"query": query, "variant": state["run"]["variant"], "limit": 12},
        )
        ids = list(
            dict.fromkeys(
                sid for candidate in search.get("candidates", []) for sid in candidate["span_ids"]
            )
        )
        state["metrics"]["initial_candidate_chunk_ids"] = [
            candidate["chunk_id"] for candidate in search.get("candidates", [])
        ]
        state["metrics"]["initial_candidate_block_span_ids"] = ids
        return await self.read(state, ids)

    async def extend_evidence(self, state, requests):
        """Merge one extra evidence batch with current spans and repack once.

        Keep current span IDs first, append deduplicated search/read groups,
        and perform one bounded read of the union. Record omitted identifiers
        and one-based additional groups with no delivered member so validation
        can disclose missing requested evidence. Repacking can omit useful
        old or new excerpts; finding a candidate does not ensure its delivery.

        Args:
            state (RunState): Current ``evidence.spans``, active run/variant and
                mutable tool/read metrics.
            requests (list[dict]): Model tool requests with ``tool``, ``query``
                and ``span_ids``. Search requires a nonblank query and no IDs;
                read requires IDs and a blank query. The action schema bounds
                the upstream list to two requests.

        Returns:
            dict: Repacked evidence with ``retrieval_queries`` and receipt
            fields ``retrieval_truncated`` and ``missing_requested_groups``;
            omissions also force incomplete/truncated coverage. The caller
            replaces the current packet; this method updates metrics/audit.

        Raises:
            ValueError: Invalid search/read fields, unsupported request tool,
                or ``duplicate_search_query`` for identical stripped queries.
                Earlier requests may already have executed when validation
                of a later request fails. Scoped tool/read failures propagate.
        """
        groups = [[span["id"] for span in state["evidence"]["spans"]]]
        queries = []
        for request in requests:
            if request["tool"] == "search_evidence":
                if request["span_ids"] or not request["query"].strip():
                    raise ValueError("invalid_search_request")
                query = request["query"].strip()
                search = await self.call_tool(
                    state,
                    "search_evidence",
                    {"query": query, "variant": state["run"]["variant"], "limit": 12},
                )
                groups.append(
                    list(
                        dict.fromkeys(
                            sid
                            for candidate in search["candidates"]
                            for sid in candidate["span_ids"]
                        )
                    )
                )
                queries.append(query)
            elif request["tool"] == "read_evidence":
                if request["query"].strip() or not request["span_ids"]:
                    raise ValueError("invalid_read_request")
                groups.append(list(dict.fromkeys(request["span_ids"])))
            else:
                raise ValueError("invalid_evidence_request")
        if len(set(queries)) != len(queries):
            raise ValueError("duplicate_search_query")
        ids = list(dict.fromkeys(sid for group in groups for sid in group))
        evidence = await self.read(state, ids)
        delivered = {span["id"] for span in evidence["spans"]}
        missing_groups = [
            index
            for index, group in enumerate(groups[1:], 1)
            if not group or not delivered.intersection(group)
        ]
        omitted = len(set(ids) - delivered)
        evidence["receipt"]["retrieval_truncated"] = bool(omitted)
        evidence["receipt"]["missing_requested_groups"] = missing_groups
        evidence["receipt"]["complete"] = evidence["receipt"]["complete"] and not omitted
        evidence["receipt"]["truncated"] = evidence["receipt"]["truncated"] or bool(omitted)
        if omitted:
            evidence["receipt"]["reason"] = "retrieved_subset_truncated"
        evidence["retrieval_queries"] = queries
        return evidence

    async def prior_history(self, run):
        """Load history before this question's original retry ancestor.

        Follow retry links to the first attempt, checking for cycles, missing
        ancestors and conversation changes. Fetch four prior messages at that
        anchor, then apply the history byte bound so retries cannot see their
        own earlier answers as preceding conversation evidence.

        Args:
            run (dict): Stored ``id``, ``conversation_id`` and optional
                ``retry_of_run_id`` identifying the history boundary.

        Returns:
            tuple[list[dict], bool]: Chronological bounded role/content hints
            and whether older or truncated history exists. No writes occur.

        Raises:
            ValueError: ``invalid_retry_ancestry`` for invalid links; the store
                can also reject a missing anchor message. Storage failures
                propagate.
        """
        ancestor_id = run.get("retry_of_run_id")
        if ancestor_id:
            visited = {run["id"]}
            while ancestor_id:
                if ancestor_id in visited:
                    raise ValueError("invalid_retry_ancestry")
                visited.add(ancestor_id)
                ancestor = await self.db(self.store.get_run, ancestor_id)
                if ancestor is None or ancestor["conversation_id"] != run["conversation_id"]:
                    raise ValueError("invalid_retry_ancestry")
                next_id = ancestor.get("retry_of_run_id")
                if not next_id:
                    break
                ancestor_id = next_id
        messages, older = await self.db(
            self.store.prior_messages, run["conversation_id"], ancestor_id or run["id"], 4
        )
        return bounded_history(messages, older)

    async def plan_query(self, state):
        """Resolve a follow-up search target or stop with clarification.

        Invoke the provider with history but no source evidence. Accept exactly
        one nonblank search request without IDs or claims, or a nonblank
        needs_clarification answer without claims/requests. History is only a
        navigation hint; no factual answer may be produced in this phase.

        Args:
            state (RunState): Run/question/scope, policy, bounded history and
                initialized generation counters and metrics.

        Returns:
            str | None: Stripped standalone query (schema maximum 512
            characters), or None after assigning ``state.answer`` with empty
            citations and an incomplete clarification receipt. Both paths
            record the provider call; clarification also updates coverage
            metrics and persists a validation audit before graph completion.

        Raises:
            ValueError: ``invalid_query_plan`` for another action shape.
                Provider, schema-validation and scoped-run failures propagate.
        """
        run = state["run"]
        payload = {
            "task_mode": "query_planning",
            "question": run["question"],
            "history": state["history"],
            "history_older_messages_present": state["history_older_messages_present"],
            "evidence": {"spans": []},
            "coverage_policy": state["policy"],
            "final_only": False,
            "available_tools": {
                "search_evidence": "Return exactly one standalone query to start retrieval."
            },
            "instructions": "Resolve only the search target from conversation history. Treat prior assistant text as a navigation hint, not verified evidence. Return one standalone search query within 512 characters or ask for clarification. Do not answer from history.",
        }
        action = await self.invoke_provider(
            state, payload, final_only=False, phase="query_planning"
        )
        if action["kind"] == "request_tools":
            requests = action["requests"]
            if (
                len(requests) != 1
                or requests[0]["tool"] != "search_evidence"
                or not requests[0]["query"].strip()
                or requests[0]["span_ids"]
                or action["claims"]
            ):
                raise ValueError("invalid_query_plan")
            return requests[0]["query"].strip()
        if (
            action["kind"] != "answer"
            or action["status"] != "needs_clarification"
            or not action["answer"].strip()
            or action["claims"]
            or action["requests"]
        ):
            raise ValueError("invalid_query_plan")
        receipt = {
            "complete": False,
            "inventory_count": run["scope"]["source_count"],
            "delivered_source_ids": [],
            "unavailable_count": run["scope"]["unavailable_count"],
            "truncated": False,
            "reason": "needs_clarification",
            "snapshot_hash": run["scope"]["snapshot_hash"],
            "provider_input_packet_sha256": packet_hash([]),
            "provider_invocation_id": f"{run['id']}:{state['generations']}",
        }
        state["metrics"]["initial_method"] = "query_clarification"
        state["metrics"]["initial_provider_span_ids"] = []
        state["metrics"]["initial_receipt_complete"] = False
        state["metrics"]["delivered_span_ids"] = []
        state["metrics"]["final_provider_span_ids"] = []
        state["metrics"]["final_receipt_complete"] = False
        state["metrics"]["coverage_complete"] = False
        state["metrics"]["guard_action"] = "query_needs_clarification"
        state["answer"] = {**action, "citations": [], "coverage": receipt}
        await self.db(
            self.store.save_audit,
            run["id"],
            "validation.completed",
            {"answer": state["answer"], "metrics": state["metrics"]},
        )
        return None

    async def initial_evidence(self, state):
        """Initialize a run and retrieve evidence for its resolved question.

        Reset counters/metrics, derive coverage policy and load bounded history.
        Plan a query when any history exists or the stripped question exceeds
        512 characters; clarification ends this node without retrieval.
        Otherwise, emit retrieval progress and search/read the initial packet.

        Args:
            state (RunState): Initially requires only the stored ``run`` dict
                with question, conversation/retry identifiers, model, variant
                and frozen scope.

        Returns:
            RunState: The same mutated state with policy, history, counters,
            metrics and either ``answer`` for clarification or ``search_query``
            and ``evidence`` with actual initial IDs/coverage. Audit/progress
            records may be committed even if later planning/retrieval fails.
        """
        state["metrics"] = {
            "model_calls": 0,
            "tool_calls": 0,
            "tool_ms": 0.0,
            "retrieval_stages": [],
            "provider_ms": 0.0,
            "evidence_bytes": 0,
            "usage": [],
            "initial_candidate_chunk_ids": None,
            "initial_candidate_block_span_ids": None,
            "initial_provider_span_ids": None,
            "initial_receipt_complete": None,
            "initial_method": None,
            "coverage_policy": None,
            "second_batch_kind": None,
            "final_provider_span_ids": None,
            "final_receipt_complete": None,
            "guard_action": None,
        }
        state["generations"] = 0
        state["answer_generations"] = 0
        state["escalations"] = 0
        state["policy"] = coverage_policy(state["run"]["question"])
        state["metrics"]["coverage_policy"] = state["policy"]
        state["history"], state["history_older_messages_present"] = await self.prior_history(
            state["run"]
        )
        query = state["run"]["question"].strip()
        if state["history"] or state["history_older_messages_present"] or len(query) > 512:
            query = await self.plan_query(state)
            if query is None:
                return state
        await self.db(
            self.store.append_event, state["run"]["id"], "stage.started", {"stage": "retrieval"}
        )
        state["metrics"]["initial_method"] = "search_read"
        state["search_query"] = query
        state["evidence"] = await self.search_read(state, query)
        state["metrics"]["initial_provider_span_ids"] = sorted(
            span["id"] for span in state["evidence"]["spans"]
        )
        state["metrics"]["initial_receipt_complete"] = bool(
            state["evidence"]["receipt"]["complete"]
        )
        await self.db(
            self.store.append_event,
            state["run"]["id"],
            "stage.completed",
            {
                "stage": "retrieval",
                "source_count": len({s["source_id"] for s in state["evidence"]["spans"]}),
            },
        )
        return state

    async def invoke_provider(self, state, payload, final_only, phase):
        """Invoke inference with an auditable identity for the delivered packet.

        Check that the run remains active, increment its generation counter,
        and for answer calls verify the evidence hash before attaching this
        invocation's ID to the receipt/payload. Persist generation progress,
        observe the call, record reported usage and hypothetical API-equivalent
        cost, then validate the returned output as ModelAction. Estimated cost
        is not the actual subscription charge. A schema-valid action still
        requires workflow/citation validation before it becomes an answer.

        Args:
            state (RunState): Run ``id``/``model``, generation counter and
                initialized metrics; answer phase also requires ``evidence``
                with spans and a packet-hash receipt. Counter, receipt and
                successful-call timing/usage/evidence-byte metrics are mutated.
            payload (dict): Provider task with ``task_mode``, ``question``,
                role/content ``history``, older-history flag, coverage policy,
                tool contract and instructions. Answer calls replace its
                ``evidence`` with the invocation-bound packet in place.
            final_only (bool): Whether the provider must return an answer
                rather than request another tool round.
            phase (str): ``query_planning`` or ``answer``, selecting telemetry
                and whether an evidence receipt is bound to the invocation.

        Returns:
            dict: Validated ModelAction: ``kind`` (answer/request_tools),
            ``status``, ``answer`` text, ``claims`` with text/evidence_ids,
            ``limitations``, ``coverage_requirement`` and ``requests`` with
            tool/query/span_ids. Does not assign ``state.action`` itself.

        Raises:
            ProviderError: Provider failure, re-raised after persisting its
                diagnostic and available usage; successful-call metrics are
                not incremented on this path.
            ValueError: ``evidence_packet_changed`` for a modified packet or
                ``scope_expired`` after a collection change. ModelAction's
                Pydantic validation error propagates for invalid output.
            asyncio.CancelledError: The run stops at a validity check or the
                task is cancelled. Audit records and generation counter can
                already be updated; neither failure nor cancellation rolls
                back earlier records or retracts input sent to the provider.
        """
        run = state["run"]
        await self.check_running(run["id"])
        state["generations"] += 1
        if phase == "answer":
            evidence = state["evidence"]
            if (
                packet_hash(evidence["spans"])
                != evidence["receipt"]["provider_input_packet_sha256"]
            ):
                raise ValueError("evidence_packet_changed")
            state["evidence"] = {
                **evidence,
                "receipt": {
                    **evidence["receipt"],
                    "provider_invocation_id": f"{run['id']}:{state['generations']}",
                },
            }
            state["evidence"] = fit_provider_evidence(payload, state["evidence"])
            payload["evidence"] = state["evidence"]
            if state["answer_generations"] == 1:
                state["metrics"]["initial_provider_span_ids"] = sorted(
                    span["id"] for span in state["evidence"]["spans"]
                )
                state["metrics"]["initial_receipt_complete"] = bool(
                    state["evidence"]["receipt"]["complete"]
                )
        operation_id = f"{run['id']}:generation:{state['generations']}"
        await self.db(
            self.store.save_audit,
            run["id"],
            "generation.started",
            {
                "generation": state["generations"],
                "phase": phase,
                "payload": payload,
                "operation_id": operation_id,
            },
        )
        await self.db(
            self.store.append_event,
            run["id"],
            "stage.started",
            {
                "stage": "generation",
                "attempt": state["generations"],
                "phase": phase,
                "model": run["model"],
            },
        )
        async with self.telemetry.observation(
            "answer.plan_query" if phase == "query_planning" else "answer.generate",
            "generation",
            payload,
            model=run["model"],
            metadata={"operation_id": operation_id},
            persist=self.db,
        ) as observation:
            try:
                result = await self.provider.generate(
                    payload,
                    role="answer",
                    model=run["model"],
                    request_id=run["id"],
                    final_only=final_only,
                )
            except ProviderError as exc:
                estimate = estimate_api_equivalent_cost(
                    run["model"], exc.usage, billing_mode="unknown"
                )
                failure = {
                    "generation": state["generations"],
                    "phase": phase,
                    "code": exc.code,
                    "usage": exc.usage,
                    "raw_provider_usage": exc.raw_provider_usage,
                    "api_equivalent_estimate": estimate,
                    "response_status": exc.response_status,
                    "response_reason": exc.response_reason,
                    "diagnostic": exc.diagnostic,
                    "exit_code": exc.exit_code,
                    "operation_id": operation_id,
                }
                await self.db(self.store.save_audit, run["id"], "generation.failed", failure)
                if observation:
                    usage = exc.usage
                    reported = {
                        key: usage[name]
                        for key, name in (("input", "input_tokens"), ("output", "output_tokens"))
                        if isinstance(usage.get(name), int)
                    }
                    components = estimate.get("cost_components_usd")
                    cost_details = (
                        {
                            "input": float(
                                Decimal(components["uncached_input"])
                                + Decimal(components["cached_input"])
                            ),
                            "output": float(components["output_including_reasoning"]),
                        }
                        if components
                        else None
                    )
                    observation.update(
                        usage_details=reported or None,
                        cost_details=cost_details,
                        metadata={
                            "provider_error_code": exc.code,
                            "usage": usage,
                            "api_equivalent_estimate": estimate,
                            "operation_id": operation_id,
                        },
                    )
                raise
            estimate = estimate_api_equivalent_cost(
                run["model"],
                result.get("usage", {}),
                billing_mode=result.get("billing_mode", "unknown"),
            )
            result["api_equivalent_estimate"] = estimate
            if observation:
                usage = result.get("usage", {})
                reported = {
                    key: usage[name]
                    for key, name in (("input", "input_tokens"), ("output", "output_tokens"))
                    if isinstance(usage.get(name), int)
                }
                components = estimate.get("cost_components_usd")
                cost_details = (
                    {
                        "input": float(
                            Decimal(components["uncached_input"])
                            + Decimal(components["cached_input"])
                        ),
                        "output": float(components["output_including_reasoning"]),
                    }
                    if components
                    else None
                )
                observation.update(
                    output=result["output"],
                    usage_details=reported or None,
                    cost_details=cost_details,
                    metadata={
                        "actual_model": run["model"],
                        "usage": usage,
                        "usage_provenance": "reported" if reported else "unknown",
                        "billing_mode": result.get("billing_mode"),
                        "api_equivalent_estimate": estimate,
                        "pricing_status": "hypothetical_api_estimate",
                        "actual_billed_cost": "unknown",
                        "operation_id": operation_id,
                    },
                )
        await self.db(
            self.store.save_audit,
            run["id"],
            "generation.completed",
            {**result, "operation_id": operation_id},
        )
        await self.check_running(run["id"])
        state["metrics"]["model_calls"] += 1
        state["metrics"]["provider_ms"] += result["elapsed_ms"]
        state["metrics"]["usage"].append(result.get("usage", {}))
        if phase == "answer":
            state["metrics"]["evidence_bytes"] += sum(
                len(s["text"].encode()) for s in state["evidence"]["spans"]
            )
        action = ModelAction.model_validate(result["output"]).model_dump()
        await self.db(
            self.store.append_event,
            run["id"],
            "stage.completed",
            {"stage": "generation", "phase": phase, "elapsed_ms": result["elapsed_ms"]},
        )
        return action

    async def generate(self, state):
        """Generate the next action from the current evidence and prompt policy.

        Count answer attempts independently from query planning. V0/V1 always
        request a final answer; V2/V3 can request tools on the first answer
        attempt and must finish on the second. V3 uses the expanded chronology,
        identity and complete-source-set instructions; other variants use P0.

        Args:
            state (RunState): Run question/model/variant, policy, bounded
                history, evidence receipt and initialized answer counters.

        Returns:
            RunState: Same state with incremented ``answer_generations`` and
            validated ``action``. Provider/audit/receipt side effects and
            failures follow ``invoke_provider``; no answer is published here.
        """
        run = state["run"]
        state["answer_generations"] += 1
        final_only = state["answer_generations"] >= 2 or run["variant"] in {"V0", "V1"}
        payload = {
            "task_mode": "answer",
            "question": run["question"],
            "coverage_policy": state["policy"],
            "history": state["history"],
            "history_older_messages_present": state["history_older_messages_present"],
            "final_only": final_only,
            "available_tools": {} if final_only else _TOOL_CONTRACT,
            "prompt_version": "P1" if run["variant"] == "V3" else "P0",
            "instructions": "Preserve event time, record time, subject, negation, medication status and units. Explain conflicts. Map every material claim to its complete supporting source set. State retrieval limits and never assert unverified absence or exhaustive coverage."
            if run["variant"] == "V3"
            else "Answer the document question with source citations, state retrieval limits, and avoid unverified absence or exhaustive coverage.",
        }
        state["action"] = await self.invoke_provider(state, payload, final_only, "answer")
        return state

    def next_node(self, state):
        """Enforce the variant's single optional evidence escalation.

        Args:
            state (RunState): Validated ``action.kind``, ``run.variant`` and
                ``escalations`` count.

        Returns:
            str: ``tools`` only for a tool request in V2/V3 before escalation;
            otherwise ``validate``, where an unfulfilled tool request becomes
            a partial budget-limit answer. Does not mutate state.
        """
        action = state["action"]
        if (
            action["kind"] == "request_tools"
            and state["run"]["variant"] in {"V2", "V3"}
            and state["escalations"] == 0
        ):
            return "tools"
        return "validate"

    async def tools(self, state):
        """Spend the one additional evidence round on an allowed request batch.

        Accept search/read requests together, or one collect_scope request on
        its own. Search/read extends and repacks existing evidence; whole-scope
        collection replaces it. After this node the graph generates a final
        answer, regardless of whether the resulting receipt is complete.

        Args:
            state (RunState): ``action.requests`` list of tool/query/span_ids
                dicts, current evidence, active run and mutable metrics.

        Returns:
            RunState: Same state with ``escalations=1``, ``second_batch_kind``
            recorded and replacement ``evidence``. Tool calls write audit and
            progress records; an incomplete packet is a valid outcome.

        Raises:
            ValueError: ``empty_tool_request`` for no requests or
                ``tool_batch_not_allowed`` for mixing collect with other
                requests. Nonempty rejected batches still consume the state
                escalation flag and set batch metrics. Evidence helper/scoped
                tool failures propagate and can leave earlier audit writes.
        """
        requests = state["action"]["requests"]
        if not requests:
            raise ValueError("empty_tool_request")
        state["escalations"] = 1
        names = {r["tool"] for r in requests}
        state["metrics"]["second_batch_kind"] = (
            "search_read" if names <= {"search_evidence", "read_evidence"} else next(iter(names))
        )
        if names <= {"search_evidence", "read_evidence"}:
            state["evidence"] = await self.extend_evidence(state, requests)
        elif len(requests) == 1 and names == {"collect_scope"}:
            state["evidence"] = await self.collect(state)
        else:
            raise ValueError("tool_batch_not_allowed")
        return state

    async def validate(self, state):
        """Build a final answer with citations bound to this exact model input.

        Reject claims with missing text/citations or IDs outside delivered
        spans, and verify the receipt's invocation, packet hash and frozen
        inventory. Require complete coverage for exhaustive question/action
        policy, not_documented status or recognized global-absence wording.
        Without it, reduce the action to a partial evidence answer. An unused
        tool request becomes a partial budget-limit answer; missing additional
        groups also cause partial output, while mere omitted excerpts add a
        limitation. Render material claims in their original order and resolve
        citation records in sorted identifier order.

        These checks establish citation membership and delivery/coverage
        consistency, not semantic support, complete claim extraction, patient
        identity resolution or clinical validity. Absence detection uses an
        English text pattern and can miss wording outside that pattern.

        Args:
            state (RunState): Active run/scope, ModelAction ``action``, policy,
                generation counter, evidence span dicts (id/source_id/text
                and source metadata), invocation-bound receipt and metrics.

        Returns:
            RunState: Same state with ``answer`` containing action fields,
            resolved ``citations`` and ``coverage`` receipt. Mutates delivered
            ID/coverage/guard metrics and persists validation audit/progress;
            it does not update the original ``action`` or publish the answer.

        Raises:
            ValueError: ``citation_outside_delivered_evidence``,
                ``claim_missing_citation``, ``claim_missing_text`` or
                ``evidence_invocation_changed`` for invalid evidence binding;
                ``supported_answer_without_claims`` for a supported answer
                with blank answer or no claims; ``answer_claims_too_long``
                when rendered claims exceed 32,000 characters. Run validity
                failures also propagate. The validation start event may be
                committed before any rejection.
            asyncio.CancelledError: The run has stopped or execution is
                cancelled; earlier audit/progress writes are retained.
        """
        run = state["run"]
        await self.check_running(run["id"])
        action = state["action"]
        await self.db(self.store.append_event, run["id"], "stage.started", {"stage": "validation"})
        spans = {s["id"]: s for s in state["evidence"]["spans"]}
        cited = {sid for claim in action["claims"] for sid in claim["evidence_ids"]}
        if cited - set(spans):
            raise ValueError("citation_outside_delivered_evidence")
        if any(not claim["evidence_ids"] for claim in action["claims"]):
            raise ValueError("claim_missing_citation")
        if any(not claim["text"].strip() for claim in action["claims"]):
            raise ValueError("claim_missing_text")
        receipt = state["evidence"]["receipt"]
        if (
            receipt.get("provider_invocation_id") != f"{run['id']}:{state['generations']}"
            or packet_hash(state["evidence"]["spans"]) != receipt["provider_input_packet_sha256"]
            or receipt["snapshot_hash"] != run["scope"]["snapshot_hash"]
            or receipt["inventory_count"] != run["scope"]["source_count"]
            or receipt["unavailable_count"] != run["scope"]["unavailable_count"]
            or set(receipt["delivered_source_ids"])
            != {span["source_id"] for span in spans.values()}
        ):
            raise ValueError("evidence_invocation_changed")
        global_absence = bool(
            re.search(
                r"\b(?:no|none|never|not)\b.{0,160}\b(?:in|across|throughout)\s+"
                r"(?:the\s+)?(?:records?|documents?|sources?|collection|library|archive)\b",
                " ".join(
                    [action["answer"], *(claim["text"] for claim in action["claims"])]
                ).casefold(),
            )
        )
        exhaustive = (
            state["policy"] == "complete_scope"
            or action["coverage_requirement"] == "complete_scope"
            or action["status"] == "not_documented"
            or global_absence
        )
        missing_groups = receipt.get("missing_requested_groups") or []
        missing_reason = (
            "No excerpts were delivered for additional evidence request "
            + ", ".join(str(index) for index in missing_groups)
            + ". The requested evidence is incomplete."
            if missing_groups
            else None
        )
        if action["kind"] != "answer":
            state["metrics"]["guard_action"] = "tool_budget_exhausted"
            action = limited_answer(
                "The configured tool budget was exhausted. Ask a narrower question or upload a smaller set to a new collection."
            )
            cited = set()
        elif exhaustive and not receipt["complete"]:
            state["metrics"]["guard_action"] = "coverage_unverified"
            action = limited_evidence_answer(
                action, spans, _coverage_limit_text(), preserve_claims=not global_absence
            )
            cited = {sid for claim in action["claims"] for sid in claim["evidence_ids"]}
        elif missing_reason:
            state["metrics"]["guard_action"] = "requested_evidence_missing"
            action = limited_evidence_answer(action, spans, missing_reason)
            cited = {sid for claim in action["claims"] for sid in claim["evidence_ids"]}
        elif action["status"] == "supported" and (
            not action["answer"].strip() or not action["claims"]
        ):
            raise ValueError("supported_answer_without_claims")
        if missing_reason and missing_reason not in action["limitations"]:
            action = {
                **action,
                "limitations": [*action["limitations"], missing_reason],
            }
        if receipt.get("retrieval_truncated") and not missing_reason:
            truncation_reason = (
                "Some additional retrieved excerpts did not fit in the evidence packet."
            )
            if truncation_reason not in action["limitations"]:
                action = {
                    **action,
                    "limitations": [*action["limitations"], truncation_reason],
                }
        if state["metrics"]["guard_action"] is None:
            state["metrics"]["guard_action"] = "none"
        if action["claims"]:
            rendered = "\n".join(claim["text"] for claim in action["claims"])
            if len(rendered) > 32000:
                raise ValueError("answer_claims_too_long")
            action = {**action, "answer": rendered}
        answer = {
            **action,
            "citations": [spans[sid] for sid in sorted(cited)],
            "coverage": receipt,
        }
        state["metrics"]["delivered_span_ids"] = sorted(spans)
        state["metrics"]["final_provider_span_ids"] = sorted(spans)
        state["metrics"]["final_receipt_complete"] = bool(receipt["complete"])
        state["metrics"]["coverage_complete"] = receipt["complete"]
        state["answer"] = answer
        await self.db(
            self.store.save_audit,
            run["id"],
            "validation.completed",
            {"answer": answer, "metrics": state["metrics"]},
        )
        await self.db(
            self.store.append_event, run["id"], "stage.completed", {"stage": "validation"}
        )
        return state

    async def execute(self, run):
        """Execute one bounded dialogue and return its publishable candidate.

        Invoke the compiled graph with a recursion limit of twelve and
        the configured wall-time timeout, then add elapsed milliseconds to its
        metrics. The caller must persist the answer through the store's atomic
        active-run/frozen-scope publication check; returning here does not
        guarantee that a collection change will still permit publication.

        Args:
            run (dict): Already-started stored run with string ``id``,
                ``conversation_id``, ``question``, ``model``, ``variant``,
                optional retry ancestor and frozen ``scope`` counts/hash.

        Returns:
            tuple[dict, dict]: Answer action plus ``citations`` and ``coverage``
            receipt, and metrics for calls, usage, elapsed milliseconds,
            UTF-8 evidence bytes, retrieval identifiers and coverage guards.
            Clarification can return without retrieving source evidence.

        Raises:
            TimeoutError: The configured run timeout elapses and cancels the
                graph task.
            asyncio.CancelledError: Caller cancellation or a stopped run.
            ValueError: Scope/evidence/action guard failures from graph nodes.
            ProviderError: Inference failure. Other storage, MCP, telemetry,
                schema and graph failures propagate. This method does not
                mark the run failed or undo committed audits/events; repeating
                execution is not idempotent and may repeat inference calls.
        """
        start = time.perf_counter()
        result = await asyncio.wait_for(
            self.graph.ainvoke({"run": run}, {"recursion_limit": 12}),
            self.settings.run_timeout_seconds,
        )
        result["metrics"]["end_to_end_ms"] = (time.perf_counter() - start) * 1000
        return result["answer"], result["metrics"]
