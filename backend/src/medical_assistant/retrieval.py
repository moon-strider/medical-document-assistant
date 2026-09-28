import base64
import binascii
import hashlib
import hmac
import json
import secrets
from contextlib import contextmanager
from time import perf_counter
from uuid import UUID

from sqlalchemy import create_engine, text

from medical_assistant.candidate_search import search_candidates
from medical_assistant.embedding import embed_query
from medical_assistant.reranking import rerank_candidates
from medical_assistant.scope import scope_matches

_SEARCH_STATEMENT_TIMEOUT_MS = 100000
_READ_STATEMENT_TIMEOUT_MS = 25000


class RetrievalError(Exception):
    """Expose a stable rejection code for the read-only evidence tool boundary.

    ``code`` (str) describes an invalid request, expired scope or unresolved
    evidence identity; the MCP adapter publishes it as a tool error. Database
    and model failures are not automatically converted to this exception here.
    """

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise RetrievalError("invalid_id") from exc


def _jsonable(row) -> dict:
    result = dict(row)
    for key, value in result.items():
        if isinstance(value, UUID):
            result[key] = str(value)
        elif hasattr(value, "isoformat"):
            result[key] = value.isoformat()
    return result


def _byte_prefix(value: str, budget: int) -> str:
    encoded = value.encode("utf-8")[:budget]
    return encoded.decode("utf-8", errors="ignore")


class Retrieval:
    """Serve evidence from the collection version frozen when a run was created.

    Each database read uses a read-only, repeatable-read transaction. Frozen
    revision/counters and the conversation's collection identity gate access to
    ready, undeleted evidence. These transactions do not hold a historical
    archive snapshot between calls; collection changes invalidate subsequent
    access. Search also rechecks scope after local ranking before returning.

    Args:
        database_url (str): Required PostgreSQL connection URL for evidence reads.
        cursor_key (bytes | None): Optional 32-byte HMAC key for collection paging.
            Omitting it creates an instance-local random key, so cursors are not
            portable across newly created instances without a shared key.

    The service opens pooled connections and may invoke local embedding/rerank
    models, but never writes application rows. Cursor authentication binds
    continuation state to a run and page size; it does not encrypt its contents.
    """

    def __init__(self, database_url: str, *, cursor_key: bytes | None = None):
        if not database_url:
            raise ValueError("read_database_url_required")
        if cursor_key is not None and len(cursor_key) != 32:
            raise ValueError("invalid_cursor_key")
        self._cursor_key = cursor_key if cursor_key is not None else secrets.token_bytes(32)
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_timeout=5,
            connect_args={
                "connect_timeout": 5,
                "options": "-c default_transaction_read_only=on -c statement_timeout=100000",
            },
        )

    def close(self) -> None:
        self.engine.dispose()

    @contextmanager
    def _transaction(self, statement_timeout_ms: int):
        with self.engine.connect() as connection:
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            connection.execute(
                text("SELECT set_config('statement_timeout', :timeout, true)"),
                {"timeout": str(statement_timeout_ms)},
            )
            yield connection

    def _scope(self, connection, run_scope_id: str) -> dict:
        """Resolve an active run's scope and reject changed collection metadata.

        Args:
            connection (sqlalchemy.engine.Connection): Current read transaction.
            run_scope_id (str): Run UUID, also the frozen scope's identity.

        Returns:
            dict: Stored scope with run/collection IDs, revision, source/ready/
            unavailable counts and ``snapshot_hash``; no document text is loaded.

        Raises:
            RetrievalError: ``invalid_id`` for a malformed run UUID;
                ``scope_expired`` for a missing/nonactive run, malformed stored
                scope, missing collection or mismatch with its current metadata
                or conversation collection. Only queued/running runs may read.
        """
        run_id = _uuid(run_scope_id)
        row = (
            connection.execute(
                text(
                    "SELECT r.scope, r.status, c.collection_id AS conversation_collection_id "
                    "FROM runs r JOIN conversations c ON c.id = r.conversation_id "
                    "WHERE r.id = :id"
                ),
                {"id": run_id},
            )
            .mappings()
            .first()
        )
        if (
            row is None
            or row["status"] not in {"queued", "running"}
            or not isinstance(row["scope"], dict)
        ):
            raise RetrievalError("scope_expired")
        scope = row["scope"]
        try:
            collection_id = _uuid(scope.get("collection_id"))
        except RetrievalError as exc:
            raise RetrievalError("scope_expired") from exc
        collection = (
            connection.execute(
                text(
                    "SELECT id, revision, source_count, ready_count, unavailable_count "
                    "FROM collections WHERE id = :id"
                ),
                {"id": collection_id},
            )
            .mappings()
            .first()
        )
        if not scope_matches(scope, collection, run_id, row["conversation_collection_id"]):
            raise RetrievalError("scope_expired")
        return scope

    def search_evidence(
        self, run_scope_id: str, query: str, variant: str = "V3", limit: int = 12
    ) -> dict:
        """Find ranked evidence references for a question within an active run.

        Embed the question locally, search the run's collection partition, and
        for V3 rerank the bounded fused pool with a local cross-encoder. A second
        scope check after ranking catches collection/run changes observed by
        that check. Returned snippets guide selection; the caller must use
        ``read_evidence`` to obtain original spans for citation support. No
        search variant proves exhaustive archive coverage or patient counts.

        Args:
            run_scope_id (str): UUID of a queued/running run with unchanged
                collection revision and source counters.
            query (str): Nonblank question of at most 512 characters after
                stripping surrounding whitespace, independent of UTF-8 byte size.
            variant (str): V0 for fixed-chunk dense search; V1/V2 for structural
                hybrid search; V3 adds reranking of up to 64 fused candidates.
            limit (int): Maximum returned candidates, from 1 through 12.

        Returns:
            dict: ``scope_applied`` (str run ID), normalized ``query``, ``variant``,
            ``no_candidates`` (bool) and ordered ``candidates`` (list[dict]). Each
            candidate includes string chunk/source IDs, source title/file hash,
            ``span_ids`` (list[str]), a snippet of at most 512 UTF-8 bytes, float
            RRF score and contributing branch names. V3 adds ``rerank_score``
            (float); full chunk text is omitted. Ordering follows fusion score
            and chunk ID, or rerank score, fusion score and chunk ID for V3.
            ``retrieval`` supplies lane/pool counts and milliseconds per stage
            and overall; V3's ``rerank`` adds model/revision and scoring statistics.
            Empty results are valid and are not evidence of archive-wide absence.
            Calls consume local model/database resources but persist no state.

        Raises:
            RetrievalError: ``invalid_search`` for query/limit violations,
                ``invalid_variant`` for unknown variants, ``embedding_dimension``
                unless the query vector has 384 components, or scope/ID rejection.
            ValueError: Candidate search/rerank validation failures propagate.
            RuntimeError: Index/source inconsistencies or local model/scoring
                failures propagate; database errors also propagate unchanged.
        """
        started = perf_counter()
        query = query.strip()
        if not query or len(query) > 512 or limit < 1 or limit > 12:
            raise RetrievalError("invalid_search")
        if variant not in {"V0", "V1", "V2", "V3"}:
            raise RetrievalError("invalid_variant")
        embedding_started = perf_counter()
        query_vector = embed_query(query)
        embedding_ms = round((perf_counter() - embedding_started) * 1000, 3)
        if len(query_vector) != 384:
            raise RetrievalError("embedding_dimension")
        with self._transaction(_SEARCH_STATEMENT_TIMEOUT_MS) as connection:
            scope = self._scope(connection, run_scope_id)
            result = search_candidates(
                connection,
                scope["collection_id"],
                query,
                variant,
                limit,
                query_vector=query_vector,
            )
        if variant == "V3":
            ranked = rerank_candidates(query, result["candidates"], limit=limit)
            result["candidates"] = ranked["candidates"]
            result["rerank"] = ranked["rerank"]
        with self._transaction(_READ_STATEMENT_TIMEOUT_MS) as connection:
            self._scope(connection, run_scope_id)
        result["retrieval"]["query_stage_ms"]["embedding"] = embedding_ms
        result["retrieval"]["search_ms"] = round((perf_counter() - started) * 1000, 3)
        candidates = [
            {key: value for key, value in candidate.items() if key != "text"}
            for candidate in result["candidates"]
        ]
        return {
            **result,
            "scope_applied": scope["id"],
            "candidates": candidates,
            "no_candidates": not candidates,
        }

    def _spans(self, connection, ids: list[str], scope: dict) -> list[dict]:
        """Resolve all requested span identities against the permitted file versions.

        Args:
            connection (sqlalchemy.engine.Connection): Current read transaction.
            ids (list[str]): Canonical span UUIDs in desired delivery order.
            scope (dict): Validated run scope with ``collection_id`` (str).

        Returns:
            list[dict]: Original extracted spans in input order, including text,
            source title/hash and location fields. Span ``sha256`` must equal its
            source's original-file hash; this checks recorded version identity,
            without recomputing a digest of span text. Empty IDs return [].

        Raises:
            RetrievalError: ``unknown_span`` if any ID is missing, outside the
                collection, deleted/unready, or from a different recorded file
                version. A partial result is never returned for unresolved IDs.
        """
        if not ids:
            return []
        rows = (
            connection.execute(
                text(
                    "SELECT sp.id, sp.source_id, sp.page, sp.line_start, sp.line_end, sp.text, "
                    "sp.bbox, sp.page_width, sp.page_height, sp.sha256, sp.section, "
                    "s.title AS source_title, s.sha256 AS source_hash "
                    "FROM spans sp JOIN sources s ON s.id = sp.source_id "
                    "WHERE sp.id = ANY(CAST(:ids AS uuid[])) "
                    "AND sp.collection_id = CAST(:collection_id AS uuid) "
                    "AND s.collection_id = CAST(:collection_id AS uuid) "
                    "AND s.status = 'ready' AND s.deleted_at IS NULL "
                    "AND sp.sha256 = s.sha256"
                ),
                {"ids": ids, "collection_id": scope["collection_id"]},
            )
            .mappings()
            .all()
        )
        found = {str(row["id"]): _jsonable(row) for row in rows}
        if set(ids) != set(found):
            raise RetrievalError("unknown_span")
        return [found[item] for item in ids]

    def read_evidence(
        self, run_scope_id: str, span_ids: list[str], budget_bytes: int = 16384
    ) -> dict:
        """Deliver original spans and a conservative receipt for evidence coverage.

        Resolve every requested identity before applying the text budget.
        Duplicate IDs are read once in first-request order. Spans are indivisible:
        one that does not fit is skipped, and later smaller spans may still fit.
        Thus ``truncated`` means requested spans were omitted, not that a span's
        text was shortened. Only UTF-8 text bytes consume the budget, not metadata
        or the serialized response size.

        Args:
            run_scope_id (str): Active run UUID whose frozen collection still matches.
            span_ids (list[str]): At most 64 UUID strings before deduplication;
                an empty list is permitted, but every supplied ID must resolve.
            budget_bytes (int): Text capacity from 1 through 16,384 UTF-8 bytes.

        Returns:
            dict: ``spans`` (list[dict]) carry string span/source IDs, full text,
            source title/hash, recorded ``sha256``, section and nullable page,
            line and geometry fields. ``delivered_source_ids`` preserves first
            delivery order. ``text_bytes`` is the delivered text size;
            ``scope_applied`` and ``snapshot_hash`` identify frozen metadata.
            ``inventory_count`` counts all sources, including ``unavailable_count``.

            ``inventory_complete`` is True only if there are no unavailable
            sources, the entire eligible inventory has at most 64 spans, every
            ready source is represented, its text fits the budget and exactly
            all inventory span IDs were delivered without omission. False can
            accompany a successful read with no truncation. The bounded inventory
            probe runs only with no unavailable sources and at most 64 ready
            sources; ``span_inventory_count`` and ``span_inventory_sha256`` are
            None if that probe is skipped or finds more than 64 spans. Otherwise
            the hash is SHA-256 of compact JSON of sorted span ID strings, not
            a digest of their text or files. No state is persisted.

        Raises:
            RetrievalError: ``window_too_large`` for ID-count/budget violations,
                ``invalid_id`` for malformed UUIDs, ``scope_expired`` for stale
                scope, or ``unknown_span`` if any supplied ID cannot be read.
                Missing IDs reject the entire request even if other spans fit.
        """
        if len(span_ids) > 64 or budget_bytes < 1 or budget_bytes > 16384:
            raise RetrievalError("window_too_large")
        ids = list(dict.fromkeys(_uuid(item) for item in span_ids))
        with self._transaction(_READ_STATEMENT_TIMEOUT_MS) as connection:
            scope = self._scope(connection, run_scope_id)
            spans = self._spans(connection, ids, scope)
            potentially_complete = scope["unavailable_count"] == 0 and scope["ready_count"] <= 64
            inventory = (
                connection.execute(
                    text(
                        "SELECT sp.id, sp.source_id, octet_length(sp.text) AS byte_count "
                        "FROM spans sp JOIN sources s ON s.id = sp.source_id "
                        "WHERE sp.collection_id = CAST(:collection_id AS uuid) "
                        "AND s.collection_id = CAST(:collection_id AS uuid) "
                        "AND s.status = 'ready' AND s.deleted_at IS NULL "
                        "AND sp.sha256 = s.sha256 LIMIT 65"
                    ),
                    {"collection_id": scope["collection_id"]},
                )
                .mappings()
                .all()
                if potentially_complete
                else []
            )
        selected = []
        byte_count = 0
        for span in spans:
            size = len(span["text"].encode("utf-8"))
            if byte_count + size <= budget_bytes:
                selected.append(span)
                byte_count += size
        truncated = len(selected) < len(spans)
        inventory_ids = {str(row["id"]) for row in inventory}
        inventory_sources = {str(row["source_id"]) for row in inventory}
        delivered_ids = {span["id"] for span in selected}
        inventory_count = len(inventory) if potentially_complete and len(inventory) <= 64 else None
        inventory_hash = (
            hashlib.sha256(
                json.dumps(sorted(inventory_ids), separators=(",", ":")).encode()
            ).hexdigest()
            if inventory_count is not None
            else None
        )
        inventory_complete = (
            potentially_complete
            and inventory_count is not None
            and len(inventory_sources) == scope["ready_count"]
            and sum(row["byte_count"] for row in inventory) <= budget_bytes
            and inventory_ids == delivered_ids
            and not truncated
        )
        return {
            "scope_applied": scope["id"],
            "spans": selected,
            "delivered_source_ids": list(dict.fromkeys(span["source_id"] for span in selected)),
            "text_bytes": byte_count,
            "truncated": truncated,
            "inventory_complete": inventory_complete,
            "inventory_count": scope["source_count"],
            "unavailable_count": scope["unavailable_count"],
            "span_inventory_count": inventory_count,
            "span_inventory_sha256": inventory_hash,
            "snapshot_hash": scope["snapshot_hash"],
        }

    def _cursor(self, scope: dict, position: list, page_bytes: int) -> str:
        """Authenticate a continuation position within one frozen run scope.

        Args:
            scope (dict): Validated run/collection IDs, revision and snapshot hash.
            position (list): Source UUID string, one-based span ordinal and
                zero-based fragment index identifying the next unread fragment.
            page_bytes (int): Text budget used to split spans; continuation must
                preserve it to keep fragment indices meaningful.

        Returns:
            str: URL-safe base64 JSON payload plus SHA-256 HMAC. The token is
            tamper-evident with this service's key, not encrypted or persisted.
        """
        payload = json.dumps(
            {
                "scope": scope["id"],
                "collection": scope["collection_id"],
                "revision": scope["revision"],
                "snapshot": scope["snapshot_hash"],
                "position": position,
                "page_bytes": page_bytes,
            },
            separators=(",", ":"),
        ).encode()
        signature = hmac.new(self._cursor_key, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(payload + signature).decode()

    def _cursor_position(self, scope: dict, cursor: str, page_bytes: int) -> list | None:
        """Validate a paging token before interpreting its source/span position.

        Args:
            scope (dict): Current validated frozen run metadata.
            cursor (str): Token from ``collect_scope``, or "" to start at the beginning.
            page_bytes (int): Same UTF-8 text budget as the token's originating page.

        Returns:
            list | None: Canonical source UUID, positive ordinal and nonnegative
            fragment index, or None for the initial page. No data is changed.

        Raises:
            RetrievalError: ``invalid_cursor`` for oversized, malformed, modified,
                differently bound or invalid-position tokens; ``invalid_id`` can
                propagate for a malformed position UUID inside an authenticated token.
        """
        if not cursor:
            return None
        if len(cursor) > 1024:
            raise RetrievalError("invalid_cursor")
        try:
            raw = base64.urlsafe_b64decode(cursor.encode())
            payload, signature = raw[:-32], raw[-32:]
            expected = hmac.new(self._cursor_key, payload, hashlib.sha256).digest()
            data = json.loads(payload)
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            if (
                data["scope"] != scope["id"]
                or data["collection"] != scope["collection_id"]
                or data["revision"] != scope["revision"]
                or data["snapshot"] != scope["snapshot_hash"]
            ):
                raise ValueError
            position = data["position"]
            if (
                data["page_bytes"] != page_bytes
                or not isinstance(position, list)
                or len(position) != 3
            ):
                raise ValueError
            if (
                type(position[1]) is not int
                or position[1] < 1
                or type(position[2]) is not int
                or position[2] < 0
            ):
                raise ValueError
            if _uuid(position[0]) != position[0]:
                raise ValueError
            return position
        except (ValueError, KeyError, TypeError, UnicodeError, binascii.Error) as exc:
            raise RetrievalError("invalid_cursor") from exc

    def collect_scope(self, run_scope_id: str, cursor: str = "", page_bytes: int = 8192) -> dict:
        """Page through eligible raw spans to support explicit whole-scope reading.

        Traverse by source UUID and span ordinal rather than retrieval relevance.
        Oversized spans are split into UTF-8-safe fragments using the page budget;
        a continuation can resume inside a span. Each response contains at most
        64 fragments and at most ``page_bytes`` text bytes. Metadata does not count
        toward that budget. Every call revalidates the frozen scope, so changed
        collection metadata expires traversal instead of silently mixing versions.

        Args:
            run_scope_id (str): Active run UUID with unchanged collection metadata.
            cursor (str): "" for the first page, then the preceding ``next_cursor``.
                Tokens are bound to run, collection, revision, snapshot, page size
                and service HMAC key; an identical cursor rereads the same position
                while its scope remains valid.
            page_bytes (int): Capacity from 1 through 8,192 UTF-8 text bytes.
                Keep it unchanged across a traversal.

        Returns:
            dict: ``spans`` (list[dict]) have original span/source IDs, location
            and file-version metadata plus fragment ``text``, zero-based
            ``fragment_index`` and total ``fragment_count`` for that span. A
            fragmented span retains the original ID/location; concatenate all
            fragments in index order to reconstruct its text. ``text_bytes``
            counts this page, ``delivered_source_ids`` preserves page order,
            ``cursor`` echoes the request and ``next_cursor`` is "" at the end.
            ``scope_applied``/``snapshot_hash`` identify frozen metadata and
            ``inventory_count``/``unavailable_count`` are scope-wide source counts.

            ``inventory_complete`` means end-of-traversal with no unavailable
            sources; it does not establish that this page alone covers the scope
            or that every ready source has extracted spans. The caller must retain
            preceding pages and verify source/fragment coverage. ``truncated`` is
            True when more pages remain or any sources are unavailable. Empty
            eligible inventories can return an empty final page. No state is written.

        Raises:
            RetrievalError: ``invalid_page_size`` for an out-of-range budget or
                a budget too small to hold the next UTF-8 character;
                ``invalid_cursor`` for rejected or unresolvable continuations;
                ``invalid_id``/``scope_expired`` for invalid run identity or scope.
        """
        if page_bytes < 1 or page_bytes > 8192:
            raise RetrievalError("invalid_page_size")
        with self._transaction(_READ_STATEMENT_TIMEOUT_MS) as connection:
            scope = self._scope(connection, run_scope_id)
            position = self._cursor_position(scope, cursor, page_bytes)
            params = {"collection_id": scope["collection_id"]}
            after = ""
            if position:
                after = "AND (sp.source_id, sp.ordinal) >= (CAST(:source AS uuid), :ordinal) "
                params.update({"source": position[0], "ordinal": position[1]})
            rows = (
                connection.execute(
                    text(
                        "SELECT sp.id, sp.source_id, sp.ordinal, sp.page, sp.line_start, sp.line_end, sp.text, "
                        "sp.bbox, sp.page_width, sp.page_height, sp.sha256, sp.section, "
                        "s.title AS source_title, s.sha256 AS source_hash "
                        "FROM spans sp JOIN sources s ON s.id = sp.source_id "
                        "WHERE sp.collection_id = CAST(:collection_id AS uuid) "
                        "AND s.collection_id = CAST(:collection_id AS uuid) "
                        "AND s.status = 'ready' AND s.deleted_at IS NULL "
                        "AND sp.sha256 = s.sha256 "
                        + after
                        + "ORDER BY sp.source_id, sp.ordinal LIMIT 65"
                    ),
                    params,
                )
                .mappings()
                .all()
            )
        page = []
        used = 0
        next_position = None
        for row_index, row in enumerate(rows):
            span = _jsonable(row)
            key = [span["source_id"], span.pop("ordinal")]
            first_fragment = position[2] if position and row_index == 0 else 0
            if position and row_index == 0 and key != position[:2]:
                raise RetrievalError("invalid_cursor")
            if len(page) >= 64:
                next_position = [*key, first_fragment]
                break
            remaining = span["text"]
            fragments = []
            while remaining:
                part = _byte_prefix(remaining, page_bytes)
                if not part:
                    raise RetrievalError("invalid_page_size")
                fragments.append(part)
                remaining = remaining[len(part) :]
            fragments = fragments or [""]
            if first_fragment >= len(fragments):
                raise RetrievalError("invalid_cursor")
            for number in range(first_fragment, len(fragments)):
                fragment = fragments[number]
                size = len(fragment.encode("utf-8"))
                if used + size > page_bytes or len(page) >= 64:
                    next_position = [*key, number]
                    break
                page.append(
                    {
                        **span,
                        "text": fragment,
                        "fragment_index": number,
                        "fragment_count": len(fragments),
                    }
                )
                used += size
            if next_position:
                break
        if position and not rows:
            raise RetrievalError("invalid_cursor")
        next_cursor = self._cursor(scope, next_position, page_bytes) if next_position else ""
        delivered = list(dict.fromkeys(item["source_id"] for item in page))
        return {
            "scope_applied": scope["id"],
            "spans": page,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "inventory_count": scope["source_count"],
            "delivered_source_ids": delivered,
            "unavailable_count": scope["unavailable_count"],
            "inventory_complete": not next_cursor and scope["unavailable_count"] == 0,
            "truncated": bool(next_cursor) or scope["unavailable_count"] > 0,
            "snapshot_hash": scope["snapshot_hash"],
            "text_bytes": used,
        }
