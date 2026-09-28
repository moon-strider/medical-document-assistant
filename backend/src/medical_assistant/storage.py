import json
import re
import threading
import uuid
from pathlib import Path

import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from medical_assistant.errors import ConflictError
from medical_assistant.scope import build_run_scope, scope_matches

_HEX = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,200}$")
_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_API_OWNER_LOCK = 739164206
_COLLECTION_SELECT = (
    "SELECT c.*,EXISTS(SELECT 1 FROM sources s WHERE s.collection_id=c.id "
    "AND s.status IN ('pending','processing')) AS has_pending_sources FROM collections c"
)


class _APIConnection(psycopg.Connection):
    def wait(self, gen, interval=0.1, timeout=None):
        return super().wait(
            gen, interval=interval, timeout=min(timeout, 10.0) if timeout is not None else 10.0
        )


def _new_id():
    return str(uuid.uuid4())


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record(row):
    if row is None:
        return None
    result = dict(row)
    for key, value in result.items():
        if isinstance(value, uuid.UUID):
            result[key] = str(value)
        elif hasattr(value, "isoformat"):
            result[key] = value.isoformat()
    return result


def _event(conn, run_id, event_type, payload):
    """Append a sequenced event within the caller's transaction.

    Lock the run before allocating its next sequence number so concurrent event
    writers serialize. The caller's state changes and event commit or roll back
    together; this helper does not commit independently.

    Args:
        conn (sqlalchemy.engine.Connection): Active transaction connection.
        run_id (str): Existing run UUID.
        event_type (str): Event name, such as ``answer.ready``.
        payload (dict): JSON-serializable event-specific fields.

    Returns:
        dict: Event with ``run_id``, integer ``seq`` starting at 1, ``type``,
        ISO-formatted ``at`` and decoded JSON ``payload``.

    Raises:
        KeyError: The run does not exist.
    """
    current = conn.execute(
        text("SELECT id FROM runs WHERE id=:id FOR UPDATE"), {"id": run_id}
    ).first()
    if current is None:
        raise KeyError(run_id)
    seq = conn.execute(
        text("SELECT COALESCE(MAX(seq),0)+1 FROM run_events WHERE run_id=:id"), {"id": run_id}
    ).scalar_one()
    row = (
        conn.execute(
            text(
                "INSERT INTO run_events(run_id,seq,type,payload) VALUES (:id,:seq,:type,CAST(:payload AS jsonb)) RETURNING *"
            ),
            {"id": run_id, "seq": seq, "type": event_type, "payload": _json(payload)},
        )
        .mappings()
        .one()
    )
    return _record(row)


class Store:
    """Persist source lifecycles, leased jobs and conversation execution state.

    Mutations use database transactions; filesystem and provider operations are
    outside this boundary. Returned row dictionaries stringify UUIDs and format
    timestamps as ISO strings while retaining decoded JSON and nullable fields.
    Database, constraint and JSON-serialization failures propagate unless a
    method explicitly handles them. Creating the store configures an engine;
    it does not migrate the schema or acquire API ownership.

    Args:
        settings (Settings): Application settings containing ``database_url``.
    """

    def __init__(self, settings):
        self.settings = settings
        self.engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 5,
                "options": "-c statement_timeout=120000 -c lock_timeout=10000",
            },
        )
        self._api_owner_lock = threading.Lock()
        self._api_owner = None
        self._api_owner_pid = None

    def acquire_api_ownership(self):
        """Try to reserve the database's single API-owner session.

        Retain a dedicated autocommit connection holding a PostgreSQL advisory
        lock. Ownership ends when that connection closes; ordinary pooled
        connections do not hold it. A failed attempt closes its connection.

        Returns:
            bool: True after acquiring and retaining ownership; False when
            another database session already holds the lock.

        Raises:
            RuntimeError: This store already retains an ownership connection.
        """
        with self._api_owner_lock:
            if self._api_owner is not None:
                raise RuntimeError("API ownership already acquired")
            url = make_url(self.settings.database_url).set(drivername="postgresql")
            connection = _APIConnection.connect(
                url.render_as_string(hide_password=False),
                autocommit=True,
                connect_timeout=5,
                options="-c statement_timeout=9000",
            )
            try:
                acquired = connection.execute(
                    "SELECT pg_try_advisory_lock(%s)", (_API_OWNER_LOCK,)
                ).fetchone()[0]
                if not acquired:
                    return False
                self._api_owner_pid = connection.execute("SELECT pg_backend_pid()").fetchone()[0]
                self._api_owner = connection
                connection = None
                return True
            finally:
                if connection is not None:
                    connection.close()

    def api_owner_healthy(self):
        """Check that the retained owner connection still has its original PID.

        Returns:
            bool: True for a successful check on the original session. False
            when no owner exists or the check fails. A failed check also clears
            local ownership and attempts to close the retained connection;
            connection errors are swallowed here.
        """
        with self._api_owner_lock:
            if self._api_owner is None:
                return False
            try:
                pid = self._api_owner.execute("SELECT pg_backend_pid()").fetchone()[0]
                if pid == self._api_owner_pid:
                    return True
            except Exception:
                pass
            connection, self._api_owner = self._api_owner, None
            self._api_owner_pid = None
            try:
                connection.close()
            except Exception:
                pass
            return False

    def release_api_ownership(self):
        """Clear local ownership, unlock the advisory lock and close its session.

        Repeated release without a retained connection is a no-op. Unlock
        errors are suppressed, but errors closing a retained connection can
        propagate. Local ownership is cleared before either operation.

        Returns:
            None: Ownership release has been attempted.
        """
        with self._api_owner_lock:
            connection, self._api_owner = self._api_owner, None
            self._api_owner_pid = None
            if connection is None:
                return
            try:
                connection.execute("SELECT pg_advisory_unlock(%s)", (_API_OWNER_LOCK,))
            except Exception:
                pass
            finally:
                connection.close()

    def migrate(self):
        """Apply unapplied numbered SQL migrations in one serialized transaction.

        A transaction advisory lock prevents concurrent migrators. Filenames
        in ``schema_migrations`` make reapplying the same migration a no-op;
        changed contents of an already recorded file are not reapplied.

        Returns:
            None: All pending migrations and their records have committed.
        """
        directory = Path(__file__).resolve().parents[3] / "migrations"
        with self.engine.begin() as conn:
            conn.execute(text("SET LOCAL statement_timeout='30min'"))
            conn.execute(text("SELECT pg_advisory_xact_lock(739164205)"))
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (filename text PRIMARY KEY,applied_at timestamptz NOT NULL DEFAULT now())"
                )
            )
            applied = set(conn.execute(text("SELECT filename FROM schema_migrations")).scalars())
            for path in sorted(directory.glob("[0-9][0-9][0-9]_*.sql")):
                if path.name in applied:
                    continue
                conn.exec_driver_sql(
                    path.read_text(encoding="utf-8"), execution_options={"no_parameters": True}
                )
                conn.execute(
                    text("INSERT INTO schema_migrations(filename) VALUES (:filename)"),
                    {"filename": path.name},
                )

    def create_collection(self, title, description=""):
        """Create a research library and its search partition atomically.

        Args:
            title (str): Nonblank title; surrounding whitespace is removed.
            description (str): Description stored without normalization.

        Returns:
            dict: Collection row with string ``id``, ``title``, ``description``,
            integer revision/source/ready/unavailable counters, ``created_at``
            and ``has_pending_sources=False``.

        Raises:
            ValueError: The title is absent or whitespace-only.
        """
        if not title or not title.strip():
            raise ValueError("Collection title is required")
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        "INSERT INTO collections(id,title,description) VALUES (:id,:title,:description) RETURNING *"
                    ),
                    {"id": _new_id(), "title": title.strip(), "description": description},
                )
                .mappings()
                .one()
            )
            conn.execute(
                text("SELECT ensure_collection_search_partition(CAST(:id AS uuid))"),
                {"id": row["id"]},
            )
            return {**_record(row), "has_pending_sources": False}

    def list_collections(self):
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(_COLLECTION_SELECT + " ORDER BY c.created_at,c.id")
                ).mappings()
            ]

    def get_collection(self, id):
        with self.engine.connect() as conn:
            return _record(
                conn.execute(text(_COLLECTION_SELECT + " WHERE c.id=:id"), {"id": id})
                .mappings()
                .first()
            )

    def source_file_keys_existing(self, keys):
        """Identify registered originals before orphan-file cleanup.

        Args:
            keys (list[str]): Candidate private-file keys.

        Returns:
            set[str]: Keys referenced by any source row, including deleted
            sources whose cleanup may still be pending. Empty input returns an
            empty set without querying; no files or database rows are changed.
        """
        if not keys:
            return set()
        with self.engine.connect() as conn:
            return set(
                conn.execute(
                    text("SELECT file_key FROM sources WHERE file_key=ANY(CAST(:keys AS text[]))"),
                    {"keys": keys},
                ).scalars()
            )

    def list_sources_page(self, collection_id, limit=100, after=None):
        """Read a newest-first library page, excluding deleted sources.

        Time-based page methods in this store use a strict ``(created_at, id)``
        cursor in descending order and fetch ``limit + 1`` records. The extra
        record is a has-more sentinel for the caller to trim before building
        the next cursor. Each call reads current state, not a frozen traversal
        snapshot; changing records can alter later pages.

        Args:
            collection_id (str): Collection UUID.
            limit (int): Requested page size; this method does not validate it.
            after (tuple[str, str] | None): Exclusive timestamp/UUID cursor;
                None starts with the newest source.

        Returns:
            list[dict]: At most ``limit + 1`` source rows, including ``id``,
            ``collection_id``, ``status``, immutable file metadata, timestamps,
            extraction/error and cleanup fields. An absent collection yields
            an empty list.
        """
        params = {"collection_id": collection_id, "limit": limit + 1}
        predicate = ""
        if after is not None:
            predicate = (
                " AND (created_at,id)<(CAST(:after_at AS timestamptz),CAST(:after_id AS uuid))"
            )
            params["after_at"], params["after_id"] = after
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT * FROM sources WHERE collection_id=:collection_id AND status<>'deleted'"
                        + predicate
                        + " ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    params,
                ).mappings()
            ]

    def get_source(self, id):
        with self.engine.connect() as conn:
            return _record(
                conn.execute(text("SELECT * FROM sources WHERE id=:id"), {"id": id})
                .mappings()
                .first()
            )

    def add_source(
        self,
        *,
        collection_id,
        title,
        filename,
        media_type,
        document_class="other",
        sha256,
        byte_count,
        file_key,
        request_id,
        source_id=None,
    ):
        """Register an uploaded original and queue ingestion atomically.

        Lock the collection, insert a pending source and one idempotent ingest
        job, then increase its revision, source and unavailable counters. This
        records file metadata only; the original must already be stored by the
        caller. Reusing a request ID with matching compared metadata returns
        the existing row, even if deleted, without creating another job. The
        comparison excludes ``file_key`` and ``source_id``.

        Args:
            collection_id (str): Existing collection UUID.
            title (str): Nonempty display title, stored verbatim.
            filename (str): Nonempty original filename.
            media_type (str): ``application/pdf`` or ``text/plain``.
            document_class (str): ``D1`` through ``D5``, or ``other``.
            sha256 (str): Exactly 64 lowercase hexadecimal characters.
            byte_count (int): Original size in bytes, greater than zero.
            file_key (str): Relative safe storage key, at most 201 characters,
                without a parent-directory component or leading slash.
            request_id (str): Nonempty idempotency key within this collection.
            source_id (str | None): Caller-provided UUID or a generated UUID.

        Returns:
            dict: New pending source or the previously registered source row,
            with file metadata, lifecycle and cleanup fields.

        Raises:
            ValueError: Hash, key or source metadata fails validation.
            KeyError: The collection does not exist.
            ConflictError: This request ID has different compared metadata.
        """
        if (
            not _HEX.fullmatch(sha256)
            or not _SAFE_KEY.fullmatch(file_key)
            or ".." in Path(file_key).parts
            or file_key.startswith("/")
        ):
            raise ValueError("Invalid source hash or file key")
        if not request_id or not filename or not title or byte_count < 1:
            raise ValueError("Incomplete source metadata")
        if media_type not in {"application/pdf", "text/plain"} or document_class not in {
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "other",
        }:
            raise ValueError("Unsupported source metadata")
        with self.engine.begin() as conn:
            collection = conn.execute(
                text("SELECT id FROM collections WHERE id=:id FOR UPDATE"), {"id": collection_id}
            ).first()
            if collection is None:
                raise KeyError(collection_id)
            existing = (
                conn.execute(
                    text(
                        "SELECT * FROM sources WHERE collection_id=:collection_id AND request_id=:request_id"
                    ),
                    {"collection_id": collection_id, "request_id": request_id},
                )
                .mappings()
                .first()
            )
            if existing is not None:
                if (
                    existing["sha256"] != sha256
                    or existing["byte_count"] != byte_count
                    or existing["media_type"] != media_type
                    or existing["document_class"] != document_class
                    or existing["filename"] != filename
                    or existing["title"] != title
                ):
                    raise ConflictError("Upload request ID reused with different content")
                return _record(existing)
            sid = source_id or _new_id()
            row = (
                conn.execute(
                    text(
                        "INSERT INTO sources(id,collection_id,request_id,title,filename,media_type,document_class,status,sha256,byte_count,file_key) VALUES (:id,:collection_id,:request_id,:title,:filename,:media_type,:document_class,'pending',:sha256,:byte_count,:file_key) RETURNING *"
                    ),
                    {
                        "id": sid,
                        "collection_id": collection_id,
                        "request_id": request_id,
                        "title": title,
                        "filename": filename,
                        "media_type": media_type,
                        "document_class": document_class,
                        "sha256": sha256,
                        "byte_count": byte_count,
                        "file_key": file_key,
                    },
                )
                .mappings()
                .one()
            )
            conn.execute(
                text(
                    "UPDATE collections SET revision=revision+1,source_count=source_count+1,unavailable_count=unavailable_count+1 WHERE id=:id"
                ),
                {"id": collection_id},
            )
            conn.execute(
                text(
                    "INSERT INTO jobs(id,kind,payload,idempotency_key,status) VALUES (:id,'source_ingest',CAST(:payload AS jsonb),:key,'ready')"
                ),
                {
                    "id": _new_id(),
                    "payload": _json({"source_id": sid}),
                    "key": f"source_ingest:{sid}",
                },
            )
            return _record(row)

    def delete_source(self, id):
        """Remove a source from retrieval and schedule private-file cleanup.

        Atomically mark the source deleted, remove its spans/chunks, adjust
        collection counts and revision, and enqueue a uniquely keyed cleanup
        job. The row and original file remain; the worker removes the file
        later. A repeated delete does not decrement counters again, but can
        requeue a ready/failed cleanup job and reset incomplete cleanup to
        ``queued``. A concurrent deletion found after locking returns its row.

        Args:
            id (str): Source UUID.

        Returns:
            dict: Retained source row with deletion and cleanup state. Return
            does not imply that filesystem cleanup has completed.

        Raises:
            KeyError: The source does not exist.
        """
        with self.engine.begin() as conn:
            current_status = conn.execute(
                text("SELECT status FROM sources WHERE id=:id"), {"id": id}
            ).scalar_one_or_none()
            if current_status is None:
                raise KeyError(id)
            if current_status == "deleted":
                conn.execute(
                    text(
                        "UPDATE jobs SET status='ready',available_at=clock_timestamp(),"
                        "worker_id=NULL,lease_until=NULL,finished_at=NULL "
                        "WHERE idempotency_key=:key AND kind='source_delete' "
                        "AND status IN ('ready','failed')"
                    ),
                    {"key": f"source_delete:{id}"},
                )
                conn.execute(
                    text(
                        "UPDATE sources SET cleanup_status='queued',cleanup_error=NULL WHERE id=:id "
                        "AND status='deleted' AND cleanup_status IS DISTINCT FROM 'complete'"
                    ),
                    {"id": id},
                )
                return self._deleted_source_with_cleanup(conn, id)
            row = (
                conn.execute(text("SELECT * FROM sources WHERE id=:id FOR UPDATE"), {"id": id})
                .mappings()
                .first()
            )
            if row["status"] == "deleted":
                return _record(row)
            result = (
                conn.execute(
                    text(
                        "UPDATE sources SET status='deleted',deleted_at=now(),error=NULL,"
                        "cleanup_status='queued',cleanup_error=NULL WHERE id=:id RETURNING *"
                    ),
                    {"id": id},
                )
                .mappings()
                .one()
            )
            conn.execute(
                text("DELETE FROM chunks WHERE collection_id=:collection_id AND source_id=:id"),
                {"collection_id": row["collection_id"], "id": id},
            )
            conn.execute(
                text("DELETE FROM spans WHERE collection_id=:collection_id AND source_id=:id"),
                {"collection_id": row["collection_id"], "id": id},
            )
            conn.execute(
                text(
                    "UPDATE collections SET revision=revision+1,source_count=source_count-1,ready_count=ready_count-:was_ready,unavailable_count=unavailable_count-:was_unavailable WHERE id=:id"
                ),
                {
                    "id": row["collection_id"],
                    "was_ready": int(row["status"] == "ready"),
                    "was_unavailable": int(row["status"] != "ready"),
                },
            )
            conn.execute(
                text(
                    "INSERT INTO jobs(id,kind,payload,idempotency_key,status) VALUES (:job_id,'source_delete',CAST(:payload AS jsonb),:key,'ready') ON CONFLICT(idempotency_key) DO NOTHING"
                ),
                {
                    "job_id": _new_id(),
                    "payload": _json({"source_id": id}),
                    "key": f"source_delete:{id}",
                },
            )
            return _record(result)

    def _deleted_source_with_cleanup(self, conn, id):
        return _record(
            conn.execute(text("SELECT * FROM sources WHERE id=:id"), {"id": id}).mappings().one()
        )

    def source_spans(self, source_id):
        """Read all published spans in original extraction order.

        Args:
            source_id (str): Source UUID.

        Returns:
            list[dict]: Span rows ordered by ascending ordinal, with ``id``,
            source/collection IDs, text, 1-based line bounds, nullable PDF page
            and geometry, source hash and section. Missing, non-ready or
            deleted sources yield no spans.
        """
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT p.* FROM spans p JOIN sources s ON s.id=p.source_id WHERE p.source_id=:id AND s.status='ready' AND s.deleted_at IS NULL ORDER BY p.ordinal"
                    ),
                    {"id": source_id},
                ).mappings()
            ]

    def source_spans_page(self, source_id, limit=100, after_id=None, anchor_id=None):
        """Page original evidence, optionally starting at a cited span.

        Unlike time-based history pages, this uses the source's ascending
        extraction ordinal. An anchor includes its span; an after cursor
        excludes it. The extra fetched record is the caller's has-more sentinel.

        Args:
            source_id (str): Source UUID.
            limit (int): Requested span count; not validated here.
            after_id (str | None): Exclusive span UUID in this source.
            anchor_id (str | None): Inclusive span UUID in this source.

        Returns:
            list[dict]: At most ``limit + 1`` span rows in extraction order,
            with the shape described by ``source_spans``. Only ready,
            non-deleted sources are exposed; otherwise the result is empty
            unless resolving a supplied cursor fails first.

        Raises:
            ValueError: Both cursors are supplied, or the cursor span does not
                exist in this source (``invalid_cursor``).
        """
        if after_id is not None and anchor_id is not None:
            raise ValueError("invalid_cursor")
        params = {"source_id": source_id, "limit": limit + 1}
        predicate = ""
        with self.engine.connect() as conn:
            position_id = after_id if after_id is not None else anchor_id
            if position_id is not None:
                position = (
                    conn.execute(
                        text(
                            "SELECT ordinal "
                            "FROM spans WHERE source_id=:source_id AND id=:position_id"
                        ),
                        {"source_id": source_id, "position_id": position_id},
                    )
                    .mappings()
                    .first()
                )
                if position is None:
                    raise ValueError("invalid_cursor")
                params["position_ordinal"] = position["ordinal"]
                operator = ">" if after_id is not None else ">="
                predicate = " AND p.ordinal" + operator + ":position_ordinal"
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT p.* FROM spans p JOIN sources s ON s.id=p.source_id "
                        "WHERE p.source_id=:source_id AND s.status='ready' AND s.deleted_at IS NULL"
                        + predicate
                        + " ORDER BY p.ordinal LIMIT :limit"
                    ),
                    params,
                ).mappings()
            ]

    def get_spans(self, ids):
        """Resolve citations only against currently published originals.

        Args:
            ids (list[str]): Span UUID strings in requested output order.

        Returns:
            list[dict]: Existing ready/non-deleted-source spans in input order,
            including duplicates when repeated in ``ids``. Missing or hidden
            spans are omitted; an empty input returns an empty list.
        """
        if not ids:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT p.* FROM spans p JOIN sources s ON s.id=p.source_id WHERE p.id=ANY(CAST(:ids AS uuid[])) AND s.status='ready' AND s.deleted_at IS NULL"
                ),
                {"ids": ids},
            ).mappings()
            by_id = {str(row["id"]): _record(row) for row in rows}
            return [by_id[id] for id in ids if id in by_id]

    def publish_source(
        self, source_id, spans, chunks, page_count, *, job_id=None, worker_id=None, attempt=None
    ):
        """Publish extracted evidence and searchable indexes in one transaction.

        Optionally lock and verify the supplied job lease's owner, attempt,
        running state and expiry, then lock the source. The caller must supply
        the ingest lease for this source. Insert ordered spans and embedded
        chunks, mark the
        processing source ready and update collection revision/availability
        counters together. Any failure rolls back every batch. Already-ready
        sources are a no-op after lease verification; their supplied extraction
        is not validated or compared with the stored extraction.

        Args:
            source_id (str): Processing source UUID.
            spans (list[dict]): Nonempty ordered extraction. Each item requires
                ``id`` (UUID string), ``line_start``/``line_end`` (1-based ints)
                and ``text`` (str); optional ``page`` (1-based int or None),
                ``bbox`` (JSON coordinates), ``page_width``/``page_height``
                (PDF points or None), and ``section`` (str). Stored ordinal is
                assigned from list order; source hash comes from the source.
            chunks (list[dict]): Nonempty items requiring ``id`` (UUID string),
                ``span_ids`` (list[str] referencing supplied spans), ``text``
                (str), ``variant`` (``fixed`` or ``structural``) and
                ``embedding`` (384 float-convertible values); optional
                ``metadata`` is a JSON-serializable dict.
            page_count (int): Positive extracted page count (one for TXT).
            job_id (str | None): Job UUID; None bypasses lease validation.
            worker_id (str | None): Owner identity when ``job_id`` is supplied.
            attempt (int | None): Claimed attempt when ``job_id`` is supplied.

        Returns:
            None: Publication committed, or the source was already ready.

        Raises:
            KeyError: The source does not exist or a required item key is absent.
            ValueError: Lease is stale/expired, source is not processing,
                extraction is empty, span IDs repeat, chunks reference unknown
                spans, or an embedding has the wrong length or invalid values.
        """
        with self.engine.begin() as conn:
            if job_id is not None:
                lease = conn.execute(
                    text(
                        "SELECT 1 FROM jobs WHERE id=:job_id AND status='running' AND worker_id=:worker_id AND attempt=:attempt FOR UPDATE"
                    ),
                    {"job_id": job_id, "worker_id": worker_id, "attempt": attempt},
                ).first()
                if lease is None:
                    raise ValueError("Ingestion lease is no longer current")
                if not conn.execute(
                    text("SELECT lease_until>clock_timestamp() FROM jobs WHERE id=:job_id"),
                    {"job_id": job_id},
                ).scalar_one():
                    raise ValueError("Ingestion lease is no longer current")
            source = (
                conn.execute(
                    text("SELECT * FROM sources WHERE id=:id FOR UPDATE"), {"id": source_id}
                )
                .mappings()
                .first()
            )
            if source is None:
                raise KeyError(source_id)
            if source["status"] == "ready":
                return
            if source["status"] != "processing":
                raise ValueError("Source is not processing")
            if not spans or not chunks or page_count < 1:
                raise ValueError("Cannot publish empty extraction")
            span_ids = {str(span["id"]) for span in spans}
            if len(span_ids) != len(spans):
                raise ValueError("Duplicate span IDs")
            if any(not set(map(str, chunk["span_ids"])).issubset(span_ids) for chunk in chunks):
                raise ValueError("Chunk references unknown spans")
            source_label = source["title"][:512] + " " + source["filename"][:512]
            span_insert = text(
                "INSERT INTO spans(id,source_id,collection_id,ordinal,page,line_start,line_end,"
                "text,bbox,page_width,page_height,sha256,section) "
                "SELECT x.id,:source_id,:collection_id,x.ordinal,x.page,x.line_start,"
                "x.line_end,x.text,x.bbox,x.page_width,x.page_height,:sha256,x.section "
                "FROM jsonb_to_recordset(CAST(:rows AS jsonb)) AS x("
                "id uuid,ordinal integer,page integer,line_start integer,line_end integer,"
                "text text,bbox jsonb,page_width double precision,page_height double precision,"
                "section text)"
            )
            for start in range(0, len(spans), 250):
                batch = spans[start : start + 250]
                conn.execute(
                    span_insert,
                    {
                        "source_id": source_id,
                        "collection_id": source["collection_id"],
                        "sha256": source["sha256"],
                        "rows": _json(
                            [
                                {
                                    "id": span["id"],
                                    "ordinal": start + offset + 1,
                                    "page": span.get("page"),
                                    "line_start": span["line_start"],
                                    "line_end": span["line_end"],
                                    "text": span["text"],
                                    "bbox": span.get("bbox"),
                                    "page_width": span.get("page_width"),
                                    "page_height": span.get("page_height"),
                                    "section": span.get("section", ""),
                                }
                                for offset, span in enumerate(batch)
                            ]
                        ),
                    },
                )
            chunk_insert = text(
                "INSERT INTO chunks(id,source_id,collection_id,span_ids,text,source_label,"
                "variant,embedding,metadata) SELECT x.id,:source_id,:collection_id,x.span_ids,"
                "x.text,:source_label,x.variant,CAST(x.embedding AS vector),x.metadata "
                "FROM jsonb_to_recordset(CAST(:rows AS jsonb)) AS x("
                "id uuid,span_ids jsonb,text text,variant text,embedding text,metadata jsonb)"
            )
            for start in range(0, len(chunks), 64):
                batch = chunks[start : start + 64]
                if any(len(chunk["embedding"]) != 384 for chunk in batch):
                    raise ValueError("Embedding must have 384 dimensions")
                conn.execute(
                    chunk_insert,
                    {
                        "source_id": source_id,
                        "collection_id": source["collection_id"],
                        "source_label": source_label,
                        "rows": _json(
                            [
                                {
                                    "id": chunk["id"],
                                    "span_ids": chunk["span_ids"],
                                    "text": chunk["text"],
                                    "variant": chunk["variant"],
                                    "embedding": "["
                                    + ",".join(str(float(x)) for x in chunk["embedding"])
                                    + "]",
                                    "metadata": chunk.get("metadata", {}),
                                }
                                for chunk in batch
                            ]
                        ),
                    },
                )
            conn.execute(
                text(
                    "UPDATE sources SET status='ready',page_count=:page_count,error=NULL WHERE id=:id"
                ),
                {"id": source_id, "page_count": page_count},
            )
            conn.execute(
                text(
                    "UPDATE collections SET revision=revision+1,ready_count=ready_count+1,unavailable_count=unavailable_count-1 WHERE id=:id"
                ),
                {"id": source["collection_id"]},
            )

    def cleanup_source(self, id):
        """Idempotently remove any remaining indexes of a deleted source.

        This database-only step follows file removal in the worker. It retains
        the source row and does not mark cleanup complete; ``finish_job`` does.

        Args:
            id (str): Deleted source UUID.

        Returns:
            None: Remaining chunks and spans have been removed.

        Raises:
            KeyError: The source does not exist.
            ValueError: The source is not deleted.
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT status,collection_id FROM sources WHERE id=:id FOR UPDATE"),
                {"id": id},
            ).first()
            if row is None:
                raise KeyError(id)
            if row[0] != "deleted":
                raise ValueError("Source is not deleted")
            conn.execute(
                text("DELETE FROM chunks WHERE collection_id=:collection_id AND source_id=:id"),
                {"collection_id": row[1], "id": id},
            )
            conn.execute(
                text("DELETE FROM spans WHERE collection_id=:collection_id AND source_id=:id"),
                {"collection_id": row[1], "id": id},
            )

    def claim_job(self, worker_id):
        """Claim one available job or recover an expired running attempt.

        Select by availability time, creation time and UUID with SKIP LOCKED
        so workers can claim different jobs. Atomically increment the attempt,
        assign the worker and a 15-minute lease; a pending ingest source also
        becomes processing. Reclaiming does not enforce an attempt cap here.

        Args:
            worker_id (str): Worker identity to record as lease owner.

        Returns:
            dict | None: Claimed job with ``id``, ``kind``, ``payload`` (for
            source jobs, ``source_id``), integer ``attempt``, ``worker_id``,
            lease/availability timestamps and status/error fields. None means
            no claimable unlocked job was found and no state changed.
        """
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        "WITH selected AS (SELECT id FROM jobs WHERE "
                        "(status='ready' AND available_at<=clock_timestamp()) OR "
                        "(status='running' AND lease_until<=clock_timestamp()) "
                        "ORDER BY available_at,created_at,id FOR UPDATE SKIP LOCKED LIMIT 1) "
                        "UPDATE jobs j SET status='running',worker_id=:worker_id,"
                        "lease_until=clock_timestamp()+interval '15 minutes',attempt=j.attempt+1 "
                        "FROM selected WHERE j.id=selected.id RETURNING j.*"
                    ),
                    {"worker_id": worker_id},
                )
                .mappings()
                .first()
            )
            if row and row["kind"] == "source_ingest":
                conn.execute(
                    text(
                        "UPDATE sources SET status='processing' WHERE id=CAST(:id AS uuid) AND status='pending'"
                    ),
                    {"id": row["payload"]["source_id"]},
                )
            return _record(row)

    def renew_job(self, id, worker_id, attempt):
        """Extend a matching, still-unexpired running lease by 15 minutes.

        Args:
            id (str): Job UUID.
            worker_id (str): Claimed owner identity.
            attempt (int): Claimed attempt number.

        Returns:
            bool: True after extending the lease; False without mutation when
            the job is absent, no longer running, owned by another attempt or
            worker, or already expired. Expired leases cannot be revived here.
        """
        with self.engine.begin() as conn:
            lease = conn.execute(
                text(
                    "SELECT 1 FROM jobs WHERE id=:id AND status='running' AND worker_id=:worker_id AND attempt=:attempt FOR UPDATE"
                ),
                {"id": id, "worker_id": worker_id, "attempt": attempt},
            ).first()
            if lease is None:
                return False
            row = conn.execute(
                text(
                    "UPDATE jobs SET lease_until=clock_timestamp()+interval '15 minutes' WHERE id=:id AND status='running' AND worker_id=:worker_id AND attempt=:attempt AND lease_until>clock_timestamp() RETURNING id"
                ),
                {"id": id, "worker_id": worker_id, "attempt": attempt},
            ).first()
            return row is not None

    def finish_job(
        self, id, error=None, *, worker_id=None, attempt=None, source_id=None, retryable=False
    ):
        """Finalize or reschedule a job only while its claimed lease is valid.

        Successful ingestion requires a ready or deleted source. An ingest
        error on either state is discarded as completed work; otherwise a
        retryable error before attempt 3 requeues with ``min(300, 10 *
        2**(attempt-1))`` seconds delay and stores the source error. Exhausted or
        permanent ingest errors mark the source failed and advance collection
        revision. Every deletion error requeues, independently of ``retryable``,
        with ``min(3600, 5 * 2**min(attempt, 10))`` seconds delay and retrying
        cleanup state; successful deletion marks cleanup complete. Job/source
        changes commit together and release worker/lease fields.

        Args:
            id (str): Job UUID.
            error (str | Exception | None): Failure description; None means
                success. Stringified errors are limited to 2000 characters,
                with an empty description replaced by ``unknown_error``.
            worker_id (str | None): Required claimed owner identity.
            attempt (int | None): Required claimed attempt number.
            source_id (str | None): Optional assertion of the ingest source ID.
            retryable (bool): Whether an ingest failure qualifies for retry.

        Returns:
            bool: True when finalization or rescheduling committed; False
            without mutation when lease identity/state does not match or the
            matching lease has expired.

        Raises:
            ValueError: Lease identity is missing, asserted source does not
                match an ingest job, or success is requested before the source
                is ready/deleted.
        """
        if worker_id is None or attempt is None:
            raise ValueError("Job lease identity is required")
        with self.engine.begin() as conn:
            job = (
                conn.execute(
                    text(
                        "SELECT kind,payload FROM jobs WHERE id=:id AND status='running' AND worker_id=:worker_id AND attempt=:attempt FOR UPDATE"
                    ),
                    {"id": id, "worker_id": worker_id, "attempt": attempt},
                )
                .mappings()
                .first()
            )
            if job is None:
                return False
            if not conn.execute(
                text("SELECT lease_until>clock_timestamp() FROM jobs WHERE id=:id"), {"id": id}
            ).scalar_one():
                return False
            if source_id is not None and (
                job["kind"] != "source_ingest" or job["payload"]["source_id"] != str(source_id)
            ):
                raise ValueError("Job source does not match")
            job_error = (str(error)[:2000] or "unknown_error") if error is not None else None
            status = "failed" if job_error else "done"
            available_at = None
            if job_error and job["kind"] == "source_ingest":
                source = (
                    conn.execute(
                        text(
                            "SELECT status,collection_id FROM sources WHERE id=CAST(:id AS uuid) FOR UPDATE"
                        ),
                        {"id": job["payload"]["source_id"]},
                    )
                    .mappings()
                    .one()
                )
                if source["status"] in {"ready", "deleted"}:
                    status = "done"
                    job_error = None
                elif retryable and attempt < 3:
                    status = "ready"
                    available_at = min(300, 10 * 2 ** (attempt - 1))
                    conn.execute(
                        text("UPDATE sources SET error=:error WHERE id=CAST(:id AS uuid)"),
                        {"id": job["payload"]["source_id"], "error": job_error},
                    )
                else:
                    conn.execute(
                        text(
                            "UPDATE sources SET status='failed',error=:error WHERE id=CAST(:id AS uuid)"
                        ),
                        {"id": job["payload"]["source_id"], "error": job_error},
                    )
                    conn.execute(
                        text("UPDATE collections SET revision=revision+1 WHERE id=:id"),
                        {"id": source["collection_id"]},
                    )
            elif job["kind"] == "source_ingest":
                source_status = conn.execute(
                    text("SELECT status FROM sources WHERE id=CAST(:source_id AS uuid)"),
                    {"source_id": job["payload"]["source_id"]},
                ).scalar_one()
                if source_status not in {"ready", "deleted"}:
                    raise ValueError("Ingestion source is not ready")
            elif job_error and job["kind"] == "source_delete":
                status = "ready"
                available_at = min(3600, 5 * 2 ** min(attempt, 10))
                conn.execute(
                    text(
                        "UPDATE sources SET cleanup_status='retrying',cleanup_error=:error "
                        "WHERE id=CAST(:source_id AS uuid) AND status='deleted'"
                    ),
                    {"source_id": job["payload"]["source_id"], "error": job_error},
                )
            elif job["kind"] == "source_delete":
                conn.execute(
                    text(
                        "UPDATE sources SET cleanup_status='complete',cleanup_error=NULL "
                        "WHERE id=CAST(:source_id AS uuid) AND status='deleted'"
                    ),
                    {"source_id": job["payload"]["source_id"]},
                )
            conn.execute(
                text(
                    "UPDATE jobs SET status=:status,error=:error,worker_id=NULL,lease_until=NULL,"
                    "available_at=clock_timestamp()+(CAST(:delay AS integer) * interval '1 second'),"
                    "finished_at=CASE WHEN :status IN ('done','failed') THEN clock_timestamp() "
                    "ELSE NULL END WHERE id=:id"
                ),
                {"id": id, "status": status, "error": job_error, "delay": available_at or 0},
            )
            return True

    def create_conversation(self, collection_id, title=""):
        """Create a dialogue bound to one existing research collection.

        Args:
            collection_id (str): Collection UUID.
            title (str): Display title, stored verbatim; may be empty.

        Returns:
            dict: Conversation with string IDs, title, integer revision and
            ISO-formatted creation timestamp. No messages or runs are created.

        Raises:
            KeyError: The collection does not exist.
        """
        with self.engine.begin() as conn:
            if (
                conn.execute(
                    text("SELECT 1 FROM collections WHERE id=:id"), {"id": collection_id}
                ).first()
                is None
            ):
                raise KeyError(collection_id)
            row = (
                conn.execute(
                    text(
                        "INSERT INTO conversations(id,collection_id,title) VALUES (:id,:collection_id,:title) RETURNING *"
                    ),
                    {"id": _new_id(), "collection_id": collection_id, "title": title},
                )
                .mappings()
                .one()
            )
            return _record(row)

    def get_conversation(self, id):
        with self.engine.connect() as conn:
            return _record(
                conn.execute(text("SELECT * FROM conversations WHERE id=:id"), {"id": id})
                .mappings()
                .first()
            )

    def list_conversations_page(self, collection_id=None, limit=50, after=None):
        """Browse dialogue summaries using the time-page contract of this store.

        Args:
            collection_id (str | None): Collection UUID filter; None includes
                conversations from all collections.
            limit (int): Page size from 1 through 200.
            after (tuple[str, str] | None): Exclusive creation timestamp/UUID
                cursor, as described by ``list_sources_page``.

        Returns:
            list[dict]: At most ``limit + 1`` summaries, newest first, containing
            ``id``, ``collection_id``, ``title``, ``revision`` and ``created_at``.

        Raises:
            ValueError: The page size is outside 1 through 200.
        """
        if not 1 <= limit <= 200:
            raise ValueError("Invalid page size")
        params = {"limit": limit + 1}
        conditions = []
        if collection_id is not None:
            conditions.append("collection_id=:collection_id")
            params["collection_id"] = collection_id
        if after is not None:
            conditions.append(
                "(created_at,id)<(CAST(:after_at AS timestamptz),CAST(:after_id AS uuid))"
            )
            params["after_at"], params["after_id"] = after
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT id,collection_id,title,revision,created_at FROM conversations"
                        + where
                        + " ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    params,
                ).mappings()
            ]

    def messages_page(self, conversation_id, limit=50, after=None):
        """Read newest-first dialogue messages for history pagination.

        Args:
            conversation_id (str): Conversation UUID.
            limit (int): Page size from 1 through 200.
            after (tuple[str, str] | None): Exclusive creation timestamp/UUID
                cursor using the ``list_sources_page`` time-page contract.

        Returns:
            list[dict]: At most ``limit + 1`` rows with ``id``, conversation/run
            IDs, ``role`` (user or assistant), ``content`` and ``created_at``.
            Missing conversations yield an empty list.

        Raises:
            ValueError: The page size is outside 1 through 200.
        """
        if not 1 <= limit <= 200:
            raise ValueError("Invalid page size")
        params = {"id": conversation_id, "limit": limit + 1}
        predicate = ""
        if after is not None:
            predicate = (
                " AND (created_at,id)<(CAST(:after_at AS timestamptz),CAST(:after_id AS uuid))"
            )
            params["after_at"], params["after_id"] = after
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT * FROM messages WHERE conversation_id=:id"
                        + predicate
                        + " ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    params,
                ).mappings()
            ]

    def conversation_runs_page(self, conversation_id, limit=50, after=None):
        """Read a dialogue's execution history independently of message history.

        Args:
            conversation_id (str): Conversation UUID.
            limit (int): Page size from 1 through 200.
            after (tuple[str, str] | None): Exclusive creation timestamp/UUID
                cursor using the ``list_sources_page`` time-page contract.

        Returns:
            list[dict]: At most ``limit + 1`` newest-first run rows, including
            identity/request fields, question/model/variant, status, scope,
            nullable answer/error/trace/retry fields, metrics and timestamps.
            Failed runs remain visible even without assistant messages.

        Raises:
            ValueError: The page size is outside 1 through 200.
        """
        if not 1 <= limit <= 200:
            raise ValueError("Invalid page size")
        params = {"id": conversation_id, "limit": limit + 1}
        predicate = ""
        if after is not None:
            predicate = (
                " AND (created_at,id)<(CAST(:after_at AS timestamptz),CAST(:after_id AS uuid))"
            )
            params["after_at"], params["after_id"] = after
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT * FROM runs WHERE conversation_id=:id"
                        + predicate
                        + " ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    params,
                ).mappings()
            ]

    def prior_messages(self, conversation_id, before_run_id, limit=5):
        """Build bounded chronological history before a run's user message.

        Select the newest earlier messages by creation timestamp and UUID,
        then reverse the retained window for prompt order. Anchoring on the
        original run supports retries without including later dialogue turns.

        Args:
            conversation_id (str): Conversation UUID containing the anchor.
            before_run_id (str): Run UUID whose user message is excluded.
            limit (int): Maximum number of earlier messages; not validated here.

        Returns:
            tuple[list[dict], bool]: Retained messages oldest first, with IDs,
            role, content and timestamp; the boolean indicates additional
            earlier messages beyond this window.

        Raises:
            ValueError: The anchor user message is missing from this dialogue
                (``retry_ancestor_message_missing``).
        """
        with self.engine.connect() as conn:
            anchor = (
                conn.execute(
                    text(
                        "SELECT created_at,id FROM messages "
                        "WHERE conversation_id=:conversation_id AND run_id=:run_id AND role='user'"
                    ),
                    {"conversation_id": conversation_id, "run_id": before_run_id},
                )
                .mappings()
                .first()
            )
            if anchor is None:
                raise ValueError("retry_ancestor_message_missing")
            rows = [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT * FROM messages WHERE conversation_id=:conversation_id "
                        "AND (created_at,id)<(CAST(:at AS timestamptz),CAST(:message_id AS uuid)) "
                        "ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    {
                        "conversation_id": conversation_id,
                        "at": anchor["created_at"],
                        "message_id": anchor["id"],
                        "limit": limit + 1,
                    },
                ).mappings()
            ]
            return list(reversed(rows[:limit])), len(rows) > limit

    def create_run(
        self, conversation_id, request_id, question, model, variant, retry_of_run_id=None
    ):
        """Queue an idempotent question and freeze its collection scope.

        Lock the dialogue, reject another active run, and snapshot collection
        identity, revision and source counters under a shared collection lock.
        Insert the queued run, user message, conversation revision increment
        and ``run.created`` event atomically. A matching request replay returns
        its existing run before checking active-run exclusion and performs no
        new writes. This method does not start inference or validate model
        availability, retry ancestry policy or collection readiness.

        Args:
            conversation_id (str): Existing conversation UUID.
            request_id (str): Nonempty idempotency key within this conversation.
            question (str): Nonblank text, compared and stored verbatim.
            model (str): Model identifier, stored without local validation.
            variant (str): Retrieval variant ``V0``, ``V1``, ``V2`` or ``V3``.
            retry_of_run_id (str | None): Optional referenced run UUID.

        Returns:
            dict: New queued run or matching existing run. Its ``scope`` dict
            contains ``id`` (run UUID), ``collection_id``, integer ``revision``,
            ``source_count``, ``ready_count``, ``unavailable_count`` and
            ``snapshot_hash``. Other fields follow ``conversation_runs_page``.

        Raises:
            ValueError: Request ID/question is empty or variant unsupported.
            KeyError: The conversation does not exist.
            ConflictError: Request ID is reused with different question, model,
                variant or retry ID, or the conversation has another active run.
        """
        if not request_id or not question.strip() or variant not in {"V0", "V1", "V2", "V3"}:
            raise ValueError("Invalid run request")
        with self.engine.begin() as conn:
            conversation = (
                conn.execute(
                    text("SELECT * FROM conversations WHERE id=:id FOR UPDATE"),
                    {"id": conversation_id},
                )
                .mappings()
                .first()
            )
            if conversation is None:
                raise KeyError(conversation_id)
            existing = (
                conn.execute(
                    text("SELECT * FROM runs WHERE conversation_id=:id AND request_id=:request_id"),
                    {"id": conversation_id, "request_id": request_id},
                )
                .mappings()
                .first()
            )
            if existing is not None:
                if (
                    existing["question"] != question
                    or existing["model"] != model
                    or existing["variant"] != variant
                    or str(existing["retry_of_run_id"] or "") != str(retry_of_run_id or "")
                ):
                    raise ConflictError("Run request ID reused with different content")
                return _record(existing)
            if conn.execute(
                text(
                    "SELECT 1 FROM runs WHERE conversation_id=:id AND status IN ('queued','running')"
                ),
                {"id": conversation_id},
            ).first():
                raise ConflictError("Conversation already has an active run")
            collection = (
                conn.execute(
                    text("SELECT * FROM collections WHERE id=:id FOR SHARE"),
                    {"id": conversation["collection_id"]},
                )
                .mappings()
                .one()
            )
            run_id = _new_id()
            scope = build_run_scope(collection, run_id)
            row = (
                conn.execute(
                    text(
                        "INSERT INTO runs(id,conversation_id,request_id,question,model,variant,status,scope,retry_of_run_id) VALUES (:id,:conversation_id,:request_id,:question,:model,:variant,'queued',CAST(:scope AS jsonb),:retry_of_run_id) RETURNING *"
                    ),
                    {
                        "id": run_id,
                        "conversation_id": conversation_id,
                        "request_id": request_id,
                        "question": question,
                        "model": model,
                        "variant": variant,
                        "scope": _json(scope),
                        "retry_of_run_id": retry_of_run_id,
                    },
                )
                .mappings()
                .one()
            )
            conn.execute(
                text(
                    "INSERT INTO messages(id,conversation_id,run_id,role,content) VALUES (:id,:conversation_id,:run_id,'user',:content)"
                ),
                {
                    "id": _new_id(),
                    "conversation_id": conversation_id,
                    "run_id": run_id,
                    "content": question,
                },
            )
            conn.execute(
                text("UPDATE conversations SET revision=revision+1 WHERE id=:id"),
                {"id": conversation_id},
            )
            _event(conn, run_id, "run.created", {"scope": scope})
            return _record(row)

    def get_run(self, id):
        with self.engine.connect() as conn:
            return _record(
                conn.execute(text("SELECT * FROM runs WHERE id=:id"), {"id": id}).mappings().first()
            )

    def get_run_by_request(self, conversation_id, request_id):
        with self.engine.connect() as conn:
            return _record(
                conn.execute(
                    text(
                        "SELECT * FROM runs WHERE conversation_id=:conversation_id AND request_id=:request_id"
                    ),
                    {"conversation_id": conversation_id, "request_id": request_id},
                )
                .mappings()
                .first()
            )

    def list_runs(self, limit=100, status="all", after=None):
        """Browse saved execution history with an optional exact status filter.

        Args:
            limit (int): Requested page size; not validated here.
            status (str): ``all`` disables filtering; otherwise matched exactly
                against stored execution status, without local validation.
            after (tuple[str, str] | None): Exclusive creation timestamp/UUID
                cursor using the ``list_sources_page`` time-page contract.

        Returns:
            list[dict]: At most ``limit + 1`` newest-first run rows with the
            identity, status, scope, result and trace fields described by
            ``conversation_runs_page``; an unmatched status yields no rows.
        """
        conditions = []
        params = {"limit": limit + 1}
        if status != "all":
            conditions.append("status=:status")
            params["status"] = status
        if after is not None:
            conditions.append(
                "(created_at,id)<(CAST(:after_at AS timestamptz),CAST(:after_id AS uuid))"
            )
            params["after_at"], params["after_id"] = after
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text(
                        "SELECT * FROM runs"
                        + where
                        + " ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    params,
                ).mappings()
            ]

    def append_event(self, run_id, type, payload):
        """Commit one durable event for later SSE replay.

        Args:
            run_id (str): Existing run UUID, including terminal runs.
            type (str): Event name; no event-name validation is applied here.
            payload (dict): JSON-serializable event-specific fields.

        Returns:
            dict: Committed event with run ID, monotonically allocated integer
            sequence, type, timestamp and payload. Repeated calls append new
            events; no idempotency key is used.

        Raises:
            KeyError: The run does not exist.
        """
        with self.engine.begin() as conn:
            return _event(conn, run_id, type, payload)

    def events(self, run_id, after=0):
        """Replay committed run events after an SSE resume sequence.

        Args:
            run_id (str): Run UUID.
            after (int): Exclusive sequence; zero starts before the first event.

        Returns:
            list[dict]: All matching events in ascending sequence order, each
            with run ID, sequence, type, timestamp and payload. A missing run
            or no newer events produces an empty list, without mutation.
        """
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text("SELECT * FROM run_events WHERE run_id=:id AND seq>:after ORDER BY seq"),
                    {"id": run_id, "after": after},
                ).mappings()
            ]

    def start_run(self, id):
        """Claim execution by atomically changing a queued run to running.

        Args:
            id (str): Run UUID.

        Returns:
            bool: True only for the caller that changes queued to running;
            False without mutation for a missing or differently staged run.
            This transition does not append an event or revalidate its scope.
        """
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "UPDATE runs SET status='running' WHERE id=:id AND status='queued' RETURNING id"
                ),
                {"id": id},
            ).first()
            return row is not None

    def run_scope_state(self, id):
        """Check whether a running answer can still use its frozen collection.

        Args:
            id (str): Run UUID.

        Returns:
            str: ``running`` when status and exact stored scope match the
            current collection; ``scope_expired`` for a running run whose
            scope is invalid or changed; ``stopped`` for a missing run/joined
            record or a non-running status. Shared locks protect this check
            only during its transaction; no state is changed or reserved.
        """
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT r.id AS run_id,r.status,r.scope,c.collection_id AS conversation_collection_id,"
                        "cl.id,cl.revision,cl.source_count,cl.ready_count,cl.unavailable_count "
                        "FROM runs r JOIN conversations c ON c.id=r.conversation_id "
                        "JOIN collections cl ON cl.id=c.collection_id "
                        "WHERE r.id=:id FOR SHARE OF r,cl"
                    ),
                    {"id": id},
                )
                .mappings()
                .first()
            )
            if row is None or row["status"] != "running":
                return "stopped"
            if scope_matches(row["scope"], row, row["run_id"], row["conversation_collection_id"]):
                return "running"
            return "scope_expired"

    def finish_run(self, id, answer, metrics, trace_id=None, trace_url=None):
        """Publish an answer only for a running run with an unchanged scope.

        Lock the run and current collection, then compare the frozen scope.
        On a match, atomically store answer/metrics/trace fields, succeed the
        run, add the assistant message, advance dialogue revision and append
        ``answer.ready``. On a mismatch, atomically fail the run with
        ``error={"code": "scope_expired"}``, finish its timestamp and append
        ``run.failed``; no supplied answer, metrics or trace fields are saved.
        This storage boundary does not validate answer content or citations.

        Args:
            id (str): Run UUID.
            answer (dict | JSON-serializable value): Validated graph output
                normally contains ``answer`` (str), ``status`` (supported,
                partial, conflicting, not_documented or needs_clarification),
                ``claims`` (dicts with ``text`` str and ``evidence_ids``
                list[str]), ``limitations`` (list[str]), ``citations`` (span
                rows) and ``coverage`` (receipt dict).
                A dict's ``answer`` defaults to empty assistant text; other
                serializable values use their string representation as text.
            metrics (dict): JSON-serializable graph diagnostics, including
                latency values in milliseconds and evidence/usage fields.
            trace_id (str | None): Optional external trace identifier.
            trace_url (str | None): Optional trace link.

        Returns:
            bool: True only when the answer was published. False without writes
            for a missing/non-running run; False WITH the failed-state and event
            writes described above when a running run's scope has expired.
            Repeated publication after either terminal outcome is a no-op.
        """
        with self.engine.begin() as conn:
            run = (
                conn.execute(
                    text(
                        "SELECT r.*,c.collection_id AS conversation_collection_id "
                        "FROM runs r JOIN conversations c ON c.id=r.conversation_id "
                        "WHERE r.id=:id FOR UPDATE OF r"
                    ),
                    {"id": id},
                )
                .mappings()
                .first()
            )
            if run is None or run["status"] != "running":
                return False
            scope = run["scope"]
            collection = (
                conn.execute(
                    text("SELECT * FROM collections WHERE id=:id FOR SHARE"),
                    {"id": run["conversation_collection_id"]},
                )
                .mappings()
                .first()
            )
            if not scope_matches(scope, collection, run["id"], run["conversation_collection_id"]):
                conn.execute(
                    text(
                        "UPDATE runs SET status='failed',error=CAST(:error AS jsonb),finished_at=now() WHERE id=:id"
                    ),
                    {"id": id, "error": _json({"code": "scope_expired"})},
                )
                _event(conn, id, "run.failed", {"code": "scope_expired"})
                return False
            conn.execute(
                text(
                    "UPDATE runs SET status='succeeded',answer=CAST(:answer AS jsonb),metrics=CAST(:metrics AS jsonb),trace_id=:trace_id,trace_url=:trace_url,finished_at=now() WHERE id=:id"
                ),
                {
                    "id": id,
                    "answer": _json(answer),
                    "metrics": _json(metrics),
                    "trace_id": trace_id,
                    "trace_url": trace_url,
                },
            )
            conn.execute(
                text(
                    "INSERT INTO messages(id,conversation_id,run_id,role,content) VALUES (:message_id,:conversation_id,:run_id,'assistant',:content)"
                ),
                {
                    "message_id": _new_id(),
                    "conversation_id": run["conversation_id"],
                    "run_id": id,
                    "content": answer.get("answer", "")
                    if isinstance(answer, dict)
                    else str(answer),
                },
            )
            conn.execute(
                text("UPDATE conversations SET revision=revision+1 WHERE id=:id"),
                {"id": run["conversation_id"]},
            )
            _event(conn, id, "answer.ready", {"answer": answer})
            return True

    def fail_run(self, id, error, status="failed"):
        """Terminate an active run and record its corresponding event atomically.

        Args:
            id (str): Run UUID.
            error (dict): JSON-serializable failure payload, normally containing
                ``code`` (str) and optional diagnostic fields; also the event
                payload. Its shape is not validated here.
            status (str): ``failed``, ``cancelled`` or ``interrupted``.

        Returns:
            bool: True after changing a queued/running run and appending its
            terminal event; False without mutation for missing/terminal runs.
            No assistant message is added and dialogue revision is unchanged.

        Raises:
            ValueError: The requested terminal status is unsupported.
        """
        if status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("Invalid terminal status")
        event_type = {
            "failed": "run.failed",
            "cancelled": "run.cancelled",
            "interrupted": "run.interrupted",
        }[status]
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "UPDATE runs SET status=:status,error=CAST(:error AS jsonb),finished_at=now() WHERE id=:id AND status IN ('queued','running') RETURNING id"
                ),
                {"id": id, "status": status, "error": _json(error)},
            ).first()
            if row:
                _event(conn, id, event_type, error)
            return row is not None

    def cancel_run(self, id):
        """Persist cancellation so later answer publication is refused.

        Args:
            id (str): Run UUID.

        Returns:
            bool: True after marking an active run cancelled with
            ``{"code": "cancelled"}`` and its event; False without mutation
            for missing/terminal runs. This does not stop a provider process
            or retract evidence already sent to it.
        """
        return self.fail_run(id, {"code": "cancelled"}, "cancelled")

    def interrupt_active_runs(self):
        """Interrupt leftover active runs at startup using the API-owner session.

        Verify the retained session identity and atomically mark all queued or
        running runs interrupted with ``process_restart`` errors and sequenced
        events. Using that same session ties cleanup to database ownership;
        another API owner cannot acquire the advisory lock during this work.

        Returns:
            int: Number of interrupted runs; repeated calls return zero when
            no new active runs exist. No inference is automatically restarted.

        Raises:
            RuntimeError: Ownership is absent or the owner session PID changed.
        """
        with self._api_owner_lock:
            connection = self._api_owner
            if connection is None:
                raise RuntimeError("API ownership is required for startup interruption")
            with connection.transaction():
                if (
                    connection.execute("SELECT pg_backend_pid()").fetchone()[0]
                    != self._api_owner_pid
                ):
                    raise RuntimeError("API ownership session changed")
                rows = connection.execute(
                    "UPDATE runs SET status='interrupted',error=%s::jsonb,finished_at=now() WHERE status IN ('queued','running') RETURNING id",
                    (_json({"code": "process_restart"}),),
                ).fetchall()
                for row in rows:
                    sequence = connection.execute(
                        "SELECT COALESCE(MAX(seq),0)+1 FROM run_events WHERE run_id=%s", (row[0],)
                    ).fetchone()[0]
                    connection.execute(
                        "INSERT INTO run_events(run_id,seq,type,payload) VALUES (%s,%s,'run.interrupted',%s::jsonb)",
                        (row[0], sequence, _json({"code": "process_restart"})),
                    )
                return len(rows)

    def interrupt_runs(self, ids):
        """Persist process-restart interruption for a selected set of active runs.

        Args:
            ids (list[str]): Run UUID strings; duplicates do not repeat updates.

        Returns:
            int: Number of queued/running runs changed to interrupted, with
            ``process_restart`` errors and events committed together. Missing
            and terminal runs are ignored; empty input performs no writes.
            Unlike startup cleanup, this does not check API ownership.
        """
        if not ids:
            return 0
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    "UPDATE runs SET status='interrupted',error=CAST(:error AS jsonb),finished_at=now() WHERE id = ANY(CAST(:ids AS uuid[])) AND status IN ('queued','running') RETURNING id"
                ),
                {"ids": ids, "error": _json({"code": "process_restart"})},
            ).all()
            for row in rows:
                _event(conn, str(row[0]), "run.interrupted", {"code": "process_restart"})
            return len(rows)

    def save_audit(self, run_id, stage, payload):
        """Append durable stage diagnostics independently of SSE publication.

        Args:
            run_id (str): Existing run UUID.
            stage (str): Stage name identifying the payload's contract.
            payload (dict): JSON-serializable stage data, such as evidence
                receipts, generation usage or validation answer/metrics.

        Returns:
            None: A new audit row has committed. Calls are append-only and
            have no deduplication; no run state or event is changed.
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO run_audit(id,run_id,stage,payload) VALUES (:id,:run_id,:stage,CAST(:payload AS jsonb))"
                ),
                {"id": _new_id(), "run_id": run_id, "stage": stage, "payload": _json(payload)},
            )

    def audit(self, run_id):
        """Read the authoritative diagnostic history of a run.

        Args:
            run_id (str): Run UUID.

        Returns:
            list[dict]: Audit rows in ascending timestamp/UUID order, containing
            ``id``, ``run_id``, ``stage``, decoded ``payload`` and ``at``. UUID
            order breaks timestamp ties, not insertion order. A missing run
            yields an empty list.
        """
        with self.engine.connect() as conn:
            return [
                _record(row)
                for row in conn.execute(
                    text("SELECT * FROM run_audit WHERE run_id=:id ORDER BY at,id"), {"id": run_id}
                ).mappings()
            ]

    def update_trace(self, run_id, trace_id, trace_url, trace_status):
        """Save trace-delivery metadata separately from answer execution status.

        Args:
            run_id (str): Run UUID.
            trace_id (str | None): External trace identifier or cleared value.
            trace_url (str | None): External trace URL or cleared value.
            trace_status (str | None): Delivery status, stored without validation.

        Returns:
            None: Update attempted; a missing run is silently ignored. Fields
            are overwritten on each call; answer/status/events are untouched.
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE runs SET trace_id=:trace_id,trace_url=:trace_url,trace_status=:trace_status WHERE id=:id"
                ),
                {
                    "id": run_id,
                    "trace_id": trace_id,
                    "trace_url": trace_url,
                    "trace_status": trace_status,
                },
            )

    def schedule_trace_reconcile(self, run_id, status):
        """Advance the persisted backoff for another trace-delivery check.

        Schedule from database wall-clock time using ``min(3600, base *
        2**min(previous_attempts, 9))`` seconds, then increment attempts. Base
        is 60 seconds for incomplete delivery and 15 for pending/error. This
        method records scheduling only; it does not change ``trace_status``.

        Args:
            run_id (str): Run UUID.
            status (str): ``pending``, ``incomplete`` or ``error``.

        Returns:
            str | None: ISO timestamp of the new retry deadline, or None with
            no mutation when the run is missing. Repeated calls advance the
            counter and reschedule, so this is not idempotent.

        Raises:
            ValueError: The status is unsupported.
        """
        if status not in {"pending", "incomplete", "error"}:
            raise ValueError("Invalid trace reconcile status")
        base = 60 if status == "incomplete" else 15
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "UPDATE runs SET trace_reconcile_after=clock_timestamp()+"
                    "(LEAST(3600,:base*power(2,LEAST(trace_reconcile_attempts,9))) * "
                    "interval '1 second'),trace_reconcile_attempts=trace_reconcile_attempts+1 "
                    "WHERE id=:id RETURNING trace_reconcile_after"
                ),
                {"id": run_id, "base": base},
            ).first()
            return row[0].isoformat() if row else None

    def reset_trace_reconcile(self, run_id):
        """Clear reconciliation scheduling after delivery or a policy reset.

        Args:
            run_id (str): Run UUID.

        Returns:
            None: Retry deadline cleared and attempt count set to zero; a
            missing run is silently ignored. Repeated calls are idempotent
            and do not change trace status or run execution status.
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE runs SET trace_reconcile_after=NULL,trace_reconcile_attempts=0 "
                    "WHERE id=:id"
                ),
                {"id": run_id},
            )
