import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from medical_assistant.errors import ConflictError
from medical_assistant.graph import Dialogue
from medical_assistant.ingest import _source_path
from medical_assistant.mcp_client import MCPGateway, MCPToolFailure
from medical_assistant.openapi import DESCRIPTION, TAGS, configure_openapi, operation
from medical_assistant.providers import get_provider
from medical_assistant.providers.common import ProviderError
from medical_assistant.schemas import CollectionInput, ConversationInput, RunInput, SessionInput
from medical_assistant.security import (
    COOKIE_NAME,
    SESSION_SECONDS,
    cookie_value,
    csrf_token,
    origin_allowed,
    service_authenticated,
    session_payload,
)
from medical_assistant.settings import get_settings
from medical_assistant.storage import Store, _record

_UUID = re.compile(r"^[0-9a-fA-F-]{36}$")
_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_RUN_STATUSES = {"all", "queued", "running", *_TERMINAL}
_LOG = logging.getLogger(__name__)


class _BodyTooLarge(Exception):
    pass


class RequestBodyLimit:
    def __init__(self, app, max_upload_bytes):
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH", "DELETE"}:
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        maximum = (
            self.max_upload_bytes + 1024 * 1024
            if path.startswith("/api/collections/") and path.endswith("/sources")
            else 65536
        )
        received = 0
        too_large = False
        rejected = False
        rejection = JSONResponse(
            status_code=413,
            content={"detail": {"code": "body_too_large", "message": "Request body exceeds limit"}},
        )

        async def limited_receive():
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > maximum:
                    too_large = True
                    raise _BodyTooLarge
            return message

        async def limited_send(message):
            nonlocal rejected
            if too_large:
                if not rejected:
                    rejected = True
                    await rejection(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except _BodyTooLarge:
            pass
        if too_large and not rejected:
            await rejection(scope, receive, send)


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path.startswith("api/") or "." in Path(path).name:
                raise
            return await super().get_response("index.html", scope)
        if (
            response.status_code == 404
            and not path.startswith("api/")
            and "." not in Path(path).name
        ):
            return await super().get_response("index.html", scope)
        return response


def _problem(status, code, message):
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def _uuid(value):
    if not _UUID.fullmatch(value):
        _problem(404, "not_found", "Resource not found")
    try:
        return str(uuid.UUID(value))
    except ValueError:
        _problem(404, "not_found", "Resource not found")


def _experiment_key(value):
    parts = value.split("~")
    if (
        len(parts) != 3
        or parts[1] not in {"V0", "V3"}
        or parts[2] not in {"gpt-6-sol", "gpt-6-luna"}
    ):
        _problem(404, "not_found", "Experiment not found")
    return _uuid(parts[0]), parts[1], parts[2]


def _encode_page_cursor(row, scope):
    payload = json.dumps(
        {"created_at": row["created_at"], "id": row["id"], "scope": scope},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_page_cursor(cursor, scope):
    if not cursor:
        return None
    try:
        if len(cursor) > 512:
            raise ValueError
        encoded = cursor.encode("ascii")
        payload = base64.b64decode(
            encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
        if base64.urlsafe_b64encode(payload).rstrip(b"=") != encoded:
            raise ValueError
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {"created_at", "id", "scope"}:
            raise ValueError
        created_at = value["created_at"]
        moment = datetime.fromisoformat(created_at)
        if moment.tzinfo is None or moment.isoformat() != created_at:
            raise ValueError
        if not isinstance(value["id"], str):
            raise ValueError
        row_id = str(uuid.UUID(value["id"]))
        if row_id != value["id"] or value["scope"] != scope:
            raise ValueError
        return created_at, row_id
    except (ValueError, TypeError, UnicodeError, binascii.Error):
        _problem(400, "invalid_cursor", "Page cursor is invalid")


def _safe_filename(name):
    filename = name.replace("\\", "/").split("/")[-1].strip()
    if not filename or len(filename) > 255 or any(ord(char) < 32 for char in filename):
        _problem(400, "invalid_filename", "A simple filename is required")
    return filename


def _source_type(filename, declared, first):
    suffix = Path(filename).suffix.casefold()
    declared = declared.split(";", 1)[0].casefold()
    if (
        suffix == ".pdf"
        and first.startswith(b"%PDF-")
        and declared in {"application/pdf", "application/octet-stream"}
    ):
        return "application/pdf"
    if (
        suffix == ".txt"
        and declared in {"text/plain", "application/octet-stream"}
        and not first.startswith(b"%PDF-")
    ):
        return "text/plain"
    _problem(415, "unsupported_file", "Upload a native PDF or UTF-8 TXT file with matching type")


def _persist_upload(settings, source_file, filename, declared, file_key):
    final = _source_path(settings, file_key)
    directory_created = not final.parent.is_dir()
    final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    published = False
    size = 0
    digest = hashlib.sha256()
    first = b""
    try:
        handle, temporary_name = tempfile.mkstemp(prefix=".upload-", dir=final.parent)
        temporary = Path(temporary_name)
        with os.fdopen(handle, "wb") as output:
            while block := source_file.read(1024 * 1024):
                if not first:
                    first = block[:8]
                size += len(block)
                if size > settings.max_upload_bytes:
                    _problem(413, "file_too_large", "File exceeds upload limit")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if not size:
            _problem(400, "empty_file", "File is empty")
        media_type = _source_type(filename, declared, first)
        if media_type == "text/plain":
            try:
                content = temporary.read_bytes()
                if b"\x00" in content:
                    raise UnicodeError
                content.decode("utf-8-sig", errors="strict")
            except UnicodeError:
                _problem(415, "invalid_text", "TXT must be UTF-8 text")
        os.replace(temporary, final)
        temporary = None
        published = True
        directory_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if directory_created:
            parent_fd = os.open(final.parent.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        return final, size, digest.hexdigest(), media_type
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if published:
            final.unlink(missing_ok=True)
        raise


def _experiment_items(conn, observed_at, campaigns):
    """Project validated campaigns into four read-only comparison cells each.

    Validate corpus/campaign membership and attempt projections before computing
    V0/V3 by Sol/Luna summaries. Primary attempt projections drive the frozen
    comparison; latest correction metrics remain separate. Current run trace
    states are attached to the projections for delivery diagnostics.

    Args:
        conn (sqlalchemy.Connection): Existing read transaction for campaign,
            corpus, attempt and run reads.
        observed_at (str): ISO timestamp identifying the caller's snapshot.
        campaigns (list[dict]): Campaign rows with id, corpus_row_id, split,
            release_id, status and created_at.

    Returns:
        list[dict]: Four cells per campaign, ordered V0/Sol, V0/Luna, V3/Sol,
        V3/Luna. Each includes a composite id, campaign/corpus identities,
        partition, model/variant, counts, primary metrics or None for an empty
        cell, latest_correction, created_at and observed_at. Empty campaigns
        produce an empty list. No evaluation or inference is started.

    Raises:
        ValueError: Stored corpus, campaign or attempt projections are invalid.
        KeyError: Required persisted campaign/projection fields are missing.
        TypeError: Stored fields cannot be processed as their expected shapes.
    """
    from medical_assistant.evaluation import (
        combine_costs,
        validate_campaign_corpus,
        validate_campaign_rows,
        validate_projection_row,
    )
    from medical_assistant.evaluation_metrics import aggregate

    if not campaigns:
        return []
    campaign_ids = [campaign["id"] for campaign in campaigns]
    corpus_ids = [campaign["corpus_row_id"] for campaign in campaigns]
    corpora = {
        str(row["id"]): _record(row)
        for row in conn.execute(
            text("SELECT * FROM evaluation_corpora WHERE id=ANY(CAST(:ids AS uuid[]))"),
            {"ids": corpus_ids},
        ).mappings()
    }
    attempts = [
        _record(row)
        for row in conn.execute(
            text(
                "SELECT id,campaign_id,case_id,variant,generator,state,metadata,metrics,attempts,first_run_id,target_run_id,judge_status,judge_output FROM evaluation_attempts WHERE campaign_id=ANY(CAST(:ids AS uuid[])) ORDER BY case_id"
            ),
            {"ids": campaign_ids},
        ).mappings()
    ]
    projections_by_id = {row["id"]: validate_projection_row(row) for row in attempts}
    run_ids = list(
        {
            run_id
            for projection in projections_by_id.values()
            for run_id in projection["execution"]["run_ids"]
        }
    )
    trace_statuses = (
        {
            str(row["id"]): row["trace_status"] or "missing"
            for row in conn.execute(
                text("SELECT id,trace_status FROM runs WHERE id=ANY(CAST(:ids AS uuid[]))"),
                {"ids": run_ids},
            ).mappings()
        }
        if run_ids
        else {}
    )
    groups = {}
    by_campaign = {}
    for attempt in attempts:
        by_campaign.setdefault(attempt["campaign_id"], []).append(attempt)
        groups.setdefault(
            (attempt["campaign_id"], attempt["variant"], attempt["generator"]), []
        ).append(attempt)
    items = []
    for campaign in campaigns:
        validate_campaign_corpus(campaign, corpora.get(campaign["corpus_row_id"]))
        validate_campaign_rows(campaign, by_campaign.get(campaign["id"], []))
        for variant in ("V0", "V3"):
            for model in ("gpt-6-sol", "gpt-6-luna"):
                rows = groups.get((campaign["id"], variant, model), [])
                projections = [projections_by_id[row["id"]] for row in rows]
                for projection in projections:
                    execution = projection["execution"]
                    execution["trace_statuses"] = [
                        trace_statuses.get(run_id, "missing") for run_id in execution["run_ids"]
                    ]
                summary = aggregate(projections)["summary"] if rows else None
                latest_correction = {
                    "attempted_rows": sum(
                        len(row["attempts"]) > 1
                        or any(item.get("judge_history") for item in row["attempts"])
                        for row in rows
                    ),
                    "completed": sum(row["state"] == "completed" for row in rows),
                    "state_counts": dict(Counter(row["state"] for row in rows)),
                    "confirmed_grounded_success": aggregate(
                        [row["metrics"]["projection"] for row in rows]
                    )["summary"]["confirmed_grounded_success"],
                }
                costs = [row["metrics"].get("cost_estimate") for row in rows]
                if summary is not None:
                    summary = {
                        **summary,
                        "api_equivalent_cost": combine_costs(costs),
                        "latest_correction": latest_correction,
                    }
                items.append(
                    {
                        "id": f"{campaign['id']}~{variant}~{model}",
                        "campaign_id": campaign["id"],
                        "corpus_id": corpora[campaign["corpus_row_id"]]["corpus_id"],
                        "partition": campaign["split"],
                        "name": f"{campaign['release_id']} · {variant} · {model}",
                        "status": campaign["status"],
                        "planned": len(rows),
                        "completed": sum(
                            projection["execution"]["state"] == "completed"
                            for projection in projections
                        ),
                        "model": model,
                        "variant": variant,
                        "metrics": summary,
                        "latest_correction": latest_correction,
                        "created_at": campaign["created_at"],
                        "observed_at": observed_at,
                    }
                )
    return items


def _experiments(store, limit, after):
    """Read one campaign page and its configuration summaries consistently.

    A repeatable-read, read-only transaction observes campaign rows and all
    dependent corpus/attempt/trace data at one database snapshot. Fetch one
    additional campaign to decide whether another page exists; the API encodes
    the returned last row as the campaign cursor.

    Args:
        store (Store): Database owner; this operation only reads its engine.
        limit (int): Number of campaigns, already checked as 1..25 by HTTP.
        after (tuple[str, str] | None): Exclusive (created_at ISO timestamp,
            campaign UUID) key, or None for the newest page.

    Returns:
        tuple[list[dict], dict | None]: Four experiment-cell records per selected
        campaign and the final selected campaign row when an older page exists,
        otherwise None. Cells within a campaign use the fixed model/variant
        order from _experiment_items; campaigns are newest first.

    Raises:
        ValueError: Stored evaluation data fails validation.
        KeyError: Required evaluation fields are missing.
        TypeError: Stored evaluation fields have unusable shapes. The HTTP
            boundary converts these three failures to a 409 error envelope.
    """
    with store.engine.connect() as conn:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        observed_at = conn.execute(text("SELECT transaction_timestamp()")).scalar_one().isoformat()
        rows = [
            _record(row)
            for row in conn.execute(
                text(
                    "SELECT * FROM evaluation_campaigns "
                    "WHERE (CAST(:after_at AS timestamptz) IS NULL OR (created_at,id)<(CAST(:after_at AS timestamptz),CAST(:after_id AS uuid))) "
                    "ORDER BY created_at DESC,id DESC LIMIT :limit"
                ),
                {
                    "after_at": after[0] if after else None,
                    "after_id": after[1] if after else None,
                    "limit": limit + 1,
                },
            ).mappings()
        ]
        campaigns = rows[:limit]
        items = _experiment_items(conn, observed_at, campaigns)
        return items, campaigns[-1] if len(rows) > limit else None


def _primary_judge_record(row):
    if not row["attempts"]:
        return {}
    first = row["attempts"][0]
    history = first.get("judge_history") or []
    return history[0] if history else first


def _experiment(store, experiment_id):
    """Read one saved comparison cell with case provenance and metric slices.

    Resolve the campaign/variant/model key, validate its stored evaluation
    data, then expose both frozen primary attempt and latest correction
    references for each case. Summary, cases and trace delivery are read in
    one repeatable-read transaction; metric breakdowns use those saved rows.

    Args:
        store (Store): Database owner; only read access is used.
        experiment_id (str): campaign UUID~V0-or-V3~gpt-6-sol-or-gpt-6-luna.

    Returns:
        dict | None: Cell summary plus cases ordered by case_id and breakdown
        mapping axis to group to planned/completion/grounding/evidence ratios.
        Ratios contain numerator, denominator and value (None when the
        denominator is zero). Required-document-class groups may overlap.
        None means the well-formed campaign key was not found. This operation
        does not start evaluations, retries, judging or inference.

    Raises:
        HTTPException: 404 for a malformed composite key or campaign UUID.
        ValueError: Persisted campaign/projection data is invalid.
        KeyError: Required persisted fields are missing.
        TypeError: Persisted fields have unusable shapes; the HTTP boundary
            normalizes the latter three failures to 409.
    """
    from medical_assistant.evaluation import validate_projection_row
    from medical_assistant.evaluation_metrics import aggregate

    campaign_id, variant, model = _experiment_key(experiment_id)
    with store.engine.connect() as conn:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        observed_at = conn.execute(text("SELECT transaction_timestamp()")).scalar_one().isoformat()
        campaign = _record(
            conn.execute(
                text("SELECT * FROM evaluation_campaigns WHERE id=:id"),
                {"id": campaign_id},
            )
            .mappings()
            .first()
        )
        item = (
            next(
                (
                    value
                    for value in _experiment_items(conn, observed_at, [campaign])
                    if value["id"] == experiment_id
                ),
                None,
            )
            if campaign is not None
            else None
        )
        if item is None:
            return None
        rows = (
            conn.execute(
                text(
                    "SELECT id,case_id,variant,generator,state,metadata,metrics,attempts,first_run_id,target_run_id,judge_status,judge_output FROM evaluation_attempts WHERE campaign_id=:campaign_id AND variant=:variant AND generator=:generator ORDER BY case_id"
                ),
                {
                    "campaign_id": item["campaign_id"],
                    "variant": item["variant"],
                    "generator": item["model"],
                },
            )
            .mappings()
            .all()
        )
        cases = [
            {
                "case_id": row["case_id"],
                "family_id": row["metadata"]["family_id"],
                "status": validate_projection_row(row)["execution"]["state"],
                "first_run_id": str(row["first_run_id"]) if row["first_run_id"] else None,
                "target_run_id": str(row["target_run_id"]) if row["target_run_id"] else None,
                "judge_status": row["judge_status"],
                "judge_verdict": (row["judge_output"] or {}).get("verdict"),
                "primary": {
                    "attempt_id": (row["metrics"].get("primary") or {}).get("attempt_id"),
                    "run_ids": (row["metrics"].get("primary") or {})
                    .get("projection", {})
                    .get("execution", {})
                    .get("run_ids", []),
                    "judge_request_id": (row["metrics"].get("primary") or {}).get(
                        "judge_request_id"
                    ),
                    "judge_verdict": (
                        (row["metrics"].get("primary") or {}).get("projection", {}).get("judge")
                        or {}
                    ).get("verdict"),
                    "judge_operation_id": _primary_judge_record(row).get("judge_operation_id"),
                    "judge_trace_id": _primary_judge_record(row).get("judge_trace_id"),
                    "judge_delivery_status": _primary_judge_record(row).get(
                        "judge_delivery_status"
                    ),
                },
                "latest": {
                    "attempt_id": row["attempts"][-1]["attempt_id"] if row["attempts"] else None,
                    "state": row["state"],
                    "judge_status": row["judge_status"],
                    "judge_verdict": (row["judge_output"] or {}).get("verdict"),
                    "retry_reason": row["attempts"][-1].get("retry_reason")
                    if row["attempts"]
                    else None,
                    "run_ids": row["metrics"]["projection"]["execution"]["run_ids"],
                    "judge_request_id": row["attempts"][-1].get("judge_request_id")
                    if row["attempts"]
                    else None,
                    "judge_operation_id": row["attempts"][-1].get("judge_operation_id")
                    if row["attempts"]
                    else None,
                    "judge_trace_id": row["attempts"][-1].get("judge_trace_id")
                    if row["attempts"]
                    else None,
                    "judge_delivery_status": row["attempts"][-1].get("judge_delivery_status")
                    if row["attempts"]
                    else None,
                },
                "metrics": {
                    key: value
                    for key, value in row["metrics"].items()
                    if key not in {"projection", "primary"}
                },
            }
            for row in rows
        ]
    from medical_assistant.evaluation import validate_projection_row

    projections = [validate_projection_row(row) for row in rows]
    fields = (
        "planned",
        "confirmed_grounded_success",
        "technical_completion",
        "judge_assessable",
        "evidence_case_availability",
        "evidence_unknown_positive_cases",
    )
    slices = aggregate(projections)["slices"] if projections else {}
    breakdown = {
        axis: {name: {key: values[key] for key in fields} for name, values in groups.items()}
        for axis, groups in slices.items()
    }
    return {**item, "cases": cases, "breakdown": breakdown}


def create_app(settings=None, store=None, provider=None, mcp=None, telemetry=None):
    settings = settings or get_settings()
    if not settings.launch_token:
        raise ValueError("PFL_LAUNCH_TOKEN is required")
    store = store or Store(settings)
    provider = provider or get_provider(settings)
    own_mcp = mcp is None
    mcp = mcp or MCPGateway(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pfl-db")
        app.state.tasks = {}
        app.state.run_admission = asyncio.Lock()
        app.state.store = store
        app.state.owner_alive = False
        mcp_started = False
        owner_watch = None
        reconcile_task = None
        try:
            if not await _db(app, store.acquire_api_ownership):
                raise RuntimeError("Another API process already owns this database")
            app.state.owner_alive = True
            await _db(app, store.migrate)
            if not await _api_owner_healthy(app):
                raise RuntimeError("API ownership lost during startup")
            await _db(app, store.interrupt_active_runs)
            if own_mcp:
                await mcp.__aenter__()
                mcp_started = True
            if telemetry is None:
                from medical_assistant.telemetry import Telemetry

                active_telemetry = Telemetry(settings, store)
            else:
                active_telemetry = telemetry
            app.state.provider = provider
            app.state.mcp = mcp
            app.state.telemetry = active_telemetry
            app.state.dialogue = Dialogue(
                store, mcp, provider, settings, active_telemetry, db_executor=app.state.pool
            )
            if not await _api_owner_healthy(app):
                raise RuntimeError("API ownership lost during startup")
            owner_watch = asyncio.create_task(_watch_api_owner(app))
            reconcile_task = asyncio.create_task(_reconcile_uploads(app, store, settings))
            yield
        finally:
            app.state.owner_alive = False
            if owner_watch is not None:
                owner_watch.cancel()
                await asyncio.gather(owner_watch, return_exceptions=True)
            if reconcile_task is not None:
                reconcile_task.cancel()
                await asyncio.gather(reconcile_task, return_exceptions=True)
            local_ids = list(app.state.tasks)
            local_tasks = tuple(app.state.tasks.values())
            for task in local_tasks:
                task.cancel()
            if local_tasks:
                await asyncio.gather(*local_tasks, return_exceptions=True)
            try:
                if local_ids:
                    await _db(app, store.interrupt_runs, local_ids)
            finally:
                try:
                    if mcp_started:
                        await mcp.__aexit__(None, None, None)
                finally:
                    try:
                        await _db(app, store.release_api_ownership)
                    finally:
                        app.state.pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        lifespan=lifespan,
        title="Medical Document Assistant · Evidence Lab",
        version="0.1.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
    )
    app.add_middleware(RequestBodyLimit, max_upload_bytes=settings.max_upload_bytes)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": "http_error", "message": str(exc.detail)}
        )
        return JSONResponse(
            status_code=exc.status_code, content={"detail": detail}, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {"code": "invalid_request", "message": "Request fields are invalid"}
            },
        )

    @app.exception_handler(ConflictError)
    async def conflict_error(request, exc):
        return JSONResponse(
            status_code=409,
            content={"detail": {"code": "conflict", "message": str(exc)[:300]}},
        )

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": str(exc)[:300],
                }
            },
        )

    @app.exception_handler(KeyError)
    async def key_error(request, exc):
        return JSONResponse(
            status_code=404,
            content={"detail": {"code": "not_found", "message": "Resource not found"}},
        )

    @app.middleware("http")
    async def guard(request, call_next):
        if not origin_allowed(request, settings.app_origin):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": {"code": "origin_denied", "message": "Host or origin is not allowed"}
                },
            )
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            path = request.url.path
            if path != "/api/session" and not app.state.owner_alive:
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "api_ownership_lost",
                            "message": "API process is unavailable",
                        }
                    },
                )
            length = request.headers.get("content-length")
            needs_length = path == "/api/session" or (
                path.startswith("/api/collections/")
                and path.endswith("/sources")
                and request.method == "POST"
            )
            if length is None and needs_length:
                return JSONResponse(
                    status_code=411,
                    content={
                        "detail": {
                            "code": "length_required",
                            "message": "Content-Length is required",
                        }
                    },
                )
            if length is not None and not length.isdecimal():
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": {"code": "invalid_length", "message": "Content-Length is invalid"}
                    },
                )
            max_body = (
                settings.max_upload_bytes + 1024 * 1024
                if path.startswith("/api/collections/") and path.endswith("/sources")
                else 65536
            )
            if length is not None and int(length) > max_body:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": {
                            "code": "body_too_large",
                            "message": "Request body exceeds limit",
                        }
                    },
                )
            if (
                path == "/api/session"
                and request.headers.get("content-type", "").split(";")[0] != "application/json"
            ):
                return JSONResponse(
                    status_code=415,
                    content={
                        "detail": {"code": "json_required", "message": "JSON body is required"}
                    },
                )
            if path != "/api/session" and path.startswith("/api/"):
                if request.headers.get("authorization"):
                    if not service_authenticated(request, settings.service_token):
                        return JSONResponse(
                            status_code=401,
                            content={
                                "detail": {
                                    "code": "unauthorized",
                                    "message": "Authentication required",
                                }
                            },
                        )
                else:
                    payload = session_payload(request, settings.launch_token)
                    if payload is None:
                        return JSONResponse(
                            status_code=401,
                            content={
                                "detail": {
                                    "code": "unauthorized",
                                    "message": "Authentication required",
                                }
                            },
                        )
                    supplied = request.headers.get("x-csrf-token", "")
                    if not hmac.compare_digest(
                        supplied, csrf_token(settings.launch_token, payload)
                    ):
                        return JSONResponse(
                            status_code=403,
                            content={
                                "detail": {
                                    "code": "csrf_denied",
                                    "message": "CSRF token is invalid",
                                }
                            },
                        )
            if (
                path.startswith("/api/")
                and path != "/api/session"
                and not await _api_owner_healthy(app)
            ):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "api_ownership_lost",
                            "message": "API process is unavailable",
                        }
                    },
                )
        return await call_next(request)

    def authorized(request: Request):
        if request.headers.get("authorization"):
            if service_authenticated(request, settings.service_token):
                return "service"
            _problem(401, "unauthorized", "Authentication required")
        payload = session_payload(request, settings.launch_token)
        if payload is None:
            _problem(401, "unauthorized", "Authentication required")
        return payload

    @app.get("/api/health", **operation("health"))
    async def health():
        if not await _api_owner_healthy(app):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "provider": settings.provider,
                    "model": settings.model,
                },
            )
        try:
            await _db(app, _ping, store)
        except Exception:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "provider": settings.provider,
                    "model": settings.model,
                    "langfuse_url": settings.langfuse_public_url,
                    "version": "0.1.0",
                },
            )
        return {
            "status": "ok",
            "provider": settings.provider,
            "model": settings.model,
            "langfuse_url": settings.langfuse_public_url,
            "version": "0.1.0",
        }

    @app.get("/api/readiness", **operation("provider_readiness"))
    async def provider_readiness():
        unavailable = {
            "provider": settings.provider,
            "ready": False,
            "connection": "unknown",
            "configuration": "unknown",
            "inference": "unverified",
        }
        if not await _api_owner_healthy(app):
            return JSONResponse(
                status_code=503, content={**unavailable, "reason": "api_ownership_lost"}
            )
        try:
            result = await app.state.provider.readiness()
        except Exception as exc:
            _LOG.error(
                "stage=provider_readiness error_type=%s traceback=%s",
                type(exc).__name__,
                "".join(traceback.format_tb(exc.__traceback__)),
            )
            return JSONResponse(status_code=503, content={**unavailable, "reason": "unavailable"})
        if not result.get("ready", False):
            return JSONResponse(status_code=503, content={**result, "ready": False})
        return result

    @app.get("/api/session", **operation("get_session"))
    async def get_session(request: Request):
        payload = session_payload(request, settings.launch_token)
        return {
            "authenticated": payload is not None,
            "csrf_token": csrf_token(settings.launch_token, payload) if payload else None,
        }

    @app.post("/api/session", **operation("start_session"))
    async def start_session(value: SessionInput, response: Response):
        if not hmac.compare_digest(value.token, settings.launch_token):
            _problem(401, "invalid_token", "Launch token is invalid")
        cookie = cookie_value(settings.launch_token)
        payload = ".".join(cookie.split(".")[:2])
        response.set_cookie(
            COOKIE_NAME,
            cookie,
            max_age=SESSION_SECONDS,
            httponly=True,
            secure=settings.app_origin.startswith("https://"),
            samesite="strict",
            path="/",
        )
        return {
            "authenticated": True,
            "csrf_token": csrf_token(settings.launch_token, payload),
        }

    @app.get("/api/collections", **operation("list_collections"))
    async def list_collections(auth=Depends(authorized)):
        return {"items": await _db(app, store.list_collections)}

    @app.post("/api/collections", **operation("create_collection"))
    async def create_collection(value: CollectionInput, auth=Depends(authorized)):
        return await _db(app, store.create_collection, value.title, value.description)

    @app.get("/api/collections/{collection_id}", **operation("get_collection"))
    async def get_collection(collection_id: str, auth=Depends(authorized)):
        collection = await _db(app, store.get_collection, _uuid(collection_id))
        if collection is None:
            _problem(404, "not_found", "Collection not found")
        return collection

    @app.get("/api/collections/{collection_id}/sources", **operation("list_sources"))
    async def list_sources(
        collection_id: str, limit: int = 100, cursor: str = "", auth=Depends(authorized)
    ):
        collection_id = _uuid(collection_id)
        if limit < 1 or limit > 100:
            _problem(400, "invalid_limit", "Source page size must be between 1 and 100")
        collection = await _db(app, store.get_collection, collection_id)
        if collection is None:
            _problem(404, "not_found", "Collection not found")
        scope = f"sources:{collection_id}"
        after = _decode_page_cursor(cursor, scope)
        rows = await _db(app, store.list_sources_page, collection_id, limit, after)
        items = rows[:limit]
        next_cursor = _encode_page_cursor(items[-1], scope) if len(rows) > limit else None
        return {"items": items, "next_cursor": next_cursor, "collection": collection}

    @app.post(
        "/api/collections/{collection_id}/sources", status_code=202, **operation("upload_source")
    )
    async def upload_source(
        collection_id: str,
        file: UploadFile = File(...),
        document_class: str = Form("other"),
        request_id: str = Form(...),
        auth=Depends(authorized),
    ):
        collection_id = _uuid(collection_id)
        if await _db(app, store.get_collection, collection_id) is None:
            _problem(404, "not_found", "Collection not found")
        if (
            document_class not in {"D1", "D2", "D3", "D4", "D5", "other"}
            or not 1 <= len(request_id) <= 100
        ):
            _problem(400, "invalid_metadata", "Document class or request ID is invalid")
        filename = _safe_filename(file.filename or "")
        source_id = str(uuid.uuid4())
        file_key = source_id
        final = None
        commit_task = None
        try:
            persist_task = asyncio.create_task(
                asyncio.to_thread(
                    _persist_upload,
                    settings,
                    file.file,
                    filename,
                    file.content_type or "application/octet-stream",
                    file_key,
                )
            )
            try:
                final, size, sha256, media_type = await asyncio.shield(persist_task)
            except asyncio.CancelledError:
                try:
                    final = (await asyncio.shield(persist_task))[0]
                except (asyncio.CancelledError, Exception):
                    pass
                raise
            commit_task = asyncio.create_task(
                _db(
                    app,
                    store.add_source,
                    collection_id=collection_id,
                    title=filename,
                    filename=filename,
                    media_type=media_type,
                    document_class=document_class,
                    sha256=sha256,
                    byte_count=size,
                    file_key=file_key,
                    request_id=request_id,
                    source_id=source_id,
                )
            )

            def cleanup_commit(finished):
                if finished.cancelled():
                    return
                try:
                    persisted = finished.result()
                except Exception:
                    asyncio.create_task(asyncio.to_thread(final.unlink, missing_ok=True))
                else:
                    if persisted["id"] != source_id:
                        asyncio.create_task(asyncio.to_thread(final.unlink, missing_ok=True))

            commit_task.add_done_callback(cleanup_commit)
            try:
                source = await asyncio.shield(commit_task)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(commit_task)
                except (asyncio.CancelledError, Exception):
                    pass
                raise
            return source
        except BaseException:
            if commit_task is None and final is not None:
                await asyncio.to_thread(final.unlink, missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/sources/{source_id}", **operation("get_source"))
    async def get_source(source_id: str, auth=Depends(authorized)):
        source = await _db(app, store.get_source, _uuid(source_id))
        if source is None:
            _problem(404, "not_found", "Source not found")
        return source

    @app.delete("/api/sources/{source_id}", **operation("delete_source"))
    async def delete_source(source_id: str, auth=Depends(authorized)):
        return await _db(app, store.delete_source, _uuid(source_id))

    @app.get("/api/sources/{source_id}/file", **operation("source_file"))
    async def source_file(source_id: str, auth=Depends(authorized)):
        source = await _db(app, store.get_source, _uuid(source_id))
        if source is None:
            _problem(404, "not_found", "Source not found")
        if source["status"] == "deleted":
            _problem(410, "source_deleted", "Source was deleted")
        path = _source_path(settings, source["file_key"])
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            _problem(410, "source_unavailable", "Source bytes are unavailable")
        if (
            len(data) != source["byte_count"]
            or hashlib.sha256(data).hexdigest() != source["sha256"]
        ):
            _problem(410, "source_unavailable", "Source bytes failed integrity check")
        disposition = f"inline; filename*=UTF-8''{quote(source['filename'], safe='')}"
        return Response(
            content=data,
            media_type=source["media_type"],
            headers={
                "Content-Disposition": disposition,
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/sources/{source_id}/spans", **operation("source_spans"))
    async def source_spans(
        source_id: str,
        limit: int = 100,
        cursor: str = "",
        anchor: str = "",
        auth=Depends(authorized),
    ):
        source_id = _uuid(source_id)
        if limit < 1 or limit > 100:
            _problem(400, "invalid_limit", "Span page size must be between 1 and 100")
        if cursor and anchor:
            _problem(400, "invalid_cursor", "Use either a cursor or an anchor")
        position = cursor or anchor
        if position:
            try:
                if str(uuid.UUID(position)) != position:
                    raise ValueError
            except ValueError:
                _problem(400, "invalid_cursor", "Span cursor is invalid")
        source = await _db(app, store.get_source, source_id)
        if source is None:
            _problem(404, "not_found", "Source not found")
        if source["status"] == "deleted":
            _problem(410, "source_deleted", "Source was deleted")
        try:
            rows = await _db(
                app, store.source_spans_page, source_id, limit, cursor or None, anchor or None
            )
        except ValueError:
            _problem(400, "invalid_cursor", "Span cursor is invalid")
        items = rows[:limit]
        return {"items": items, "next_cursor": items[-1]["id"] if len(rows) > limit else None}

    @app.get("/api/spans/{span_id}", **operation("get_span"))
    async def get_span(span_id: str, auth=Depends(authorized)):
        spans = await _db(app, store.get_spans, [_uuid(span_id)])
        if not spans:
            _problem(404, "not_found", "Span not found")
        span = spans[0]
        source = await _db(app, store.get_source, span["source_id"])
        return {
            **span,
            "source_title": source["title"],
            "source_hash": source["sha256"],
            "source_filename": source["filename"],
            "source_media_type": source["media_type"],
        }

    @app.get("/api/conversations", **operation("list_conversations"))
    async def list_conversations(
        collection_id: str | None = None,
        limit: int = 100,
        cursor: str = "",
        auth=Depends(authorized),
    ):
        if collection_id is not None:
            collection_id = _uuid(collection_id)
        if limit < 1 or limit > 100:
            _problem(400, "invalid_limit", "Conversation page size must be between 1 and 100")
        scope = f"conversations:{collection_id or 'all'}"
        after = _decode_page_cursor(cursor, scope)
        rows = await _db(app, store.list_conversations_page, collection_id, limit, after)
        items = rows[:limit]
        return {
            "items": items,
            "next_cursor": _encode_page_cursor(items[-1], scope) if len(rows) > limit else None,
        }

    @app.post("/api/conversations", **operation("create_conversation"))
    async def create_conversation(value: ConversationInput, auth=Depends(authorized)):
        return await _db(app, store.create_conversation, _uuid(value.collection_id), value.title)

    @app.get("/api/conversations/{conversation_id}", **operation("get_conversation"))
    async def get_conversation(
        conversation_id: str,
        messages_limit: int = 100,
        messages_cursor: str = "",
        runs_limit: int = 50,
        runs_cursor: str = "",
        auth=Depends(authorized),
    ):
        conversation_id = _uuid(conversation_id)
        if messages_limit < 1 or messages_limit > 100 or runs_limit < 1 or runs_limit > 100:
            _problem(400, "invalid_limit", "Conversation page size must be between 1 and 100")
        conversation = await _db(app, store.get_conversation, conversation_id)
        if conversation is None:
            _problem(404, "not_found", "Conversation not found")
        message_scope = f"messages:{conversation_id}"
        run_scope = f"conversation_runs:{conversation_id}"
        messages_after = _decode_page_cursor(messages_cursor, message_scope)
        runs_after = _decode_page_cursor(runs_cursor, run_scope)
        message_rows, run_rows = await asyncio.gather(
            _db(app, store.messages_page, conversation_id, messages_limit, messages_after),
            _db(app, store.conversation_runs_page, conversation_id, runs_limit, runs_after),
        )
        messages = message_rows[:messages_limit]
        runs = run_rows[:runs_limit]
        return {
            **conversation,
            "messages": messages,
            "runs": runs,
            "messages_next_cursor": _encode_page_cursor(messages[-1], message_scope)
            if len(message_rows) > messages_limit
            else None,
            "runs_next_cursor": _encode_page_cursor(runs[-1], run_scope)
            if len(run_rows) > runs_limit
            else None,
        }

    @app.post(
        "/api/conversations/{conversation_id}/runs", status_code=202, **operation("create_run")
    )
    async def create_run(conversation_id: str, value: RunInput, auth=Depends(authorized)):
        if not await _api_owner_healthy(app):
            _problem(503, "api_ownership_lost", "API process is unavailable")
        conversation_id = _uuid(conversation_id)
        retry = _uuid(value.retry_of_run_id) if value.retry_of_run_id else None
        if retry:
            original = await _db(app, store.get_run, retry)
            if (
                original is None
                or original["conversation_id"] != conversation_id
                or original["status"] not in _TERMINAL
            ):
                _problem(
                    400,
                    "invalid_retry",
                    "Retry must reference a completed run in this conversation",
                )
        async with app.state.run_admission:
            if not app.state.owner_alive:
                _problem(503, "api_ownership_lost", "API process is unavailable")
            existing = await _db(app, store.get_run_by_request, conversation_id, value.request_id)
            if len(app.state.tasks) >= settings.max_active_runs and (
                existing is None
                or existing["status"] == "queued"
                and existing["id"] not in app.state.tasks
            ):
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "run_capacity",
                        "message": "Too many active runs; retry shortly",
                    },
                    headers={"Retry-After": "1"},
                )
            run = await _db(
                app,
                store.create_run,
                conversation_id,
                value.request_id,
                value.question,
                value.model,
                value.variant,
                retry,
            )
            if not await _api_owner_healthy(app):
                await _db(app, store.interrupt_runs, [run["id"]])
                _problem(503, "api_ownership_lost", "API process is unavailable")
            if run["status"] == "queued" and run["id"] not in app.state.tasks:
                task = asyncio.create_task(_execute_run(app, run))
                app.state.tasks[run["id"]] = task
                task.add_done_callback(
                    lambda finished, run_id=run["id"], request_id=run["request_id"]: (
                        _run_task_finished(app, run_id, request_id, finished)
                    )
                )
        return run

    @app.get("/api/runs", **operation("list_runs"))
    async def list_runs(
        limit: int = 100,
        status: str = "all",
        cursor: str = "",
        auth=Depends(authorized),
    ):
        if limit < 1 or limit > 100:
            _problem(400, "invalid_limit", "Run page size must be between 1 and 100")
        if status not in _RUN_STATUSES:
            _problem(400, "invalid_status", "Run status is invalid")
        after = _decode_page_cursor(cursor, f"runs:{status}")
        rows = await _db(app, store.list_runs, limit, status, after)
        items = rows[:limit]
        next_cursor = (
            _encode_page_cursor(items[-1], f"runs:{status}") if len(rows) > limit else None
        )
        return {"items": items, "next_cursor": next_cursor}

    @app.get("/api/runs/{run_id}", **operation("get_run"))
    async def get_run(run_id: str, auth=Depends(authorized)):
        run = await _db(app, store.get_run, _uuid(run_id))
        if run is None:
            _problem(404, "not_found", "Run not found")
        return run

    @app.get("/api/runs/{run_id}/events", **operation("stream_events"))
    async def stream_events(
        run_id: str, request: Request, after: int = 0, auth=Depends(authorized)
    ):
        run_id = _uuid(run_id)
        if await _db(app, store.get_run, run_id) is None:
            _problem(404, "not_found", "Run not found")
        last_id = request.headers.get("last-event-id")
        if last_id:
            match = re.fullmatch(r"([0-9a-fA-F-]{36}):(\d+)", last_id)
            if match is None or _uuid(match.group(1)) != run_id:
                _problem(400, "invalid_cursor", "Last-Event-ID is invalid")
            after = max(after, int(match.group(2)))
        if after < 0:
            _problem(400, "invalid_cursor", "Event cursor is invalid")

        async def events():
            cursor = after
            last_ping = time.monotonic()
            while True:
                if await request.is_disconnected():
                    return
                batch = await _db(app, store.events, run_id, cursor)
                current = await _db(app, store.get_run, run_id)
                if current is not None and current["status"] in _TERMINAL:
                    batch.extend(
                        await _db(
                            app,
                            store.events,
                            run_id,
                            batch[-1]["seq"] if batch else cursor,
                        )
                    )
                for event in batch:
                    cursor = event["seq"]
                    yield f"id: {run_id}:{cursor}\nevent: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
                if current is None or current["status"] in _TERMINAL:
                    return
                if time.monotonic() - last_ping >= 15:
                    yield ": keepalive\n\n"
                    last_ping = time.monotonic()
                await asyncio.sleep(0.5)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/cancel", **operation("cancel_run"))
    async def cancel_run(run_id: str, auth=Depends(authorized)):
        run_id = _uuid(run_id)
        run = await _db(app, store.get_run, run_id)
        if run is None:
            _problem(404, "not_found", "Run not found")
        if await _db(app, store.cancel_run, run_id):
            task = app.state.tasks.get(run_id)
            try:
                await provider.cancel(run_id)
            except Exception as exc:
                await _db(
                    app,
                    store.save_audit,
                    run_id,
                    "provider.cancel_failed",
                    {"type": type(exc).__name__},
                )
            finally:
                if task is not None:
                    task.cancel()
        return await _db(app, store.get_run, run_id)

    @app.get("/api/runs/{run_id}/export", **operation("export_run"))
    async def export_run(run_id: str, auth=Depends(authorized)):
        run = await _db(app, store.get_run, _uuid(run_id))
        if run is None:
            _problem(404, "not_found", "Run not found")
        if run["status"] != "succeeded":
            _problem(409, "run_incomplete", "Run has no completed answer")
        markdown = _run_markdown(run)
        return PlainTextResponse(
            markdown,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=run-{run['id']}.md",
                "Cache-Control": "private, no-store",
            },
        )

    @app.get("/api/stats", **operation("stats"))
    async def stats(auth=Depends(authorized)):
        return await _db(app, _stats, store)

    @app.get("/api/experiments", **operation("experiments"))
    async def experiments(limit: int = 10, cursor: str = "", auth=Depends(authorized)):
        if limit < 1 or limit > 25:
            _problem(400, "invalid_limit", "Campaign page size must be between 1 and 25")
        after = _decode_page_cursor(cursor, "experiments")
        try:
            items, last = await _db(app, _experiments, store, limit, after)
            return {
                "items": items,
                "next_cursor": _encode_page_cursor(last, "experiments") if last else None,
            }
        except (ValueError, KeyError, TypeError) as exc:
            _problem(409, "invalid_experiment_data", str(exc)[:300])

    @app.get("/api/experiments/{experiment_id}", **operation("experiment"))
    async def experiment(experiment_id: str, auth=Depends(authorized)):
        try:
            result = await _db(app, _experiment, store, experiment_id)
        except (ValueError, KeyError, TypeError) as exc:
            _problem(409, "invalid_experiment_data", str(exc)[:300])
        if result is None:
            _problem(404, "not_found", "Experiment not found")
        return result

    @app.api_route(
        "/api/{unmatched:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def unmatched_api(unmatched: str, auth=Depends(authorized)):
        _problem(404, "not_found", "Endpoint not found")

    frontend = Path(settings.frontend_dir)
    if not frontend.is_absolute():
        frontend = Path(__file__).resolve().parents[3] / frontend
    if frontend.is_dir():
        app.mount("/", SPAStaticFiles(directory=frontend, html=True), name="frontend")
    configure_openapi(app)
    return app


async def _db(app, fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app.state.pool, partial(fn, *args, **kwargs))


def _lose_api_ownership(app):
    if not app.state.owner_alive:
        return
    app.state.owner_alive = False
    for task in tuple(app.state.tasks.values()):
        task.cancel()
    request_shutdown = getattr(app.state, "request_shutdown", None)
    if request_shutdown is not None:
        request_shutdown()


async def _api_owner_healthy(app):
    if not app.state.owner_alive:
        return False
    try:
        if await _db(app, app.state.store.api_owner_healthy):
            return True
    except Exception:
        pass
    _lose_api_ownership(app)
    return False


async def _watch_api_owner(app):
    while True:
        await asyncio.sleep(1)
        if app.state.owner_alive and await _api_owner_healthy(app):
            continue
        if getattr(app.state, "request_shutdown", None) is not None:
            return
        await _recover_api_owner(app)


async def _recover_api_owner(app):
    store = app.state.store
    local_tasks = tuple(app.state.tasks.values())
    if local_tasks:
        await asyncio.gather(*local_tasks, return_exceptions=True)
    while not app.state.owner_alive:
        try:
            await _db(app, store.release_api_ownership)
            acquired = await _db(app, store.acquire_api_ownership)
            if acquired:
                try:
                    await _db(app, store.interrupt_active_runs)
                    if not await _db(app, store.api_owner_healthy):
                        raise RuntimeError("API ownership lost during recovery")
                except Exception:
                    await _db(app, store.release_api_ownership)
                    raise
                app.state.owner_alive = True
                _LOG.warning("stage=api_ownership_recovered")
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.error(
                "stage=api_ownership_recovery error_type=%s traceback=%s",
                type(exc).__name__,
                "".join(traceback.format_tb(exc.__traceback__)),
            )
        await asyncio.sleep(1)


def _ping(store):
    with store.engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _reconcile_upload_batch(store, entries, cutoff):
    stale = []
    for entry in entries:
        try:
            if entry.is_dir() or entry.stat().st_mtime >= cutoff:
                continue
        except FileNotFoundError:
            continue
        name = entry.name
        if not name.startswith(".upload-"):
            try:
                uuid.UUID(name)
            except ValueError:
                continue
        stale.append(entry)
    existing = store.source_file_keys_existing([entry.name for entry in stale])
    for entry in stale:
        if entry.name not in existing:
            try:
                Path(entry.path).unlink()
            except FileNotFoundError:
                pass


async def _reconcile_uploads(app, store, settings):
    root = Path(settings.data_dir) / "uploads"
    if not await asyncio.to_thread(root.is_dir):
        return
    cutoff = time.time() - 3600
    entries = await asyncio.to_thread(os.scandir, root)
    try:
        while True:
            batch = await asyncio.to_thread(lambda: list(islice(entries, 256)))
            if not batch:
                return
            await _db(app, _reconcile_upload_batch, store, batch, cutoff)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _LOG.error(
            "stage=upload_reconciliation error_type=%s traceback=%s",
            type(exc).__name__,
            "".join(traceback.format_tb(exc.__traceback__)),
        )
    finally:
        await asyncio.to_thread(entries.close)


def _stats(store):
    """Summarize saved run execution, answer and trace state independently.

    Read all aggregates in one repeatable-read, read-only transaction. Answer
    counts include succeeded runs only; trace counts include all runs and a
    separate terminal-only view. Prepopulate known status names with zeros
    while retaining other persisted status names.

    Args:
        store (Store): Database owner whose engine supplies the read snapshot.

    Returns:
        dict: total_runs, latency_n, nullable latency_p50_ms, runtime/answer/
        trace/terminal_trace_status_counts mappings, latency_measure and an ISO
        observed_at timestamp. Latency includes only succeeded runs with numeric
        end_to_end_ms, measured in milliseconds inside dialogue execution;
        queue and trace-delivery time are excluded. No model calls or writes occur.
    """
    with store.engine.connect() as conn:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        observed_at = conn.execute(text("SELECT transaction_timestamp()")).scalar_one().isoformat()
        total = (
            conn.execute(
                text(
                    "SELECT COUNT(*) AS total_runs,"
                    "COUNT(*) FILTER (WHERE status='succeeded' AND jsonb_typeof(metrics->'end_to_end_ms')='number') AS latency_n,"
                    "percentile_cont(0.5) WITHIN GROUP (ORDER BY (metrics->>'end_to_end_ms')::double precision) "
                    "FILTER (WHERE status='succeeded' AND jsonb_typeof(metrics->'end_to_end_ms')='number') AS latency_p50_ms "
                    "FROM runs"
                )
            )
            .mappings()
            .one()
        )
        by_status = list(
            conn.execute(
                text("SELECT status,COUNT(*) AS count FROM runs GROUP BY status ORDER BY status")
            ).mappings()
        )
        by_answer = list(
            conn.execute(
                text(
                    "SELECT COALESCE(answer->>'status','missing') AS status,COUNT(*) AS count "
                    "FROM runs WHERE status='succeeded' GROUP BY 1 ORDER BY 1"
                )
            ).mappings()
        )
        by_trace = list(
            conn.execute(
                text(
                    "SELECT COALESCE(trace_status,'missing') AS status,COUNT(*) AS count "
                    "FROM runs GROUP BY 1 ORDER BY 1"
                )
            ).mappings()
        )
        terminal_trace = list(
            conn.execute(
                text(
                    "SELECT COALESCE(trace_status,'missing') AS status,COUNT(*) AS count "
                    "FROM runs WHERE status IN ('succeeded','failed','cancelled','interrupted') "
                    "GROUP BY 1 ORDER BY 1"
                )
            ).mappings()
        )
        trace_names = (
            "missing",
            "pending",
            "incomplete",
            "original_materialized",
            "recovered_from_audit",
            "unrecoverable",
        )
        answer_names = (
            "supported",
            "partial",
            "conflicting",
            "not_documented",
            "needs_clarification",
            "missing",
        )
        runtime_names = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
        return {
            **_record(total),
            "runtime_status_counts": {
                **dict.fromkeys(runtime_names, 0),
                **{row["status"]: row["count"] for row in by_status},
            },
            "answer_status_counts": {
                **dict.fromkeys(answer_names, 0),
                **{row["status"]: row["count"] for row in by_answer},
            },
            "trace_status_counts": {
                **dict.fromkeys(trace_names, 0),
                **{row["status"]: row["count"] for row in by_trace},
            },
            "terminal_trace_status_counts": {
                **dict.fromkeys(trace_names, 0),
                **{row["status"]: row["count"] for row in terminal_trace},
            },
            "latency_measure": "succeeded dialogue.execute end_to_end_ms; excludes queue and trace delivery",
            "observed_at": observed_at,
        }


def _run_markdown(run):
    """Render a saved answer with source identity for a portable Markdown export.

    Args:
        run (dict): Run with question and nullable answer. Answer may contain
            answer prose, citations (source_id/id/source_title/page/line_start)
            and limitations. The caller separately requires succeeded status.

    Returns:
        str: Question heading, saved answer prose, source bullets and optional
        limitations, ending with a newline. PDF locations use page; TXT uses
        line_start. The source and span IDs remain explicit, but source files
        are not embedded. Rendering does not modify or persist the run.
    """
    answer = run["answer"] or {}
    lines = [f"# {run['question']}", "", answer.get("answer", ""), "", "## Sources", ""]
    for span in answer.get("citations", []):
        location = f"page {span['page']}" if span.get("page") else f"line {span['line_start']}"
        lines.append(
            f"- {span.get('source_title', 'Source')} ({location}; source {span['source_id']}; span {span['id']})"
        )
    if answer.get("limitations"):
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in answer["limitations"])
    return "\n".join(lines) + "\n"


def _log_run_failure(run, stage, exc):
    _LOG.error(
        "run_id=%s request_id=%s stage=%s error_type=%s error_code=%s traceback=%s",
        run["id"],
        run["request_id"],
        stage,
        type(exc).__name__,
        getattr(exc, "code", None),
        "".join(traceback.format_tb(exc.__traceback__)),
    )


def _run_task_finished(app, run_id, request_id, finished):
    app.state.tasks.pop(run_id, None)
    if finished.cancelled():
        return
    exc = finished.exception()
    if exc is None:
        return
    _LOG.error(
        "run_id=%s request_id=%s stage=background_task error_type=%s traceback=%s",
        run_id,
        request_id,
        type(exc).__name__,
        "".join(traceback.format_tb(exc.__traceback__)),
    )
    _lose_api_ownership(app)


async def _execute_run(app, run):
    """Execute an admitted question and publish only under its frozen scope.

    Verify API ownership, atomically move a queued run to running, and reload
    its saved state before executing the bounded dialogue graph under telemetry.
    Publication is delegated to storage, which checks that the run is still
    active and its collection version has not changed. A completed provider
    answer can therefore be withheld after cancellation or source changes.

    Args:
        app (FastAPI): Running application with store, provider, telemetry,
            dialogue and its database executor initialized in app.state.
        run (dict): Admitted run record containing id and request_id; the full
            current question/model/scope is reloaded after successful start.

    Returns:
        None: Returns without generation when ownership is lost or starting
        the queued run fails. Otherwise publishes a succeeded answer/metrics or
        saves a failed/interrupted state. Dialogue progress/audits may persist
        before any terminal result. Trace delivery is a separate lifecycle.

    Raises:
        asyncio.CancelledError: After requesting run interruption and provider
            cancellation; cancellation failure is privately audited. Existing
            terminal states are not overwritten by interruption.
        Exception: Ownership/start/load failures before the execution try block,
            failed interruption/cancellation auditing, or failure to persist a
            terminal error can propagate to the task completion callback.
            Execution/provider exceptions are normally audited and normalized
            to run_failed, except scope_expired, whose public error asks for a
            new run. No automatic retry is started.
    """
    store = app.state.store
    provider = app.state.provider
    telemetry = app.state.telemetry
    stage = "ownership_check"
    if not await _api_owner_healthy(app):
        await _db(app, store.interrupt_runs, [run["id"]])
        return
    stage = "start_run"
    if not await _db(app, store.start_run, run["id"]):
        return
    stage = "load_run"
    run = await _db(app, store.get_run, run["id"])
    trace_id = None
    trace_url = None
    try:
        async with telemetry.trace(run, persist=lambda fn, *args: _db(app, fn, *args)) as trace:
            trace_id = getattr(trace, "trace_id", None)
            trace_url = getattr(trace, "trace_url", None)
            stage = "dialogue"
            answer, metrics = await app.state.dialogue.execute(run)
            if not await _api_owner_healthy(app):
                raise asyncio.CancelledError
            stage = "finish_run"
            published = await _db(
                app, store.finish_run, run["id"], answer, metrics, trace_id, trace_url
            )
            trace.update(
                output={
                    "published": published,
                    "answer": answer if published else None,
                    "metrics": metrics,
                }
            )
    except asyncio.CancelledError:
        stage = "interrupt_run"
        await _db(app, store.interrupt_runs, [run["id"]])
        try:
            await provider.cancel(run["id"])
        except Exception as exc:
            await _db(
                app,
                store.save_audit,
                run["id"],
                "provider.cancel_failed",
                {"type": type(exc).__name__},
            )
        raise
    except Exception as exc:
        _log_run_failure(run, stage, exc)
        exception_audit = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        if isinstance(exc, ProviderError):
            exception_audit["code"] = exc.code
            for field in (
                "response_status",
                "response_reason",
                "exit_code",
                "usage",
                "raw_provider_usage",
            ):
                value = getattr(exc, field, None)
                if value is not None:
                    exception_audit[field] = value
        try:
            await _db(app, store.save_audit, run["id"], "run.exception", exception_audit)
        except Exception as audit_exc:
            _log_run_failure(run, "save_exception_audit", audit_exc)
        scope_expired = (
            isinstance(exc, (ValueError, MCPToolFailure)) and str(exc) == "scope_expired"
        )
        try:
            await _db(
                app,
                store.fail_run,
                run["id"],
                {"code": "scope_expired", "message": "Collection changed; start a new run"}
                if scope_expired
                else {"code": "run_failed", "message": "Run failed; see private audit for details"},
            )
        except Exception as terminal_exc:
            _log_run_failure(run, "persist_terminal_status", terminal_exc)
            raise


app = create_app()
