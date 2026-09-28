import logging
import os
import socket
import threading
import time
import uuid

import psycopg
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from medical_assistant.errors import TransientIngestError
from medical_assistant.ingest import _source_path, process_source
from medical_assistant.settings import get_settings
from medical_assistant.storage import Store

_LOG = logging.getLogger(__name__)


class LeaseLostError(RuntimeError):
    """Stop ingestion when this worker can no longer confirm its claimed lease.

    The worker excludes this RuntimeError subtype from retryable ingestion
    failures. Finalization still checks the database lease and may refuse the
    stale attempt, leaving recovery to a later claim by another worker.
    """

    pass


def _heartbeat(store, job, worker_id, stop, lost, state):
    """Renew a running job while maintaining a conservative local lease deadline.

    Wait 60 seconds between successful renewals and retry renewal exceptions
    after 5 seconds. Exceptions are logged without extending the deadline;
    a rejected renewal or reaching the deadline signals lease loss. A
    successful renewal sets the monotonic deadline 895 seconds ahead, five
    seconds short of the database lease duration.

    Args:
        store (Store): Storage providing lease renewal.
        job (dict): Claimed job containing ``id`` (UUID str) and ``attempt`` (int).
        worker_id (str): Claimed owner identity.
        stop (threading.Event): Signal to stop this heartbeat thread.
        lost (threading.Event): Mutable signal set when the lease is unconfirmed.
        state (dict[str, float]): Shared ``deadline`` in monotonic seconds,
            updated after confirmed renewal.

    Returns:
        None: Heartbeat stopped or lease loss was signalled. Renewal exceptions
        are contained here; this function does not finalize the job.
    """
    interval = 60
    while not stop.wait(interval):
        if time.monotonic() >= state["deadline"]:
            lost.set()
            return
        try:
            if not store.renew_job(job["id"], worker_id, job["attempt"]):
                lost.set()
                return
            state["deadline"] = time.monotonic() + 895
            interval = 60
        except Exception:
            _LOG.exception("Job lease renewal failed")
            interval = 5


def _delete_source(store, source_id, settings):
    """Remove a deleted source's original before cleaning any leftover indexes.

    Validate the source's deleted state and confined, nonsymlink storage path,
    then unlink its file and clean database spans/chunks. An already absent file
    is accepted so a retry can finish after partial filesystem success. File
    deletion and database cleanup are separate operations; job finalization
    later records completion or schedules another attempt.

    Args:
        store (Store): Source metadata and cleanup storage.
        source_id (str): UUID of a source already marked deleted.
        settings (Settings): Settings containing the private storage root.

    Returns:
        None: File removal and database cleanup completed; cleanup status is
        not updated by this function.

    Raises:
        KeyError: The source does not exist.
        ValueError: The source is not deleted or its storage path is unsafe.
        OSError: The original cannot be removed because of a filesystem error.
    """
    source = store.get_source(source_id)
    if source is None:
        raise KeyError(source_id)
    if source["status"] != "deleted":
        raise ValueError("Source is not deleted")
    path = _source_path(settings, source["file_key"])
    if path.is_symlink():
        raise ValueError("Source path is a symlink")
    path.unlink(missing_ok=True)
    store.cleanup_source(source_id)


def run_once(store, settings, worker_id):
    """Claim and process at most one ingestion or deletion job.

    Start a heartbeat and dispatch by job kind, then signal the heartbeat to
    stop and wait up to two seconds before finalizing the claimed attempt.
    Ingestion receives a callback that checks local lease confirmation between
    expensive stages; deletion has no
    such callback. Processing exceptions are logged and converted to a job
    error. Ingest retries qualify for transient-ingest, OS/database, timeout
    or RuntimeError failures except LeaseLostError; ``Store.finish_job`` owns
    the attempt cap and backoff. Deletion errors follow its deletion retry
    policy regardless of the classification here.

    Args:
        store (Store): Storage providing job, source and lease operations.
        settings (Settings): Parser/model settings and private file root.
        worker_id (str): Worker identity recorded on the claimed lease.

    Returns:
        bool: False when no job was claimed. True when a claimed job reached
        finalization, including processing failure or a False finalization
        result due to a stale lease; it does not assert successful ingestion
        or deletion. Claim/finalization database failures propagate instead.
    """
    job = store.claim_job(worker_id)
    if job is None:
        return False
    stop = threading.Event()
    lost = threading.Event()
    state = {"deadline": time.monotonic() + 895}
    heart = threading.Thread(
        target=_heartbeat, args=(store, job, worker_id, stop, lost, state), daemon=True
    )
    heart.start()
    error = None
    retryable = False

    def ensure_lease():
        """Abort ingestion when heartbeat loss or local lease expiry is observed.

        Raises:
            LeaseLostError: The shared loss event is set or the monotonic
                confirmation deadline has been reached.
        """
        if lost.is_set() or time.monotonic() >= state["deadline"]:
            raise LeaseLostError("Job lease is no longer confirmed")

    try:
        if job["kind"] == "source_ingest":
            process_source(
                store,
                job["payload"]["source_id"],
                settings,
                lease={"job_id": job["id"], "worker_id": worker_id, "attempt": job["attempt"]},
                ensure_lease=ensure_lease,
            )
        elif job["kind"] == "source_delete":
            _delete_source(store, job["payload"]["source_id"], settings)
        else:
            raise ValueError(f"Unknown job kind: {job['kind']}")
    except Exception as exc:
        error = str(exc) or type(exc).__name__
        retryable = isinstance(
            exc,
            (
                TransientIngestError,
                OSError,
                OperationalError,
                InterfaceError,
                psycopg.OperationalError,
                TimeoutError,
                SQLAlchemyTimeoutError,
                RuntimeError,
            ),
        ) and not isinstance(exc, LeaseLostError)
        _LOG.exception("Job %s failed", job["id"])
    finally:
        stop.set()
        heart.join(timeout=2)
        store.finish_job(
            job["id"], error, worker_id=worker_id, attempt=job["attempt"], retryable=retryable
        )
    return True


def main():
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    store = Store(settings)
    store.migrate()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    while True:
        try:
            if not run_once(store, settings, worker_id):
                time.sleep(1)
        except KeyboardInterrupt:
            break
        except Exception:
            _LOG.exception("Worker loop failed")
            time.sleep(3)


if __name__ == "__main__":
    main()
