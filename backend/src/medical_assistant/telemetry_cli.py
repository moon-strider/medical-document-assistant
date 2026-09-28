import argparse
import logging
import time

from sqlalchemy import text

from medical_assistant.settings import get_settings
from medical_assistant.storage import Store
from medical_assistant.telemetry import Telemetry

_LOG = logging.getLogger(__name__)


def run_once(store, telemetry, limit=25):
    """Reconcile one bounded batch of due terminal runs and schedule retries.

    Select succeeded/failed/cancelled/interrupted runs with NULL, pending, or
    incomplete trace status, ordered by due time then ID. Use the durable
    trace_reconcile_after deadline when present; otherwise require finish at
    least five seconds ago. Without a client, only NULL-status rows are selected
    so their audit can still be classified. This is a database selection, not a
    claimed/leased queue, so concurrent passes may select the same run.

    Reconcile each selected run independently. Pending/incomplete results get
    durable exponential retry deadlines (15-second base for pending/error,
    60 seconds for incomplete, capped at one hour); late scores reset backoff
    through Telemetry.score. A per-run error is logged and scheduling attempted
    with error policy without aborting later runs. No inference is performed.

    Args:
        store (Store): Database engine plus trace-retry scheduling methods.
        telemetry (Telemetry): Optional client and reconcile_run boundary.
        limit (int): Maximum selected batch size, from 1 through 100.

    Returns:
        dict[str, int]: checked count, counts for original_materialized,
        recovered_from_audit, pending, incomplete, unrecoverable, and errors.
        checked counts returned reconciliation results; errors may include a
        subsequent scheduling failure for an already counted result, so those
        two counters are not guaranteed to partition selected runs.

    Raises:
        ValueError: limit is outside the supported range. Database selection
            failures propagate before per-run error handling."""
    if not 1 <= limit <= 100:
        raise ValueError("reconciliation batch limit must be between 1 and 100")
    counts = {
        "checked": 0,
        "original_materialized": 0,
        "recovered_from_audit": 0,
        "pending": 0,
        "incomplete": 0,
        "unrecoverable": 0,
        "errors": 0,
    }
    with store.engine.connect() as connection:
        run_ids = [
            str(row[0])
            for row in connection.execute(
                text(
                    "WITH due AS ("
                    " (SELECT id,trace_reconcile_after AS due_at FROM runs"
                    " WHERE status IN ('succeeded','failed','cancelled','interrupted')"
                    " AND (trace_status IN ('pending','incomplete') OR trace_status IS NULL)"
                    " AND (:client_available OR trace_status IS NULL)"
                    " AND trace_reconcile_after IS NOT NULL AND trace_reconcile_after<=now()"
                    " ORDER BY trace_reconcile_after,id LIMIT :limit)"
                    " UNION ALL"
                    " (SELECT id,finished_at AS due_at FROM runs"
                    " WHERE status IN ('succeeded','failed','cancelled','interrupted')"
                    " AND (trace_status IN ('pending','incomplete') OR trace_status IS NULL)"
                    " AND (:client_available OR trace_status IS NULL)"
                    " AND trace_reconcile_after IS NULL"
                    " AND finished_at<=now()-interval '5 seconds'"
                    " ORDER BY finished_at,id LIMIT :limit)"
                    ") SELECT id FROM due ORDER BY due_at,id LIMIT :limit"
                ),
                {"limit": limit, "client_available": telemetry.client is not None},
            )
        ]
    for run_id in run_ids:
        try:
            result = telemetry.reconcile_run(run_id)
            counts["checked"] += 1
            counts[result["trace_status"]] += 1
            if result["trace_status"] in {"pending", "incomplete"}:
                store.schedule_trace_reconcile(run_id, result["trace_status"])
        except Exception:
            counts["errors"] += 1
            _LOG.exception("Trace reconciliation failed for run %s", run_id)
            try:
                store.schedule_trace_reconcile(run_id, "error")
            except Exception:
                _LOG.exception("Trace reconciliation scheduling failed for run %s", run_id)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("interval must be at least 5 seconds")
    if not 1 <= args.limit <= 100:
        parser.error("limit must be between 1 and 100")
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    store = Store(settings)
    telemetry = Telemetry(settings, store)
    while True:
        counts = run_once(store, telemetry, limit=args.limit)
        if counts["checked"] or counts["errors"]:
            _LOG.info("Trace reconciliation: %s", counts)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
