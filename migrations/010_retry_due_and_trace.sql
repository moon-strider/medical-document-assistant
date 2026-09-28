ALTER TABLE jobs ADD COLUMN available_at timestamptz NOT NULL DEFAULT now();
CREATE INDEX jobs_due_idx ON jobs(available_at,created_at,id) WHERE status='ready';

ALTER TABLE sources ADD COLUMN cleanup_status text;
ALTER TABLE sources ADD COLUMN cleanup_error text;
ALTER TABLE sources ADD CONSTRAINT sources_cleanup_status_ck
 CHECK (cleanup_status IS NULL OR cleanup_status IN ('queued','retrying','complete'));
ALTER TABLE runs ADD COLUMN trace_reconcile_after timestamptz;
ALTER TABLE runs ADD COLUMN trace_reconcile_attempts integer NOT NULL DEFAULT 0;
CREATE INDEX runs_trace_reconcile_due_idx ON runs(trace_reconcile_after,id)
 WHERE status IN ('succeeded','failed','cancelled','interrupted')
 AND (trace_status IS NULL OR trace_status IN ('pending','incomplete'))
 AND trace_reconcile_after IS NOT NULL;
CREATE INDEX runs_trace_reconcile_initial_idx ON runs(finished_at,id)
 WHERE status IN ('succeeded','failed','cancelled','interrupted')
 AND (trace_status IS NULL OR trace_status IN ('pending','incomplete'))
 AND trace_reconcile_after IS NULL;

CREATE INDEX conversations_created_page_idx ON conversations(created_at DESC,id DESC);
CREATE INDEX conversations_collection_page_idx ON conversations(collection_id,created_at DESC,id DESC);
DROP INDEX conversations_collection_idx;
CREATE INDEX runs_conversation_page_idx ON runs(conversation_id,created_at DESC,id DESC);
GRANT SELECT (id,collection_id) ON conversations TO assistant_readonly;
