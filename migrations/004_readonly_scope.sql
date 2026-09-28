ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM assistant_readonly;
REVOKE SELECT ON TABLE conversations, messages, run_events, run_audit, jobs, evaluation_campaigns, evaluation_attempts FROM assistant_readonly;
GRANT SELECT ON TABLE collections, sources, spans, chunks, runs TO assistant_readonly;
