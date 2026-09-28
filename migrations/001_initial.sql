CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS collections (
 id uuid PRIMARY KEY,
 title text NOT NULL,
 description text NOT NULL DEFAULT '',
 revision bigint NOT NULL DEFAULT 0,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sources (
 id uuid PRIMARY KEY,
 collection_id uuid NOT NULL REFERENCES collections(id),
 request_id text NOT NULL,
 title text NOT NULL,
 filename text NOT NULL,
 media_type text NOT NULL CHECK (media_type IN ('application/pdf','text/plain')),
 document_class text NOT NULL CHECK (document_class IN ('D1','D2','D3','D4','D5','other')),
 status text NOT NULL CHECK (status IN ('pending','processing','ready','failed','deleted')),
 sha256 char(64) NOT NULL,
 byte_count bigint NOT NULL CHECK (byte_count > 0),
 page_count integer,
 error text,
 file_key text NOT NULL UNIQUE,
 created_at timestamptz NOT NULL DEFAULT now(),
 deleted_at timestamptz,
 UNIQUE(collection_id,request_id)
);
CREATE INDEX IF NOT EXISTS sources_collection_status_idx ON sources(collection_id,status,created_at);
CREATE TABLE IF NOT EXISTS spans (
 id uuid PRIMARY KEY,
 source_id uuid NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
 page integer,
 line_start integer NOT NULL,
 line_end integer NOT NULL,
 text text NOT NULL,
 bbox jsonb,
 page_width double precision,
 page_height double precision,
 sha256 char(64) NOT NULL,
 section text NOT NULL DEFAULT '',
 created_at timestamptz NOT NULL DEFAULT now(),
 CHECK (page IS NULL OR page > 0),
 CHECK (line_start > 0 AND line_end >= line_start)
);
CREATE INDEX IF NOT EXISTS spans_source_order_idx ON spans(source_id,page,line_start,id);
CREATE TABLE IF NOT EXISTS chunks (
 id uuid PRIMARY KEY,
 source_id uuid NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
 span_ids jsonb NOT NULL,
 text text NOT NULL,
 variant text NOT NULL CHECK (variant IN ('fixed','structural')),
 embedding vector(384) NOT NULL,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
 search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_source_variant_idx ON chunks(source_id,variant);
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin(search_tsv);
CREATE TABLE IF NOT EXISTS jobs (
 id uuid PRIMARY KEY,
 kind text NOT NULL,
 payload jsonb NOT NULL,
 idempotency_key text NOT NULL UNIQUE,
 status text NOT NULL CHECK (status IN ('ready','running','done','failed')),
 attempt integer NOT NULL DEFAULT 0,
 worker_id text,
 lease_until timestamptz,
 error text,
 created_at timestamptz NOT NULL DEFAULT now(),
 finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status,lease_until,created_at);
CREATE TABLE IF NOT EXISTS conversations (
 id uuid PRIMARY KEY,
 collection_id uuid NOT NULL REFERENCES collections(id),
 title text NOT NULL DEFAULT '',
 revision bigint NOT NULL DEFAULT 0,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_collection_idx ON conversations(collection_id,created_at);
CREATE TABLE IF NOT EXISTS runs (
 id uuid PRIMARY KEY,
 conversation_id uuid NOT NULL REFERENCES conversations(id),
 request_id text NOT NULL,
 question text NOT NULL,
 model text NOT NULL,
 variant text NOT NULL CHECK (variant IN ('V0','V1','V2','V3')),
 status text NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled','interrupted')),
 scope jsonb NOT NULL,
 answer jsonb,
 error jsonb,
 trace_id text,
 trace_url text,
 trace_status text,
 metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 finished_at timestamptz,
 retry_of_run_id uuid REFERENCES runs(id),
 UNIQUE(conversation_id,request_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_conversation ON runs(conversation_id) WHERE status IN ('queued','running');
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_at DESC,id DESC);
CREATE TABLE IF NOT EXISTS messages (
 id uuid PRIMARY KEY,
 conversation_id uuid NOT NULL REFERENCES conversations(id),
 run_id uuid NOT NULL REFERENCES runs(id),
 role text NOT NULL CHECK (role IN ('user','assistant')),
 content text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(run_id,role)
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages(conversation_id,created_at,id);
CREATE TABLE IF NOT EXISTS run_events (
 run_id uuid NOT NULL REFERENCES runs(id),
 seq bigint NOT NULL,
 type text NOT NULL,
 at timestamptz NOT NULL DEFAULT now(),
 payload jsonb NOT NULL DEFAULT '{}'::jsonb,
 PRIMARY KEY(run_id,seq)
);
CREATE TABLE IF NOT EXISTS run_audit (
 id uuid PRIMARY KEY,
 run_id uuid NOT NULL REFERENCES runs(id),
 stage text NOT NULL,
 payload jsonb NOT NULL,
 at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS run_audit_run_idx ON run_audit(run_id,at);
