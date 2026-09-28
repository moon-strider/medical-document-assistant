CREATE TABLE IF NOT EXISTS evaluation_campaigns (
 id uuid PRIMARY KEY,
 release_id text NOT NULL,
 split text NOT NULL CHECK (split IN ('development','acceptance')),
 status text NOT NULL CHECK (status IN ('importing','imported','frozen','running','incomplete','complete')),
 manifest_sha256 char(64) NOT NULL,
 questions_sha256 char(64),
 gold_sha256 char(64),
 extraction_map_sha256 char(64),
 artifact_manifest_sha256 char(64) NOT NULL,
 config_sha256 char(64),
 rubric_sha256 char(64),
 code_sha256 char(64),
 selected_variant text,
 selected_generator text,
 planned_count integer NOT NULL DEFAULT 0,
 source_map jsonb NOT NULL DEFAULT '{"collection_id":null,"collection_revision":null,"documents":{}}'::jsonb,
 config jsonb NOT NULL DEFAULT '{}'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 frozen_at timestamptz,
 updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(split,release_id)
);
CREATE INDEX IF NOT EXISTS evaluation_campaigns_status_idx ON evaluation_campaigns(status,created_at);
ALTER TABLE evaluation_campaigns ALTER COLUMN source_map SET DEFAULT '{"collection_id":null,"collection_revision":null,"documents":{}}'::jsonb;
CREATE TABLE IF NOT EXISTS evaluation_attempts (
 id uuid PRIMARY KEY,
 campaign_id uuid NOT NULL REFERENCES evaluation_campaigns(id),
 case_id text NOT NULL,
 variant text NOT NULL CHECK (variant IN ('V0','V1','V2','V3')),
 generator text NOT NULL CHECK (generator IN ('gpt-6-sol','gpt-6-luna')),
 metadata jsonb NOT NULL,
 state text NOT NULL CHECK (state IN ('pending','running','first_turn_failed','target_failed','completed','needs_review')),
 attempts jsonb NOT NULL DEFAULT '[]'::jsonb,
 first_run_id uuid,
 target_run_id uuid,
 judge_output jsonb,
 judge_status text,
 judge_na_reason text,
 metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(campaign_id,case_id,variant,generator)
);
CREATE INDEX IF NOT EXISTS evaluation_attempts_campaign_state_idx ON evaluation_attempts(campaign_id,state,case_id);
DO $$
BEGIN
 IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='assistant_readonly') THEN
  EXECUTE 'REVOKE ALL ON evaluation_campaigns,evaluation_attempts FROM assistant_readonly';
 END IF;
END
$$;
