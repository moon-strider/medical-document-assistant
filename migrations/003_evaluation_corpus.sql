CREATE TABLE IF NOT EXISTS evaluation_corpora (
 id uuid PRIMARY KEY,
 corpus_id text NOT NULL UNIQUE,
 status text NOT NULL CHECK (status IN ('importing','ready')),
 manifest_sha256 char(64) NOT NULL,
 artifact_manifest_sha256 char(64) NOT NULL,
 source_map jsonb NOT NULL DEFAULT '{"collection_id":null,"collection_revision":null,"documents":{}}'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS evaluation_corpora_status_idx ON evaluation_corpora(status,created_at);
ALTER TABLE evaluation_campaigns ADD COLUMN IF NOT EXISTS corpus_row_id uuid;
ALTER TABLE evaluation_campaigns ADD COLUMN IF NOT EXISTS fingerprint char(64);
DO $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='evaluation_campaigns_corpus_row_id_fkey') THEN
  ALTER TABLE evaluation_campaigns ADD CONSTRAINT evaluation_campaigns_corpus_row_id_fkey FOREIGN KEY (corpus_row_id) REFERENCES evaluation_corpora(id);
 END IF;
END
$$;
ALTER TABLE evaluation_campaigns ALTER COLUMN corpus_row_id SET NOT NULL;
ALTER TABLE evaluation_campaigns ALTER COLUMN fingerprint SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS evaluation_campaigns_fingerprint_idx ON evaluation_campaigns(fingerprint);
ALTER TABLE evaluation_campaigns DROP CONSTRAINT IF EXISTS evaluation_campaigns_split_release_id_key;
ALTER TABLE evaluation_campaigns DROP CONSTRAINT IF EXISTS evaluation_campaigns_status_check;
ALTER TABLE evaluation_campaigns ADD CONSTRAINT evaluation_campaigns_status_check CHECK (status IN ('frozen','running','incomplete','complete'));
DO $$
BEGIN
 IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='assistant_readonly') THEN
  EXECUTE 'REVOKE ALL ON evaluation_corpora FROM assistant_readonly';
 END IF;
END
$$;
