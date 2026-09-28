DO $$
BEGIN
 IF EXISTS (
  SELECT 1 FROM pg_index definition
  JOIN pg_class index_relation ON index_relation.oid=definition.indexrelid
  WHERE definition.indrelid='runs'::regclass
   AND index_relation.relname='runs_created_idx'
   AND definition.indnkeyatts=1
 ) THEN
  DROP INDEX runs_created_idx;
 END IF;
END $$;
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_at DESC,id DESC);
CREATE INDEX IF NOT EXISTS runs_status_page_idx ON runs(status,created_at DESC,id DESC);
