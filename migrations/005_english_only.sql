DO $$
BEGIN
 IF EXISTS (
  SELECT 1 FROM pg_attribute
  WHERE attrelid='chunks'::regclass AND attname='language' AND NOT attisdropped
 ) THEN
  ALTER TABLE chunks DROP COLUMN search_tsv;
  ALTER TABLE chunks DROP COLUMN language;
  ALTER TABLE chunks ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS
   (to_tsvector('english'::regconfig, text)) STORED;
 END IF;
 IF EXISTS (
  SELECT 1 FROM pg_attribute
  WHERE attrelid='sources'::regclass AND attname='language' AND NOT attisdropped
 ) THEN
  ALTER TABLE sources DROP COLUMN language;
 END IF;
END
$$;
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin(search_tsv);
