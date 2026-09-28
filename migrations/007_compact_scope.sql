DO $$
BEGIN
 IF NOT EXISTS (
  SELECT 1 FROM pg_attribute
  WHERE attrelid='collections'::regclass AND attname='source_count' AND NOT attisdropped
 ) THEN
  ALTER TABLE collections
   ADD COLUMN source_count bigint NOT NULL DEFAULT 0,
   ADD COLUMN ready_count bigint NOT NULL DEFAULT 0,
   ADD COLUMN unavailable_count bigint NOT NULL DEFAULT 0;
  UPDATE collections c SET
   source_count=inventory.source_count,
   ready_count=inventory.ready_count,
   unavailable_count=inventory.unavailable_count
  FROM (
   SELECT collection_id,
    count(*) AS source_count,
    count(*) FILTER (WHERE status='ready') AS ready_count,
    count(*) FILTER (WHERE status IN ('pending','processing','failed')) AS unavailable_count
   FROM sources
   WHERE status<>'deleted' AND deleted_at IS NULL
   GROUP BY collection_id
  ) inventory
  WHERE c.id=inventory.collection_id;
 END IF;
END $$;

DO $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='collections_source_counts_ck') THEN
  ALTER TABLE collections ADD CONSTRAINT collections_source_counts_ck
   CHECK (source_count>=0 AND ready_count>=0 AND unavailable_count>=0
    AND source_count=ready_count+unavailable_count);
 END IF;
END $$;

ALTER TABLE spans ADD COLUMN IF NOT EXISTS collection_id uuid;
DO $$
BEGIN
 IF EXISTS (
  SELECT 1 FROM pg_attribute
  WHERE attrelid='spans'::regclass AND attname='collection_id'
   AND NOT attisdropped AND NOT attnotnull
 ) THEN
  UPDATE spans p SET collection_id=s.collection_id
  FROM sources s WHERE p.source_id=s.id AND p.collection_id IS NULL;
  ALTER TABLE spans ALTER COLUMN collection_id SET NOT NULL;
 END IF;
END $$;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id uuid;
DO $$
BEGIN
 IF EXISTS (
  SELECT 1 FROM pg_attribute
  WHERE attrelid='chunks'::regclass AND attname='collection_id'
   AND NOT attisdropped AND NOT attnotnull
 ) THEN
  UPDATE chunks c SET collection_id=s.collection_id
  FROM sources s WHERE c.source_id=s.id AND c.collection_id IS NULL;
  ALTER TABLE chunks ALTER COLUMN collection_id SET NOT NULL;
 END IF;
END $$;

CREATE INDEX IF NOT EXISTS sources_collection_status_id_idx ON sources(collection_id,status,id);
CREATE INDEX IF NOT EXISTS sources_collection_page_idx ON sources(collection_id,created_at DESC,id DESC) WHERE status<>'deleted';
CREATE INDEX IF NOT EXISTS spans_collection_order_idx ON spans(collection_id,source_id,(COALESCE(page,2147483647)),line_start,id);
CREATE INDEX IF NOT EXISTS spans_source_page_idx ON spans(source_id,(COALESCE(page,0)),line_start,id);
