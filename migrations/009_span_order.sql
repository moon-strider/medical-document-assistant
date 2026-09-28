ALTER TABLE spans ADD COLUMN ordinal integer;

WITH ordered AS (
 SELECT id,
  row_number() OVER (PARTITION BY source_id ORDER BY page NULLS FIRST,line_start,id) AS value
 FROM spans
)
UPDATE spans p SET ordinal=ordered.value::integer
FROM ordered WHERE p.id=ordered.id;

ALTER TABLE spans ALTER COLUMN ordinal SET NOT NULL;
ALTER TABLE spans ADD CONSTRAINT spans_ordinal_positive_ck CHECK (ordinal>0);

DROP INDEX spans_source_order_idx;
DROP INDEX spans_collection_order_idx;
DROP INDEX spans_source_page_idx;
CREATE UNIQUE INDEX spans_source_ordinal_uq ON spans(source_id,ordinal);
CREATE INDEX spans_collection_ordinal_idx ON spans(collection_id,source_id,ordinal);
