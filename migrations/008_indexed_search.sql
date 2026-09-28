DO $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname='pg_textsearch') THEN
  RAISE EXCEPTION 'pg_textsearch_required';
 END IF;
END $$;

CREATE OR REPLACE FUNCTION normalize_search_identifier(value text) RETURNS text
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE
 normalized text := lower(rtrim(value, '.,;:'));
BEGIN
 WHILE right(normalized,1)=')'
  AND length(normalized)-length(replace(normalized,')',''))
   > length(normalized)-length(replace(normalized,'(','')) LOOP
  normalized := left(normalized,length(normalized)-1);
 END LOOP;
 RETURN normalized;
END $$;

CREATE OR REPLACE FUNCTION search_identifier_tokens(value text) RETURNS text[]
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
 SELECT COALESCE(
  array_agg(DISTINCT normalize_search_identifier(token)
   ORDER BY normalize_search_identifier(token)),
  ARRAY[]::text[]
 )
 FROM (
  SELECT m[1] AS token FROM regexp_matches(
   value,
   '(10[.][0-9]{4,9}/[A-Za-z0-9._;()/:+-]+|[A-Za-z][A-Za-z0-9]*[-./:][A-Za-z0-9/._:-]+|[A-Za-z]{2,}[0-9][A-Za-z0-9]*)',
   'g'
  ) AS m
  UNION ALL
  SELECT m[1] FROM regexp_matches(
   value, '(10[.][0-9]{4,9}/[A-Za-z0-9._;()/:+-]+)', 'g'
  ) AS m
  UNION ALL
  SELECT m[1] FROM regexp_matches(
   value, '(NCT[0-9]{8})(?![0-9])', 'gi'
  ) AS m
 ) AS tokens
 WHERE token ~ '[0-9]'
$$;

CREATE OR REPLACE FUNCTION search_identifier_lexemes(value text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
 SELECT COALESCE(string_agg(
  'pflid' || encode(sha256(convert_to(token, 'UTF8')), 'hex'),
  ' ' ORDER BY token
 ), '')
 FROM unnest(search_identifier_tokens(value)) AS token
$$;

CREATE OR REPLACE FUNCTION ensure_collection_search_partition(collection_uuid uuid)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE
 partition_name text := 'chunks_c' || replace(collection_uuid::text, '-', '');
BEGIN
 IF to_regclass('public.' || partition_name) IS NULL THEN
  EXECUTE format(
   'CREATE TABLE public.%I PARTITION OF public.chunks FOR VALUES IN (%L)',
   partition_name, collection_uuid
  );
 END IF;
 EXECUTE format(
  'CREATE INDEX IF NOT EXISTS %I ON public.%I (source_id)',
  partition_name || '_source', partition_name
 );
 EXECUTE format(
  'CREATE INDEX IF NOT EXISTS %I ON public.%I USING hnsw (embedding vector_cosine_ops) WHERE variant = ''fixed''',
  partition_name || '_fixed_hnsw', partition_name
 );
 EXECUTE format(
  'CREATE INDEX IF NOT EXISTS %I ON public.%I USING hnsw (embedding vector_cosine_ops) WHERE variant = ''structural''',
  partition_name || '_structural_hnsw', partition_name
 );
 EXECUTE format(
  'CREATE INDEX IF NOT EXISTS %I ON public.%I USING bm25 (search_text) WITH (text_config = ''english'') WHERE variant = ''structural''',
  partition_name || '_bm25', partition_name
 );
 EXECUTE format('GRANT SELECT ON public.%I TO assistant_readonly', partition_name);
END $$;

DO $$
BEGIN
 IF NOT EXISTS (
  SELECT 1 FROM pg_partitioned_table WHERE partrelid='public.chunks'::regclass
 ) THEN
  ALTER TABLE public.chunks RENAME TO chunks_before_indexed_search;
  CREATE TABLE public.chunks (
   id uuid NOT NULL,
   source_id uuid NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
   collection_id uuid NOT NULL REFERENCES collections(id),
   span_ids jsonb NOT NULL,
   text text NOT NULL,
   source_label text NOT NULL,
   variant text NOT NULL CHECK (variant IN ('fixed','structural')),
   embedding vector(384) NOT NULL,
   metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
   search_text text GENERATED ALWAYS AS (
    source_label || E'\n' || text || E'\n' ||
    search_identifier_lexemes(source_label || E'\n' || text)
   ) STORED,
   search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, source_label || E'\n' || text)) STORED,
   identifier_tokens text[] GENERATED ALWAYS AS (search_identifier_tokens(source_label || E'\n' || text)) STORED,
   created_at timestamptz NOT NULL DEFAULT now(),
   PRIMARY KEY (collection_id,id)
  ) PARTITION BY LIST (collection_id);
  PERFORM ensure_collection_search_partition(id) FROM collections;
  INSERT INTO public.chunks (
   id,source_id,collection_id,span_ids,text,source_label,variant,embedding,metadata,created_at
  )
  SELECT c.id,c.source_id,c.collection_id,c.span_ids,c.text,
   left(s.title,512) || ' ' || left(s.filename,512),c.variant,
   c.embedding,c.metadata,c.created_at
  FROM public.chunks_before_indexed_search c
  JOIN sources s ON s.id=c.source_id
  WHERE s.collection_id=c.collection_id
   AND s.status='ready' AND s.deleted_at IS NULL;
  DROP TABLE public.chunks_before_indexed_search;
 END IF;
END $$;

GRANT SELECT ON public.chunks TO assistant_readonly;
