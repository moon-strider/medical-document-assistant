CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin(search_tsv);
