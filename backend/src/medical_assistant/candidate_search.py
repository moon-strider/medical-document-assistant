import hashlib
import re
from time import perf_counter
from uuid import UUID

from sqlalchemy import text

_IDENTIFIER = re.compile(
    r"10\.[0-9]{4,9}/[A-Za-z0-9._;()/:+-]+|"
    r"[A-Za-z][A-Za-z0-9]*[-./:][A-Za-z0-9/._:-]+|"
    r"[A-Za-z]{2,}[0-9][A-Za-z0-9]*"
)
_DOI = re.compile(r"10\.[0-9]{4,9}/[A-Za-z0-9._;()/:+-]+")
_TRIAL = re.compile(r"NCT[0-9]{8}(?![0-9])", re.IGNORECASE)
_PHRASE = re.compile(r'"([^"\n]{3,120})"')
_BRANCH_LIMIT = 100
_LITERAL_LIMIT = 20
_RRF_LIMIT = 64


def _identifiers(value: str) -> list[str]:
    """Extract identifier-like query terms for a dedicated literal search lane.

    Recognizes DOI/trial forms and digit-bearing structured tokens, lowercases
    them and removes trailing punctuation/unbalanced closing parentheses. This
    supports document identifiers rather than authoritative patient resolution.

    Args:
        value (str): Search question, which may also contain ordinary prose.

    Returns:
        list[str]: Up to eight distinct normalized terms, longest first and
        lexicographically ordered within equal lengths. No matches yields [].
    """

    def normalize(token: str) -> str:
        token = token.lower().rstrip(".,;:")
        while token.endswith(")") and token.count(")") > token.count("("):
            token = token[:-1]
        return token

    return sorted(
        {
            normalize(match.group())
            for pattern in (_IDENTIFIER, _DOI, _TRIAL)
            for match in pattern.finditer(value)
            if any(character.isdigit() for character in match.group())
        },
        key=lambda term: (-len(term), term),
    )[:8]


def _prefix(value: str, budget: int) -> str:
    return value.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def _ids(connection, sql: str, params: dict) -> list[str]:
    return [str(value) for value in connection.execute(text(sql), params).scalars()]


def _interleave(lanes: list[list[str]], limit: int) -> list[str]:
    """Share bounded candidate slots across ordered evidence lanes.

    Args:
        lanes (list[list[str]]): Ranked chunk IDs, in lane priority order.
        limit (int): Maximum distinct IDs to select.

    Returns:
        list[str]: Round-robin first occurrences, preserving each lane's relative
        order while skipping duplicates. Empty/exhausted lanes can return fewer
        than ``limit`` IDs. Input lanes are not changed.
    """
    selected = []
    seen = set()
    positions = [0] * len(lanes)
    while len(selected) < limit:
        advanced = False
        for index, lane in enumerate(lanes):
            if positions[index] == len(lane):
                continue
            item = lane[positions[index]]
            positions[index] += 1
            advanced = True
            if item not in seen:
                selected.append(item)
                seen.add(item)
            if len(selected) == limit:
                break
        if not advanced:
            break
    return selected


def _literal_ids(connection, partition: str, query: str, terms: list[str]) -> list[str]:
    """Find bounded exact identifier matches without letting one source dominate.

    Each normalized identifier becomes a synthetic BM25 token via SHA-256,
    matching ingestion's identifier index representation. The hash is an index
    term, not a source integrity check. At most 100 index-ordered chunks per term
    are inspected; an exact identifier-token check rejects spurious matches.
    English query relevance orders matches, then round-robin selection shares
    slots across sources and identifier terms. Missing results do not prove
    absence outside these candidate pools.

    Args:
        connection (sqlalchemy.engine.Connection): Active database transaction.
        partition (str): Trusted collection-specific structural chunk table name.
        query (str): Original question used to rank verified matches.
        terms (list[str]): Normalized identifier terms from ``_identifiers``.

    Returns:
        list[str]: Up to 20 distinct chunk UUID strings in interleaved order.
        Database failures propagate; no persistent data is modified.
    """
    term_lanes = []
    bm25_index = f"{partition}_bm25"
    for term in terms:
        encoded = "pflid" + hashlib.sha256(term.encode("utf-8")).hexdigest()
        rows = connection.execute(
            text(
                "WITH id_pool AS MATERIALIZED ("
                "SELECT c.id, c.source_id, c.identifier_tokens, c.search_tsv, "
                f"c.search_text <@> to_bm25query(:encoded, '{bm25_index}') AS id_score "
                f"FROM {partition} c WHERE c.variant = 'structural' "
                f"ORDER BY c.search_text <@> to_bm25query(:encoded, '{bm25_index}') "
                "LIMIT 100) "
                "SELECT id, source_id FROM id_pool "
                "WHERE id_score < 0 AND identifier_tokens @> CAST(:term AS text[]) "
                "ORDER BY ts_rank_cd(search_tsv, websearch_to_tsquery('english', :query)) DESC, "
                "source_id, id"
            ),
            {"term": [term], "encoded": encoded, "query": query},
        ).all()
        groups = {}
        for chunk_id, source_id in rows:
            groups.setdefault(str(source_id), []).append(str(chunk_id))
        term_lanes.append(_interleave(list(groups.values()), _BRANCH_LIMIT))
    return _interleave(term_lanes, _LITERAL_LIMIT)


def _pool(branches: dict[str, list[str]], limit: int) -> tuple[list[str], dict[str, dict]]:
    """Fuse unlike retrieval scores while protecting literal/phrase candidates.

    Reciprocal rank fusion adds ``1 / (60 + rank)`` for each lane appearance,
    avoiding a comparison between raw cosine and BM25 scores. Up to 20 slots
    at a 64-candidate limit, otherwise up to four slots, are reserved by
    interleaving literal and phrase lanes. Remaining slots follow fusion order;
    the final selected pool is again ordered by decreasing fusion score and
    ascending chunk ID. Protected candidates can displace higher fusion scores.

    Args:
        branches (dict[str, list[str]]): Ranked unique chunk IDs per search lane.
        limit (int): Pool capacity: 64 before V3 reranking, otherwise the requested
            output limit. Empty branches yield an empty pool.

    Returns:
        tuple[list[str], dict[str, dict]]: Selected IDs and fusion metadata for
        every encountered ID, including float ``score`` and contributing
        ``branches`` (list[str]). Neither input lanes nor their scores are changed.
    """
    fused = {}
    for branch, ids in branches.items():
        for rank, chunk_id in enumerate(ids, 1):
            item = fused.setdefault(chunk_id, {"score": 0.0, "branches": []})
            item["score"] += 1 / (60 + rank)
            item["branches"].append(branch)
    ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id]["score"], chunk_id))
    protected = _interleave(
        [branches.get("literal", []), branches.get("phrase", [])],
        min(limit, 20 if limit == _RRF_LIMIT else 4),
    )
    selected = protected[:]
    selected_set = set(selected)
    for chunk_id in ordered:
        if len(selected) >= limit:
            break
        if chunk_id not in selected_set:
            selected.append(chunk_id)
            selected_set.add(chunk_id)
    selected.sort(key=lambda chunk_id: (-fused[chunk_id]["score"], chunk_id))
    return selected, fused


def search_candidates(
    connection,
    collection_id: str,
    query: str,
    variant: str,
    limit: int = 12,
    *,
    query_vector: list[float],
) -> dict:
    """Retrieve a collection-local pool from complementary evidence search lanes.

    V0 searches fixed chunks by approximate HNSW cosine distance. V1/V2/V3
    search structural chunks using semantic distance and indexed English BM25;
    digit-bearing identifiers and up to two quoted phrases add verified match
    lanes. Dense and BM25 retrieval each inspect at most 100 chunks. Identifier
    and phrase verification also starts from at most 100 BM25 candidates per
    term/phrase, with at most 20 retained per lane. Phrase matching follows
    English text-search token semantics, not raw substring equality.

    Fusion protects some literal/phrase slots and produces at most ``limit``
    candidates for V0/V1/V2 or 64 for V3's downstream reranker. Approximate search
    and every lane's cap can omit valid evidence; empty results cannot establish
    archive-wide absence. This function checks selected sources are ready,
    undeleted and in the collection, but the caller must validate run scope and
    provide a consistent transaction.

    Args:
        connection (sqlalchemy.engine.Connection): Active PostgreSQL transaction.
            Transaction-local HNSW search settings are changed; no rows are written.
        collection_id (str): Collection UUID selecting its own chunk partition.
        query (str): Question, stripped before search; 1 through 512 characters
            after stripping, not a UTF-8 byte limit.
        variant (str): V0, V1, V2 or V3; V1 and V2 have identical candidate search.
        limit (int): Requested final size from 1 through 12; V3 instead returns
            up to 64 candidates for subsequent reranking.
        query_vector (list[float]): Precomputed 384-dimensional query embedding.

    Returns:
        dict: Normalized ``query``, ``variant``, ``no_candidates`` and ordered
        ``candidates``. Each candidate has string chunk/source IDs, source title
        and original-file SHA-256 ``source_hash``, ordered ``span_ids`` (list[str]),
        a UTF-8-safe ``snippet`` of at most 512 bytes, float RRF ``score`` and
        contributing ``branches``. V3 also includes full chunk ``text`` for the
        reranker. ``retrieval`` records lane counts, selected ``rrf_pool_size``,
        collection-partition scope and stage/total timings in milliseconds;
        its embedding stage is zero because embedding is supplied by the caller.

    Raises:
        ValueError: Invalid query/variant/limit, malformed collection UUID, or
            an embedding dimension other than 384.
        RuntimeError: A selected chunk or its ready collection source cannot
            be resolved (``search_index_scope_mismatch``). Database errors propagate.
    """
    started = perf_counter()
    query = query.strip()
    if not query or len(query) > 512 or variant not in {"V0", "V1", "V2", "V3"}:
        raise ValueError("invalid_search")
    if limit < 1 or limit > 12:
        raise ValueError("invalid_search_limit")
    collection_id = str(UUID(collection_id))
    partition = f"chunks_c{UUID(collection_id).hex}"
    chunk_variant = "fixed" if variant == "V0" else "structural"
    timings = {name: 0.0 for name in ("embedding", "dense", "bm25", "literal", "phrase", "fusion")}
    if len(query_vector) != 384:
        raise ValueError("embedding_dimension")
    vector_literal = "[" + ",".join(str(value) for value in query_vector) + "]"
    connection.execute(text("SET LOCAL hnsw.ef_search = 200"))
    connection.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))

    stage = perf_counter()
    dense = _ids(
        connection,
        f"SELECT c.id FROM {partition} c WHERE c.variant = '{chunk_variant}' "
        "ORDER BY c.embedding <=> CAST(:vector AS vector) LIMIT 100",
        {"vector": vector_literal},
    )
    timings["dense"] = round((perf_counter() - stage) * 1000, 3)
    branches = {"dense": dense}

    if variant != "V0":
        stage = perf_counter()
        bm25_index = f"{partition}_bm25"
        bm25_rows = connection.execute(
            text(
                f"SELECT c.id, c.search_text <@> to_bm25query(:query, '{bm25_index}') AS bm25_score "
                f"FROM {partition} c WHERE c.variant = 'structural' "
                f"ORDER BY c.search_text <@> to_bm25query(:query, '{bm25_index}') LIMIT 100"
            ),
            {"query": query},
        ).all()
        bm25 = [str(row[0]) for row in bm25_rows if row[1] < 0]
        timings["bm25"] = round((perf_counter() - stage) * 1000, 3)
        branches["bm25"] = bm25

        stage = perf_counter()
        terms = _identifiers(query)
        literal = _literal_ids(connection, partition, query, terms) if terms else []
        timings["literal"] = round((perf_counter() - stage) * 1000, 3)
        branches["literal"] = literal

        stage = perf_counter()
        phrase_lanes = []
        for quoted in _PHRASE.findall(query)[:2]:
            matches = _ids(
                connection,
                "WITH phrase_pool AS MATERIALIZED ("
                "SELECT c.id, c.search_tsv, "
                f"c.search_text <@> to_bm25query(:phrase, '{bm25_index}') AS phrase_score "
                f"FROM {partition} c WHERE c.variant = 'structural' "
                f"ORDER BY c.search_text <@> to_bm25query(:phrase, '{bm25_index}') "
                "LIMIT 100) "
                "SELECT id FROM phrase_pool "
                "WHERE phrase_score < 0 "
                "AND search_tsv @@ phraseto_tsquery('english', :phrase) "
                "ORDER BY ts_rank_cd(search_tsv, "
                "phraseto_tsquery('english', :phrase)) DESC, id LIMIT 20",
                {"phrase": quoted},
            )
            phrase_lanes.append(matches)
        phrase = _interleave(phrase_lanes, _LITERAL_LIMIT)
        timings["phrase"] = round((perf_counter() - stage) * 1000, 3)
        branches["phrase"] = phrase

    stage = perf_counter()
    pool_limit = _RRF_LIMIT if variant == "V3" else limit
    selected, fused = _pool(branches, pool_limit)
    if selected:
        rows = (
            connection.execute(
                text(
                    f"SELECT c.id, c.source_id, c.span_ids, c.text FROM {partition} c "
                    "WHERE c.collection_id = CAST(:collection_id AS uuid) "
                    "AND c.id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"ids": selected, "collection_id": collection_id},
            )
            .mappings()
            .all()
        )
        found = {str(row["id"]): row for row in rows}
        if set(selected) != set(found):
            raise RuntimeError("search_index_scope_mismatch")
        sources = (
            connection.execute(
                text(
                    "SELECT id, collection_id, title, sha256, status, deleted_at "
                    "FROM sources WHERE id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"ids": list({str(row["source_id"]) for row in rows})},
            )
            .mappings()
            .all()
        )
        source_by_id = {str(row["id"]): row for row in sources}
        candidates = []
        for chunk_id in selected:
            row = found[chunk_id]
            source = source_by_id.get(str(row["source_id"]))
            if (
                source is None
                or str(source["collection_id"]) != collection_id
                or source["status"] != "ready"
                or source["deleted_at"] is not None
            ):
                raise RuntimeError("search_index_scope_mismatch")
            full_text = row["text"]
            candidate = {
                "chunk_id": chunk_id,
                "source_id": str(row["source_id"]),
                "source_title": source["title"],
                "source_hash": source["sha256"],
                "span_ids": [str(item) for item in row["span_ids"]],
                "snippet": _prefix(full_text, 512),
                "score": fused[chunk_id]["score"],
                "branches": fused[chunk_id]["branches"],
            }
            if variant == "V3":
                candidate["text"] = full_text
            candidates.append(candidate)
    else:
        candidates = []
    timings["fusion"] = round((perf_counter() - stage) * 1000, 3)
    result = {
        "query": query,
        "variant": variant,
        "candidates": candidates,
        "no_candidates": not candidates,
        "retrieval": {
            "dense_candidates": len(branches["dense"]),
            "bm25_candidates": len(branches.get("bm25", [])),
            "literal_candidates": len(branches.get("literal", [])),
            "phrase_candidates": len(branches.get("phrase", [])),
            "rrf_pool_size": len(selected),
            "query_stage_ms": timings,
            "index_scope": "collection_partition",
        },
    }
    result["retrieval"]["search_ms"] = round((perf_counter() - started) * 1000, 3)
    return result
