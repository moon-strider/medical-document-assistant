import math
from functools import lru_cache
from threading import Lock
from time import perf_counter

from medical_assistant.settings import get_settings

_model_lock = Lock()
_max_pairs = 64
_batch_size = 8
_max_length = 512


def _load_model(*, local_files_only: bool):
    settings = get_settings()
    import torch
    from sentence_transformers import CrossEncoder

    if settings.reranker_threads < 1 or settings.reranker_threads > 8:
        raise ValueError("invalid_reranker_threads")
    torch.set_num_threads(settings.reranker_threads)
    return CrossEncoder(
        settings.reranker_model,
        revision=settings.reranker_revision,
        trust_remote_code=False,
        device="cpu",
        local_files_only=local_files_only,
        max_length=_max_length,
        model_kwargs={"use_safetensors": True},
    )


@lru_cache(maxsize=1)
def _model():
    try:
        return _load_model(local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            "Pinned reranker files unavailable; run uv run --project . python -m medical_assistant.reranking"
        ) from exc


def rerank_candidates(query: str, candidates: list[dict], limit: int = 12) -> dict:
    """Select the most relevant passages from an already bounded retrieval pool.

    The pinned local CPU cross-encoder scores query/passage pairs jointly, so
    relevance can change from the independent dense/BM25 fusion order. It cannot
    discover evidence omitted by candidate retrieval. Prediction is serialized
    by a process-local lock, uses batches of eight and a 512-token model window;
    longer tokenized pairs are scored after model truncation, not rejected.

    Args:
        query (str): Nonblank question of at most 512 characters, passed to the
            model without whitespace normalization.
        candidates (list[dict]): At most 64 records with nonblank ``text`` of at
            most 16,384 UTF-8 bytes and ``chunk_id`` (str) for tie-breaking.
            Optional ``score`` is the numeric fusion score, defaulting to zero.
            Other source/span/snippet metadata passes through unchanged.
        limit (int): Maximum returned candidates, from 1 through 12.

    Returns:
        dict: ``candidates`` are new dicts without ``text``, with finite float
        ``rerank_score``, ordered by decreasing rerank score, decreasing fusion
        score, then ascending chunk ID. Input records are not mutated.
        ``rerank`` reports status, pinned model/revision, input/scored counts,
        the count of pairs exceeding 512 tokens, and elapsed milliseconds.
        Empty input returns ``skipped_empty`` without loading/scoring a model.
        Nonempty calls may populate the process model cache; no files are
        downloaded and no application data is written.

    Raises:
        ValueError: Invalid query, pool size, limit, candidate text, or configured
            thread count; nonnumeric fusion scores also propagate conversion errors.
        RuntimeError: Pinned model files are unavailable, prediction count differs
            from the pool, or a model score is nonfinite.
    """
    if (
        not query.strip()
        or len(query) > 512
        or len(candidates) > _max_pairs
        or limit < 1
        or limit > 12
    ):
        raise ValueError("invalid_rerank_request")
    settings = get_settings()
    if not candidates:
        return {
            "candidates": [],
            "rerank": {
                "status": "skipped_empty",
                "model": settings.reranker_model,
                "revision": settings.reranker_revision,
                "input_count": 0,
                "scored_pairs": 0,
                "truncated_pairs": 0,
                "elapsed_ms": 0.0,
            },
        }
    if any(
        not isinstance(item.get("text"), str)
        or not item["text"].strip()
        or len(item["text"].encode("utf-8")) > 16384
        for item in candidates
    ):
        raise ValueError("candidate_text_required")
    pairs = [(query, item["text"]) for item in candidates]
    start = perf_counter()
    with _model_lock:
        model = _model()
        pair_lengths = model.tokenizer(
            [query] * len(pairs),
            [item["text"] for item in candidates],
            truncation=False,
            return_length=True,
        )["length"]
        scores = model.predict(
            pairs,
            batch_size=_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
        )
    if len(scores) != len(candidates):
        raise RuntimeError("reranker_score_count_mismatch")
    ranked = []
    for item, value in zip(candidates, scores, strict=True):
        score = float(value)
        if not math.isfinite(score):
            raise RuntimeError("reranker_nonfinite_score")
        ranked.append(
            {**{key: value for key, value in item.items() if key != "text"}, "rerank_score": score}
        )
    ranked.sort(
        key=lambda item: (-item["rerank_score"], -float(item.get("score", 0)), item["chunk_id"])
    )
    return {
        "candidates": ranked[:limit],
        "rerank": {
            "status": "ok",
            "model": settings.reranker_model,
            "revision": settings.reranker_revision,
            "input_count": len(candidates),
            "scored_pairs": len(pairs),
            "truncated_pairs": sum(length > _max_length for length in pair_lengths),
            "elapsed_ms": round((perf_counter() - start) * 1000, 3),
        },
    }


if __name__ == "__main__":
    from medical_assistant.embedding import _load_model as _load_embedding_model

    _load_embedding_model(local_files_only=False)
    _load_model(local_files_only=False)
    print("Pinned embedding and reranker models prepared in the local Hugging Face cache.")
