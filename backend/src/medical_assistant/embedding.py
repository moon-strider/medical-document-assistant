from bisect import bisect_right
from functools import lru_cache
from threading import Lock

from medical_assistant.settings import get_settings

_model_lock = Lock()


@lru_cache(maxsize=1)
def _tokenizer():
    return _model().tokenizer


@lru_cache(maxsize=1)
def _model():
    try:
        return _load_model(local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            "Pinned embedding files unavailable; run uv run --project . python -m medical_assistant.embedding"
        ) from exc


def _load_model(*, local_files_only):
    """Load the configured E5 encoder with the local ingestion resource policy.

    The pinned default is multilingual-e5-small, whose 384-dimensional output
    matches the stored vector schema. Inference uses CPU and the configured
    Torch thread count; inputs are truncated at 512 model tokens. This sets the
    process-wide Torch thread count as well as constructing a model.

    Args:
        local_files_only (bool): Whether loading must use cached artifacts.
            False permits the preparation command to download model files.

    Returns:
        SentenceTransformer: Encoder configured for CPU inference.
    """
    settings = get_settings()
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(settings.embedding_threads)
    model = SentenceTransformer(
        settings.embedding_model,
        revision=settings.embedding_revision,
        trust_remote_code=False,
        device="cpu",
        local_files_only=local_files_only,
    )
    model.max_seq_length = 512
    return model


def _encode(texts: list[str]) -> list[list[float]]:
    """Encode a prefixed batch for cosine retrieval in input order.

    A process-local lock serializes inference on the shared encoder. The model
    normalizes each vector to unit length and converts its coordinates to plain
    floats. The pinned default encoder produces 384 coordinates; this function
    does not independently enforce the storage dimension.

    Args:
        texts (list[str]): Model inputs already carrying E5 query or passage
            prefixes. The encoder may truncate inputs beyond 512 tokens.

    Returns:
        list[list[float]]: One normalized vector per input, in the same order;
            an empty input produces an empty list without loading the model.

    Raises:
        RuntimeError: Pinned local embedding files are unavailable.
    """
    if not texts:
        return []
    with _model_lock:
        vectors = _model().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return [[float(value) for value in vector] for vector in vectors]


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed source chunks in the E5 passage role for index publication.

    Args:
        texts (list[str]): Extracted chunk text in publication order. Each input
            receives the literal ``passage: `` prefix before encoding.

    Returns:
        list[list[float]]: Unit-normalized passage vectors in input order,
            384-dimensional with the pinned default model. Empty input returns
            an empty list; inputs over the encoder's 512-token limit truncate.

    Raises:
        RuntimeError: Pinned local embedding files are unavailable.
    """
    return _encode(["passage: " + text for text in texts])


def embed_query(text: str) -> list[float]:
    """Embed one search request in the E5 query role.

    Args:
        text (str): Search text; the literal ``query: `` prefix is added before
            encoding, including for empty text.

    Returns:
        list[float]: Unit-normalized query vector, 384-dimensional with the
            pinned default model, suitable for comparison with passage vectors.
            Inputs beyond the encoder's 512-token limit truncate.

    Raises:
        RuntimeError: Pinned local embedding files are unavailable.
    """
    return _encode(["query: " + text])[0]


def _mapped_windows(parts: list[tuple[str, str]], prefix: list[tuple[str, str]] | None = None):
    """Split evidence into token windows while retaining source-span provenance.

    Content lines are joined with newlines and tokenizer offsets map each
    window back to the spans it touches. Repeated header text consumes part of
    the 384-token content budget and its span IDs precede body IDs. Windows
    overlap by up to 32 body tokens; a span/byte boundary can reduce overlap.
    Adding tokens after the first is stopped at 64 distinct spans or 16,384
    bytes of full referenced source text, rather than bytes of the decoded
    excerpt. The first token and initial prefix can already exceed those caps.

    Args:
        parts (list[tuple[str, str]]): Ordered ``(text, span_id)`` body lines.
            Span IDs identify stable source records, not decoded token offsets.
        prefix (list[tuple[str, str]] | None): Header lines repeated in every
            window, with the same text/ID shape. None means no headers.

    Returns:
        list[tuple[str, list[str]]]: Decoded text and deduplicated source IDs in
            header-first, first-touch order. No body tokens yields no windows,
            even if a prefix exists. Decoding need not preserve original spacing.

    Raises:
        ValueError: Prefix tokens leave fewer than 64 body tokens, or a
            nonempty decoded window has no supporting source span.
        RuntimeError: Pinned local embedding files are unavailable.
    """
    prefix = prefix or []
    text = "\n".join(value for value, _ in parts)
    prefix_text = "".join(value + "\n" for value, _ in prefix)
    tokenizer = _tokenizer()
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    width = 384 - len(prefix_ids)
    if width < 64:
        raise ValueError("chunk_prefix_too_long")
    if not token_ids:
        return []
    ranges = []
    position = 0
    for value, span_id in parts:
        ranges.append((position, position + len(value), span_id))
        position += len(value) + 1
    starts = [line_start for line_start, _, _ in ranges]
    sizes = {span_id: len(value.encode("utf-8")) for value, span_id in parts + prefix}
    chunks = []
    start = 0
    while start < len(token_ids):
        span_ids = list(dict.fromkeys(span_id for _, span_id in prefix))
        seen = set(span_ids)
        source_bytes = sum(sizes[span_id] for span_id in seen)
        end = start
        while end < min(start + width, len(token_ids)):
            token_start, token_end = offsets[end]
            index = max(0, bisect_right(starts, token_start) - 1)
            added = []
            while index < len(ranges) and ranges[index][0] < token_end:
                line_start, line_end, span_id = ranges[index]
                if line_start < token_end and token_start < line_end and span_id not in seen:
                    added.append(span_id)
                index += 1
            if end > start and (
                len(seen) + len(added) > 64
                or source_bytes + sum(sizes[span_id] for span_id in added) > 16384
            ):
                break
            for span_id in added:
                span_ids.append(span_id)
                seen.add(span_id)
                source_bytes += sizes[span_id]
            end += 1
        piece = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        chunk_text = (prefix_text + piece).strip()
        if chunk_text and not span_ids:
            raise ValueError("chunk_without_source_span")
        chunks.append((chunk_text, span_ids))
        if end >= len(token_ids):
            break
        start = max(start + 1, end - 32)
    return chunks


def standalone_table_headers(spans: list[dict]) -> set[str]:
    """Recognize delimited identity/context lines preceding prose.

    A short pipe/tab line is treated as a document header when it begins a
    page/section region or explicitly names patient/subject, and the next
    nonempty line on the same page is prose. This is a layout heuristic, not a
    patient identity resolver or a general table-header classifier.

    Args:
        spans (list[dict]): Reading-order records with ``id`` and ``text``
            strings, ``page`` as int or None, and ``section`` as str. Blank
            records do not participate in adjacency checks.

    Returns:
        set[str]: IDs to carry as document context during structural chunking.
    """
    lines = [span for span in spans if span["text"].strip()]
    headers = set()
    for index, span in enumerate(lines):
        value = span["text"].strip()
        if len(value) > 200 or not ("|" in value or "\t" in value):
            continue
        key = (span["page"], span["section"])
        previous = lines[index - 1] if index else None
        following = lines[index + 1] if index + 1 < len(lines) else None
        fields = [field.strip().casefold() for field in value.replace("\t", "|").split("|")]
        explicit_identity = any(
            field in {"patient", "subject"} or field.startswith(("patient:", "subject:"))
            for field in fields
        )
        if (
            (
                previous is None
                or (previous["page"], previous["section"]) != key
                or explicit_identity
            )
            and following is not None
            and following["page"] == span["page"]
            and not ("|" in following["text"] or "\t" in following["text"])
        ):
            headers.add(span["id"])
    return headers


def uncovered_span_chunks(spans: list[dict], used_ids: set[str]):
    """Give each otherwise unrepresented nonempty span a retrieval window.

    Args:
        spans (list[dict]): Ordered source records with ``id`` and ``text``
            strings, including headers that structural processing may only carry
            as context or may replace before emitting a chunk.
        used_ids (set[str]): IDs already referenced by emitted chunks.

    Returns:
        list[tuple[str, list[str]]]: Standalone token windows for uncovered
            spans, in source order, with no repeated document/section prefix.
    """
    chunks = []
    for span in spans:
        if span["text"].strip() and span["id"] not in used_ids:
            chunks.extend(_mapped_windows([(span["text"], span["id"])]))
    return chunks


def structural_chunks_with_context(
    spans: list[dict],
    document_header: list[tuple[str, str]],
    section_header: list[tuple[str, str]],
    table_header: list[tuple[str, str]],
    standalone_headers: set[str],
    *,
    complete: bool = True,
):
    """Build structural evidence windows and return context for the next block.

    Prose is grouped under the current document and section headers. A heading
    flushes prose and replaces section context; a standalone identity header
    replaces document context. Pipe/tab rows are emitted individually: the
    first row establishes table-header context and later rows repeat it, keeping
    column labels attached to their values. These are textual heuristics;
    supplied row order is preserved and cells are not semantically reclassified.

    Args:
        spans (list[dict]): Reading-order records with ``id`` and ``text``
            strings. Blank text is skipped.
        document_header (list[tuple[str, str]]): Active ``(text, span_id)``
            document context carried from a preceding block.
        section_header (list[tuple[str, str]]): Active section context in the
            same shape.
        table_header (list[tuple[str, str]]): Active first table row in the same
            shape, allowing a table to continue across block boundaries.
        standalone_headers (set[str]): IDs recognized as document-context rows.
        complete (bool): Whether to append standalone windows for nonempty
            input spans absent from every emitted chunk. False lets ingestion
            perform this coverage pass across all blocks instead.

    Returns:
        tuple[list[tuple[str, list[str]]], list[tuple[str, str]],
            list[tuple[str, str]], list[tuple[str, str]]]: Windows with source
            IDs, followed by the final document, section and table context.
            Use these returned lists for continuation; inputs are not mutated.

    Raises:
        ValueError: Repeated context leaves too little token budget, or a window
            has text without a supporting span.
    """
    lines = [(span["text"], span["id"]) for span in spans if span["text"].strip()]
    chunks = []
    prose = []

    def flush():
        if prose:
            chunks.extend(_mapped_windows(prose, document_header + section_header))
            prose.clear()

    for value, span_id in lines:
        stripped = value.strip()
        line = (stripped, span_id)
        table = "|" in stripped or "\t" in stripped
        if span_id in standalone_headers:
            flush()
            document_header = [line]
            section_header = []
            table_header = []
        elif not table and (
            (stripped.startswith("#") and len(stripped) < 200)
            or (stripped.endswith(":") and len(stripped) < 200)
            or (stripped.isupper() and len(stripped) < 120)
        ):
            flush()
            section_header = [line]
            table_header = []
        elif table:
            flush()
            if not table_header:
                table_header = [line]
                chunks.extend(_mapped_windows([line], document_header + section_header))
            else:
                chunks.extend(
                    _mapped_windows([line], document_header + section_header + table_header)
                )
        else:
            table_header = []
            prose.append(line)
    flush()
    if complete:
        chunks.extend(uncovered_span_chunks(spans, {sid for _, ids in chunks for sid in ids}))
    return chunks, document_header, section_header, table_header


def token_chunks_with_spans(spans: list[dict], variant: str):
    """Produce either baseline or contextual windows from ordered evidence.

    Fixed chunking joins nonempty spans into overlapping token windows without
    headers. Structural chunking recognizes headings and table rows, repeats
    supporting header spans, and adds windows for any uncovered nonempty span.
    Both use the 384-token window policy with up to 32 body tokens of overlap.

    Args:
        spans (list[dict]): Source records with ``id`` and ``text`` strings;
            structural detection also requires ``page`` (int or None) and
            ``section`` (str).
        variant (str): Exactly ``fixed`` or ``structural``.

    Returns:
        list[tuple[str, list[str]]]: Chunk text paired with supporting source
            IDs. An input without nonempty body text yields an empty list.

    Raises:
        ValueError: The variant is unsupported or mapped-window validation fails.
        RuntimeError: Pinned local embedding files are unavailable.
    """
    if variant not in {"fixed", "structural"}:
        raise ValueError("invalid_chunk_variant")
    lines = [(span["text"], span["id"]) for span in spans if span["text"].strip()]
    if variant == "fixed":
        return _mapped_windows(lines)
    chunks, _, _, _ = structural_chunks_with_context(
        spans, [], [], [], standalone_table_headers(spans)
    )
    return chunks


if __name__ == "__main__":
    _load_model(local_files_only=False)
    print("Pinned embedding model prepared in the local Hugging Face cache.")
