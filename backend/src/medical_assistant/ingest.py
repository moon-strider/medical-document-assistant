import hashlib
import json
import os
import resource
import selectors
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from medical_assistant.errors import TransientIngestError

MAX_SPANS = 200000
MAX_PARSER_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SPAN_TEXT_BYTES = 4096


def _source_path(settings, file_key):
    root = (Path(settings.data_dir) / "uploads").resolve()
    candidate = root / file_key
    current = root
    for component in Path(file_key).parts:
        current /= component
        if current.is_symlink():
            raise ValueError("Source path contains a symlink")
    target = candidate.resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Invalid source path")
    return target


def _line(text, page, number, bbox, width, height, section):
    """Create one source span with the coordinates needed to reopen evidence.

    Args:
        text (str): Extracted line or serialized table row; outer whitespace is
            stripped while internal cell separators are retained.
        page (int | None): One-based PDF page, or None for TXT.
        number (int): One-based source line number; copied to both line bounds.
        bbox (list[float] | None): PDF ``[x0, top, x1, bottom]`` rectangle in
            page coordinate units, or None for TXT.
        width (float | None): PDF page width in the same units, or None for TXT.
        height (float | None): PDF page height in the same units, or None for TXT.
        section (str): Current textual heading, or an empty string.

    Returns:
        dict: New UUID-string ``id``, ``text``, ``page``, ``line_start``,
            ``line_end``, ``bbox``, ``page_width``, ``page_height`` and ``section``.
            Document-wide ordinal and registered source hash are added later.
    """
    value = text.strip()
    return {
        "id": str(uuid.uuid4()),
        "page": page,
        "line_start": number,
        "line_end": number,
        "text": value,
        "bbox": bbox,
        "page_width": width,
        "page_height": height,
        "section": section,
    }


def _parse_txt(path, max_bytes):
    """Extract strict UTF-8 text into line-addressable evidence without OCR.

    UTF-8 BOM is accepted. Nonblank original lines retain their one-based file
    line numbers, including gaps caused by blank lines. Hash-prefixed headings
    update section context and remain evidence themselves. TXT has a logical
    page count of one but no PDF page coordinates on its spans.

    Args:
        path (Path): Original file to read in full.
        max_bytes (int): Maximum permitted original file size in bytes.

    Returns:
        dict: ``spans`` as ordered line records from ``_line`` and
            ``page_count`` equal to 1. Text is trimmed at each line's edges.

    Raises:
        ValueError: File size exceeds the upload limit, there are over 200,000
            original lines, a line exceeds 65,536 UTF-8 bytes, or no nonblank
            text can be extracted.
        UnicodeDecodeError: Original bytes are not valid UTF-8.
        OSError: Original bytes cannot be read.
    """
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("Text file exceeds upload limit")
    content = data.decode("utf-8-sig", errors="strict")
    lines = content.splitlines()
    if len(lines) > MAX_SPANS:
        raise ValueError("Text file has too many lines")
    spans = []
    section = ""
    for number, raw in enumerate(lines, 1):
        if len(raw.encode("utf-8")) > 65536:
            raise ValueError("Text line is too long")
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            section = raw.lstrip("# ").strip()[:200]
        spans.append(_line(raw, None, number, None, None, None, section))
    if not spans:
        raise ValueError("Text file has no extractable text")
    return {"spans": spans, "page_count": 1}


def _pdf_piece(pairs, text=None):
    """Join PDF words into one evidence piece while keeping word provenance.

    Words are grouped into rows when their tops differ by at most three page
    units, then ordered left to right within each row. A caller-supplied text
    value can preserve table cell separators independently of this word order.

    Args:
        pairs (list[tuple[int, dict]]): Nonempty word-index/word records;
            words require ``text``, ``x0``, ``x1``, ``top`` and ``bottom``.
        text (str | None): Serialized row text, or None to join word text with
            spaces in the resulting geometric order.

    Returns:
        dict: Ordered ``pairs``, matching integer ``word_ids``, ``text``, and
            enclosing ``bbox`` as ``[x0, top, x1, bottom]`` in page units.
    """
    by_top = sorted(pairs, key=lambda pair: (float(pair[1]["top"]), float(pair[1]["x0"])))
    rows = []
    for pair in by_top:
        top = float(pair[1]["top"])
        if not rows or top - rows[-1][0] > 3:
            rows.append((top, [pair]))
        else:
            rows[-1][1].append(pair)
    ordered = [
        pair for _, row in rows for pair in sorted(row, key=lambda item: float(item[1]["x0"]))
    ]
    words = [word for _, word in ordered]
    return {
        "pairs": ordered,
        "word_ids": [index for index, _ in ordered],
        "text": text if text is not None else " ".join(str(word["text"]) for word in words),
        "bbox": [
            min(float(word["x0"]) for word in words),
            min(float(word["top"]) for word in words),
            max(float(word["x1"]) for word in words),
            max(float(word["bottom"]) for word in words),
        ],
    }


def _pdf_rows(pieces):
    """Group evidence pieces into geometric rows for table/layout decisions.

    Args:
        pieces (list[dict]): Pieces carrying ``bbox`` as
            ``[x0, top, x1, bottom]`` in PDF page units.

    Returns:
        list[list[dict]]: Top-to-bottom rows, each ordered left to right; pieces
            whose tops differ from the row anchor by at most three units share
            that row. Empty input returns an empty list.
    """
    ordered = sorted(pieces, key=lambda piece: (piece["bbox"][1], piece["bbox"][0]))
    rows = []
    for piece in ordered:
        if not rows or abs(piece["bbox"][1] - rows[-1][0]) > 3:
            rows.append((piece["bbox"][1], [piece]))
        else:
            rows[-1][1].append(piece)
    return [sorted(row, key=lambda piece: piece["bbox"][0]) for _, row in rows]


def _pdf_grid_tables(page, words, page_number):
    """Serialize supported ruled tables without losing column position.

    Word centers assign text to detected table cells. Rows retain the detector's
    cell order and use `` | `` separators, including empty cells, so values stay
    aligned with their column labels. Each nonempty row keeps the union of its
    word IDs and bounding box for later coverage checks and source highlighting.

    Args:
        page (pdfplumber.page.Page): Page with edges and table detection support.
        words (list[dict]): Extracted word records with text and PDF coordinates.
        page_number (int): One-based page used in rejection messages.

    Returns:
        tuple[list[dict], set[int]]: Table blocks with ``top``, ``bottom``,
            ``kind='table'`` and row ``pieces``, plus word indexes consumed by
            those tables. Pages without edges yield no detected grid tables.

    Raises:
        ValueError: More than 512 edges, 16 tables or 1,024 detected rows;
            overlapping tables/cells; merged cells; or table words that cannot
            be assigned completely to nonempty rows.
    """
    if len(page.edges) > 512:
        raise ValueError(f"PDF page {page_number} has unsupported table geometry")
    tables = page.find_tables() if page.edges else []
    if len(tables) > 16 or sum(len(table.rows) for table in tables) > 1024:
        raise ValueError(f"PDF page {page_number} has too many table rows")
    blocks = []
    used = set()
    for table in tables:
        table_ids = {
            index
            for index, word in enumerate(words)
            if table.bbox[0] <= (float(word["x0"]) + float(word["x1"])) / 2 <= table.bbox[2]
            and table.bbox[1] <= (float(word["top"]) + float(word["bottom"])) / 2 <= table.bbox[3]
        }
        if used & table_ids:
            raise ValueError(f"PDF page {page_number} has overlapping tables")
        assigned = set()
        pieces = []
        for row in table.rows:
            cells = []
            row_pairs = []
            for cell in row.cells:
                if cell is None:
                    raise ValueError(f"PDF page {page_number} has unsupported merged table cells")
                pairs = [
                    (index, words[index])
                    for index in table_ids
                    if cell[0]
                    <= (float(words[index]["x0"]) + float(words[index]["x1"])) / 2
                    < cell[2]
                    and cell[1]
                    <= (float(words[index]["top"]) + float(words[index]["bottom"])) / 2
                    < cell[3]
                ]
                ids = {index for index, _ in pairs}
                if assigned & ids:
                    raise ValueError(f"PDF page {page_number} has overlapping table cells")
                assigned.update(ids)
                row_pairs.extend(pairs)
                cells.append(_pdf_piece(pairs)["text"] if pairs else "")
            if row_pairs:
                pieces.append(_pdf_piece(row_pairs, " | ".join(cells)))
        if assigned != table_ids or not pieces:
            raise ValueError(f"PDF page {page_number} has incomplete table cells")
        used.update(table_ids)
        blocks.append(
            {"top": table.bbox[1], "bottom": table.bbox[3], "kind": "table", "pieces": pieces}
        )
    return blocks, used


def _pdf_textlines(page, words, excluded, page_number):
    """Recover non-table text lines with explicit ownership of every word.

    Each remaining word center must belong to exactly one detected horizontal
    text line, allowing a 1.5-unit margin. Large horizontal gaps split a line
    into pieces so later layout logic can distinguish prose columns from rows.

    Args:
        page (pdfplumber.page.Page): Page exposing horizontal text-line objects
            and width in PDF page units.
        words (list[dict]): Word records with text and PDF coordinates.
        excluded (set[int]): Word indexes already consumed by ruled tables.
        page_number (int): One-based page used in rejection messages.

    Returns:
        list[dict]: Provenance-bearing pieces split at gaps greater than the
            larger of 100 units or one sixth of page width. This result is not
            yet the final page reading order.

    Raises:
        ValueError: A remaining word has zero or multiple text-line owners.
    """
    textlines = page.objects.get("textlinehorizontal", [])
    groups = {}
    for index, word in enumerate(words):
        if index in excluded:
            continue
        cx = (float(word["x0"]) + float(word["x1"])) / 2
        cy = (float(word["top"]) + float(word["bottom"])) / 2
        owners = [
            number
            for number, line in enumerate(textlines)
            if float(line["x0"]) - 1.5 <= cx <= float(line["x1"]) + 1.5
            and float(line["top"]) - 1.5 <= cy <= float(line["bottom"]) + 1.5
        ]
        if len(owners) != 1:
            raise ValueError(f"PDF page {page_number} has unsupported text layout")
        groups.setdefault(owners[0], []).append((index, word))
    pieces = []
    for pairs in groups.values():
        ordered = sorted(pairs, key=lambda pair: float(pair[1]["x0"]))
        segment = [ordered[0]]
        for left, right in zip(ordered, ordered[1:]):
            if float(right[1]["x0"]) - float(left[1]["x1"]) > max(100, float(page.width) / 6):
                pieces.append(_pdf_piece(segment))
                segment = []
            segment.append(right)
        pieces.append(_pdf_piece(segment))
    return pieces


def _pdf_borderless_table(rows):
    """Apply the conservative geometry/content heuristic for unruled tables.

    Recognition requires at least three rows and columns, equal column counts,
    left-edge alignment within 14 page units, short nondigit column labels in
    the first row, and a digit somewhere below. Column text must also leave
    sufficient horizontal separation. This avoids treating ambiguous aligned
    prose as a table; it is not a semantic table or medical-value classifier.

    Args:
        rows (list[list[dict]]): Top-to-bottom rows of left-to-right pieces,
            each with ``text`` and ``bbox`` in page units.

    Returns:
        bool: True only if every heuristic condition holds; False otherwise.
    """
    if len(rows) < 3 or len(rows[0]) < 3:
        return False
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        return False
    if any(
        max(row[column]["bbox"][0] for row in rows) - min(row[column]["bbox"][0] for row in rows)
        > 14
        for column in range(count)
    ):
        return False
    if any(
        len(piece["text"].split()) > 3 or any(char.isdigit() for char in piece["text"])
        for piece in rows[0]
    ):
        return False
    if not any(any(char.isdigit() for char in piece["text"]) for row in rows[1:] for piece in row):
        return False
    return all(
        max(row[column]["bbox"][2] - row[column]["bbox"][0] for row in rows)
        < 0.8 * min(row[column + 1]["bbox"][0] - row[column]["bbox"][0] for row in rows)
        for column in range(count - 1)
    )


def _pdf_borderless_blocks(pieces):
    """Lift contiguous aligned rows into table blocks before prose ordering.

    Candidate regions have at least three pieces per row, a constant column
    count, and no more than 30 page units between consecutive row tops. Regions
    passing the borderless-table heuristic become pipe-separated row pieces,
    preserving left-to-right columns and all word IDs. Their original pieces
    are removed from the residual prose list.

    Args:
        pieces (list[dict]): Non-grid text pieces with ``pairs``, ``text`` and
            ``bbox`` from PDF extraction.

    Returns:
        tuple[list[dict], list[dict]]: Table blocks containing geometry and row
            pieces, followed by unconsumed original pieces in input order.
            Neither input records nor the input list are mutated.
    """
    rows = _pdf_rows(pieces)
    blocks = []
    consumed = set()
    index = 0
    while index < len(rows):
        count = len(rows[index])
        end = index + 1
        while (
            count >= 3
            and end < len(rows)
            and len(rows[end]) == count
            and rows[end][0]["bbox"][1] - rows[end - 1][0]["bbox"][1] <= 30
        ):
            end += 1
        region = rows[index:end]
        if _pdf_borderless_table(region):
            table_pieces = [
                _pdf_piece(
                    [(word_id, word) for piece in row for word_id, word in piece["pairs"]],
                    " | ".join(piece["text"] for piece in row),
                )
                for row in region
            ]
            blocks.append(
                {
                    "top": min(piece["bbox"][1] for piece in table_pieces),
                    "bottom": max(piece["bbox"][3] for piece in table_pieces),
                    "kind": "table",
                    "pieces": table_pieces,
                }
            )
            consumed.update(id(piece) for row in region for piece in row)
        index = end
    return blocks, [piece for piece in pieces if id(piece) not in consumed]


def _pdf_columns(pieces, page_width, page_number):
    """Choose a supported two-column prose order or reject ambiguous regions.

    The widest qualifying horizontal gap separates two vertically overlapping
    groups, each with at least two pieces. Both groups must contain enough
    sentence-like prose. This restricts the accepted layout before assigning a
    left-column-then-right-column reading order; it does not support arbitrary
    newsletters or overlapping text regions.

    Args:
        pieces (list[dict]): Region pieces with ``text`` and bounding boxes.
        page_width (float): Page width in PDF coordinate units; determines the
            minimum gap together with the fixed 40-unit threshold.
        page_number (int): One-based page used in rejection messages.

    Returns:
        tuple[list[dict], list[dict]] | None: Left and right columns, each
            ordered top to bottom then left to right, or None when no gap has
            both the required width and at least eight units of vertical overlap.

    Raises:
        ValueError: A qualifying split lacks sufficient supported prose in
            either column.
    """
    ordered = sorted(pieces, key=lambda piece: piece["bbox"][0])
    candidates = []
    for split in range(2, len(ordered) - 1):
        left, right = ordered[:split], ordered[split:]
        gap = min(piece["bbox"][0] for piece in right) - max(piece["bbox"][2] for piece in left)
        overlap = min(
            max(piece["bbox"][3] for piece in left), max(piece["bbox"][3] for piece in right)
        ) - max(min(piece["bbox"][1] for piece in left), min(piece["bbox"][1] for piece in right))
        if gap >= max(40, page_width / 16) and overlap >= 8:
            candidates.append((gap, left, right))
    if not candidates:
        return None
    _, left, right = max(candidates, key=lambda item: item[0])
    for column in (left, right):
        prose = [piece["text"] for piece in column if not piece["text"].isupper()]
        if (
            len(prose) < 2
            or sum(text.rstrip().endswith((".", "?", "!")) for text in prose) < len(prose) / 2
            or sum(len(text.split()) for text in prose) < 3 * len(prose)
        ):
            raise ValueError(f"PDF page {page_number} has unsupported multi-region layout")
    return (
        sorted(left, key=lambda piece: (piece["bbox"][1], piece["bbox"][0])),
        sorted(right, key=lambda piece: (piece["bbox"][1], piece["bbox"][0])),
    )


def _pdf_text_blocks(pieces, page_width, page_number):
    """Organize residual prose into ordered bands with a supported reading path.

    Rows separated by more than 18 page units start a new band. Each band is
    either a single text region or validated two-column prose. Multiple pieces
    on one row without a supported column split are rejected to avoid silently
    inventing a reading order.

    Args:
        pieces (list[dict]): Residual text pieces with ``text``, ``bbox`` and
            word provenance after table extraction.
        page_width (float): Page width in PDF coordinate units.
        page_number (int): One-based page used in rejection messages.

    Returns:
        list[dict]: Top-to-bottom blocks with geometry and either ``kind='text'``
            plus ordered ``pieces``, or ``kind='columns'`` plus two ordered
            ``columns`` and the band's original ``pieces``.

    Raises:
        ValueError: A band has an unsupported multi-region arrangement.
    """
    rows = _pdf_rows(pieces)
    bands = []
    for row in rows:
        top = min(piece["bbox"][1] for piece in row)
        bottom = max(piece["bbox"][3] for piece in row)
        if not bands or top - bands[-1]["bottom"] > 18:
            bands.append({"top": top, "bottom": bottom, "pieces": list(row)})
        else:
            bands[-1]["bottom"] = max(bottom, bands[-1]["bottom"])
            bands[-1]["pieces"].extend(row)
    blocks = []
    for band in bands:
        columns = _pdf_columns(band["pieces"], page_width, page_number)
        if columns is None and any(len(row) > 1 for row in _pdf_rows(band["pieces"])):
            raise ValueError(f"PDF page {page_number} has unsupported multi-region layout")
        if columns:
            blocks.append({**band, "kind": "columns", "columns": columns})
        else:
            blocks.append(
                {
                    **band,
                    "kind": "text",
                    "pieces": sorted(
                        band["pieces"], key=lambda piece: (piece["bbox"][1], piece["bbox"][0])
                    ),
                }
            )
    return blocks


def _parse_pdf(path, max_pages):
    """Extract supported native-text PDFs into citation-addressable evidence.

    Pages are processed in order. Ruled and recognized borderless tables retain
    row/column order; remaining supported prose follows geometric block order
    and left column before right column. Short uppercase headings supply section
    context, local to each column when appropriate. Coverage checks require every
    extracted word to appear exactly once. They do not prove completeness of the
    PDF's visual content: large image regions are rejected because OCR is absent.

    Args:
        path (Path): Original PDF file read by pdfplumber.
        max_pages (int): Maximum accepted number of PDF pages.

    Returns:
        dict: ``spans`` in accepted reading order and integer ``page_count``.
            Each span has UUID ``id``, trimmed ``text``, one-based ``page`` and
            per-page ``line_start``/``line_end``, page-unit ``bbox`` and page
            dimensions, and string ``section``. Table rows use pipe separators.

    Raises:
        ValueError: Empty/over-limit documents; unsupported image, table or
            region layout; lost/duplicated words; lines over 65,536 UTF-8 bytes;
            more than 32 MiB extracted text or 200,000 spans; or no usable text.
        OSError: The original cannot be read. PDF-library failures otherwise
            propagate to the parser subprocess boundary.
    """
    import pdfplumber

    spans = []
    extracted_bytes = 0
    with pdfplumber.open(path, laparams={"boxes_flow": None}) as pdf:
        if len(pdf.pages) > max_pages:
            raise ValueError("PDF exceeds page limit")
        if not pdf.pages:
            raise ValueError("PDF has no pages")
        for page_number, page in enumerate(pdf.pages, 1):
            words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
            largest_image_area = max(
                (
                    max(0, float(image["x1"]) - float(image["x0"]))
                    * max(0, float(image["bottom"]) - float(image["top"]))
                    for image in page.images
                ),
                default=0,
            )
            page_area = float(page.width) * float(page.height)
            if largest_image_area >= 0.5 * page_area or (
                not words and largest_image_area >= 0.05 * page_area
            ):
                raise ValueError(
                    f"PDF page {page_number} contains image content that may be missing from extracted text; OCR is unavailable"
                )
            table_blocks, table_ids = _pdf_grid_tables(page, words, page_number)
            text_pieces = _pdf_textlines(page, words, table_ids, page_number)
            borderless_blocks, text_pieces = _pdf_borderless_blocks(text_pieces)
            blocks = sorted(
                [
                    *table_blocks,
                    *borderless_blocks,
                    *_pdf_text_blocks(text_pieces, float(page.width), page_number),
                ],
                key=lambda block: (
                    block["top"],
                    min(piece["bbox"][0] for piece in block["pieces"]),
                ),
            )
            if any(left["bottom"] > right["top"] + 3 for left, right in zip(blocks, blocks[1:])):
                raise ValueError(f"PDF page {page_number} has overlapping regions")
            page_section = ""
            ordered = []
            for block in blocks:
                if block["kind"] == "columns":
                    for column in block["columns"]:
                        section = page_section
                        for piece in column:
                            if (
                                len(piece["word_ids"]) <= 5
                                and piece["text"].isupper()
                                and len(piece["text"]) < 160
                            ):
                                section = piece["text"]
                            ordered.append((piece, section))
                else:
                    for piece in block["pieces"]:
                        if block["kind"] == "text" and (
                            len(piece["word_ids"]) <= 5
                            and piece["text"].isupper()
                            and len(piece["text"]) < 160
                        ):
                            page_section = piece["text"]
                        ordered.append((piece, page_section))
            delivered = [word_id for piece, _ in ordered for word_id in piece["word_ids"]]
            if sorted(delivered) != list(range(len(words))):
                raise ValueError(f"PDF page {page_number} lost or duplicated text")
            for line_number, (piece, section) in enumerate(ordered, 1):
                value = piece["text"]
                if len(value.encode("utf-8")) > 65536:
                    raise ValueError("PDF text line is too long")
                extracted_bytes += len(value.encode("utf-8"))
                if extracted_bytes > 32 * 1024 * 1024:
                    raise ValueError("Extracted PDF text exceeds limit")
                spans.append(
                    _line(
                        value,
                        page_number,
                        line_number,
                        piece["bbox"],
                        float(page.width),
                        float(page.height),
                        section,
                    )
                )
                if len(spans) > MAX_SPANS:
                    raise ValueError("PDF has too many text lines")
            page.close()
        if not spans:
            raise ValueError("PDF has no usable text layer; OCR is unavailable")
        return {"spans": spans, "page_count": len(pdf.pages)}


def _parser_limit():
    """Apply parser child-process resource caps before executing extraction.

    CPU time is capped at 150 seconds, open files at 64, and output-file size at
    zero. Non-macOS children also receive a 1 GiB address-space cap. These are
    process resource limits, not an operating-system security sandbox; parent
    supervision separately bounds elapsed time and pipe output.

    Returns:
        None: Resource limits are changed in the calling child process.
    """
    resource.setrlimit(resource.RLIMIT_CPU, (150, 150))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def _parse_subprocess(path, media_type, settings):
    """Extract an original in a bounded child and classify ingestion failures.

    The child runs isolated Python with a minimal locale environment and parser
    resource caps. The parent drains both pipes until the configured wall-time
    deadline, caps stdout at 64 MiB, and retains only the last 2,000 stderr bytes.
    A child still running on exit is killed as a process group and reaped.
    Recognized final ``ValueError: PDF ...``/``Text ...`` lines become permanent
    input errors; other nonzero exits are transient, even if the underlying
    parser exception describes invalid document data. JSON is decoded here
    without validating its record shape.

    Args:
        path (Path): Original file passed to the child without copying it.
        media_type (str): Parser dispatch type, ``text/plain`` or
            ``application/pdf`` for the supported workflow.
        settings (Settings): ``max_upload_bytes`` and ``max_pdf_pages`` are
            passed to extraction; ``parser_timeout_seconds`` bounds wall time.

    Returns:
        dict: Parsed JSON with ordered ``spans`` and ``page_count`` on the
            supported child path; the decoder itself permits other JSON shapes.

    Raises:
        ValueError: Stdout exceeds 64 MiB or the child reports a recognized
            permanent PDF/TXT rejection.
        TransientIngestError: Child startup failure, timeout, other nonzero exit,
            or malformed JSON output. Lower-level pipe/selector errors propagate.
    """
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "medical_assistant.ingest",
        "_parse",
        str(path),
        media_type,
        str(settings.max_upload_bytes),
        str(settings.max_pdf_pages),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            cwd="/",
            preexec_fn=_parser_limit,
            start_new_session=True,
        )
    except OSError as exc:
        raise TransientIngestError("Parser could not be started") from exc
    output = bytearray()
    errors = bytearray()
    deadline = time.monotonic() + settings.parser_timeout_seconds
    with selectors.DefaultSelector() as selector:
        try:
            for stream in (process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransientIngestError("Parser timed out")
                for key, _ in selector.select(timeout=remaining):
                    try:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.fileobj is process.stdout:
                        if len(output) + len(chunk) > MAX_PARSER_OUTPUT_BYTES:
                            raise ValueError("Parser output exceeds limit")
                        output.extend(chunk)
                    else:
                        errors = (errors + chunk)[-2000:]
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise TransientIngestError("Parser timed out") from exc
            if process.returncode:
                detail = errors.decode("utf-8", errors="replace").strip()
                last_line = detail.splitlines()[-1] if detail else ""
                message = "The document could not be parsed. Check that the file is valid and contains readable text."
                if last_line.startswith("ValueError: PDF ") or last_line.startswith(
                    "ValueError: Text "
                ):
                    message = last_line.removeprefix("ValueError: ")[:240]
                error = (
                    ValueError if message.startswith(("PDF ", "Text ")) else TransientIngestError
                )
                raise error(message) from RuntimeError(detail or str(process.returncode))
            try:
                return json.loads(output)
            except json.JSONDecodeError as exc:
                raise TransientIngestError("Parser returned invalid output") from exc
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            process.stdout.close()
            process.stderr.close()


def _fragment_spans(spans):
    """Bound source-span byte size while preserving source-location metadata.

    Lines are split at UTF-8 boundaries into at most 4,096-byte fragments,
    preferring a space/tab in the latter half of each full fragment. The first
    fragment retains its source ID and later fragments receive new UUIDs. All
    fragments retain the original line bounds and PDF rectangle; their boxes
    therefore describe the whole original line rather than a sub-line region.

    Args:
        spans (list[dict]): Ordered parser records with ``id`` and ``text``;
            all other provenance fields are copied unchanged.

    Returns:
        list[dict]: New records in source order with ``text`` fragments and
            contiguous one-based ``ordinal`` values. Fragment text concatenates
            to the original text; empty input text yields no record.

    Raises:
        ValueError: Fragmentation would exceed 200,000 records.
    """
    fragments = []
    for span in spans:
        encoded = span["text"].encode("utf-8")
        start = 0
        while start < len(encoded):
            end = min(start + MAX_SPAN_TEXT_BYTES, len(encoded))
            if end < len(encoded):
                while encoded[end] & 0xC0 == 0x80:
                    end -= 1
                whitespace = max(
                    encoded.rfind(b" ", start + MAX_SPAN_TEXT_BYTES // 2, end),
                    encoded.rfind(b"\t", start + MAX_SPAN_TEXT_BYTES // 2, end),
                )
                if whitespace >= 0:
                    end = whitespace + 1
            fragment = {**span, "text": encoded[start:end].decode("utf-8")}
            if start:
                fragment["id"] = str(uuid.uuid4())
            fragment["ordinal"] = len(fragments) + 1
            fragments.append(fragment)
            if len(fragments) > MAX_SPANS:
                raise ValueError("Document has too many text fragments")
            start = end
    return fragments


def _blocks(spans, variant):
    """Group spans into bounded work blocks for the two index variants.

    Fixed blocks stop at page changes or a target of 1,200 UTF-8 bytes;
    structural blocks stop at page/section changes or 1,800 bytes. Each span
    contributes its text bytes plus one separator byte. A single span may exceed
    the target because spans are never split here; token windowing follows.

    Args:
        spans (list[dict]): Reading-order records with ``text``, ``page`` and
            ``section``. Records are referenced, not copied or mutated.
        variant (str): ``fixed`` selects baseline grouping; any other value
            selects structural grouping. The caller supplies supported variants.

    Returns:
        list[list[dict]]: Nonempty blocks in input order, or an empty list.
    """
    blocks = []
    current = []
    size = 0
    boundary = None
    limit = 1200 if variant == "fixed" else 1800
    for span in spans:
        key = span["page"] if variant == "fixed" else (span["page"], span["section"])
        span_size = len(span["text"].encode("utf-8")) + 1
        if current and (size + span_size > limit or key != boundary):
            blocks.append(current)
            current = []
            size = 0
        current.append(span)
        size += span_size
        boundary = key
    if current:
        blocks.append(current)
    return blocks


def process_source(store, source_id, settings, *, lease=None, ensure_lease=None):
    """Build and atomically publish both searchable indexes for a claimed source.

    Registered byte count and SHA-256 are checked before bounded PDF/TXT parsing.
    Source spans are fragmented, then fixed and structural chunks retain span
    IDs for citations. Structural context continues across work blocks. Document
    context resets at page boundaries; section and table context reset at page
    or section changes. Uncovered nonempty spans receive their own chunks.
    Passage embeddings are computed in batches of 32. Extraction
    and embeddings stay in memory until the store transaction inserts spans and
    chunks, marks the source ready and increments the collection revision and
    ready count while decrementing its unavailable count.

    Args:
        store (Store): Storage boundary providing ``get_source`` and
            ``publish_source``. Source records require ``status``, ``file_key``,
            ``media_type``, ``byte_count`` and ``sha256``.
        source_id (str): Registered source UUID string; its job must already
            have claimed the source into ``processing`` state.
        settings (Settings): Upload storage root, byte/page limits and parser
            timeout. The embedding module reads its own configured model settings.
        lease (dict | None): Optional publication fence with ``job_id`` and
            ``worker_id`` UUID strings and integer ``attempt``. The store checks
            the running job identity and unexpired lease in its transaction.
        ensure_lease (Callable[[], None] | None): Optional lease guard called
            before parsing, before/between embedding batches and before
            publication. It may raise to abort; it does not interrupt an active
            parser call or encoder batch.

    Returns:
        None: Ready or deleted sources return immediately without processing.
            Successful processing publishes once; the store also treats an
            already-ready source as a no-op at publication. This function does
            not finish the job or mark failures; the worker owns those changes.

    Raises:
        KeyError: The registered source no longer exists.
        ValueError: Source state/path, hash/size, extraction/chunking, embedding
            count or publication validation fails, including a stale lease.
        TransientIngestError: Original bytes are unavailable or subprocess
            extraction reports a transient failure.
        RuntimeError: Pinned local embedding files are unavailable. Lease-guard
            and storage exceptions also propagate; publication is transactional.
    """
    source = store.get_source(source_id)
    if source is None:
        raise KeyError(source_id)
    if source["status"] == "ready":
        return
    if source["status"] == "deleted":
        return
    if source["status"] != "processing":
        raise ValueError("Source must be claimed before processing")
    path = _source_path(settings, source["file_key"])
    if not path.is_file() or path.is_symlink():
        raise TransientIngestError("Source bytes are unavailable")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TransientIngestError("Source bytes are unavailable") from exc
    if (
        len(data) != source["byte_count"]
        or len(data) > settings.max_upload_bytes
        or hashlib.sha256(data).hexdigest() != source["sha256"]
    ):
        raise ValueError("Source bytes do not match registered hash and size")
    del data
    if ensure_lease is not None:
        ensure_lease()
    parsed = _parse_subprocess(path, source["media_type"], settings)
    spans = _fragment_spans(parsed["spans"])
    if sum(len(span["text"].encode("utf-8")) for span in spans) > 32 * 1024 * 1024:
        raise ValueError("Extracted text exceeds limit")
    from medical_assistant.embedding import (
        embed_passages,
        standalone_table_headers,
        structural_chunks_with_context,
        token_chunks_with_spans,
    )

    chunks = []
    document_header = []
    section_header = []
    table_header = []
    structural_key = None
    structural_used_ids = set()
    standalone_headers = standalone_table_headers(spans)
    for variant in ("fixed", "structural"):
        for block in _blocks(spans, variant):
            if variant == "structural":
                key = (block[0]["page"], block[0]["section"])
                if key != structural_key:
                    if structural_key is None or key[0] != structural_key[0]:
                        document_header = []
                    section_header = []
                    table_header = []
                    structural_key = key
                block_chunks, document_header, section_header, table_header = (
                    structural_chunks_with_context(
                        block,
                        document_header,
                        section_header,
                        table_header,
                        standalone_headers,
                        complete=False,
                    )
                )
            else:
                block_chunks = token_chunks_with_spans(block, variant)
            for chunk_text, chunk_span_ids in block_chunks:
                if not chunk_text.strip():
                    continue
                if variant == "structural":
                    structural_used_ids.update(chunk_span_ids)
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "span_ids": chunk_span_ids,
                        "text": chunk_text,
                        "variant": variant,
                        "metadata": {"page": block[0]["page"], "section": block[0]["section"]},
                    }
                )
    for span in spans:
        if span["text"].strip() and span["id"] not in structural_used_ids:
            for chunk_text, chunk_span_ids in token_chunks_with_spans([span], "fixed"):
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "span_ids": chunk_span_ids,
                        "text": chunk_text,
                        "variant": "structural",
                        "metadata": {"page": span["page"], "section": span["section"]},
                    }
                )
    if not chunks or {chunk["variant"] for chunk in chunks} != {"fixed", "structural"}:
        raise ValueError("Chunking produced no complete indices")
    if ensure_lease is not None:
        ensure_lease()
    for start in range(0, len(chunks), 32):
        if ensure_lease is not None:
            ensure_lease()
        batch = chunks[start : start + 32]
        embeddings = embed_passages([chunk["text"] for chunk in batch])
        if len(embeddings) != len(batch):
            raise ValueError("Embedding count mismatch")
        for chunk, embedding in zip(batch, embeddings, strict=True):
            chunk["embedding"] = embedding
    if ensure_lease is not None:
        ensure_lease()
    store.publish_source(source_id, spans, chunks, parsed["page_count"], **(lease or {}))


def _main():
    if len(sys.argv) != 6 or sys.argv[1] != "_parse":
        raise SystemExit(2)
    _, _, filename, media_type, max_bytes, max_pages = sys.argv
    path = Path(filename)
    if media_type == "text/plain":
        parsed = _parse_txt(path, int(max_bytes))
    elif media_type == "application/pdf":
        from pdfminer.psexceptions import PSException
        from pdfplumber.utils.exceptions import PdfminerException

        try:
            parsed = _parse_pdf(path, int(max_pages))
        except PdfminerException as exc:
            if exc.args and isinstance(exc.args[0], PSException):
                raise ValueError("PDF file is invalid or unsupported") from exc
            raise
        except PSException as exc:
            raise ValueError("PDF file is invalid or unsupported") from exc
    else:
        raise ValueError("Unsupported media type")
    sys.stdout.write(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    _main()
