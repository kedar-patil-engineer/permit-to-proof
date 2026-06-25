"""The ingest stage: read a permit PDF into ordered, traceable text segments.

This stage never calls the language model and never interprets meaning. It
only turns pages of text into Segment records with page numbers and character
positions into the full document text, so any later quote can be traced back
to where it came from (Part B2, B4.1).

Segments are split at blank lines and at the start of numbered or lettered
conditions, which is how permit obligations are typically laid out. A segment
is roughly one condition, which keeps grounding meaningful.
"""

from __future__ import annotations

import io
import re
from typing import List, Tuple, Union

import pdfplumber

from .schema import Segment

# A line that begins a new condition, e.g. "Condition 3.1.", "SECTION 4 -",
# "1.", "(a)", or a bullet. Used to start a fresh segment so that each segment
# is roughly one obligation.
_CONDITION_START = re.compile(
    r"^\s*(?:"
    r"(?:Condition|Section|Part|Article|Paragraph|Chapter)\s+[A-Za-z0-9]+[.)]?"
    r"|\(?\d{1,3}(?:\.\d{1,3})*\)?[.)]\s"
    r"|\(?[A-Za-z]\)\s"
    r"|[•●\-\*]\s"
    r")",
    re.IGNORECASE,
)

PdfSource = Union[str, bytes, io.BytesIO]


def _split_page_into_blocks(text: str) -> List[str]:
    """Group a page's lines into condition sized blocks."""
    blocks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            joined = " ".join(line.strip() for line in current).strip()
            if joined:
                blocks.append(joined)
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _CONDITION_START.match(raw_line) and current:
            flush()
        current.append(line)
    flush()
    return blocks


def _open_pdf(source: PdfSource):
    if isinstance(source, (bytes, bytearray)):
        return pdfplumber.open(io.BytesIO(source))
    if isinstance(source, io.BytesIO):
        source.seek(0)
        return pdfplumber.open(source)
    return pdfplumber.open(source)


# ---------------------------------------------------------------------------
# Table handling. Real permits put emission limits in tables, which plain text
# extraction mashes into unreadable run-on lines. We detect tables, render them
# as clean pipe-separated rows the model can parse, and pull the surrounding
# prose separately so a limit row is never lost in a garbled paragraph.
# ---------------------------------------------------------------------------

def _safe_find_tables(page):
    try:
        return page.find_tables() or []
    except Exception:
        return []


def _safe_extract(table):
    try:
        return table.extract() or []
    except Exception:
        return []


def _inside(obj, bboxes) -> bool:
    x0 = obj.get("x0", 0.0); x1 = obj.get("x1", 0.0)
    top = obj.get("top", 0.0); bottom = obj.get("bottom", 0.0)
    for bx0, btop, bx1, bbottom in bboxes:
        if x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbottom + 1:
            return True
    return False


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _render_table(rows: List) -> str:
    """Render a table as clean lines (header + one row per line, pipe separated).

    Keeping it as one block, with the header on the first line, gives the model
    the column meaning while letting it copy a single row verbatim into
    source_quote so grounding still works.
    """
    lines = []
    for row in rows:
        cells = [_clean_cell(c) for c in row]
        if not any(cells):
            continue
        line = " | ".join(cells).strip()
        line = re.sub(r"(?:\s*\|\s*){2,}", " | ", line).strip(" |")
        if line:
            lines.append(line)
    if not lines:
        return ""
    return "TABLE:\n" + "\n".join(lines)


def ingest_pdf(source: PdfSource) -> Tuple[str, List[Segment]]:
    """Read a PDF and return (full_text, segments).

    full_text is every segment joined by newlines; each segment's start_char
    and end_char index into it, so full_text[start:end] == segment.text. This
    invariant is what lets the UI show exactly where a quote lives. Tables are
    emitted as their own clean segments so table-bound limits survive.
    """
    segments: List[Segment] = []
    parts: List[str] = []
    state = {"cursor": 0, "index": 0}

    def add(text: str, page_number: int) -> None:
        text = text.strip()
        if not text:
            return
        start = state["cursor"]
        end = start + len(text)
        segments.append(Segment(
            segment_id="S%04d" % state["index"], text=text, page=page_number,
            start_char=start, end_char=end))
        parts.append(text)
        state["index"] += 1
        state["cursor"] = end + 1  # account for the newline join separator

    with _open_pdf(source) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = _safe_find_tables(page)
            if tables:
                bboxes = [t.bbox for t in tables]
                try:
                    page_text = page.filter(lambda o: not _inside(o, bboxes)).extract_text() or ""
                except Exception:
                    page_text = page.extract_text() or ""
            else:
                page_text = page.extract_text() or ""

            for block in _split_page_into_blocks(page_text):
                add(block, page_number)
            for table in tables:
                add(_render_table(_safe_extract(table)), page_number)

    return "\n".join(parts), segments


def segments_to_text_map(segments: List[Segment]) -> dict:
    """Convenience map of segment_id -> text for the verification stage."""
    return {s.segment_id: s.text for s in segments}
