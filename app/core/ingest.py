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


def ingest_pdf(source: PdfSource) -> Tuple[str, List[Segment]]:
    """Read a PDF and return (full_text, segments).

    full_text is every segment joined by newlines; each segment's start_char
    and end_char index into it, so full_text[start:end] == segment.text. This
    invariant is what lets the UI show exactly where a quote lives.
    """
    segments: List[Segment] = []
    parts: List[str] = []
    cursor = 0
    seg_index = 0

    with _open_pdf(source) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            for block in _split_page_into_blocks(page_text):
                start = cursor
                end = start + len(block)
                segments.append(
                    Segment(
                        segment_id="S%04d" % seg_index,
                        text=block,
                        page=page_number,
                        start_char=start,
                        end_char=end,
                    )
                )
                parts.append(block)
                seg_index += 1
                cursor = end + 1  # account for the newline join separator

    full_text = "\n".join(parts)
    return full_text, segments


def segments_to_text_map(segments: List[Segment]) -> dict:
    """Convenience map of segment_id -> text for the verification stage."""
    return {s.segment_id: s.text for s in segments}
