import re
import uuid

from rag_assistant.config import settings
from rag_assistant.ingestion.models import Chunk, ContentElement

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
AUDIO_PARENT_WINDOW_SECONDS = 90  # CLAUDE.md §13's ~90s topic-window size

# PyMuPDF4LLM emits markdown, so code arrives fenced. Matching the fence lets a
# code block stay one chunk no matter how long it is (CLAUDE.md §3's "returns
# the code block intact, never split mid-block" criterion), which a blind
# fixed-size split cannot honour.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)

# Markdown pipe tables, kept whole for the same reason as code: half a table is
# not a useful retrieval result, and the header row is what gives the body rows
# meaning.
#
# This also makes §20.1's `table_lookup` routing rule live. Nothing in the
# pipeline had ever set element_type="table", so that rule filtered against an
# always-empty set - the exact dead-routing bug §13 describes fixing for "code",
# left in place for its sibling (CLAUDE.md §7 item 33). Requires the separator
# row (|---|---|), so ordinary prose containing a pipe character cannot match.
_TABLE_RE = re.compile(
    r"^[ \t]*\|.*\|[ \t]*\r?\n[ \t]*\|[ \t:\-|]+\|[ \t]*\r?\n(?:[ \t]*\|.*\|[ \t]*\r?\n?)*",
    re.MULTILINE,
)


def _new_chunk(
    source_id: str,
    text: str,
    element: ContentElement,
    parent_id: str,
    element_type: str | None = None,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        source_id=source_id,
        text=text,
        element_type=element_type or element.element_type,
        chunk_index=chunk_index,
        page=element.page,
        start_time=element.start_time,
        end_time=element.end_time,
        speaker=element.speaker,
        multi_speaker=element.multi_speaker,
        language=element.language,
        # Carried through so §17's indexed low_confidence payload field is
        # actually populated - without this the signal would stop at chunking.
        low_confidence=element.low_confidence,
        confidence_score=element.confidence_score,
        ocr=element.ocr,
        parent_id=parent_id,
        embedding_model_version=settings.embedding_model_version,
        chunking_version=settings.chunking_version,
    )


def _split_prose(text: str) -> list[str]:
    """Fixed-size + overlap split, used only for non-code spans."""
    pieces = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return pieces


def _split_preserving_blocks(text: str) -> list[tuple[str, str | None]]:
    """Splits into (piece, element_type) pairs.

    Fenced code blocks and markdown tables come back WHOLE - deliberately
    unsplit even when longer than CHUNK_SIZE, because half a code block or a
    table stripped of its header is not a useful retrieval result. Everything
    between them is ordinary prose, split fixed-size with overlap.
    """
    blocks: list[tuple[int, int, str]] = [
        (m.start(), m.end(), "code") for m in _FENCED_CODE_RE.finditer(text)
    ]
    blocks += [(m.start(), m.end(), "table") for m in _TABLE_RE.finditer(text)]
    blocks.sort()

    pieces: list[tuple[str, str | None]] = []
    cursor = 0
    for start, end, kind in blocks:
        if start < cursor:
            continue  # a table inside a fenced block, say - the outer one wins
        pieces.extend((p, None) for p in _split_prose(text[cursor:start]))
        block = text[start:end].strip()
        if block:
            pieces.append((block, kind))
        cursor = end
    pieces.extend((p, None) for p in _split_prose(text[cursor:]))
    return pieces


def chunk_elements(elements: list[ContentElement], source_id: str) -> list[Chunk]:
    """Structure-light chunking with real parent-child links (CLAUDE.md §32
    V2 scope call #2 - page-level/time-window grouping, not a full Docling
    rewrite; see the Session 2 plan for why).

    PDF pages ("paragraph" elements): fenced code blocks are kept intact as
    their own chunks and tagged element_type="code" (which is what makes
    §20.1's code_lookup routing filter able to find anything); the prose
    around them is split fixed-size with overlap. parent_id is the page, so a
    chunk never spans two pages and citations stay exact.

    Audio/video/YouTube ("transcript_segment" elements): one chunk per ASR
    segment, unsplit (splitting mid-segment would need word-level timestamps
    this project doesn't request) - parent_id groups consecutive segments
    into ~90s windows for context expansion (§20.3-style grouping, applied
    here to parent-child rather than broad-scope retrieval).
    """
    chunks: list[Chunk] = []
    # Position within each parent group, so expansion can rebuild a page or a
    # time window in document order rather than by uuid (see Chunk.chunk_index).
    next_index: dict[str, int] = {}

    def _take_index(parent_id: str) -> int:
        index = next_index.get(parent_id, 0)
        next_index[parent_id] = index + 1
        return index

    for element in elements:
        if element.element_type == "transcript_segment":
            window_index = int((element.start_time or 0) // AUDIO_PARENT_WINDOW_SECONDS)
            parent_id = f"{source_id}:window:{window_index}"
            chunks.append(
                _new_chunk(
                    source_id,
                    element.text,
                    element,
                    parent_id,
                    chunk_index=_take_index(parent_id),
                )
            )
            continue

        parent_id = f"{source_id}:page:{element.page}"
        for piece, block_type in _split_preserving_blocks(element.text):
            chunks.append(
                _new_chunk(
                    source_id,
                    piece,
                    element,
                    parent_id,
                    element_type=block_type,
                    chunk_index=_take_index(parent_id),
                )
            )
    return chunks
