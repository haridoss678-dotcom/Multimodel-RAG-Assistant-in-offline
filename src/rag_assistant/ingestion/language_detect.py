import pymupdf as fitz

# Legacy non-Unicode Tamil fonts extract as garbage text even from a
# "digital" PDF (CLAUDE.md §13) - pages using them are force-routed to OCR.
_LEGACY_TAMIL_FONT_MARKERS = ("bamini", "tab", "tscii", "tam-tab")

_TAMIL_UNICODE_RANGE = range(0x0B80, 0x0C00)


def detect_language(text: str) -> str:
    """Very small, dependency-free language guess: en / ta / mixed, based on
    Tamil Unicode block presence (CLAUDE.md §15's ContentElement.language)."""
    if not text.strip():
        return "en"
    tamil_chars = sum(1 for ch in text if ord(ch) in _TAMIL_UNICODE_RANGE)
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return "en"
    tamil_ratio = tamil_chars / letters
    if tamil_ratio > 0.6:
        return "ta"
    if tamil_ratio > 0.05:
        return "mixed"
    return "en"


def is_legacy_tamil_font(doc: fitz.Document, page_number_zero_indexed: int) -> bool:
    """Checks embedded font names on a page for known non-Unicode Tamil font
    markers (Bamini/TSCII/TAB) - such pages must be OCR'd, not text-extracted,
    since their "text" is meaningless glyph codes (CLAUDE.md §13)."""
    page = doc[page_number_zero_indexed]
    for font in page.get_fonts():
        font_name = font[3].lower()  # font[3] is the font's base name
        if any(marker in font_name for marker in _LEGACY_TAMIL_FONT_MARKERS):
            return True
    return False
