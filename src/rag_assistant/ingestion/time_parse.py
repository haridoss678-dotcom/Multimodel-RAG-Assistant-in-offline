import re


def parse_time_reference(text: str) -> float | None:
    """Parses a classifier-extracted time reference like '32 minutes',
    '1:15:00', or '90 seconds' into seconds (CLAUDE.md §20.2's temporal
    retrieval). Returns None if unparseable - callers fall back to normal
    semantic search rather than failing."""
    if not text:
        return None
    text = text.strip().lower()

    match = re.match(r"^(\d+):(\d{2})(?::(\d{2}))?$", text)
    if match:
        parts = [int(p) for p in match.groups() if p is not None]
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    match = re.search(r"(\d+(?:\.\d+)?)\s*h", text)
    hours = float(match.group(1)) * 3600 if match else 0
    match = re.search(r"(\d+(?:\.\d+)?)\s*m", text)
    minutes = float(match.group(1)) * 60 if match else 0
    match = re.search(r"(\d+(?:\.\d+)?)\s*s", text)
    seconds = float(match.group(1)) if match else 0

    total = hours + minutes + seconds
    return total if total > 0 else None


# Time expressions as a person actually writes them in a question. Kept
# separate from parse_time_reference, which parses a value the classifier
# already extracted; this finds one in raw prose.
#
# Why it exists (CLAUDE.md §7 item 52): temporal retrieval routes on a parsed
# time value, and the 0.6B classifier drops `time_reference` often enough that
# "what was discussed around 20 minutes?" fell through to ordinary semantic
# search on a live run. A regex over the question is the reliable half of that
# signal - the same reasoning §20.2 already gives for not routing on `intent`.
_TIME_IN_QUESTION_RE = re.compile(
    r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)"  # 1:15, 1:15:00
    r"|(?<!\d)(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b"  # 2 hours
    r"|(?<!\d)(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|m)\b"  # 32 minutes
    r"|(?<!\d)(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s)\b",  # 90 seconds
    re.IGNORECASE,
)


def extract_time_reference(question: str) -> float | None:
    """Finds a time reference in the question itself, in seconds.

    Returns None when there is none - callers then fall back to ordinary
    semantic search rather than inventing a window.
    """
    if not question:
        return None
    total = 0.0
    found = False
    for match in _TIME_IN_QUESTION_RE.finditer(question):
        clock_form, hours, _hu, minutes, _mu, seconds, _su = match.groups()
        if clock_form:
            parsed = parse_time_reference(clock_form)
            if parsed is not None:
                return parsed
            continue
        if hours:
            total += float(hours) * 3600
        elif minutes:
            total += float(minutes) * 60
        elif seconds:
            total += float(seconds)
        found = True
    return total if found and total > 0 else None
