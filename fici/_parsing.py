"""Shared string-parsing helpers used by the searcher and verifier.

Keeping the title-extraction heuristic in one place guarantees that the
string we send to the APIs is the same string we later score against the
API response — any changes to the heuristic stay in lock-step on both
sides of the pipeline.
"""

from __future__ import annotations

import re
from typing import Optional


LEADING_MARKER = re.compile(r"^\s*(?:\[\s*\d{1,3}\s*\]|\(\s*\d{1,3}\s*\)|\d{1,3}\.)\s+")
URL_OR_DOI_TAIL = re.compile(r"(https?://\S+|doi:\s*\S+|arXiv:\s*\S+)", re.IGNORECASE)
_YEAR_TOKEN = re.compile(r"\b(19|20)\d{2}[a-z]?\b")

# Venue / publisher keywords that often appear at the end of a bibliographic
# entry. We trim them off the extracted title so the API query isn't polluted
# with "In Proceedings of ..." or "Advances in Neural Information Processing
# Systems" when the heuristic over-extracts.
_VENUE_PREFIXES = (
    "in proceedings of",
    "proceedings of",
    "in advances in",
    "advances in",
    "in international conference",
    "international conference",
    "transactions on",
    "journal of",
    "arxiv preprint",
    "arxiv",
)

# Trailing "(2022)" / "(2022a)" style year markers that slip into the title
# segment when a citation omits a period before the year parenthesis
# (common in Springer LNCS entries: "...Title (2022)").
_TRAILING_YEAR_PAREN = re.compile(r"\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)\s*$")

# Minimum/maximum lengths for a plausible title.
_MIN_TITLE_LEN = 8
_MAX_TITLE_LEN = 250


def strip_markers(raw_citation: str) -> str:
    """Strip leading enumeration markers ("[12]", "12.") and trailing URLs/DOIs."""
    cleaned = LEADING_MARKER.sub("", raw_citation).strip()
    cleaned = URL_OR_DOI_TAIL.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def extract_suspected_title(raw_citation: str) -> Optional[str]:
    """Heuristically extract the paper title from a raw reference string.

    Handles the three dominant patterns in CS bibliographies:

    1. **Quoted** (IEEE / some NeurIPS styles)::

           [1] J. Smith, "Title of the paper," Venue, 2021.
           -> return the quoted substring.

    2. **Springer / LNCS** (``Authors: Title. In: Venue ...``)::

           Alaofi, M., Clarke, C.L.: Generative IR evaluation. In: ...
           -> title is the clause between the post-author colon and the
              next sentence break.

    3. **Author-Year** (ACM acmart / many NeurIPS / ICLR / APA styles)::

           Smith, J. (2021). Title of the paper. Venue, 1-10.
           Bender et al. 2021. On the dangers of stochastic parrots. FAccT.
           -> title is the clause that follows the year.

    Returns ``None`` when no plausible title can be isolated. Callers should
    treat that as "fall back to the cleaned raw string".
    """
    text = strip_markers(raw_citation)
    if not text:
        return None

    # Pattern 1: quoted title (straight or curly quotes).
    m = re.search(r'["“]([^"”]{%d,%d})["”]' % (_MIN_TITLE_LEN, _MAX_TITLE_LEN), text)
    if m:
        return _cleanup(m.group(1))

    # Pattern 2: Springer / LNCS "Authors: Title. Venue ..."
    # The first ``: `` in the string demarcates the author list from the
    # title in this style. We only trust it when the prefix looks like an
    # author list (commas and/or period-initials) so we don't mis-split
    # titles that happen to contain a colon themselves.
    candidate = _extract_after_author_colon(text)
    if candidate:
        return candidate

    # Pattern 3: "... YYYY[.|)]  Title. Venue ..."
    m = _YEAR_TOKEN.search(text)
    if m:
        tail = text[m.end():].lstrip(" ).,:;")
        if tail:
            # Title typically ends at the next period followed by whitespace +
            # a capital letter (start of venue) or end of string. Fall back to
            # the first period if that pattern doesn't match.
            end_match = re.search(r"\.(?=\s+[A-Z]|\s*$)", tail)
            if end_match:
                candidate = tail[: end_match.start()].strip()
            else:
                candidate = tail.split(".")[0].strip()
            candidate = _cleanup(candidate)
            if candidate and _MIN_TITLE_LEN <= len(candidate) <= _MAX_TITLE_LEN:
                return candidate

    # Fallback: longest dot-delimited segment that isn't pure numerics.
    parts = [p.strip() for p in text.split(".") if p.strip()]
    candidates = [
        _cleanup(p) for p in parts if _MIN_TITLE_LEN <= len(p) <= _MAX_TITLE_LEN and not p.isdigit()
    ]
    candidates = [c for c in candidates if c]
    if candidates:
        return max(candidates, key=len)

    return None


def _extract_after_author_colon(text: str) -> Optional[str]:
    """Return the title from a Springer/LNCS-style "Authors: Title. ..." string.

    Triggering conditions (all must hold):

    * A ``": "`` appears within the first 300 characters.
    * The prefix before the colon looks like an author list — it contains at
      least one comma or a single-letter initial followed by a period.
    * The prefix contains **no year token**. In Author-Year styles (APA,
      ACM acmart) the year sits between the author list and the title, so
      the first ``": "`` is almost always inside the title itself (e.g.
      "...: Can Language Models Be Too Big?") rather than the author-list
      delimiter.

    Returns ``None`` otherwise so the caller can fall through to the next
    heuristic.
    """
    colon_idx = text.find(": ")
    if colon_idx < 0 or colon_idx > 300:
        return None

    prefix = text[:colon_idx]
    if "," not in prefix and not re.search(r"\b[A-Z]\.", prefix):
        return None
    if _YEAR_TOKEN.search(prefix):
        # The colon is inside the title (Author-Year subtitle style), not
        # the author-list delimiter.
        return None

    tail = text[colon_idx + 2 :].lstrip()
    if not tail:
        return None

    # Title ends at the next period followed by whitespace + a capital letter
    # (start of the venue line, typically "In:" or "Proceedings of ...").
    end_match = re.search(r"\.(?=\s+[A-Z]|\s*$)", tail)
    candidate = tail[: end_match.start()].strip() if end_match else tail.split(".")[0].strip()

    candidate = _cleanup(candidate)
    if candidate and _MIN_TITLE_LEN <= len(candidate) <= _MAX_TITLE_LEN:
        return candidate
    return None


def _cleanup(candidate: str) -> str:
    """Tidy up an extracted title: trim trailing venue markers and punctuation."""
    text = candidate.strip().strip(",;:").strip()
    if not text:
        return ""

    # Drop a trailing "(YYYY)" that slipped in from the end of the citation
    # when the title segment wasn't period-terminated.
    text = _TRAILING_YEAR_PAREN.sub("", text).strip().strip(",;:").strip()

    # Drop a leading "In " that commonly precedes the venue in ACM entries —
    # it shouldn't show up in a title but does occasionally when the heuristic
    # over-extracts one segment.
    if text.lower().startswith("in "):
        tail = text[3:].lstrip()
        if any(tail.lower().startswith(p[3:]) for p in _VENUE_PREFIXES if p.startswith("in ")):
            # It's actually a venue, not a title.
            return ""
        text = tail

    # If any venue prefix appears mid-string, cut the title off there.
    lower = text.lower()
    for prefix in _VENUE_PREFIXES:
        idx = lower.find(prefix)
        if idx > _MIN_TITLE_LEN:
            text = text[:idx].rstrip(" ,;:-")
            lower = text.lower()

    return text.strip()
