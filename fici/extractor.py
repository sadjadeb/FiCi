"""Phase 1: Heuristic PDF reference extraction using PyMuPDF.

Targets common conference layouts:
    * NeurIPS / ICLR: single-column body, numeric bracketed references "[1]".
    * ACM (acmart / SIG conf): two-column, numeric references "1." or "[1]".
    * Some CS venues use plain Author-Year entries as a fallback style.

The extractor operates in three stages:
    1. Locate the "References"/"Bibliography" section header.
    2. Collect the raw text of everything that follows it on subsequent pages,
       trimming any trailing "Appendix"/"Supplementary" section if present.
    3. Split the collected text into individual citation strings using a set
       of prioritized regex splitters.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyMuPDF is required for PDF extraction. Install it via `pip install PyMuPDF`."
    ) from exc


# Headers that typically mark the start of the bibliography.
_REFERENCE_HEADER_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^\s*references\s*$", re.IGNORECASE),
    re.compile(r"^\s*bibliography\s*$", re.IGNORECASE),
    re.compile(r"^\s*references\s+and\s+notes\s*$", re.IGNORECASE),
    # Numbered section like "7 References" or "7. References".
    re.compile(r"^\s*\d+\.?\s+references\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\.?\s+bibliography\s*$", re.IGNORECASE),
)

# Sections that typically follow the bibliography and should be excluded.
_POST_REFERENCE_STOP_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^\s*appendix(?:\s+[a-z])?\s*$", re.IGNORECASE),
    re.compile(r"^\s*supplementary(?:\s+material)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*a\s+appendix\s*$", re.IGNORECASE),
    re.compile(r"^\s*checklist\s*$", re.IGNORECASE),
)

# Primary splitter: bracketed numeric references like "[1]", "[12]".
_BRACKETED_NUM_SPLIT = re.compile(r"(?m)(?=^\s*\[\s*\d{1,3}\s*\]\s+)")

# Secondary splitter: "1." / "12." at the start of a line, used by many ACM/SIG
# conf templates. We require a following space and a capital letter to avoid
# breaking on things like "1.5 MB" inside a citation.
_DOTTED_NUM_SPLIT = re.compile(r"(?m)(?=^\s*\d{1,3}\.\s+[A-Z])")

# Fallback splitter: author-year style references. Each entry usually starts
# with a capitalized surname, possibly with initials, followed by a comma or
# period and more authors/title. This is inherently fuzzy.
_AUTHOR_YEAR_SPLIT = re.compile(
    r"(?m)(?=^\s*[A-Z][A-Za-z'\-]+(?:,\s+[A-Z]\.|,\s+[A-Z][A-Za-z'\-]+|\s+et\s+al\.?))"
)

# Minimum viable citation length — anything shorter is almost certainly noise.
_MIN_CITATION_LEN = 25


# ---------------------------------------------------------------------- #
# Plain-text bullet-list parsing (.txt input)
# ---------------------------------------------------------------------- #
# Characters that commonly mark the start of a citation in a bullet- or
# dash-prefixed list. ASCII first, then the various Unicode bullets and
# dashes that survive a copy/paste from Word, Google Docs, or Markdown.
_TEXT_BULLET_PREFIXES: Tuple[str, ...] = (
    "•", "●", "◦", "‣", "▪", "■", "□", "◾",
    "*", "+", ">",
    "—", "–", "-",
)


def parse_text_entries(raw_text: str) -> List[str]:
    """Parse a plain-text bullet list (or one-per-line file) into citations.

    Two source styles are recognised automatically:

    * **Bulleted** — the user's example, where each citation begins with
      a bullet character (``•``, ``*``, ``-`` …). Unmarked lines that
      follow are folded into the current entry so that wrapped citations
      survive intact.
    * **Flat**    — no bullets anywhere in the file. Each non-empty line
      is taken as a single citation.

    Leading enumeration markers like ``[12]`` or ``12.`` are *not*
    stripped here — they're handled downstream by
    :func:`fici._parsing.strip_markers`, just as for PDF-extracted
    citations.
    """
    if raw_text.startswith("\ufeff"):
        raw_text = raw_text[1:]

    lines = raw_text.splitlines()
    has_bullets = any(_starts_with_bullet(line) for line in lines)

    entries: List[str] = []
    current: List[str] = []

    def _flush() -> None:
        if current:
            joined = " ".join(current).strip()
            if joined:
                entries.append(joined)
            current.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _flush()
            continue

        if not has_bullets:
            _flush()
            current.append(stripped)
            continue

        unmarked = _strip_bullet(stripped)
        if unmarked is not None:
            _flush()
            if unmarked:
                current.append(unmarked)
        elif current:
            current.append(stripped)
        else:
            current.append(stripped)

    _flush()
    return entries


def _starts_with_bullet(line: str) -> bool:
    return _strip_bullet(line.lstrip()) is not None


def _strip_bullet(stripped_line: str) -> Optional[str]:
    """Return the line with its leading bullet removed, or ``None`` if absent."""
    for marker in _TEXT_BULLET_PREFIXES:
        if stripped_line.startswith(marker):
            rest = stripped_line[len(marker):]
            # Require whitespace (or end of line) after the marker so we
            # don't mistake "-x" inside a token for a list bullet.
            if not rest or rest[0] in (" ", "\t", "\u00a0"):
                return rest.lstrip(" \t\u00a0")
    return None


# ---------------------------------------------------------------------- #
# Line-end hyphenation
# ---------------------------------------------------------------------- #
# Common compound-word prefixes. When one of these appears as the LEFT
# fragment of a line-end hyphen the hyphen is almost certainly real
# ("self-\nsupervised", "pre-\ntrained") rather than a typesetter soft-wrap.
_COMPOUND_LEFT = frozenset({
    "pre", "post", "non", "self", "semi", "sub", "super", "multi",
    "cross", "anti", "meta", "inter", "intra", "bi", "tri", "uni",
    "mid", "mini", "micro", "macro", "quasi", "neo", "auto", "ultra",
    "hyper", "co", "re",
})
# Common particles that form the RIGHT fragment of a hyphenated compound
# ("state-of-\nthe-art", "end-\nto-end", "Text-\nto-Text").
_COMPOUND_RIGHT = frozenset({
    "of", "to", "the", "by", "in", "on", "as", "up", "off",
    "end", "art", "out", "all", "one",
})

# Pattern for a line-end hyphen with fully alphabetic fragments on both
# sides — exactly the shape that can be either a soft wrap or a real
# hyphen. ``[^\W\d_]`` matches any Unicode letter (including accented
# glyphs like "í" in "Martínez") but excludes digits and underscores.
_EOL_HYPHEN_RE = re.compile(r"([^\W\d_]+)-\n([^\W\d_]+)", re.UNICODE)


def _dehyphenate_line_breaks(text: str) -> str:
    """Resolve line-end hyphenation without mangling real hyphens.

    Drops the hyphen when it looks like a typesetter soft-wrap
    ("informa-\\ntion" -> "information") but preserves it (and rejoins the
    fragments into a single token) when any of the following hold:

    * the fragment after the hyphen starts with an uppercase letter —
      proper noun / compound name, e.g. "Fei-\\nFei", "Meta-\\nLearning";
    * the fragment before the hyphen is a known compound prefix such as
      ``pre``, ``self``, ``multi``, ``cross`` — e.g. "self-\\nsupervised";
    * the fragment after the hyphen is a known compound particle such as
      ``of``, ``to``, ``the``, ``end`` — e.g. "state-of-\\nthe-art".

    When in doubt we keep the hyphen, because a spurious hyphen (``infor-
    mation``) is tolerated by the downstream fuzzy matcher whereas a
    missing hyphen (``FeiFei``) breaks tokenization for the API search.
    """
    def _resolve(match: "re.Match[str]") -> str:
        left, right = match.group(1), match.group(2)
        keep_hyphen = (
            right[0].isupper()
            or left.lower() in _COMPOUND_LEFT
            or right.lower() in _COMPOUND_RIGHT
        )
        return f"{left}-{right}" if keep_hyphen else f"{left}{right}"
    return _EOL_HYPHEN_RE.sub(_resolve, text)


class ReferenceExtractor:
    """Extract individual raw citation strings from a PDF, BibTeX, or text file.

    The extractor dispatches by file extension:

    * ``.pdf``            — locate the References section and split it
      into individual entries with the heuristic regex pipeline below.
    * ``.bib`` / ``.bbl`` — parse BibTeX ``@entry{...}`` blocks via
      :mod:`fici.bibtex` and render each into a free-text citation
      string, mirroring the shape of PDF-extracted entries so the rest
      of the pipeline runs unchanged.
    * ``.txt``            — parse a bullet-/dash-prefixed list (or a
      flat one-per-line file) where each non-empty entry is treated as
      a single raw citation. Useful for ad-hoc lists pasted from
      Word / Google Docs / Markdown notes.

    Example:
        extractor = ReferenceExtractor()
        refs = extractor.extract("paper.pdf")
        refs = extractor.extract("references.bib")
        refs = extractor.extract("citations.txt")
    """

    # File extensions handled by the BibTeX path. ``.bbl`` is included
    # because users sometimes paste raw BibTeX into a ``.bbl`` file even
    # though that's technically the LaTeX-formatted bibliography output;
    # we let the parser decide and degrade gracefully if it sees nothing
    # parseable.
    _BIBTEX_EXTENSIONS = frozenset({".bib", ".bbl"})
    _TEXT_EXTENSIONS = frozenset({".txt"})

    def __init__(
        self,
        min_citation_length: int = _MIN_CITATION_LEN,
        max_citation_length: int = 1200,
    ) -> None:
        self.min_citation_length = min_citation_length
        self.max_citation_length = max_citation_length

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, input_path: str | Path) -> List[str]:
        """Extract a list of raw citation strings from ``input_path``.

        Returns an empty list if no entries can be located. ``input_path``
        may be a PDF, a ``.bib`` / ``.bbl`` BibTeX file, or a ``.txt``
        bullet/line list of citations.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in self._BIBTEX_EXTENSIONS:
            from .bibtex import parse_bibtex_file
            entries = parse_bibtex_file(path)
            return self._postprocess_entries(entries)

        if suffix in self._TEXT_EXTENSIONS:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            entries = parse_text_entries(raw_text)
            return self._postprocess_entries(entries)

        raw_bib = self._extract_bibliography_text(path)
        if not raw_bib:
            return []

        return self._split_into_citations(raw_bib)

    # ------------------------------------------------------------------ #
    # Stage 1 & 2: locate the bibliography and collect its text
    # ------------------------------------------------------------------ #
    def _extract_bibliography_text(self, pdf_path: Path) -> str:
        """Return the raw text of the bibliography section, or empty string."""
        with fitz.open(pdf_path) as doc:
            # Pull per-page plain text preserving line breaks.
            pages: List[str] = [page.get_text("text") for page in doc]

        start_page, start_line = self._find_reference_header(pages)
        if start_page is None:
            return ""

        collected: List[str] = []
        for page_idx in range(start_page, len(pages)):
            page_text = pages[page_idx]
            lines = page_text.splitlines()

            # On the first page, skip everything up to and including the header.
            begin = start_line + 1 if page_idx == start_page else 0

            for i in range(begin, len(lines)):
                if self._is_post_reference_stop(lines[i]):
                    # Hit an appendix/supplementary boundary — stop collecting.
                    return "\n".join(collected).strip()
                collected.append(lines[i])

        return "\n".join(collected).strip()

    @staticmethod
    def _find_reference_header(pages: List[str]) -> Tuple[Optional[int], int]:
        """Locate the last "References"/"Bibliography" header.

        We prefer the LAST occurrence in the document because body text can
        occasionally contain phrases like "see the references" in earlier
        pages, whereas the actual section header is late in the paper.
        """
        last: Tuple[Optional[int], int] = (None, -1)
        for page_idx, page_text in enumerate(pages):
            for line_idx, line in enumerate(page_text.splitlines()):
                stripped = line.strip()
                if not stripped or len(stripped) > 40:
                    continue
                if any(pat.match(stripped) for pat in _REFERENCE_HEADER_PATTERNS):
                    last = (page_idx, line_idx)
        return last

    @staticmethod
    def _is_post_reference_stop(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 40:
            return False
        return any(pat.match(stripped) for pat in _POST_REFERENCE_STOP_PATTERNS)

    # ------------------------------------------------------------------ #
    # Stage 3: split the raw bibliography text into individual entries
    # ------------------------------------------------------------------ #
    def _split_into_citations(self, raw_text: str) -> List[str]:
        """Try splitters in order of specificity until one yields >= 3 entries."""
        normalized = self._normalize_whitespace(raw_text)

        for splitter in (
            self._split_bracketed_numbers,
            self._split_dotted_numbers,
            self._split_author_year,
        ):
            entries = splitter(normalized)
            entries = self._postprocess_entries(entries)
            if len(entries) >= 3:
                return entries

        # Last resort: return whatever the most permissive splitter produced,
        # even if it's short — better than nothing for small reference lists.
        fallback = self._postprocess_entries(self._split_author_year(normalized))
        return fallback

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse soft wraps while keeping hard line breaks between entries.

        PyMuPDF preserves the PDF's line-breaks. References often wrap mid
        citation, so we join intra-entry wraps but keep blank lines and lines
        that look like the start of a new reference marker.
        """
        # Resolve hyphens at line ends: drop for typesetter soft-wraps
        # ("informa-\ntion" -> "information") but preserve for real hyphens
        # in compound names / prefixes ("Fei-\nFei" -> "Fei-Fei").
        text = _dehyphenate_line_breaks(text)

        out_lines: List[str] = []
        for line in text.split("\n"):
            if not line.strip():
                out_lines.append("")  # preserve blank separators
                continue
            out_lines.append(line.rstrip())

        # Rejoin continuation lines: a line that doesn't look like a new entry
        # marker should be appended to the previous non-empty line.
        merged: List[str] = []
        new_entry_markers = (
            re.compile(r"^\s*\[\s*\d{1,3}\s*\]\s+"),
            re.compile(r"^\s*\d{1,3}\.\s+[A-Z]"),
        )
        for line in out_lines:
            if not line:
                merged.append("")
                continue
            starts_new = any(m.match(line) for m in new_entry_markers)
            if starts_new or not merged or not merged[-1]:
                merged.append(line)
            else:
                merged[-1] = f"{merged[-1]} {line.lstrip()}"

        return "\n".join(merged)

    @staticmethod
    def _split_bracketed_numbers(text: str) -> List[str]:
        chunks = _BRACKETED_NUM_SPLIT.split(text)
        return [c for c in chunks if re.match(r"^\s*\[\s*\d{1,3}\s*\]", c)]

    @staticmethod
    def _split_dotted_numbers(text: str) -> List[str]:
        chunks = _DOTTED_NUM_SPLIT.split(text)
        return [c for c in chunks if re.match(r"^\s*\d{1,3}\.\s+[A-Z]", c)]

    @staticmethod
    def _split_author_year(text: str) -> List[str]:
        chunks = _AUTHOR_YEAR_SPLIT.split(text)
        return [c for c in chunks if c.strip()]

    def _postprocess_entries(self, entries: Iterable[str]) -> List[str]:
        cleaned: List[str] = []
        for entry in entries:
            normalized = re.sub(r"\s+", " ", entry).strip()
            if not normalized:
                continue
            if len(normalized) < self.min_citation_length:
                continue
            if len(normalized) > self.max_citation_length:
                normalized = normalized[: self.max_citation_length].rstrip()
            cleaned.append(normalized)
        return cleaned
