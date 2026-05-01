"""Load citations directly from a BibTeX file.

This module sidesteps PDF extraction entirely for users who already have a
BibTeX bibliography (``.bib`` or ``.bbl``). Each ``@entry{...}`` block is
parsed and rendered into a single free-text citation string that mimics
the shape produced by :mod:`fici.extractor` (``Authors (Year). Title.
Venue.``), so the rest of the pipeline — title extraction, search,
verification — runs unchanged.

We deliberately avoid pulling in a third-party BibTeX library: the subset
of the format we need to handle is small and stable, and a focused parser
keeps FiCi's dependency footprint minimal.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


logger = logging.getLogger(__name__)


# Entry types that are not bibliographic records and must be skipped.
_NON_RECORD_ENTRY_TYPES = frozenset({"comment", "string", "preamble"})

# Field-extraction priority for the venue. The first non-empty match wins.
_VENUE_FIELDS: Tuple[str, ...] = (
    "journal", "booktitle", "publisher", "school", "institution", "howpublished",
)


def parse_bibtex_file(path: str | Path) -> List[str]:
    """Parse ``path`` and return one rendered citation string per BibTeX entry.

    Returns an empty list if the file is unreadable as text or contains no
    parseable entries. The order of entries in the file is preserved.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    entries = list(_iter_entries(text))
    citations = [_render_citation(e) for e in entries]
    citations = [c for c in citations if c]
    logger.info("Parsed %d BibTeX entries from %s", len(citations), p)
    return citations


# ---------------------------------------------------------------------- #
# Low-level parsing
# ---------------------------------------------------------------------- #
def _iter_entries(text: str) -> Iterator[Dict[str, str]]:
    """Yield ``{'type': ..., 'key': ..., <field>: ...}`` dicts from raw BibTeX."""
    i = 0
    while True:
        m = re.search(r"@(\w+)\s*\{", text[i:])
        if not m:
            return
        entry_type = m.group(1).lower()
        body_start = i + m.end()  # position just after '{'
        body_end = _find_matching_close(text, body_start - 1)
        if body_end < 0:
            # Unbalanced braces — abandon further parsing rather than risk
            # silently misattributing fields.
            logger.warning("BibTeX entry starting near offset %d has no matching '}'", i + m.start())
            return

        if entry_type in _NON_RECORD_ENTRY_TYPES:
            i = body_end + 1
            continue

        body = text[body_start:body_end]
        key, _, rest = body.partition(",")
        entry: Dict[str, str] = {"type": entry_type, "key": key.strip()}
        entry.update(_parse_fields(rest))
        yield entry
        i = body_end + 1


def _find_matching_close(text: str, open_brace_pos: int) -> int:
    """Return the index of the ``}`` matching the ``{`` at ``open_brace_pos``."""
    depth = 0
    n = len(text)
    j = open_brace_pos
    while j < n:
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _parse_fields(body: str) -> Dict[str, str]:
    """Extract ``field = value`` pairs from a comma-delimited entry body."""
    fields: Dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break
        m = re.match(r"(\w+)\s*=\s*", body[i:])
        if not m:
            break
        name = m.group(1).lower()
        i += m.end()
        if i >= n:
            break

        ch = body[i]
        if ch == "{":
            end = _find_matching_close(body, i)
            if end < 0:
                break
            raw_value = body[i + 1 : end]
            i = end + 1
        elif ch == '"':
            end = i + 1
            while end < n and body[end] != '"':
                if body[end] == "\\" and end + 1 < n:
                    end += 2
                    continue
                end += 1
            if end >= n:
                break
            raw_value = body[i + 1 : end]
            i = end + 1
        else:
            m2 = re.match(r"([^\s,]+)", body[i:])
            if not m2:
                break
            raw_value = m2.group(1)
            i += m2.end()

        fields[name] = _clean_latex(raw_value)
    return fields


# ---------------------------------------------------------------------- #
# Field cleanup + citation rendering
# ---------------------------------------------------------------------- #
_LATEX_CMD_WITH_ARG = re.compile(r"\\[a-zA-Z]+\s*\{([^{}]*)\}")
_LATEX_BARE_CMD = re.compile(r"\\[a-zA-Z]+")
_LATEX_ESCAPED_CHAR = re.compile(r"\\([&%_$#{}\\])")

# LaTeX accent commands → Unicode combining marks. We compose the
# resulting <base><combining> pair with NFC so the output uses the
# precomposed glyph (``é``) rather than ``e\u0301`` whenever one exists.
_LATEX_ACCENT_TO_COMBINING = {
    "'":  "\u0301",   # acute      (\'e -> é)
    "`":  "\u0300",   # grave      (\`e -> è)
    "^":  "\u0302",   # circumflex (\^e -> ê)
    '"':  "\u0308",   # diaeresis  (\"o -> ö)
    "~":  "\u0303",   # tilde      (\~n -> ñ)
    "=":  "\u0304",   # macron     (\=a -> ā)
    ".":  "\u0307",   # dot above  (\.z -> ż)
    "u":  "\u0306",   # breve      (\u{a} -> ă)
    "v":  "\u030c",   # caron      (\v{c} -> č)
    "H":  "\u030b",   # dbl acute  (\H{o} -> ő)
    "c":  "\u0327",   # cedilla    (\c{c} -> ç)
    "k":  "\u0328",   # ogonek     (\k{a} -> ą)
    "d":  "\u0323",   # dot below  (\d{a} -> ạ)
    "b":  "\u0331",   # macron blw (\b{a} -> a̱)
}
# Standalone LaTeX letter commands that map to a single Unicode codepoint.
_LATEX_LETTER_LITERALS = {
    "\\ss": "ß",
    "\\ae": "æ", "\\AE": "Æ",
    "\\oe": "œ", "\\OE": "Œ",
    "\\o":  "ø", "\\O":  "Ø",
    "\\l":  "ł", "\\L":  "Ł",
    "\\aa": "å", "\\AA": "Å",
    "\\i":  "\u0131", "\\j": "\u0237",   # dotless i / j
}
_LATEX_LETTER_LITERAL_RE = re.compile(
    r"\\(ss|ae|AE|oe|OE|aa|AA|[OoLlIiJj])(?![a-zA-Z])"
)
# ``\<acc>{<letter>}`` and ``\<acc>{\i}`` / ``\<acc>{\j}`` (dotless).
_LATEX_ACCENT_BRACED_RE = re.compile(
    r"""\\(['`^"~=.uvHckdb])\s*\{\s*(?:\\([ij])|([a-zA-Z]))\s*\}"""
)
# ``\<acc><letter>`` for the symbol-style accents only (letter accents
# like ``\u``, ``\v``, ``\c`` always require braces in BibTeX).
_LATEX_ACCENT_BARE_RE = re.compile(r"""\\(['`^"~=.])([a-zA-Z])""")


def _decode_latex_accents(text: str) -> str:
    """Convert LaTeX accent escapes to their Unicode equivalents."""
    def _braced(match: "re.Match[str]") -> str:
        accent = match.group(1)
        combining = _LATEX_ACCENT_TO_COMBINING.get(accent)
        if combining is None:
            return match.group(0)
        if match.group(2):
            base = "i" if match.group(2) == "i" else "j"
        else:
            base = match.group(3)
        return unicodedata.normalize("NFC", base + combining)

    def _bare(match: "re.Match[str]") -> str:
        combining = _LATEX_ACCENT_TO_COMBINING.get(match.group(1))
        if combining is None:
            return match.group(0)
        return unicodedata.normalize("NFC", match.group(2) + combining)

    text = _LATEX_ACCENT_BRACED_RE.sub(_braced, text)
    text = _LATEX_ACCENT_BARE_RE.sub(_bare, text)
    text = _LATEX_LETTER_LITERAL_RE.sub(
        lambda m: _LATEX_LETTER_LITERALS["\\" + m.group(1)], text
    )
    return text


def _clean_latex(text: str) -> str:
    """Strip the LaTeX punctuation that BibTeX values are loaded with."""
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    # Decode accent and letter escapes BEFORE the generic ``\cmd`` stripper
    # so commands like ``\'`` and ``\ss`` aren't swept away as no-ops.
    text = _decode_latex_accents(text)
    # Iterate ``\cmd{x}`` -> ``x`` until convergence so that nested commands
    # like ``\textbf{\textit{x}}`` collapse cleanly.
    while True:
        new = _LATEX_CMD_WITH_ARG.sub(r"\1", text)
        if new == text:
            break
        text = new
    text = _LATEX_BARE_CMD.sub("", text)
    text = _LATEX_ESCAPED_CHAR.sub(r"\1", text)
    text = text.replace("~", " ")
    text = re.sub(r"-{2,3}", "-", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _format_authors(raw: str) -> str:
    """Convert BibTeX's ``First1 Last1 and First2 Last2`` to ``Last1, F.; Last2, F.``."""
    names = [n.strip() for n in re.split(r"\s+and\s+", raw) if n.strip()]
    formatted: List[str] = []
    for name in names:
        if "," in name:
            # Already in "Last, First" form; keep as-is.
            formatted.append(name)
            continue
        parts = name.split()
        if len(parts) == 1:
            formatted.append(parts[0])
        else:
            last = parts[-1]
            initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
            formatted.append(f"{last}, {initials}")
    return "; ".join(formatted)


def _render_citation(entry: Dict[str, str]) -> str:
    """Build a free-text citation that mirrors typical PDF-extracted output."""
    parts: List[str] = []

    authors = entry.get("author") or entry.get("editor")
    if authors:
        parts.append(_format_authors(authors))

    year = entry.get("year")
    if year:
        parts.append(f"({year})")

    title = entry.get("title")
    if title:
        parts.append(title.rstrip(".") + ".")

    for venue_field in _VENUE_FIELDS:
        venue = entry.get(venue_field)
        if venue:
            parts.append(venue.rstrip(".") + ".")
            break

    doi = entry.get("doi")
    if doi:
        parts.append(f"doi:{doi}")

    return " ".join(p for p in parts if p).strip()
