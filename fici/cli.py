"""Command-line entry point for FiCi.

Registered as the ``fici`` console script in ``pyproject.toml``:

    fici paper.pdf --email you@example.org

Also supports module execution:

    python -m fici paper.pdf --email you@example.org
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import urlparse

import requests

from . import __version__
from .models import CitationReport, Verdict
from .pipeline import FiCiPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fici",
        description=(
            "FiCi: detect fake/hallucinated citations in a scientific PDF "
            "or BibTeX bibliography. Extracts the references, queries "
            "OpenAlex (with Crossref / arXiv / Semantic Scholar fallbacks), "
            "and scores each entry with rapidfuzz."
        ),
    )
    parser.add_argument(
        "input",
        help=(
            "Input: a paper PDF, a BibTeX source (.bib / .bbl), a "
            "plain-text bullet/line list (.txt), or an http(s) URL to a "
            "PDF on the web (e.g. an arXiv link)."
        ),
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Contact email for OpenAlex polite pool (strongly recommended).",
    )
    parser.add_argument(
        "--openalex-api-key",
        default=None,
        help=(
            "OpenAlex premium API key, forwarded as the `api_key` query "
            "parameter. Use this once the polite-pool daily limit has been "
            "reached."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent API workers (default: 4; set 1 to disable).",
    )
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=90.0,
        help=(
            "Score >= this is marked Verified; anything below is "
            "Suspicious/Mismatch (default: 90)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full reports as JSON on stdout (disables progress output).",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help=(
            "After the run, write a Markdown report to the current working "
            "directory as <pdf-basename>-fici-<timestamp>.md (timestamp is "
            "YYYYMMDD-HHMMSS local time). Written in addition to stdout."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-reference progress lines.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"fici {__version__}"
    )
    return parser


def _best_hit_url(report: CitationReport) -> str:
    """Return the most useful URL for the matched hit, or ``<no match>``."""
    hit = report.best_hit
    if hit is None:
        return "<no match>"
    if hit.id_url:
        return hit.id_url
    if hit.doi:
        return f"https://doi.org/{hit.doi}"
    return "<no link>"


# Column width for the title field in progress output — truncated with an
# ellipsis when longer, padded with spaces when shorter, so the ``url=``
# column always lines up.
_TITLE_COL_WIDTH = 80


def _fixed_width(text: str, width: int) -> str:
    """Truncate with ``…`` or right-pad so ``text`` occupies exactly ``width`` chars."""
    if len(text) > width:
        return text[: max(0, width - 1)] + "…"
    return text.ljust(width)


def _print_progress(done: int, total: int, report: CitationReport) -> None:
    # With concurrency, completion order != document order; print both the
    # completion counter and the original citation index.
    title = _fixed_width(report.suspected_title or "<unknown>", _TITLE_COL_WIDTH)
    # Only surface the source URL for verified matches — for suspicious /
    # likely-fake / errored reports the "best hit" is by definition not
    # trustworthy and printing its link would be misleading.
    url_suffix = (
        f"  url={_best_hit_url(report)}" if report.verdict is Verdict.VERIFIED else ""
    )
    sys.stderr.write(
        f"[{done:>3}/{total}] ref#{report.index:<3} "
        f"{report.verdict.value:<22} "
        f"score={report.score:>5.1f}  "
        f"title={title}"
        f"{url_suffix}\n"
    )
    sys.stderr.flush()


def _print_human_summary(reports: List[CitationReport]) -> None:
    summary = FiCiPipeline.summarize(reports)
    print("\n=== FiCi Summary ===")
    for label, count in summary.items():
        print(f"  {label:<22} {count}")

    flagged = [r for r in reports if r.verdict.value != "Verified"]
    if flagged:
        print("\nFlagged citations:")
        for r in flagged:
            snippet = r.raw_text[:120] + ("..." if len(r.raw_text) > 120 else "")
            print(f"  [{r.index}] {r.verdict.value} (score={r.score}): {snippet}")


def _build_md_report(reports: List[CitationReport], pdf_path: str) -> str:
    """Render *reports* as a Markdown document and return the string."""
    summary = FiCiPipeline.summarize(reports)
    total = summary.pop("total")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# FiCi Citation Report")
    lines.append("")
    lines.append(f"**File:** `{pdf_path}`  ")
    lines.append(f"**Generated:** {timestamp}  ")
    lines.append(f"**Total references:** {total}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---------|------:|")
    for label, count in summary.items():
        lines.append(f"| {label} | {count} |")
    lines.append("")

    lines.append("## All References")
    lines.append("")
    lines.append("| # | Verdict | Score | Source | Title | URL |")
    lines.append("|--:|---------|------:|--------|-------|-----|")
    for r in reports:
        title = (r.suspected_title or r.raw_text[:80]).replace("|", "\\|")
        source = r.source_used or "—"
        # Only surface URLs for verified matches — same rule as progress output.
        if r.verdict is Verdict.VERIFIED:
            url = _best_hit_url(r)
            url_cell = f"[link]({url})" if url.startswith("http") else url
        else:
            url_cell = "—"
        lines.append(
            f"| {r.index} | {r.verdict.value} | {r.score:.1f} | {source} | {title} | {url_cell} |"
        )
    lines.append("")

    flagged = [r for r in reports if r.verdict is not Verdict.VERIFIED]
    if flagged:
        lines.append("## Flagged Citations")
        lines.append("")
        for r in flagged:
            lines.append(f"### [{r.index}] {r.verdict.value} (score={r.score:.1f})")
            lines.append("")
            lines.append(f"> {r.raw_text}")
            lines.append("")
            if r.reason:
                lines.append(f"**Reason:** {r.reason}")
                lines.append("")
            url = _best_hit_url(r)
            if url.startswith("http"):
                lines.append(f"**Closest match:** {url}")
                lines.append("")

    return "\n".join(lines)


def _default_md_report_path(input_ref: str) -> Path:
    """Return ``<cwd>/<stem>-fici-<YYYYMMDD-HHMMSS>.md`` for *input_ref*.

    ``input_ref`` may be a local path or an ``http(s)://`` URL; in the
    URL case the basename of the URL path (sans query/fragment) is used
    as the report stem.
    """
    if re.match(r"^https?://", input_ref, re.IGNORECASE):
        url_path = urlparse(input_ref).path
        stem = Path(url_path).stem or "report"
    else:
        stem = Path(input_ref).resolve().stem or "report"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / f"{stem}-fici-{ts}.md"


def _save_md_report(reports: List[CitationReport], path: Path, pdf_path: str) -> None:
    """Write the Markdown report to *path*."""
    path.write_text(_build_md_report(reports, pdf_path), encoding="utf-8")
    sys.stderr.write(f"fici: report saved to {path.resolve()}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``fici`` console script.

    Returns a POSIX exit code:
        0 - all references verified.
        1 - at least one reference flagged (suspicious / likely fake / error).
        2 - bad input (e.g. PDF not found or unreadable).
    """
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = FiCiPipeline(
        email=args.email,
        openalex_api_key=args.openalex_api_key,
        verify_threshold=args.verify_threshold,
        max_workers=args.workers,
    )

    progress_cb = None if (args.quiet or args.json) else _print_progress

    try:
        reports = pipeline.run(args.input, progress=progress_cb)
    except FileNotFoundError as exc:
        sys.stderr.write(f"fici: {exc}\n")
        return 2
    except (requests.RequestException, ValueError) as exc:
        # Surface clean errors for URL fetches (network failures, HTML
        # landing pages, oversize downloads) without a raw traceback.
        sys.stderr.write(f"fici: failed to load input: {exc}\n")
        return 2
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        sys.stderr.write(f"fici: unexpected error: {exc!r}\n")
        return 2

    if args.json:
        json.dump([r.to_dict() for r in reports], sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        _print_human_summary(reports)

    if args.save_output:
        _save_md_report(reports, _default_md_report_path(args.input), args.input)

    any_flagged = any(r.verdict.value != "Verified" for r in reports)
    return 1 if any_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
