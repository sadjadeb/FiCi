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
import sys
from typing import List, Optional, Sequence

from . import __version__
from .models import CitationReport, Verdict
from .pipeline import FiCiPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fici",
        description=(
            "FiCi: detect fake/hallucinated citations in a scientific PDF. "
            "Extracts the bibliography, queries OpenAlex (with Crossref as a "
            "fallback), and scores each reference with rapidfuzz."
        ),
    )
    parser.add_argument("pdf", help="Path to the paper PDF.")
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
        reports = pipeline.run(args.pdf, progress=progress_cb)
    except FileNotFoundError as exc:
        sys.stderr.write(f"fici: {exc}\n")
        return 2
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        sys.stderr.write(f"fici: unexpected error: {exc!r}\n")
        return 2

    if args.json:
        json.dump([r.to_dict() for r in reports], sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        _print_human_summary(reports)

    any_flagged = any(r.verdict.value != "Verified" for r in reports)
    return 1 if any_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
