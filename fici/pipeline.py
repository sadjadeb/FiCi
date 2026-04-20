"""Top-level orchestrator that glues the four FiCi phases together."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from .extractor import ReferenceExtractor
from .models import CitationReport, Verdict
from .searcher import CitationSearcher
from .verifier import CitationVerifier


logger = logging.getLogger(__name__)

# Conservative default — OpenAlex's polite pool and Crossref both tolerate
# ~10 req/s. Four workers keeps us comfortably below that while still giving
# a large speed-up over sequential execution for typical 20-60 entry
# bibliographies.
_DEFAULT_MAX_WORKERS = 4


class FiCiPipeline:
    """End-to-end pipeline: PDF -> references -> API search -> verdicts.

    Parameters
    ----------
    email:
        Contact email used for the OpenAlex polite pool (and Crossref
        ``User-Agent``). Strongly recommended.
    verify_threshold:
        Score cutoff for the verdict classifier — matches at or above this
        value are marked ``Verified``, everything else ``Suspicious`` (see
        :class:`fici.verifier.CitationVerifier`).
    max_workers:
        Default number of concurrent API workers used by :meth:`run`. Set to
        ``1`` to disable concurrency entirely. Can be overridden per call
        via the ``max_workers`` argument of :meth:`run`.
    extractor, searcher, verifier:
        Optionally inject custom component instances (useful for testing).

    Example
    -------
        pipeline = FiCiPipeline(email="you@example.org")
        reports = pipeline.run("paper.pdf")
        for r in reports:
            print(r.index, r.verdict.value, r.score, r.suspected_title)
    """

    def __init__(
        self,
        email: Optional[str] = None,
        *,
        openalex_api_key: Optional[str] = None,
        verify_threshold: float = 90.0,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        extractor: Optional[ReferenceExtractor] = None,
        searcher: Optional[CitationSearcher] = None,
        verifier: Optional[CitationVerifier] = None,
    ) -> None:
        self.extractor = extractor or ReferenceExtractor()
        self.searcher = searcher or CitationSearcher(email=email, openalex_api_key=openalex_api_key)
        self.verifier = verifier or CitationVerifier(verify_threshold=verify_threshold)
        self.max_workers = max(1, int(max_workers))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        pdf_path: str | Path,
        *,
        progress: Optional[Callable[[int, int, CitationReport], None]] = None,
        max_workers: Optional[int] = None,
    ) -> List[CitationReport]:
        """Execute the full pipeline against ``pdf_path``.

        Parameters
        ----------
        pdf_path:
            Path to the PDF on disk.
        progress:
            Optional callback ``(completed_count, total, report)`` invoked
            after each citation is verified. With concurrency enabled the
            callback may fire in completion order (not document order); use
            ``report.index`` to recover the original ordering.
        max_workers:
            Override the pipeline's default worker count for this call.
            ``1`` forces sequential execution.

        Returns
        -------
        list[CitationReport]
            One report per extracted citation, in document order.
        """
        references = self.extractor.extract(pdf_path)
        total = len(references)
        logger.info("Extracted %d candidate references from %s", total, pdf_path)

        workers = self._resolve_workers(max_workers, total)

        if workers <= 1 or total <= 1:
            return self._run_sequential(references, total, progress)
        return self._run_concurrent(references, total, progress, workers)

    def check_reference(self, raw_citation: str, *, index: int = 0) -> CitationReport:
        """Run search + verification on a single reference string.

        Query strategy:

        1. **OpenAlex** — broadest CS coverage, polite-pool access.
        2. **Crossref** — strong DOI / bibliographic coverage; catches
           published papers OpenAlex may have indexed poorly.
        3. **arXiv**    — preprints that neither of the above may have
           ingested yet.

        After each step we run the verifier. If the verdict is ``Verified``
        we return immediately; otherwise we advance to the next backend and
        finally return whichever report across all three is strongest
        (``Verified`` beats anything else; within the same tier the higher
        score wins).

        Exceptions during any individual backend call are caught inside
        :meth:`_verify_with_source` and surface as ``ERROR`` verdicts, so a
        single failing reference never aborts the full scan.
        """
        reports: List[CitationReport] = []

        for source, fetch in (
            ("openalex", self.searcher.search_openalex),
            ("crossref", self.searcher.search_crossref),
            ("arxiv", self.searcher.search_arxiv),
        ):
            report = self._verify_with_source(
                raw_citation,
                index=index,
                source=source,
                fetch=fetch,
            )
            if report.verdict is Verdict.VERIFIED:
                return report
            reports.append(report)
            logger.debug(
                "%s verdict for ref %d was %s (score=%.1f); escalating.",
                source, index, report.verdict.value, report.score,
            )

        # None of the three backends verified — return the strongest report.
        best = reports[0]
        for r in reports[1:]:
            best = self._pick_better_report(best, r)
        return best

    def _verify_with_source(
        self,
        raw_citation: str,
        *,
        index: int,
        source: str,
        fetch: Callable[[str], list],
    ) -> CitationReport:
        """Fetch hits from a single backend and run the verifier on them.

        On backend error, returns a report whose verdict is ``ERROR`` so the
        caller can still choose between sources.
        """
        try:
            hits = fetch(raw_citation)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("%s search failed for reference %d: %s", source, index, exc)
            return CitationReport(
                index=index,
                raw_text=raw_citation,
                verdict=Verdict.ERROR,
                score=0.0,
                reason=f"{source} backend raised: {exc!r}",
                source_used=source,
            )
        return self.verifier.verify(index, raw_citation, hits)

    @staticmethod
    def _pick_better_report(a: CitationReport, b: CitationReport) -> CitationReport:
        """Return whichever of two reports represents a stronger conclusion.

        Priority order:
            1. ``VERIFIED`` always beats anything else.
            2. A real verdict (``SUSPICIOUS``) beats ``ERROR``.
            3. Within the same tier, the higher score wins.
        """
        rank = {
            Verdict.VERIFIED: 2,
            Verdict.SUSPICIOUS: 1,
            Verdict.ERROR: 0,
        }
        a_rank = rank.get(a.verdict, 0)
        b_rank = rank.get(b.verdict, 0)
        if a_rank != b_rank:
            return a if a_rank > b_rank else b
        return a if a.score >= b.score else b

    # ------------------------------------------------------------------ #
    # Execution strategies
    # ------------------------------------------------------------------ #
    def _run_sequential(
        self,
        references: List[str],
        total: int,
        progress: Optional[Callable[[int, int, CitationReport], None]],
    ) -> List[CitationReport]:
        reports: List[CitationReport] = []
        for i, raw in enumerate(references, start=1):
            report = self.check_reference(raw, index=i)
            reports.append(report)
            if progress is not None:
                progress(i, total, report)
        return reports

    def _run_concurrent(
        self,
        references: List[str],
        total: int,
        progress: Optional[Callable[[int, int, CitationReport], None]],
        workers: int,
    ) -> List[CitationReport]:
        """Dispatch reference checks across a thread pool.

        The underlying work is I/O bound (HTTP + JSON parsing), so threads
        are the right primitive — we get parallel network waits without
        fighting the GIL. ``requests.Session`` is safe to share for
        concurrent GETs in practice, so the pipeline's single
        :class:`CitationSearcher` is reused across workers.
        """
        results: List[Optional[CitationReport]] = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fici") as pool:
            future_to_index = {
                pool.submit(self.check_reference, raw, index=i): i - 1
                for i, raw in enumerate(references, start=1)
            }
            for future in as_completed(future_to_index):
                slot = future_to_index[future]
                try:
                    report = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Worker raised unexpectedly: %s", exc)
                    report = CitationReport(
                        index=slot + 1,
                        raw_text=references[slot],
                        verdict=Verdict.ERROR,
                        score=0.0,
                        reason=f"Worker raised: {exc!r}",
                    )
                results[slot] = report
                completed += 1
                if progress is not None:
                    progress(completed, total, report)

        # All slots are populated by the time we exit the pool; the cast is
        # just for the type-checker.
        return [r for r in results if r is not None]

    def _resolve_workers(self, override: Optional[int], total: int) -> int:
        requested = self.max_workers if override is None else int(override)
        requested = max(1, requested)
        # No point in spinning up more threads than there are references.
        return min(requested, max(1, total))

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def summarize(reports: List[CitationReport]) -> dict:
        """Aggregate verdict counts across a list of reports."""
        counts = {v.value: 0 for v in Verdict}
        for r in reports:
            counts[r.verdict.value] = counts.get(r.verdict.value, 0) + 1
        counts["total"] = len(reports)
        return counts
