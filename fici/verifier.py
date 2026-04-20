"""Phase 4: verify API responses against the raw citation with rapidfuzz.

Scoring model (all values are 0-100):

    title_score = rapidfuzz.fuzz.token_sort_ratio(api_title, suspected_title)
    author_bonus = +up to 10 if any API author surname appears in the raw text
    final_score  = min(100, title_score + author_bonus)

Verdicts:
    * "Highly Likely Fake"    -> no results from either API.
    * "Suspicious/Mismatch"   -> results found but final_score < mismatch_threshold.
    * "Verified"              -> final_score >= verify_threshold.
    * Scores between the two thresholds are treated as suspicious as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rapidfuzz import fuzz, utils

from ._parsing import LEADING_MARKER, extract_suspected_title
from .models import CitationReport, SearchHit, Verdict


# Minimum title length (in characters) before we allow partial_ratio to
# contribute to scoring. Short, generic titles ("Introduction", "Results")
# would otherwise false-match almost any API response.
_PARTIAL_RATIO_MIN_LEN = 20

# Preprocessor applied to every rapidfuzz comparison. ``default_process``
# lowercases, replaces non-alphanumeric characters with spaces, and strips
# whitespace. This is important because OpenAlex/Crossref sometimes return
# titles in ALL CAPS, with curly quotes, or with differing punctuation
# (e.g. trailing periods, em-dashes) compared to the cited form, and the
# raw rapidfuzz ratios are case- and punctuation-sensitive.
_FUZZ_PROCESSOR = utils.default_process


@dataclass
class VerifierConfig:
    """Thresholds for verdict assignment."""

    verify_threshold: float = 85.0     # >= this -> Verified
    mismatch_threshold: float = 75.0   # < this -> Suspicious
    author_bonus_max: float = 10.0


class CitationVerifier:
    """Evaluate candidate hits against a raw reference string."""

    def __init__(
        self,
        *,
        verify_threshold: float = 85.0,
        mismatch_threshold: float = 75.0,
        author_bonus_max: float = 10.0,
    ) -> None:
        self.config = VerifierConfig(
            verify_threshold=verify_threshold,
            mismatch_threshold=mismatch_threshold,
            author_bonus_max=author_bonus_max,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def verify(
        self,
        index: int,
        raw_citation: str,
        hits: List[SearchHit],
    ) -> CitationReport:
        """Produce a :class:`CitationReport` for ``raw_citation``."""
        suspected_title = extract_suspected_title(raw_citation)

        if not hits:
            return CitationReport(
                index=index,
                raw_text=raw_citation,
                suspected_title=suspected_title,
                verdict=Verdict.LIKELY_FAKE,
                score=0.0,
                reason="No results returned from OpenAlex or Crossref.",
                best_hit=None,
                candidates_considered=0,
                source_used=None,
            )

        best_hit, best_score = self._pick_best_hit(raw_citation, suspected_title, hits)
        verdict, reason = self._classify(best_score, best_hit)

        return CitationReport(
            index=index,
            raw_text=raw_citation,
            suspected_title=suspected_title,
            verdict=verdict,
            score=round(best_score, 2),
            reason=reason,
            best_hit=best_hit,
            candidates_considered=len(hits),
            source_used=best_hit.source if best_hit else None,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _pick_best_hit(
        self,
        raw_citation: str,
        suspected_title: Optional[str],
        hits: List[SearchHit],
    ) -> Tuple[Optional[SearchHit], float]:
        cleaned_raw = LEADING_MARKER.sub("", raw_citation).strip()
        best: Optional[SearchHit] = None
        best_score = 0.0

        for hit in hits:
            if not hit.title:
                continue

            # Title similarity is computed from several angles and we keep
            # the best. Each metric handles a different failure mode:
            #   - token_sort_ratio: strict full-string match.
            #   - token_set_ratio:  forgiving of extra tokens on either side.
            #   - partial_ratio:    handles the common case where the citation
            #                       omits a subtitle (e.g. "On the Dangers of
            #                       Stochastic Parrots" vs "... : Can Language
            #                       Models Be Too Big?"). Gated by a minimum
            #                       length on the shorter string so we don't
            #                       false-match short generic prefixes.
            #   - token_set_ratio against the full raw string: safety net for
            #     cases where title extraction misfires.
            title_vs_suspected = 0.0
            if suspected_title:
                scores = [
                    fuzz.token_sort_ratio(hit.title, suspected_title, processor=_FUZZ_PROCESSOR),
                    fuzz.token_set_ratio(hit.title, suspected_title, processor=_FUZZ_PROCESSOR),
                ]
                if min(len(hit.title), len(suspected_title)) >= _PARTIAL_RATIO_MIN_LEN:
                    scores.append(
                        fuzz.partial_ratio(hit.title, suspected_title, processor=_FUZZ_PROCESSOR)
                    )
                title_vs_suspected = max(scores)
            title_vs_raw = fuzz.token_set_ratio(hit.title, cleaned_raw, processor=_FUZZ_PROCESSOR)
            title_score = max(title_vs_suspected, title_vs_raw)

            # Secondary signal: does any API author surname appear in the raw
            # reference? Small bonus to break ties and reward corroborating
            # metadata.
            author_bonus = self._author_bonus(hit, cleaned_raw)

            combined = min(100.0, title_score + author_bonus)
            if combined > best_score:
                best = hit
                best_score = combined

        return best, best_score

    def _author_bonus(self, hit: SearchHit, cleaned_raw: str) -> float:
        if not hit.authors:
            return 0.0
        raw_lower = cleaned_raw.lower()
        surnames = {self._surname(a).lower() for a in hit.authors if a}
        surnames.discard("")
        if not surnames:
            return 0.0
        matches = sum(1 for s in surnames if s and s in raw_lower)
        if matches == 0:
            return 0.0
        ratio = matches / len(surnames)
        return round(self.config.author_bonus_max * ratio, 2)

    @staticmethod
    def _surname(full_name: str) -> str:
        """Best-effort surname extraction."""
        parts = [p for p in full_name.strip().split() if p]
        return parts[-1] if parts else ""

    def _classify(
        self, score: float, best_hit: Optional[SearchHit]
    ) -> Tuple[Verdict, str]:
        if best_hit is None:
            return (
                Verdict.SUSPICIOUS,
                "API returned results but none contained a usable title.",
            )
        if score >= self.config.verify_threshold:
            return Verdict.VERIFIED, (
                f"Title similarity {score:.1f} ≥ {self.config.verify_threshold:.0f}."
            )
        if score >= self.config.mismatch_threshold:
            return Verdict.SUSPICIOUS, (
                f"Borderline match: score {score:.1f} is between "
                f"{self.config.mismatch_threshold:.0f} and {self.config.verify_threshold:.0f}."
            )
        return Verdict.SUSPICIOUS, (
            f"Low title similarity ({score:.1f} < {self.config.mismatch_threshold:.0f}); "
            "closest match appears to differ from the cited work."
        )

