"""Data containers shared across the FiCi pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    """Verdict labels assigned to a citation after verification."""

    VERIFIED = "Verified"
    SUSPICIOUS = "Suspicious/Mismatch"
    LIKELY_FAKE = "Highly Likely Fake"
    ERROR = "Error"


@dataclass
class SearchHit:
    """Normalized representation of a single result from OpenAlex or Crossref."""

    source: str  # "openalex" or "crossref"
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    id_url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class CitationReport:
    """Full result for a single extracted reference string."""

    index: int
    raw_text: str
    suspected_title: Optional[str] = None
    verdict: Verdict = Verdict.ERROR
    score: float = 0.0
    reason: str = ""
    best_hit: Optional[SearchHit] = None
    candidates_considered: int = 0
    source_used: Optional[str] = None  # "openalex" / "crossref" / None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data
