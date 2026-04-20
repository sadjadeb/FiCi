"""FiCi: a lightweight fake-citation detector for scientific PDFs.

Public API:
    FiCiPipeline       - orchestrates extraction, search, and verification.
    ReferenceExtractor - Phase 1: PDF -> list of raw citation strings.
    CitationSearcher   - Phases 2+3: OpenAlex (primary) + Crossref (fallback).
    CitationVerifier   - Phase 4: fuzzy matching + verdict scoring.
    CitationReport     - per-citation result record.
    Verdict            - enum of verdict labels.
"""

from .models import CitationReport, SearchHit, Verdict
from .extractor import ReferenceExtractor
from .searcher import CitationSearcher
from .verifier import CitationVerifier
from .pipeline import FiCiPipeline

__all__ = [
    "FiCiPipeline",
    "ReferenceExtractor",
    "CitationSearcher",
    "CitationVerifier",
    "CitationReport",
    "SearchHit",
    "Verdict",
]

__version__ = "0.1.0"
