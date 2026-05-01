"""FiCi: a lightweight fake-citation detector for scientific PDFs and BibTeX.

Public API:
    FiCiPipeline       - orchestrates extraction, search, and verification.
    ReferenceExtractor - Phase 1: PDF / .bib / .bbl -> raw citation strings.
    CitationSearcher   - Phases 2+3: OpenAlex (primary) + Crossref / arXiv /
                         Semantic Scholar fallbacks.
    CitationVerifier   - Phase 4: fuzzy matching + verdict scoring.
    CitationReport     - per-citation result record.
    Verdict            - enum of verdict labels.
    parse_bibtex_file  - load a BibTeX file directly into citation strings.
"""

from .models import CitationReport, SearchHit, Verdict
from .extractor import ReferenceExtractor
from .searcher import CitationSearcher
from .verifier import CitationVerifier
from .pipeline import FiCiPipeline
from .bibtex import parse_bibtex_file

__all__ = [
    "FiCiPipeline",
    "ReferenceExtractor",
    "CitationSearcher",
    "CitationVerifier",
    "CitationReport",
    "SearchHit",
    "Verdict",
    "parse_bibtex_file",
]

__version__ = "0.1.0"
