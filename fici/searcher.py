"""Phases 2 & 3: structure a raw citation and search external databases.

Primary backend: OpenAlex (https://api.openalex.org/works).
Fallback:        Crossref bibliographic query (https://api.crossref.org/works).

Both APIs accept the raw citation string as a free-text query, which lets us
avoid a structured-parse dependency (e.g. ``anystyle``). We do apply a small
amount of pre-processing to strip leading enumeration markers like ``[12]``
and trailing URLs that hurt retrieval quality.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlencode

import requests

from ._parsing import extract_suspected_title, strip_markers
from .models import SearchHit


logger = logging.getLogger(__name__)

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"

# Upper bound on the query length sent to either API. OpenAlex in particular
# penalises very long `search` strings.
_MAX_QUERY_LEN = 300


@dataclass
class SearcherConfig:
    """Configuration for the searcher (timeouts, retries, politeness)."""

    email: Optional[str] = None
    user_agent: str = "FiCi/0.1 (+https://github.com/)"
    max_results: int = 5
    timeout: float = 15.0
    retries: int = 2
    backoff: float = 1.5
    request_sleep: float = 0.1  # small delay between calls to be polite


class CitationSearcher:
    """Query OpenAlex (primary) and Crossref (fallback) for a citation string.

    Parameters
    ----------
    email:
        Contact email forwarded to OpenAlex (polite pool) and Crossref
        (``User-Agent`` header). Strongly recommended — without it both APIs
        apply stricter rate limits.
    """

    def __init__(
        self,
        email: Optional[str] = None,
        *,
        max_results: int = 5,
        timeout: float = 15.0,
        retries: int = 2,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = SearcherConfig(
            email=email,
            max_results=max_results,
            timeout=timeout,
            retries=retries,
        )
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(self, raw_citation: str) -> List[SearchHit]:
        """Return a list of candidate hits for ``raw_citation``.

        Tries OpenAlex first; if it returns no hits OR errors, falls back to
        Crossref. The returned list may be empty if neither backend resolves
        the string.
        """
        query = self._prepare_query(raw_citation)
        if not query:
            return []

        hits = self._search_openalex(query)
        if hits:
            return hits

        logger.debug("OpenAlex returned no hits; falling back to Crossref.")
        return self._search_crossref(query)

    # ------------------------------------------------------------------ #
    # Query normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare_query(raw_citation: str) -> str:
        """Extract the most searchable portion of a raw reference string.

        OpenAlex's ``search`` parameter ranks across title, abstract, and
        fulltext — so author names, venue names, and publication years in
        the raw citation dilute the ranking signal and return noisy,
        irrelevant top hits (e.g. querying a Bender et al. 2021 FAccT
        citation can otherwise surface unrelated papers whose abstract
        happens to mention "language models" and "2021").

        We therefore send only the **title** whenever the title-extraction
        heuristic succeeds. If it fails (unusual template, missing year,
        over-aggressive line wrapping), we fall back to the cleaned raw
        string so the call still has a chance of resolving.
        """
        title = extract_suspected_title(raw_citation)
        if title:
            return title[:_MAX_QUERY_LEN]

        # Fallback: strip leading numbering + trailing URLs/DOIs and cap length.
        return strip_markers(raw_citation)[:_MAX_QUERY_LEN]

    # ------------------------------------------------------------------ #
    # OpenAlex backend
    # ------------------------------------------------------------------ #
    def _search_openalex(self, query: str) -> List[SearchHit]:
        params = {
            "search": query,
            "per-page": str(self.config.max_results),
        }
        if self.config.email:
            # Polite pool: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication
            params["mailto"] = self.config.email

        url = f"{OPENALEX_ENDPOINT}?{urlencode(params)}"
        data = self._request_json(url, source="openalex")
        if not data:
            return []

        results = data.get("results", []) or []
        return [self._openalex_to_hit(r) for r in results]

    @staticmethod
    def _openalex_to_hit(result: dict) -> SearchHit:
        title = result.get("title") or result.get("display_name")
        authorships = result.get("authorships") or []
        authors: List[str] = []
        for a in authorships:
            author = (a or {}).get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)

        venue = None
        host = result.get("primary_location") or {}
        source = host.get("source") or {}
        if isinstance(source, dict):
            venue = source.get("display_name")

        return SearchHit(
            source="openalex",
            title=title,
            authors=authors,
            year=result.get("publication_year"),
            doi=(result.get("doi") or "").replace("https://doi.org/", "") or None,
            venue=venue,
            id_url=result.get("id"),
            raw=result,
        )

    # ------------------------------------------------------------------ #
    # Crossref backend
    # ------------------------------------------------------------------ #
    def _search_crossref(self, query: str) -> List[SearchHit]:
        params = {
            "query.bibliographic": query,
            "rows": str(self.config.max_results),
        }
        if self.config.email:
            # Crossref's polite pool uses the User-Agent header, but also
            # accepts a `mailto` query parameter.
            params["mailto"] = self.config.email

        url = f"{CROSSREF_ENDPOINT}?{urlencode(params)}"
        data = self._request_json(url, source="crossref")
        if not data:
            return []

        items = (data.get("message") or {}).get("items", []) or []
        return [self._crossref_to_hit(i) for i in items]

    @staticmethod
    def _crossref_to_hit(item: dict) -> SearchHit:
        titles = item.get("title") or []
        title = titles[0] if titles else None

        authors: List[str] = []
        for a in item.get("author") or []:
            given = a.get("given", "").strip()
            family = a.get("family", "").strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        year = None
        issued = item.get("issued") or {}
        date_parts = issued.get("date-parts") or []
        if date_parts and date_parts[0]:
            try:
                year = int(date_parts[0][0])
            except (ValueError, TypeError):
                year = None

        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else None

        return SearchHit(
            source="crossref",
            title=title,
            authors=authors,
            year=year,
            doi=item.get("DOI"),
            venue=venue,
            id_url=item.get("URL"),
            raw=item,
        )

    # ------------------------------------------------------------------ #
    # HTTP helper
    # ------------------------------------------------------------------ #
    def _request_json(self, url: str, *, source: str) -> Optional[dict]:
        headers = {"Accept": "application/json", "User-Agent": self._user_agent()}
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.retries + 1):
            try:
                resp = self._session.get(url, headers=headers, timeout=self.config.timeout)
                # 429/5xx are retryable; 4xx (other) indicates a bad query.
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}")
                if resp.status_code >= 400:
                    logger.warning("%s returned HTTP %s for query.", source, resp.status_code)
                    return None
                if self.config.request_sleep:
                    time.sleep(self.config.request_sleep)
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < self.config.retries:
                    sleep_for = self.config.backoff ** attempt
                    logger.debug("%s attempt %d failed (%s); retrying in %.1fs", source, attempt + 1, exc, sleep_for)
                    time.sleep(sleep_for)
                    continue
                logger.warning("%s request failed after %d attempts: %s", source, attempt + 1, exc)
        if last_exc is not None:
            logger.debug("Final %s error: %s", source, last_exc)
        return None

    def _user_agent(self) -> str:
        if self.config.email:
            return f"{self.config.user_agent} mailto:{self.config.email}"
        return self.config.user_agent
