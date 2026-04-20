"""Phases 2 & 3: structure a raw citation and search external databases.

Primary backend:        OpenAlex  (https://api.openalex.org/works).
Secondary backend:      Crossref  (https://api.crossref.org/works).
Tertiary backend:       arXiv     (http://export.arxiv.org/api/query).

All three APIs accept the extracted title as a free-text query, which lets
us avoid a structured-parse dependency (e.g. ``anystyle``). We strip
leading enumeration markers like ``[12]`` and trailing URLs/DOIs before
querying, and extract the paper title when possible so the search isn't
polluted by author names or venue strings.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlencode

import requests

from ._parsing import extract_suspected_title, strip_markers
from .models import SearchHit


logger = logging.getLogger(__name__)

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"

# Upper bound on the query length sent to each API. OpenAlex in particular
# penalises very long `search` strings.
_MAX_QUERY_LEN = 300

# arXiv asks clients to wait ~3 seconds between requests. Because arXiv is a
# third-tier fallback that only fires for already-unverified citations, we use
# a smaller sleep but still larger than the default 100 ms per-worker pause.
_ARXIV_REQUEST_SLEEP = 1.0

# Atom / arXiv XML namespaces.
_ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class SearcherConfig:
    """Configuration for the searcher (timeouts, retries, politeness)."""

    email: Optional[str] = None
    openalex_api_key: Optional[str] = None
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
    openalex_api_key:
        Optional OpenAlex premium API key. When provided, it is sent as the
        ``api_key`` query parameter on OpenAlex requests, bypassing the
        polite-pool daily cap for users who have hit their limit.
    """

    def __init__(
        self,
        email: Optional[str] = None,
        *,
        openalex_api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: float = 15.0,
        retries: int = 2,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = SearcherConfig(
            email=email,
            openalex_api_key=openalex_api_key,
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

        Tries OpenAlex first; if it returns no hits (or errors) falls back
        to Crossref, then to arXiv. The returned list may be empty if none
        of the backends resolves the string.

        Note: the :class:`FiCiPipeline` prefers calling the per-source
        ``search_*`` methods separately so it can escalate only when a
        backend fails to **verify** (not just when it returns nothing).
        This method is kept for standalone use.
        """
        hits = self.search_openalex(raw_citation)
        if hits:
            return hits
        logger.debug("OpenAlex returned no hits; falling back to Crossref.")
        hits = self.search_crossref(raw_citation)
        if hits:
            return hits
        logger.debug("Crossref returned no hits; falling back to arXiv.")
        return self.search_arxiv(raw_citation)

    def search_openalex(self, raw_citation: str) -> List[SearchHit]:
        """Search OpenAlex only. Returns [] if the query is empty or on error."""
        query = self._prepare_query(raw_citation)
        if not query:
            return []
        return self._search_openalex(query)

    def search_crossref(self, raw_citation: str) -> List[SearchHit]:
        """Search Crossref only. Returns [] if the query is empty or on error."""
        query = self._prepare_query(raw_citation)
        if not query:
            return []
        return self._search_crossref(query)

    def search_arxiv(self, raw_citation: str) -> List[SearchHit]:
        """Search arXiv only. Returns [] if the query is empty or on error."""
        query = self._prepare_query(raw_citation)
        if not query:
            return []
        return self._search_arxiv(query)

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
        if self.config.openalex_api_key:
            # Premium / authenticated access — raises the daily cap once the
            # polite pool is exhausted.
            params["api_key"] = self.config.openalex_api_key

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
    # arXiv backend
    # ------------------------------------------------------------------ #
    def _search_arxiv(self, query: str) -> List[SearchHit]:
        # arXiv supports field-prefixed queries; we scope to titles so author
        # names and abstracts don't pollute ranking. The title itself is
        # wrapped in double quotes to request a phrase match; arXiv gracefully
        # degrades to a token search if the exact phrase isn't present.
        sanitized = query.replace('"', "")
        search_query = f'ti:"{sanitized}"'
        params = {
            "search_query": search_query,
            "max_results": str(self.config.max_results),
        }
        url = f"{ARXIV_ENDPOINT}?{urlencode(params)}"

        resp = self._request_response(
            url,
            source="arxiv",
            accept="application/atom+xml",
            request_sleep=_ARXIV_REQUEST_SLEEP,
        )
        if resp is None:
            return []

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            logger.warning("arxiv returned malformed XML: %s", exc)
            return []

        entries = root.findall("atom:entry", _ARXIV_NAMESPACES)
        return [self._arxiv_to_hit(e) for e in entries]

    @staticmethod
    def _arxiv_to_hit(entry: ET.Element) -> SearchHit:
        ns = _ARXIV_NAMESPACES

        title_el = entry.find("atom:title", ns)
        title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else None

        authors: List[str] = []
        for a in entry.findall("atom:author", ns):
            name_el = a.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        year = None
        published = entry.find("atom:published", ns)
        if published is not None and published.text and len(published.text) >= 4:
            try:
                year = int(published.text[:4])
            except ValueError:
                year = None

        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        id_el = entry.find("atom:id", ns)
        id_url = id_el.text.strip() if id_el is not None and id_el.text else None

        return SearchHit(
            source="arxiv",
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            venue="arXiv",
            id_url=id_url,
            raw={"xml": ET.tostring(entry, encoding="unicode")},
        )

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #
    def _request_json(self, url: str, *, source: str) -> Optional[dict]:
        resp = self._request_response(url, source=source, accept="application/json")
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            logger.warning("%s returned non-JSON body: %s", source, exc)
            return None

    def _request_response(
        self,
        url: str,
        *,
        source: str,
        accept: str = "application/json",
        request_sleep: Optional[float] = None,
    ) -> Optional[requests.Response]:
        """Execute a GET with retry/backoff, returning the Response or None.

        Parameters
        ----------
        request_sleep:
            Per-call override for the post-success politeness sleep. When
            ``None`` the searcher's configured ``request_sleep`` is used.
        """
        headers = {"Accept": accept, "User-Agent": self._user_agent()}
        sleep_after = self.config.request_sleep if request_sleep is None else request_sleep
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.retries + 1):
            try:
                resp = self._session.get(url, headers=headers, timeout=self.config.timeout)
                # 429/5xx are retryable; other 4xx statuses indicate a bad query.
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}")
                if resp.status_code >= 400:
                    logger.warning("%s returned HTTP %s for query.", source, resp.status_code)
                    return None
                if sleep_after:
                    time.sleep(sleep_after)
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.config.retries:
                    sleep_for = self.config.backoff ** attempt
                    logger.debug(
                        "%s attempt %d failed (%s); retrying in %.1fs",
                        source, attempt + 1, exc, sleep_for,
                    )
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
