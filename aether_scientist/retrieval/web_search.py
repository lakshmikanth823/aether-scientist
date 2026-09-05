import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WebHit:
    """Represents a search hit from the web."""

    title: str
    url: str
    snippet: str


class WebSearchEngine:
    """Lightweight web search engine with graceful degradation."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self.headers = {"User-Agent": "AetherScientist/1.3 (research-agent)"}

    def _search_ddg(self, query: str, max_results: int) -> list[WebHit]:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        hits: list[WebHit] = []
        for r in soup.select(".result"):
            a = r.select_one(".result__title a")
            snip = r.select_one(".result__snippet")
            if a and a.get("href"):
                url = str(a["href"])
                if url.startswith("//duckduckgo.com/l/?uddg="):
                    from urllib.parse import parse_qs, urlparse

                    qs = parse_qs(urlparse("https:" + url).query)
                    url = qs.get("uddg", [url])[0]
                title = a.get_text(strip=True)
                snippet = snip.get_text(strip=True) if snip else ""
                hits.append(WebHit(title=title, url=url, snippet=snippet))
                if len(hits) >= max_results:
                    break
        return hits

    def _search_wiki(self, query: str, max_results: int) -> list[WebHit]:
        import httpx

        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": query, "limit": max_results, "format": "json"}
        resp = httpx.get(url, params=params, headers=self.headers, timeout=self.timeout)
        if resp.status_code != 200:
            return []
        data = resp.json()
        titles, snippets, urls = data[1], data[2], data[3]
        return [
            WebHit(title=t, url=u, snippet=s)
            for t, s, u in zip(titles, snippets, urls, strict=False)
            if u
        ]

    def search(self, query: str, max_results: int = 3) -> list[WebHit]:
        """Search the web for relevant pages, returning top hits."""
        if not query.strip():
            return []
        try:
            hits = self._search_ddg(query, max_results)
            if not hits:
                hits = self._search_wiki(query, max_results)
            return hits[:max_results]
        except Exception as e:
            logger.warning(f"Web search failed for query '{query}': {e}")
            return []
