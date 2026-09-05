import logging
import re

from aether_scientist.retrieval.chunker import Chunk, chunk

logger = logging.getLogger(__name__)

USER_AGENT = "AetherScientist/1.3 (research-agent; +https://github.com/aetherscientist/aether-scientist)"


def clean_html(html: str) -> str:
    """Clean HTML by stripping scripts, styles, navigation, and excessive whitespace."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
    except Exception:
        no_script = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE
        )
        no_tags = re.sub(r"<[^>]+>", " ", no_script)
        text = no_tags

    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def fetch_url(url: str, timeout: float = 5.0) -> str:
    """Fetch raw HTML content from URL with strict timeout."""
    import httpx

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            return resp.text
        logger.warning(f"HTTP {resp.status_code} fetching {url}")
    except Exception as e:
        logger.warning(f"Failed fetching {url}: {e}")
    return ""


def fetch_and_chunk(
    urls: list[str],
    size: int = 512,
    overlap: int = 64,
    timeout: float = 5.0,
    max_chars: int = 15000,
) -> list[Chunk]:
    """Fetch web pages, extract clean text, and chunk for retrieval synthesis."""
    all_chunks: list[Chunk] = []
    for url in urls:
        if not url or not url.startswith(("http://", "https://")):
            continue
        raw_html = fetch_url(url, timeout=timeout)
        if not raw_html:
            continue
        text = clean_html(raw_html)
        if not text:
            continue
        bounded_text = text[:max_chars]
        url_chunks = chunk(bounded_text, size=size, overlap=overlap, doc_id=url)
        all_chunks.extend(url_chunks)
    return all_chunks
