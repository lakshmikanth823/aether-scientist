from aether_scientist.retrieval.web_ingest import clean_html, fetch_and_chunk, fetch_url


def test_clean_html_strips_scripts_and_styles():
    raw_html = """
    <html>
      <head>
        <title>Test Page</title>
        <style>body { color: red; }</style>
        <script>console.log("secret tracker");</script>
      </head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <header><h1>Site Header</h1></header>
        <main>
          <p>Quantum computing utilizes qubits for superposition.</p>
          <p>Superposition allows parallel state evaluation.</p>
        </main>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    cleaned = clean_html(raw_html)
    assert "Quantum computing utilizes qubits" in cleaned
    assert "Superposition allows parallel state evaluation." in cleaned
    assert "color: red" not in cleaned
    assert "console.log" not in cleaned
    assert "Site Header" not in cleaned
    assert "Copyright 2026" not in cleaned


def test_clean_html_empty_and_whitespace():
    assert clean_html("") == ""
    assert clean_html("   \n\t  ") == ""
    assert clean_html("<p>Hello    world</p>") == "Hello world"


def test_fetch_url_mock(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = "<p>Real content</p>"

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse())
    content = fetch_url("https://example.com/page")
    assert content == "<p>Real content</p>"


def test_fetch_url_failure_returns_empty(monkeypatch):
    def fake_get(*args, **kwargs):
        raise ConnectionError("Host unreachable")

    monkeypatch.setattr("httpx.get", fake_get)
    assert fetch_url("https://invalid.domain.xyz") == ""


def test_fetch_and_chunk_produces_chunks(monkeypatch):
    html_page = """
    <html>
      <body>
        <article>
          <h1>CRISPR Mechanics</h1>
          <p>CRISPR-Cas9 is a targeted genome editing technique discovered in bacterial immune systems.</p>
          <p>The guide RNA directs Cas9 to cleave matching double-stranded DNA sequences.</p>
        </article>
      </body>
    </html>
    """
    monkeypatch.setattr("aether_scientist.retrieval.web_ingest.fetch_url", lambda url, **kwargs: html_page)

    urls = ["https://science.org/crispr-mechanics"]
    chunks = fetch_and_chunk(urls, size=80, overlap=20)
    assert len(chunks) >= 1
    assert all(c.doc_id == "https://science.org/crispr-mechanics" for c in chunks)
    assert any("genome editing technique" in c.text for c in chunks)


def test_fetch_and_chunk_invalid_urls():
    assert fetch_and_chunk([]) == []
    assert fetch_and_chunk(["not-a-url", "ftp://invalid"]) == []
