from aether_scientist.retrieval.web_search import WebSearchEngine


def test_web_search_ddg_mock(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = """
        <html>
          <body>
            <div class="result">
              <h2 class="result__title">
                <a href="https://example.com/crispr">CRISPR Gene Editing Overview</a>
              </h2>
              <a class="result__snippet">CRISPR allows geneticists to edit genomes.</a>
            </div>
            <div class="result">
              <h2 class="result__title">
                <a href="https://example.com/cas9">Cas9 Nuclease Mechanisms</a>
              </h2>
              <a class="result__snippet">Cas9 functions as scissors guided by RNA.</a>
            </div>
          </body>
        </html>
        """

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: FakeResponse())

    engine = WebSearchEngine()
    hits = engine.search("CRISPR Cas9", max_results=2)
    assert len(hits) == 2
    assert hits[0].title == "CRISPR Gene Editing Overview"
    assert hits[0].url == "https://example.com/crispr"
    assert "allows geneticists" in hits[0].snippet
    assert hits[1].url == "https://example.com/cas9"


def test_web_search_wiki_fallback_mock(monkeypatch):
    class FakeDDGFailure:
        status_code = 500
        text = ""

    class FakeWikiResponse:
        status_code = 200

        def json(self):
            return [
                "CRISPR",
                ["CRISPR", "CRISPR gene editing"],
                ["A family of DNA sequences.", "A genetic engineering technique."],
                [
                    "https://en.wikipedia.org/wiki/CRISPR",
                    "https://en.wikipedia.org/wiki/CRISPR_gene_editing",
                ],
            ]

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: FakeDDGFailure())
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeWikiResponse())

    engine = WebSearchEngine()
    hits = engine.search("CRISPR", max_results=2)
    assert len(hits) == 2
    assert hits[0].title == "CRISPR"
    assert hits[0].url == "https://en.wikipedia.org/wiki/CRISPR"
    assert "family of DNA" in hits[0].snippet


def test_web_search_network_failure_graceful(monkeypatch):
    def fake_post(*args, **kwargs):
        raise RuntimeError("Connection timed out")

    def fake_get(*args, **kwargs):
        raise RuntimeError("DNS resolution error")

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("httpx.get", fake_get)

    engine = WebSearchEngine()
    hits = engine.search("anything")
    assert hits == []


def test_web_search_empty_query():
    engine = WebSearchEngine()
    assert engine.search("") == []
    assert engine.search("   ") == []
