import json
from pathlib import Path

from typer.testing import CliRunner

from aether_scientist.cli import app

runner = CliRunner()


def test_cli_domains():
    result = runner.invoke(app, ["domains"])
    assert result.exit_code == 0
    assert "physics" in result.output
    assert "chemistry" in result.output
    assert "biology" in result.output


def test_cli_domains_json():
    result = runner.invoke(app, ["domains", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "domains" in data
    assert "physics" in data["domains"]


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "version" in result.output
    assert "model" in result.output


def test_cli_info_json():
    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "version" in data
    assert "model" in data
    assert "domains" in data


def test_cli_ask(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.model.InferenceEngine.generate",
        lambda **kwargs: {
            "text": "Entropy is the measure of molecular disorder.",
            "model": "distilgpt2",
            "tokens_generated": 7,
            "elapsed_seconds": 0.01,
        },
    )
    result = runner.invoke(app, ["ask", "What is entropy?", "--domain", "physics"])
    assert result.exit_code == 0
    assert "analysis_result" in result.output


def test_cli_ask_json(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.model.InferenceEngine.generate",
        lambda **kwargs: {
            "text": "Entropy is the measure of molecular disorder.",
            "model": "distilgpt2",
            "tokens_generated": 7,
            "elapsed_seconds": 0.01,
        },
    )
    result = runner.invoke(app, ["ask", "What is entropy?", "--domain", "physics", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["domain"] == "physics"
    assert "analysis_result" in data


def test_cli_ask_bad_domain(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.model.InferenceEngine.generate",
        lambda **kwargs: {"text": "dummy"},
    )
    result = runner.invoke(app, ["ask", "Test query", "--domain", "nonexistent"])
    assert result.exit_code == 2


def test_cli_analyze():
    result = runner.invoke(
        app, ["analyze", "-p", "paper1.pdf", "-p", "paper2.pdf", "--focus", "mechanisms"]
    )
    assert result.exit_code == 0
    assert "summary" in result.output
    assert "mechanisms" in result.output


def test_docker_files_exist():
    root = Path(__file__).parent.parent
    dockerfile = root / "Dockerfile"
    docker_compose = root / "docker-compose.yml"
    dockerignore = root / ".dockerignore"

    assert dockerfile.exists()
    assert "FROM python:3.11-slim" in dockerfile.read_text()
    assert docker_compose.exists()
    assert "services:" in docker_compose.read_text()
    assert dockerignore.exists()


def test_cli_ingest_and_search(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text(
        "Quantum entanglement occurs when paired particles share quantum states.",
        encoding="utf-8",
    )

    res_ingest = runner.invoke(app, ["ingest", str(doc), "--json"])
    assert res_ingest.exit_code == 0
    data = json.loads(res_ingest.output)
    assert data["status"] == "indexed"
    assert data["docs"] == 1

    res_search = runner.invoke(app, ["search", "entanglement", "-k", "1", "--json"])
    assert res_search.exit_code == 0
    search_data = json.loads(res_search.output)
    assert len(search_data["hits"]) >= 1


def test_cli_ask_stream(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.stream",
        lambda *args, **kwargs: iter(["Entanglement ", "is ", "real."]),
    )
    result = runner.invoke(app, ["ask", "What is entanglement?", "--stream"])
    assert result.exit_code == 0
    assert "Entanglement" in result.output


def test_cli_ask_rag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "Grounded answer statement [1]."},
    )
    result = runner.invoke(app, ["ask", "What is entanglement?", "--rag", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "analysis_result" in data


def test_cli_ask_rag_offline_warning(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "Grounded answer statement [1]."},
    )
    result = runner.invoke(app, ["ask", "What is entanglement?", "--rag"])
    assert result.exit_code == 0
    assert "Using 'offline' profile" in result.output



def test_cli_bench(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.core.inference.InferenceEngine.generate",
        lambda **kwargs: {"text": "Farad"},
    )
    result = runner.invoke(app, ["bench", "--n", "2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["total"] == 2
    assert "accuracy" in data


def test_cli_research_empty_index(monkeypatch):
    monkeypatch.setattr(
        "aether_scientist.retrieval.store.VectorStore.__len__",
        lambda self: 0,
    )
    result = runner.invoke(app, ["research", "What is entropy?"])
    assert result.exit_code == 2
    assert "No documents indexed" in result.output


def test_cli_research_success(tmp_path, monkeypatch):
    from aether_scientist.agent.pipeline import ResearchAgent, ResearchResult
    from aether_scientist.agent.state import ResearchState

    state = ResearchState(question="Explain entropy")
    res = ResearchResult(
        question="Explain entropy",
        report_md="# Research Report: Explain entropy\n\n## References\n[1] Source 1",
        state=state,
        sources=[],
        confidence=0.9,
        coverage={},
    )
    events = [
        {"event": "step", "kind": "plan", "description": "Decomposing"},
        {"event": "result", "result_object": res, "data": res.to_dict()},
    ]
    monkeypatch.setattr(ResearchAgent, "stream", lambda *args, **kwargs: iter(events))
    out_file = tmp_path / "research_report.md"

    result = runner.invoke(
        app,
        ["research", "Explain entropy", "--out", str(out_file), "--profile", "fast"],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    assert "# Research Report" in out_file.read_text(encoding="utf-8")

