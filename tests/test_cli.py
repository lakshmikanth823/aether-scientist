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
