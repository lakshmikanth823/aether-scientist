import json
from pathlib import Path

import typer

app = typer.Typer(
    name="aether",
    help="AetherScientist — Scientific Research LLM",
    add_completion=False,
)


def _output(data: dict, as_json: bool = False) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2))
    else:
        for key, value in data.items():
            typer.echo(f"  {key}: {value}")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Scientific question to ask"),
    domain: str = typer.Option("auto", "--domain", "-d", help="Domain"),
    model: str = typer.Option("distilgpt2", "--model", "-m", help="Model name"),
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens"),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens live"),
    use_rag: bool = typer.Option(False, "--rag", help="Use RAG grounding"),
) -> None:
    """Ask a scientific question and get an AI-generated answer."""
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.inference import InferenceEngine
    from aether_scientist.core.model import AetherScientist
    from aether_scientist.domains import available_domains

    try:
        if stream:
            for token in InferenceEngine.stream(f"Question: {query}\nAnswer:", model_name=model):
                typer.echo(token, nl=False)
            typer.echo()
            return

        if domain != "auto" and domain not in available_domains():
            raise ValueError(f"Unknown domain '{domain}'. Available: {available_domains()}")

        domains_list = [domain] if domain != "auto" else ["physics", "chemistry", "biology"]
        config = AetherConfig(model_name=model, max_new_tokens=max_tokens, domains=domains_list)
        scientist = AetherScientist(config)
        result = scientist.analyze(query, use_rag=use_rag)

        res = result.get("analysis_result", {})
        if use_rag and not as_json and res.get("sources"):
            typer.echo(f"\nAnswer:\n{res.get('generated_text', '')}")
            typer.echo(f"\nConfidence: {res.get('confidence', 0.0)}")
            typer.echo("\nSources:")
            for i, s in enumerate(res["sources"]):
                typer.echo(f"  [{i + 1}] {s.get('title')} ({s.get('score')}): {s.get('snippet')}")
        else:
            _output(result, as_json)
    except (ValueError, KeyError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2) from e
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def analyze(
    papers: list[str] = typer.Option(..., "--papers", "-p", help="Paper paths"),  # noqa: B008
    focus: str = typer.Option("", "--focus", "-f", help="Analysis focus"),
    domain: str = typer.Option("auto", "--domain", "-d", help="Domain"),
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Analyze scientific papers."""
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.model import AetherScientist

    try:
        config = AetherConfig()
        scientist = AetherScientist(config)
        result = scientist.synthesize_papers(papers, domain=domain if domain != "auto" else None)
        if focus:
            result["focus"] = focus
        _output(result, as_json)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="Documents to index"),  # noqa: B008
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Ingest and index scientific documents into knowledge store."""
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.model import AetherScientist

    try:
        config = AetherConfig()
        scientist = AetherScientist(config)
        stats = scientist.rag_engine.index(paths)
        store_path = Path(scientist.config.rag.cache_dir) / "default_store"
        scientist.rag_engine.store.save(store_path)
        data = {
            "status": "indexed",
            "docs": stats.docs,
            "chunks": stats.chunks,
            "elapsed_seconds": stats.elapsed_seconds,
        }
        _output(data, as_json)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(4, "-k", help="Top-k results"),
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search indexed knowledge base."""
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.model import AetherScientist

    try:
        config = AetherConfig()
        scientist = AetherScientist(config)
        hits = scientist.rag_engine.retrieve(query, k=k)
        results = [
            {
                "score": round(h.score, 4),
                "title": h.title,
                "source": h.source,
                "snippet": h.chunk.text[:150],
            }
            for h in hits
        ]
        _output({"query": query, "hits": results}, as_json)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def bench(
    n: int = typer.Option(20, "--n", help="Number of questions"),
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run scientific benchmark evaluation on bundled quiz."""
    from aether_scientist.benchmarks import run_eval

    try:
        report = run_eval(n=n)
        _output(report.to_dict(), as_json)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def domains(as_json: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List available domain adapters."""
    from aether_scientist.domains import available_domains

    _output({"domains": available_domains()}, as_json)


@app.command()
def info(as_json: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show configuration, model, and version info."""
    from aether_scientist import __version__
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.inference import InferenceEngine

    config = AetherConfig()
    data = {
        "version": __version__,
        "model": config.model_name,
        "model_loaded": InferenceEngine.is_loaded(),
        "domains": config.domains,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "rag_backend": config.rag.backend,
        "device": config.device,
    }
    _output(data, as_json)


if __name__ == "__main__":
    app()
