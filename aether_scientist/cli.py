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
    profile: str = typer.Option(None, "--profile", "-p", help="Model profile"),
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
        prof = profile or model
        if stream:
            for tok in InferenceEngine.stream(query=query, profile=prof, max_new_tokens=max_tokens):
                typer.echo(tok, nl=False)
            typer.echo()
            return

        if domain != "auto" and domain not in available_domains():
            raise ValueError(f"Unknown domain '{domain}'. Available: {available_domains()}")

        domains_list = [domain] if domain != "auto" else ["physics", "chemistry", "biology"]
        cfg = AetherConfig(profile=prof, max_new_tokens=max_tokens, domains=domains_list)
        result = AetherScientist(cfg).analyze(query, use_rag=use_rag)

        res = result.get("analysis_result", {})
        if use_rag and not as_json and res.get("sources"):
            typer.echo(
                f"\nAnswer:\n{res.get('generated_text', '')}\n\n"
                f"Confidence: {res.get('confidence', 0.0)}\n\nSources:"
            )
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
        scientist = AetherScientist(AetherConfig())
        res = scientist.synthesize_papers(papers, domain=domain if domain != "auto" else None)
        if focus:
            res["focus"] = focus
        _output(res, as_json)
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
        sc = AetherScientist(AetherConfig())
        stats = sc.rag_engine.index(paths)
        sc.rag_engine.store.save(Path(sc.config.rag.cache_dir) / "default_store")
        _output({"status": "indexed", "docs": stats.docs, "chunks": stats.chunks}, as_json)
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
        hits = AetherScientist(AetherConfig()).rag_engine.retrieve(query, k=k)
        results = [
            {"score": round(h.score, 4), "title": h.title, "snippet": h.chunk.text[:150]}
            for h in hits
        ]
        _output({"query": query, "hits": results}, as_json)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def bench(
    n: int = typer.Option(20, "--n", help="Questions count"),
    profile: list[str] = typer.Option(  # noqa: B008
        None, "--profile", "-p", help="Profile(s) to benchmark"
    ),
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run scientific benchmark evaluation on bundled quiz."""
    from aether_scientist.benchmarks.eval import run_eval

    try:
        report = run_eval(n=n, profiles=profile)
        out = report.to_dict() if hasattr(report, "to_dict") else report
        if not as_json and isinstance(out, dict) and "summary" in out:
            hdr = f"{'Profile':<9} {'Model':<35} {'Accuracy':<10} {'Latency(ms)':<12} {'Device':<6}"
            typer.echo(f"{hdr}\n{'-' * len(hdr)}")
            for r in out["summary"]:
                typer.echo(
                    f"{r['profile']:<10} {r['model']:<35} {r['accuracy']:<10.2%} "
                    f"{r['avg_latency_ms']:<12.2f} {r['device']:<6}"
                )
        else:
            _output(out, as_json)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def profiles(as_json: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List available model profiles."""
    from aether_scientist.core.profiles import list_profiles

    items = list_profiles()
    if as_json:
        typer.echo(json.dumps(items, indent=2))
    else:
        hdr = f"{'Profile':<10} {'Model':<35} {'Tokens':<7} {'Temp':<5} {'Chat':<5} Description"
        typer.echo(f"{hdr}\n{'-' * len(hdr)}")
        for p in items:
            typer.echo(
                f"{p['name']:<10} {p['model']:<35} {p['max_new_tokens']:<7} "
                f"{p['temperature']:<5} {str(p['chat']):<5} {p['description']}"
            )


@app.command()
def domains(as_json: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """List available domain adapters."""
    from aether_scientist.domains import available_domains

    _output({"domains": available_domains()}, as_json)


@app.command()
def info(as_json: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show configuration, model, device, and version info."""
    from aether_scientist import __version__
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.inference import InferenceEngine

    config = AetherConfig()
    data = {
        "version": __version__,
        "profile": config.profile,
        "model": config.model_name,
        "device": InferenceEngine.device,
        "model_loaded": InferenceEngine.is_loaded(),
        "domains": config.domains,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "rag_backend": config.rag.backend,
    }
    _output(data, as_json)


if __name__ == "__main__":
    app()
