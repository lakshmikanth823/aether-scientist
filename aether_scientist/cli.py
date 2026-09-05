import json
from pathlib import Path

import typer

app = typer.Typer(
    name="aether", help="AetherScientist — Scientific Research LLM", add_completion=False
)


def _output(data: dict, as_json: bool = False) -> None:
    out = (
        json.dumps(data, indent=2)
        if as_json
        else "\n".join(f"  {k}: {v}" for k, v in data.items())
    )
    typer.echo(out)


def _err(e: Exception, code: int = 1) -> None:
    typer.echo(f"Error: {e}", err=True)
    raise typer.Exit(code=code) from e


def _scientist(cfg=None):
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.model import AetherScientist
    return AetherScientist(cfg or AetherConfig())

@app.command()
def ask(
    query: str = typer.Argument(..., help="Scientific question to ask"),
    domain: str = typer.Option("auto", "-d", "--domain"),
    model: str = typer.Option("distilgpt2", "-m", "--model"),
    profile: str = typer.Option(None, "-p", "--profile"),
    as_json: bool = typer.Option(False, "--json"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    stream: bool = typer.Option(False, "--stream"),
    use_rag: bool = typer.Option(False, "--rag"),
) -> None:
    """Ask a scientific question and get an AI-generated answer."""
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.inference import InferenceEngine
    from aether_scientist.domains import available_domains

    try:
        prof = profile or model
        if stream:
            for tok in InferenceEngine.stream(query=query, profile=prof, max_new_tokens=max_tokens):
                typer.echo(tok, nl=False)
            return typer.echo()

        if domain != "auto" and domain not in available_domains():
            raise ValueError(f"Unknown domain '{domain}'. Available: {available_domains()}")
        cfg = AetherConfig(
            profile=prof, max_new_tokens=max_tokens,
            domains=[domain] if domain != "auto" else ["physics", "chemistry", "biology"],
        )
        result = _scientist(cfg).analyze(query, use_rag=use_rag)

        res = result.get("analysis_result", {})
        if use_rag and not as_json and res.get("sources"):
            ans, conf = res.get("generated_text", ""), res.get("confidence", 0.0)
            typer.echo(f"\nAnswer:\n{ans}\n\nConfidence: {conf}\n\nSources:")
            for i, s in enumerate(res["sources"]):
                typer.echo(f"  [{i + 1}] {s.get('title')} ({s.get('score')}): {s.get('snippet')}")
        else:
            _output(result, as_json)
    except (ValueError, KeyError) as e:
        _err(e, 2)
    except Exception as e:
        _err(e, 1)

@app.command()
def analyze(
    papers: list[str] = typer.Option(..., "--papers", "-p"),  # noqa: B008
    focus: str = typer.Option("", "-f", "--focus"), domain: str = typer.Option("auto", "-d"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Analyze scientific papers."""
    try:
        res = _scientist().synthesize_papers(papers, domain=domain if domain != "auto" else None)
        if focus:
            res["focus"] = focus
        _output(res, as_json)
    except Exception as e:
        _err(e, 1)

@app.command()
def ingest(
    paths: list[str] = typer.Argument(...), as_json: bool = typer.Option(False, "--json")  # noqa: B008
) -> None:
    """Ingest and index scientific documents into knowledge store."""
    try:
        sc = _scientist()
        stats = sc.rag_engine.index(paths)
        sc.rag_engine.store.save(Path(sc.config.rag.cache_dir) / "default_store")
        _output({"status": "indexed", "docs": stats.docs, "chunks": stats.chunks}, as_json)
    except Exception as e:
        _err(e, 1)

@app.command()
def search(
    query: str = typer.Argument(...), k: int = typer.Option(4, "-k"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Search indexed knowledge base."""
    try:
        hits = _scientist().rag_engine.retrieve(query, k=k)
        res = [
            {"score": round(h.score, 4), "title": h.title, "snippet": h.chunk.text[:140]}
            for h in hits
        ]
        _output({"query": query, "hits": res}, as_json)
    except Exception as e:
        _err(e, 1)

@app.command()
def bench(
    n: int = typer.Option(20, "--n"), as_json: bool = typer.Option(False, "--json"),
    profile: list[str] = typer.Option(None, "-p", "--profile"),  # noqa: B008
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
        _err(e, 1)

@app.command()
def research(
    question: str = typer.Argument(...), profile: str = typer.Option(None, "-p", "--profile"),
    k: int = typer.Option(3, "-k"), as_json: bool = typer.Option(False, "--json"),
    out: Path = typer.Option(None, "-o", "--out"),  # noqa: B008
    llm_planner: bool = typer.Option(False, "--llm-planner"),
) -> None:
    """Run autonomous multi-step research inquiry and generate cited report."""
    from aether_scientist.agent.pipeline import ResearchAgent
    from aether_scientist.agent.report import write_report
    from aether_scientist.core.config import AetherConfig

    try:
        agent = ResearchAgent(AetherConfig(profile=profile) if profile else AetherConfig())
        result = None
        for ev in agent.stream(question, k=k, use_llm_planner=llm_planner, profile=profile):
            if ev.get("event") == "step":
                typer.echo(f"[*] {ev.get('description', '')}", err=True)
            elif ev.get("event") == "result":
                result = ev.get("result_object")
        if not result:
            raise RuntimeError("Research agent produced no output.")
        if out:
            write_report(result.report_md, out)
            typer.echo(f"Report written to {out}", err=True)
        if as_json:
            typer.echo(json.dumps(result.to_dict(), indent=2))
        elif not out:
            typer.echo(result.report_md)
    except IndexError as e:
        _err(e, 2)
    except Exception as e:
        _err(e, 1)

@app.command()
def profiles(as_json: bool = typer.Option(False, "--json")) -> None:
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
def domains(as_json: bool = typer.Option(False, "--json")) -> None:
    """List available domain adapters."""
    from aether_scientist.domains import available_domains
    _output({"domains": available_domains()}, as_json)

@app.command()
def info(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show configuration, model, device, and version info."""
    from aether_scientist import __version__
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.inference import InferenceEngine

    c = AetherConfig()
    _output({
        "version": __version__, "profile": c.profile, "model": c.model_name,
        "device": InferenceEngine.device, "model_loaded": InferenceEngine.is_loaded(),
        "domains": c.domains, "max_new_tokens": c.max_new_tokens,
        "temperature": c.temperature, "rag_backend": c.rag.backend,
    }, as_json)


if __name__ == "__main__":
    app()
