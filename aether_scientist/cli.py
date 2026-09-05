import json

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
) -> None:
    """Ask a scientific question and get an AI-generated answer."""
    from aether_scientist.core.config import AetherConfig
    from aether_scientist.core.model import AetherScientist
    from aether_scientist.domains import available_domains

    try:
        if domain != "auto" and domain not in available_domains():
            raise ValueError(f"Unknown domain '{domain}'. Available: {available_domains()}")
        domains_list = [domain] if domain != "auto" else ["physics", "chemistry", "biology"]
        config = AetherConfig(
            model_name=model,
            max_new_tokens=max_tokens,
            domains=domains_list,
        )
        scientist = AetherScientist(config)
        result = scientist.analyze(query)
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
def domains(
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List available domain adapters."""
    from aether_scientist.domains import available_domains

    data = {"domains": available_domains()}
    _output(data, as_json)


@app.command()
def info(
    as_json: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
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
        "device": config.device,
    }
    _output(data, as_json)


if __name__ == "__main__":
    app()
