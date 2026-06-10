"""Command-line interface for Kaggle Agent."""

import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file",
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path]) -> None:
    """Kaggle Agent - Self-evolving competition agent.

    Automatically compete in Kaggle competitions and learn from experience.
    """
    ctx.ensure_object(dict)

    # Load configuration
    try:
        cfg = Config.load(config)
        ctx.obj["config"] = cfg
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("competition")
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    help="Resume an existing competition run",
)
@click.pass_context
def run(ctx: click.Context, competition: str, resume: bool) -> None:
    """Start or resume a competition run.

    COMPETITION is the Kaggle competition slug (e.g., 'titanic').
    """
    cfg: Config = ctx.obj["config"]
    click.echo(f"Starting competition: {competition}")
    click.echo(f"Config: {cfg}")

    if resume:
        click.echo("Resuming from previous state...")
    else:
        click.echo("Starting fresh...")

    # TODO: Initialize orchestrator and start competition
    click.echo("Orchestrator not yet implemented. Coming soon!")


@main.command()
@click.argument("competition")
@click.argument("guidance", nargs=-1, required=True)
@click.pass_context
def guide(ctx: click.Context, competition: str, guidance: tuple) -> None:
    """Inject guidance for a running competition.

    Example: kagent guide titanic "Try target encoding for high cardinality features"
    """
    cfg: Config = ctx.obj["config"]
    guidance_text = " ".join(guidance)

    click.echo(f"Adding guidance for {competition}:")
    click.echo(f"  {guidance_text}")

    # TODO: Add guidance to competition's guidance queue
    click.echo("Guidance system not yet implemented. Coming soon!")


@main.command()
@click.argument("competition")
@click.pass_context
def status(ctx: click.Context, competition: str) -> None:
    """Show status of a competition run."""
    cfg: Config = ctx.obj["config"]

    click.echo(f"Status for {competition}:")
    click.echo("  Current phase: Not implemented")
    click.echo("  Experiments: 0")
    click.echo("  Best CV: N/A")
    click.echo("  Submissions: 0")


@main.command()
@click.argument("competition")
@click.pass_context
def stop(ctx: click.Context, competition: str) -> None:
    """Stop a running competition (graceful exit)."""
    click.echo(f"Stopping competition: {competition}")
    click.echo("Stop signal sent. Agent will exit after current experiment.")


@main.command(name="list")
@click.pass_context
def list_competitions(ctx: click.Context) -> None:
    """List all competition runs."""
    cfg: Config = ctx.obj["config"]
    competitions_path = cfg.resolve_path("competitions")

    if not competitions_path.exists():
        click.echo("No competitions found.")
        return

    click.echo("Competition runs:")
    for comp_dir in sorted(competitions_path.iterdir()):
        if comp_dir.is_dir():
            click.echo(f"  - {comp_dir.name}")


@main.command()
@click.argument("competition")
@click.pass_context
def retro(ctx: click.Context, competition: str) -> None:
    """Show retrospective for a completed competition."""
    cfg: Config = ctx.obj["config"]
    comp_path = cfg.resolve_path("competitions") / competition
    retro_path = comp_path / "retrospective.md"

    if not retro_path.exists():
        click.echo(f"No retrospective found for {competition}")
        return

    with open(retro_path, "r") as f:
        click.echo(f.read())


if __name__ == "__main__":
    main()
