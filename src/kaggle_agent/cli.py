"""Command-line interface for Kaggle Agent.

Commands:
- run: Start or resume a competition
- guide: Inject guidance for a running competition
- status: Show competition status
- stop: Request graceful stop
- list: List all competitions
- retro: Show retrospective
"""

import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config
from .interaction import CompetitionInterface
from .llm import LLMRouter
from .llm.cursor import CursorTaskPending
from .orchestrator import Orchestrator
from .tools import MockKaggleClient


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock Kaggle client (for testing)",
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path], mock: bool) -> None:
    """Kaggle Agent - Self-evolving competition agent.

    Automatically compete in Kaggle competitions and learn from experience.
    """
    ctx.ensure_object(dict)

    # Load configuration
    try:
        cfg = Config.load(config)
        ctx.obj["config"] = cfg
        ctx.obj["mock"] = mock
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _build_orchestrator(
    cfg: Config,
    competition: str,
    use_mock: bool,
    resume: bool,
) -> Orchestrator:
    """Create orchestrator with cursor-aware LLM router."""
    comp_path = cfg.resolve_path("competitions") / competition
    tasks_dir = comp_path / "agent_tasks"
    llm_router = LLMRouter.from_config(cfg, cursor_tasks_dir=tasks_dir)

    if use_mock:
        kaggle_client = MockKaggleClient()
    else:
        from .tools import KaggleClient

        kaggle_client = KaggleClient(dry_run=cfg.kaggle.dry_run)

    return Orchestrator(
        config=cfg,
        competition=competition,
        llm_router=llm_router,
        kaggle_client=kaggle_client,
        resume=resume,
    )


@main.command()
@click.argument("competition")
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    help="Resume an existing competition run",
)
@click.option(
    "--max-experiments",
    "-n",
    type=int,
    help="Maximum number of experiments to run",
)
@click.pass_context
def run(ctx: click.Context, competition: str, resume: bool, max_experiments: Optional[int]) -> None:
    """Start or resume a competition run.

    COMPETITION is the Kaggle competition slug (e.g., 'titanic').
    """
    cfg: Config = ctx.obj["config"]
    use_mock: bool = ctx.obj["mock"]

    click.echo(f"🚀 Starting competition: {competition}")

    if resume:
        click.echo("📂 Resuming from previous state...")

    # Initialize LLM router and orchestrator
    try:
        orchestrator = _build_orchestrator(cfg, competition, use_mock, resume)
    except Exception as e:
        click.echo(f"Error initializing agent: {e}", err=True)
        sys.exit(1)

    # Override max experiments if specified
    if max_experiments:
        cfg.budget.max_experiments_per_competition = max_experiments

    try:
        orchestrator.run()
    except CursorTaskPending as pending:
        click.echo("\n⏸️  Paused — waiting for Cursor agent response.")
        click.echo(f"   Task file:   {pending.task_path}")
        click.echo(f"   Write to:    {pending.response_path}")
        click.echo(f"\n   Then run: kagent continue {competition}")
        sys.exit(0)
    except KeyboardInterrupt:
        click.echo("\n⚠️ Interrupted by user. Stopping gracefully...")
        orchestrator.stop()
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        raise

    # Show final status
    status = orchestrator.get_status()
    click.echo("\n✅ Competition completed!")
    click.echo(f"   Phase: {status['phase']}")
    click.echo(f"   Experiments: {status['experiment_count']}")
    click.echo(f"   Best CV: {status['best_cv_score']}")
    click.echo(f"   LLM Cost: ${status['llm_cost']:.4f}")


@main.command(name="continue")
@click.argument("competition")
@click.pass_context
def continue_run(ctx: click.Context, competition: str) -> None:
    """Resume a competition after Cursor agent writes task response."""
    cfg: Config = ctx.obj["config"]
    use_mock: bool = ctx.obj["mock"]

    click.echo(f"▶️  Continuing competition: {competition}")

    try:
        orchestrator = _build_orchestrator(cfg, competition, use_mock, resume=True)
        orchestrator.run()
    except CursorTaskPending as pending:
        click.echo("\n⏸️  Still waiting for Cursor agent response.")
        click.echo(f"   Task file:   {pending.task_path}")
        click.echo(f"   Write to:    {pending.response_path}")
        sys.exit(0)
    except KeyboardInterrupt:
        click.echo("\n⚠️ Interrupted by user. Stopping gracefully...")
        orchestrator.stop()
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        raise

    status = orchestrator.get_status()
    click.echo("\n✅ Step completed!")
    click.echo(f"   Phase: {status['phase']}")
    click.echo(f"   Experiments: {status['experiment_count']}")
    click.echo(f"   Best CV: {status['best_cv_score']}")


@main.command()
@click.argument("competition")
@click.pass_context
def task(ctx: click.Context, competition: str) -> None:
    """Show the current pending Cursor agent task, if any."""
    cfg: Config = ctx.obj["config"]
    tasks_dir = cfg.resolve_path("competitions") / competition / "agent_tasks"

    from .llm.cursor import CursorAgentProvider

    provider = CursorAgentProvider(name="cursor", model="cursor-agent", tasks_dir=tasks_dir)
    pending = provider.get_pending_task()

    if pending is None:
        click.echo(f"No pending Cursor task for {competition}.")
        return

    task_path, response_path = pending
    click.echo(f"📋 Pending task for {competition}:")
    click.echo(f"   Read:   {task_path}")
    click.echo(f"   Write:  {response_path}")
    click.echo("\n--- Task preview ---\n")
    click.echo(task_path.read_text()[:4000])


@main.command()
@click.argument("competition")
@click.argument("guidance", nargs=-1, required=True)
@click.option(
    "--source",
    "-s",
    default="user",
    help="Source of guidance (user, system, auto)",
)
@click.pass_context
def guide(ctx: click.Context, competition: str, guidance: tuple, source: str) -> None:
    """Inject guidance for a running competition.

    Example: kagent guide titanic "Try target encoding for high cardinality features"
    """
    cfg: Config = ctx.obj["config"]
    guidance_text = " ".join(guidance)

    competitions_path = cfg.resolve_path("competitions")
    interface = CompetitionInterface(competition, competitions_path)

    guidance_id = interface.add_guidance(guidance_text, source)

    click.echo(f"💡 Guidance added for {competition}:")
    click.echo(f"   ID: {guidance_id}")
    click.echo(f"   Content: {guidance_text}")


@main.command()
@click.argument("competition")
@click.pass_context
def status(ctx: click.Context, competition: str) -> None:
    """Show status of a competition run."""
    cfg: Config = ctx.obj["config"]

    competitions_path = cfg.resolve_path("competitions")
    interface = CompetitionInterface(competition, competitions_path)

    status_info = interface.get_status()

    click.echo(f"📊 Status for {competition}:")
    click.echo(f"   Phase: {status_info['phase']}")
    click.echo(f"   Experiments: {status_info['experiment_count']}")
    click.echo(f"   Best CV Score: {status_info['best_cv_score'] or 'N/A'}")
    click.echo(f"   LLM Cost: ${status_info['llm_cost_usd']:.4f}")
    click.echo(f"   Guidance Queue: {status_info['guidance_pending']} pending")
    click.echo(f"   Guidance Adoption: {status_info['guidance_adoption_rate']}")

    if status_info['notes']:
        click.echo(f"\n📝 Recent Notes:")
        for note in status_info['notes']:
            click.echo(f"   - {note}")


@main.command()
@click.argument("competition")
@click.pass_context
def stop(ctx: click.Context, competition: str) -> None:
    """Stop a running competition (graceful exit)."""
    cfg: Config = ctx.obj["config"]

    competitions_path = cfg.resolve_path("competitions")
    interface = CompetitionInterface(competition, competitions_path)

    if interface.request_stop():
        click.echo(f"🛑 Stop signal sent for {competition}")
        click.echo("   Agent will exit after current experiment completes.")
    else:
        click.echo(f"⚠️ Failed to send stop signal for {competition}", err=True)


@main.command(name="list")
@click.pass_context
def list_competitions(ctx: click.Context) -> None:
    """List all competition runs."""
    cfg: Config = ctx.obj["config"]
    competitions_path = cfg.resolve_path("competitions")

    if not competitions_path.exists():
        click.echo("No competitions found.")
        return

    click.echo("📋 Competition runs:")
    for comp_dir in sorted(competitions_path.iterdir()):
        if comp_dir.is_dir() and not comp_dir.name.startswith("."):
            # Check for state
            state_path = comp_dir / "state.json"
            if state_path.exists():
                import json
                with open(state_path, "r") as f:
                    state = json.load(f)
                phase = state.get("phase", "UNKNOWN")
                experiments = state.get("experiment_count", 0)
                click.echo(f"   📁 {comp_dir.name} ({phase}, {experiments} experiments)")
            else:
                click.echo(f"   📁 {comp_dir.name} (no state)")


@main.command()
@click.argument("competition")
@click.pass_context
def retro(ctx: click.Context, competition: str) -> None:
    """Show retrospective for a completed competition."""
    cfg: Config = ctx.obj["config"]
    comp_path = cfg.resolve_path("competitions") / competition
    retro_path = comp_path / "retrospective.md"

    if not retro_path.exists():
        click.echo(f"📝 No retrospective found for {competition}")
        click.echo("   Run the competition to completion to generate one.")
        return

    click.echo(retro_path.read_text())


@main.command()
@click.pass_context
def config_check(ctx: click.Context) -> None:
    """Check configuration and API keys."""
    cfg: Config = ctx.obj["config"]

    click.echo("🔍 Configuration Check:\n")

    # Check LLM providers
    click.echo("LLM Providers:")
    for provider in cfg.llm.providers:
        if provider.type == "cursor":
            status = "✅ No API key needed"
        else:
            api_key = cfg.get_api_key(provider.name)
            status = "✅ Set" if api_key and api_key != "dummy-key" else "❌ Not set"
        click.echo(f"   {provider.name} ({provider.type}): {status}")

    # Check Kaggle
    click.echo("\nKaggle API:")
    import os
    has_username = bool(os.environ.get("KAGGLE_USERNAME"))
    has_key = bool(os.environ.get("KAGGLE_KEY"))
    if has_username and has_key:
        click.echo("   ✅ KAGGLE_USERNAME and KAGGLE_KEY set")
    else:
        click.echo("   ❌ Kaggle credentials not set (or using mock)")

    # Show paths
    click.echo("\nPaths:")
    for path_type in ["knowledge", "competitions"]:
        path = cfg.resolve_path(path_type)
        exists = "✅" if path.exists() else "❌ (will create)"
        click.echo(f"   {path_type}: {path} {exists}")


if __name__ == "__main__":
    main()
