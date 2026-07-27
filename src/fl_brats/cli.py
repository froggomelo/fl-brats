"""Infrastructure-only Typer command-line interface."""

import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from fl_brats import __version__
from fl_brats.config.loader import load_config, resolved_dict
from fl_brats.config.schema import ProjectConfig
from fl_brats.config.validation import format_validation_error, validate_config
from fl_brats.tracking.local import LocalTracker
from fl_brats.tracking.mlflow import MLflowTracker
from fl_brats.utils.environment import collect_environment, write_environment_report
from fl_brats.utils.logging import configure_logging
from fl_brats.utils.paths import (
    create_run_directory,
    create_standard_directories,
    ensure_writable_directory,
)

app = typer.Typer(
    help="FL-BraTS infrastructure commands; no training or federated runtime.", no_args_is_help=True
)
Overrides = Annotated[
    list[str] | None, typer.Argument(help="Hydra overrides, e.g. environment=debug")
]


def _config(overrides: list[str] | None) -> DictConfig:
    return load_config(overrides)


def _validated(overrides: list[str] | None) -> ProjectConfig:
    try:
        return validate_config(resolved_dict(_config(overrides)))
    except ValidationError as error:
        typer.echo(f"Configuration is invalid:\n{format_validation_error(error)}", err=True)
        raise typer.Exit(1) from error


@app.command()
def version() -> None:
    """Print only the package version."""
    typer.echo(__version__)


@app.command()
def info(overrides: Overrides = None) -> None:
    """Print safe platform, dependency, Git, and execution-mode information."""
    report = collect_environment()
    values = resolved_dict(_config(overrides))
    payload = asdict(report)
    payload.update(
        {
            "project_name": values["project_name"],
            "package_version": __version__,
            "execution_mode": values["environment"]["mode"],
        }
    )
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@app.command("show-config")
def show_config(overrides: Overrides = None) -> None:
    """Print fully resolved Hydra configuration without creating runtime state."""
    typer.echo(OmegaConf.to_yaml(_config(overrides), resolve=True))


@app.command("validate-config")
def validate_config_command(overrides: Overrides = None) -> None:
    """Resolve and validate the infrastructure configuration."""
    _validated(overrides)
    typer.echo("Configuration is valid.")


@app.command("create-dirs")
def create_dirs(overrides: Overrides = None) -> None:
    """Create configured user-writable data, output, log, cache, and checkpoint roots."""
    created = create_standard_directories(_validated(overrides))
    for name, path in created.items():
        typer.echo(f"{name}: {path}")


def _environment_messages(config: ProjectConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if sys.version_info[:2] != (3, 11):
        errors.append("Python 3.11 is required")
    for package in ("torch", "monai", "flwr", "hydra", "mlflow"):
        try:
            __import__(package)
        except ImportError:
            errors.append(f"Required import unavailable: {package}")
    for name in ("output_root", "log_root", "cache_root", "checkpoint_root"):
        try:
            ensure_writable_directory(getattr(config, name))
        except (OSError, PermissionError) as error:
            errors.append(f"{name}: {error}")
    if not shutil.which("uv"):
        warnings.append("uv is not on PATH")
    if not shutil.which("git"):
        warnings.append("git is not on PATH")
    if not shutil.which("apptainer"):
        warnings.append("Apptainer is not on PATH")
    if not shutil.which("sbatch"):
        warnings.append("SLURM sbatch is not on PATH")
    environment = config.environment
    if environment.mode == "apptainer" and (
        environment.image_path is None or not Path(environment.image_path).expanduser().exists()
    ):
        errors.append(f"Apptainer image is missing: {environment.image_path}")
    if environment.use_gpu is True and not collect_environment().cuda_available:
        warnings.append("GPU requested but CUDA is unavailable")
    return errors, warnings


@app.command("check-environment")
def check_environment(overrides: Overrides = None) -> None:
    """Check local infrastructure prerequisites; warnings do not fail the command."""
    config = _validated(overrides)
    errors, warnings = _environment_messages(config)
    for message in warnings:
        typer.echo(f"WARNING: {message}")
    for message in errors:
        typer.echo(f"ERROR: {message}", err=True)
    if errors:
        raise typer.Exit(1)
    typer.echo("Environment check passed.")


@app.command("write-environment-report")
def write_report(
    path: Annotated[Path, typer.Option("--path", "-p")],
    format: Annotated[str, typer.Option()] = "json",
) -> None:
    """Write a non-sensitive environment report in JSON, YAML, or plain text."""
    if format not in {"json", "yaml", "text"}:
        raise typer.BadParameter("format must be json, yaml, or text")
    report_format = cast(Literal["json", "yaml", "text"], format)
    typer.echo(write_environment_report(collect_environment(), path, report_format).as_posix())


@app.command("init-run")
def init_run(overrides: Overrides = None) -> None:
    """Initialize directories, metadata tracking, config, and report without training."""
    config = _validated(overrides)
    create_standard_directories(config)
    run_dir = create_run_directory(config.output_root, config.run_name)
    resolved = resolved_dict(_config(overrides))
    config_path = run_dir / "resolved-config.yaml"
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    report_path = write_environment_report(
        collect_environment(), run_dir / "environment.json", "json"
    )
    logger = configure_logging(config.logging)
    tracker = (
        LocalTracker(config.tracking.root_dir or config.output_root / "tracking")
        if config.tracking.backend == "local"
        else MLflowTracker(
            config.tracking.tracking_uri or "file:./mlruns", config.tracking.experiment_name
        )
    )
    tracker.start_run(config.run_name)
    tracker.log_parameters({"project_name": config.project_name, "seed": config.seed})
    tracker.log_artifact(config_path)
    tracker.log_artifact(report_path)
    tracker.set_tags({"scope": "infrastructure"})
    tracker.end_run()
    logger.info("Initialized infrastructure run at %s", run_dir)
    typer.echo(f"Initialized run: {run_dir}")
