"""Infrastructure-only Typer command-line interface."""

import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from fl_brats import __version__
from fl_brats.comparison.baselines import compare
from fl_brats.config.loader import load_config, resolved_dict
from fl_brats.config.schema import ProjectConfig
from fl_brats.config.validation import format_validation_error, validate_config
from fl_brats.data.brats import build_dataloader, build_dataset, records_from_index
from fl_brats.data.indexing import build_dataset_index, sha256_file
from fl_brats.data.partitioning import create_partition
from fl_brats.data.splits import create_split
from fl_brats.data.transforms import build_transforms
from fl_brats.data.validation import validate_index
from fl_brats.losses.segmentation import create_loss
from fl_brats.models.factory import create_model, model_report
from fl_brats.tracking.local import LocalTracker
from fl_brats.tracking.mlflow import MLflowTracker
from fl_brats.training.checkpointing import load_checkpoint, save_checkpoint
from fl_brats.training.engine import evaluate, train_one_epoch
from fl_brats.utils.environment import collect_environment, write_environment_report
from fl_brats.utils.logging import configure_logging
from fl_brats.utils.paths import (
    create_run_directory,
    create_standard_directories,
    ensure_writable_directory,
)
from fl_brats.utils.reproducibility import seed_everything

app = typer.Typer(
    help="FL-BraTS reproducible BraTS centralized and FedAvg baseline commands.",
    no_args_is_help=True,
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


@app.command("index-data")
def index_data(
    root: Path = typer.Option(..., "--root"), output_dir: Path = typer.Option(..., "--output-dir")
) -> None:
    """Create an immutable-source BraTS dataset index and validation summary."""
    index, summary, _ = build_dataset_index(root, output_dir)
    typer.echo(f"dataset index: {index}\nsummary: {summary}")


@app.command("validate-data")
def validate_data(
    index_path: Path = typer.Option(..., "--index-path"), allow_incomplete: bool = False
) -> None:
    """Validate index completeness and duplicate IDs without silently excluding data."""
    errors = validate_index(index_path, allow_incomplete)
    if errors:
        typer.echo("\n".join(errors), err=True)
        raise typer.Exit(1)
    typer.echo("Dataset index is valid.")


@app.command("create-split")
def create_split_command(
    index_path: Path = typer.Option(..., "--index-path"),
    manifest_path: Path = typer.Option(..., "--manifest-path"),
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> None:
    """Write a fixed deterministic subject-level train/validation/test manifest."""
    result = create_split(
        index_path, manifest_path, seed, train_fraction, validation_fraction, test_fraction
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-split")
def inspect_split(manifest_path: Path = typer.Option(..., "--manifest-path")) -> None:
    typer.echo(manifest_path.read_text(encoding="utf-8"))


@app.command("create-partition")
def create_partition_command(
    split_path: Path = typer.Option(..., "--split-path"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    fractions: str = "[0.5,0.25,0.15,0.1]",
    seed: int = 42,
) -> None:
    """Create and persist a fixed quantity-skewed training-client partition."""
    values = json.loads(fractions)
    result = create_partition(split_path, output_dir, values, seed)
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-partition")
def inspect_partition(manifest_path: Path = typer.Option(..., "--manifest-path")) -> None:
    typer.echo(manifest_path.read_text(encoding="utf-8"))


@app.command("create-initial-weights")
def create_initial_weights(
    model_config: Path = typer.Option(..., "--model-config"),
    output_dir: Path = typer.Option(Path("artifacts/initial_weights"), "--output-dir"),
    seed: int = 42,
) -> None:
    """Create one hashed U-Net initialization shared by both controlled baselines."""
    import torch

    values = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    model = create_model(values)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"unet_seed_{seed}.pt"
    torch.save(model.state_dict(), path)
    digest = sha256_file(path)
    metadata = {"seed": seed, "initial_weight_hash": digest, "model": model_report(model, values)}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    typer.echo(str(path))


@app.command("train-centralized")
def train_centralized(
    index_path: Path = typer.Option(..., "--index-path"),
    split_path: Path = typer.Option(..., "--split-path"),
    partition_path: Path = typer.Option(..., "--partition-path"),
    initial_weights: Path = typer.Option(..., "--initial-weights"),
    output_root: Path = typer.Option(Path("outputs/centralized"), "--output-root"),
    epochs: int = typer.Option(50, "--epochs", min=1),
    batch_size: int = typer.Option(1, "--batch-size", min=1),
    workers: int = typer.Option(0, "--workers", min=0),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Train on the union of frozen client subjects and evaluate fixed global sets."""
    import torch

    split = json.loads(split_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    assigned = sorted(
        subject for client in partition["clients"] for subject in client["subject_ids"]
    )
    if assigned != sorted(split["train_subjects"]):
        raise typer.BadParameter("partition training subjects do not exactly match the split")
    if partition["source_training_split_hash"] != split["split_manifest_hash"]:
        raise typer.BadParameter("partition was generated from a different split")
    model_values = yaml.safe_load(Path("configs/model/unet_baseline.yaml").read_text())
    data_values = yaml.safe_load(Path("configs/data/brats2021.yaml").read_text())
    loss_values = yaml.safe_load(Path("configs/loss/dice_ce.yaml").read_text())
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    (run_dir / "checkpoints").mkdir(parents=True)
    seed_everything(seed)
    selected = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    model = create_model(model_values).to(selected)
    model.load_state_dict(
        torch.load(initial_weights, map_location="cpu", weights_only=True), strict=True
    )
    loss_fn = create_loss(loss_values)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cache_base = Path(os.environ.get("FL_BRATS_CACHE_ROOT", "cache")) / "brats2021" / run_dir.name

    def loader(subjects: list[str], training: bool, name: str) -> tuple[Any, int]:
        records = records_from_index(index_path, subjects)
        dataset = build_dataset(
            records,
            build_transforms(data_values, training),
            data_values.get("cache_mode", "dataset"),
            cache_base / name,
        )
        return build_dataloader(dataset, batch_size, training, workers, seed), len(records)

    train_loader, train_count = loader(split["train_subjects"], True, "train")
    validation_loader, validation_count = loader(split["validation_subjects"], False, "validation")
    test_loader, test_count = loader(split["test_subjects"], False, "test")
    best, best_epoch, started = float("-inf"), 0, time.monotonic()
    metrics_file = run_dir / "metrics.jsonl"
    for epoch in range(1, epochs + 1):
        training = train_one_epoch(model, train_loader, loss_fn, optimizer, selected)
        validation = evaluate(model, validation_loader, loss_fn, selected)
        metrics_file.write_text(
            metrics_file.read_text()
            + json.dumps(
                {
                    "epoch": epoch,
                    **training,
                    **{f"validation_{k}": v for k, v in validation.items()},
                }
            )
            + "\n"
            if metrics_file.exists()
            else json.dumps(
                {
                    "epoch": epoch,
                    **training,
                    **{f"validation_{k}": v for k, v in validation.items()},
                }
            )
            + "\n"
        )
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_validation_metric": best,
            },
        )
        if validation["mean_dice"] > best:
            best, best_epoch = validation["mean_dice"], epoch
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                {"model_state": model.state_dict(), "epoch": epoch, "best_validation_metric": best},
            )
    model.load_state_dict(
        load_checkpoint(run_dir / "checkpoints" / "best.pt")["model_state"], strict=True
    )
    test = evaluate(model, test_loader, loss_fn, selected)
    summary = {
        "run_id": run_dir.name,
        "seed": seed,
        "initial_weight_hash": sha256_file(initial_weights),
        "dataset_index_hash": sha256_file(index_path),
        "split_manifest_hash": split["split_manifest_hash"],
        "partition_manifest_hash": partition["partition_manifest_hash"],
        "model_parameter_count": model_report(model, model_values)["total_parameters"],
        "training_subject_count": train_count,
        "validation_subject_count": validation_count,
        "test_subject_count": test_count,
        "best_validation_epoch": best_epoch,
        "best_validation_mean_dice": best,
        **{f"test_{k}": v for k, v in test.items()},
        "total_training_time": time.monotonic() - started,
        "model_config": model_values,
        "loss_config": loss_values,
        "preprocessing_config": data_values,
        "evaluation_policy": "empty ground truth and empty prediction score Dice 1",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(str(run_dir))


@app.command("evaluate-checkpoint")
def evaluate_checkpoint(checkpoint: Path = typer.Option(..., "--checkpoint")) -> None:
    """Inspect checkpoint metadata; full evaluation is performed by the shared engine."""
    typer.echo(
        json.dumps(
            {k: v for k, v in load_checkpoint(checkpoint).items() if k != "model_state"},
            default=str,
            indent=2,
        )
    )


@app.command("compare-baselines")
def compare_baselines(
    centralized_summary: Path = typer.Option(..., "--centralized-summary"),
    federated_summary: Path = typer.Option(..., "--federated-summary"),
    output_dir: Path = typer.Option(Path("outputs/comparison"), "--output-dir"),
    allow_mismatch: bool = False,
) -> None:
    result = compare(centralized_summary, federated_summary, output_dir, strict=not allow_mismatch)
    typer.echo(json.dumps(result, indent=2))
