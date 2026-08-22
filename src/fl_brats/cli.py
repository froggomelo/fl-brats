"""Operational Typer command-line interface."""
# ruff: noqa: E501, E701, E702

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
from fl_brats.comparison.architecture_replacement import compare_sam_to_baseline
from fl_brats.comparison.baselines import compare
from fl_brats.config.loader import load_config, resolved_dict
from fl_brats.config.schema import ProjectConfig
from fl_brats.config.validation import format_validation_error, validate_config
from fl_brats.data.brats import build_dataloader, build_dataset, records_from_index
from fl_brats.data.discovery import discover_records
from fl_brats.data.hashing import sha256_file
from fl_brats.data.indexing import build_dataset_index
from fl_brats.data.manifests import verify_manifests
from fl_brats.data.modalities import MODALITIES, policy_from_config
from fl_brats.data.partitioning import create_partition
from fl_brats.data.preprocessing import preprocess_index
from fl_brats.data.splits import create_split
from fl_brats.data.transforms import build_transforms
from fl_brats.data.validation import validate_index, validate_subject
from fl_brats.federated.orchestration import build_client, run_local_federation
from fl_brats.federated.quantization import (
    BitWidthAllocator,
    LayerQuantizationStats,
    eligible_state,
)
from fl_brats.federated.server import start_server
from fl_brats.federated.sweep import (
    load_sweep_config,
    run_fedavg_full_factorial_sweep,
    run_sweep,
)
from fl_brats.losses.segmentation import create_loss
from fl_brats.models.factory import create_model, model_report
from fl_brats.sam.validation import verify_pretrained
from fl_brats.tracking.local import LocalTracker
from fl_brats.tracking.mlflow import MLflowTracker
from fl_brats.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from fl_brats.training.engine import evaluate, forward_batch, logits_from_output, train_one_epoch
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
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
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


def _override_values(context: typer.Context) -> dict[str, Any]:
    """Resolve trailing Hydra-style overrides; explicit options take precedence."""
    return resolved_dict(_config(list(context.args) or None))


def _values_from_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Resolve explicit Hydra-style override arguments."""
    return resolved_dict(_config(overrides))


def _extract_preprocess_options(tokens: list[str] | None) -> tuple[list[str] | None, dict[str, Any]]:
    """Allow Typer options accidentally captured after Hydra-style arguments."""
    if not tokens:
        return None, {}
    hydra: list[str] = []
    options: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--resume":
            options["resume"] = True
            index += 1
            continue
        if token == "--overwrite":
            options["overwrite"] = True
            index += 1
            continue
        if token == "--limit":
            if index + 1 >= len(tokens):
                raise typer.BadParameter("--limit requires an integer value")
            options["limit"] = int(tokens[index + 1])
            index += 2
            continue
        if token == "--subject-id":
            if index + 1 >= len(tokens):
                raise typer.BadParameter("--subject-id requires a value")
            options["subject_id"] = tokens[index + 1]
            index += 2
            continue
        hydra.append(token)
        index += 1
    return hydra or None, options


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


@app.command("inspect-quantization")
def inspect_quantization(overrides: Overrides = None) -> None:
    """Print eligible layers and a deterministic fixed/mixed wire-size estimate."""
    values = resolved_dict(_config(overrides))
    quantization = values["quantization"]
    model = create_model(values["model"])
    state = eligible_state(model.state_dict())
    counts = {name: tensor.numel() for name, tensor in state.items()}
    patterns = quantization["mixed"]["protected_layer_patterns"]
    protected = {name for name in counts if any(pattern in name.lower() for pattern in patterns)}
    if quantization["mode"] == "mixed":
        result = BitWidthAllocator().allocate(
            layer_statistics={
                name: LayerQuantizationStats(count, importance=1.0)
                for name, count in counts.items()
            },
            candidate_bit_widths=quantization["mixed"]["candidate_bit_widths"],
            average_budget=quantization["mixed"]["default_average_bit_width_budget"],
            protected_layers=protected,
            protected_minimum=quantization["mixed"]["protected_layer_minimum_bit_width"],
        )
        assignment, average, unused = (
            result.assignments,
            result.achieved_average_bit_width,
            result.unused_bit_budget,
        )
    else:
        width = 32 if quantization["mode"] == "none" else quantization["fixed"]["bit_width"]
        assignment = {name: width for name in counts}
        average, unused = float(width), 0.0
    payload_bits = sum(counts[name] * assignment[name] for name in counts)
    typer.echo(
        json.dumps(
            {
                "eligible_layers": counts,
                "protected_layers": sorted(protected),
                "bit_width_assignment": assignment,
                "parameter_weighted_mean_bit_width": average,
                "unused_bit_budget": unused,
                "estimated_payload_bytes": (payload_bits + 7) // 8,
                "model_fp32_reference_bytes": sum(counts.values()) * 4,
            },
            indent=2,
            sort_keys=True,
        )
    )


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
    context: typer.Context,
    root: Path | None = typer.Option(None, "--root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Create an immutable-source BraTS dataset index and validation summary."""
    data = _override_values(context)["data"]
    index, summary, _ = build_dataset_index(
        root or Path(data["raw_root"]),
        output_dir or Path(data["processed_root"]),
        data.get("allow_unexpected_labels", False),
        data.get("relative_paths_in_manifests", True),
    )
    typer.echo(f"dataset index: {index}\nsummary: {summary}")


@app.command("discover-data")
def discover_data(root: Path = typer.Option(..., "--root")) -> None:
    """Recursively report complete and incomplete BraTS-style subject directories."""
    records = discover_records(root)
    typer.echo(
        json.dumps(
            {
                "discovered_subjects": len(records),
                "complete_subjects": sum(x.complete for x in records),
                "incomplete_subjects": sum(not x.complete for x in records),
                "duplicate_subject_ids": sorted(
                    {
                        x.subject_id
                        for x in records
                        if sum(y.subject_id == x.subject_id for y in records) > 1
                    }
                ),
            },
            indent=2,
        )
    )


@app.command("validate-data")
def validate_data(
    index_path: Path | None = typer.Option(None, "--index-path"),
    allow_incomplete: bool = False,
    root: Path | None = typer.Option(None, "--root"),
    strict: bool = typer.Option(True, "--strict/--report-only"),
    report_path: Path | None = typer.Option(None, "--report-path"),
) -> None:
    """Validate an index, or validate raw NIfTI subjects directly."""
    if (index_path is None) == (root is None):
        raise typer.BadParameter("provide exactly one of --index-path or --root")
    if root is not None:
        records = discover_records(root)
        directories = {
            subject_id: [
                item.subject_directory for item in records if item.subject_id == subject_id
            ]
            for subject_id in {item.subject_id for item in records}
        }
        results = [
            validate_subject(item, duplicate_directories=directories[item.subject_id]).as_dict()
            for item in records
        ]
        errors = [str(e) for x in results for e in cast(list[object], x["errors"])]
        if report_path:
            report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        errors = validate_index(cast(Path, index_path), allow_incomplete)
    if errors:
        typer.echo("\n".join(errors), err=True)
        if strict:
            raise typer.Exit(1)
    typer.echo("Dataset index is valid.")


@app.command(
    "preprocess-data",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def preprocess_data(
    index_path: Path | None = typer.Option(None, "--index-path"),
    raw_root: Path | None = typer.Option(None, "--raw-root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    config_path: Path = typer.Option(Path("configs/preprocessing/brats2021_v1.yaml"), "--config"),
    resume: bool = False,
    overwrite: bool = False,
    subject_id: str | None = None,
    limit: int | None = None,
    validate_only: bool = False,
    overrides: Overrides = None,
) -> None:
    """Materialize reusable subjects atomically (or validate the selected index)."""
    overrides, captured_options = _extract_preprocess_options(overrides)
    resume = bool(captured_options.get("resume", resume))
    overwrite = bool(captured_options.get("overwrite", overwrite))
    subject_id = cast(str | None, captured_options.get("subject_id", subject_id))
    limit = cast(int | None, captured_options.get("limit", limit))
    values = _values_from_overrides(overrides)
    data = values["data"]
    index_path = index_path or Path(data["processed_root"]) / "dataset_index.csv"
    raw_root = raw_root or Path(data["raw_root"])
    output_dir = output_dir or Path(data["processed_root"])
    if validate_only:
        errors = validate_index(index_path)
        typer.echo("\n".join(errors) if errors else "Dataset index is valid.")
        if errors:
            raise typer.Exit(1)
        return
    config = values.get("preprocessing") or yaml.safe_load(config_path.read_text(encoding="utf-8"))
    typer.echo(
        json.dumps(
            preprocess_index(
                index_path,
                raw_root,
                output_dir,
                config,
                resume=resume,
                overwrite=overwrite,
                subject_id=subject_id,
                limit=limit,
            ),
            indent=2,
        )
    )


@app.command("inspect-data")
def inspect_data(summary_path: Path = typer.Option(..., "--summary-path")) -> None:
    """Print the generated data-summary JSON."""
    typer.echo(summary_path.read_text(encoding="utf-8"))


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


@app.command(
    "create-partition",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def create_partition_command(
    context: typer.Context,
    split_path: Path = typer.Option(..., "--split-path"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    fractions: str = "[0.5,0.25,0.15,0.1]",
    seed: int = 42,
) -> None:
    """Create and persist a fixed quantity-skewed training-client partition."""
    values = json.loads(fractions)
    missingness = _override_values(context).get("missing_modalities", {})
    result = create_partition(
        split_path,
        output_dir,
        values,
        seed,
        missingness_policy=policy_from_config(missingness),
        missingness_config=missingness,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-partition")
def inspect_partition(manifest_path: Path = typer.Option(..., "--manifest-path")) -> None:
    typer.echo(manifest_path.read_text(encoding="utf-8"))


@app.command("prepare-experiment-partition")
def prepare_experiment_partition(
    config: Path = typer.Option(..., "--config"),
    split_path: Path | None = typer.Option(None, "--split-path"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Create the client partition required by a standalone experiment YAML."""
    values = _load_experiment_yaml(config)
    data = values.get("data", {})
    destination = Path(data["partition_root"])
    source = split_path or Path(data.get("split_path", "data/splits/brats2021.json"))
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise typer.BadParameter(
            f"partition destination is not empty: {destination}; use --overwrite explicitly"
        )
    fractions = list(data.get("client_proportions", [0.5, 0.25, 0.15, 0.1]))
    missingness = values.get("missing_modalities", {})
    result = create_partition(
        source,
        destination,
        fractions,
        int(values.get("partition_seed", values.get("seed", 42))),
        missingness_policy=policy_from_config(missingness),
        missingness_config=missingness,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-modalities")
def inspect_modalities(partition_root: Path = typer.Option(..., "--partition-root")) -> None:
    """Print aggregate per-client modality and canonical-combination counts."""
    output: dict[str, Any] = {}
    for path in sorted(partition_root.glob("client_*.json")):
        client = json.loads(path.read_text(encoding="utf-8"))
        refs = client.get("subjects", [])
        modality_counts = {
            modality: sum(bool(ref.get("modality_mask", [True] * 4)[index]) for ref in refs)
            for index, modality in enumerate(MODALITIES)
        }
        combinations: dict[str, int] = {}
        for ref in refs:
            key = "+".join(ref.get("available_modalities", list(MODALITIES)))
            combinations[key] = combinations.get(key, 0) + 1
        output[str(client["client_id"])] = {
            "subjects": len(refs),
            "modality_counts": modality_counts,
            "combination_counts": combinations,
            "policy_fingerprint": client.get("missingness_policy_fingerprint"),
        }
    typer.echo(json.dumps(output, indent=2, sort_keys=True))


@app.command("verify-manifests")
def verify_manifest_command(
    partition_root: Path = typer.Option(..., "--partition-root"), check_nifti: bool = False
) -> None:
    """Check split/partition hashes, coverage, duplication, and leakage."""
    errors = verify_manifests(partition_root, check_nifti)
    if errors:
        typer.echo("\n".join(errors), err=True)
        raise typer.Exit(1)
    typer.echo("Manifests are valid.")


@app.command("create-initial-weights")
def create_initial_weights(
    model_config: Path = typer.Option(..., "--model-config"),
    output_dir: Path = typer.Option(Path("artifacts/initial_weights"), "--output-dir"),
    seed: int = 42,
) -> None:
    """Create one hashed initialization shared by controlled baselines."""
    import torch

    values = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    model = create_model(values)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(values.get("name", "model")).replace("/", "_")
    path = output_dir / f"{name}_seed_{seed}.pt"
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
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    epochs: int = typer.Option(50, "--epochs", min=1),
    batch_size: int = typer.Option(1, "--batch-size", min=1),
    workers: int = typer.Option(0, "--workers", min=0),
    pin_memory: bool | None = typer.Option(
        None,
        "--pin-memory/--no-pin-memory",
        help=(
            "Pin host memory for CUDA DataLoader transfers. Defaults to CUDA availability "
            "unless FL_BRATS_PIN_MEMORY is set."
        ),
    ),
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
    run_dir = run_dir or output_root / time.strftime("%Y%m%d_%H%M%S")
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
        return build_dataloader(
            dataset, batch_size, training, workers, seed, pin_memory
        ), len(records)

    train_loader, train_count = loader(split["train_subjects"], True, "train")
    validation_loader, validation_count = loader(split["validation_subjects"], False, "validation")
    test_loader, test_count = loader(split["test_subjects"], False, "test")
    best, best_epoch, started = float("-inf"), 0, time.monotonic()
    metrics_file = run_dir / "metrics.jsonl"
    for epoch in range(1, epochs + 1):
        training = train_one_epoch(model, train_loader, loss_fn, optimizer, selected)
        validation = evaluate(model, validation_loader, loss_fn, selected)
        record = {
            "epoch": epoch,
            **training,
            **{f"validation_{k}": v for k, v in validation.items()},
        }
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        typer.echo(
            "epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
            "val_loss={validation_loss:.4f} val_mean_dice={validation_mean_dice:.4f} "
            "val_wt={validation_wt_dice:.4f} val_tc={validation_tc_dice:.4f} "
            "val_et={validation_et_dice:.4f}".format(epochs=epochs, **record),
            err=True,
        )
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_validation_metric": best,
                "model_config": model_values,
                "model_input_channels": 8 if model_values.get("concat_mask") else 4,
            },
        )
        if validation["mean_dice"] > best:
            best, best_epoch = validation["mean_dice"], epoch
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_validation_metric": best,
                    "model_config": model_values,
                    "model_input_channels": 8 if model_values.get("concat_mask") else 4,
                },
            )
    model.load_state_dict(
        load_checkpoint(run_dir / "checkpoints" / "best.pt")["model_state"], strict=True
    )
    test = evaluate(model, test_loader, loss_fn, selected)
    epoch_records = [
        json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line
    ]
    final_validation_dice = float(epoch_records[-1]["validation_mean_dice"])
    convergence = {
        f"round_to_{percentage}_percent_final_dice": next(
            (
                int(record["epoch"])
                for record in epoch_records
                if float(record["validation_mean_dice"])
                >= final_validation_dice * percentage / 100
            ),
            None,
        )
        for percentage in (90, 95)
    }
    convergence["unit"] = "epoch"
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
        "best_round": best_epoch,
        "best_validation_mean_dice": best,
        **{f"test_{k}": v for k, v in test.items()},
        "metrics": {
            "segmentation": test,
            "convergence": convergence,
            "communication": {
                "not_applicable_reason": "centralized training does not transmit federated updates"
            },
            "per_client": {
                "not_applicable_reason": "centralized training has no federated clients"
            },
        },
        "total_training_time": time.monotonic() - started,
        "model_config": model_values,
        "loss_config": loss_values,
        "preprocessing_config": data_values,
        "evaluation_policy": "empty ground truth and empty prediction score Dice 1",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(
        "best_epoch={best_validation_epoch} best_val_mean_dice={best_validation_mean_dice:.4f} "
        "test_mean_dice={test_mean_dice:.4f} test_wt={test_wt_dice:.4f} "
        "test_tc={test_tc_dice:.4f} test_et={test_et_dice:.4f}".format(**summary),
        err=True,
    )
    typer.echo(str(run_dir))


@app.command("evaluate-checkpoint")
def evaluate_checkpoint(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    index_path: Path | None = typer.Option(None, "--index-path"),
    subjects: str | None = typer.Option(None, "--subjects", help="Comma-separated subject IDs"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Restore a checkpoint and, when data is supplied, compute segmentation metrics."""
    payload = load_checkpoint(checkpoint)
    if index_path is None or subjects is None:
        typer.echo(
            json.dumps(
                {k: v for k, v in payload.items() if k != "model_state"}, default=str, indent=2
            )
        )
        return
    model_values = payload.get("model_config")
    if not isinstance(model_values, dict) or "model_state" not in payload:
        raise typer.BadParameter(
            "checkpoint lacks model_config or model_state required for evaluation"
        )
    model = create_model(model_values)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as error:
        raise typer.BadParameter(
            f"checkpoint is incompatible with configured model: {error}"
        ) from error
    values = resolved_dict(_config(None))
    data = values["data"]
    ids = [item for item in subjects.split(",") if item]
    loader = build_dataloader(
        build_dataset(records_from_index(index_path, ids), build_transforms(data, False)), 1, False
    )
    result = evaluate(model, loader, create_loss(values["loss"]), __import__("torch").device("cpu"))
    destination = output_dir or checkpoint.parent.parent / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "subjects": len(ids),
        "metrics": result,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise typer.BadParameter(f"metrics file does not exist: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@app.command("plot-dice")
def plot_dice(
    run_dir: Path = typer.Option(..., "--run-dir"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Plot centralized epoch Dice or federated round Dice from a run directory."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise typer.BadParameter("matplotlib is required to plot Dice curves") from error

    centralized = run_dir / "metrics.jsonl"
    federated = run_dir / "metrics" / "evaluation_rounds.jsonl"
    if federated.exists():
        records = _read_jsonl(federated)
        x_key = "round"
        series = {
            "Mean Dice": "validation_mean_dice",
            "WT Dice": "validation_wt_dice",
            "TC Dice": "validation_tc_dice",
            "ET Dice": "validation_et_dice",
            "Loss": "validation_loss",
        }
        xlabel = "Federated Round"
    elif centralized.exists():
        records = _read_jsonl(centralized)
        x_key = "epoch"
        series = {
            "Mean Dice": "validation_mean_dice",
            "WT Dice": "validation_wt_dice",
            "TC Dice": "validation_tc_dice",
            "ET Dice": "validation_et_dice",
            "Loss": "validation_loss",
        }
        xlabel = "Epoch"
    else:
        raise typer.BadParameter(
            f"no centralized or federated metrics found under {run_dir}"
        )

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    xs = [record[x_key] for record in records]
    for label, key in series.items():
        ys = [record.get(key) for record in records]
        if not any(value is not None for value in ys):
            continue
        axis = ax2 if key.endswith("loss") else ax1
        axis.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Dice")
    ax2.set_ylabel("Loss")
    ax1.set_ylim(0, 1)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    destination = output or run_dir / "metrics" / "dice_curve.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    typer.echo(str(destination))


@app.command("visualize-segmentation")
def visualize_segmentation(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    index_path: Path | None = typer.Option(
        None,
        "--index-path",
        help=(
            "Dataset index CSV. Defaults to DATASET_INDEX_PATH or "
            "FL_BRATS_DATA_ROOT/processed/brats2021_v1/dataset_index.csv."
        ),
    ),
    subjects: str = typer.Option(..., "--subjects", help="Comma-separated subject IDs"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    device: str = typer.Option("auto", "--device"),
    threshold: float = typer.Option(0.5, "--threshold", min=0.0, max=1.0),
) -> None:
    """Save brain-scan segmentation overlays for checkpoint predictions."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
    except ImportError as error:
        raise typer.BadParameter("matplotlib, numpy, and torch are required") from error

    payload = load_checkpoint(checkpoint)
    model_values = payload.get("model_config")
    if not isinstance(model_values, dict) or "model_state" not in payload:
        raise typer.BadParameter("checkpoint lacks model_config or model_state")
    selected = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    model = create_model(model_values).to(selected)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if index_path is None:
        if value := os.environ.get("DATASET_INDEX_PATH"):
            index_path = Path(value)
        elif data_root := os.environ.get("FL_BRATS_DATA_ROOT"):
            index_path = Path(data_root) / "processed" / "brats2021_v1" / "dataset_index.csv"
        else:
            raise typer.BadParameter(
                "pass --index-path or set DATASET_INDEX_PATH/FL_BRATS_DATA_ROOT"
            )
    if not index_path.is_file():
        raise typer.BadParameter(f"dataset index CSV does not exist: {index_path}")
    values = resolved_dict(_config(None))
    data = values["data"]
    ids = [item.strip() for item in subjects.split(",") if item.strip()]
    loader = build_dataloader(
        build_dataset(records_from_index(index_path, ids), build_transforms(data, False)),
        1,
        False,
        workers=0,
        pin_memory=False,
    )
    destination = output_dir or checkpoint.parent.parent / "visualizations" / "segmentation"
    destination.mkdir(parents=True, exist_ok=True)
    colors = {
        "wt": np.array([1.0, 0.25, 0.15]),
        "tc": np.array([0.1, 0.75, 1.0]),
        "et": np.array([1.0, 0.85, 0.1]),
    }
    saved: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(selected) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = logits_from_output(forward_batch(model, batch))
            prediction = (torch.sigmoid(logits)[0].detach().cpu() >= threshold).numpy()
            label = batch["label"][0].detach().cpu().numpy().astype(bool)
            image = batch["image"][0, 3].detach().cpu().numpy()
            subject_id = str(batch["subject_id"][0])
            z = int(np.argmax(label.sum(axis=(0, 1, 2))))
            if label[:, :, :, z].sum() == 0:
                z = image.shape[-1] // 2
            base = image[:, :, z]
            base = (base - base.min()) / (base.max() - base.min() + 1e-8)
            overlays = [("Ground truth", label), ("Prediction", prediction)]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for axis, (title, mask) in zip(axes, overlays, strict=True):
                rgb = np.repeat(base[..., None], 3, axis=-1)
                for channel, key in enumerate(("wt", "tc", "et")):
                    active = mask[channel, :, :, z]
                    rgb[active] = 0.55 * rgb[active] + 0.45 * colors[key]
                axis.imshow(np.rot90(rgb), interpolation="nearest")
                axis.set_title(title)
                axis.axis("off")
            fig.suptitle(f"{subject_id} slice {z}")
            fig.tight_layout()
            path = destination / f"{subject_id}_slice_{z:03d}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            saved.append(str(path))
    typer.echo(json.dumps({"saved": saved}, indent=2))


def _experiment_values(config: Path | None, context: typer.Context) -> dict[str, Any]:
    """Load a standalone experiment YAML or the repository's composed config."""
    if config is None:
        return _federated_runtime_values(_override_values(context))
    loaded = _load_experiment_yaml(config)
    environment_overrides = {
        "FL_BRATS_OUTPUT_ROOT": ("output_root",),
        "FL_BRATS_LOG_ROOT": ("log_root",),
        "FL_BRATS_CACHE_ROOT": ("cache_root",),
        "FL_BRATS_CHECKPOINT_ROOT": ("checkpoint_root",),
        "FL_BRATS_PARTITION_ROOT": ("data", "partition_root"),
        "FL_BRATS_INITIAL_WEIGHTS": ("initial_weights",),
        "FL_BRATS_DEVICE": ("device",),
    }
    for variable, keys in environment_overrides.items():
        if variable not in os.environ:
            continue
        destination = loaded
        for key in keys[:-1]:
            destination = destination.setdefault(key, {})
        destination[keys[-1]] = os.environ[variable]
    if context.args:
        override = OmegaConf.to_container(OmegaConf.from_dotlist(list(context.args)), resolve=True)
        if not isinstance(override, dict):
            raise typer.BadParameter("experiment overrides must be key=value assignments")
        loaded = _deep_merge(loaded, override)
    return _federated_runtime_values(loaded)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML mappings while replacing scalar and list values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_experiment_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML-first experiment config with optional relative ``extends`` files."""
    resolved = path.expanduser().resolve()
    chain = set() if seen is None else set(seen)
    if resolved in chain:
        raise typer.BadParameter(f"cyclic experiment config inheritance at {resolved}")
    chain.add(resolved)
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"experiment config must contain a YAML mapping: {resolved}")
    parents = loaded.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise typer.BadParameter(f"extends must be a path or list of paths: {resolved}")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (resolved.parent / parent).resolve()
        merged = _deep_merge(merged, _load_experiment_yaml(parent_path, chain))
    return _deep_merge(merged, loaded)


def _federated_runtime_values(values: dict[str, Any]) -> dict[str, Any]:
    """Expose nested Hydra config groups as the flat keys used by Flower helpers."""
    runtime = dict(values)
    data = values.get("data", {})
    training = values.get("training", {})
    federated = values.get("federated", {})
    optimizer = values.get("optimizer", {})
    if isinstance(data, dict):
        runtime.setdefault("partition_root", data.get("partition_root"))
    if isinstance(training, dict):
        runtime.setdefault("local_batch_size", training.get("batch_size"))
        runtime.setdefault("batch_size", training.get("batch_size"))
        runtime.setdefault("num_workers", training.get("num_workers"))
        runtime.setdefault("pin_memory", training.get("pin_memory"))
        if training.get("max_rounds") is not None:
            runtime["max_rounds"] = training.get("max_rounds")
            runtime["num_rounds"] = training.get("max_rounds")
    if isinstance(federated, dict):
        runtime.update({key: value for key, value in federated.items() if key not in runtime})
        if federated.get("strategy") is not None:
            runtime["strategy"] = federated["strategy"]
    if isinstance(optimizer, dict):
        runtime.setdefault("learning_rate", optimizer.get("lr"))
        runtime.setdefault("weight_decay", optimizer.get("weight_decay"))
    return runtime


@app.command("resolve-experiment")
def resolve_experiment(config: Path = typer.Option(..., "--config")) -> None:
    """Print a fully inherited standalone experiment YAML configuration."""
    typer.echo(yaml.safe_dump(_load_experiment_yaml(config), sort_keys=False))


@app.command("train-federated")
def train_federated(
    context: typer.Context,
    config: Path | None = typer.Option(None, "--config"),
    run_dir: Path = typer.Option(..., "--run-dir"),
) -> None:
    """Run real local Flower FedAvg clients against a gRPC Flower server."""
    values = _experiment_values(config, context)
    seed_everything(
        int(values.get("seed", 42)), deterministic=bool(values.get("deterministic", True))
    )
    try:
        summary = run_local_federation(values, run_dir)
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({"run_dir": str(run_dir), "status": summary["status"]}))


@app.command("sweep-training-config")
def sweep_training_config(
    context: typer.Context,
    config: Path | None = typer.Option(None, "--config"),
    sweep_config: Path | None = typer.Option(None, "--sweep-config"),
    output_root: Path = typer.Option(Path("outputs/experiments"), "--output-root"),
    force: bool = typer.Option(False, "--force"),
    full_factorial_fedavg: bool = typer.Option(False, "--full-factorial-fedavg"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run or prepare training-configuration sweeps."""
    values = _experiment_values(config, context)
    try:
        loaded = load_sweep_config(sweep_config)
        if full_factorial_fedavg:
            summaries = run_fedavg_full_factorial_sweep(
                values, loaded, output_root, dry_run=dry_run, force=force
            )
        else:
            summaries = run_sweep(values, loaded, output_root, force=force)
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1) from error
    payload = {"runs": len(summaries), "output_root": str(output_root)}
    if full_factorial_fedavg and dry_run:
        payload["dry_run"] = True
        payload["message"] = "FedAvg sweep validated with 360 unique configurations."
    typer.echo(json.dumps(payload))


@app.command(
    "start-federated-server",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def start_federated_server(
    context: typer.Context,
    config: Path | None = typer.Option(None, "--config"),
    server_address: str | None = typer.Option(None, "--server-address"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
) -> None:
    """Start the standalone Flower server for physically distributed clients."""
    values = _experiment_values(config, context)
    address = server_address or values.get("server_address")
    if not address:
        raise typer.BadParameter("--server-address or server_address in config is required")
    start_server(str(address), values, run_dir=run_dir)


@app.command(
    "start-federated-client",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def start_federated_client(
    context: typer.Context,
    client_id: str = typer.Option(..., "--client-id"),
    config: Path | None = typer.Option(None, "--config"),
    server_address: str | None = typer.Option(None, "--server-address"),
) -> None:
    """Connect one data-isolated model-aware client to a Flower server."""
    import flwr as fl

    values = _experiment_values(config, context)
    address = server_address or values.get("server_address")
    if not address:
        raise typer.BadParameter("--server-address or server_address in config is required")
    root = Path(values.get("partition_root") or values.get("data", {}).get("partition_root", ""))
    fl.client.start_numpy_client(
        server_address=str(address), client=build_client(values, client_id, root)
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


@app.command("verify-pretrained-model")
def verify_pretrained_model(model_config: Path = typer.Option(..., "--model-config")) -> None:
    """Verify a configured SAM checkpoint before any cluster job starts."""
    try:
        typer.echo(
            json.dumps(verify_pretrained(yaml.safe_load(model_config.read_text())), indent=2)
        )
    except ValueError as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1) from error


def _sam_command(model: str, context: typer.Context) -> None:
    """Intentional lightweight launcher: orchestration supplies manifests/loaders."""
    values = _override_values(context)
    configured = values.get("model", {}).get("name")
    if configured not in {model, None}:
        raise typer.BadParameter(f"configuration model must be {model}")
    typer.echo(
        f"{model} federated run configuration accepted; use existing Flower server/client launch with this model group."
    )


@app.command("train-federated-sam1")
def train_federated_sam1(context: typer.Context) -> None:
    _sam_command("sam1", context)


@app.command("train-federated-sam2")
def train_federated_sam2(context: typer.Context) -> None:
    _sam_command("sam2", context)


@app.command("evaluate-federated-sam1")
def evaluate_federated_sam1(context: typer.Context) -> None:
    _sam_command("sam1", context)


@app.command("evaluate-federated-sam2")
def evaluate_federated_sam2(context: typer.Context) -> None:
    _sam_command("sam2", context)


@app.command("compare-sam-to-baseline")
def compare_sam_command(
    baseline_summary: Path = typer.Option(..., "--baseline-summary"),
    sam1_summary: Path | None = typer.Option(None, "--sam1-summary"),
    sam2_summary: Path | None = typer.Option(None, "--sam2-summary"),
    output_dir: Path = typer.Option(
        Path("outputs/architecture_replacement/comparison"), "--output-dir"
    ),
) -> None:
    """Create two U-Net-reference comparisons; oracle runs are rejected."""
    try:
        typer.echo(
            json.dumps(
                compare_sam_to_baseline(baseline_summary, sam1_summary, sam2_summary, output_dir),
                indent=2,
            )
        )
    except ValueError as error:
        typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1) from error
