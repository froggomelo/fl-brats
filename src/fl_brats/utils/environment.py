"""Non-sensitive environment inspection and report serialization."""

import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import psutil
import yaml

from fl_brats.utils.git import get_git_branch, get_git_commit, is_git_dirty


@dataclass
class EnvironmentReport:
    """Portable, non-secret infrastructure report."""

    os: str
    kernel: str
    architecture: str
    python_version: str
    executable: str
    hostname: str
    username: str
    cpu_model: str | None
    physical_cpus: int | None
    logical_cpus: int | None
    total_ram: int
    available_ram: int
    torch_version: str | None
    cuda_compiled: str | None
    cuda_available: bool
    cudnn_version: int | None
    gpu_count: int
    gpu_names: list[str]
    gpu_memory: list[int | None]
    monai_version: str | None
    flower_version: str | None
    hydra_version: str | None
    omegaconf_version: str | None
    mlflow_version: str | None
    optuna_version: str | None
    uv_version: str | None
    apptainer_version: str | None
    slurm_job_id: str | None
    slurm_node_name: str | None
    slurm_partition: str | None
    cwd: str
    git_branch: str | None
    git_commit: str | None
    git_dirty: bool | None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=False
        )
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except OSError:
        return None


def collect_environment() -> EnvironmentReport:
    """Collect safe environment facts without exposing environment variables."""
    cpu_model = platform.processor() or None
    torch_version = cuda_compiled = None
    cuda_available = False
    cudnn_version = None
    gpu_names: list[str] = []
    gpu_memory: list[int | None] = []
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_compiled = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        cudnn_version = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(index))
                gpu_memory.append(torch.cuda.get_device_properties(index).total_memory)
    except ImportError:
        pass
    memory = psutil.virtual_memory()
    return EnvironmentReport(
        os=platform.system(),
        kernel=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        executable=sys.executable,
        hostname=socket.gethostname(),
        username=os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        cpu_model=cpu_model,
        physical_cpus=psutil.cpu_count(logical=False),
        logical_cpus=psutil.cpu_count(),
        total_ram=memory.total,
        available_ram=memory.available,
        torch_version=torch_version,
        cuda_compiled=cuda_compiled,
        cuda_available=cuda_available,
        cudnn_version=cudnn_version,
        gpu_count=len(gpu_names),
        gpu_names=gpu_names,
        gpu_memory=gpu_memory,
        monai_version=_version("monai"),
        flower_version=_version("flwr"),
        hydra_version=_version("hydra-core"),
        omegaconf_version=_version("omegaconf"),
        mlflow_version=_version("mlflow"),
        optuna_version=_version("optuna"),
        uv_version=_command_version("uv"),
        apptainer_version=_command_version("apptainer"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_node_name=os.environ.get("SLURMD_NODENAME"),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        cwd=str(Path.cwd()),
        git_branch=get_git_branch(),
        git_commit=get_git_commit(),
        git_dirty=is_git_dirty(),
    )


def write_environment_report(
    report: EnvironmentReport, path: str | Path, format: Literal["json", "yaml", "text"]
) -> Path:
    """Serialize a report to a user-selected JSON, YAML, or text format."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Some optional dependency versions (notably torch's TorchVersion) are
    # str subclasses that PyYAML does not represent. Normalize them first.
    payload: dict[str, Any] = json.loads(json.dumps(asdict(report), default=str))
    if format == "json":
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif format == "yaml":
        target.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    else:
        target.write_text(
            "\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n", encoding="utf-8"
        )
    return target
