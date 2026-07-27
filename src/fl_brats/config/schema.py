"""Pydantic schemas for infrastructure-only configuration."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EnvironmentConfig(BaseModel):
    """Execution environment settings."""

    model_config = ConfigDict(extra="allow")
    mode: Literal["local", "debug", "apptainer", "cluster"] = "local"
    use_container: bool = False
    container_runtime: Literal["none", "apptainer"] = "none"
    scheduler: Literal["none", "slurm"] = "none"
    node_rank: int | None = 0
    world_size: int | None = 1
    server_address: str | None = None
    server_port: int | None = None
    use_gpu: bool | Literal["auto"] = "auto"
    image_path: str | None = None

    @field_validator("node_rank")
    @classmethod
    def validate_rank(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("node_rank must be zero or greater")
        return value

    @field_validator("world_size")
    @classmethod
    def validate_world_size(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("world_size must be at least one")
        return value

    @field_validator("server_port")
    @classmethod
    def validate_port(cls, value: int | None) -> int | None:
        if value is not None and not 1024 <= value <= 65535:
            raise ValueError("server_port must be between 1024 and 65535")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> "EnvironmentConfig":
        if self.mode == "cluster" and self.scheduler != "slurm":
            raise ValueError("cluster mode requires scheduler=slurm")
        if self.mode == "apptainer" and not self.image_path:
            raise ValueError("apptainer mode requires image_path")
        if self.mode == "cluster" and self.server_address is not None and not self.server_port:
            raise ValueError("cluster server_address requires server_port")
        return self


class TrackingConfig(BaseModel):
    """Infrastructure tracking backend settings."""

    model_config = ConfigDict(extra="allow")
    backend: Literal["local", "mlflow"] = "local"
    enabled: bool = True
    root_dir: str | None = None
    tracking_uri: str | None = None
    experiment_name: str


class LoggingConfig(BaseModel):
    """Application logging settings."""

    model_config = ConfigDict(extra="allow")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: bool = True
    file: bool = True
    rich: bool = True
    filename: str


class ProjectConfig(BaseModel):
    """Complete validated infrastructure configuration."""

    # Baseline configuration groups are intentionally extensible while the
    # infrastructure fields below retain their typed validation.
    model_config = ConfigDict(extra="allow")
    project_name: str
    experiment_name: str
    run_name: str
    seed: int = Field(ge=0, le=2**32 - 1)
    deterministic: bool
    debug: bool
    output_root: Path
    data_root: Path
    log_root: Path
    cache_root: Path
    checkpoint_root: Path
    device: str
    num_workers: int = Field(ge=0)
    precision: Literal["float32", "float16", "bfloat16"]
    resume: str | None = None
    tags: list[str] = Field(default_factory=list)
    create_missing_directories: bool = True
    environment: EnvironmentConfig
    tracking: TrackingConfig
    logging: LoggingConfig
