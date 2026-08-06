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


class MissingModalitiesConfig(BaseModel):
    """Strict configuration for explicit BraTS modality availability."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    mode: Literal["none", "complete", "arbitrary", "mixed"] = "none"
    modalities: list[str] = Field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    seed: int = Field(default=42, ge=0)
    require_at_least_one: bool = True
    complete: dict[str, object] = Field(default_factory=dict)
    arbitrary: dict[str, object] = Field(default_factory=dict)
    baseline: dict[str, object] = Field(default_factory=dict)
    evaluation: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_policy(self) -> "MissingModalitiesConfig":
        from fl_brats.data.modalities import MODALITIES, policy_from_config

        if self.modalities != list(MODALITIES):
            raise ValueError("modalities must be exactly [t1, t1ce, t2, flair] in canonical order")
        policy_from_config(self.model_dump())
        if self.mode == "none" and self.enabled:
            raise ValueError("missing_modalities.enabled=true requires a non-none mode")
        strategy = self.baseline.get("unavailable_input_strategy", "zero_fill")
        if strategy not in {"zero_fill", "learned_token"}:
            raise ValueError(
                "baseline unavailable_input_strategy must be zero_fill or learned_token"
            )
        return self


class FixedQuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bit_width: int = Field(default=8, ge=1, le=16)


class MixedQuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_bit_widths: list[int] = Field(default_factory=lambda: [2, 4, 8, 16])
    default_bit_width: int = Field(default=8, ge=1, le=16)
    default_average_bit_width_budget: float = Field(default=8.0, ge=1.0, le=16.0)
    client_average_bit_width_budgets: dict[str, float] = Field(default_factory=dict)
    allocation_strategy: Literal[
        "uniform", "magnitude", "sensitivity", "learned", "fedmpq_inspired"
    ] = "magnitude"
    update_allocation_every_rounds: int = Field(default=1, ge=1)
    minimum_layer_bit_width: int = Field(default=2, ge=1, le=16)
    maximum_layer_bit_width: int = Field(default=16, ge=1, le=16)
    protected_layer_minimum_bit_width: int = Field(default=8, ge=1, le=16)
    protected_layer_patterns: list[str] = Field(
        default_factory=lambda: ["output", "final", "classifier", "normalization"]
    )

    @model_validator(mode="after")
    def validate_mixed(self) -> "MixedQuantizationConfig":
        if not self.candidate_bit_widths or len(set(self.candidate_bit_widths)) != len(
            self.candidate_bit_widths
        ):
            raise ValueError("mixed candidate_bit_widths must be non-empty and unique")
        if any(
            not self.minimum_layer_bit_width <= bit <= self.maximum_layer_bit_width
            for bit in self.candidate_bit_widths
        ):
            raise ValueError("candidate bit widths are outside configured min/max range")
        if (
            not self.minimum_layer_bit_width
            <= self.default_average_bit_width_budget
            <= self.maximum_layer_bit_width
        ):
            raise ValueError("default average bit-width budget is outside achievable range")
        if any(
            not self.minimum_layer_bit_width <= value <= self.maximum_layer_bit_width
            for value in self.client_average_bit_width_budgets.values()
        ):
            raise ValueError("client average bit-width budget is outside achievable range")
        return self


class QuantizationConfig(BaseModel):
    """Strict communication-time quantization configuration."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    mode: Literal["none", "fixed", "mixed"] = "none"
    transmitted_object: Literal["weights", "delta"] = "delta"
    scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    granularity: Literal["per_tensor", "per_channel"] = "per_tensor"
    rounding: Literal["nearest", "stochastic"] = "nearest"
    quantization_aware_training: bool = False
    fixed: FixedQuantizationConfig = Field(default_factory=FixedQuantizationConfig)
    mixed: MixedQuantizationConfig = Field(default_factory=MixedQuantizationConfig)
    error_feedback: dict[str, object] = Field(default_factory=lambda: {"enabled": False})
    sparsification: dict[str, object] = Field(
        default_factory=lambda: {
            "enabled": False,
            "mode": "topk_fraction",
            "fraction": 1.0,
            "error_feedback": True,
        }
    )
    communication: dict[str, object] = Field(
        default_factory=lambda: {
            "quantize_uplink": True,
            "quantize_downlink": False,
            "include_metadata_in_byte_count": True,
            "use_real_serialized_size": True,
        }
    )

    @model_validator(mode="after")
    def validate_quantization(self) -> "QuantizationConfig":
        if self.enabled and self.mode == "none":
            raise ValueError("enabled quantization requires fixed or mixed mode")
        if not self.enabled and self.mode != "none":
            raise ValueError("quantization mode requires enabled=true")
        fraction = self.sparsification.get("fraction", 1.0)
        if not 0 < float(fraction) <= 1:
            raise ValueError("sparsification fraction must be in (0, 1]")
        if self.rounding == "stochastic" and not self.enabled:
            raise ValueError(
                "stochastic rounding requires enabled quantization and configured seed"
            )
        return self


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
    pin_memory: bool = False
    precision: Literal["float32", "float16", "bfloat16"]
    resume: str | None = None
    tags: list[str] = Field(default_factory=list)
    create_missing_directories: bool = True
    environment: EnvironmentConfig
    tracking: TrackingConfig
    logging: LoggingConfig
    missing_modalities: MissingModalitiesConfig = Field(default_factory=MissingModalitiesConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
