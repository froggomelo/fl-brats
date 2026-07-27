# Configuration

Hydra composes `configs/config.yaml` from environment, tracking, and logging groups. Use `uv run fedbrats show-config environment=debug` for resolved values and `validate-config` before use. Environment-variable interpolation supports portable paths; Pydantic validation rejects invalid modes, ports, worker counts, and precision.
