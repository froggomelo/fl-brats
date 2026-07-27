"""Fine-tuning policy with explicit trainable scope."""

import torch


def configure_fine_tuning(model: torch.nn.Module, mode: str) -> list[str]:
    allowed = {"decoder_only", "prompt_decoder", "adapter_decoder", "full_model"}
    if mode not in allowed:
        raise ValueError(f"fine_tuning_mode must be one of {sorted(allowed)}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = mode == "full_model" or any(
            token in name for token in (("decoder",) if mode == "decoder_only" else ("decoder", "prompt", "projection", "adapter"))
        )
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
