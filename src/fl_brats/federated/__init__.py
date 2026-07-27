"""Flower FedAvg adapters; model and training remain framework-independent."""
"""Federated Flower adapters and communication compression support."""

from .quantization import BitWidthAllocator, QuantizedClientUpdate, dequantize_tensor, quantize_tensor

__all__ = ["BitWidthAllocator", "QuantizedClientUpdate", "dequantize_tensor", "quantize_tensor"]
