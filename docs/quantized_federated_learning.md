# Quantized federated communication

U-Net updates are large and repeated every federated round.  This project can
therefore transmit packed integer **deltas** rather than FP32 model weights.
The server always dequantizes to FP32 before sample-weighted FedAvg; integer
codes are never averaged.

`fixed` uses one width for every floating state entry. `mixed` assigns a width
per layer with a parameter-weighted average budget. The deterministic greedy
allocator starts at the minimum width, protects layers matching configured
patterns, and selects upgrades by estimated error reduction per extra bit.
For example, clients with budgets 8, 6, 4, and 2 can receive different layer
assignments; their weighted means never exceed their respective budgets.

This is communication-only quantization and the `fedmpq_inspired` strategy is
a budget-aware heuristic, **not** a reproduction of FedMPQ. Local master
weights/gradients remain FP32. QAT and true integer training are not provided.
Metadata (shape, dtype, scale, zero-point, layer name and checksum) and packed
payload bytes are included in the primary application-level byte count. Network
headers and NIC-level traffic remain outside this measurement.

Use `fedbrats inspect-quantization model=unet_baseline` to inspect eligible
layers and a deterministic allocation. Typical Hydra overrides are:

```bash
fedbrats inspect-quantization quantization.enabled=true quantization.mode=fixed quantization.fixed.bit_width=8
fedbrats inspect-quantization quantization.enabled=true quantization.mode=mixed quantization.mixed.default_average_bit_width_budget=4
fedbrats inspect-quantization quantization.enabled=true quantization.mode=mixed quantization.mixed.client_average_bit_width_budgets.client_0=8
```

Missing-modality masks are batch data, never state updates; quantization is
independent of that policy. Sparse selective communication and QAT are reserved
extension points rather than silently conflated with quantization.
