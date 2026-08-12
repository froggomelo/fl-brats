# Communication-efficient federated updates

U-Net updates are large and repeated every federated round.  This project can
therefore reduce uplink communication in two separable ways: packed integer
**deltas** and selective Top-rho sparse delta transmission. The server always
reconstructs/dequantizes to FP32 before sample-weighted FedAvg; integer codes
are never averaged directly.

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
independent of that policy. QAT remains a reserved extension point rather than
silently conflated with communication-time compression.

## Selective Top-rho transmission

Selective communication applies to client-to-server updates. Each client
computes the ordinary local delta against the received global model, adds its
client-local residual, ranks all eligible floating update entries globally by
absolute value, and sends only the largest `ceil(rho * N)` entries as sparse
flat indices plus selected FP32 values. Unsent entries reconstruct as zero
updates on the server, not as zero model weights.

The residual is updated as `u_t - reconstructed(compressed_t)`, so omitted
changes persist across rounds for the same client. With quantization disabled,
selected values remain FP32:

```bash
fedbrats train-federated quantization.enabled=false quantization.sparsification.enabled=true quantization.sparsification.fraction=0.10
```

The intended sweep fractions are ordinary config values, for example `1.0`,
`0.50`, `0.25`, `0.10`, and `0.01`. Downlink remains the full global model
unless another mechanism is explicitly added.

Sparse and quantized communication can be combined in fixed-bit mode. In that
path the client selects Top-rho entries first, then quantizes only the selected
values before serialization.
