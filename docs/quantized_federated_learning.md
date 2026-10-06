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
unless the quantized broadcast below is enabled.

Sparse and quantized communication can be combined in fixed-bit mode. In that
path the client selects Top-rho entries first, then quantizes only the selected
values before serialization.

## Quantized broadcast (downlink)

`quantization.communication.quantize_downlink: true` quantizes the
server-to-client broadcast (`federated/downlink.py`); `downlink_bit_width` is
2, 4 or 8 (default 8). The server and every client keep the same reconstructed
model, the view. The first broadcast of a process (round 1, or the first round
after a resume) is the full FP32 model. Every later broadcast carries the
change since the last one, `global - view`, quantized symmetrically per tensor,
and both sides add the dequantized change to their view; the server updates its
view from the bytes it sent, so the copies stay bit-identical. What rounding
drops is still in `global - view` next round, so the error is resent rather
than accumulated (error feedback with no client state).

Clients train from the view. The server keeps the FP32 global model, which is
validated and checkpointed; compressed uplink updates are applied to it, and a
dense client average is moved onto it (`mean - view` is applied). Every client
receives the same payload, one byte per value at 8 bits plus a JSON header of
names, shapes and scales and a SHA-256 checksum: 4.82 MB for the 4,810,074
parameter U-Net against 19.24 MB in FP32. `download_payload_bytes` is the size
of the payload the client received. Encoding takes a few seconds per round on
the CPU.

```bash
fedbrats train-federated --config configs/comm_eff_uneven/best_p_0_1_downlink_8bit.yaml --run-dir outputs/...
```

## Encoding the sparse indices

`quantization.sparsification.index_encoding` chooses how the selected positions
are sent; the encoding is lossless, so training is the same either way.

- `flat` (default): one uint32 flat index per selected value.
- `csr`: compressed sparse row, layer by layer (`encode_csr_indices`). A layer
  is a matrix with one row per output channel (a vector is one row). Each layer
  with selected entries sends its row pointers (rows + 1 counts) and the column
  of every entry, both in the narrowest unsigned type that fits. Conv weights
  of the U-Net have fewer than 65,536 columns, so a column costs 2 bytes or
  less. Layers with nothing selected send nothing; the header lists every
  layer's entry count.

Every sparse client record logs `upload_index_bytes`, `upload_value_bytes`
and the index bytes of both encodings (`upload_index_bytes_flat`,
`upload_index_bytes_csr`), whichever was sent. On the top 10 % of a real U-Net
update CSR takes 0.97 MB against 1.92 MB flat; gap coding would take less (see
the comm-eff section of `Experiments.md`).

```yaml
quantization:
  sparsification: {enabled: true, fraction: 0.1, index_encoding: csr}
```
