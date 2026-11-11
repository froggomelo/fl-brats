# Communication efficiency

This guide covers the methods that shrink what clients and the server send each round, how the payload bytes are counted, and the configuration that selects each method. Read it before running `fedmpq_uneven`, `selective_transmission_uneven`, `comm_eff_uneven`, or any family that combines compression with a missing-modality method.

## Overview

A full U-Net update is 19.24 MB in 32-bit floating point (FP32), and every client sends one every round. Three independent mechanisms reduce that traffic.

| Mechanism | Direction | Main keys | Code |
|---|---|---|---|
| Mixed-precision quantisation | Client to server | `quantization.enabled`, `quantization.mode` | `src/fl_brats/federated/quantization.py` |
| Selective top-rho transmission | Client to server | `quantization.sparsification.*` | `src/fl_brats/federated/sparsification.py` |
| Quantised broadcast | Server to client | `quantization.communication.quantize_downlink` | `src/fl_brats/federated/downlink.py` |

All three compress what is sent, not how the model trains. Local weights, gradients, and optimiser state stay in FP32, and quantisation-aware training is not provided. The server always decodes every client update to FP32 before averaging. Integer codes are never averaged directly.

Defaults for every key are in `configs/quantization/default.yaml`. An experiment YAML overrides them under `quantization:`.

## Mixed-precision quantisation of uplink updates

Each client sends its update, the difference between its trained weights and the model it received, instead of the weights themselves. With `quantization.enabled: true` and sparsification off, every floating-point tensor of that update is quantised and packed.

The method follows FedMPQ, a federated mixed-precision quantisation method in which each client fits per-layer bit widths to its own average bit budget. The implementation here is a budget-aware heuristic inspired by FedMPQ. It is not a reproduction of the paper.

### Quantiser

The default quantiser is symmetric and per tensor with round-to-nearest. A tensor quantised at b bits uses integer codes from -(2^(b-1) - 1) to 2^(b-1) - 1 and one scale equal to the largest absolute value divided by the largest code. Codes are packed little-endian, least-significant code first, with b bits each.

| Key | Values | Default |
|---|---|---|
| `quantization.scheme` | `symmetric`, `asymmetric` | `symmetric` |
| `quantization.granularity` | `per_tensor`, `per_channel` | `per_tensor` |
| `quantization.rounding` | `nearest`, `stochastic` | `nearest` |
| `quantization.error_feedback.enabled` | `true`, `false` | `false` |

With error feedback enabled, each client keeps the quantisation error of every tensor and adds it to the next round's update. No experiment config enables it for dense quantisation.

### Fixed and mixed modes

`quantization.mode: fixed` sends every tensor at `quantization.fixed.bit_width`. `quantization.mode: mixed` assigns a width per tensor from `mixed.candidate_bit_widths` under a client budget.

The mixed allocator is deterministic and greedy. Every tensor starts at the smallest candidate width. A tensor whose name contains any of `mixed.protected_layer_patterns`, compared in lower case, starts at the smallest candidate of at least `mixed.protected_layer_minimum_bit_width`. The allocator then repeatedly raises one tensor to its next candidate width, choosing the largest estimated error reduction per extra bit, while the parameter-weighted mean width stays within the budget. The estimated error of a tensor at b bits is its importance divided by 4^b. In dense mode the importance is the L2 norm of the tensor's trained local weights. A budget that the minimum widths already exceed stops the run with an error.

The budget of a client is `mixed.client_average_bit_width_budgets.<client_id>`, or `mixed.default_average_bit_width_budget` for a client not listed. No U-Net or FedAMM parameter name contains the default protected patterns. No tensor is protected in the shipped configs.

`configs/fedmpq_uneven/heterogeneous_2_4_8bit.yaml` gives the largest client 8 bits and the smallest two 2 bits.

```yaml
quantization:
  enabled: true
  mode: mixed
  mixed:
    candidate_bit_widths: [2, 4, 8]
    default_average_bit_width_budget: 4.0
    client_average_bit_width_budgets: {client_0: 8.0, client_1: 4.0, client_2: 2.0, client_3: 2.0}
    protected_layer_minimum_bit_width: 8
    protected_layer_patterns: [output, final, classifier, normalization]
```

The schema also accepts `transmitted_object`, `quantization_aware_training`, `mixed.default_bit_width`, `mixed.allocation_strategy`, `mixed.update_allocation_every_rounds`, `mixed.minimum_layer_bit_width`, `mixed.maximum_layer_bit_width`, `communication.quantize_uplink`, `communication.include_metadata_in_byte_count`, and `communication.use_real_serialized_size`. Training reads none of them. The allocator always runs every round with the greedy rule above. `fedbrats validate-config` checks these keys, including the minimum and maximum widths, on the composed Hydra configuration only. Experiment YAMLs passed to `train-federated --config` are not schema-validated.

### Inspecting an allocation

`fedbrats inspect-quantization` builds the model of the composed Hydra configuration and prints the eligible tensors, the protected tensors, the width of each, the achieved mean width, and the estimated payload against FP32. It gives every tensor the same importance. The printed allocation therefore depends on tensor sizes alone and differs from a training allocation.

```bash
fedbrats inspect-quantization model=unet_baseline \
  quantization.enabled=true quantization.mode=fixed quantization.fixed.bit_width=8
fedbrats inspect-quantization model=fedamm \
  quantization.enabled=true quantization.mode=mixed quantization.mixed.default_average_bit_width_budget=4
```

### Wire format

A quantised update is one uint8 array. It holds a 4-byte header length, a JSON header with each tensor's name, shape, dtype, bit width, scale, zero point, and code count, the packed codes, and a 64-character SHA-256 checksum of header and codes. The server rejects a packet whose checksum or round does not match.

## Selective top-rho transmission

Selective transmission, also called top-k sparsification, sends only the largest entries of each update. Set `quantization.sparsification.enabled: true` and choose the fraction rho with `fraction`.

```yaml
quantization:
  sparsification: {enabled: true, method: topk, fraction: 0.1, error_feedback: true, scope: global}
```

Each client computes its update against the model it received and adds its residual from earlier rounds. It ranks all floating-point entries of the whole model together by absolute value and keeps the largest `ceil(fraction × N)`, where N is the number of floating-point parameters. It sends the flat positions of those entries and their values. An entry that is not sent counts as a zero update on the server, not a zero weight.

The residual is the part of the corrected update that the server will not reconstruct. It covers the unsent entries and the quantisation error of the sent ones. Each client keeps its residual between rounds and stores it with its optimiser state under the run directory. A resumed run continues with the stored residual. Set `error_feedback: false` to drop the residual.

| Key | Values | Default |
|---|---|---|
| `sparsification.enabled` | `true`, `false` | `false` |
| `sparsification.fraction` | (0, 1] | `1.0` |
| `sparsification.error_feedback` | `true`, `false` | `true` |
| `sparsification.index_encoding` | `flat`, `csr` | `flat` |
| `sparsification.method` | `topk` only | `topk` |
| `sparsification.scope` | `global` only | `global` |

With `quantization.enabled: false` the selected values travel as FP32. With quantisation enabled, the client selects first and then quantises only the selected values. In `fixed` mode all selected values share one width. In `mixed` mode the allocator above runs over the layers that have selected entries. Each layer is weighted by its number of selected entries, and its importance is the L2 norm of those entries. Each layer then gets its own width and scale, which requires `granularity: per_tensor`.

The server sums the decoded sparse updates weighted by client sample count and adds the result to its FP32 global model. A round cannot mix sparse and dense client payloads.

A sparse update is two uint8 arrays. The first is the 24-byte marker `FLBRATS_SPARSE_UPDATE_V1`. The second holds a 4-byte header length, a JSON header, the index bytes, the value bytes, and a SHA-256 checksum.

### Encoding the sparse indices

`sparsification.index_encoding` chooses how the selected positions are sent. Both encodings are lossless. Training is identical with either.

| Encoding | Bytes per selected entry |
|---|---|
| `flat` | One uint32 flat index, or uint64 for models with 2^32 or more parameters |
| `csr` | Compressed sparse row, layer by layer, as described below |

In `csr` each layer is a matrix with one row per output channel, and a one-dimensional tensor is a single row. A layer with selected entries sends its row pointers, which are rows + 1 counts, and the column of every entry. Both use the narrowest unsigned type that fits. Every U-Net convolution has fewer than 65,536 columns. A column therefore costs at most 2 bytes. A layer with no selected entries sends nothing, and the header lists every layer's entry count.

Every sparse client record logs the index bytes of both encodings, whichever was sent. No shipped config sets `csr`. The measured comparison is in the "Encoding the sparse indices" section of [Experiments.md](../Experiments.md).

```bash
fedbrats train-federated --config configs/comm_eff_uneven/best_p_0_1.yaml \
  --run-dir outputs/<run> quantization.sparsification.index_encoding=csr
```

A client at fraction p with mean width b sends about p × N × (4 + b/8) bytes with flat indices, against N × b/8 bytes for quantisation alone. Sparsification at a large p can therefore upload more than quantisation alone.

## Quantised broadcast

`quantization.communication.quantize_downlink: true` quantises the server-to-client broadcast. `downlink_bit_width` is 2, 4, or 8 and defaults to 8.

```yaml
quantization:
  communication: {quantize_downlink: true, downlink_bit_width: 8}
```

The server and every client keep the same reconstructed model, called the view. The first broadcast of a process is the full FP32 model. That is round 1, or the first round after a resume. Every later broadcast carries the change since the last broadcast, `global - view`, quantised symmetrically per tensor. Both sides add the dequantised change to their view. The server updates its view from the bytes it sent, and the copies stay bit-identical. What rounding drops stays in `global - view` and goes out in the next broadcast. The error is resent rather than accumulated, which is error feedback with no client state.

Clients train from the view. The server keeps the FP32 global model, which is the model it validates and checkpoints. Compressed uplink updates apply to that model directly. When the uplink is dense, the server applies the client average minus the view to its FP32 model.

Every client receives the same payload. It holds the 4-byte marker `FLDL`, a header length, a JSON header of tensor names, shapes, and scales, the packed codes, and a SHA-256 checksum. At 8 bits the 4,810,074-parameter U-Net takes 4.82 MB per client against 19.24 MB in FP32.

```bash
python scripts/launch_experiment.py configs/comm_eff_uneven/best_p_0_1_downlink_8bit.yaml
```

## How payload bytes are counted

Bytes are counted at the application level from the exact arrays each side hands to Flower. Network headers, gRPC framing, and traffic at the network interface are not measured. Flower exposes no network-only transfer time, and `network_transmission_seconds` is always null.

| Payload | What `upload_payload_bytes` or `download_payload_bytes` counts |
|---|---|
| Dense FP32 uplink | The sum of the NumPy array sizes |
| Quantised uplink | The full packet, including header, codes, and checksum |
| Sparse uplink | The marker array plus the full packet, including header, indices, values, and checksum |
| FP32 broadcast | The sum of the NumPy array sizes the client received |
| Quantised broadcast | The full packet the client received |

The run directory records the counts in four files.

| File | Content |
|---|---|
| `metrics/clients.jsonl` | Per client and round, `upload_payload_bytes`, `download_payload_bytes`, and for sparse uplinks `upload_index_bytes`, `upload_value_bytes`, `upload_index_bytes_flat`, and `upload_index_bytes_csr` |
| `communication/rounds.jsonl` | Per round, the bytes uploaded and downloaded by all clients |
| `state.json` | Running totals `uploaded_bytes` and `downloaded_bytes` |
| `communication/summary.json` | The totals, the curve of validation Dice, the segmentation overlap score, against cumulative bytes, and its Pareto frontier |

FedAMM prototypes and the server prototype library travel in Flower's metrics and configuration dictionaries. They are not counted.

## Combining compression with FedAMM

Compression applies unchanged to FedAMM, the prototype-based missing-modality method, and to FedAvg-MM, the FedAMM network trained with plain Federated Averaging (FedAvg). [Missing modalities](missing_modalities.md) describes both. Under a sparse or quantised uplink, FedAMM still exchanges prototypes and still adds the prototype losses. Aggregation changes. The server decodes each update and applies the sample-weighted mean to every tensor, including the modality encoders. Modality-weighted encoder aggregation is not applied. Global top-rho selection does not keep the layers of each modality complete.

`configs/fedamm_comm_eff_uneven/best_p_0_1.yaml` shows the full combination used in the report.

```yaml
extends: ../fedamm_uneven/alpha_0_1.yaml
federated: {strategy: fedamm}
quantization:
  enabled: true
  mode: mixed
  mixed:
    candidate_bit_widths: [2, 4, 8]
    client_average_bit_width_budgets: {client_0: 8.0, client_1: 4.0, client_2: 2.0, client_3: 2.0}
  sparsification: {enabled: true, method: topk, fraction: 0.25, error_feedback: true, scope: global}
```

## Config folders

| Folder | Files | Setting |
|---|---|---|
| `configs/quantization/` | `default.yaml` | Defaults for every key, with compression off |
| `configs/fedmpq_uneven/` | `homogeneous_4bit.yaml` | Dense fixed 4-bit uplink |
| | `heterogeneous_2_4_8bit.yaml` | Dense mixed uplink with budgets 8, 4, 2, and 2 bits |
| `configs/selective_transmission_uneven/` | `p_0_25.yaml`, `p_0_1.yaml`, `p_0_01.yaml` | Top-rho with FP32 values at rho 0.25, 0.1, and 0.01 |
| `configs/comm_eff_uneven/` | `best.yaml`, `best_p_0_1.yaml` | Top-rho at 0.25 or 0.1 with mixed 2/4/8-bit values |
| | `best_p_0_1_downlink_{2,4,8}bit.yaml` | `best_p_0_1` with a quantised broadcast |
| | `best_p_0_1_k5.yaml`, `best_p_0_1_k8.yaml` | `best_p_0_1` with 5 or 8 clients |
| `configs/fedamm_comm_eff_uneven/` | `best.yaml`, `best_p_0_1.yaml` | FedAMM at dropout alpha 0.1 with the `comm_eff_uneven` settings |
| | `dirichlet_alpha_0_1.yaml`, `dirichlet_alpha_1.yaml`, `per_client_flair_largest.yaml` | `best_p_0_1` on the other missing-modality partitions |
| | `dirichlet_alpha_0_1_k5.yaml`, `dirichlet_alpha_0_1_k8.yaml` | The Dirichlet run with 5 or 8 clients |
| `configs/fedavg_mm_comm_eff_uneven/` | `best_p_0_1.yaml`, `dirichlet_alpha_0_1*.yaml`, `per_client_flair_largest.yaml` | FedAvg-MM with the `best_p_0_1` compression |
| `configs/fedamm_comm_eff_ablation_uneven/` | `fedmpq_*.yaml`, `selective_*.yaml` | FedAMM with quantisation alone or selective transmission alone |

The 5- and 8-client configs add per-client budgets from `configs/_experiment_base/clients_lorenz/k{5,8}_bit_budgets.yaml`. The largest client gets 8 bits, the second 4, and the rest 2.

Launch any of these with the experiment launcher. [Experiments](experiments.md) describes its options and outputs.

```bash
python scripts/launch_experiment.py configs/comm_eff_uneven/best_p_0_1.yaml
```
