# Experiments

## Centralized
K = 1
Rounds = 100
B = 1

3 runs

### Centralized trained to convergence (added 2026-10-06)
K = 1
Rounds = up to 750 (the E = 5 budget), at least 150 (the E = 1 budget), then early stopping after 50 epochs without a validation gain of 0.001
B = 1

3 runs (configs/centralized/converged.yaml). Added after review: the 100-epoch runs were still improving on two of three seeds (best validation at epochs 99 and 93) and saw 100 passes over the training data against 150 for FedAvg E = 1, 750 for E = 5 and 1,500 for E = 10. The checkpoint with the best validation Dice is tested, as before. The 100-epoch runs are kept as the original protocol.

### Centralized with a fixed 200 epochs (added 2026-10-06)
K = 1
Epochs = 200, no early stopping before epoch 200
B = 1

3 runs (configs/centralized/fixed_200.yaml), queued after every other run. A fixed budget beside the converged runs; the checkpoint with the best validation Dice is tested, as before.

## Federated Baseline Uneven
K = 4
E = 1
Rounds = 100 at minimum and extened using 50 epoch stopping condition
B = 1
Beta = 50% 25% 15% 10%
Alpha = Full

3 runs

## Federated Baseline Even
K = 4
E = 1
Rounds = 100 at minimum and extened using 50 epoch stopping condition
B = 1
Beta = 25% 25% 25% 25%
Alpha = full

3 runs

## Federated Baseline with Missing Modalities Uneven
K = 4
E = 1
Rounds = 100 at minimum and extened using 50 epoch stopping condition
B = 1
Beta = 50% 25% 15% 10%
Alpha = 0.1

3 runs

Alpha = 0.2

3 runs

### Zero-filled U-Net
The full-modality FedAvg U-Net and its initial weights; missing modalities are
zero channels, with no mask channels. Runs on the Dirichlet partition below
instead of this one (configs unet_zero_fill_alpha_0_{1,2}.yaml are kept but not
scheduled).

## Missing Modalities, FedAMM-style Dirichlet Partition Uneven
Each client draws Dirichlet(alpha * ones(15)) over the 15 modality combinations,
holds one subject of each, and draws the rest from its own proportions (the FedAMM
authors' scheme). Here alpha is the Dirichlet concentration, not the per-modality
dropout probability above; small alpha means each client is dominated by one or
two combinations. Same clients, K, E, B, rounds and evaluation as above. See
docs/missing_modalities.md.

FedAMM
Alpha = 0.001, 0.1, 1
3 runs each

FedAvg-MM (modality-specific encoders; prototypes and modality weighting off)
Alpha = 0.001, 0.1, 1
3 runs each

Zero-filled U-Net
Alpha = 0.001, 0.1, 1
3 runs each

FedAMM + comm-eff (p = 0.1, b = {2,4,8})
Alpha = 0.1, 1
3 runs each

## Training Configuration Sweeps

### Client Sweeps
K = 3 

1 run

K = 5

1 run

K = 8

1 run

The splits above halve each client's share (K = 8: 439, 219, 109, 54, 27, 13,
6 and 8 patients): the "extreme imbalance" runs. A second sweep keeps the K = 4
split's shape (piecewise-linear Lorenz rule, docs/fixed_quantity_skew.md):

K = 3: 58.3 / 26.7 / 15.0 % (511, 233, 131 patients)
1 run

K = 5: 40 / 25 / 16 / 11 / 8 % (351, 218, 140, 96, 70)
1 run

K = 8: 25 / 25 / 12.5 / 12.5 / 7.5 / 7.5 / 5 / 5 % (223, 218, 109, 109, 65, 65, 43, 43)
1 run

### Local Epoch Sweeps
E = 5 
1 run

E = 10 
1 run

### Mini-batch Sweeps
B = 4
1 run

B = 8
1 run

### Modality Sweep
Alpha = 0.001
1 run

## FedAMM with Missing Modalities Uneven
Alpha = 0.1
3 runs

Alpha = 0.001
3 runs

Alpha = 0.2
3 runs

## FedMPQ Tuning Uneven
4-bit homogenous
1 run

heterogenous 
b = {2,4,8}
1 run

## Selective Transmission Uneven 
p = 0.25
1 run

p = 0.01
1 run

p = 0.1
1 run

## Best Communication Effecient Uneven
p = from above
b = {2,4,8}

3 runs

p = 0.25 had the highest validation Dice of the tuning runs, but each selected
value carries a 4-byte index, so p = 0.25 with 2/4/8-bit uploads 21.7 MB per
round against 9.7 MB for the quantization alone. Rerun with p = 0.1, the
highest-Dice p whose combination uploads less in total (about 8.7 MB):

p = 0.1, b = {2,4,8}
3 runs

### Quantized broadcast (added 2026-10-06)
p = 0.1, b = {2,4,8} uplink, 8-bit downlink
1 run (seed 42, configs/comm_eff_uneven/best_p_0_1_downlink_8bit.yaml)

Lecturer feedback: try quantizing the broadcast on the best configuration. After
round 1 the server sends the change since its last broadcast, quantized to 8
bits per tensor, with error feedback (docs/quantized_federated_learning.md).
Download per client per round: 4.82 MB against 19.24 MB.

Result (seed 42, 150 rounds, against best_p_0_1 seed 42): best validation Dice
0.8533 against 0.8525, test Dice 0.859 against 0.857; download 2.95 GB against
11.54 GB over the run, upload unchanged at 1.31 GB, total 4.25 GB against
12.85 GB. With CSR indices (logged, not sent) the upload would be about 0.73 GB.

Follow-up (added 2026-10-07): seeds 43 and 44 of the 8-bit broadcast, and a
4-bit and a 2-bit broadcast on seed 42
(configs/comm_eff_uneven/best_p_0_1_downlink_{4,2}bit.yaml), to see where the
quantized broadcast starts to cost Dice. Expected download per run: about
1.5 GB at 4 bits and 0.75 GB at 2 bits against 11.5 GB in FP32.

### Encoding the sparse indices
Each selected value carries a 4-byte flat index, so at p = 0.1 the four clients
send 7.70 MB of indices and 0.96 MB of values per round. Measured on the top 10 %
of a real global update (best_p_0_1 seed 42, round 149 to 150), per client:

| Index coding | p = 0.1 | bits per index | p = 0.01 |
|---|---|---|---|
| flat uint32 (current) | 1.924 MB | 32.0 | 0.192 MB |
| CSR, uint16 column index + row pointers | 0.971 MB | 16.2 | 0.105 MB |
| bitmap (1 bit per parameter) | 0.601 MB | 10.0 | 0.601 MB |
| gaps + varint | 0.482 MB | 8.0 | 0.060 MB |
| bitmap + lzma | 0.281 MB | 4.7 | 0.058 MB |
| gaps + Golomb-Rice | 0.289 MB | 4.8 | 0.049 MB |
| entropy bound | 0.282 MB | 4.7 | 0.049 MB |

The index coding is lossless, so the Dice of every run stays the same; only the
uplink bytes change. With Golomb-Rice gaps, best_p_0_1 would upload about 2.1 MB
per round against 8.7 MB now and 9.7 MB for 2/4/8-bit alone, and the 2-bit
clients 0.41 MB each against 1.2 MB quantized only.

CSR is implemented (`sparsification.index_encoding: csr`, 2026-10-06). Every
sparse run from now on logs the index bytes of both encodings per client and
round, so the best_p_0_1 downlink run gives the exact CSR bytes too.

## Combined FedAMM + Comm_eff
Alpha = from above
p = from above
b {2,4,8}
3 runs

Rerun with p = 0.1, as above:
3 runs

FedAvg-MM + comm-eff (added 2026-10-07): the same q = 0.1 partition and the
same compression (top 10 %, 2/4/8-bit) with FedAvg instead of FedAMM, so the two
missing-modality methods are compared under the same compression
(configs/fedavg_mm_comm_eff_uneven/best_p_0_1.yaml).
3 runs

## Model Agnostic
E = 1
B =1
Alpha = full

SAM1 
3 runs

SAM2 
3 runs

# Folder Structure
I want the sub folders to be structured like this for folders like scripts, logs, configs and whateve is relevant:

centralized/
fedavg_uneven/
fedavg_even/
fedavg_missing_modalities_uneven/
fedavg_train_configs_uneven/
    client_sweeps/
    local_epochs_sweeps/
    mini_batch_sweeps/
    modality_sweep/
fedamm_uneven/
fedmpq_uneven/
selective_transmission_uneven/
comm_eff_uneven/
fedamm_comm_eff_uneven/
model_agnotic_uneven/
    sam1/
    sam2/
data_preprocessing/
environment/
depedencies
scripts/
    monitoring/
metrics/
