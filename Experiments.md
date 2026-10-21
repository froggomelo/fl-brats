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

### Dataset cache check (added 2026-10-07)
K = 4, E = 1, B = 1, seed 42, 50 rounds (configs/cache_check_uneven/)

FedAvg on the uneven split exactly as the campaign runs it (server and four
clients on five nodes), twice without the dataset cache (cache_mode: dataset)
and twice with the persistent cache, each in a fresh directory on the client
nodes so it is built in round 1 and read afterwards. Compares losses and round
times. An earlier one-round check on a single node (job 64800, 6 October) found
no measurable difference in the losses and a 36 % faster round once the cache
existed.

Result (50 rounds each): test Dice 0.829 / 0.818 with the cache (mean 0.823)
and 0.822 / 0.823 without (mean 0.822); median round 98 / 70 s against
198 / 199 s, training 1.42 / 1.07 h against 2.79 / 2.80 h. No systematic effect
on accuracy; the cache halves the training time.

### Centralized with a fixed 200 epochs (added 2026-10-06)
K = 1
Epochs = 200, no early stopping before epoch 200
B = 1

3 runs (configs/centralized/fixed_200.yaml), queued after every other run. A fixed budget beside the converged runs; the checkpoint with the best validation Dice is tested, as before.

Results (test mean Dice, seeds 42 / 43 / 44, mean): centralized 100 epochs
0.870 / 0.864 / 0.865 (0.866); trained to convergence (283 / 279 / 254 epochs)
0.881 / 0.876 / 0.878 (0.879); fixed 200 epochs 0.879 / 0.878 / 0.877 (0.878).
FedAvg on the same split: E = 1 0.864, E = 5 0.886, E = 10 0.887 (means). The
100-epoch reference was undertrained; trained longer, centralized beats FedAvg
E = 1 and trails E = 5 and E = 10 by about 0.008.

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
zero channels, with no mask channels. First run on the Dirichlet partition
below; added on this one as well on 2026-10-08 (q = 0.1 and 0.2, seeds 42-44,
configs unet_zero_fill_alpha_0_{1,2}.yaml), the control for FedAMM and FedAvg-MM
at q.
6 runs

Result (test Dice, seeds 42 / 43 / 44, mean; then the mean over the 15 forced
modality combinations and with a single scan):
q = 0.1: zero-filled U-Net 0.816 / 0.811 / 0.811, 0.813 (0.629, 0.391);
FedAvg-MM 0.830 (0.679, 0.488); FedAMM 0.829 (0.672, 0.479).
q = 0.2: zero-filled U-Net 0.784 / 0.785 / 0.780, 0.783 (0.667, 0.496);
FedAvg-MM 0.805 (0.702, 0.547); FedAMM 0.804 (0.700, 0.550).
The missing-modality architectures beat zero-filling by about 0.02 with each
subject's own scans and by 0.05 to 0.10 when scans are absent.

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

## Per-client Missing Modality Uneven (added 2026-10-19)
Every subject of a client lacks the same scan, and the largest client loses
FLAIR: client 0 (437 subjects) no FLAIR, client 1 (218) no T2, client 2 (131)
no T1ce, client 3 (89) no T1. No client ever trains on all four. Same subjects,
K, E, B and rounds as the uneven split (mode `complete`,
configs/_experiment_base/missing_per_client/flair_largest.yaml). Held-out
subjects each miss one scan, drawn with the training shares (49.9 % no FLAIR,
24.9 % no T2, 15.0 % no T1ce, 10.2 % no T1). See docs/missing_modalities.md.

FedAMM, FedAvg-MM, zero-filled U-Net, FedAMM + comm-eff, FedAvg-MM + comm-eff
(comm-eff = top 10 %, 2/4/8-bit; configs/*/per_client_flair_largest.yaml)
3 runs each, 15 runs on batch

All 15 completed 150 rounds (none early-stopped) and were evaluated by
2026-10-21 18:55 UTC.

Result, test Dice (seeds 42 / 43 / 44, mean ± SD; upload per run, all four
clients):
FedAMM 0.804 / 0.782 / 0.804, 0.797 ± 0.013, 2.43 GB;
FedAvg-MM 0.798 / 0.795 / 0.818, 0.804 ± 0.013, 2.43 GB;
zero-filled U-Net 0.777 / 0.789 / 0.783, 0.783 ± 0.006, 11.54 GB;
FedAMM + comm-eff 0.792 / 0.788 / 0.799, 0.793 ± 0.005, 0.30 GB;
FedAvg-MM + comm-eff 0.792 / 0.788 / 0.805, 0.795 ± 0.009, 0.30 GB.

Test Dice with every test subject forced to one set of scans (mean of the three
seeds; all four / no FLAIR / no T2 / no T1ce / no T1):

| Method | All four | No FLAIR | No T2 | No T1ce | No T1 |
|---|---|---|---|---|---|
| FedAMM | 0.860 | 0.850 | 0.833 | 0.542 | 0.827 |
| FedAvg-MM | 0.858 | 0.848 | 0.832 | 0.571 | 0.840 |
| Zero-filled U-Net | 0.823 | 0.827 | 0.833 | 0.574 | 0.791 |
| FedAMM + comm-eff | 0.855 | 0.846 | 0.823 | 0.550 | 0.818 |
| FedAvg-MM + comm-eff | 0.861 | 0.848 | 0.831 | 0.537 | 0.825 |

FedAvg-MM and FedAMM are within the seed spread of each other (0.007 apart, SD
0.013), and the compression costs each of them at most 0.009, also within it.
Both are ahead of the zero-filled U-Net, which also uploads almost five times
as much as the uncompressed modality-specific models. Which scan is missing
matters far more than the method: without T1ce, which only the 131-subject
client lacks, test Dice falls to 0.54 to 0.57; without FLAIR, the scan the
largest client never sees, it stays at 0.83 to 0.85. The modality-specific
methods do best with all four scans although no client ever had them together.
The zero-filled U-Net does not: it scores 0.823 with all four against 0.827
without FLAIR, the input it saw most often in training. Against q = 0.1
(FedAMM 0.829, FedAvg-MM 0.830), the per-client split costs about 0.03.

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

Three seeds (42 / 43 / 44): test Dice 0.859 / 0.866 / 0.861, mean 0.862 ± 0.004,
against 0.857 / 0.866 / 0.862, mean 0.862 ± 0.005, with the FP32 broadcast;
total traffic per run 4.25 GB against 12.85 GB.

Bit width of the broadcast (seed 42, test Dice / total traffic per run): FP32
0.857 / 12.85 GB, 8-bit 0.859 / 4.25 GB, 4-bit 0.859 / 2.82 GB, 2-bit 0.859 /
2.10 GB. Even a 2-bit broadcast costs no Dice here; error feedback resends what
the rounding drops.

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

FedAvg-MM + comm-eff (added 2026-10-07): the same partitions and the same
compression (top 10 %, 2/4/8-bit) with FedAvg instead of FedAMM, so the two
missing-modality methods are compared under the same compression. First on the
Dirichlet partition at alpha 0.1
(configs/fedavg_mm_comm_eff_uneven/dirichlet_alpha_0_1.yaml), then at q = 0.1
(configs/fedavg_mm_comm_eff_uneven/best_p_0_1.yaml).
3 + 3 runs

Dirichlet alpha 0.1, test Dice (seeds 42 / 43 / 44, mean ± SD, traffic per run):
FedAvg-MM + comm-eff 0.736 / 0.723 / 0.729, 0.729 ± 0.007, 2.73 GB; FedAMM +
comm-eff 0.729 / 0.724 / 0.720, 0.724 ± 0.005, 2.73 GB; FedAvg-MM 0.727 /
0.726 / 0.726, 0.726 ± 0.001, 4.86 GB; FedAMM 0.724 / 0.724 / 0.736, 0.728 ±
0.007, 4.86 GB. All four within the seed spread; the compression costs FedAvg-MM
nothing here.

q = 0.1, test Dice (seeds 42 / 43 / 44, mean): FedAvg-MM + comm-eff 0.829 /
0.829 / 0.822, 0.827, 2.73 GB; FedAMM + comm-eff 0.830 / 0.829 / 0.827, 0.829;
FedAvg-MM 0.832 / 0.833 / 0.824, 0.830, 4.86 GB; FedAMM 0.830 / 0.832 / 0.827,
0.829. Again all within the seed spread.

Ablation (added 2026-10-13): FedAMM with each half of the compression alone,
FedMPQ 2/4/8-bit only and top 10 % only (FP32 values), at q = 0.1 and on the
Dirichlet partition at alpha 0.1
(configs/fedamm_comm_eff_ablation_uneven/{fedmpq,selective}_{alpha_0_1,dirichlet_alpha_0_1}.yaml).
3 seeds each, 12 runs on batch. Batch lost 98 of its 100 nodes on 2026-10-13, so
interim copies of all 12 ran on bigbatch (RTX 3090, bigbatch/ configs); the
batch runs stay queued and are the reported results.

Batch recovered on 2026-10-17; all 12 batch runs completed 150 rounds (none
early-stopped) and were evaluated by 2026-10-18 23:40 UTC.

Result on batch (RTX 3060), test Dice (seeds 42 / 43 / 44, mean ± SD; mean over
the 15 forced combinations; upload per run, all four clients):
q = 0.1: FedAMM 0.830 / 0.832 / 0.827, 0.829 ± 0.002 (0.672), 2.43 GB;
FedMPQ only 0.830 / 0.832 / 0.825, 0.829 ± 0.004 (0.669), 0.33 GB;
top 10 % only 0.830 / 0.830 / 0.824, 0.828 ± 0.003 (0.669), 0.50 GB;
both 0.830 / 0.829 / 0.827, 0.829 ± 0.002 (0.674), 0.30 GB.
Dirichlet alpha 0.1: FedAMM 0.724 / 0.724 / 0.736, 0.728 ± 0.007 (0.721);
FedMPQ only 0.718 / 0.714 / 0.720, 0.717 ± 0.003 (0.710);
top 10 % only 0.731 / 0.720 / 0.719, 0.723 ± 0.007 (0.717);
both 0.729 / 0.724 / 0.720, 0.724 ± 0.005 (0.718).
At q = 0.1 neither half costs FedAMM anything. On the Dirichlet partition
FedMPQ alone sits 0.011 below uncompressed FedAMM, a little beyond the seed
spread, while top 10 % alone and both are within it; with three seeds this is
a possible small cost, not a firm one. FedMPQ alone gives most of the upload
saving; adding top 10 % on top cuts only 11 % more, because each sent value
carries a 4-byte index.

Interim bigbatch (RTX 3090) copies, run while batch was down (13 to 17
October), same seeds: q = 0.1 FedMPQ only 0.833 / 0.826 / 0.826, 0.828 ±
0.004; top 10 % only 0.834 / 0.829 / 0.828, 0.830 ± 0.004. Dirichlet FedMPQ
only 0.726 / 0.724 / 0.713, 0.721 ± 0.007; top 10 % only 0.727 / 0.721 /
0.723, 0.724 ± 0.003. Per-setting means agree with batch within 0.004; single
seeds differ by up to 0.010 (more on the Dirichlet partition), the size of the
seed spread. Upload bytes are identical on both GPUs.

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
