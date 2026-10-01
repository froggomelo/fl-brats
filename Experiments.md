# Experiments

## Centralized
K = 1
Rounds = 100
B = 1

3 runs

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

## Combined FedAMM + Comm_eff
Alpha = from above
p = from above
b {2,4,8}
3 runs

Rerun with p = 0.1, as above:
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
