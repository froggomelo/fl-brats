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

## Training Configuration Sweeps

### Client Sweeps
K = 3 

1 run

K = 5

1 run

K = 8

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

## Combined FedAMM + Comm_eff
Alpha = from above
p = from above
b {2,4,8}
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
