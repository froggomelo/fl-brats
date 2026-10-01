# Fixed quantity skew

Only training IDs are partitioned. The default client fractions are 0.50, 0.25, 0.15, and 0.10; floor rounding is used and the final client receives the remainder. Client JSON files contain IDs and metadata, never copied images. `verify-manifests` checks hashes, coverage, duplicates, and leakage.

Each client subject includes its dataset-index row and relative modality paths.
Use `verify-manifests --check-nifti` to additionally open every referenced
NIfTI file.

## Client-count sweeps

`create_partition` takes any client fractions. `data.partition_remainder: largest`
floors every client and gives the rounding remainder to the largest client
instead of the final one.

Two client-count sweeps exist. The first (`client_sweeps/k{3,5,8}.yaml`) halves
each client's share, so at K = 8 the smallest clients hold 13, 6 and 8 of the
875 training patients: extreme imbalance. The second
(`client_sweeps/k{3,5,8}_lorenz.yaml`) keeps the shape of the K = 4 split with a
piecewise-linear Lorenz rule. Sort clients largest first; the K = 4 shares
50/25/15/10 give the cumulative points (fraction of clients, fraction of
patients) (0, 0), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90) and (1, 1). Client i
(0-based) of K gets F((i + 1) / K) - F(i / K), where F interpolates those points
linearly. Sizes are floored from the training split and the remainder goes to
the largest client:

| K | Shares (%) | Patients |
|---|---|---|
| 3 | 58.3 / 26.7 / 15.0 | 511 / 233 / 131 |
| 4 | 50 / 25 / 15 / 10 (the existing uneven split) | 437 / 218 / 131 / 89 |
| 5 | 40 / 25 / 16 / 11 / 8 | 351 / 218 / 140 / 96 / 70 |
| 8 | 25 / 25 / 12.5 / 12.5 / 7.5 / 7.5 / 5 / 5 | 223 / 218 / 109 / 109 / 65 / 65 / 43 / 43 |

The K = 4 point is the existing uneven partition and its runs.
