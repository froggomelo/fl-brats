# Cluster node health

Last checked 2026-09-24. Campaign files list these under `known_bad_nodes`,
and `scripts/campaign.py` adds any node that fails the job-start GPU check
(`FL_BRATS_GPU_UNAVAILABLE` in a job's `.err`, exit code 75) to
`logs/campaign/<name>/state.json`. Every later submission excludes them.

## GPU present but unusable

The GPU shows on the PCI bus, the 595.71.05 kernel driver is loaded, and
`/dev/nvidia0` exists with open permissions, but `nvidia-smi` reports
"Unable to determine the device handle for GPU0: 0000:17:00.0: Unknown Error"
and PyTorch sees no CUDA device. This usually means the GPU needs a reset or a
node reboot, which only the cluster admins can do. `dmesg` is not readable by
users, so the Xid code is unknown.

| Partition | Node | GPU |
|---|---|---|
| bigbatch | mscluster44, 45, 57, 59, 62, 65, 75, 83 | RTX 3090 |
| batch | mscluster129 | RTX 3060 |

Only idle bigbatch nodes could be probed. Busy bigbatch nodes are untested;
the job-start check catches any bad ones.

## Unsupported GPUs

stampede nodes carry a GTX 1060 (compute capability 6.1). The image's PyTorch
has no kernels for it ("no kernel image is available for execution on the
device"), so stampede cannot run GPU work. The job-start check requires
compute capability 7.0 or higher.

## Slow I/O

mscluster124 (batch) held a client in uninterruptible I/O wait for more than
25 minutes on its first round while the same data loaded normally elsewhere.
It is not excluded. The campaign's stall detection resubmits runs that stop
making progress.

## Preprocessing cache

Batch nodes have a local 500 GB SSD; `/var/tmp` lives on it and Apptainer
binds it. Experiments cache preprocessed volumes there
(`data.cache_location: node`, `/var/tmp/$USER/fl-brats-cache`). An entry is
about 75 MB against 10 MB for the `.nii.gz` source, so a cache on NFS
(`cache/persistent`) moves the bottleneck to network I/O. Remove old node
caches with `rm -rf /var/tmp/$USER/fl-brats-cache` in a job on that node.

## Partition facts

| Partition | Node | Per-user limits | Wall time |
|---|---|---|---|
| batch | 6 CPUs, 30 GB, RTX 3060 12 GB | 13 running, 100 submitted jobs | 24 h |
| bigbatch | 14 CPUs, 124 GB, RTX 3090 24 GB | 6 running, 48 submitted jobs | 72 h |
| stampede | 16 CPUs, 30 GB, GTX 1060 6 GB (unsupported) | 2 running jobs | |

GRES requests are rejected on both partitions; the GPU comes with the node.
A federated run needs five jobs (server and four clients), so bigbatch fits one
run at a time and batch fits two.
