# Cluster node health

Last checked 2026-10-13. Campaign files list these under `known_bad_nodes`,
and `scripts/monitoring/campaign.py` adds any node that fails the job-start GPU check
(`FL_BRATS_GPU_UNAVAILABLE` in a job's `.err`, exit code 75) to
`logs/monitoring/campaign/<name>/state.json`. Every later submission excludes them.

## GPU present but unusable

The GPU shows on the PCI bus, the 595.71.05 kernel driver is loaded, and
`/dev/nvidia0` exists with open permissions, but `nvidia-smi` reports
"Unable to determine the device handle for GPU0: 0000:17:00.0: Unknown Error"
and PyTorch sees no CUDA device. This usually means the GPU needs a reset or a
node reboot, which only the cluster admins can do. `dmesg` is not readable by
users, so the Xid code is unknown.

| Partition | Node | GPU |
|---|---|---|
| bigbatch | mscluster44, 45, 57, 59, 62, 65, 75, 83 (2026-09-24); 51, 76 (2026-10-13) | RTX 3090 |
| batch | mscluster129 | RTX 3060 |

mscluster50 (bigbatch) fails differently: "Failed to initialize NVML:
Driver/library version mismatch" (2026-10-13), so its driver was updated
without a reboot.

Only idle bigbatch nodes could be probed (on 2026-10-13: 50-56, 71-73, 76, 87;
52-56, 71-73 and 87 passed a CUDA matmul in the run image). Busy bigbatch nodes
are untested; the job-start check catches any bad ones.

## Unsupported GPUs

stampede nodes carry a GTX 1060 (compute capability 6.1). The image's PyTorch
has no kernels for it ("no kernel image is available for execution on the
device"), so stampede cannot run GPU work. The job-start check requires
compute capability 7.0 or higher.

## Slow I/O

mscluster124 (batch) held a client in uninterruptible I/O wait for more than
25 minutes on its first round while the same data loaded normally elsewhere.
It was not excluded until 2026-10-13, when it and mscluster125 were drained
with "Kill task failed" during the batch outage below; both are excluded now.

## Nodes that cannot start jobs

mscluster82 (bigbatch, 2026-10-13) failed every job within a second with exit
code 6 and wrote nothing to the job's `.err`, so the GPU check never ran.
Because it looked idle, Slurm sent it every new job: three training attempts
and four periodic evaluations failed there in 15 minutes. `campaign.py` now
excludes a node after three such failures (FAILED within 5 s, in the last three
hours) and does not count attempts that landed on an excluded node against the
run.

mscluster48 (bigbatch, 2026-10-15) failed the same way, but only one run's
server kept landing there, so the first rule (at least two job names) missed
it and the run went stuck after three attempts. The rule now also accepts three
failures of one job name when a job of that name (the same run's clients) ran
for over a minute on another node, which rules out a broken config.

## Slow nodes

mscluster74 (bigbatch, 2026-10-14) passed the GPU check, but a FedAMM server
on it spent 312 s per round on validation against 26 s on other bigbatch
nodes (client training was normal), which would have stretched a 3 h run to
15 h. It is excluded; the run resumed from its round 4 checkpoint elsewhere.

mscluster61 (bigbatch, 2026-10-14) ran a periodic evaluation at about 1,300 s
per checkpoint against about 460 s on other bigbatch nodes. It is excluded;
the evaluation resumed elsewhere, keeping the four checkpoints it had scored.

mscluster77 (bigbatch, 2026-10-14) took a client's round 2 training message
and never replied (25+ minutes on a step that takes about 40 s; the other
three clients replied normally). It is excluded; the run resumed from its
round 1 checkpoint elsewhere.

## Batch outage, 2026-10-13

At 07:07 that day 98 of the 100 batch nodes stopped responding (Slurm state
`down*`, "Not responding") and the other two were drained. The FedAMM comm-eff
ablation then ran interim copies on bigbatch while its batch runs waited in
the queue.

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
