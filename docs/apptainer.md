# Apptainer

Apptainer is the cluster runtime. Use no-sudo builds where allowed or a remote builder, bind user-owned data/output paths, and add `--nv` only for GPUs. `--cleanenv` avoids host leakage. Set user-owned cache paths when troubleshooting builds.
