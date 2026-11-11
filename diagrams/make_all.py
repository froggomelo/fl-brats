"""Regenerate every diagram into diagrams/out/ (PNG for slides and email, PDF for the thesis)."""

import fig_client_sizes
import fig_combinations
import fig_compression
import fig_fedamm
import fig_modality_counts
import fig_partitions
import fig_pipeline
import fig_results
import fig_system

for module in (fig_pipeline, fig_client_sizes, fig_modality_counts, fig_combinations, fig_system, fig_partitions,
               fig_fedamm, fig_compression, fig_results):
    module.main()
    print(f"drew {module.__name__}")
