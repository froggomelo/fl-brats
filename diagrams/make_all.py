"""Regenerate every diagram into diagrams/out/ (PNG for slides and email, PDF for the thesis)."""

import fig_client_sizes
import fig_combinations
import fig_modality_counts
import fig_pipeline

for module in (fig_pipeline, fig_client_sizes, fig_modality_counts, fig_combinations):
    module.main()
    print(f"drew {module.__name__}")
