# BraTS validation

Every modality and segmentation is opened with nibabel and checked for 3-D shape, finite affine/spacing, orientation, matching spatial geometry, and non-empty data. Segmentation labels are integer-valued BraTS labels `0, 1, 2, 4`; unexpected labels are errors by default. Validation reports retain all subject failures.
