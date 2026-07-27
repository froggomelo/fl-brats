"""NIfTI resampling helpers; segmentation always uses nearest neighbour."""

from typing import Any

from nibabel.processing import resample_to_output


def resample_image(
    image: Any, spacing: tuple[float, float, float], segmentation: bool = False
) -> Any:
    return resample_to_output(image, voxel_sizes=spacing, order=0 if segmentation else 1)  # type: ignore[no-untyped-call]
