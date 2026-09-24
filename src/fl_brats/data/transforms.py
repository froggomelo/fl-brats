"""Shared MONAI preprocessing for both baseline modes."""

from typing import Any

from monai.transforms import (  # type: ignore[attr-defined]
    Compose,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandSpatialCropSamplesd,
    Spacingd,
    SpatialPadd,
)

from .labels import brats_to_regions


class BraTSRegionsd(MapTransform):
    """Convert the label to WT/TC/ET channels.

    A MONAI Transform subclass so PersistentDataset keeps caching past it: the
    cache stops at the first transform that is random or not a MONAI Transform.
    """

    def __init__(self, key: str = "label") -> None:
        super().__init__(keys=[key])
        self.key = key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data[self.key] = brats_to_regions(data[self.key].squeeze(0))
        return data


def build_transforms(config: dict[str, Any], training: bool) -> Compose:
    keys = ["image", "label"]
    transforms: list[Any] = [
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Orientationd(keys, axcodes=config.get("orientation", "RAS")),
        Spacingd(keys, pixdim=config.get("spacing", [1.0, 1.0, 1.0]), mode=("bilinear", "nearest")),
        CropForegroundd(keys, source_key="image"),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        BraTSRegionsd(),
        SpatialPadd(keys, spatial_size=config.get("patch_size", [96, 96, 96])),
        DivisiblePadd(keys, k=config.get("divisible_by", 16)),
    ]
    if training:
        # Slice models (SAM) train on whole axial slices sampled from the full
        # volume, so they skip the 3D patch crop (data.train_crop: false).
        if config.get("train_crop", True):
            transforms.append(
                RandSpatialCropSamplesd(
                    keys,
                    roi_size=config.get("patch_size", [96, 96, 96]),
                    num_samples=config.get("num_samples", 1),
                    random_size=False,
                )
            )
        transforms.append(RandFlipd(keys, prob=config.get("flip_probability", 0.5), spatial_axis=0))
    transforms += [EnsureTyped(keys)]
    return Compose(transforms)
