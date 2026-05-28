"""Convert NIfTI volumes to HDF5 for training pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import h5py
import SimpleITK as sitk


@dataclass
class H5Converter:
    """Batch-convert NIfTI images (and optional labels) to compressed H5 files."""

    image_dir: str
    output_dir: str
    mask_dir: Optional[str] = None
    labeled: bool = True
    match_mask: Optional[Callable[[str, list[str]], Optional[str]]] = None

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _list_nii(self, directory: str) -> list[str]:
        return sorted(
            name
            for name in os.listdir(directory)
            if name.endswith(".nii") or name.endswith(".nii.gz")
        )

    def _default_mask_name(self, img_name: str, mask_names: list[str]) -> Optional[str]:
        if img_name in mask_names:
            return img_name
        stem = img_name.split("_0000")[0]
        for ext in (".nii.gz", ".nii"):
            candidate = stem + ext
            if candidate in mask_names:
                return candidate
        return None

    def _output_name(self, img_name: str) -> str:
        return img_name.split(".")[0] + ".h5"

    def iter_pairs(self) -> Iterable[tuple[str, Optional[str]]]:
        img_names = self._list_nii(self.image_dir)
        mask_names = self._list_nii(self.mask_dir) if self.mask_dir else []
        matcher = self.match_mask or self._default_mask_name

        for img_name in img_names:
            mask_name = None
            if self.labeled and self.mask_dir:
                mask_name = matcher(img_name, mask_names)
                if mask_name is None:
                    print(f"Skipping {img_name}: no matching mask found.")
                    continue
            yield img_name, mask_name

    def convert_one(self, img_name: str, mask_name: Optional[str] = None) -> str:
        img_path = os.path.join(self.image_dir, img_name)
        sitk_img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(sitk_img)

        savepath = os.path.join(self.output_dir, self._output_name(img_name))
        with h5py.File(savepath, "w") as f:
            f.create_dataset("image", data=img, compression="gzip")
            if mask_name is not None and self.mask_dir:
                mask_path = os.path.join(self.mask_dir, mask_name)
                sitk_mask = sitk.ReadImage(mask_path)
                mask = sitk.GetArrayFromImage(sitk_mask)
                f.create_dataset("label", data=mask, compression="gzip")

        print(f"Saved {img_name} -> {savepath}")
        return savepath

    def run(self) -> int:
        count = 0
        for img_name, mask_name in self.iter_pairs():
            self.convert_one(img_name, mask_name)
            count += 1
        return count


def convert_nii_to_h5(
    image_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
    labeled: bool = True,
    match_mask: Optional[Callable[[str, list[str]], Optional[str]]] = None,
) -> int:
    """Convenience wrapper around :class:`H5Converter`."""
    return H5Converter(
        image_dir=image_dir,
        output_dir=output_dir,
        mask_dir=mask_dir,
        labeled=labeled,
        match_mask=match_mask,
    ).run()


def synapse_mask_matcher(img_name: str, mask_names: list[str]) -> Optional[str]:
    """Map ``img0001.nii`` to ``label0001.nii`` for Synapse-style naming."""
    base_num = img_name.replace("img", "").replace(".nii", "").replace(".nii.gz", "")
    for candidate in (f"label{base_num}.nii", f"label{base_num}.nii.gz"):
        if candidate in mask_names:
            return candidate
    return None
