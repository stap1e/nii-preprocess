"""NIfTI read/write helpers."""

import os

import SimpleITK as sitk


def _build_temp_output_path(final_path: str) -> str:
    if final_path.endswith(".nii.gz"):
        return final_path[:-7] + ".tmp.nii.gz"
    root, ext = os.path.splitext(final_path)
    return root + ".tmp" + ext


def remove_file_if_exists(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def safe_write_image(image: sitk.Image, final_path: str) -> None:
    """Write a NIfTI file atomically via a temporary path."""
    temp_path = _build_temp_output_path(final_path)
    remove_file_if_exists(temp_path)
    try:
        sitk.WriteImage(image, temp_path)
        os.replace(temp_path, final_path)
    except Exception:
        remove_file_if_exists(temp_path)
        raise
