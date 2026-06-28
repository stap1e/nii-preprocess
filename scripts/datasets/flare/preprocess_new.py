import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize

from med_preprocess.io.nii_io import _build_temp_output_path, remove_file_if_exists


DEFAULT_IMAGE_DIR = r"E:/FLARE25/validation/Validation-Public-Images"
DEFAULT_OUTPUT_DIR = r"E:/FLARE25/test/nii_val_img_6"
DEFAULT_SPACING = (1.254798173904419, 1.254798173904419, 2.5)
CLIP_RANGE = (-40.0, 325.0)
TARGET_AXCODES = ("R", "A", "S")


def parse_spacing(value: str) -> tuple[float, float, float]:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("spacing must be formatted as x,y,z")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spacing", type=parse_spacing, default=DEFAULT_SPACING)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def list_nii_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.name.endswith(".nii") or path.name.endswith(".nii.gz")
    )


def normalize_to_unit_range(data: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(
        data,
        nan=CLIP_RANGE[0],
        posinf=CLIP_RANGE[1],
        neginf=CLIP_RANGE[0],
    )
    data = np.clip(data, CLIP_RANGE[0], CLIP_RANGE[1]).astype(np.float32, copy=False)
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std > 0:
        data = (data - mean) / std
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    if max_value > min_value:
        data = (data - min_value) / (max_value - min_value)
    else:
        data = np.zeros_like(data, dtype=np.float32)
    return data.astype(np.float32, copy=False)


def resample_to_spacing(
    data: np.ndarray,
    in_spacing: tuple[float, float, float],
    out_spacing: tuple[float, float, float],
) -> np.ndarray:
    out_shape = tuple(
        max(1, int(round(size * src / dst)))
        for size, src, dst in zip(data.shape, in_spacing, out_spacing)
    )
    if out_shape == data.shape:
        return data.astype(np.float32, copy=False)
    return resize(
        data,
        output_shape=out_shape,
        order=3,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32, copy=False)


def resample_affine(
    affine: np.ndarray,
    in_spacing: tuple[float, float, float],
    out_spacing: tuple[float, float, float],
) -> np.ndarray:
    in_spacing_arr = np.asarray(in_spacing, dtype=np.float64)
    out_spacing_arr = np.asarray(out_spacing, dtype=np.float64)
    direction = affine[:3, :3] @ np.diag(1.0 / in_spacing_arr)
    out_affine = affine.copy()
    out_affine[:3, :3] = direction @ np.diag(out_spacing_arr)
    return out_affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    spacing: tuple[float, float, float],
) -> None:
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_zooms(spacing)
    image.header.set_data_dtype(np.float32)
    temp_path = Path(_build_temp_output_path(str(output_path)))
    remove_file_if_exists(str(temp_path))
    try:
        nib.save(image, str(temp_path))
        os.replace(temp_path, output_path)
    except Exception:
        remove_file_if_exists(str(temp_path))
        raise


def preprocess_one(
    input_path: Path,
    output_path: Path,
    out_spacing: tuple[float, float, float],
) -> None:
    image = nib.as_closest_canonical(nib.load(str(input_path)))
    if nib.aff2axcodes(image.affine) != TARGET_AXCODES:
        raise RuntimeError(f"failed to orient {input_path.name} to RAS")
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"expected 3D image, got shape {data.shape} for {input_path.name}")
    in_spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    data = normalize_to_unit_range(data)
    data = resample_to_spacing(data, in_spacing, out_spacing)
    affine = resample_affine(image.affine, in_spacing, out_spacing)
    save_nifti(data, affine, output_path, out_spacing)
    saved = nib.load(str(output_path))
    if nib.aff2axcodes(saved.affine) != TARGET_AXCODES:
        raise RuntimeError(f"saved orientation is not RAS for {output_path.name}")
    zooms = tuple(round(float(value), 4) for value in saved.header.get_zooms()[:3])
    expected = tuple(round(float(value), 4) for value in out_spacing)
    if zooms != expected:
        raise RuntimeError(f"saved spacing {zooms} != {expected} for {output_path.name}")
    if str(saved.get_data_dtype()) != "float32":
        raise RuntimeError(f"saved dtype is not float32 for {output_path.name}")


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_nii_files(image_dir)
    if args.limit > 0:
        files = files[:args.limit]
    total = len(files)
    processed = 0
    skipped = 0
    for index, input_path in enumerate(files, start=1):
        output_path = output_dir / input_path.name
        remove_file_if_exists(_build_temp_output_path(str(output_path)))
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{total}] skip existing {input_path.name}")
            continue
        print(f"[{index}/{total}] process {input_path.name}")
        preprocess_one(input_path, output_path, args.spacing)
        processed += 1
    print(f"done: processed={processed}, skipped={skipped}, total={total}, output={output_dir}")


if __name__ == "__main__":
    main()
