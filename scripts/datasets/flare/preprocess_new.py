import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize

from med_preprocess.io.nii_io import _build_temp_output_path, remove_file_if_exists


DEFAULT_IMAGE_DIR = r"E:/FLARE25/validation/Validation-Public-Images"
DEFAULT_MASK_DIR = ""
DEFAULT_OUTPUT_DIR = r"E:/FLARE25/test/nii_val_img_6"
DEFAULT_MASK_OUTPUT_DIR = r"E:/FLARE25/test/nii_val_mask_6"
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
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mask-output-dir", default=DEFAULT_MASK_OUTPUT_DIR)
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


def nii_case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def build_mask_lookup(mask_files: list[Path]) -> dict[str, Path]:
    lookup = {}
    for mask_path in mask_files:
        case_id = nii_case_id(mask_path)
        if case_id in lookup:
            raise RuntimeError(f"duplicate mask case id {case_id}: {lookup[case_id].name}, {mask_path.name}")
        lookup[case_id] = mask_path
    return lookup


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
    order: int,
) -> np.ndarray:
    out_shape = tuple(
        max(1, int(round(size * src / dst)))
        for size, src, dst in zip(data.shape, in_spacing, out_spacing)
    )
    if out_shape == data.shape:
        return data
    return resize(
        data,
        output_shape=out_shape,
        order=order,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )


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
    dtype: np.dtype,
) -> None:
    data = data.astype(dtype, copy=False)
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_zooms(spacing)
    image.header.set_data_dtype(dtype)
    temp_path = Path(_build_temp_output_path(str(output_path)))
    remove_file_if_exists(str(temp_path))
    try:
        nib.save(image, str(temp_path))
        os.replace(temp_path, output_path)
    except Exception:
        remove_file_if_exists(str(temp_path))
        raise


def load_canonical_data(input_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image = nib.as_closest_canonical(nib.load(str(input_path)))
    if nib.aff2axcodes(image.affine) != TARGET_AXCODES:
        raise RuntimeError(f"failed to orient {input_path.name} to RAS")
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"expected 3D image, got shape {data.shape} for {input_path.name}")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    return data, image.affine, spacing


def validate_saved(
    output_path: Path,
    spacing: tuple[float, float, float],
    dtype_name: str,
) -> None:
    saved = nib.load(str(output_path))
    if nib.aff2axcodes(saved.affine) != TARGET_AXCODES:
        raise RuntimeError(f"saved orientation is not RAS for {output_path.name}")
    zooms = tuple(round(float(value), 4) for value in saved.header.get_zooms()[:3])
    expected = tuple(round(float(value), 4) for value in spacing)
    if zooms != expected:
        raise RuntimeError(f"saved spacing {zooms} != {expected} for {output_path.name}")
    if str(saved.get_data_dtype()) != dtype_name:
        raise RuntimeError(f"saved dtype is not {dtype_name} for {output_path.name}")


def preprocess_image(
    input_path: Path,
    output_path: Path,
    out_spacing: tuple[float, float, float],
) -> None:
    data, affine, in_spacing = load_canonical_data(input_path)
    data = normalize_to_unit_range(data)
    data = resample_to_spacing(data, in_spacing, out_spacing, order=3).astype(np.float32, copy=False)
    affine = resample_affine(affine, in_spacing, out_spacing)
    save_nifti(data, affine, output_path, out_spacing, np.float32)
    validate_saved(output_path, out_spacing, "float32")


def preprocess_mask(
    input_path: Path,
    output_path: Path,
    out_spacing: tuple[float, float, float],
) -> None:
    data, affine, in_spacing = load_canonical_data(input_path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = resample_to_spacing(data, in_spacing, out_spacing, order=0)
    data = np.rint(data).astype(np.int32, copy=False)
    affine = resample_affine(affine, in_spacing, out_spacing)
    save_nifti(data, affine, output_path, out_spacing, np.int32)
    validate_saved(output_path, out_spacing, "int32")


def cleanup_temp(*paths: Path | None) -> None:
    for path in paths:
        if path is not None:
            remove_file_if_exists(_build_temp_output_path(str(path)))


def remove_outputs(*paths: Path | None) -> None:
    for path in paths:
        if path is not None:
            remove_file_if_exists(str(path))


def should_skip(
    image_output_path: Path,
    mask_output_path: Path | None,
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    image_exists = image_output_path.exists()
    if mask_output_path is None:
        return image_exists
    mask_exists = mask_output_path.exists()
    if image_exists and mask_exists:
        return True
    if image_exists != mask_exists:
        print("found partial output, regenerating current sample")
        remove_outputs(image_output_path, mask_output_path)
    return False


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if str(args.mask_dir).strip() else None
    output_dir = Path(args.output_dir)
    mask_output_dir = Path(args.mask_output_dir) if mask_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if mask_output_dir is not None:
        mask_output_dir.mkdir(parents=True, exist_ok=True)
    image_files = list_nii_files(image_dir)
    if args.limit > 0:
        image_files = image_files[:args.limit]
    mask_lookup = build_mask_lookup(list_nii_files(mask_dir)) if mask_dir is not None else {}
    mode = "labeled" if mask_dir is not None else "unlabeled"
    total = len(image_files)
    processed = 0
    skipped = 0
    print(f"mode={mode}, total={total}")
    for index, image_path in enumerate(image_files, start=1):
        image_output_path = output_dir / image_path.name
        mask_path = None
        mask_output_path = None
        if mask_output_dir is not None:
            case_id = nii_case_id(image_path)
            mask_path = mask_lookup.get(case_id)
            if mask_path is None:
                raise RuntimeError(f"missing mask for {image_path.name}, case id {case_id}")
            mask_output_path = mask_output_dir / mask_path.name
        cleanup_temp(image_output_path, mask_output_path)
        if should_skip(image_output_path, mask_output_path, args.overwrite):
            skipped += 1
            print(f"[{index}/{total}] skip existing {image_path.name}")
            continue
        if mask_path is None:
            print(f"[{index}/{total}] process unlabeled image {image_path.name}")
            preprocess_image(image_path, image_output_path, args.spacing)
        else:
            print(f"[{index}/{total}] process image {image_path.name}, mask {mask_path.name}")
            try:
                preprocess_mask(mask_path, mask_output_path, args.spacing)
                preprocess_image(image_path, image_output_path, args.spacing)
            except Exception:
                remove_outputs(image_output_path, mask_output_path)
                raise
        processed += 1
    print(f"done: mode={mode}, processed={processed}, skipped={skipped}, total={total}, output={output_dir}")
    if mask_output_dir is not None:
        print(f"mask_output={mask_output_dir}")


if __name__ == "__main__":
    main()
