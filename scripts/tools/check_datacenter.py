#!/usr/bin/env python3
"""Comprehensive diagnostic: are NIfTI files in two folders from the same data center?

Pipeline:
  1. Scan every .nii / .nii.gz in folder A and B, extracting per-file header
     metadata and (optional) voxel intensity statistics into a folder profile.
  2. Check intra-folder consistency (same-center data is usually uniform).
  3. Cross-compare the two profiles across modality, orientation, spacing,
     physical size, and intensity dimensions, then vote on a verdict.

Only depends on nibabel + numpy (already in requirements.txt).
"""

import argparse
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# per-file profile
# ---------------------------------------------------------------------------

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def profile_file(path: str, with_intensity: bool = True) -> Optional[Dict[str, Any]]:
    """Extract metadata and (optional) intensity stats from one NIfTI file."""
    try:
        img = nib.load(path)
    except Exception as e:
        print(f"  [skip] failed to read {path}: {e}")
        return None

    header = img.header
    affine = img.affine
    zooms = tuple(np.round(header.get_zooms(), 4))
    shape = tuple(int(s) for s in img.shape[:3])
    dtype = str(img.get_data_dtype())
    axcodes = nib.aff2axcodes(affine)

    try:
        qform_code = int(header["qform_code"])
        sform_code = int(header["sform_code"])
    except Exception:
        qform_code, sform_code = -1, -1

    pixdim = tuple(np.round(header["pixdim"][1:4], 4))
    scl_slope = _safe_float(header["scl_slope"])
    scl_inter = _safe_float(header["scl_inter"])

    record: Dict[str, Any] = {
        "name": os.path.basename(path),
        "axcodes": axcodes,
        "zooms": zooms,
        "shape": shape,
        "dtype": dtype,
        "qform_code": qform_code,
        "sform_code": sform_code,
        "pixdim": pixdim,
        "scl_slope": scl_slope,
        "scl_inter": scl_inter,
        "affine": np.asarray(affine, dtype=float),
        "phys_size": tuple(round(s * z, 2) for s, z in zip(shape, zooms)),
    }

    if with_intensity:
        try:
            data = np.asarray(img.get_fdata(), dtype=np.float64)
        except Exception as e:
            print(f"  [warn] intensity stats failed {path}: {e}")
            data = None
        if data is not None:
            neg_frac = float(np.mean(data < 0))
            record.update({
                "int_min": float(np.min(data)),
                "int_max": float(np.max(data)),
                "int_mean": float(np.mean(data)),
                "int_std": float(np.std(data)),
                "int_p1": float(np.percentile(data, 1)),
                "int_p50": float(np.percentile(data, 50)),
                "int_p99": float(np.percentile(data, 99)),
                "neg_frac": neg_frac,
            })

    return record


# ---------------------------------------------------------------------------
# folder-level profile
# ---------------------------------------------------------------------------

def _list_nii(directory: str) -> List[str]:
    return sorted(
        os.path.join(directory, n)
        for n in os.listdir(directory)
        if n.endswith(".nii") or n.endswith(".nii.gz")
    )


def _num_summary(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"min": float("nan"), "median": float("nan"),
                "max": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def profile_folder(directory: str, with_intensity: bool = True) -> Dict[str, Any]:
    """Scan a folder and return an aggregated profile."""
    files = _list_nii(directory)
    print(f"\nScanning folder: {directory}")
    print(f"  found {len(files)} NIfTI file(s)")

    records: List[Dict[str, Any]] = []
    for i, fp in enumerate(files):
        rec = profile_file(fp, with_intensity=with_intensity)
        if rec is not None:
            records.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(files)} ...")
    print(f"  valid files: {len(records)}/{len(files)}")

    if not records:
        return {"directory": directory, "count": 0, "records": []}

    axcodes_c = Counter(r["axcodes"] for r in records)
    dtype_c = Counter(r["dtype"] for r in records)

    zooms_arr = np.array([r["zooms"] for r in records], dtype=float)
    shape_arr = np.array([r["shape"] for r in records], dtype=float)
    phys_arr = np.array([r["phys_size"] for r in records], dtype=float)

    profile: Dict[str, Any] = {
        "directory": directory,
        "count": len(records),
        "records": records,
        "axcodes_dist": axcodes_c,
        "dtype_dist": dtype_c,
        "zooms_summary": [_num_summary(zooms_arr[:, k].tolist()) for k in range(3)],
        "shape_summary": [_num_summary(shape_arr[:, k].tolist()) for k in range(3)],
        "phys_summary": [_num_summary(phys_arr[:, k].tolist()) for k in range(3)],
    }

    if with_intensity and "int_min" in records[0]:
        for key in ("int_min", "int_max", "int_mean", "int_std",
                    "int_p1", "int_p50", "int_p99", "neg_frac"):
            profile[key] = _num_summary([r[key] for r in records])

    return profile


# ---------------------------------------------------------------------------
# intra-folder consistency
# ---------------------------------------------------------------------------

def _consistency_label(unique_ratio: float) -> str:
    if unique_ratio <= 0.05:
        return "highly consistent"
    elif unique_ratio <= 0.20:
        return "mostly consistent"
    elif unique_ratio <= 0.50:
        return "some variation"
    return "highly variable"


def report_intra_consistency(profile: Dict[str, Any], label: str) -> None:
    """Print intra-folder consistency report for one folder."""
    print(f"\n{'=' * 60}")
    print(f"  Phase 1 - folder {label} intra-consistency")
    print(f"  path: {profile['directory']}")
    print(f"  files: {profile['count']}")
    print("=" * 60)

    if profile["count"] == 0:
        print("  (no valid files)")
        return

    n = profile["count"]

    axcodes_c: Counter = profile["axcodes_dist"]
    dominant_ax = axcodes_c.most_common(1)[0]
    ax_ratio = (len(axcodes_c) - 1) / n
    print(f"\n  [1] Orientation (axcodes)")
    for ax, cnt in axcodes_c.most_common():
        print(f"      {ax}  x{cnt}")
    print(f"      dominant: {dominant_ax[1]}/{n} "
          f"({dominant_ax[1]/n*100:.1f}%)  ->  {_consistency_label(ax_ratio)}")

    dtype_c: Counter = profile["dtype_dist"]
    print(f"\n  [2] Data type (dtype)")
    for dt, cnt in dtype_c.most_common():
        print(f"      {dt}  x{cnt}")

    print(f"\n  [3] Voxel spacing (mm)")
    for k, axis_name in enumerate(("x", "y", "z")):
        s = profile["zooms_summary"][k]
        print(f"      axis {axis_name}: min={s['min']:.4f}  median={s['median']:.4f}  "
              f"max={s['max']:.4f}  std={s['std']:.4f}")

    print(f"\n  [4] Matrix shape")
    for k, axis_name in enumerate(("x", "y", "z")):
        s = profile["shape_summary"][k]
        print(f"      axis {axis_name}: min={s['min']:.0f}  median={s['median']:.0f}  "
              f"max={s['max']:.0f}")

    print(f"\n  [5] Physical size (shape x spacing, mm)")
    for k, axis_name in enumerate(("x", "y", "z")):
        s = profile["phys_summary"][k]
        print(f"      axis {axis_name}: min={s['min']:.1f}  median={s['median']:.1f}  "
              f"max={s['max']:.1f}")

    if "int_min" in profile:
        print(f"\n  [6] Intensity statistics")
        for key, desc in (
            ("int_min", "min   "), ("int_p1", "P1    "), ("int_p50", "median"),
            ("int_p99", "P99   "), ("int_max", "max   "), ("int_mean", "mean  "),
            ("int_std", "std   "), ("neg_frac", "neg%   "),
        ):
            s = profile[key]
            if key == "neg_frac":
                print(f"      {desc}: median={s['median']:.4f}  "
                      f"min={s['min']:.4f}  max={s['max']:.4f}")
            else:
                print(f"      {desc}: median={s['median']:.2f}  "
                      f"min={s['min']:.2f}  max={s['max']:.2f}")
        med_neg = profile["neg_frac"]["median"]
        modality = "CT (has negative HU)" if med_neg > 0.02 else "MRI / non-negative (guess)"
        print(f"      -> inferred modality: {modality}")


# ---------------------------------------------------------------------------
# cross-folder comparison
# ---------------------------------------------------------------------------

def _categorical_similarity(c1: Counter, c2: Counter) -> float:
    """Jaccard similarity over key sets. 1=identical, 0=disjoint."""
    s1, s2 = set(c1.keys()), set(c2.keys())
    if not s1 and not s2:
        return 1.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def _num_overlap(s1: Dict[str, float], s2: Dict[str, float],
                 rel_tol: float = 0.15) -> Tuple[float, bool]:
    """Compare medians of two numeric distributions. Returns (rel_diff, close?)."""
    m1, m2 = s1["median"], s2["median"]
    if np.isnan(m1) or np.isnan(m2):
        return (0.0, True) if m1 == m2 else (float("inf"), False)
    if m1 == 0 and m2 == 0:
        return 0.0, True
    denom = max(abs(m1), abs(m2))
    if denom == 0:
        return (0.0, True) if m1 == m2 else (float("inf"), False)
    rel_diff = abs(m1 - m2) / denom
    return rel_diff, rel_diff <= rel_tol


def compare_folders(pa: Dict[str, Any], pb: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    """Compare two folder profiles. Returns list of (dimension, same_source?, note)."""
    verdicts: List[Tuple[str, bool, str]] = []

    if pa["count"] == 0 or pb["count"] == 0:
        verdicts.append(("validity", False, "a folder has no valid files; cannot compare"))
        return verdicts

    # 0. modality (CT vs MRI)
    if "neg_frac" in pa and "neg_frac" in pb:
        a_ct = pa["neg_frac"]["median"] > 0.02
        b_ct = pb["neg_frac"]["median"] > 0.02
        same = a_ct == b_ct
        desc = f"A={'CT' if a_ct else 'MRI/neg'}, B={'CT' if b_ct else 'MRI/neg'}"
        verdicts.append(("modality(CT/MRI)", same, desc))

    # 1. orientation
    sim = _categorical_similarity(pa["axcodes_dist"], pb["axcodes_dist"])
    a_dom = pa["axcodes_dist"].most_common(1)[0][0]
    b_dom = pb["axcodes_dist"].most_common(1)[0][0]
    same = sim >= 0.5 and a_dom == b_dom
    desc = f"dominant A={a_dom}, B={b_dom}, jaccard={sim:.2f}"
    verdicts.append(("orientation", same, desc))

    # 2. dtype
    sim_dt = _categorical_similarity(pa["dtype_dist"], pb["dtype_dist"])
    same = sim_dt >= 0.5
    desc = (f"A={list(pa['dtype_dist'])}, B={list(pb['dtype_dist'])}, "
            f"jaccard={sim_dt:.2f}")
    verdicts.append(("dtype", same, desc))

    # 3. voxel spacing
    spacing_ok = 0
    parts = []
    for k, axis_name in enumerate(("x", "y", "z")):
        rd, ok = _num_overlap(pa["zooms_summary"][k], pb["zooms_summary"][k], 0.15)
        spacing_ok += int(ok)
        parts.append(f"{axis_name}:d={rd:.1%}")
    same = spacing_ok >= 2
    desc = "median rel-diff " + ", ".join(parts) + f"  ({spacing_ok}/3 axes close)"
    verdicts.append(("spacing", same, desc))

    # 4. physical size
    phys_ok = 0
    parts = []
    for k, axis_name in enumerate(("x", "y", "z")):
        rd, ok = _num_overlap(pa["phys_summary"][k], pb["phys_summary"][k], 0.25)
        phys_ok += int(ok)
        parts.append(f"{axis_name}:d={rd:.1%}")
    same = phys_ok >= 2
    desc = "median rel-diff " + ", ".join(parts) + f"  ({phys_ok}/3 axes close)"
    verdicts.append(("physical_size", same, desc))

    # 5. intensity distribution
    if "int_min" in pa and "int_min" in pb:
        med_rd, med_ok = _num_overlap(pa["int_p50"], pb["int_p50"], 0.20)
        p1_rd, p1_ok = _num_overlap(pa["int_p1"], pb["int_p1"], 0.25)
        p99_rd, p99_ok = _num_overlap(pa["int_p99"], pb["int_p99"], 0.25)
        same = (med_ok and (p1_ok or p99_ok)) or (p1_ok and p99_ok)
        desc = (f"P1 d={p1_rd:.1%}({'close' if p1_ok else 'far'}), "
                f"median d={med_rd:.1%}({'close' if med_ok else 'far'}), "
                f"P99 d={p99_rd:.1%}({'close' if p99_ok else 'far'})")
        verdicts.append(("intensity", same, desc))

    return verdicts


def report_comparison(pa: Dict[str, Any], pb: Dict[str, Any]) -> None:
    """Print cross-folder comparison report and final verdict."""
    print(f"\n{'=' * 60}")
    print("  Phase 2 - A vs B cross-center comparison")
    print("=" * 60)

    verdicts = compare_folders(pa, pb)

    if len(verdicts) == 1 and verdicts[0][0] == "validity":
        print(f"\n  {verdicts[0][2]}")
        return

    print(f"\n  {'dimension':<20}{'verdict':<10}{'note'}")
    print("  " + "-" * 56)
    for name, same, desc in verdicts:
        tag = "SAME" if same else "DIFF"
        print(f"  {name:<20}{tag:<10}{desc}")

    pro = sum(1 for _, same, _ in verdicts if same)
    total = len(verdicts)
    ratio = pro / total if total else 0.0

    print(f"\n  vote: {pro}/{total} dimensions support same-source (ratio {ratio:.0%})")
    if ratio >= 0.75:
        conclusion = "very likely from the SAME data center"
    elif ratio >= 0.50:
        conclusion = "possibly from the same data center (manual review advised)"
    elif ratio >= 0.25:
        conclusion = "possibly from DIFFERENT data centers"
    else:
        conclusion = "very likely from DIFFERENT data centers"

    print(f"\n{'=' * 60}")
    print(f"  CONCLUSION: {conclusion}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether NIfTI files in two folders come from the same data center."
    )
    parser.add_argument("--dir-a", required=True, help="First folder of .nii/.nii.gz files")
    parser.add_argument("--dir-b", required=True, help="Second folder of .nii/.nii.gz files")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Header-only mode: skip voxel intensity stats (much faster, less signal)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with_intensity = not args.quick

    print("=" * 60)
    print("  Data Center Consistency Diagnostic")
    print(f"  intensity stats: {'ON' if with_intensity else 'OFF (--quick)'}")
    print("=" * 60)

    pa = profile_folder(args.dir_a, with_intensity=with_intensity)
    pb = profile_folder(args.dir_b, with_intensity=with_intensity)

    report_intra_consistency(pa, "A")
    report_intra_consistency(pb, "B")
    report_comparison(pa, pb)


if __name__ == "__main__":
    main()
