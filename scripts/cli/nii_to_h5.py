#!/usr/bin/env python3
"""CLI: convert preprocessed NIfTI volumes to HDF5."""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/cli/nii_to_h5.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from med_preprocess.io.h5_converter import H5Converter, synapse_mask_matcher


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NIfTI to H5 for training.")
    parser.add_argument("--image-dir", required=True, help="Directory of image NIfTI files")
    parser.add_argument("--output-dir", required=True, help="Directory for output .h5 files")
    parser.add_argument("--mask-dir", default=None, help="Directory of label NIfTI files")
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Image-only mode (no label dataset written)",
    )
    parser.add_argument(
        "--synapse-naming",
        action="store_true",
        help="Use img#### / label#### filename pairing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    converter = H5Converter(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        labeled=not args.unlabeled and args.mask_dir is not None,
        match_mask=synapse_mask_matcher if args.synapse_naming else None,
    )
    count = converter.run()
    print(f"Converted {count} volume(s).")


if __name__ == "__main__":
    main()
