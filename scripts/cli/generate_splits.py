#!/usr/bin/env python3
"""CLI: build train/val/test list files from a directory of samples."""

import argparse
import random
import sys
from pathlib import Path
from typing import Set

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from med_preprocess.splits.writer import SplitPlan, write_split_lists


def parse_args():
    parser = argparse.ArgumentParser(description="Generate split list text files.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--extension", default="h5", choices=["h5", "nii", "nii.gz"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--id-regex",
        default=r"_(\d+)",
        help="Regex with one capture group for numeric ID in filenames",
    )
    parser.add_argument(
        "--split",
        action="append",
        metavar="NAME=COUNT",
        help='e.g. train_l=160 val=20 (repeat for each split)',
    )
    parser.add_argument(
        "--output",
        action="append",
        metavar="SPLIT=PATH",
        help='Map split name to output txt path, e.g. train_l=/data/train_l.txt',
    )
    return parser.parse_args()


def _parse_kv(items, label):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f'{label} must be NAME=VALUE, got "{item}"')
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _id_from_name_factory(regex: str):
    import re

    pattern = re.compile(regex)

    def _id_from_name(name: str) -> int:
        match = pattern.search(name)
        if not match:
            raise ValueError(f"Could not parse ID from {name} with pattern {regex}")
        return int(match.group(1))

    return _id_from_name


def main():
    args = parse_args()
    split_sizes = {k: int(v) for k, v in _parse_kv(args.split, "--split").items()}
    output_files = _parse_kv(args.output, "--output")

    id_fn = _id_from_name_factory(args.id_regex)
    names = [
        n
        for n in Path(args.data_dir).iterdir()
        if n.name.endswith(f".{args.extension}")
    ]
    ids = sorted({id_fn(n.name) for n in names})
    plan = SplitPlan(ids=ids, sizes=split_sizes, seed=args.seed)

    groups = {k: v for k, v in plan.groups.items() if k in output_files}

    write_split_lists(
        data_dir=args.data_dir,
        output_files=output_files,
        groups=groups,
        id_from_name=id_fn,
        extension=args.extension,
        seed=args.seed,
    )
    random.seed(args.seed)
    print("Wrote split files:", output_files)
    for name, group in groups.items():
        print(f"  {name}: {len(group)} cases")


if __name__ == "__main__":
    main()
