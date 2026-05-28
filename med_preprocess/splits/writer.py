"""Write train/val/test list files from H5 or NIfTI directories."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set


@dataclass
class SplitWriter:
    """Assign samples to split text files based on numeric IDs."""

    data_dir: str
    output_files: Dict[str, str]
    id_from_name: Callable[[str], int]
    groups: Dict[str, Set[int]]
    extension: str = "h5"
    seed: int = 2025
    unlabeled_output: Optional[str] = None

    def __post_init__(self) -> None:
        random.seed(self.seed)
        for path in self.output_files.values():
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if os.path.exists(path):
                os.remove(path)
        if self.unlabeled_output and os.path.exists(self.unlabeled_output):
            os.remove(self.unlabeled_output)

    def _stem(self, name: str) -> str:
        if name.endswith(f".{self.extension}"):
            return name[: -(len(self.extension) + 1)]
        return os.path.splitext(name)[0]

    def _split_for_id(self, sample_id: int) -> Optional[str]:
        for split_name, ids in self.groups.items():
            if sample_id in ids:
                return split_name
        return None

    def run(self) -> None:
        for name in sorted(os.listdir(self.data_dir)):
            if not name.endswith(f".{self.extension}"):
                continue
            stem = self._stem(name)
            if self.groups:
                sample_id = self.id_from_name(name)
                split = self._split_for_id(sample_id)
                if split is None:
                    raise ValueError(f"No split group for id {sample_id} ({name})")
                out_path = self.output_files[split]
            else:
                if not self.unlabeled_output:
                    raise ValueError("unlabeled_output required when groups is empty")
                out_path = self.unlabeled_output

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(stem + "\n")


def write_split_lists(
    data_dir: str,
    output_files: Dict[str, str],
    groups: Dict[str, Set[int]],
    id_from_name: Callable[[str], int],
    extension: str = "h5",
    seed: int = 2025,
    unlabeled_output: Optional[str] = None,
) -> None:
    SplitWriter(
        data_dir=data_dir,
        output_files=output_files,
        id_from_name=id_from_name,
        groups=groups,
        extension=extension,
        seed=seed,
        unlabeled_output=unlabeled_output,
    ).run()


@dataclass
class SplitPlan:
    """Describe random splits before writing list files."""

    ids: List[int]
    sizes: Dict[str, int]
    seed: int = 2025

    groups: Dict[str, Set[int]] = field(init=False)

    def __post_init__(self) -> None:
        random.seed(self.seed)
        remaining = list(self.ids)
        self.groups = {}
        for name, size in self.sizes.items():
            picked = random.sample(remaining, size)
            self.groups[name] = set(picked)
            remaining = [i for i in remaining if i not in picked]
        if remaining:
            self.groups["_remainder"] = set(remaining)
