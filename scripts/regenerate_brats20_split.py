#!/usr/bin/env python3
"""
Regenerate train.txt, val.txt, and test.txt under the BraTS 2020 data root.

Uses BRATS_TRAIN_FOLDERS_20 from dataset/brats_paths.py (same value as
get_brats_folder_20() in dataset/brats.py), unless --data-root is passed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic 70%% / 10%% / rest split by sorted BraTS20_Training_* folder names."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="BraTS 2020 folder containing BraTS20_Training_* (default: BRATS_TRAIN_FOLDERS_20 in dataset/brats_paths.py)",
    )
    args = parser.parse_args()

    if args.data_root:
        root = os.path.abspath(args.data_root)
    else:
        from dataset.brats_paths import BRATS_TRAIN_FOLDERS_20

        root = os.path.abspath(BRATS_TRAIN_FOLDERS_20)

    paths = sorted(glob.glob(os.path.join(root, "BraTS20_Training_*")))
    ids = [os.path.basename(p) for p in paths if os.path.isdir(p)]
    n = len(ids)
    if n == 0:
        print(f"No BraTS20_Training_* folders under: {root}", file=sys.stderr)
        sys.exit(1)

    n_tr = int(0.70 * n)
    n_va = int(0.10 * n)
    train_ids = ids[:n_tr]
    val_ids = ids[n_tr : n_tr + n_va]
    test_ids = ids[n_tr + n_va :]

    print(f"data_root: {root}")
    print(f"N={n}  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    for name, part in (("train.txt", train_ids), ("val.txt", val_ids), ("test.txt", test_ids)):
        out_path = os.path.join(root, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
