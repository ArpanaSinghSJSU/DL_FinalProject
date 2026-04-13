# Implementation strategy: BraTS 2020 deterministic split + loader

Reference for how the **70 / 10 / ~20** split and **`dataset/brats.py`** behavior are implemented in this repo.

---

## 1. Deterministic 70 / 10 / ~20 split (sorted folder names)

Under the BraTS 2020 data root, split files list **one subject ID per line** (folder basename).

| File        | Count | Range (inclusive)   | Approx. share |
|------------|-------|---------------------|---------------|
| `train.txt` | 258   | `BraTS20_Training_001` … `_258` | ≈ 69.9% |
| `val.txt`   | 36    | `_259` … `_294`                 | ≈ 9.8%  |
| `test.txt`  | 75    | `_295` … `_369`                 | ≈ 20.3% |

**Rule:** With **N = 369** cases, use `n_tr = int(0.70 * N)` and `n_va = int(0.10 * N)`; assign the remainder to test so **all 369 cases appear exactly once** across the three files.

**Data root (split files live next to `BraTS20_Training_*` folders):**

`/Users/arpanasingh/Desktop/SJSU/Spring2026/DL/Prj/m3ae/dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/`

---

## 2. Loader change in `dataset/brats.py`

### Previous behavior

`get_datasets_brats20_rf` **concatenated** `train.txt` and `val.txt` for training and used **`test.txt` for validation**. That is **not** a standard train / val / **held-out test** split (it behaves like train∪val vs “test-as-val”).

### Current behavior

- **Training** uses **`train.txt` only**.
- **Validation** uses **`val.txt` only**.
- **`test.txt`** is the **held-out test** list. Those subjects are **not** loaded by `get_datasets_brats20_rf` for train or validation; use a **separate final-eval script or dataloader** if you need test-set metrics in code.

**Helper added:** `_brats20_paths_for_id_list` (maps ID strings to full paths in a stable way).

### `pretrain.py` note

With `--dataset brats20`, **`pretrain.py` still only builds train and val loaders**. The **~20%** subject IDs in **`test.txt`** are for **reporting reproducibility** or **later evaluation code**, not for the current pretraining loop.

### Regenerating the split

If you change `BRATS_TRAIN_FOLDERS_20` or refresh the dataset:

1. Collect all `BraTS20_Training_*` folder **basenames** and **sort** them.
2. Slice: `[0 : n_tr)`, `[n_tr : n_tr + n_va)`, `[n_tr + n_va :]` into `train.txt`, `val.txt`, and `test.txt` respectively, with `n_tr = int(0.70 * N)` and `n_va = int(0.10 * N)`.

Or use the checked-in script (from the `m3ae` repo root):

```bash
python scripts/regenerate_brats20_split.py
```

Optional override without editing `brats.py`:

```bash
python scripts/regenerate_brats20_split.py --data-root /path/to/MICCAI_BraTS2020_TrainingData
```

**Script:** `scripts/regenerate_brats20_split.py`

#### What it would do

- Read **`BRATS_TRAIN_FOLDERS_20`** from **`dataset/brats_paths.py`** (same path **`get_brats_folder_20()`** in `dataset/brats.py` uses), unless you pass **`--data-root`**.
- Find all **`BraTS20_Training_*`** folders, **sort** their names, apply the same **70 / 10 / rest** rule, and **overwrite** the three **`.txt`** files next to those folders.

#### Why that’s useful

- You don’t have to remember the slice math or copy-paste a one-off command.
- If you **move the dataset** or get a **different N**, one command regenerates a **consistent** split.
- The **logic lives in the repo**, so it’s obvious how the lists were produced.

#### What it is not

- It’s **not required for training**; your current `.txt` files are already valid if they match this rule.
- It’s optional **quality of life** and **reproducibility**, not part of the model itself.

---

*See also `SPLIT_STRATEGIES_BRATS.md` for broader split options and upstream vs local loader semantics.*
