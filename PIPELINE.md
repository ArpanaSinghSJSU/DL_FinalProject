# M3AE end-to-end pipeline (code map)

Here’s an **end-to-end map** of the M3AE codebase so you can read it on CPU and later run the heavy parts on Colab. This is about **what lives where**, not about installing CUDA on your laptop.

---

## Big picture: two stages

| Stage | Script | Model mode | What it learns |
|--------|--------|------------|----------------|
| **1. Pretrain** | `pretrain.py` | `Unet_missing(..., pre_train=True)` | Fill in **masked voxels / modalities** so the network reconstructs the multimodal patch (representation learning). |
| **2. Finetune (segment)** | `train_18.py` | `Unet_missing(..., pre_train=False)` | Same backbone; now optimized with **Dice-style loss** on **ET / TC / WT** labels, with **simulated missing modalities** at training time. |

Checkpoints from stage 1 are loaded in stage 2 (with some keys deleted / `strict=False` in `train_18.py`).

---

## 1. Configuration and data roots

| Piece | Role |
|--------|------|
| `dataset/brats_paths.py` | **BraTS folder paths** only (no ML imports). Your BraTS2020 root + split `.txt` location. |
| `dataset/brats.py` | **Paths → file lists → `Brats` dataset**; **`get_datasets_brats20_rf`** reads `train.txt` / `val.txt` (your deterministic split). |

**Pipeline:** `brats_paths` → `get_datasets_brats20_rf` → list of `Path`s per subject → `Brats(patients_dir, ...)`.

---

## 2. Loading volumes and building labels (`Brats` in `brats.py`)

| Logic | Where |
|--------|--------|
| Find **T1, T1ce, T2, FLAIR, seg** (`.nii` or `.nii.gz`) | `Brats.__init__` |
| Read NIfTI with SimpleITK | `Brats.load_nii` |
| **Normalize** (per-volume min–max or z-score) | `__getitem__` via `irm_min_max_preprocess` / `zscore_normalise` |
| **BraTS label → 3 binary channels** | ET (label 4), TC (4∨1), WT (TC∨2) in `__getitem__` |
| **Train:** tight bbox → **pad/crop to `patch_shape³`** (e.g. 128³) | `pad_or_crop_image`, `training=True` |
| **Val/test:** crop to brain bbox, no fixed patch in the same way | `training=False` branch |
| Batch dict | `image`, `label`, `crop_indexes`, `patient_id`, etc. |

So: **all multimodal geometry and preprocessing** for the U-Net lives here.

---

## 3. Stage 1 — Pretraining (`pretrain.py`)

| Piece | Role |
|--------|------|
| Dataset | `get_datasets_brats20_rf` (or BraTS18 path via `get_datasets_train_rf_forpretrain`). |
| Model | `Unet_missing(..., pre_train=True, mask_ratio=..., mdp=...)` from `model/Unet.py`. |
| **Masking** | In `Unet_missing.forward`, branch `pre_train and self.training`: `MaskEmbeeding2` builds mask; inputs are **`x * mask + limage * (1-mask)`** using **`limage`** and **`raw_input`** (learnable / structured over the patch). |
| **Loss** | Main: **MSE** between model output and **full input** `inputs`; extra: **L2 on `limage`** around its spatial mean (`pretrain.py` ~179–183). |
| **Labels** | Not used for the segmentation objective here; still in the batch from `Brats`. |
| **Checkpoints** | `utils.save_checkpoint` → **`runs/<exp_name>/model_1/`** (every `--val` epochs). |

Conceptually: **masked autoencoding / inpainting** on **3D multimodal patches** so the U-Net learns features that survive missing voxels/modalities.

---

## 4. The network core (`model/Unet.py` + `model/models.py`)

| Piece | Role |
|--------|------|
| `UNet3D_g` (inside `Unet_missing`) | **3D encoder–decoder**; returns `uout` and auxiliary **`style` / `content`** (used in finetuning). |
| `Unet_missing.limage` | **Learnable “completion” tensor** (placeholder for missing regions/modalities), aligned with patch locations in forward. |
| `Unet_missing.raw_input` | **`proj(...)`** on a tensor of ones — **positional / token layout** for masking (`MaskEmbeeding2`). |
| Forward | After masking, **`self.unet(x)`** runs the segmentation-style head; in pretrain the target supervision is on **reconstructing inputs**, not Dice. |

So: **masking policy** is in **`Unet_missing.forward`** and **`MaskEmbeeding2`**; **conv geometry** is in the U-Net blocks.

---

## 5. Stage 2 — Fine-tuning (`train_18.py`)

| Piece | Role |
|--------|------|
| **Weights** | `torch.load(args.checkpoint)` then **`load_state_dict(..., strict=False)`** after deleting some `up1conv` keys — loads **pretrained U-Net + `limage`-related state** (you must set a real checkpoint path). |
| Dataset (as shipped) | **`get_datasets_train_rf_withvalid`** → **BraTS 2018** paths — **not** your BraTS2020 loader unless you wire it. |
| **Input masking** | **`MaskEmbeeding1`** in the training loop: **modality dropout**, using **frozen `limage` / `raw_input` from pretrain** to fill missing modalities. |
| **Loss** | **`EDiceLoss`** on predictions vs **3-class label**; optional **deep supervision**, **KL** between branches, etc. (later lines in `train_18.py`). |
| **Metrics** | Per **mask pattern** (which modalities present) via `mask_codes` and `mean_results`. |

So: **pretrain** teaches **how to complete**; **finetune** uses that completion **explicitly** when modalities are dropped and trains **D**ice on tumor regions.

---

## 6. Losses (`loss/`)

| File | Role |
|------|------|
| `loss/dice.py` | **`EDiceLoss`** (train), **`EDiceLoss_Val`** (validation / metrics). |
| `loss/__init__.py` | Re-exports Dice loss for imports. |

Pretrain MSE is **inline in `pretrain.py`**, not in `loss/`.

---

## 7. Utilities

| File | Typical use |
|------|-------------|
| `utils.py` | Checkpoints, meters, **dice_metric**, inference helpers used by training/validation. |
| `configs/*.yaml` | Model/yacs-style config if you extend config-driven runs. |

---

## Testing

2. Your local “test” split (test.txt) — still under the same 369 training cases

You created a held-out list of 75 subjects in test.txt (from the same MICCAI 2020 training data). That is a custom test split for your experiment, not the challenge’s hidden test set.

How it is used in this repo right now

get_datasets_brats20_rf only loads train.txt (train) and val.txt (validation). It does not load test.txt.

So nothing in M3AE currently runs training or validation on those 75 cases unless you add an evaluation script (or extend the loader) that reads test.txt and runs inference + Dice against _seg.

So: the testing dataset for your project is “defined” by test.txt; it is “used” only when you add code to use it.

Short summary

| What | Where | Used by M3AE as-is? |
|------|--------|----------------------|
| Challenge test/val (no/limited local labels) | Separate downloads / portal | No in your current BraTS2020 pretrain path |
| Your test.txt holdout (75 cases, have _seg) | Same tree as train; list in test.txt | Not loaded — you add eval when ready |

If you want, we can add a small eval script that loads test.txt, runs the model, and reports Dice for fair “local test set” numbers.

---

## Reading order on CPU (no GPU needed)

1. `dataset/brats_paths.py` → `dataset/brats.py` (`Brats`, `get_datasets_brats20_rf`).  
2. `pretrain.py` (data load → loop → loss).  
3. `model/Unet.py` → `Unet_missing.forward` + `MaskEmbeeding2` usage.  
4. `train_18.py` (checkpoint load → masking with `MaskEmbeeding1` → `EDiceLoss`).  
5. `loss/dice.py` for how ET/TC/WT are scored.

---

## CPU vs Colab

- **Understanding / mapping:** your laptop is enough: follow imports and the tables above.  
- **Running training:** scripts call **`.cuda()`**; on CPU you’d have to branch to `.cpu()` / `device` everywhere, which is tedious; **Colab (or any GPU box)** is the practical place to **execute** long runs.  
- **Optional sanity check on CPU:** a **single** forward with tiny tensors after temporarily forcing CPU is possible for debugging, but not required for “understanding the pipeline.”

If you want a **single diagram** (e.g. data → mask → U-Net → loss) as a figure for notes, say whether you prefer **markdown / Mermaid** or **bullet-only** and I’ll match that style.
