# BraTS 2024 Brain Tumour Segmentation — Preprocessing Pipeline

Preprocessing pipeline for the **BraTS 2024** brain MRI segmentation challenge, built as part of an engineering thesis. The goal is to turn raw 3D NIfTI volumes into clean, normalised 2D axial slices ready for training a deep learning segmentation model (PyTorch — planned).

---

## Project structure

```
BraTS_thesis/
├── configs/
│   └── data_config.yaml          # data configuration (planned: paths, target size, clip range)
├── data/
│   └── patients_z_boundaries.json  # first/last brain-containing axial slice per GLI patient
├── notebooks/
│   └── eda_exploration.ipynb     # exploratory data analysis (Colab notebook)
├── src/brain_mri/
│   ├── data/
│   │   ├── nifti_loader.py       # patient loading, modality discovery, intensity normalisation
│   │   ├── preprocessing.py      # clipping, z-score, cropping/padding, slice validation
│   │   └── split.py              # patient-level train/val/test split (70/15/15, saved to JSON)
│   └── slice_extraction.py       # 2D slice extraction pipeline (5-channel image + mask)
├── pyproject.toml
└── roadmap.md                    # detailed preprocessing TODO (in Polish)
```

## Data

The project uses two BraTS 2024 training datasets, stored on Google Drive as ZIPs and unpacked to `.../engineering_thesis/brats/unpacked/`:

| Dataset | Path | Patients | Modalities available |
|---|---|---|---|
| **BraTS-GLI** (adult glioma, pre-operative) | `BraTS-GLI/training_data1_v2` | 1350 | T1, T1CE, T2, FLAIR, SEG |
| **BraTS-MEN-RT** (meningioma, radiotherapy) | `BraTS-MEN/BraTS-MEN-RT-Train-v2` | 500 | T1CE, GTV only |

- Every patient folder contains one `.nii.gz` file per modality, identified by file suffix: `t1n` (T1 native), `t1c` (T1 contrast-enhanced), `t2w` (T2 weighted), `t2f` (FLAIR), `seg` (segmentation mask).
- Volumes share a common shape of **240 × 240 × 155** (H × W × D) for GLI patients; the MEN-RT set is incomplete by design (only T1CE + a binary GTV mask, as it targets radiotherapy planning).
- Splits (`split.py`) are done **at the patient level** (no patient leaks across sets): 70 % train / 15 % validation / 15 % test with seed 42, persisted to JSON so slices can be generated after splitting.

---

## MRI theory — imaging modalities

Each GLI patient comes as four co-registered structural MRI scans plus a segmentation mask:

| Channel | Name | File suffix | What it shows | Typical use in tumour analysis |
|---|---|---|---|---|
| **T1** | T1-weighted, native | `t1n` | Anatomical structure; fat bright, water/CSF dark | Brain anatomy reference; tumour boundaries look dark relative to white matter |
| **T1CE** | T1-weighted + gadolinium contrast | `t1c` | Areas where the blood–brain barrier is broken light up | Highlights the **enhancing tumour (ET)** — the actively growing, vascularised core |
| **T2** | T2-weighted | `t2w` | Water content bright (oedema, CSF, inflammation) | Shows oedema and infiltrative changes, but CSF is also bright |
| **FLAIR** | Fluid-attenuated inversion recovery | `t2f` | T2-like, but CSF signal is **suppressed** | Best single view of **peritumoral oedema / SNFH**; also the most reliable modality for locating brain tissue (used for z-boundary scanning) |
| **SEG** | Segmentation mask | `seg` | Integer labels per voxel | Ground truth for training/evaluation |
| **GTV** (MEN-RT only) | Gross tumour volume | `gtv` | Binary RT planning contour | Meningioma delineation for radiotherapy |

**Key intuition:** T1CE highlights where contrast leaks into tumour tissue (active core), while FLAIR highlights oedema and infiltrative signal far beyond the enhancing core. Combining all four channels lets a model separate tumour subregions that look identical on a single sequence.

## Segmentation labels (GLI)

BraTS 2024 labels each voxel with one of the following classes (0 = background):

| Label | Notebook name | BraTS 2024 nomenclature | Meaning |
|---|---|---|---|
| 1 | **NETC** | Non-enhancing tumour core / "cellular tumour" (formerly NCR/NET) | Tumour cells that do **not** take up contrast |
| 2 | **SNFH** | Surrounding non-enhancing FLAIR hyperintensity (formerly oedema, ED) | Oedema + infiltrated tissue around the tumour; visible on T2/FLAIR |
| 3 | **ET** | Enhancing tumour | Active, vascularised core visible on T1CE |
| 4 | **RC** | Resection cavity | Post-surgical cavity filled with fluid/air |

These labels compose into the standard **hierarchical evaluation regions** used by the BraTS challenge:

- **WTW** — whole tumour window = labels **1 + 2 + 3**
- **TCW** — tumour core window = labels **1 + 3**
- **ETW** — enhancing tumour window = label **3**
- **RCW** — resection cavity window = label **4**

Class imbalance matters: in a 100-patient GLI random sample (seed 42) SNFH was present in **100/100** patients, RC in **89/100**, ET in **68/100**, and NETC in only **43/100** — so NETC is the rarest subregion and class imbalance must be handled at training time.

---

## Preprocessing pipeline (implemented)

The pipeline lives in `src/brain_mri/` and is installed as an editable package (`pip install -e .`).

### 1. Patient loading — `data/nifti_loader.py`

- `find_patient_files()` maps modality names to files by suffix (`t1n`, `t1c`, `t2w`, `t2f`, `seg`).
- `load_patient()` returns `(images, segmentation, brain)`:
  - **images**: `float32` array of shape `(4, 240, 240, 155)` — T1, T1CE, T2, FLAIR,
  - **segmentation**: `int16` label mask (not clipped/normalised),
  - **brain**: `uint8` union mask of voxels `> 0` across all modalities (acts as a 5th "brain channel" downstream).
- Per modality: **percentile clipping** → **z-score normalisation**.

### 2. Intensity normalisation — `data/preprocessing.py`

- `clip_percentiles()` clips brain-tissue intensities to the **[1st, 99th] percentile** (robustness against outliers/hot spots; background `<= 0` is left untouched).
- `z_score_normalise()` computes mean/std **only over non-zero (brain) voxels**; the background stays **exactly 0**, so masks and empty space remain unambiguous. EDA confirmed: brain voxels end up with mean ≈ 0 and std ≈ 1.
- Ordering matters: clip first, then normalise, so extreme outliers do not inflate the std.

### 3. Slice extraction — `slice_extraction.py`

- `extract_patient_slices(patient_dir, boundaries_path, target_size=(224, 224))`:
  1. Loads the patient (4 normalised modalities + SEG + brain mask),
  2. reads the valid axial range for that patient from `data/patients_z_boundaries.json`,
  3. for each `z`: crops/pads image, mask and brain mask to **224 × 224** (`crop_or_pad_slice()` — centre crop / zero-pad),
  4. validates the slice (`validate_slice()`: correct channel count, matching shapes, all finite),
  5. stacks channels into a **(5, 224, 224)** image — `T1, T1CE, T2, FLAIR, brain-mask` — with an `int16` **(224, 224)** mask,
  6. returns a list of `{"patient_id", "image", "mask", "z_idx"}` dicts.
- Invalid slices (all background, NaN/inf, shape mismatch) are skipped.

### 4. Z boundaries — `data/patients_z_boundaries.json`

- Generated in the EDA notebook by scanning each GLI patient's **FLAIR** volume for the first/last axial slice containing brain tissue (`volume > 0`).
- Rationale: cropping empty slices early removes ~thousands of useless slices per dataset, cuts disk/memory usage, and biases the slice set toward informative slices. Takes ~4 min for 1350 patients.

### 5. Patient-level split — `data/split.py`

- `split_patients()` / `get_or_create_splits()`: 70/15/15 train/val/test **by patient** (seed 42), persisted to `data_splits.json` so the split is stable across runs and slice generation.

---

## What the EDA notebook shows (`notebooks/eda_exploration.ipynb`)

1. **Colab setup** — clone + `pip install -e`, Drive mount.
2. **Dataset overview** — patient counts, modality completeness (all 500 MEN-RT patients are "incomplete" for the 5-modality GLI schema — expected, they ship T1CE + GTV only), shape uniformity check on a 100-patient sample.
3. **Single-patient GLI exploration** — modality montages at the best segmentation slice, FLAIR + SEG overlays, per-modality intensity statistics and histograms, per-label voxel counts.
4. **Z boundaries & slice extraction** — FLAIR-based z-boundary scan, `extract_patient_slices()` demo, best-slice visualisation, verification that normalised brain voxels are mean ≈ 0 / std ≈ 1 with background exactly 0.
5. **100-patient segmentation analysis** — tumour/cavity volume distributions, per-class presence and totals, within-tumour label percentages, smallest/largest tumour cases.
6. **MEN-RT exploration** — T1CE + GTV overlays for radiotherapy patients.

Key EDA findings:

- Raw intensities are non-normalised with a large background: ~6 M of 7.22 M voxels are exactly 0, medians are 0 → per-volume z-score on brain voxels is required.
- Label presence in the 100-patient GLI sample: **SNFH 100 %**, **RC 89 %**, **ET 68 %**, **NETC 43 %** of patients.
- MEN-RT contains no full 4-modality data — it is treated as a separate single-modality (T1CE) + GTV task.

---

## Setup

```bash
git clone https://github.com/AdaBro24/BraTS_thesis.git
cd BraTS_thesis
pip install -e .
```

Requirements: Python ≥ 3.10, `numpy`, `nibabel`, `scikit-learn`, `pyyaml`. The raw BraTS 2024 ZIPs are expected on Google Drive under `/content/drive/MyDrive/engineering_thesis/brats/` (the project is developed in Google Colab).

---

## Roadmap (short)

Done: patient loading, clipping + z-score normalisation, brain mask, crop/pad to 224×224, slice validation, z-boundary generation, patient-level splits, slice extraction.

Next up: PyTorch `Dataset`/`DataLoader`, augmentations (train only), HDF5 export of extracted slices, unit tests, then model training and evaluation (Dice score per hierarchical region: WTW / TCW / ETW).

See `roadmap.md` for the detailed checklist.