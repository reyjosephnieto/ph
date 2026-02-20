# TDA Pipeline

Retinal fundus classification pipeline using cubical persistence (H0/H1/HS) and Hu moments, with perturbation stress tests and plotting.

## Requirements
- Python 3.9+
- Packages: numpy, scipy, scikit-learn, opencv-python, gudhi, matplotlib

Install:
```bash
pip install numpy scipy scikit-learn opencv-python gudhi matplotlib
```

## Dataset Setup
1. Download the FIVES dataset and extract it into the workspace root.
2. Folder name must be:
   `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/`

## One-Time Ingest (Staging)
Run once to build staged tensors:
```bash
python 0_ingest.py
```
Creates:
- `data/clean_cohort.npz` (images, labels, indices)
- `data/clean_images.npy`, `data/clean_labels.npy`
- `data/raw_images.npy` (raw green channel, no CLAHE, no resize; stacked or object array)
- `data/raw_paths.npy` (string paths aligned with stacked order)
- `data/train_indices.npy`, `data/test_indices.npy`

## Run the Pipeline
From the workspace root:
```bash
python 7_orchestrate.py
```

### Orchestrator Order
1) `1_precompute.py` builds cached persistence/geometry features for each perturbation.
2) `2_audit.py` audits feature statistics on standard train.
3) `3_signal.py` measures signal via 5-fold CV on standard train.
4) `4_generalise.py` evaluates train/test generalization on standard H0+H1.
5) `5_ablate.py` performs k-fold stress tests across perturbations.
6) `6_plot.py` renders grouped panels, per-protocol plots, and orthogonality.

## Shared Configuration (fives_shared.py)
Protocol groups:
- Mechanical: rotation, resolution, blur
- Radiometric: drift, gamma, contrast
- Failure: gau_noise, spepper_noise, poi_noise, bit_depth

Perturbation ranges (current defaults):
- gamma: 0.1 → 3.0 (step 0.1)
- contrast: 0.5 → 3.0 (step 0.1)
- drift: -150 → 150 (step 10)
- rotation: -10 → 10 (step 1)
- blur: 0.0 → 1.5 (step 0.15)
- gau_noise: 0.0 → 0.20 (step 0.02)
- poi_noise: 0.0 → 0.20 (step 0.02)
- spepper_noise: 0.0000 → 0.0250 (step 0.0025)
- bit_depth: 2–8
- resolution: 64–2048

Reproducibility:
- `seed_everything()` is used in steps 2+.
- Stochastic perturbations are seeded via `make_rng()`.

## Feature Definitions
- Per-diagram stats: `[top-5 persistence sum, total persistence]`.
- H0 = H0 sublevel, H1 = H1 sublevel, HS = H0 superlevel (computed on the inverted image). Each is 2D.
- H0H1HS = 6D (H0 + H1 + HS).
- Hu = 7D log Hu moments.
- Audit vector = 13D (H0 + H1 + HS + Hu).

## Step Details

### 1_precompute.py
- Loads staged arrays (memmap when possible).
- Uses `train_indices.npy` / `test_indices.npy` for splits.
- Non-resolution protocols use `clean_images.npy` (already CLAHE).
- Resolution uses `raw_images.npy` (green only, no CLAHE) and applies CLAHE inside persistence.
- If `raw_images.npy` is object dtype, falls back to `raw_paths.npy` + `fs.load_image`.
- Writes cache streams: `cache_parts/{split}_{perturbation}_{level}.pkl`.
- Skips existing cache files.

### 2_audit.py
- Welch t-test + Cohen’s d on standard train features.
- Logs to `perf_log.jsonl`.

### 3_signal.py
- 5-fold CV on standard train cache.
- Evaluates H0, H1, HS, H0H1HS, Hu.
- Saves feature matrices to `cache_parts/ablation_features.pkl`.

### 4_generalise.py
- Train/test check on standard H0+H1 (4D).
- Logs to `perf_log.jsonl`.

### 5_ablate.py
- 5-fold CV on combined train+test with fixed folds across perturbations.
- Feature sets: H0, H1, HS, H0H1HS, Hu.
- Outputs:
  - `results_mechanical.csv`, `results_radiometric.csv`, `results_failure.csv`
  - `stress_test_results_{protocol}.csv`
- Logs metrics to `perf_log.jsonl`.

### 6_plot.py
- Reads the latest `5_ablate` run from `perf_log.jsonl`.
- Grouped panels:
  - `plot_panel_mechanical.png` (rotation, blur, resolution)
  - `plot_panel_radiometric.png` (drift, gamma, contrast)
  - `plot_panel_failure.png` (gau_noise, spepper_noise, poi_noise)
- Per-protocol plots: `plot_single_{protocol}.png` (bit_depth is single only).
- Orthogonality plot: `plot_orthogonality.png` (rotation vs drift, H0/H1/HS/H0H1HS/Hu).
- Output dir: `plot_results/`.

## Outputs
- `data/`: staged tensors
- `cache_parts/`: cached persistence/geometry streams
- `perf_log.jsonl`: metrics log
- `results_mechanical.csv`, `results_radiometric.csv`, `results_failure.csv`
- `stress_test_results_{protocol}.csv`
- `plot_results/plot_panel_*.png`
- `plot_results/plot_single_{protocol}.png`
- `plot_results/plot_orthogonality.png`

## Additional Experiments
Standalone rotation experiments that write to `wobble/`:

### 1) Points under rotation — `experiment_shattering_binary.py`
- Rotates 30 isolated pixels and measures connected components.

Run:
```bash
python experiment_shattering_binary.py
```

### 2) Thin loops under rotation — `experiment_loops_binary.py`
- Rotates five thin rectangular loops and measures hole count.

Run:
```bash
python experiment_loops_binary.py
```

## Notes
- Run `0_ingest.py` once before the orchestrator.
- If perturbation ranges or feature definitions change, clear `cache_parts/` and rerun precompute + downstream steps.
