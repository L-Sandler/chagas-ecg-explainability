# Reproducing this work

Order: data, verify, environment, run, compare. Status of each step is in the last section.

## 0. Prerequisites

- `uv` 0.11.7 exactly (`pyproject.toml` has `required-version = "==0.11.7"`).
- Reported numbers come only from the pod environment: Linux x86_64, NVIDIA driver with CUDA >= 12.4,
  `torch 2.6.0+cu124`. An Intel Mac resolves `torch 2.2.2` (CPU); use it for smoke tests, the
  length scan and analysis, not for reported numbers.

## 1. Data

```bash
bash scripts/download_data.sh        # ~72 GB; skips files already present; ends with the verify below
```

Sources and versions are in `spec/data-manifest.json`: CODE-15% (Zenodo 4916206 v1.0.0, 18 HDF5
parts and `exams.csv`), the Challenge 2025 Chagas labels, SaMi-Trop (Zenodo 4905618 v1.0.0),
PTB-XL 1.0.3 (PhysioNet). Zenodo publishes md5 for the zips; the manifest records them next to
the sha256 of the unzipped HDF5 that the code reads.

## 2. Verify the data

```bash
python3 scripts/data_manifest.py --verify                 # full sha256 of every file
python3 scripts/data_manifest.py --verify --allow-missing # a machine that holds only part0, e.g. the Mac
```

Expect `verify (full sha256): PASSED` and `manifest sha256 c916c22a...` for the full set (the
manifest also hashes itself when printed; compare it with `python3 scripts/data_manifest.py --manifest-sha256`).

## 3. Environment

```bash
bash scripts/setup_runpod.sh         # pod: pinned uv, `uv sync --frozen`, env_check on the GPU
uv sync --frozen                     # Mac / other machine
.venv/bin/python scripts/env_check.py [--require-gpu]
```

Expect `uv.lock` sha256 `a97e8418c74f9a00c33f5816de5774ef47c86174c40ac6a901dcbe62532e8556`
and, on the pod, `torch 2.6.0+cu124` with a finite AMP forward/backward.

## 4. Check the inputs the training code sees

```bash
python scripts/scan_lengths.py       # all 18 parts; writes reports/*length_scan*
```

Compare `reports/length_scan_summary.csv` and `reports/length_scan_flagged.csv` with the tracked
copies. Locked-split labels are masked in these outputs by design.

## 5. Run

Runs are manifest-driven (`spec/runs/*.json`) through `scripts/launch_run.py`, which refuses to
launch unless the code pin, the lock, the CUDA path and the data manifest all match. On a pod,
follow `.claude/skills/experiment-facilitator/SKILL.md`; setup details are in `spec/gpu-setup.md`.

Local smoke test (checks wiring only, not results):

```bash
uv run --frozen python src/train.py --hdf5 data/code15/exams_part0.hdf5 \
    --labels data/code15/code15_chagas_labels.csv --fast --epochs 1
```

## 6. Compare

`python3 scripts/collect_run.py lightning_logs/<run>` writes `runs/<run>/` (`meta.json`,
`env.json`, `summary.json`, `val_preds.csv.gz`, `train.log.gz`). Compare your `summary.json`
and `env.json` with the tracked ones. `notes/runs.md` holds the verbatim run records.

## Seeds and determinism

Split seed 42 (notes/decisions/0004) and `seed_everything(42)` in training. Runs are seeded and
recorded, not bitwise-deterministic (cuDNN and AMP kernels): expect metrics to match to the
precision the log prints, not bit for bit.

## What is verified and what is not (2026-10-11)

- Verified on the pod: lock and venv agree, GPU forward/backward, full sha256 of all inputs,
  the 18-part length scan, the launcher dry-run header.
- Verified on the Mac: `uv sync --frozen`, 1-epoch `--fast` smoke run, part0 scan counts.
- Not exercised: the PTB-XL download branch of `download_data.sh` (data was already on the
  volume), a full launch with the env sidecar, a run's `collect_run.py` output (no run has been
  made with the locked environment yet).
- W&B project is private; checkpoints are not in the repo.
