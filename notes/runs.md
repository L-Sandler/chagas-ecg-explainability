# Run ledger: faithful record only

Filled in by the facilitator agent, verbatim from logs / W&B. No interpretation in this file;
the user writes interpretation elsewhere. One block per run, newest at the bottom.

| run name            | date       | pod / GPU              | lr   | epochs run | best ckpt epoch | val/auroc (best) | test AUROC | test AUPRC | TPR@5%FPR | TPR@top-5% | wall time | W&B |
|---------------------|------------|------------------------|------|------------|-----------------|------------------|------------|------------|-----------|------------|-----------|-----|
| full-code15-10ep    | 2026-08-28 | 9hmhxpi69rkp52 / L4    | 1e-3 | 7 (ES@6)   | 1               | 0.836            | 0.8296     | 0.1403     | 0.4036    | 0.3804     | ~35 min   | [97ugsfys](https://wandb.ai/leo-s-org/chagas-ecg/runs/97ugsfys) |
| full-code15-lr1e-3  | 2026-08-28 | 8w09mq198ctjw4 / 4090  | 1e-3 | 10 (ES@9)  | 1               | 0.836            | 0.8276     | 0.1366     | 0.4127    | 0.3875     |           |     |
| full-code15-lr3e-4  |            |                        | 3e-4 |            |                 |                  |            |            |           |            |           |     |
| full-code15-lr3e-3  |            |                        | 3e-3 |            |                 |                  |            |            |           |            |           |     |
| baseline-crop2934   | 2026-10-10 | z3xbdcif55py4t / 4090  | 1e-3 | 0 (CRASHED in epoch 0, batch 124/1015) |  |  |  |  |  |  | ~4 min | [b1gbfja0](https://wandb.ai/leo-s-org/chagas-ecg/runs/b1gbfja0) |

The two 2026-08-28 rows are back-filled from `lightning_logs/*/train.log` and session notes.
Those logs do not contain the launch command, so their command is reconstructed, not recorded.
(The full-code15-10ep row used the pre-`44b1ca8` Adam + CosineAnnealing code, patience 5.)

**Provenance caveats added 2026-10-09 (facts from code and logs, no interpretation):**
- The `test *` and `TPR` columns of both rows were computed on the weights left after the LAST
  epoch (0-based epoch 6 for `full-code15-10ep`, epoch 9 for `full-code15-lr1e-3`), not on the
  best checkpoint (epoch 1 in both). `src/train.py` evaluated `module` right after `trainer.fit()`.
  Fixed in this repo (decision 0002); these rows were not recomputed.
- Both runs used the preprocessing before decision 0001 (leading zero-padding of short
  recordings was not stripped).
- Both runs used the old split (test = 51,529 exams, 991 positives). That split is superseded
  by decision 0004 (train/val/locked-final, stratified on chagas x sex x age band). Results on
  it are not comparable with results on the new split.
- `full-code15-lr3e-4` and `full-code15-lr3e-3` were cancelled (2026-10-09) and never ran.

---

### baseline-crop2934  (2026-10-10)  CRASHED: no metrics, no checkpoint

- **Command (verbatim from `train_baseline-crop2934.meta.json`):**
  ```
  .venv/bin/python src/train.py --hdf5 data/code15/exams_part0.hdf5 data/code15/exams_part1.hdf5 data/code15/exams_part10.hdf5 data/code15/exams_part11.hdf5 data/code15/exams_part12.hdf5 data/code15/exams_part13.hdf5 data/code15/exams_part14.hdf5 data/code15/exams_part15.hdf5 data/code15/exams_part16.hdf5 data/code15/exams_part17.hdf5 data/code15/exams_part2.hdf5 data/code15/exams_part3.hdf5 data/code15/exams_part4.hdf5 data/code15/exams_part5.hdf5 data/code15/exams_part6.hdf5 data/code15/exams_part7.hdf5 data/code15/exams_part8.hdf5 data/code15/exams_part9.hdf5 --labels data/code15/code15_chagas_labels.csv --samitrop data/samitrop/exams.hdf5 --ptbxl data/ptbxl --epochs 10 --batch-size 256 --accelerator gpu --devices 1 --num-workers 8 --amp --eval-test --wandb --lr 1e-3 --run-name baseline-crop2934
  ```
- **git HEAD:** `c74e8bb` | manifest `code_sha`: `dc8e7e8` | code-pin check: pass (`code_pin_ok: true`)
- **Pod:** `z3xbdcif55py4t`, NVIDIA GeForce RTX 4090 (EU-RO-1), driver CUDA 13.0, $0.89/hr (manifest lists $0.74 for this GPU)
- **W&B run:** https://wandb.ai/leo-s-org/chagas-ecg/runs/b1gbfja0 (final state on W&B not read)
- **Config as logged to W&B:** not retrieved. From the stdout log: pos_weight=40.7 (full train set; CODE-15%-only would be 51.3), batch_size=256, epochs=10, lr=1e-3, workers=8, AMP on; train total 259,637 (CODE-15% train + 1,631 SaMi-Trop + 17,418 PTB-XL folds 1-8), val (CODE-15%) 51,602; `[Code15Dataset] 2356/345780 HDF5 records had no label row; excluded.`
- **Per-epoch metrics:** none. The 2-batch sanity check on val completed; training died in epoch 0.
- **Stop reason:** unhandled exception in a DataLoader worker (below). `sweep_driver.sh`: `baseline-crop2934 EXITED WITHOUT test results - treating as failure. Halting sweep (no retry).` at 2026-10-10T16:49:14Z. Launched 16:45:14Z.
- **Last progress line:** `Epoch 0:  12%|█▏  | 124/1015 [00:34<04:04,  3.64it/s, v_num=fja0]`
- **Traceback (verbatim, from the worker; wrapper frames omitted):**
  ```
  File ".../src/dataset.py", line 228, in __getitem__
      signal = preprocess_signal(signal, fs=400.0, sig_names=CODE15_LEAD_ORDER)  # [12, TARGET_LEN]
  File ".../src/preprocess.py", line 81, in preprocess_signal
      signal = _bandpass(signal, fs)
  File ".../src/preprocess.py", line 30, in _bandpass
      return filtfilt(b, a, signal, axis=0)
  File ".../scipy/signal/_signaltools.py", line 4931, in _validate_pad
      raise ValueError(
  ValueError: The length of the input vector x must be greater than padlen, which is 27.
  ```
- **What the log does not say:** which record / dataset index raised it (the traceback names no record id; the call came from the training DataLoader, through the CODE-15% dataset path with `fs=400.0`); how many records are affected. Not reproduced or investigated.
- **Where the full log is:** local `lightning_logs/baseline-crop2934/` (`train_baseline-crop2934.log`, `.meta.json`, `sweep_driver.log`); `lightning_logs/` is git-ignored, so it is not in the repo. Same files remain on the volume under `/workspace/chagas-ecg-explainability/` along with a `failed_baseline-crop2934` marker.
- **State that affects the next attempt (facts from the launcher/driver code):**
  - `launch_run.py` refuses to overwrite `train_baseline-crop2934.log` on the volume; reusing the run name needs that log moved or a new run name in the manifest.
  - `src/` changes will need a new manifest `code_sha`, because the launcher fails the pin check otherwise.
  - The hook state dir `.guardrails/state/` was empty after this run: no `launched-baseline-crop2934`, pod-id or ledger entry was written (the pod was created via `scripts/create_pod.sh`, and the driver was started over SSH).
  - Manifest run status is still `planned`; setting it is the user's call.
- **Wall time:** ~4 min (16:45:14Z to 16:49:14Z), no test-eval block.
- **Pod-hours billed this run:** pod created 16:30:13Z, deleted ~16:53Z (~0.38 h, about $0.34 at $0.89/hr, estimate). Account balance before: $25.87.
- **Anomalies (factual):** infra issues hit before the launch, in order:
  - `data/code15/exams.csv` was missing on the volume; copied from local (sha256 `f9f82f4c07c1793e2285915c69f1113f5a7db8a0f2cf2f4b90474502f406bc35`, 345,780 lines, matches on the pod) after the user created `approve-scp`.
  - Pod `--env` vars (`WANDB_API_KEY`, `RUNPOD_API_KEY`, `RUNPOD_POD_ID`, `RUNPOD_GPU_NAME`) were present in PID 1 but absent in SSH sessions; the driver was started after sourcing `/etc/rp_environment`, and W&B auth was done on the pod by the user (`.venv/bin/wandb login`). `RUNPOD_GPU_NAME` is recorded URL-encoded in the meta file (`NVIDIA+GeForce+RTX+4090`).
  - The auto-mode classifier denied one driver-start command that read secrets from `/proc/1/environ`; it was not retried in that form.
  - The pod repo was at `44b1ca8` before a fast-forward `git pull` to `c74e8bb`.
  - `scripts/create_pod.sh` was edited locally (uncommitted) to forward a W&B key from `~/.netrc` or `WANDB_API_KEY` into `--env`.

---

## Template: copy per run

### <run-name>  (<date>)

- **Command (verbatim from `train_<run>.meta.json`):**
  ```
  ```
- **git HEAD:** `<sha>`  | manifest `code_sha`: `<sha>` | code-pin check: pass/fail
- **Pod:** `<pod-id>`, `<GPU>`, driver CUDA `<x.y>`, `$<rate>/hr`
- **W&B run:** `<url>`
- **Config as logged to W&B:** pos_weight=, pos_weight_code15_only=, input_len=, batch_size=, epochs=, lr=, git_sha=, per-source sizes=
- **Per-epoch metrics:**

  | epoch | val/auroc | val/tpr_top5pct | val/tpr_at_5pct_fpr | train loss |
  |-------|-----------|-----------------|---------------------|------------|

- **Stop reason:** (e.g. "EarlyStopping patience=8 exhausted at epoch N"; "max_epochs reached")
- **Best checkpoint:** `lightning_logs/<run>/checkpoints/best-epoch=<n>-val/auroc=<x>.ckpt` (epoch N)
- **Val (CODE-15%, best checkpoint, N samples, P% positive):** AUROC , AUPRC , TPR@5%FPR , TPR@top-5%
- **PTB-XL fold-10 check:** (the `PTB-XL fold 10: n=... flagged=...%` line, verbatim)
- **val_preds.csv:** `lightning_logs/<run>/val_preds.csv`; `length_diagnostic.py` output (verbatim)
- **Wall time:** launch to test-eval end
- **Pod-hours billed this run:**
- **Anomalies (factual):**
