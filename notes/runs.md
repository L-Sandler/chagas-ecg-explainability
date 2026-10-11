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

### setup + length scan  (2026-10-11)  no training; not a run in the table above

Purpose (from the approved plan): rebuild the pod environment from `uv.lock`, record every input
file's sha256 in `spec/data-manifest.json`, and scan the real length of all 18 CODE-15% parts
(approved after the `baseline-crop2934` crash).

- **Code:** pod `git pull` took the repo from `c74e8bb` to `3da92c4`, later to `657d9d9`. Manifest
  `code_sha` `774ec79`, code-pin check pass (`code_pin_ok: true` in the dry-run header below).
- **Pod:** `m9hjemzxxb9blt`, RTX 4090, EU-RO-1, driver 580.159.04 (CUDA 13.0), `$0.89/hr`, created
  2026-10-11 20:13:45Z, deleted ~20:42Z. Ledger: 0.48 h = $0.42; total GPU spend $10.76 of $50.
  RunPod balance before creating: $25.25. `runpodctl pod list` returned `[]` after delete.
- **Environment before (hand-installed, every earlier number):** full list in
  `runs/_env-pre-lock.json`. torch `2.6.0+cu124`, numpy `2.4.4`, scipy `1.17.1`, h5py `3.16.0`,
  pytorch-lightning `2.6.1`, python `3.11.10`, cuDNN 90100. It also had both `nvidia-*-cu12`
  and `nvidia-*-cu13` packages (e.g. `nvidia-cublas 13.1.0.3`, `nvidia-cudnn-cu13 9.19.0.56`,
  `cuda-toolkit 13.0.2`). Moved to `/workspace/chagas-ecg-explainability/.venv-handinstalled-2026-06`
  (not deleted).
- **Environment after (`bash scripts/setup_runpod.sh`, `env_check.py --require-gpu`, verbatim):**
  ```
  {
    "python": "3.11.10",
    "platform": "Linux-6.8.0-124-generic-x86_64-with-glibc2.35",
    "git_head": "3da92c4",
    "uv": "uv 0.11.7 (x86_64-unknown-linux-gnu)",
    "uv_lock_sha256": "a97e8418c74f9a00c33f5816de5774ef47c86174c40ac6a901dcbe62532e8556",
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "h5py": "3.16.0",
    "pytorch_lightning": "2.6.1",
    "torch": "2.6.0+cu124",
    "torch_cuda": "12.4",
    "cudnn": 90100,
    "cuda_available": true,
    "gpu": "NVIDIA GeForce RTX 4090",
    "driver": "580.159.04",
    "driver_cuda_max": "13.0",
    "forward_backward": {"device": "cuda", "amp_fp16": true, "logits_shape": [4],
                         "loss": 0.6917272806167603, "grads_finite": true},
    "failures": []
  }
  ```
  The Mac's `uv.lock` sha256 is the same (`a97e8418...`). `uv sync --frozen --check` printed
  `Would make no changes` / `venv matches uv.lock`.
- **Setup anomaly (first attempt failed):** while `data_manifest.py --verify` was reading the
  volume, `uv sync --frozen` stopped with
  ```
  error: Failed to install: jupyterlab-4.5.7-py3-none-any.whl (jupyterlab==4.5.7)
    Caused by: Stale file handle (os error 116) at path ".../.venv/lib/python3.11/site-packages/jupyterlab/tests/mock_packages/test_no_hyphens/test_no_hyphens/.tmpLhTaJj"
  ```
  The script was rerun with `UV_CONCURRENT_INSTALLS=4 UV_LINK_MODE=copy`, with nothing else
  running: `Installed 12 packages in 1m 24s`, then the check above. Logs on the volume:
  `setup_runpod.log` (failed), `setup_runpod_retry.log`. The `env_check` step then took ~3 min
  (`ps` showed the process in a filesystem wait, `request_wait_answer`) and passed.
- **Data manifest:** `data_manifest.py --write` on the pod hashed parts 1-17 (kept the locally
  recorded hashes for part0, `exams.csv`, labels CSV and SaMi-Trop); 23 entries. Committed as
  `657d9d9`. `data_manifest.py --verify` (full sha256) on the pod:
  `ptbxl: 87203 files, 0 bad` and
  `verify (full sha256): PASSED  manifest sha256 c916c22a241ae46c4b5f379ceefd7b0202d27b5ab3846b909e289e67cda69520`.
  The same sha256 as the local copy of `spec/data-manifest.json`. Both CSVs' md5 equal Zenodo's
  published md5 (`md5_matches_published: true`). The zips are not on the volume, so the
  published zip md5s are recorded, not compared.
- **Length scan (`scan_lengths.py`, all 18 parts, 345,797 records; locked-split labels masked):**

  | bin | n_records | n_labelled | n_chagas_pos (train+val) | n_train | n_val | n_locked | n_unlabelled |
  |---|---|---|---|---|---|---|---|
  | all-zero (real_len == 0) | 688 | 669 | 2 | 482 | 89 | 98 | 19 |
  | 1..27 | 4 | 4 | 0 | 3 | 1 | 0 | 0 |
  | 28..2933 | 4395 | 4373 | 5 | 3048 | 657 | 668 | 22 |
  | == 2934 | 201084 | 199591 | 5308 | 139808 | 30050 | 29733 | 1493 |
  | 2935..4095 | 8863 | 8795 | 7 | 6220 | 1336 | 1239 | 68 |
  | == 4096 | 130763 | 129992 | 261 | 91027 | 19469 | 19496 | 771 |

  Records with any non-finite value: 0. Most common real_len: 2934 (201084), 4096 (130763),
  3874 (1261), 0 (688), 4000 (173). Part0 alone: 39 / 0 / 273 / 11675 / 477 / 7537, equal to the
  2026-10-10 scan.
- **Records with real_len 1..27 (`reports/length_scan_flagged.csv`, verbatim rows):**
  ```
  file,row,exam_id,real_len,first_nz,last_nz,n_nonzero_rows,max_abs,labelled,chagas,split
  exams_part12.hdf5,14591,2829945,27,2108,2134,27,281.1352,True,False,train
  exams_part15.hdf5,8559,2795803,27,1576,1602,27,281.1352,True,False,train
  exams_part7.hdf5,895,1441593,27,3559,3585,27,243.90517,True,False,val
  exams_part9.hdf5,15730,207914,27,1842,1868,27,219.28545,True,False,train
  ```
  All four have real_len 27 and none has a length in 1..26. The 2026-10-10 traceback reads
  `ValueError: The length of the input vector x must be greater than padlen, which is 27.`
  Which record raised it in that run is not recorded in that log.
- **Files:** `reports/length_scan_summary.csv`, `reports/length_scan_flagged.csv` (tracked;
  sha256 `77c3389c...` and `cb2090ec...`, equal to the pod's). Per-record scan
  `reports/code15_length_scan.csv` (345,797 rows, 29,168,967 bytes, sha256
  `12c943cec545f355e866406c9301ca613a30a0c4c7540a25bacaf77bb0966723`) is on the volume only
  (git-ignored).
- **Launcher dry-run (`python3 scripts/launch_run.py baseline-crop2934 --manifest spec/runs/baseline.json --dry-run`, over SSH):**
  ```
  "git_head": "657d9d9", "manifest_code_sha": "774ec79", "code_pin_ok": true,
  "driver_cuda": 13.0, "pod_id": null,
  "gpu": "| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |",
  "env": {"python": "3.11.10", "torch": "2.6.0+cu124", "torch_cuda": "12.4", "cudnn": 90100,
          "driver": "580.159.04", "driver_cuda_max": "13.0", "gpu": "NVIDIA GeForce RTX 4090",
          "pytorch_lightning": "2.6.1", "numpy": "1.26.4", "scipy": "1.17.1", "h5py": "3.16.0",
          "uv": "uv 0.11.7 (x86_64-unknown-linux-gnu)",
          "uv_lock_sha256": "a97e8418c74f9a00c33f5816de5774ef47c86174c40ac6a901dcbe62532e8556"},
  "env_file": "train_baseline-crop2934.env.json",
  "data_manifest_sha256": "c916c22a241ae46c4b5f379ceefd7b0202d27b5ab3846b909e289e67cda69520",
  "data_inputs_checked": 22 paths (18 parts, labels CSV, exams.csv, samitrop hdf5, data/ptbxl),
  "n_hdf5_parts": 18   ...   DRY RUN - nothing launched.
  ```
  Command string as in the 2026-10-10 entry above. `pod_id` is null and `gpu` is the
  `nvidia-smi` table line because the SSH session has no `RUNPOD_*` env (same as 2026-10-10).
- **Not done:** no training run, no launch marker, `baseline-crop2934` status still `planned`.
  The PTB-XL download branch of `scripts/download_data.sh` was not exercised (data already on
  the volume). `scripts/launch_run.py` full launch path (env sidecar write) was not exercised.
- **Anomalies (factual):**
  - `create_pod.sh` did not provision on the first three attempts: the sandbox denied
    `api.runpod.io` (the redirect/pipe forms are not on the sandbox exemption), and the hook
    recorded a stale pod twice (from a piped exit code, and from a `bash -n` syntax check naming
    the script); the user cleared `.guardrails/state/pod-*` each time. The fourth attempt
    created the pod. `scripts/create_pod.sh` now redacts keys from its output (`1544b56`).
  - A first `git pull` on the pod needed `git checkout -- spec/data-manifest.json` because the
    pod-written manifest differed from the committed one before the push.
  - Two pod-side background jobs (`--write`, then `--verify`) ran concurrently with the first
    `uv sync` attempt; see the stale-file-handle error above.
  - An unmasked per-bin summary of part0 alone was viewed by the agent while testing
    `scan_lengths.py` on the Mac, before label masking was added: 1 Chagas-positive among the
    39 all-zero records, 4 of them in the locked split. The pod outputs above are masked.

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
