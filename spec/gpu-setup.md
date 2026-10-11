# GPU pod setup (RunPod)

Reference for provisioning, environment and data. Updated 2026-10-11. The run-by-run launch
flow, guardrails and termination rules are in `.claude/skills/experiment-facilitator/SKILL.md`;
this file does not repeat them. Reproducing from a clean machine: `REPRODUCING.md`.

## 1. What a pod needs

| Item | Value | Source |
|---|---|---|
| Datacenter | EU-RO-1 | `spec/runs/baseline.json` `infra` |
| Network volume | `w6ahd2sms6`, mounted at `/workspace` (repo, data, venv and pinned uv persist on it) | same |
| Template | `runpod-torch-v240` (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) | same |
| Min driver CUDA | 12.4 (the locked torch is `2.6.0+cu124`) | `pyproject.toml`, `uv.lock` |
| GPU order | RTX 4090, L4, RTX A6000, RTX 2000 Ada (first with stock) | manifest `gpu_preference_order` |
| Excluded | Blackwell cards (need cu128) and AMD MI300X | manifest `gpu_id_note` |

Create pods with `./scripts/create_pod.sh` (walks the order against live stock, forwards the
RunPod and W&B keys into the pod env). Stock is live and low-stock listings are racy: re-check,
never assume. Delete with `runpodctl pod delete <id>` (there is no `terminate`); confirm
`runpodctl pod list` returns `[]`.

Observed so far (2026-08-28 and 2026-10-10; different pipeline versions, so rates, not benchmarks):

| GPU | Billed rate | Epoch time once the page cache is warm |
|---|---|---|
| RTX 4090 | $0.89/hr (2026-10-10; $0.74 on 2026-08-28; manifest lists $0.74) | ~2 min |
| L4 | $0.49/hr | ~5 min |

Epoch 0 is slower on a cold cache (about 3.8 it/s on the 4090). Spend is tracked by
`python3 .claude/hooks/guard.py --ledger` against the $50 global budget.

## 2. Environment (from the lock, never re-resolved on the pod)

```bash
git pull                              # the commit the manifest's code_sha names must be pushed first
bash scripts/setup_runpod.sh
```

`setup_runpod.sh` installs the uv version named by `pyproject.toml` (`required-version`) into
`/workspace/bin`, runs `uv sync --frozen`, checks `uv sync --frozen --check`, then
`scripts/env_check.py --require-gpu`: torch CUDA vs driver CUDA, GPU visible, one AMP
forward/backward of `ECGResNet`. It prints the environment as JSON.

- Linux resolves `torch 2.6.0+cu124` from `https://download.pytorch.org/whl/cu124`
  (`[tool.uv.sources]` in `pyproject.toml`). The Intel Mac resolves `torch 2.2.2` (CPU).
- Numbers reported anywhere come only from the pod environment. The Mac is for smoke tests, the
  length scan and analysis.
- Do not `uv add`, `uv lock` or `uv pip install` on the pod. Changing the lock is a local change
  that gets committed and pushed, and the manifest's `code_sha` moves with it.
- Call `.venv/bin/python` directly (as `launch_run.py` does). `uv run` without `--frozen` can
  re-resolve and resync the venv.
- Before the lock existed the pod venv was hand-installed. That package list is recorded in
  `runs/_env-pre-lock.json` once the setup session has run (see `notes/runs.md`).

## 3. Data

```bash
bash scripts/download_data.sh         # skips files already present, then verifies
python3 scripts/data_manifest.py --verify            # full sha256 of every file, ~once per pod
python3 scripts/data_manifest.py --verify --quick    # presence + size only
```

| Input | Where | Size |
|---|---|---|
| CODE-15% parts 0-17 | `data/code15/exams_part{0..17}.hdf5` (Zenodo 4916206 v1.0.0) | part0 is 3.93 GB; ~63 GB for parts 1-17 (volume, 2026-08-28) |
| Labels | `data/code15/code15_chagas_labels.csv` (PhysioNet Challenge 2025) | 7.1 MB |
| Split universe | `data/code15/exams.csv`: training refuses to start without it | 35.5 MB |
| SaMi-Trop | `data/samitrop/exams.hdf5`, `exams.csv` (Zenodo 4905618 v1.0.0) | 641 MB |
| PTB-XL 1.0.3 | `data/ptbxl/`, files directly in that folder, checked against its own `SHA256SUMS.txt` | 3.2 GB |

`spec/data-manifest.json` records path, size and sha256 for each file, plus the source URL and
Zenodo's published md5 of the zip each HDF5 came from. PTB-XL rsync (port 873) is blocked on
RunPod, so the script uses the HTTPS zip.

## 4. Before launching a run

1. `bash scripts/setup_runpod.sh` passes (or `.venv/bin/python scripts/env_check.py --require-gpu`
   and `uv sync --frozen --check` on a pod that is already set up).
2. `python3 scripts/data_manifest.py --verify` passes.
3. `python3 scripts/launch_run.py <run> --manifest <manifest> --dry-run` passes its checks.
4. Launch only through `launch_run.py` or `sweep_driver.sh`. A direct `src/train.py` call is
   blocked by the hook; a local `--fast` smoke run is allowed.
5. After the run: `python3 scripts/collect_run.py lightning_logs/<run>` writes the tracked
   `runs/<run>/` (meta, env, parsed summary, val predictions, log without progress bars).

Local smoke test (Mac, CPU, ~10 s; checks wiring, not results):

```bash
uv run --frozen python src/train.py --hdf5 data/code15/exams_part0.hdf5 \
    --labels data/code15/code15_chagas_labels.csv --fast --epochs 1
```

## 5. `src/train.py` flags used by the manifests

| Flag | Meaning |
|---|---|
| `--hdf5 <paths...>` | CODE-15% parts (the launcher expands a glob) |
| `--labels`, `--samitrop`, `--ptbxl`, `--exams-csv` | label CSV and the extra sources |
| `--epochs`, `--batch-size`, `--lr` | training length, batch, learning rate |
| `--accelerator gpu --devices 1 --num-workers N` | N is 8 in the baseline manifest; 0 is safe on the Mac |
| `--amp` | fp16 mixed precision |
| `--eval-val` (alias `--eval-test`) | val-only report after training; the locked-final split is never read here |
| `--wandb`, `--run-name` | W&B logging; the name namespaces the checkpoint dir and `val_preds.csv` |
| `--fast` | tiny subset for smoke runs |

## 6. Pod gotchas (observed)

- `--env` variables reach PID 1 only; SSH sessions do not see them. Start the driver after
  `P="$PATH"; . /etc/rp_environment; PATH="$P"` (gives `RUNPOD_*`, not `WANDB_API_KEY`).
- W&B: the user runs `.venv/bin/wandb login` on each pod (`~/.netrc` is on container disk).
- `runpodctl pod create --wait` echoes the env block including keys; pipe through `sed`.
- RunPod injects SSH keys at pod creation only; a pod that lost access must be recreated.
- SSH sessions drop mid-command: detach long work, recover state from the log.
- No `rsync` on the pod image: pull with `scp`. Checkpoint paths contain a literal `val/`
  directory (metric name `val/auroc`); use `find`, not a `*.ckpt` glob.
- Stopped pods can fail to restart when host capacity shifts: delete rather than stop.
