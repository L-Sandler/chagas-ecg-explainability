#!/usr/bin/env python3
"""Manifest-driven training launcher. Runs ON THE POD.

    python3 scripts/launch_run.py <run-name> [--manifest spec/runs/lr-sweep.json] [--dry-run]

What it enforces (so the run record does not depend on anyone's memory):
  * run-name must exist in the manifest with status "planned"
  * working tree clean (tracked files) and src/, pyproject.toml, uv.lock identical to code_sha
  * driver CUDA >= manifest min_cuda_version
  * refuses to overwrite an existing log for the same run
  * writes a header (timestamp, HEAD, code-pin check, host, GPU, full command) to the log
    and a sidecar train_<run>.meta.json, then launches detached (nohup semantics)
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: list) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def fail(msg: str) -> None:
    print(f"LAUNCH REFUSED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--manifest", default="spec/runs/lr-sweep.json")
    ap.add_argument("--dry-run", action="store_true", help="print the command and checks, launch nothing")
    a = ap.parse_args()

    manifest = json.loads((ROOT / a.manifest).read_text())
    runs = {r["name"]: r for r in manifest["runs"]}
    if a.run_name not in runs:
        fail(f"'{a.run_name}' not in manifest {a.manifest}; registered: {sorted(runs)}")
    run = runs[a.run_name]
    if run.get("status") != "planned":
        fail(f"run '{a.run_name}' has status '{run.get('status')}', expected 'planned'")

    # --- code identity checks ---
    head = sh(["git", "rev-parse", "--short", "HEAD"])
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        fail(f"tracked files modified on pod (HEAD {head}):\n{dirty}")
    pinned = manifest["code_paths_pinned"]
    diff = subprocess.run(["git", "diff", "--quiet", manifest["code_sha"], "HEAD", "--", *pinned], cwd=ROOT)
    code_pin_ok = diff.returncode == 0
    if not code_pin_ok:
        fail(f"{pinned} differ between manifest code_sha {manifest['code_sha']} and HEAD {head}")

    # --- driver check ---
    smi = sh(["nvidia-smi"])
    cuda_line = next((l for l in smi.splitlines() if "CUDA Version" in l), "")
    try:
        drv = float(cuda_line.split("CUDA Version:")[1].split()[0])
    except (IndexError, ValueError):
        drv = 0.0
    if drv < float(manifest["infra"]["min_cuda_version"]):
        fail(f"driver CUDA {drv} < required {manifest['infra']['min_cuda_version']}; nvidia-smi said: {cuda_line!r}")

    # --- build command ---
    py = manifest["infra"]["python"]
    argv = [py, "src/train.py", *manifest["common_args"], *run["args"], "--run-name", a.run_name]
    expanded = []
    for tok in argv:
        if "*" in tok:
            matches = sorted(str(p) for p in ROOT.glob(tok))
            if not matches:
                fail(f"glob '{tok}' matched nothing")
            expanded.extend(os.path.relpath(m, ROOT) for m in matches)
        else:
            expanded.append(tok)
    argv = expanded

    log = ROOT / f"train_{a.run_name}.log"
    meta = ROOT / f"train_{a.run_name}.meta.json"
    if log.exists() and not a.dry_run:
        fail(f"{log.name} already exists; will not overwrite a prior run log")

    header = {
        "run_name": a.run_name,
        "launched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "gpu": os.environ.get("RUNPOD_GPU_NAME")
        or next((l.strip() for l in smi.splitlines() if "NVIDIA" in l and "|" in l), ""),
        "driver_cuda": drv,
        "git_head": head,
        "manifest_code_sha": manifest["code_sha"],
        "code_pin_ok": code_pin_ok,
        "manifest": a.manifest,
        "command": " ".join(shlex.quote(t) for t in argv),
        "n_hdf5_parts": sum(1 for t in argv if t.endswith(".hdf5") and "code15" in t),
        "log": str(log),
    }

    print(json.dumps(header, indent=2))
    if a.dry_run:
        print("DRY RUN - nothing launched.")
        return

    with log.open("w") as fh:
        fh.write("=== LAUNCH HEADER (scripts/launch_run.py) ===\n")
        fh.write(json.dumps(header, indent=2) + "\n")
        fh.write("=== nvidia-smi ===\n" + smi + "\n=== TRAINING OUTPUT ===\n")
        fh.flush()
        proc = subprocess.Popen(
            argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,  # survives SSH drop, like nohup + disown
        )
    header["pid"] = proc.pid
    meta.write_text(json.dumps(header, indent=2) + "\n")
    print(f"LAUNCHED pid={proc.pid}  log={log}  meta={meta}")


if __name__ == "__main__":
    main()
