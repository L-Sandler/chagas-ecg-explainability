#!/usr/bin/env python3
"""Record the environment a run used, and prove the GPU path works. Prints one JSON object.

    .venv/bin/python scripts/env_check.py [--require-gpu] [--packages] [--no-forward]

  --require-gpu  exit 1 unless CUDA is available (the pod setup passes this)
  --packages     also list every installed distribution (importlib.metadata)
  --no-forward   skip the ECGResNet forward/backward pass (for recording an old venv as it is)

Exit code is non-zero if any check failed; the failures are listed under "failures".
"""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def version(mod: str) -> str:
    try:
        return importlib.import_module(mod).__version__
    except Exception as e:  # noqa: BLE001 - a missing package is a recorded fact, not a crash
        return f"UNAVAILABLE ({type(e).__name__})"


def sha256_of(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def smi() -> dict:
    out = {"driver": None, "driver_cuda_max": None, "gpus": []}
    q = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    for line in q.splitlines():
        name, drv, mem = [x.strip() for x in line.split(",")]
        out["driver"] = drv
        out["gpus"].append({"name": name, "memory": mem})
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", run(["nvidia-smi"]))
    out["driver_cuda_max"] = m.group(1) if m else None
    return out


def forward_backward(torch, failures: list) -> dict:
    """One forward + backward of ECGResNet on random input, AMP fp16 when on a GPU."""
    sys.path.insert(0, str(ROOT / "src"))
    from model import ECGResNet  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = ECGResNet().to(dev)
    x = torch.randn(4, 12, 2934, device=dev)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0], device=dev)
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
    loss.backward()
    grads_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    res = {"device": dev, "amp_fp16": dev == "cuda", "logits_shape": list(logits.shape),
           "loss": float(loss), "grads_finite": grads_finite}
    if not (torch.isfinite(loss).item() and grads_finite):
        failures.append("forward/backward produced a non-finite loss or gradient")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--packages", action="store_true")
    ap.add_argument("--no-forward", action="store_true")
    a = ap.parse_args()

    failures = []
    uv_bin = shutil.which("uv") or ("/workspace/bin/uv" if Path("/workspace/bin/uv").exists() else None)
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "git_head": run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]) or None,
        "uv": run([uv_bin, "--version"]) if uv_bin else None,
        "uv_lock_sha256": sha256_of(ROOT / "uv.lock"),
        "numpy": version("numpy"), "scipy": version("scipy"), "h5py": version("h5py"),
        "pytorch_lightning": version("pytorch_lightning"),
    }
    try:
        import torch  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"UNAVAILABLE ({type(e).__name__}: {e})"
        info["failures"] = ["torch import failed"]
        print(json.dumps(info, indent=2))
        return 1

    nv = smi()
    info.update({
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": nv["driver"], "driver_cuda_max": nv["driver_cuda_max"], "nvidia_smi_gpus": nv["gpus"],
    })

    if a.require_gpu and not torch.cuda.is_available():
        failures.append("CUDA is not available but --require-gpu was given")
    if torch.version.cuda and nv["driver_cuda_max"]:
        want = tuple(int(v) for v in torch.version.cuda.split("."))
        have = tuple(int(v) for v in nv["driver_cuda_max"].split("."))
        if want > have:
            failures.append(f"torch is built for CUDA {torch.version.cuda}; driver supports only "
                            f"{nv['driver_cuda_max']}")
    if not a.no_forward:
        try:
            info["forward_backward"] = forward_backward(torch, failures)
        except Exception as e:  # noqa: BLE001
            failures.append(f"forward/backward failed: {type(e).__name__}: {e}")
    if a.packages:
        info["packages"] = {d.metadata["Name"]: d.version for d in md.distributions()}
        info["packages"] = dict(sorted(info["packages"].items(), key=lambda kv: kv[0].lower()))

    info["failures"] = failures
    print(json.dumps(info, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
