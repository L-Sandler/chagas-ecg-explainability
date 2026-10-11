#!/usr/bin/env python3
"""Turn a pulled lightning_logs/<run>/ into the tracked record runs/<run>/.

    python3 scripts/collect_run.py lightning_logs/<run> [--out runs] [--force]

Writes (standard library only; nothing is interpreted, every value is copied from the log):
  meta.json         the launcher's train_<run>.meta.json (command, SHA, pod, GPU)
  env.json          the launcher's train_<run>.env.json (env_check output), if present
  summary.json      values parsed from the log: sizes, pos_weight, per-epoch val metrics, the
                    results blocks, PTB-XL line, traceback tail on a crash, checkpoint file list
  val_preds.csv.gz  val_preds.csv, gzipped, if the run wrote one
  train.log.gz      the log with progress-bar lines removed
Checkpoints are not copied (too large for git); their names and sizes go in summary.json.
"""
import argparse
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAR = re.compile(r"\d+%\||it/s\]|s/it\]|\?it/s|it/s,")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_LINE = re.compile(r"Epoch (\d+):\s+100%\|.*?\|\s*\d+/\d+ \[([^\]]*)\]")
EXC_LINE = re.compile(r"^[A-Za-z_.]*(Error|Exception)\b[^\n]*:")
RESULT_START = re.compile(r"=== TEST SET RESULTS")
KV = re.compile(r"([A-Za-z_/@%0-9\-]+)=([-+0-9.eE]+)")


def log_lines(text: str) -> list:
    """Split on both \\r and \\n (tqdm rewrites a line with \\r), drop ANSI escapes and blanks."""
    return [ANSI.sub("", s).rstrip() for s in re.split(r"[\r\n]", text) if ANSI.sub("", s).strip()]


def parse(lines: list) -> dict:
    s = {"epochs": {}, "results_blocks": [], "dataset_notes": [], "traceback_tail": None}
    section = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = EPOCH_LINE.search(ln)
        if m and "val/auroc" in m.group(2):  # the bar line printed after validation carries the metrics
            s["epochs"][int(m.group(1))] = {k: float(v) for k, v in KV.findall(m.group(2))}
        if ln.startswith("--- "):
            section = ln.strip("- ").strip()
        if "[Code15Dataset]" in ln and ln not in s["dataset_notes"]:
            s["dataset_notes"].append(ln.strip())
        for key, pat in (("train_total", r"Train total: (\d+) samples\s+\|\s+Val \(CODE-15%\): (\d+) samples"),
                         ("pos_weight", r"Pos weight \(full train set\): ([0-9.]+)\s+\(CODE-15% train only would be ([0-9.]+)\)")):
            mm = re.search(pat, ln)
            if mm:
                s[key] = [float(x) if "." in x else int(x) for x in mm.groups()]
        for key, pat in (("best_checkpoint", r"Evaluating BEST checkpoint: (.*)"),
                         ("checkpoint_epoch_0based", r"checkpoint epoch \(0-based\): (\d+)\s+\|\s+monitored val/auroc: ([0-9.]+)"),
                         ("ptbxl_fold10_line", r"(PTB-XL fold 10: n=.*)")):
            mm = re.search(pat, ln)
            if mm:
                s[key] = mm.group(1) if len(mm.groups()) == 1 else list(mm.groups())
        if RESULT_START.search(ln):
            block = {"section": section, "lines": []}
            i += 1
            while i < len(lines) and not lines[i].startswith("====="):
                block["lines"].append(lines[i].strip())
                i += 1
            s["results_blocks"].append(block)
        if "Traceback (most recent call last)" in ln and s["traceback_tail"] is None:
            win = [x for x in lines[i: i + 150] if not BAR.search(x)]
            tb = []
            for j, x in enumerate(win):  # through the exception line; a re-raised worker trace continues
                tb.append(x)
                nxt = win[j + 1] if j + 1 < len(win) else ""
                if j and EXC_LINE.match(x.strip()) and not nxt.startswith("Original Traceback"):
                    break
            s["traceback_tail"] = tb
        i += 1
    s["epochs"] = {str(k): v for k, v in sorted(s["epochs"].items())}
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="pulled lightning_logs/<run>/")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing runs/<run>/")
    a = ap.parse_args()

    src = Path(a.run_dir)
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 1
    run = src.name
    dst = Path(a.out) / run
    if dst.exists() and not a.force:
        print(f"{dst} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    missing = []
    for name, cands in (("meta.json", [f"train_{run}.meta.json", "meta.json"]),
                        ("env.json", [f"train_{run}.env.json", "env.json"])):
        found = next((src / c for c in cands if (src / c).is_file()), None)
        if found:
            shutil.copyfile(found, dst / name)
        else:
            missing.append(name)

    log = next((src / c for c in (f"train_{run}.log", "train.log") if (src / c).is_file()), None)
    summary = {"run": run, "source_dir": str(src), "missing_files": missing}
    if log:
        lines = log_lines(log.read_text(errors="replace"))
        summary.update(parse(lines))
        summary["log_file"] = log.name
        with gzip.open(dst / "train.log.gz", "wt") as f:
            f.write("\n".join(x for x in lines if not BAR.search(x)) + "\n")
    else:
        missing.append("train log")
    drv = src / "sweep_driver.log"
    if drv.is_file():
        summary["sweep_driver_log_lines"] = [x for x in drv.read_text().splitlines() if x.strip()]
    vp = src / "val_preds.csv"
    if vp.is_file():
        with open(vp, "rb") as fi, gzip.open(dst / "val_preds.csv.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        missing.append("val_preds.csv")
    summary["checkpoints"] = sorted(
        ({"path": os.path.relpath(os.path.join(d, f), src), "bytes": os.path.getsize(os.path.join(d, f))}
         for d, _, fs in os.walk(src) for f in fs if f.endswith(".ckpt")), key=lambda c: c["path"])
    summary["missing_files"] = missing
    (dst / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {dst}/ : {sorted(p.name for p in dst.iterdir())}")
    if missing:
        print(f"not found in {src}: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
