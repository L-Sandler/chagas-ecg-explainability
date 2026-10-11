#!/usr/bin/env python3
"""Real (non-zero-padded) length of every CODE-15% record. Read-only; changes no data.

    python scripts/scan_lengths.py [--hdf5 data/code15/exams_part*.hdf5] [--out-dir reports]

Length rule is identical to preprocess._strip_zero_padding: the span from the first to the last
row that has any non-zero value (all-zero record -> 0). filtfilt (butter order 4 band-pass,
padlen 27) raises when that span is 1..27 samples.

Writes, in --out-dir:
  code15_length_scan.csv     one row per record (large; git-ignored, kept on the volume)
  length_scan_summary.csv    counts per length bin, overall and per part file
  length_scan_flagged.csv    records with real length 0 or 1..27, with label and split
Locked-final labels are masked: `chagas` is blank for every row whose split is "locked" (the
split is still shown, so you can see how many locked records a rule would touch), and
n_chagas_pos counts train + val only. The rule for short and all-zero records is then chosen
without any locked-final label in view (notes/decisions/0004).
The same summary is printed to stdout so it can be pasted into notes/runs.md verbatim.
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PADLEN = 27   # scipy filtfilt default padlen = 3 * max(len(a), len(b)) = 27 for the order-4 band-pass
TARGET_LEN = 2934
CHUNK = 256
BINS = [
    ("all-zero (real_len == 0)", lambda s: s == 0),
    (f"1..{PADLEN} (filtfilt raises)", lambda s: (s >= 1) & (s <= PADLEN)),
    (f"{PADLEN + 1}..{TARGET_LEN - 1} (shorter than crop)", lambda s: (s > PADLEN) & (s < TARGET_LEN)),
    (f"== {TARGET_LEN}", lambda s: s == TARGET_LEN),
    (f"{TARGET_LEN + 1}..4095", lambda s: (s > TARGET_LEN) & (s < 4096)),
    ("== 4096 (no padding)", lambda s: s == 4096),
]


def part_index(p: Path) -> int:
    m = re.search(r"part(\d+)", p.name)
    return int(m.group(1)) if m else -1


def scan_file(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as f:
        tr, ids = f["tracings"], f["exam_id"][:]
        for lo in range(0, tr.shape[0], CHUNK):
            x = tr[lo: lo + CHUNK]                      # [n, T, 12]
            nz = np.any(x != 0, axis=2)                 # [n, T]; NaN != 0 counts as non-zero, as in preprocess
            has = nz.any(axis=1)
            first = np.argmax(nz, axis=1)
            last = x.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1)
            rows.append(pd.DataFrame({
                "file": path.name,
                "row": np.arange(lo, lo + x.shape[0]),
                "exam_id": ids[lo: lo + x.shape[0]].astype(np.int64),
                "T": x.shape[1],
                "first_nz": np.where(has, first, -1),
                "last_nz": np.where(has, last, -1),
                "real_len": np.where(has, last - first + 1, 0),
                "n_nonzero_rows": nz.sum(axis=1),
                "all_finite": np.isfinite(x).all(axis=(1, 2)),
                "max_abs": np.nanmax(np.abs(np.where(np.isfinite(x), x, 0)), axis=(1, 2)),
            }))
    print(f"scanned {path.name}: {tr.shape[0]} records", flush=True)
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    scopes = [("ALL", df)] + [(name, g) for name, g in df.groupby("file", sort=False)]
    for scope, g in scopes:
        for label, rule in BINS:
            m = rule(g.real_len)
            sub = g[m]
            out.append({
                "scope": scope, "bin": label, "n_records": int(m.sum()),
                "n_labelled": int(sub.labelled.sum()),
                "n_chagas_pos": int((sub.chagas == True).sum()),  # noqa: E712 - train+val only (locked is masked)
                "n_train": int((sub.split == "train").sum()),
                "n_val": int((sub.split == "val").sum()),
                "n_locked": int((sub.split == "locked").sum()),
                "n_unlabelled": int((~sub.labelled).sum()),
            })
        out.append({"scope": scope, "bin": "any non-finite value", "n_records": int((~g.all_finite).sum())})
    res = pd.DataFrame(out)
    counts = [c for c in res.columns if c.startswith("n_")]
    return res.astype({c: "Int64" for c in counts})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", nargs="+", help="default: data/code15/exams_part*.hdf5")
    ap.add_argument("--labels", default=str(ROOT / "data/code15/code15_chagas_labels.csv"))
    ap.add_argument("--exams-csv", default=str(ROOT / "data/code15/exams.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    a = ap.parse_args()

    paths = [Path(p) for p in a.hdf5] if a.hdf5 else list((ROOT / "data/code15").glob("exams_part*.hdf5"))
    paths = sorted(paths, key=part_index)
    if not paths:
        print("no HDF5 parts found", file=sys.stderr)
        return 1
    df = pd.concat([scan_file(p) for p in paths], ignore_index=True)

    from dataset import build_split_table  # noqa: PLC0415 - same label/split join the trainer uses
    tab = build_split_table(a.labels, a.exams_csv)[["exam_id", "chagas", "split"]]
    df = df.merge(tab, on="exam_id", how="left")
    df["labelled"] = df.split.notna()
    df["split"] = df.split.fillna("unlabelled")
    df["chagas"] = df.chagas.astype("object")
    df.loc[df.split == "locked", "chagas"] = None  # mask before anything is written or printed

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "code15_length_scan.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "length_scan_summary.csv", index=False)
    flagged = df[df.real_len <= PADLEN][["file", "row", "exam_id", "real_len", "first_nz", "last_nz",
                                        "n_nonzero_rows", "max_abs", "labelled", "chagas", "split"]]
    flagged.sort_values(["real_len", "file", "row"]).to_csv(out / "length_scan_flagged.csv", index=False)

    print(f"\nTOTAL records: {len(df)} in {len(paths)} part files")
    print(summary[summary.scope == "ALL"].drop(columns="scope").to_string(index=False))
    print(f"\nrecords with real_len 1..{PADLEN}: {int(((df.real_len >= 1) & (df.real_len <= PADLEN)).sum())}")
    print("most common real_len values:")
    print(df.real_len.value_counts().head(10).rename_axis("real_len").to_string())
    short = df[(df.real_len >= 1) & (df.real_len <= PADLEN)]
    if len(short):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(short.drop(columns=["T"]).head(40).to_string(index=False))
    print(f"\nwrote {out / 'code15_length_scan.csv'}, length_scan_summary.csv, length_scan_flagged.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
