"""
Print verified descriptive statistics for all three ECG datasets.

Usage:
    uv run python src/dataset_stats.py

Output matches the numbers in spec/dataset-stats.md. Run this script to
regenerate those numbers if the data files change.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_SPLIT_SEED = 42


def _patient_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mirrors dataset.py _patient_level_splits exactly."""
    patients = df.groupby("patient_id").agg(chagas_any=("chagas", "any")).reset_index()

    def _split(pids):
        tr, tmp = train_test_split(pids, test_size=0.30, random_state=_SPLIT_SEED)
        v, te = train_test_split(tmp, test_size=0.50, random_state=_SPLIT_SEED)
        return tr, v, te

    pos_tr, pos_v, pos_te = _split(patients[patients["chagas_any"]]["patient_id"].values)
    neg_tr, neg_v, neg_te = _split(patients[~patients["chagas_any"]]["patient_id"].values)
    return {
        "train": df[df["patient_id"].isin(set(pos_tr) | set(neg_tr))],
        "val":   df[df["patient_id"].isin(set(pos_v)  | set(neg_v))],
        "test":  df[df["patient_id"].isin(set(pos_te) | set(neg_te))],
    }


def stats_code15() -> None:
    # Collect all part files that are present locally
    import glob
    all_parts = sorted(glob.glob("data/code15/exams_part*.hdf5"))
    hdf5 = all_parts[0] if all_parts else "data/code15/exams_part0.hdf5"
    labels_csv = "data/code15/code15_chagas_labels.csv"
    if not (os.path.exists(hdf5) and os.path.exists(labels_csv)):
        print("CODE-15%: SKIP (files not present)")
        return

    # Collect all part files present locally
    import glob
    all_parts = sorted(glob.glob("data/code15/exams_part*.hdf5"))

    hdf5_ids: set[int] = set()
    part_info = []
    for path in all_parts:
        with h5py.File(path, "r") as f:
            shape = f["tracings"].shape
            dtype = f["tracings"].dtype
            ids = set(int(x) for x in f["exam_id"][:])
        part_info.append((path, shape, dtype, len(ids)))
        hdf5_ids |= ids

    n_hdf5 = sum(p[1][0] for p in part_info)

    labels = pd.read_csv(labels_csv)

    # Full-label-CSV stats (all 18 parts worth of labels)
    print("=== CODE-15% full label CSV (all 18 parts) ===")
    print(f"  Total labeled exams:     {len(labels):,}")
    print(f"  Unique patients:         {labels['patient_id'].nunique():,}")
    print(f"  Chagas positive:         {labels['chagas'].sum():,} ({labels['chagas'].mean()*100:.2f}%)")

    # Parts present locally
    print(f"\n  Parts present locally: {len(all_parts)} of 18 (part0–part17)")
    for path, shape, dtype, nids in part_info:
        name = os.path.basename(path)
        print(f"    {name}: {shape[0]:,} records  shape={shape}  dtype={dtype}")

    matched = labels[labels["exam_id"].isin(hdf5_ids)].reset_index(drop=True)
    n_unlabeled = n_hdf5 - len(matched)

    exams_per_pt = matched.groupby("patient_id").size()
    pos_pts = matched.groupby("patient_id")["chagas"].any()

    print(f"\n  Locally matched exams:   {len(matched):,} ({n_unlabeled} HDF5 records unlabeled, dropped)")
    print(f"  Unique patients:         {matched['patient_id'].nunique():,}")
    print(f"  Chagas positive exams:   {matched['chagas'].sum():,} / {len(matched):,} ({matched['chagas'].mean()*100:.2f}%)")
    print(f"  Chagas positive patients:{pos_pts.sum():,} / {len(pos_pts):,} ({pos_pts.mean()*100:.2f}%)")
    print(f"  Exams per patient:       min={exams_per_pt.min()}  max={exams_per_pt.max()}  "
          f"mean={exams_per_pt.mean():.2f}  median={exams_per_pt.median():.0f}")
    print(f"  Patients with >1 exam:   {(exams_per_pt > 1).sum():,}")

    splits = _patient_splits(matched)
    print(f"\n  Patient-level 70/15/15 splits (seed={_SPLIT_SEED}):")
    print(f"  {'split':6}  {'exams':>6}  {'patients':>8}  {'positive':>8}  {'prevalence':>10}")
    for name, df in splits.items():
        print(f"  {name:6}  {len(df):6}  {df['patient_id'].nunique():8}  "
              f"{df['chagas'].sum():8}  {df['chagas'].mean()*100:9.2f}%")

    # Leakage check
    ps = {name: set(df["patient_id"]) for name, df in splits.items()}
    leaks = [(a, b, ps[a] & ps[b]) for a, b in [("train","val"),("train","test"),("val","test")]]
    any_leak = any(len(overlap) > 0 for _, _, overlap in leaks)
    print(f"\n  Patient leakage check: {'FAIL' if any_leak else 'PASS — zero overlap across all splits'}")
    if any_leak:
        for a, b, overlap in leaks:
            if overlap:
                print(f"    {a} ∩ {b}: {len(overlap)} shared patients")


def stats_samitrop() -> None:
    hdf5 = "data/samitrop/exams.hdf5"
    csv = "data/samitrop/exams.csv"
    if not (os.path.exists(hdf5) and os.path.exists(csv)):
        print("SaMi-Trop: SKIP (files not present)")
        return

    with h5py.File(hdf5, "r") as f:
        ts_shape = f["tracings"].shape
        ts_dtype = f["tracings"].dtype

    df = pd.read_csv(csv)

    print("\n=== SaMi-Trop ===")
    print(f"  HDF5 shape:     {ts_shape}  dtype={ts_dtype}")
    print(f"  HDF5 records:   {ts_shape[0]}")
    print(f"  CSV rows:       {len(df)}")
    print(f"  CSV columns:    {list(df.columns)}")
    print(f"  Chagas label:   ALL POSITIVE (implicit, per prepare_samitrop_data.py:178)")
    print(f"  Index-based 70/15/15 split:")
    n = len(df)
    print(f"    train: {int(n*0.70)}  val: {int(n*0.85)-int(n*0.70)}  test: {n-int(n*0.85)}")


def stats_ptbxl() -> None:
    records_file = "data/ptbxl/RECORDS"
    if not os.path.exists(records_file):
        print("PTB-XL: SKIP (RECORDS file not present)")
        return

    with open(records_file, encoding="utf-8") as fh:
        text = fh.read()
    all_paths = re.findall(r"records\d+/\d+/\d+_(?:lr|hr)", text)
    lr = [p for p in all_paths if p.endswith("_lr")]
    hr = [p for p in all_paths if p.endswith("_hr")]
    uids = {os.path.basename(p).rsplit("_", 1)[0] for p in all_paths}

    print("\n=== PTB-XL 1.0.3 ===")
    print(f"  RECORDS entries: {len(all_paths)} ({len(lr)} lr + {len(hr)} hr)")
    print(f"  Unique recordings: {len(uids)}")
    print(f"  Chagas label:      ALL NEGATIVE (implicit, per prepare_ptbxl_data.py:132)")
    print(f"  Used resolution:   500 Hz (_hr), resampled to 400 Hz in preprocessing")


def main() -> None:
    print("Dataset Descriptive Statistics")
    print("=" * 60)
    stats_code15()
    stats_samitrop()
    stats_ptbxl()
    print("\n" + "=" * 60)
    print("To revalidate preprocessing on real data: uv run python src/audit_preprocessing.py")


if __name__ == "__main__":
    main()
