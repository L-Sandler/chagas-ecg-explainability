"""
Datasets for CODE-15% (HDF5), SaMi-Trop (HDF5), and PTB-XL (WFDB).

CODE-15% HDF5 layout (each exams_partN.hdf5):
    exam_id:  int64[N]          — exam IDs
    tracings: float32[N, T, 12] — raw ECG signals (T ≈ 4096, 12 leads)

Labels come from a separate CSV with columns: exam_id, patient_id, chagas (bool).

CODE-15% spans 18 part files (exams_part0.hdf5 … exams_part17.hdf5), each with
~20,000 records. Pass all available parts for full training; part0 alone is fine
for local prototyping.

Splitting strategy (notes/decisions/0004): patient-level 70/15/15 train / val / locked,
stratified on chagas x sex x age band, over the full labelled universe (labels CSV joined to
exams.csv). A fixed seed (42) makes it reproducible without a saved manifest, and it does not
depend on which HDF5 parts are loaded. "locked" is the final held-out set: opening it needs
unlock_final=True. No patient appears in more than one split.

SaMi-Trop HDF5 layout (exams.hdf5):
    tracings: float64[1631, 4096, 12] — all records Chagas-positive (serologically
    confirmed). No exam_id key; row index aligns with rows in exams.csv.
    All 1,631 records are used for training only — too small to split and
    withholding confirmed positives from training is not warranted.

PTB-XL uses the official strat_fold split (Wagner et al. 2020):
    folds 1–8 = train, 9 = val, 10 = test.
    All records are treated as Chagas-negative (geographic assumption, same as
    the PhysioNet 2025 challenge). Records read from records100/ (100 Hz WFDB),
    resampled to 400 Hz by preprocess_ecg.
"""

import os

import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from preprocess import preprocess_ecg, preprocess_signal, CODE15_LEAD_ORDER, SAMITROP_LEAD_ORDER

_SPLIT_SEED = 42

# notes/decisions/0004: patient-level 70/15/15 train / val / locked-final.
# "locked" is read once, at the very end; everything else (selection, tuning, explainability
# exploration, the equity audit) uses "val".
SPLIT_NAMES = ("train", "val", "locked")
_SPLIT_FRACTIONS = (0.70, 0.15)  # train, val; locked is the remainder

# 10-year age bands with the young end merged (0004/0005). np.digitize is left-closed:
# 39 -> 0, 40 -> 1, ..., 80 -> 5.
AGE_BAND_EDGES = (40, 50, 60, 70, 80)
AGE_BAND_LABELS = ("<40", "40-49", "50-59", "60-69", "70-79", "80+")

_SPLIT_TABLE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def age_band(age) -> np.ndarray:
    """Band index (0..5) into AGE_BAND_LABELS."""
    return np.digitize(age, AGE_BAND_EDGES)


def _assign_patient_splits(patients: pd.DataFrame) -> pd.Series:
    """
    patients: one row per patient with columns patient_id, chagas_any, is_male, band.
    Returns Series patient_id -> split name.

    Within every (chagas_any, sex, age band) stratum the patients are shuffled with a fixed
    seed and cut 70/15/15, so every subgroup is proportionally represented in every split.
    Deterministic given the SET of patients (strata are visited in sorted order and members
    are sorted before shuffling), so row order and file order do not matter.
    """
    rng = np.random.RandomState(_SPLIT_SEED)
    assignment: dict = {}
    for _, grp in patients.groupby(["chagas_any", "is_male", "band"], sort=True):
        pids = rng.permutation(np.sort(grp["patient_id"].values))
        n = len(pids)
        n_train = int(round(_SPLIT_FRACTIONS[0] * n))
        n_val = int(round(_SPLIT_FRACTIONS[1] * n))
        for pid in pids[:n_train]:
            assignment[pid] = "train"
        for pid in pids[n_train:n_train + n_val]:
            assignment[pid] = "val"
        for pid in pids[n_train + n_val:]:
            assignment[pid] = "locked"
    return pd.Series(assignment)


def build_split_table(labels_csv: str, exams_csv: str) -> pd.DataFrame:
    """
    One row per labelled CODE-15% exam, with its patient-level split, computed over the FULL
    labelled universe (labels CSV joined to exams.csv for age/sex). Membership therefore never
    depends on which HDF5 parts happen to be loaded (closes the old part0-only split BUG).

    Columns: exam_id, patient_id, chagas, age, is_male, band, split.
    A patient's age is their minimum recorded age (proxy for first exam: exams.csv has no
    date); sex is constant per patient in this data.
    """
    key = (os.path.abspath(labels_csv), os.path.abspath(exams_csv))
    if key in _SPLIT_TABLE_CACHE:
        return _SPLIT_TABLE_CACHE[key]
    if not os.path.exists(exams_csv):
        raise FileNotFoundError(
            f"{exams_csv} not found. The patient split needs CODE-15% exams.csv (age, sex): "
            f"it is the split universe. Pass exams_csv= or place it next to the labels CSV."
        )

    labels = pd.read_csv(labels_csv)
    demo = pd.read_csv(exams_csv, usecols=["exam_id", "age", "is_male"])
    df = labels.merge(demo, on="exam_id", how="left")
    missing = int(df[["age", "is_male"]].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} labelled exams have no age/sex in {exams_csv}")

    patients = (
        df.groupby("patient_id")
        .agg(chagas_any=("chagas", "any"), is_male=("is_male", "first"), age=("age", "min"))
        .reset_index()
    )
    patients["band"] = age_band(patients["age"].values)
    patient_split = _assign_patient_splits(patients)

    df["band"] = age_band(df["age"].values)
    df["split"] = df["patient_id"].map(patient_split)
    _SPLIT_TABLE_CACHE[key] = df
    return df


class Code15Dataset(Dataset):
    """
    CODE-15% dataset backed by one or more HDF5 part files.

    hdf5_paths:  path or list of paths to exams_partN.hdf5 files.
                 Pass all 18 parts for full training; part0 alone for prototyping.
    labels_csv:  path to CSV with columns 'exam_id', 'patient_id', 'chagas'
    split:       'train' | 'val' | 'locked'
    max_samples: cap total samples (useful for CPU smoke tests)
    exams_csv:   CODE-15% exams.csv (age, sex); default: exams.csv next to labels_csv
    unlock_final: must be True to open the 'locked' split (see notes/decisions/0004)

    Splitting is patient-level, stratified on chagas x sex x age band, 70/15/15 with seed 42,
    computed over the full labelled universe (independent of which HDF5 parts are loaded).
    No patient appears in more than one split.
    """

    def __init__(
        self,
        hdf5_paths: str | list[str],
        labels_csv: str,
        split: str = "train",
        max_samples: int | None = None,
        exams_csv: str | None = None,
        unlock_final: bool = False,
    ):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}; got {split!r}")
        if split == "locked" and not unlock_final:
            raise PermissionError(
                "The 'locked' split is the final held-out set: read it once, at the very end. "
                "Development, tuning and the equity audit use 'val'. Pass unlock_final=True "
                "only for the final evaluation (notes/decisions/0004)."
            )

        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self.hdf5_paths = list(hdf5_paths)

        # Build position lookup: exam_id → (file_index, row_in_file)
        self._id_to_pos: dict[int, tuple[int, int]] = {}
        for file_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                for row, eid in enumerate(f["exam_id"][:]):
                    self._id_to_pos[int(eid)] = (file_idx, row)

        if exams_csv is None:
            exams_csv = os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "exams.csv")
        table = build_split_table(labels_csv, exams_csv)
        labeled_ids = set(table["exam_id"])
        n_total = len(self._id_to_pos)
        n_unlabeled = sum(1 for eid in self._id_to_pos if eid not in labeled_ids)
        if n_unlabeled > 0:
            print(
                f"[Code15Dataset] {n_unlabeled}/{n_total} HDF5 records had no label row; "
                f"excluded."
            )
        table = table[table["exam_id"].isin(self._id_to_pos)]
        df = table[table["split"] == split]

        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.df = df.reset_index(drop=True)
        self.split = split
        self._hdf5_handles: dict[int, h5py.File] = {}  # opened lazily per-worker

    def _get_hdf5(self, file_idx: int) -> h5py.File:
        if file_idx not in self._hdf5_handles:
            self._hdf5_handles[file_idx] = h5py.File(self.hdf5_paths[file_idx], "r")
        return self._hdf5_handles[file_idx]

    def real_lengths(self) -> np.ndarray:
        """Original (unpadded) length in samples of every row of self.df: the span between the
        first and last non-zero sample. ~2934 for 7 s recordings, 4096 for 10 s ones.
        Reads the raw HDF5 once; used for the length diagnostic (notes/decisions/0006)."""
        out = np.zeros(len(self.df), dtype=int)
        for i, eid in enumerate(self.df["exam_id"].values):
            file_idx, row = self._id_to_pos[int(eid)]
            nz = np.flatnonzero(np.any(self._get_hdf5(file_idx)["tracings"][row] != 0, axis=1))
            out[i] = nz[-1] - nz[0] + 1 if nz.size else 0
        return out

    def pos_weight(self) -> float:
        """Ratio of negatives to positives — pass to BCEWithLogitsLoss."""
        n_pos = self.df["chagas"].sum()
        n_neg = len(self.df) - n_pos
        return float(n_neg) / max(float(n_pos), 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_idx, row_in_file = self._id_to_pos[int(row["exam_id"])]

        f = self._get_hdf5(file_idx)
        signal = f["tracings"][row_in_file]  # [T, 12], float32
        signal = preprocess_signal(signal, fs=400.0, sig_names=CODE15_LEAD_ORDER)  # [12, TARGET_LEN]

        label = float(row["chagas"])
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.float32)


class SamiTropDataset(Dataset):
    """
    SaMi-Trop dataset backed by exams.hdf5.

    All 1,631 records are serologically confirmed Chagas-positive; label is
    hardcoded to 1.0 (no chagas column exists in the CSV).

    The HDF5 has no exam_id key — row index aligns with rows in exams.csv.
    All records are used for training; the dataset is too small to split and
    withholding confirmed positives from training is not warranted.

    HDF5 handles are opened lazily per DataLoader worker so that forked
    processes each hold their own file descriptor.

    hdf5_path: path to exams.hdf5
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, "r") as f:
            self._len = f["tracings"].shape[0]
        self._hdf5_handle: h5py.File | None = None

    def _get_hdf5(self) -> h5py.File:
        if self._hdf5_handle is None:
            self._hdf5_handle = h5py.File(self.hdf5_path, "r")
        return self._hdf5_handle

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):
        signal = self._get_hdf5()["tracings"][idx]  # [4096, 12], float64
        signal = preprocess_signal(signal, fs=400.0, sig_names=SAMITROP_LEAD_ORDER)
        return torch.from_numpy(signal), torch.tensor(1.0, dtype=torch.float32)


class PTBXLDataset(Dataset):
    """
    PTB-XL dataset for Chagas-negative training and evaluation.

    All records are treated as Chagas-negative (label=0.0). Uses the official
    strat_fold split from the PTB-XL paper (Wagner et al. 2020) to respect
    patient grouping: folds 1–8 = train, 9 = val, 10 = test.

    Records are read from records100/ (100 Hz WFDB) and resampled to 400 Hz
    by preprocess_ecg.

    ptbxl_root: directory containing ptbxl_database.csv and records100/
    split:      'train' | 'val' | 'test'
    """

    _FOLD_MAP: dict[str, tuple[int, ...]] = {
        "train": tuple(range(1, 9)),
        "val":   (9,),
        "test":  (10,),
    }

    def __init__(self, ptbxl_root: str, split: str = "train"):
        if split not in self._FOLD_MAP:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        self.ptbxl_root = ptbxl_root
        db = pd.read_csv(os.path.join(ptbxl_root, "ptbxl_database.csv"))
        folds = self._FOLD_MAP[split]
        self.df = db[db["strat_fold"].isin(folds)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]["filename_lr"]  # e.g. records100/00000/00001_lr
        full_path = os.path.join(self.ptbxl_root, rel_path)
        signal = preprocess_ecg(full_path)  # [12, TARGET_LEN], 100→400 Hz resampling in preprocess_ecg
        return torch.from_numpy(signal), torch.tensor(0.0, dtype=torch.float32)
