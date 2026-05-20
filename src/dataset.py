"""
Datasets for CODE-15% (HDF5), SaMi-Trop (HDF5), and PTB-XL (WFDB).

CODE-15% HDF5 layout (each exams_partN.hdf5):
    exam_id:  int64[N]          — exam IDs
    tracings: float32[N, T, 12] — raw ECG signals (T ≈ 4096, 12 leads)

Labels come from a separate CSV with columns: exam_id, patient_id, chagas (bool).

CODE-15% spans 18 part files (exams_part0.hdf5 … exams_part17.hdf5), each with
~20,000 records. Pass all available parts for full training; part0 alone is fine
for local prototyping.

Splitting strategy: patient-level stratified 70/15/15 (train/val/test).
A fixed seed (42) is baked in so splits are reproducible across runs without
a saved manifest. No patient appears in more than one split.

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
from sklearn.model_selection import train_test_split

from preprocess import preprocess_ecg, preprocess_signal, CODE15_LEAD_ORDER, SAMITROP_LEAD_ORDER

_SPLIT_SEED = 42


def _patient_level_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Stratified 70/15/15 split at the patient level.

    Stratification is by whether a patient has any chagas-positive exam so
    that the rare-positive class is proportionally represented in every split.
    Returns {'train': df, 'val': df, 'test': df} with zero patient overlap.
    """
    patients = (
        df.groupby("patient_id")
        .agg(chagas_any=("chagas", "any"))
        .reset_index()
    )

    def _split(pids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train, tmp = train_test_split(pids, test_size=0.30, random_state=_SPLIT_SEED)
        val, test = train_test_split(tmp, test_size=0.50, random_state=_SPLIT_SEED)
        return train, val, test

    pos_p = patients[patients["chagas_any"]]["patient_id"].values
    neg_p = patients[~patients["chagas_any"]]["patient_id"].values

    pos_train, pos_val, pos_test = _split(pos_p)
    neg_train, neg_val, neg_test = _split(neg_p)

    splits = {
        "train": set(pos_train) | set(neg_train),
        "val":   set(pos_val)   | set(neg_val),
        "test":  set(pos_test)  | set(neg_test),
    }
    return {
        name: df[df["patient_id"].isin(pids)].reset_index(drop=True)
        for name, pids in splits.items()
    }


class Code15Dataset(Dataset):
    """
    CODE-15% dataset backed by one or more HDF5 part files.

    hdf5_paths:  path or list of paths to exams_partN.hdf5 files.
                 Pass all 18 parts for full training; part0 alone for prototyping.
    labels_csv:  path to CSV with columns 'exam_id', 'patient_id', 'chagas'
    split:       'train' | 'val' | 'test'
    max_samples: cap total samples (useful for CPU smoke tests)

    Splitting is patient-level stratified 70/15/15 with seed 42.
    No patient appears in more than one split.
    """

    def __init__(
        self,
        hdf5_paths: str | list[str],
        labels_csv: str,
        split: str = "train",
        max_samples: int | None = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self.hdf5_paths = list(hdf5_paths)

        # Build position lookup: exam_id → (file_index, row_in_file)
        self._id_to_pos: dict[int, tuple[int, int]] = {}
        for file_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                for row, eid in enumerate(f["exam_id"][:]):
                    self._id_to_pos[int(eid)] = (file_idx, row)

        labels = pd.read_csv(labels_csv)
        labels = labels[labels["exam_id"].isin(self._id_to_pos)].reset_index(drop=True)

        all_splits = _patient_level_splits(labels)
        df = all_splits[split]

        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.df = df.reset_index(drop=True)
        self._hdf5_handles: dict[int, h5py.File] = {}  # opened lazily per-worker

    def _get_hdf5(self, file_idx: int) -> h5py.File:
        if file_idx not in self._hdf5_handles:
            self._hdf5_handles[file_idx] = h5py.File(self.hdf5_paths[file_idx], "r")
        return self._hdf5_handles[file_idx]

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
        signal = preprocess_signal(signal, fs=400.0, sig_names=CODE15_LEAD_ORDER)  # [12, 4000]

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
        signal = preprocess_ecg(full_path)  # [12, 4000], 100→400 Hz resampling in preprocess_ecg
        return torch.from_numpy(signal), torch.tensor(0.0, dtype=torch.float32)
