"""
Datasets for CODE-15% (HDF5) and SaMi-Trop / PTB-XL (WFDB).

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
"""

import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from preprocess import preprocess_ecg, preprocess_signal, CODE15_LEAD_ORDER

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


class WFDBDataset(Dataset):
    """
    Generic WFDB dataset (SaMi-Trop or PTB-XL).

    records_csv: CSV with columns 'path' (WFDB record path, no extension) and 'chagas'
    split:       'train' | 'val' | 'test' — index-based 70/15/15
    """

    def __init__(self, records_csv: str, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        df = pd.read_csv(records_csv)
        n = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        if split == "train":
            self.df = df.iloc[:train_end].reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.df = df.iloc[val_end:].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = preprocess_ecg(row["path"])  # [12, 4000] float32
        label = float(row["chagas"])
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.float32)
