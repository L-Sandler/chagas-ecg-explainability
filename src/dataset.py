"""
Datasets for CODE-15% (HDF5) and SaMi-Trop (WFDB).

CODE-15% HDF5 layout (each exams_partN.hdf5):
    exam_id:  int64[N]          — exam IDs
    tracings: float32[N, T, 12] — raw ECG signals (T ≈ 4096, 12 leads)

Labels come from a separate CSV with columns: exam_id, chagas (bool).
"""

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

from preprocess import preprocess_ecg, TARGET_LEN


class Code15Dataset(Dataset):
    """
    CODE-15% dataset backed by a single HDF5 part file.

    hdf5_path:  path to exams_partN.hdf5
    labels_csv: path to CSV with columns 'exam_id' and 'chagas'
    split:      'train' | 'val' (90/10 by index after joining)
    max_samples: cap total samples (useful for CPU smoke tests)
    """

    def __init__(
        self,
        hdf5_path: str,
        labels_csv: str,
        split: str = "train",
        max_samples: int | None = None,
    ):
        with h5py.File(hdf5_path, "r") as f:
            hdf5_ids = f["exam_id"][:]         # int64[N]

        # Build position lookup: exam_id → row index in HDF5
        self._id_to_pos = {int(eid): i for i, eid in enumerate(hdf5_ids)}

        labels = pd.read_csv(labels_csv)
        labels = labels[labels["exam_id"].isin(self._id_to_pos)].reset_index(drop=True)

        n = len(labels)
        cutoff = int(n * 0.9)
        if split == "train":
            df = labels.iloc[:cutoff]
        else:
            df = labels.iloc[cutoff:]

        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self._hdf5 = None  # opened lazily per-worker

    def _get_hdf5(self):
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.hdf5_path, "r")
        return self._hdf5

    def pos_weight(self) -> float:
        """Ratio of negatives to positives — pass to BCEWithLogitsLoss."""
        n_pos = self.df["chagas"].sum()
        n_neg = len(self.df) - n_pos
        return float(n_neg) / max(float(n_pos), 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pos = self._id_to_pos[int(row["exam_id"])]

        f = self._get_hdf5()
        signal = f["tracings"][pos]  # [T, 12], float32

        signal = signal.T  # [12, T]
        signal = np.clip(signal, -5.0, 5.0)

        # Fix length to TARGET_LEN (4000)
        if signal.shape[1] >= TARGET_LEN:
            signal = signal[:, :TARGET_LEN]
        else:
            pad = TARGET_LEN - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)))

        # Per-lead z-score
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std

        label = float(row["chagas"])
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.float32)


class WFDBDataset(Dataset):
    """
    Generic WFDB dataset (SaMi-Trop or PTB-XL).

    records_csv: CSV with columns 'path' (WFDB record path, no extension) and 'chagas'
    """

    def __init__(self, records_csv: str, split: str = "train"):
        df = pd.read_csv(records_csv)
        n = len(df)
        cutoff = int(n * 0.9)
        self.df = (df.iloc[:cutoff] if split == "train" else df.iloc[cutoff:]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = preprocess_ecg(row["path"])  # [12, 4000] float32
        label = float(row["chagas"])
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.float32)
