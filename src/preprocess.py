import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample

TARGET_FS = 400
TARGET_LEN = 4000  # 10 seconds at 400 Hz
LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# HDF5-based datasets store leads in this order (confirmed from dataset schemas).
# Both match LEAD_ORDER, so reordering is a no-op — but passing explicitly makes
# the assumption auditable rather than implicit.
CODE15_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMITROP_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    b, a = butter(4, [0.5, 40.0], btype="band", fs=fs)
    return filtfilt(b, a, signal, axis=0)


def _resample_to_target(signal: np.ndarray, fs: float) -> np.ndarray:
    if fs == TARGET_FS:
        return signal
    n_target = int(round(signal.shape[0] * TARGET_FS / fs))
    return resample(signal, n_target, axis=0)


def _fix_length(signal: np.ndarray) -> np.ndarray:
    n = signal.shape[0]
    if n >= TARGET_LEN:
        return signal[:TARGET_LEN]
    pad = TARGET_LEN - n
    return np.pad(signal, ((0, pad), (0, 0)))


def _reorder_leads(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
    names_upper = [s.upper() for s in sig_names]
    indices = [names_upper.index(lead) for lead in LEAD_ORDER]
    return signal[:, indices]


def _normalize(signal: np.ndarray) -> np.ndarray:
    # Clip artifact amplitudes before normalizing
    signal = np.clip(signal, -5.0, 5.0)
    mean = signal.mean(axis=0)
    std = signal.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid divide-by-zero on flat/missing leads
    return (signal - mean) / std


def preprocess_signal(signal: np.ndarray, fs: float, sig_names: list[str]) -> np.ndarray:
    """
    Shared preprocessing pipeline for all datasets.

    Args:
        signal:    [T, 12] array (any float dtype)
        fs:        sampling frequency of the input signal in Hz
        sig_names: lead names in the order they appear in signal columns

    Returns:
        [12, 4000] float32 — reordered, bandpassed, resampled, length-fixed, z-scored
    """
    signal = signal.astype(np.float64)
    signal = _reorder_leads(signal, sig_names)
    signal = _bandpass(signal, fs)
    signal = _resample_to_target(signal, fs)
    signal = _fix_length(signal)
    signal = _normalize(signal)
    return signal.T.astype(np.float32)  # [12, 4000]


def preprocess_ecg(path: str) -> np.ndarray:
    """Read a WFDB record and return a preprocessed [12, 4000] float32 array."""
    record = wfdb.rdrecord(path)
    signal = record.p_signal.copy()  # [T, 12], float64
    return preprocess_signal(signal, record.fs, record.sig_name)
