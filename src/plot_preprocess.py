"""Quick visual check: raw vs preprocessed for a single record."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from preprocess import preprocess_ecg, LEAD_ORDER

RECORD = "data/code15/wfdb/1000010"
LEAD_IDX = 0  # Lead I


def main():
    record = wfdb.rdrecord(RECORD)
    raw = record.p_signal[:, 0]  # Lead I, raw
    fs = record.fs
    t_raw = np.arange(len(raw)) / fs

    processed = preprocess_ecg(RECORD)  # [12, 4000]
    proc = processed[LEAD_IDX]
    t_proc = np.arange(len(proc)) / 400

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=False)

    axes[0].plot(t_raw, raw, lw=0.6, color="steelblue")
    axes[0].set_title(f"Raw — {LEAD_ORDER[LEAD_IDX]}")
    axes[0].set_ylabel("mV")
    axes[0].set_xlabel("Time (s)")

    axes[1].plot(t_proc, proc, lw=0.6, color="darkorange")
    axes[1].set_title(f"Preprocessed — {LEAD_ORDER[LEAD_IDX]} (filtered, normalized)")
    axes[1].set_ylabel("z-score")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    out = "preprocessing_check.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
