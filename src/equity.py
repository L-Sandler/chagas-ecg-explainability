"""
Subgroup performance report: sex and age band (notes/decisions/0005).

Reports, for the whole evaluation split and for each subgroup:
  n, n_pos, prevalence, AUROC, AUPRC, TPR at the GLOBAL top-5% threshold, and the fraction of
  the group flagged at that threshold, each with a 95% interval from a bootstrap that
  resamples PATIENTS (exams of one patient stay together).

"Global top-5%": the flagged set is the top 5% of ALL exams in the evaluated split by
predicted probability (the challenge metric), then each group is scored against that single
cutoff. The cutoff is recomputed inside every bootstrap resample. Groups with fewer than
MIN_POS positives are marked `small` and should not be interpreted.

Usage (development and audit use val; locked-final is final evaluation only):
    python src/equity.py --checkpoint <ckpt> --hdf5 data/code15/exams_part*.hdf5 \
        --labels data/code15/code15_chagas_labels.csv --split val
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import AGE_BAND_LABELS, SPLIT_NAMES, Code15Dataset

TOP_FRAC = 0.05
MIN_POS = 50
METRICS = ("auroc", "auprc", "tpr_top5", "flagged_frac")


def group_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean row masks: Overall, sex (2), age bands (6). df needs is_male and band columns."""
    sex = df["is_male"].astype(int).to_numpy()
    band = df["band"].to_numpy()
    masks = {"Overall": np.ones(len(df), dtype=bool),
             "Sex: Female": sex == 0,
             "Sex: Male": sex == 1}
    for i, label in enumerate(AGE_BAND_LABELS):
        masks[f"Age: {label}"] = band == i
    return masks


def flag_top_k(probs: np.ndarray, k_frac: float = TOP_FRAC) -> np.ndarray:
    """Boolean mask of the top ceil(k_frac * N) exams by probability (the global cutoff)."""
    k = int(np.ceil(k_frac * len(probs)))
    flagged = np.zeros(len(probs), dtype=bool)
    flagged[np.argsort(-probs, kind="stable")[:k]] = True
    return flagged


def _group_metrics(y: np.ndarray, p: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    n, n_pos = len(y), int(y.sum())
    nan = float("nan")
    return {
        "auroc": roc_auc_score(y, p) if 0 < n_pos < n else nan,
        "auprc": average_precision_score(y, p) if n_pos > 0 else nan,
        "tpr_top5": float(flagged[y == 1].mean()) if n_pos > 0 else nan,
        "flagged_frac": float(flagged.mean()) if n > 0 else nan,
    }


def subgroup_report(df: pd.DataFrame, probs: np.ndarray, n_boot: int = 1000,
                    seed: int = 0) -> pd.DataFrame:
    """
    df:    one row per evaluated exam, aligned with probs. Columns: patient_id, chagas,
           is_male, band (as produced by Code15Dataset.df).
    probs: predicted probabilities, same order as df.
    """
    y = df["chagas"].to_numpy().astype(int)
    p = np.asarray(probs, dtype=float)
    masks = group_masks(df)
    flagged = flag_top_k(p)

    rows = {}
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        rows[name] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                      **_group_metrics(y[m], p[m], flagged[m])}
        rows[name]["prevalence"] = rows[name]["n_pos"] / rows[name]["n"]

    # Patient-level bootstrap via integer weights: resampling patients with replacement is
    # the same as repeating each of their exams by how often the patient was drawn.
    pcodes, uniques = pd.factorize(df["patient_id"])
    n_patients = len(uniques)
    rng = np.random.default_rng(seed)
    draws = {name: {k: [] for k in METRICS} for name in rows}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)
        idx = np.repeat(np.arange(len(y)), counts[pcodes])
        yb, pb = y[idx], p[idx]
        fb = flag_top_k(pb)
        for name, m in masks.items():
            if name not in rows:
                continue
            mb = m[idx]
            res = _group_metrics(yb[mb], pb[mb], fb[mb])
            for k in METRICS:
                draws[name][k].append(res[k])

    out = []
    for name, r in rows.items():
        rec = {"group": name, "n": r["n"], "n_pos": r["n_pos"], "prevalence": r["prevalence"]}
        for k in METRICS:
            rec[k] = r[k]
            vals = np.asarray(draws[name][k], dtype=float)
            if n_boot and np.isfinite(vals).any():
                rec[f"{k}_lo"], rec[f"{k}_hi"] = np.nanpercentile(vals, [2.5, 97.5])
            else:
                rec[f"{k}_lo"] = rec[f"{k}_hi"] = float("nan")
        rec["small"] = r["n_pos"] < MIN_POS
        out.append(rec)
    return pd.DataFrame(out)


def format_report(rep: pd.DataFrame) -> str:
    def ci(r, k):
        return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
    lines = [f"{'group':<14}{'n':>8}{'pos':>6}{'prev%':>7}  {'AUROC':<22}{'TPR@top5% (global)':<24}"
             f"{'flagged%':<10}"]
    for _, r in rep.iterrows():
        mark = "  <- too few positives to interpret" if r["small"] else ""
        lines.append(f"{r['group']:<14}{r['n']:>8}{r['n_pos']:>6}{100 * r['prevalence']:>7.2f}  "
                     f"{ci(r, 'auroc'):<22}{ci(r, 'tpr_top5'):<24}{100 * r['flagged_frac']:<10.2f}{mark}")
    return "\n".join(lines)


@torch.no_grad()
def predict_probs(module, ds: Code15Dataset, batch_size: int = 256, num_workers: int = 0,
                  device: str = "cpu") -> np.ndarray:
    """Probabilities in ds.df row order."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    module.model.to(device).eval()
    out = [torch.sigmoid(module.model(x.to(device))).cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def main() -> None:
    from train import ECGLightningModule  # heavy import kept out of module scope

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preds", default=None,
                    help="val_preds.csv written by train.py (notes/decisions/0007). Skips the "
                         "checkpoint and HDF5 files; the split is taken from the file.")
    ap.add_argument("--hdf5", nargs="+", default=["data/code15/exams_part0.hdf5"])
    ap.add_argument("--labels", default="data/code15/code15_chagas_labels.csv")
    ap.add_argument("--exams-csv", default=None)
    ap.add_argument("--split", default="val", choices=list(SPLIT_NAMES))
    ap.add_argument("--unlock-final", action="store_true",
                    help="Open the locked-final split. Final evaluation only.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="reports/equity")
    args = ap.parse_args()

    if args.preds:
        df = pd.read_csv(args.preds)
        df["is_male"] = df["is_male"].astype(bool)
        probs = df["prob"].to_numpy()
        n_exams, source = len(df), args.preds
    else:
        if not args.checkpoint:
            ap.error("give --checkpoint or --preds")
        ds = Code15Dataset(args.hdf5, args.labels, split=args.split,
                           exams_csv=args.exams_csv, unlock_final=args.unlock_final)
        module = ECGLightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
        probs = predict_probs(module, ds, args.batch_size, args.num_workers, args.device)
        df, n_exams, source = ds.df, len(ds), args.checkpoint

    rep = subgroup_report(df, probs, n_boot=args.n_boot)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.split}")
    rep.to_csv(stem + "_subgroups.csv", index=False)
    pd.DataFrame({"exam_id": df["exam_id"], "prob": probs}).to_csv(stem + "_preds.csv", index=False)
    print(f"source: {source}\nsplit: {args.split}  exams: {n_exams}  "
          f"bootstrap: {args.n_boot} patient-level resamples\n")
    print(format_report(rep))
    print(f"\nwrote {stem}_subgroups.csv and {stem}_preds.csv")


if __name__ == "__main__":
    main()
