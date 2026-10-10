"""
Length diagnostic (notes/decisions/0006): does performance depend on the recording's original
length, which in CODE-15% tracks the label (7 s recordings: 93% of positives, 60% of negatives)?

Reads val_preds.csv written by train.py (columns include chagas, prob, n_real) and reports, per
original-length group and overall: n, positives, prevalence, AUROC, and TPR / flagged fraction at
the single GLOBAL top-5% cutoff (same convention as equity.py). No bootstrap: this is a quick
look, not a reported table.

Usage:
    python src/length_diagnostic.py lightning_logs/<run>/val_preds.csv
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from equity import _group_metrics, flag_top_k
from preprocess import TARGET_LEN

# Original unpadded length: shorter than the crop (still padded), a 7 s recording (~2934 real
# samples), or a 10 s recording (4096 stored samples).
GROUPS = (
    (f"<{TARGET_LEN} (padded)", lambda n: n < TARGET_LEN),
    (f"{TARGET_LEN}-4095 (7 s)", lambda n: (n >= TARGET_LEN) & (n < 4096)),
    ("4096 (10 s)", lambda n: n >= 4096),
)


def length_report(df: pd.DataFrame) -> pd.DataFrame:
    y = df["chagas"].to_numpy().astype(int)
    p = df["prob"].to_numpy(dtype=float)
    n_real = df["n_real"].to_numpy()
    flagged = flag_top_k(p)
    rows = []
    for name, f in (("Overall", lambda n: np.ones(len(n), dtype=bool)), *GROUPS):
        m = f(n_real)
        if m.sum() == 0:
            continue
        r = _group_metrics(y[m], p[m], flagged[m])
        rows.append({"group": name, "n": int(m.sum()), "n_pos": int(y[m].sum()),
                     "prevalence": y[m].mean(), **r})
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    df = pd.read_csv(sys.argv[1])
    rep = length_report(df)
    print(f"{'group':<20}{'n':>8}{'pos':>6}{'prev%':>7}{'AUROC':>8}{'TPR@top5%':>11}{'flagged%':>10}")
    for _, r in rep.iterrows():
        print(f"{r['group']:<20}{r['n']:>8}{r['n_pos']:>6}{100 * r['prevalence']:>7.2f}"
              f"{r['auroc']:>8.3f}{r['tpr_top5']:>11.3f}{100 * r['flagged_frac']:>10.2f}")


if __name__ == "__main__":
    main()
