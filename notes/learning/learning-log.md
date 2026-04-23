# Learning Log

---

## Phase 1 — Chagas Disease and Clinical ECG Basics

**Done signal question:** What does RBBB look like on an ECG, why does Chagas cause it, and why does that make ECG useful for screening?

**Answer:** RBBB shows as a wider-than-normal QRS complex on the ECG. Chagas causes it by scarring the right bundle branch — the fast conduction pathway to the right ventricle — so the signal travels slowly through muscle tissue instead, arriving late. ECG is useful for screening because it captures this electrical delay cheaply and non-invasively, before the patient has any symptoms.

---

## Phase 2 — ECG Signal Processing (Concepts)

**What I learned:** The four preprocessing steps for ECG data — bandpass filtering (0.5–40 Hz), resampling to 400 Hz, per-lead normalization — and why order matters (filter before resample to avoid aliasing artifacts corrupting the signal before cleaning).

**Key distinction made:** Heavy breathing causes two separate artifacts through two different mechanisms: mechanical chest movement → baseline wander (low frequency, <0.5 Hz); respiratory muscle contraction → EMG noise (high frequency, >40 Hz). The bandpass filter handles both ends simultaneously.

**Open question raised:** Whether the 40 Hz upper cutoff discards diagnostically relevant signal. Accepted as convention for now — Chagas-relevant features (RBBB, conduction delays) live well within 0–40 Hz. Logged in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

**Done signal status:** Complete — `preprocess_ecg` implemented in `src/preprocess.py`. Verified visually with `src/plot_preprocess.py` on a CODE-15% record. Output is `[12, 4000]` float32, z-scored per lead, baseline flat.

---

## Phase 3 — 1D Deep Learning for ECG (Concepts)

**Why 1D ResNet:** Has built-in understanding of spatial/temporal relationships unlike transformers which need to learn them. Current SOTA papers use this architecture for ECG Chagas analysis, giving a good benchmark. LSTMs are similar in functionality but generally more expensive to train.

**SE blocks:** An MLP that produces scalar importance weights for each feature channel. Sits before the residual add (not before the projection layer — those are different things). Global average pool collapses the time dimension first, then the MLP outputs per-channel weights that scale the feature map.

**ECG Transformer:** Uses attention to classify. Input is patched — 4000 samples → 80 patches of 50 samples — to make O(n²) attention cost tractable. Multi-headed attention lets the model learn how tokens attend to each other from multiple perspectives simultaneously.

**Class imbalance:** Only ~3% positive in CODE-15%, so a model predicting all-negative gets 97% accuracy with no clinical value. Use AUROC (ranking quality across all thresholds) and TPR@5% (fraction of true positives captured in the top 5% of risk-ranked patients).

**Done signal status:** Conceptual phase complete. Implementation (4-block 1D ResNet, training on CODE-15% subset, AUROC > 0.6) — pending.

---
