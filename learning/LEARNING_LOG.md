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
