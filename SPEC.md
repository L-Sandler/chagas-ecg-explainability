# Chagas Disease Detection from 12-Lead ECG — Project Spec

## Challenge

**The George B. Moody PhysioNet Challenge 2025**
*Detection of Chagas Disease from the ECG*

The challenge is formally over (deadline August 2025), but all datasets and evaluation code are public. Framing this project against a real published competition with a leaderboard and scored results is a direct signal to any research lab that you're working on something grounded and benchmarkable.

---

## Problem Statement

Chagas disease is a parasitic infection (*Trypanosoma cruzi*) affecting ~8 million people in Latin America. Most cases are asymptomatic until serious cardiac complications develop. The chronic form causes **Chagas cardiomyopathy**, which produces distinct electrical abnormalities detectable on ECG — right bundle branch block (RBBB), left anterior fascicular block (LAFB), AV block, and arrhythmias.

The bottleneck: serological testing (the confirmatory test) has limited capacity in endemic regions. ECG is cheap, non-invasive, and widely available. The ML task is to **rank patients by Chagas probability so that limited testing capacity is directed at the highest-risk individuals**.

This is a triage problem, not just a classification problem. The evaluation metric reflects this directly.

---

## Evaluation Metric

**Challenge score: True Positive Rate (TPR) at the top 5% of risk-ranked patients.**

This is not AUROC or F1. The model ranks all patients by predicted probability, the top 5% are "sent for testing," and the score is the fraction of true Chagas-positive patients captured within that 5%. A perfect ranker captures all positives in the top 5%; a random ranker captures ~5%.

The winning team scored 0.323 on the test set. A well-implemented baseline should target 0.2+.

**Also report:** AUROC, AUPRC (for comparability with published baselines).

---

## Datasets

| Dataset | Records | Sampling Rate | Leads | Labels | Split |
|---|---|---|---|---|---|
| **CODE-15%** | 335,621 | 400 Hz | 12 | Self-reported (weak labels) | Training |
| **SaMi-Trop** | 1,631 | 400 Hz | 12 | Serologically confirmed | Training |
| **PTB-XL** | 21,799 | 500 Hz | 12 | Geographic assumption (negative) | Training |

**Total training data: ~378k ECG recordings.** Hidden validation and test sets (REDS-II, SaMi-Trop 3, ELSA-Brasil) were used by the challenge organizers — these represent different population distributions and are the source of the domain shift problem (see below).

**Access:** All training datasets are publicly available through PhysioNet. No credentialing required for CODE-15%.

---

## Architecture

### 1. Signal Preprocessing Pipeline

Raw ECG → usable input tensor:

```
Raw 12-lead ECG (variable length, variable Hz)
    → Resample to uniform rate (400 Hz)
    → Bandpass filter (0.5–40 Hz) — removes baseline wander and high-freq noise
    → Normalize per-lead (zero mean, unit variance)
    → Fixed-length window (10 seconds = 4,000 samples at 400 Hz)
    → Tensor shape: [batch, 12 leads, 4000 time steps]
```

Preprocessing is clinically meaningful — each step has a reason grounded in ECG signal characteristics, not just ML convention.

### 2. Model — 1D CNN with Attention

**Primary: 1D ResNet with SE (Squeeze-and-Excitation) blocks**

Standard 2D ResNet adapted for 1D temporal signals. SE blocks add channel-wise attention — the model learns which ECG leads matter more for a given prediction. This is the architecture class used by the original CODE-15% paper (which achieved cardiologist-level performance on arrhythmia detection from the same dataset).

```
Input [B, 12, 4000]
    → 1D Conv blocks (residual connections)
    → SE block (channel attention over 12 leads)
    → Global average pooling
    → Dense → sigmoid → Chagas probability
```

**Alternate: ECG Transformer (PatchTST-style)**
- Segment each lead into patches, treat as token sequence
- Multi-head attention over temporal patches
- More parameters, better at capturing long-range waveform dependencies
- Use as a comparison model, not primary

**Why not a plain MLP or generic LSTM:** 1D CNNs are the established baseline for raw ECG. They respect the temporal structure of the signal and have published benchmarks on this exact dataset. Starting here is defensible in a research setting.

### 3. Training Setup

- Framework: PyTorch + PyTorch Lightning
- Experiment tracking: Weights & Biases
- Loss: Binary cross-entropy with class weighting (Chagas is a rare positive class — prevalence ~3% in CODE-15%)
- Optimizer: AdamW with cosine LR schedule
- Mixed precision (FP16) to reduce GPU memory and training time
- Batch size: 256–512

**Class imbalance strategy:** CODE-15% labels are weak (self-reported Chagas). SaMi-Trop labels are serologically confirmed. Weight confirmed-label samples more heavily in the loss, or oversample SaMi-Trop during training.

### 4. Domain Shift — the Key Technical Challenge

The winning team's score dropped 64% between the validation population (Brazilian public health system patients) and the test population (ELSA-Brasil, a different Brazilian cohort). This isn't a bug — it reflects a real problem in clinical ML: models trained on one hospital's population don't always generalize.

Approaches to mitigate this (pick one to implement and analyze):
- **Domain-adversarial training** — add a domain classifier head that tries to identify data source; train the feature extractor to fool it
- **Data augmentation** — simulate domain shift via signal-level augmentation (amplitude scaling, noise injection, lead dropout)
- **Post-hoc calibration** — Platt scaling or temperature scaling to improve probability calibration across domains

This is a strong discussion point for any research lab conversation.

### 5. Explainability Layer

Since the model is a 1D CNN, standard 2D methods adapt directly:

**Grad-CAM (1D):** Compute gradients of the predicted Chagas probability with respect to the last convolutional feature map. The result is a saliency map over the 4,000-sample waveform — highlighting which time regions drove the prediction.

**SE block attention:** The SE block outputs per-lead weights — directly interpretable as "which of the 12 leads mattered most for this patient."

**Clinical interpretability:** A Chagas prediction driven by activity in leads V1–V2 during the QRS complex is consistent with RBBB, a known Chagas marker. Being able to narrate this connection is what turns a model into a research contribution.

---

## Evaluation Framework

| Metric | Purpose |
|---|---|
| **TPR@5%** (challenge score) | Primary — matches competition metric |
| **AUROC** | Comparability with published baselines |
| **AUPRC** | More informative than AUROC under class imbalance |
| **Calibration (ECE)** | How well probabilities reflect true risk |
| **Domain shift delta** | Score difference between CODE-15% val vs held-out distribution |
| **Saliency map alignment** | Do Grad-CAM peaks align with known Chagas ECG regions (QRS, RBBB)? |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Signal processing | `neurokit2`, `scipy`, `wfdb` (PhysioNet reader) |
| Model | PyTorch, custom 1D ResNet |
| Training | PyTorch Lightning |
| Experiment tracking | Weights & Biases |
| Explainability | `captum` (Grad-CAM 1D), manual SE weight extraction |
| Evaluation | `sklearn`, custom TPR@5% scorer |
| Compute | Kaggle free (prototyping) → RunPod community (~$2/hr, training) |

---

## Compute Budget

| Phase | Task | Platform | Estimated Cost |
|---|---|---|---|
| Prototyping | Preprocessing pipeline, small subset | Kaggle free | $0 |
| Baseline training | 1D ResNet on full CODE-15% | RunPod A100 ~4–6 hrs | $8–12 |
| Experiments | Architecture comparison, augmentation | RunPod A100 ~8–12 hrs | $16–24 |
| Final runs | Calibration, explainability | RunPod A100 ~4 hrs | $8 |
| **Total** | | | **~$35–50** |

Mixed-precision (FP16) cuts memory and training time by ~40% — use it from the start.

---

## Build Phases

| Phase | Weeks | Deliverable |
|---|---|---|
| **1. Setup + Data** | 1–2 | PhysioNet account, wfdb pipeline reads all three datasets, preprocessing outputs clean tensors |
| **2. Baseline Model** | 3–4 | 1D ResNet trains and converges, AUROC > 0.75, TPR@5% logged |
| **3. Model Improvement** | 5–6 | SE attention, class weighting, augmentation; TPR@5% improvement over baseline |
| **4. Domain Shift** | 7 | At least one mitigation strategy implemented and measured |
| **5. Explainability** | 8 | Grad-CAM visualization over waveform, SE lead weights, case study on 3–5 patients |
| **6. Polish** | 9+ | W&B report, benchmark comparison table, write-up for portfolio |

---

## Success Criteria

- TPR@5% ≥ 0.20 on CODE-15% held-out validation
- AUROC ≥ 0.80
- Grad-CAM peaks visible and narrated for at least 3 representative cases
- Domain shift delta measured and discussed
- Code reproducible from a single `train.py` command

---

## Why This Project for a Research Lab

- Built on a real, published, peer-reviewed competition — your results are directly comparable to 41 submitted teams
- Chagas disease is a global health problem with genuine unmet need
- Domain shift analysis shows awareness of the gap between benchmark ML and clinical deployment
- Explainability layer shows you're thinking about clinician trust, not just accuracy
- ECG signal processing is a transferable skill to any cardiology, wearables, or physiological monitoring lab
