# Learning Plan

Assumes: solid ML/DL + PyTorch background. Gaps: ECG signal processing, clinical cardiology basics, 1D deep learning architectures, PhysioNet tooling.

Each phase has a **goal** and a **done signal** — a concrete thing you should be able to do or explain without looking it up.

---

## Teaching instructions for Claude

When explaining any concept in this plan — clinical, signal-processing, architectural, or statistical — do not rely on training data alone. At the start of a teaching session for a given phase or topic:

1. **Search for current academic sources** using web search. Prioritise, in order: peer-reviewed journal articles (PubMed, Nature, Lancet, JACC, NEJM, IEEE, etc.), official dataset or software papers, recent review articles (within the last 5 years where possible), and authoritative clinical bodies (WHO, AHA, ESC) for clinical facts.
2. **Prefer recent sources.** For clinical cardiology and epidemiology, search for reviews from the past 5 years before falling back to older foundational papers. For ML methods, the original method paper plus any recent benchmark that uses it on ECG.
3. **Cite every substantive claim** inline as you teach — author, year, and journal/venue at minimum. If a claim cannot be traced to a specific source found via search, flag it explicitly as unverified rather than stating it as fact.
4. **Show the source link** when introducing a new paper so I can verify it myself.
5. **Follow the teaching-style preference**: one concept at a time, short messages, end each with a single understanding-check question before moving on.

If a search returns nothing recent or relevant, say so rather than filling the gap with training-data recall.

---

## Phase 1 — Chagas Disease and Clinical ECG Basics
**Time: 2–3 days**

### What is Chagas disease

- Caused by *Trypanosoma cruzi*, a parasite transmitted by triatomine insects
- Concentrated in Latin America, millions infected globally
- Most chronic cases are asymptomatic until cardiac damage develops
- **Chagas cardiomyopathy** is the main lethal complication — it disrupts the heart's electrical conduction system
- No widely available vaccine; serological testing is confirmatory but capacity-limited

### How the heart's electrical system works (enough to read an ECG)

The ECG records electrical activity as the heart contracts. Each beat produces a waveform with named components:

```
P wave       → atria depolarize (contract)
PR interval  → conduction delay at AV node
QRS complex  → ventricles depolarize (main pump contraction)
T wave       → ventricles repolarize (reset)
```

A **12-lead ECG** records this from 12 different electrode positions on the body, giving 12 different "views" of the same electrical event. Each lead emphasizes different parts of the heart.

### ECG abnormalities specific to Chagas

These are the patterns your model is implicitly learning to detect:

| Abnormality | What it means | ECG signature |
|---|---|---|
| **RBBB** (Right Bundle Branch Block) | Delay in right ventricle conduction | Wide QRS, RSR' pattern in V1 |
| **LAFB** (Left Anterior Fascicular Block) | Damage to left conduction pathway | Left axis deviation |
| **1st degree AV block** | Slow AV node conduction | Prolonged PR interval |
| **Ventricular arrhythmias** | Abnormal beats originating in ventricles | Premature ventricular complexes |

You don't need to diagnose these by eye — but you need to know they exist so you can interpret Grad-CAM results. If your saliency map highlights the QRS region in V1, that's consistent with RBBB detection, which is a known Chagas marker.

### Done signal
Explain in one paragraph: what RBBB looks like on an ECG, why Chagas disease causes it, and why that makes ECG useful for Chagas screening.

---

## Phase 2 — ECG Signal Processing
**Time: 3–4 days**

This is the most important phase to get right before touching the model. Garbage preprocessing = garbage model.

### What an ECG looks like as data

A 12-lead ECG at 400 Hz, 10 seconds long:
- Shape: `[12, 4000]` — 12 leads × 4000 time steps
- Values: millivolts (mV), typically in the range -2.0 to +2.0
- Stored in PhysioNet as WFDB format (`.dat` + `.hea` header files)

### Key preprocessing steps

**1. Reading WFDB files**
```python
import wfdb
record = wfdb.rdrecord('path/to/record')
signal = record.p_signal  # shape: [n_samples, n_leads]
```

**2. Resampling**
CODE-15% is 400 Hz, PTB-XL is 500 Hz. Resample everything to 400 Hz using `scipy.signal.resample`.

**3. Bandpass filtering (0.5–40 Hz)**
- Below 0.5 Hz: baseline wander (slow drift from breathing or electrode movement) — remove it
- Above 40 Hz: high-frequency noise, muscle artifacts — remove it
- Use a Butterworth filter: `scipy.signal.butter` + `scipy.signal.filtfilt`

**4. Normalization**
Normalize per-lead to zero mean, unit variance. This makes training stable across different patients and recording equipment.

**5. Windowing**
Clip or pad to a fixed 10-second window (4,000 samples at 400 Hz). Most records are already 10 seconds; handle edge cases with zero-padding.

**6. Lead ordering**
Ensure consistent lead order across datasets: I, II, III, aVR, aVL, aVF, V1–V6.

### Tools
- `wfdb` — PhysioNet's official Python reader
- `neurokit2` — ECG-specific signal processing (peak detection, quality checks, feature extraction)
- `scipy.signal` — filtering and resampling
- `matplotlib` — plot the raw and processed waveform to verify your pipeline visually

### What to watch out for
- **Missing leads:** Some records have NaN or zero-filled leads — detect and handle
- **Clipping:** Amplitudes outside ±5 mV are likely artifact — clip before normalizing
- **Label leakage:** CODE-15% labels are self-reported; don't treat them as ground truth for final evaluation

### Done signal
Write a function `preprocess_ecg(path) -> np.ndarray` that reads a WFDB record, filters, resamples, normalizes, and returns a `[12, 4000]` float32 array. Plot one raw waveform and one processed waveform side by side. The processed version should be smooth with no baseline wander.

---

## Phase 3 — 1D Deep Learning for ECG
**Time: 3–4 days**

You know 2D CNNs and transformers. This phase is the adaptation to 1D temporal signals.

### 1D ResNet

Identical logic to 2D ResNet — replace `Conv2d` with `Conv1d`, same residual structure. Key differences:
- Kernel sizes are typically 7–15 (covering ~17–37ms at 400 Hz — the width of a QRS complex)
- Pooling is 1D
- Input: `[B, 12, 4000]`, output after global average pooling: `[B, n_channels]`

Start from an established published 1D ResNet architecture for ECG (search for current baselines) to have a benchmark to compare against.

### Squeeze-and-Excitation (SE) blocks

SE blocks add channel attention — after each conv block, a small MLP learns to weight each channel (lead) differently:

```
Feature map [B, C, T]
    → Global avg pool → [B, C]
    → FC → ReLU → FC → Sigmoid → [B, C]
    → Multiply back: rescale each channel
```

For 12-lead ECG, this means the model learns which leads matter more for a given prediction. This is your explainability hook for the SE weights.

### ECG Transformer (secondary model)

PatchTST-style: divide each lead's 4,000-sample signal into patches (e.g., 50 samples each = 80 patches per lead), treat each patch as a token, run a standard transformer encoder. The attention maps show which time windows the model focused on.

More parameters and compute than 1D ResNet, but useful as a comparison.

### Class imbalance

Chagas prevalence in CODE-15% is ~3%. This means ~97% of samples are negative — a model that predicts "no Chagas" on everything gets 97% accuracy but is useless. Strategies:
- `pos_weight` in `nn.BCEWithLogitsLoss` — weight positive samples more heavily
- Oversample SaMi-Trop (confirmed labels) in each batch
- **Don't use accuracy as your metric** — use AUROC and TPR@5%

### Done signal
Implement a 1D ResNet with at least 4 residual blocks that trains on a small subset of CODE-15% and produces a probability output. Training loss should decrease over 10 epochs. AUROC on validation > 0.6 (chance is 0.5).

---

## Phase 4 — Explainability for 1D Signals
**Time: 2 days**

### Grad-CAM on 1D CNN

Grad-CAM computes a saliency map over the input by backpropagating gradients from the predicted class to the last convolutional layer.

For 1D signals, the result is a 1D heatmap over time: which 25ms windows of the ECG drove the "Chagas positive" prediction?

Using `captum` (PyTorch):
```python
from captum.attr import GuidedGradCam
gc = GuidedGradCam(model, model.layer4)  # last conv layer
attr = gc.attribute(input_tensor, target=1)  # target=1 = Chagas positive
```

Overlay the heatmap on the raw ECG waveform to produce a visualization.

### SE lead weights

After each forward pass, extract the SE block's per-lead weights. For a Chagas-positive prediction, leads V1–V2 should have high weight (RBBB is visible there). Visualize as a bar chart over the 12 leads.

### Clinical narration

For 3–5 representative high-confidence predictions, write a one-paragraph narrative:
- What did the model predict?
- Which time region was highlighted by Grad-CAM?
- Which leads were weighted by SE?
- Is this consistent with known Chagas ECG patterns?

This narration is what a researcher or clinician evaluates — not the heatmap alone.

### Done signal
Produce a Grad-CAM visualization for one true positive prediction from SaMi-Trop (confirmed Chagas). Describe in one sentence whether the highlighted region is clinically plausible.

---

## Phase 5 — Domain Shift and Generalization
**Time: 2 days (conceptual + implementation)**

### Why domain shift matters here

The 2025 challenge's most important finding: the winning team's score dropped substantially between the validation set (Brazilian public health system) and the ELSA-Brasil test set (a different Brazilian cohort). Same country, different demographic composition, different disease prevalence.

This is the gap between "benchmark ML" and "clinical deployment" — and being able to discuss it is exactly what separates a portfolio project from a research contribution.

### What causes it

- **Population difference:** ELSA-Brasil skews toward higher-income, better-nourished patients; Chagas prevalence is lower
- **Recording equipment differences:** different ECG machines produce subtly different signal characteristics
- **Label quality differences:** CODE-15% uses weak labels; confirmed-label datasets have different error structures

### Mitigation approaches

Pick one to implement:

**Option A — Signal augmentation (simplest)**
During training, randomly apply: amplitude scaling, Gaussian noise, baseline wander injection, random lead dropout. Forces the model to learn invariant features rather than dataset-specific artifacts.

**Option B — Probability calibration (most practically useful)**
After training, fit a temperature scaling layer on a held-out validation set. This adjusts predicted probabilities without changing the ranking — useful for TPR@5% since the metric depends on ranking, not raw probabilities.

**Option C — Analyze where the model fails**
Plot AUROC separately for CODE-15%, SaMi-Trop, and PTB-XL. Identify which data source has the worst performance. Hypothesize why. This is a valid research contribution even without a fix.

### Done signal
Measure AUROC separately on CODE-15% validation and SaMi-Trop. Report the difference. Write one paragraph explaining what might cause it.

---

## Phase 6 — Tooling and Infrastructure
**Time: 1–2 days**

Do this just before starting the build. Prove each integration works before writing real training code.

**PyTorch Lightning**
- Write a `LightningModule` with `training_step`, `validation_step`, `configure_optimizers`
- Add `ModelCheckpoint` callback to save best model by validation AUROC

**Weights & Biases**
- `wandb.init(project="chagas-ecg")`
- Log: loss, AUROC, TPR@5%, learning rate per epoch
- Log: one Grad-CAM visualization per validation epoch

**Kaggle Notebooks (prototyping)**
- Upload a small subset of CODE-15% to Kaggle dataset storage
- Verify your preprocessing pipeline runs end-to-end
- Confirm training loop runs without OOM errors before switching to paid compute

**RunPod (training)**
- Spin up an A100 instance on RunPod community tier
- Use a PyTorch Docker image — no environment setup needed
- `rsync` or `git clone` your code, download data from PhysioNet directly

### Done signal
A training run completes end-to-end (even 1 epoch on a small subset) with metrics logged to W&B. You can see the run in the W&B dashboard.

---

## Learning Order

```
Phase 1 (Chagas + ECG basics)
    ↓
Phase 2 (Signal processing) ← most important to get right
    ↓
Phase 3 (1D deep learning)
    ↓
Phase 4 (Explainability)    Phase 5 (Domain shift)
         ↓                           ↓
              Phase 6 (Infrastructure) ← do this before building
```

Phases 4 and 5 can run in parallel once Phase 3 is solid.

---

## Questions to answer before talking to a researcher

1. What ECG abnormality is most associated with Chagas cardiomyopathy, and where in the waveform does it appear?
2. Why is TPR@5% a better metric for this task than AUROC?
3. What does a bandpass filter between 0.5–40 Hz remove, and why does it matter?
4. Why does a model that achieves 97% accuracy on CODE-15% fail as a clinical tool?
5. Why does the challenge winning team's score drop between validation and test populations?
6. What does your Grad-CAM saliency map show, and is it clinically plausible?
7. How would you deploy this model as a screening tool in a resource-limited setting?