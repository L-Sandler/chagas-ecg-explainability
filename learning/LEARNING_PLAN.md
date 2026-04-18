# Learning Plan

Assumes: solid ML/DL + PyTorch background. Gaps: ECG signal processing, clinical cardiology basics, 1D deep learning architectures, PhysioNet tooling.

Each phase has a **goal** and a **done signal** — a concrete thing you should be able to do or explain without looking it up.

> **Note for teaching sessions:** When explaining any clinical or technical concept covered in this plan, always cite the specific paper or source that the claim comes from. If a fact about Chagas ECG patterns, signal processing, or model design can't be traced to a source listed here, flag it as unverified.

---

## Phase 1 — Chagas Disease and Clinical ECG Basics
**Time: 2–3 days**

### What is Chagas disease

- Caused by *Trypanosoma cruzi*, a parasite transmitted by triatomine insects
- ~8 million infected globally, concentrated in Latin America
- Most chronic cases are asymptomatic until cardiac damage develops
- **Chagas cardiomyopathy** is the main lethal complication — it disrupts the heart's electrical conduction system
- No widely available vaccine; serological testing is confirmatory but capacity-limited

### How the heart's electrical system works (enough to read an ECG)

The ECG records electrical activity as the heart contracts. Each beat produces a waveform with named components:

```
P wave  → atria depolarize (contract)
PR interval → conduction delay at AV node
QRS complex → ventricles depolarize (main pump contraction)
T wave  → ventricles repolarize (reset)
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

### Resources

**Primary — read these:**

1. **Rassi A Jr, Rassi A, Marin-Neto JA. "Chagas disease." *Lancet*, 375(9723):1388–1402, 2010.**
   — The definitive clinical review. Covers epidemiology, pathophysiology, ECG findings, and treatment. This is the paper researchers cite when discussing Chagas cardiomyopathy. Read sections on cardiac manifestations and ECG abnormalities.
   DOI: 10.1016/S0140-6736(10)60061-X | [PubMed](https://pubmed.ncbi.nlm.nih.gov/20399979/) | [Full text (Lancet)](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60061-X/abstract)

2. **Nunes MCP, Dones W, Morillo CA, et al. "Chagas disease: an overview of clinical and epidemiological aspects." *Journal of the American College of Cardiology*, 62(9):767–776, 2013.**
   — Focused on the cardiac complications and their clinical management. More accessible than Rassi for ECG-specific findings.
   DOI: 10.1016/j.jacc.2013.05.046 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/23770163/) | [Full text (JACC)](https://www.jacc.org/doi/10.1016/j.jacc.2013.05.046)

3. **Ribeiro ALP, et al. "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications*, 11:1760, 2020.**
   — The CODE-15% paper. Read the introduction and results. This is the paper behind your primary dataset and your baseline architecture.
   DOI: 10.1038/s41467-020-15432-4 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32273514/) | [Full text (Nature)](https://www.nature.com/articles/s41467-020-15432-4) | [GitHub](https://github.com/antonior92/automatic-ecg-diagnosis)

4. **Goldberger AL, et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*, 101(23):e215–e220, 2000.**
   — The original PhysioNet paper. Cite this when referencing any PhysioNet dataset.
   DOI: 10.1161/01.CIR.101.23.e215 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/10851218/) | [Full text (Circulation)](https://www.ahajournals.org/doi/10.1161/01.cir.101.23.e215)

**Reference (use as needed, not cover-to-cover):**

5. **Life in the Fast Lane ECG Library — litfl.com/ecg-library/**
   — Written and reviewed by emergency physicians. Useful for looking up specific ECG patterns (e.g., "what does RBBB look like"). Use for visual pattern reference, not for citation in writing.

6. **WHO Chagas disease fact sheet — who.int/news-room/fact-sheets/detail/chagas-disease-(american-trypanosomiasis)**
   — Authoritative epidemiology numbers. Use for global prevalence figures.

### Done signal
Explain in one paragraph: what RBBB looks like on an ECG, why Chagas disease causes it, and why that makes ECG useful for Chagas screening. Cite Rassi et al. (2010) or Nunes et al. (2013) for each clinical claim.

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

### Resources

1. **Sörnmo L, Laguna P. *Bioelectrical Signal Processing in Cardiac and Neurological Applications*. Elsevier, 2005.**
   — The standard textbook for ECG signal processing. Chapter 6 covers filtering and artifact removal. Dense but authoritative — use as a reference when you need to understand *why* a preprocessing choice is made.
   [Publisher page](https://www.elsevier.com/books/bioelectrical-signal-processing-in-cardiac-and-neurological-applications/sornmo/978-0-12-437552-9)

2. **Luz EJS, et al. "ECG-based heartbeat classification for arrhythmia detection: A survey." *Computer Methods and Programs in Biomedicine*, 127:144–164, 2016.**
   — Survey of preprocessing pipelines used in ECG ML. Useful for understanding what choices are standard vs. novel in your pipeline.
   DOI: 10.1016/j.cmpb.2015.12.008 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26775002/)

3. **Makowski D, et al. "NeuroKit2: A Python toolbox for neurophysiological signal processing." *Behavior Research Methods*, 53:1689–1696, 2021.**
   — The paper behind the `neurokit2` library. Cite this when using neurokit2 in your writeup.
   DOI: 10.3758/s13428-020-01516-y | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33528817/) | [Full text (Springer)](https://link.springer.com/article/10.3758/s13428-020-01516-y)

4. **Wagner P, et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data*, 7:154, 2020.**
   — The PTB-XL dataset paper. Read the data collection and preprocessing sections — they describe the recording conditions and label structure you'll need to understand to use it correctly.
   DOI: 10.1038/s41597-020-0495-6 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32451379/) | [Full text (Nature)](https://www.nature.com/articles/s41597-020-0495-6) | [Dataset (PhysioNet)](https://physionet.org/content/ptb-xl/)

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

Start with the architecture from the CODE-15% paper (Ribeiro et al., 2020) — it's a 1D ResNet and the code is on GitHub. This gives you a published baseline to compare against.

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

### Resources

1. **Ribeiro ALP, et al. "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications*, 11:1760, 2020.**
   — Your baseline architecture. The GitHub repo (github.com/antonior92/automatic-ecg-diagnosis) has a PyTorch implementation to start from.
   DOI: 10.1038/s41467-020-15432-4 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32273514/) | [Full text (Nature)](https://www.nature.com/articles/s41467-020-15432-4) | [GitHub](https://github.com/antonior92/automatic-ecg-diagnosis)

2. **He K, et al. "Deep residual learning for image recognition." *CVPR*, 2016.**
   — The original ResNet paper. You know this, but re-read it with 1D in mind — the residual block logic is identical. Useful to cite when describing your architecture.
   DOI: 10.1109/CVPR.2016.90 | [arXiv](https://arxiv.org/abs/1512.03385) | [CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

3. **Hu J, et al. "Squeeze-and-excitation networks." *CVPR*, 2018.**
   — The original SE block paper. Read Section 3. When you explain SE weights as a lead-importance mechanism, this is what you cite.
   DOI: 10.1109/CVPR.2018.00745 | [arXiv](https://arxiv.org/abs/1709.01507) | [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)

4. **Nie Y, et al. "A time series is worth 64 words: Long-term forecasting with transformers." *ICLR*, 2023.**
   — The PatchTST paper. Read the patch tokenization section (Section 3). Your ECG transformer follows this design.
   URL: https://openreview.net/forum?id=Jbdc0vTOcol | [arXiv](https://arxiv.org/abs/2211.14730)

5. **Hannun AY, et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." *Nature Medicine*, 25:65–69, 2019.**
   — Stanford's landmark ECG deep learning paper. Context for the state of the field before the CODE-15% paper. Useful for framing your project's place in the literature.
   DOI: 10.1038/s41591-018-0268-3 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/30617320/) | [Full text (Nature)](https://www.nature.com/articles/s41591-018-0268-3)

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

### Resources

1. **Selvaraju RR, et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV*, 2017.**
   — The original Grad-CAM paper. Read Sections 1–3. Your 1D adaptation follows the same gradient flow logic.
   DOI: 10.1109/ICCV.2017.74 | [arXiv](https://arxiv.org/abs/1610.02391) | [CVF (open access)](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)

2. **Kokhlikyan N, et al. "Captum: A unified and generic model interpretability library for PyTorch." *arXiv*, 2020.**
   — The paper behind the `captum` library. Cite this when using it in your writeup.
   arXiv: 2009.07896 | [arXiv](https://arxiv.org/abs/2009.07896) | [GitHub](https://github.com/pytorch/captum)

3. **Strodthoff N, et al. "Deep learning for ECG analysis: Benchmarks and insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics*, 25(5):1519–1528, 2021.**
   — Benchmarks explainability methods on ECG specifically, using PTB-XL. Directly relevant to your interpretability evaluation.
   DOI: 10.1109/JBHI.2020.3022989 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32915745/) | [arXiv](https://arxiv.org/abs/2004.13701)

4. **Tonekaboni S, Joshi S, McCradden MD, Goldenberg A. "What clinicians want: Contextualizing explainable machine learning for clinical end use." *MLHC Proceedings*, 2019.**
   — Qualitative study of what clinicians actually find useful in ML explanations. Important background for framing your clinical narration section.
   URL: https://proceedings.mlr.press/v106/tonekaboni19a.html | [arXiv](https://arxiv.org/abs/1905.05134)

### Done signal
Produce a Grad-CAM visualization for one true positive prediction from SaMi-Trop (confirmed Chagas). Describe in one sentence whether the highlighted region is clinically plausible. Cite Selvaraju et al. (2017) when describing the method.

---

## Phase 5 — Domain Shift and Generalization
**Time: 2 days (conceptual + implementation)**

### Why domain shift matters here

The 2025 challenge's most important finding: the winning team's score dropped 64% between the validation set (Brazilian public health system) and the ELSA-Brasil test set (a different Brazilian cohort). Same country, different demographic composition, different disease prevalence.

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

### Resources

1. **Finlayson SG, et al. "The clinician and dataset shift in artificial intelligence." *New England Journal of Medicine*, 385:283–286, 2021.**
   — Short, accessible clinical perspective on dataset shift in deployed ML. Useful framing for why this matters beyond the benchmark.
   DOI: 10.1056/NEJMc2104626 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34260843/) | [Full text (NEJM)](https://www.nejm.org/doi/full/10.1056/NEJMc2104626)

2. **Subbaswamy A, Saria S. "From development to deployment: Dataset shift, causality, and shift-stable models in health AI." *Biostatistics*, 21(2):345–352, 2020.**
   — More technical treatment of shift types (covariate shift, concept drift) in clinical ML. Read sections 1–2.
   DOI: 10.1093/biostatistics/kxz041 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/31588496/) | [Full text (Oxford)](https://academic.oup.com/biostatistics/article/21/2/345/5572660)

3. **Guo C, et al. "On calibration of modern neural networks." *ICML*, 2017.**
   — The temperature scaling paper. If you implement Option B, this is what you cite.
   URL: https://proceedings.mlr.press/v70/guo17a.html | [arXiv](https://arxiv.org/abs/1706.04599)

4. **Shorten C, Khoshgoftaar TM. "A survey on image data augmentation for deep learning." *Journal of Big Data*, 6:60, 2019.**
   — Covers augmentation strategies including signal-level methods. Useful background for Option A.
   DOI: 10.1186/s40537-019-0197-0 | [Full text (open access)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)

5. **PhysioNet Challenge 2025 Description Paper** — check physionet.org/content/challenge-2025 for the official challenge paper once published. This is the primary citation for the 64% score drop finding.

### Done signal
Measure AUROC separately on CODE-15% validation and SaMi-Trop. Report the difference. Write one paragraph explaining what might cause it, citing Finlayson et al. (2021) or Subbaswamy & Saria (2020).

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
- Spin up an A100 instance on RunPod community tier (~$1.19/hr)
- Use a PyTorch Docker image — no environment setup needed
- `rsync` or `git clone` your code, download data from PhysioNet directly

### Resources

1. **Falcon W, The PyTorch Lightning Team. "PyTorch Lightning." *GitHub*, 2019.**
   — Cite as the software reference for PyTorch Lightning.
   URL: https://github.com/Lightning-AI/pytorch-lightning | [Docs](https://lightning.ai/docs/pytorch/stable/)

2. **Biewald L. "Experiment tracking with Weights and Biases." *GitHub*, 2020.**
   — The software citation for W&B.
   URL: https://www.wandb.com | [Docs](https://docs.wandb.ai/)

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

## Master Reference List

| Paper | Phase | Why it matters |
|---|---|---|
| Rassi et al., *Lancet* 2010 | 1 | Definitive Chagas clinical review |
| Nunes et al., *JACC* 2013 | 1 | Chagas cardiac manifestations |
| Ribeiro et al., *Nature Comms* 2020 | 1, 3 | CODE-15% dataset + baseline architecture |
| Goldberger et al., *Circulation* 2000 | 1 | PhysioNet citation |
| Sörnmo & Laguna, textbook 2005 | 2 | ECG signal processing theory |
| Luz et al., *CMPB* 2016 | 2 | ECG preprocessing survey |
| Makowski et al., *BRM* 2021 | 2 | NeuroKit2 citation |
| Wagner et al., *Scientific Data* 2020 | 2 | PTB-XL dataset paper |
| He et al., *CVPR* 2016 | 3 | ResNet architecture |
| Hu et al., *CVPR* 2018 | 3 | SE block paper |
| Nie et al., *ICLR* 2023 | 3 | PatchTST / patch tokenization |
| Hannun et al., *Nature Medicine* 2019 | 3 | ECG deep learning context |
| Selvaraju et al., *ICCV* 2017 | 4 | Grad-CAM method |
| Kokhlikyan et al., *arXiv* 2020 | 4 | Captum citation |
| Strodthoff et al., *JBHI* 2021 | 4 | Explainability benchmarks on ECG |
| Tonekaboni et al., *MLHC* 2019 | 4 | What clinicians want from XAI |
| Finlayson et al., *NEJM* 2021 | 5 | Dataset shift in clinical AI |
| Subbaswamy & Saria, *Biostatistics* 2020 | 5 | Shift types in health AI |
| Guo et al., *ICML* 2017 | 5 | Temperature scaling / calibration |

---

## Questions to answer before talking to a researcher

1. What ECG abnormality is most associated with Chagas cardiomyopathy, and where in the waveform does it appear? *(cite Rassi et al., 2010)*
2. Why is TPR@5% a better metric for this task than AUROC?
3. What does a bandpass filter between 0.5–40 Hz remove, and why does it matter? *(cite Sörnmo & Laguna)*
4. Why does a model that achieves 97% accuracy on CODE-15% fail as a clinical tool?
5. Why does the challenge winning team's score drop between validation and test populations? *(cite Finlayson et al., 2021)*
6. What does your Grad-CAM saliency map show, and is it clinically plausible? *(cite Selvaraju et al., 2017)*
7. How would you deploy this model as a screening tool in a resource-limited setting?
