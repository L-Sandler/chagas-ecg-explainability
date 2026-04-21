# Open Questions

Questions raised during learning sessions that are unresolved, worth revisiting, or represent genuine uncertainty in the approach.

Format: question, phase where it arose, current status.

---

## Signal Processing

**Q: Is the 0.5–40 Hz bandpass cutoff correct for Chagas detection?**
Phase: 2 — ECG Signal Processing
Status: Open — accepted convention for now

The 40 Hz upper cutoff is a clinical/engineering convention partly set by older hardware limits. Some research suggests QRS morphology information exists up to ~150 Hz. For Chagas (RBBB, conduction delays), the relevant features are well within 0–40 Hz, so the convention is likely safe. Worth revisiting if the model shows unexpected behavior or if ablation studies suggest higher frequencies carry signal.

---
