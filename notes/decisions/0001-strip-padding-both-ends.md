---
status: approved
applies_to: src/preprocess.py
---
# 0001 — Strip zero-padding at both ends of a recording

DRAFT by Claude from the 2026-10-09 chat decision ("fix F1 now, F2 later"). Edit freely; set `status: approved` yourself.

**Decision:** `_strip_zero_padding` removes all-zero rows at the start AND end of a recording, as the challenge's own converter does (`python-example-2025/prepare_code15_data.py:188-195`). How the stripped signal is re-padded or cropped back to 4000 samples is NOT changed here (that is the open F2 length question).

**Why:** Short CODE-15% and SaMi-Trop recordings hold ~2934 real samples with ~581 zeros before and ~581 after. Only the trailing zeros were removed, so the bandpass filter ran across a 0→signal step, the z-score statistics included ~581 zeros, and the model saw a ~1.45 s flat stretch at the start of the input.

**Alternatives considered:** leave as is (rejected: artifact enters every input); fix behind a flag so old checkpoints can still run (not chosen).

**Effect on reported numbers or checkpoints:** model inputs change for ~60% of CODE-15% records. Existing checkpoints and the 0.8296 / 0.8276 test numbers were produced without this fix; any new result requires a retrain.

**How it is verified:** on part0 records with ~2934 real samples the output has no leading flat region; a full-length record is unchanged apart from the existing truncation to 4000.
