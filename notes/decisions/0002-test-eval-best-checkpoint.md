---
status: approved
applies_to: src/train.py
---
# 0002 — Report test metrics from the best-validation checkpoint

DRAFT by Claude. **You have not decided this one yet**; it changes what the reported numbers mean, so it is yours to accept, reject or reshape.

**Decision (proposed):** after `trainer.fit`, load `trainer.checkpoint_callback.best_model_path` and run the test evaluation on those weights, printing which checkpoint/epoch was used.

**Why:** `_eval_test(module, ...)` currently runs on the weights left after the last epoch. Per the logs, run 1 ended at epoch 6 and run 2 at epoch 9, while the best validation AUROC was epoch 1 (0.836) in both. `explain.py` loads the best checkpoint, so the reported metrics and the explained model are different models.

**Alternatives considered:** keep last-epoch evaluation and say so in the paper (cheaper, but then explain.py must load last-epoch weights, which are not saved); evaluate both and report both.

**Effect on reported numbers or checkpoints:** 0.8296 / 0.8276 (and the TPR@top-5% values) describe last-epoch weights, not the best checkpoint. They must be recomputed or labelled as such.

**How it is verified:** `--fast` smoke run prints the checkpoint path and epoch used for the test block.
