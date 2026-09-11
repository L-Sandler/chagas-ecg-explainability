# Run ledger: faithful record only

Filled in by the facilitator agent, verbatim from logs / W&B. No interpretation in this file;
the user writes interpretation elsewhere. One block per run, newest at the bottom.

| run name            | date       | pod / GPU              | lr   | epochs run | best ckpt epoch | val/auroc (best) | test AUROC | test AUPRC | TPR@5%FPR | TPR@top-5% | wall time | W&B |
|---------------------|------------|------------------------|------|------------|-----------------|------------------|------------|------------|-----------|------------|-----------|-----|
| full-code15-10ep    | 2026-08-28 | 9hmhxpi69rkp52 / L4    | 1e-3 | 7 (ES@6)   | 1               | 0.836            | 0.8296     | 0.1403     | 0.4036    | 0.3804     | ~35 min   | [97ugsfys](https://wandb.ai/leo-s-org/chagas-ecg/runs/97ugsfys) |
| full-code15-lr1e-3  | 2026-08-28 | 8w09mq198ctjw4 / 4090  | 1e-3 | 10 (ES@9)  | 1               | 0.836            | 0.8276     | 0.1366     | 0.4127    | 0.3875     |           |     |
| full-code15-lr3e-4  |            |                        | 3e-4 |            |                 |                  |            |            |           |            |           |     |
| full-code15-lr3e-3  |            |                        | 3e-3 |            |                 |                  |            |            |           |            |           |     |

The two 2026-08-28 rows are back-filled from `lightning_logs/*/train.log` and session notes.
Those logs do not contain the launch command, so their command is reconstructed, not recorded.
(The full-code15-10ep row used the pre-`44b1ca8` Adam + CosineAnnealing code, patience 5.)

---

## Template: copy per run

### <run-name>  (<date>)

- **Command (verbatim from `train_<run>.meta.json`):**
  ```
  ```
- **git HEAD:** `<sha>`  | manifest `code_sha`: `<sha>` | code-pin check: pass/fail
- **Pod:** `<pod-id>`, `<GPU>`, driver CUDA `<x.y>`, `$<rate>/hr`
- **W&B run:** `<url>`
- **Config as logged to W&B:** pos_weight=, batch_size=, epochs=, lr=, git_sha=, per-source sizes=
- **Per-epoch metrics:**

  | epoch | val/auroc | val/tpr_top5pct | val/tpr_at_5pct_fpr | train loss |
  |-------|-----------|-----------------|---------------------|------------|

- **Stop reason:** (e.g. "EarlyStopping patience=8 exhausted at epoch N"; "max_epochs reached")
- **Best checkpoint:** `lightning_logs/<run>/checkpoints/best-epoch=<n>-val/auroc=<x>.ckpt` (epoch N)
- **Test (CODE-15% held-out, N samples, P% positive):** AUROC , AUPRC , TPR@5%FPR , TPR@top-5%
- **PTB-XL fold-10 check:** (what the script printed, verbatim)
- **Wall time:** launch to test-eval end
- **Pod-hours billed this run:**
- **Anomalies (factual):**
