# GPU Training Setup

Target platform: **RunPod** (Linux, CUDA 12.1).
Estimated cost: ~$2/hr on RTX 3090 community GPU, ~$4/hr on A100.

---

## 1. Instance Setup

Recommended pod template: **RunPod PyTorch 2.1** (includes CUDA 12.1, Python 3.11).

If starting from a bare image or custom template, install CUDA drivers and verify:

```bash
nvidia-smi            # confirm GPU is visible, CUDA version
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Install Dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone the repo
git clone <your-repo-url>
cd chagas-ecg-explainability

# Install Python deps — pyproject.toml will pull torch>=2.3 on Linux
uv sync

# Verify the correct CUDA torch is installed
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `False`, install the CUDA wheel explicitly:

```bash
uv add torch --index https://download.pytorch.org/whl/cu121
```

---

## 3. Upload Data

The HDF5 and WFDB files must be present at `data/` before training.
Three options (fastest to slowest):

**Option A — RunPod volume mount (recommended)**
Attach a network volume in the RunPod UI. Download the data once, reuse across pods.

**Option B — direct download on pod**
```bash
mkdir -p data/code15
cd data/code15
curl -L "https://zenodo.org/records/4916206/files/exams_part0.zip?download=1" -o exams_part0.zip
curl -L "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip" -o code15_chagas_labels.zip
unzip exams_part0.zip && unzip code15_chagas_labels.zip
```

For the full 18-part CODE-15% dataset:
```bash
for i in $(seq 0 17); do
  curl -L "https://zenodo.org/records/4916206/files/exams_part${i}.zip?download=1" -o exams_part${i}.zip
  unzip exams_part${i}.zip && rm exams_part${i}.zip
done
```

**Option C — rsync from local machine**
```bash
rsync -avz --progress data/ <runpod-user>@<pod-ip>:<pod-path>/data/
```

---

## 4. Run the Audit Before Training

Always verify the data is healthy before any training run:

```bash
uv run python src/audit_preprocessing.py --n-samples 20
```

Expected: `PASS=43  FAIL=0  WARN=0  SKIP=0`.
Any FAIL is a blocker — do not proceed.

---

## 5. Training Command

Three tiers — pick the one that matches your data and goal:

| Tier | Data needed | Positives | Purpose |
|---|---|---|---|
| Smoke test | part0 only (already local) | 283 train | Verify CUDA path, ~2 min |
| Part0 full run | part0 only (already local) | 283 train | Real training, no extra download |
| Full dataset | All 18 parts (~67 GB) | ~4,600 train | Production-quality run |

---

**Tier 2 — Part 0 only, full training run (no extra download needed):**

```bash
uv run python src/train.py \
    --hdf5 data/code15/exams_part0.hdf5 \
    --labels data/code15/code15_chagas_labels.csv \
    --epochs 30 --batch-size 256 \
    --accelerator gpu --devices 1 --num-workers 4 --amp \
    --eval-test --wandb
```

This trains on 13,974 exams (283 positives). Expect ~45 sec/epoch on RTX 3090 → ~23 min total.
Use this for initial GPU validation and hyperparameter tuning before committing to a full run.

---

**Tier 3 — all 18 CODE-15% parts (recommended for real training):**

```bash
uv run python src/train.py \
    --hdf5 data/code15/exams_part0.hdf5 \
          data/code15/exams_part1.hdf5 \
          data/code15/exams_part2.hdf5 \
          data/code15/exams_part3.hdf5 \
          data/code15/exams_part4.hdf5 \
          data/code15/exams_part5.hdf5 \
          data/code15/exams_part6.hdf5 \
          data/code15/exams_part7.hdf5 \
          data/code15/exams_part8.hdf5 \
          data/code15/exams_part9.hdf5 \
          data/code15/exams_part10.hdf5 \
          data/code15/exams_part11.hdf5 \
          data/code15/exams_part12.hdf5 \
          data/code15/exams_part13.hdf5 \
          data/code15/exams_part14.hdf5 \
          data/code15/exams_part15.hdf5 \
          data/code15/exams_part16.hdf5 \
          data/code15/exams_part17.hdf5 \
    --labels data/code15/code15_chagas_labels.csv \
    --epochs 30 \
    --batch-size 256 \
    --lr 1e-3 \
    --accelerator gpu \
    --devices 1 \
    --num-workers 4 \
    --amp \
    --eval-test \
    --wandb
```

Or with shell glob expansion (bash/zsh):

```bash
uv run python src/train.py \
    --hdf5 data/code15/exams_part*.hdf5 \
    --labels data/code15/code15_chagas_labels.csv \
    --epochs 30 --batch-size 256 \
    --accelerator gpu --devices 1 --num-workers 4 --amp \
    --eval-test --wandb
```

This trains on 343,424 exams with 6,561 positives (vs. 19,939 / 403 with part0 only).

**Flag reference:**

| Flag | Default | Purpose |
|---|---|---|
| `--accelerator gpu` | `auto` | Explicit GPU; required on RunPod |
| `--devices 1` | 1 | Number of GPUs |
| `--num-workers 4` | 0 | DataLoader workers; 0 is safe on Mac, 4–8 on Linux |
| `--amp` | off | 16-bit mixed precision (~40% memory savings, ~30% speedup) |
| `--eval-test` | off | Evaluate the held-out test split after training ends |
| `--wandb` | off | Log metrics to W&B |
| `--batch-size 256` | 64 | GPU allows larger batches than CPU prototype |

**Smoke test on GPU (verify CUDA path before a long run):**

```bash
uv run python src/train.py \
    --hdf5 data/code15/exams_part0.hdf5 \
    --labels data/code15/code15_chagas_labels.csv \
    --fast --epochs 2 --batch-size 64 \
    --accelerator gpu --devices 1 --num-workers 4 --amp
```

Expected: training starts within ~30s, GPU utilization visible in `nvidia-smi`.

---

## 6. Checkpoint and Resume

PyTorch Lightning saves checkpoints to `lightning_logs/version_N/checkpoints/`.
The `ModelCheckpoint` callback keeps the best `val/auroc` checkpoint.

To resume from a checkpoint:

```bash
uv run python src/train.py \
    --hdf5 data/code15/exams_part0.hdf5 \
    --labels data/code15/code15_chagas_labels.csv \
    --epochs 30 --batch-size 256 \
    --accelerator gpu --devices 1 --num-workers 4 --amp
# Add to trainer in train.py: trainer.fit(..., ckpt_path="path/to/checkpoint.ckpt")
```

Download checkpoints from RunPod before terminating the pod:

```bash
rsync -avz <pod-ip>:<pod-path>/lightning_logs/ ./lightning_logs/
```

---

## 7. Mixed Precision Notes

`--amp` enables `precision="16-mixed"` (FP16 forward + backward, FP32 master weights).
This is safe for this model: BatchNorm and the BCE loss are not precision-sensitive at this scale.

- A100 also supports `bfloat16` (`precision="bf16-mixed"`) — numerically more stable than FP16, no overflow risk. Change the `precision` line in `train.py` if desired.
- RTX 3090 has native FP16 Tensor Cores; BF16 is also supported but slower than on A100.

---

## 8. Expected Training Time

| Hardware | Batch size | AMP | Full part0 / epoch |
|---|---|---|---|
| Mac CPU (M1/M2) | 32–64 | off | ~20 min/epoch |
| RTX 3090 | 256 | on | ~45 sec/epoch |
| A100 40 GB | 512 | on | ~25 sec/epoch |

30 epochs at 45 sec = ~23 min total on RTX 3090.
