"""
Explainability harness: attribution overlays for a trained ECGResNet checkpoint.

Wire-up focus — runs on the existing epoch-12 checkpoint (trained on 1/18 of
CODE-15%), so explanations are not expected to be clinically meaningful yet.
This module exists to prove the plumbing (checkpoint -> record -> attribution
-> overlay), not to produce trustworthy insight. Re-run once a properly
trained checkpoint exists.

Usage:
    # Single record by exam_id
    python src/explain.py --checkpoint <path> --record <exam_id> --method gradcam

    # Auto-pick the highest-confidence true-positive and false-positive in the
    # test split and render both methods for each
    python src/explain.py --checkpoint <path> --method both
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from dataset import Code15Dataset
from preprocess import LEAD_ORDER
from train import ECGLightningModule

OUT_DIR = "reports/explainability"


def load_module(checkpoint_path: str, device: str = "cpu") -> ECGLightningModule:
    module = ECGLightningModule.load_from_checkpoint(checkpoint_path, map_location=device)
    module.model.to(device)
    module.model.eval()
    return module


def get_record(ds: Code15Dataset, exam_id: int) -> tuple[torch.Tensor, float]:
    matches = ds.df.index[ds.df["exam_id"] == exam_id]
    if len(matches) == 0:
        raise ValueError(f"exam_id {exam_id} not found in split")
    signal, label = ds[int(matches[0])]
    return signal, float(label)


@torch.no_grad()
def predict_prob(module: ECGLightningModule, signal: torch.Tensor, device: str = "cpu") -> float:
    x = signal.unsqueeze(0).to(device)
    logit = module.model(x)
    return float(torch.sigmoid(logit).item())


def pick_interesting_records(
    module: ECGLightningModule, ds: Code15Dataset, device: str = "cpu", max_scan: int = 1000
) -> dict[str, int]:
    """Scan up to max_scan records and return the exam_id of the
    highest-confidence true positive and highest-confidence false positive
    (a negative record the model is most fooled by)."""
    n = min(len(ds), max_scan)
    best = {"tp": (-1.0, None), "fp": (-1.0, None)}
    for i in range(n):
        signal, label = ds[i]
        prob = predict_prob(module, signal, device)
        exam_id = int(ds.df.iloc[i]["exam_id"])
        if label == 1.0 and prob > best["tp"][0]:
            best["tp"] = (prob, exam_id)
        elif label == 0.0 and prob > best["fp"][0]:
            best["fp"] = (prob, exam_id)
    picks = {}
    if best["tp"][1] is not None:
        picks["tp"] = best["tp"][1]
    if best["fp"][1] is not None:
        picks["fp"] = best["fp"][1]
    return picks


class GradCAM1D:
    """
    Grad-CAM for the 1D ECGResNet.

    Target layer: model.blocks[-1] — the last residual/SE block, whose output
    is the final feature map fed into global average pooling (model.gap).
    This is the deepest layer whose spatial (time) dimension is still
    meaningful before the classifier collapses it to a single vector.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.blocks[-1]
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def attribute(self, x: torch.Tensor) -> np.ndarray:
        """x: [1, 12, T]. Returns a (T,) importance array in [0, 1]."""
        self.model.zero_grad()
        x = x.clone().requires_grad_(True)
        logit = self.model(x)  # [1]
        logit.backward()

        weights = self.gradients.mean(dim=2, keepdim=True)  # [1, C, 1]
        cam = F.relu((weights * self.activations).sum(dim=1))  # [1, T']
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()


def integrated_gradients_attribution(module: ECGLightningModule, x: torch.Tensor) -> np.ndarray:
    """x: [1, 12, T]. Returns a (12, T) per-lead-per-timestep attribution array."""
    ig = IntegratedGradients(module.model)
    baseline = torch.zeros_like(x)
    attr = ig.attribute(x, baselines=baseline)
    return attr.squeeze(0).detach().cpu().numpy()


def _plot_lead(ax, t: np.ndarray, sig: np.ndarray, attr: np.ndarray, lead_name: str) -> None:
    attr_range = attr.max() - attr.min()
    attr_norm = (attr - attr.min()) / attr_range if attr_range > 1e-8 else np.zeros_like(attr)
    ax.imshow(
        attr_norm[np.newaxis, :], aspect="auto", cmap="Reds", alpha=0.5,
        extent=[t[0], t[-1], sig.min() - 0.1, sig.max() + 0.1], origin="lower",
    )
    ax.plot(t, sig, color="black", lw=0.6)
    ax.set_ylabel(lead_name, rotation=0, labelpad=20, fontsize=8)
    ax.set_yticks([])


def plot_explanation(signal: np.ndarray, attribution: np.ndarray, title: str, out_path: str) -> None:
    """signal: [12, T]. attribution: (T,) shared across leads, or (12, T) per-lead."""
    n_leads, T = signal.shape
    t = np.arange(T) / 400.0
    per_lead = attribution.ndim == 2

    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 1.4 * n_leads), sharex=True)
    for i, ax in enumerate(axes):
        attr_i = attribution[i] if per_lead else attribution
        _plot_lead(ax, t, signal[i], attr_i, LEAD_ORDER[i])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def explain_record(
    module: ECGLightningModule, ds: Code15Dataset, exam_id: int, method: str,
    tag: str, out_dir: str, device: str = "cpu",
) -> None:
    signal, label = get_record(ds, exam_id)
    prob = predict_prob(module, signal, device)
    x = signal.unsqueeze(0).to(device)

    if method in ("gradcam", "both"):
        cam = GradCAM1D(module.model).attribute(x)
        title = f"Grad-CAM | exam {exam_id} | label={label:.0f} pred={prob:.3f}"
        out_path = os.path.join(out_dir, f"{tag}_{exam_id}_gradcam.png")
        plot_explanation(signal.numpy(), cam, title, out_path)

    if method in ("ig", "both"):
        ig_attr = integrated_gradients_attribution(module, x)
        title = f"Integrated Gradients | exam {exam_id} | label={label:.0f} pred={prob:.3f}"
        out_path = os.path.join(out_dir, f"{tag}_{exam_id}_ig.png")
        plot_explanation(signal.numpy(), ig_attr, title, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5", nargs="+", default=["data/code15/exams_part0.hdf5"])
    parser.add_argument("--labels", default="data/code15/code15_chagas_labels.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--record", type=int, default=None,
                         help="exam_id to explain. Omit to auto-pick the highest-confidence "
                              "true-positive and false-positive in the split.")
    parser.add_argument("--method", default="both", choices=["gradcam", "ig", "both"])
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    module = load_module(args.checkpoint, args.device)
    ds = Code15Dataset(args.hdf5, args.labels, split=args.split)

    if args.record is not None:
        explain_record(module, ds, args.record, args.method, "record", args.out_dir, args.device)
        return

    picks = pick_interesting_records(module, ds, args.device)
    if "tp" in picks:
        print(f"True positive: exam_id={picks['tp']}")
        explain_record(module, ds, picks["tp"], args.method, "tp", args.out_dir, args.device)
    if "fp" in picks:
        print(f"False positive (highest-confidence negative): exam_id={picks['fp']}")
        explain_record(module, ds, picks["fp"], args.method, "fp", args.out_dir, args.device)


if __name__ == "__main__":
    main()
