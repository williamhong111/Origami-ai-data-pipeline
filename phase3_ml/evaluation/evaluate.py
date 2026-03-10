"""
evaluate.py — Evaluation & Domain Randomization Stress Test
==============================================================
Part 3 & 5 of Phase 3:
    - Domain randomization stress test (lighting, camera, texture)
    - Performance degradation table
    - Inference visualization (GT vs Predictions)
    - Sample batch visualization (RGB / Depth / Seg / BBox)
    - Training curves (Loss, mAP, LR)

Usage:
    cd ~/Desktop/files/phase3_ml
    source origami_env/bin/activate

    # Run all evaluations
    python3 evaluation/evaluate.py

    # With trained checkpoint
    python3 evaluation/evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import json
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.origami_dataset import OrigamiMultimodalDataset
from models.yolo_detector import build_model


# ======================================================================
# Domain Randomization Transforms
# ======================================================================
class DomainRandomizer:
    """
    Simulates Isaac Sim domain randomization variations
    for stress testing model robustness.
    """

    @staticmethod
    def lighting_change(rgb: torch.Tensor, intensity: float = 0.3) -> torch.Tensor:
        """Brightness shift to simulate lighting variation."""
        shift = (torch.rand(1).item() * 2 - 1) * intensity
        return torch.clamp(rgb + shift, 0, 1)

    @staticmethod
    def camera_angle(rgb: torch.Tensor, shift_pixels: int = 20) -> torch.Tensor:
        """Translation to simulate camera angle change."""
        _, H, W = rgb.shape
        dx = np.random.randint(-shift_pixels, shift_pixels + 1)
        dy = np.random.randint(-shift_pixels, shift_pixels + 1)
        result = torch.zeros_like(rgb)

        sx1, sy1 = max(0, dx), max(0, dy)
        sx2, sy2 = min(W, W + dx), min(H, H + dy)
        dx1, dy1 = max(0, -dx), max(0, -dy)
        dx2 = dx1 + (sx2 - sx1)
        dy2 = dy1 + (sy2 - sy1)

        result[:, dy1:dy2, dx1:dx2] = rgb[:, sy1:sy2, sx1:sx2]
        return result

    @staticmethod
    def texture_randomization(rgb: torch.Tensor, noise_level: float = 0.15) -> torch.Tensor:
        """Gaussian noise to simulate texture variation."""
        noise = torch.randn_like(rgb) * noise_level
        return torch.clamp(rgb + noise, 0, 1)

    @staticmethod
    def combined(rgb: torch.Tensor) -> torch.Tensor:
        """Apply all randomizations together."""
        rgb = DomainRandomizer.lighting_change(rgb, intensity=0.2)
        rgb = DomainRandomizer.camera_angle(rgb, shift_pixels=10)
        rgb = DomainRandomizer.texture_randomization(rgb, noise_level=0.1)
        return rgb


# ======================================================================
# Stress Test
# ======================================================================
def evaluate_scenario(model, dataset, device, scenario_name, transform_fn=None):
    """Evaluate model under one domain randomization scenario."""
    model.eval()
    tp, fp, total_gt = 0, 0, 0
    total_conf = 0.0
    total_dets = 0
    num_samples = min(len(dataset), 200)

    for i in range(num_samples):
        sample = dataset[i]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        gt_labels = sample["labels"]

        # Apply transform
        if transform_fn:
            rgb = transform_fn(rgb.squeeze(0)).unsqueeze(0)

        with torch.no_grad():
            preds = model(rgb, depth)

        pred = preds[0]
        mask = pred["scores"] > 0.3
        pred_labels = pred["labels"][mask]
        pred_scores = pred["scores"][mask]

        n_gt = (gt_labels > 0).sum().item()
        total_gt += n_gt
        total_dets += len(pred_labels)
        if len(pred_scores) > 0:
            total_conf += pred_scores.mean().item()

        # Match predictions to GT
        gt_set = set(gt_labels[gt_labels > 0].numpy().tolist())
        pred_set = set(pred_labels.cpu().numpy().tolist())
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)

    accuracy = tp / max(total_gt, 1) * 100
    avg_det = total_dets / num_samples
    avg_conf = total_conf / num_samples * 100

    return {
        "scenario": scenario_name,
        "accuracy": accuracy,
        "avg_detections": avg_det,
        "avg_confidence": avg_conf,
    }


def run_stress_test(model, dataset, device, output_dir="evaluation"):
    """
    Part 3 Task 3.1: Domain Randomization Stress Test.

    Produces the required table:
        | Scenario | Accuracy | Drop % |
    """
    print("\n" + "=" * 60)
    print("  Domain Randomization Stress Test")
    print("=" * 60)

    scenarios = {
        "Baseline (clean)":       None,
        "Lighting Changes":       DomainRandomizer.lighting_change,
        "Camera Angle Shift":     DomainRandomizer.camera_angle,
        "Texture Randomization":  DomainRandomizer.texture_randomization,
        "Combined (all)":         DomainRandomizer.combined,
    }

    results = []
    baseline_acc = None

    for name, transform in scenarios.items():
        print(f"\n  Testing: {name}...")
        result = evaluate_scenario(model, dataset, device, name, transform)

        if baseline_acc is None:
            baseline_acc = result["accuracy"]
            result["drop_pct"] = 0.0
        else:
            result["drop_pct"] = baseline_acc - result["accuracy"]

        results.append(result)
        print(f"    Accuracy: {result['accuracy']:.1f}%  |  "
              f"Drop: {result['drop_pct']:.1f}%  |  "
              f"Confidence: {result['avg_confidence']:.1f}%")

    # ── Save results ──
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "stress_test_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table (required by assignment)
    table = """# Domain Randomization Stress Test Results

| Scenario | Accuracy (%) | Drop (%) | Avg Confidence (%) | Avg Detections |
|----------|:------------:|:--------:|:------------------:|:--------------:|
"""
    for r in results:
        table += (f"| {r['scenario']} | {r['accuracy']:.1f} | "
                  f"{r['drop_pct']:.1f} | {r['avg_confidence']:.1f} | "
                  f"{r['avg_detections']:.2f} |\n")

    table_path = os.path.join(output_dir, "stress_test_table.md")
    with open(table_path, "w") as f:
        f.write(table)

    print(f"\n  Saved → {json_path}")
    print(f"  Saved → {table_path}")
    return results


# ======================================================================
# Inference Visualization (GT vs Predictions)
# ======================================================================
def visualize_predictions(model, dataset, device, output_dir="evaluation", num_samples=6):
    """
    Part 5 Task 4: Before/After prediction frames.

    Shows: Input RGB | Ground Truth boxes | Predicted boxes
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    n = min(num_samples, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    colors = {0: "gray", 1: "red", 2: "blue", 3: "green", 4: "orange"}
    names = OrigamiMultimodalDataset.CLASS_NAMES

    for i in range(n):
        sample = dataset[i]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(rgb, depth)

        img = sample["rgb"].permute(1, 2, 0).numpy()
        H, W = img.shape[:2]
        pred = preds[0]

        # ── Col 0: Input RGB ──
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input RGB", fontsize=10)
        axes[i, 0].axis("off")

        # ── Col 1: Ground Truth ──
        axes[i, 1].imshow(img)
        gt_boxes = sample["bbox"].numpy()
        gt_labels = sample["labels"].numpy()
        for j in range(len(gt_labels)):
            if gt_labels[j] > 0:
                x1, y1, x2, y2 = gt_boxes[j] * [W, H, W, H]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=colors.get(gt_labels[j], "white"),
                    facecolor="none",
                )
                axes[i, 1].add_patch(rect)
                axes[i, 1].text(
                    x1, y1 - 2, names.get(gt_labels[j], "?"),
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=colors.get(gt_labels[j], "gray"), alpha=0.7),
                )
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis("off")

        # ── Col 2: Predictions ──
        axes[i, 2].imshow(img)
        p_boxes = pred["boxes"].cpu().numpy()
        p_labels = pred["labels"].cpu().numpy()
        p_scores = pred["scores"].cpu().numpy()
        for j in range(len(p_labels)):
            if p_scores[j] > 0.3:
                x1, y1, x2, y2 = p_boxes[j]
                lbl = p_labels[j]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=colors.get(lbl, "white"),
                    facecolor="none", linestyle="--",
                )
                axes[i, 2].add_patch(rect)
                axes[i, 2].text(
                    x1, y1 - 2, f"{names.get(lbl, '?')} {p_scores[j]:.2f}",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=colors.get(lbl, "gray"), alpha=0.7),
                )
        axes[i, 2].set_title("Predictions", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("Object Detection: Ground Truth vs Predictions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "inference_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved → {path}")
    return path


# ======================================================================
# Batch Visualization (RGB / Depth / Seg / BBox)
# ======================================================================
def visualize_batch_samples(dataset, output_dir="evaluation", num_samples=4):
    """
    Part 5 Task 2: Sample batch visualization.

    Shows 4 columns: RGB | Depth | Segmentation | BBox overlay
    """
    os.makedirs(output_dir, exist_ok=True)
    n = min(num_samples, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    colors = {1: "red", 2: "blue", 3: "green", 4: "orange"}

    for i in range(n):
        sample = dataset[i]
        img = sample["rgb"].permute(1, 2, 0).numpy()
        depth = sample["depth"].squeeze(0).numpy()
        seg = sample["segmentation"].numpy()
        H, W = img.shape[:2]

        # RGB
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("RGB", fontsize=10)
        axes[i, 0].axis("off")

        # Depth
        axes[i, 1].imshow(depth, cmap="plasma")
        axes[i, 1].set_title("Depth", fontsize=10)
        axes[i, 1].axis("off")

        # Segmentation
        axes[i, 2].imshow(seg, cmap="tab10", vmin=0, vmax=5)
        axes[i, 2].set_title("Segmentation", fontsize=10)
        axes[i, 2].axis("off")

        # BBox overlay
        axes[i, 3].imshow(img)
        bboxes = sample["bbox"].numpy()
        labels = sample["labels"].numpy()
        for j in range(len(labels)):
            if labels[j] > 0:
                x1, y1, x2, y2 = bboxes[j] * [W, H, W, H]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=colors.get(labels[j], "white"),
                    facecolor=colors.get(labels[j], "gray"), alpha=0.2,
                )
                axes[i, 3].add_patch(rect)
                border = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=colors.get(labels[j], "white"),
                    facecolor="none",
                )
                axes[i, 3].add_patch(border)
        axes[i, 3].set_title("BBox Overlay", fontsize=10)
        axes[i, 3].axis("off")

    plt.suptitle("Sample Batch Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "batch_visualization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved → {path}")
    return path


# ======================================================================
# Training Curves
# ======================================================================
def visualize_training_curves(history_path, output_dir="evaluation"):
    """
    Part 5 Task 3: Loss vs Epoch, mAP vs Epoch.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(len(history["train_loss"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mAP
    axes[1].plot(epochs, history["val_mAP"], "g-", label="Val mAP", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].set_title("mAP vs Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR
    axes[2].plot(epochs, history["lr"], "m-", label="LR", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Object Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved → {path}")
    return path


# ======================================================================
# CLI — run all evaluations
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate detection model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Build model ──
    model = build_model(num_classes=5, use_depth=True, device=device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}\n")

    # ── Load val dataset ──
    val_ds = OrigamiMultimodalDataset(
        hdf5_path=args.hdf5_path,
        config_path=args.config_path,
        split="val",
        max_episodes=args.max_episodes,
        synthetic_mode=True,
    )

    # ── Run all evaluations ──
    print("\n[1/4] Batch visualization...")
    visualize_batch_samples(val_ds, args.output_dir)

    print("\n[2/4] Stress test...")
    run_stress_test(model, val_ds, device, args.output_dir)

    print("\n[3/4] Inference visualization...")
    visualize_predictions(model, val_ds, device, args.output_dir)

    # Training curves (if history exists)
    history_path = os.path.join("checkpoints", "training_history.json")
    if os.path.exists(history_path):
        print("\n[4/4] Training curves...")
        visualize_training_curves(history_path, args.output_dir)
    else:
        print("\n[4/4] Skipping training curves (no history file found)")

    print(f"\n{'=' * 60}")
    print(f"  All evaluations complete")
    print(f"  Outputs → {args.output_dir}/")
    print(f"{'=' * 60}")
