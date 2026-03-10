"""
train.py — Training Pipeline for Object Detection
====================================================
Part 2 of Phase 3: Training Pipeline

Implements:
    - Train loop with gradient clipping
    - Validation loop with mAP evaluation
    - Early stopping
    - Model checkpoint saving (best + periodic)
    - TensorBoard logging (loss, mAP, learning rate)

Usage:
    cd ~/Desktop/files/phase3_ml
    source origami_env/bin/activate

    # Train with synthetic data (no HDF5 needed)
    python3 training/train.py --epochs 15 --batch-size 4

    # Train with real Isaac Sim data
    python3 training/train.py \
        --hdf5-path ../mimic_dataset_1k.hdf5 \
        --config-path ../phase2_pipeline/source_configs/isaac_sim.yaml \
        --epochs 30 --batch-size 4

    # Resume from checkpoint
    python3 training/train.py --resume checkpoints/best_model.pth
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.origami_dataset import OrigamiMultimodalDataset
from models.yolo_detector import build_model


# ======================================================================
# Early Stopping
# ======================================================================
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ======================================================================
# Target preparation
# ======================================================================
def prepare_targets(batch, device):
    """
    Convert batch dict → list of target dicts for Faster R-CNN.

    Faster R-CNN expects:
        [{"boxes": [N, 4] absolute pixels, "labels": [N] int64}, ...]
    """
    B = batch["rgb"].shape[0]
    H, W = batch["rgb"].shape[2], batch["rgb"].shape[3]
    targets = []

    for i in range(B):
        boxes = batch["bbox"][i]       # [max_N, 4] normalized
        labels = batch["labels"][i]    # [max_N]

        # Filter out padding (label == 0)
        valid = labels > 0
        boxes = boxes[valid]
        labels = labels[valid]

        if len(boxes) == 0:
            # Faster R-CNN needs at least one box
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
            labels = torch.tensor([0], dtype=torch.int64, device=device)
        else:
            # Normalized [x1,y1,x2,y2] → absolute pixels
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= W
            boxes[:, [1, 3]] *= H
            # Ensure valid boxes (x2 > x1, y2 > y1)
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=boxes[:, 0] + 1)
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=boxes[:, 1] + 1)
            boxes = boxes.to(device)
            labels = labels.to(device)

        targets.append({"boxes": boxes, "labels": labels})

    return targets


# ======================================================================
# IoU + simple mAP
# ======================================================================
def box_iou(box1, box2):
    """IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def compute_map(predictions, targets, iou_threshold=0.5):
    """
    Simplified mAP: match predictions to ground truth by IoU,
    compute precision/recall, return F1 as mAP proxy.
    """
    tp, fp, total_gt = 0, 0, 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu()
        pred_labels = pred["labels"].cpu()
        pred_scores = pred["scores"].cpu()
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        total_gt += len(gt_boxes)

        # Filter low confidence
        mask = pred_scores > 0.3
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        matched = set()
        for i in range(len(pred_boxes)):
            best_iou, best_j = 0, -1
            for j in range(len(gt_boxes)):
                if j in matched:
                    continue
                iou = box_iou(pred_boxes[i], gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0 and pred_labels[i] == gt_labels[best_j]:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    return 2 * precision * recall / max(precision + recall, 1e-6)


# ======================================================================
# Train one epoch
# ======================================================================
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None):
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        targets = prepare_targets(batch, device)

        # Forward
        loss_dict = model(rgb, depth, targets)
        loss = sum(loss_dict.values())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_comp = {k: v / max(num_batches, 1) for k, v in loss_components.items()}

    # TensorBoard
    if writer:
        writer.add_scalar("train/total_loss", avg_loss, epoch)
        for k, v in avg_comp.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    return avg_loss, avg_comp


# ======================================================================
# Validate
# ======================================================================
@torch.no_grad()
def validate(model, dataloader, device, epoch, writer=None):
    """Run validation: compute loss + mAP."""
    total_loss = 0.0
    num_batches = 0
    all_preds, all_targets = [], []

    for batch in dataloader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        targets = prepare_targets(batch, device)

        # Get losses (train mode needed)
        model.train()
        with torch.enable_grad():
            loss_dict = model(rgb, depth, targets)
            total_loss += sum(v.item() for v in loss_dict.values())
        num_batches += 1

        # Get predictions (eval mode)
        model.eval()
        preds = model(rgb, depth)
        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = total_loss / max(num_batches, 1)
    mAP = compute_map(all_preds, all_targets)

    if writer:
        writer.add_scalar("val/total_loss", avg_loss, epoch)
        writer.add_scalar("val/mAP", mAP, epoch)

    return avg_loss, mAP


# ======================================================================
# Checkpoint
# ======================================================================
def save_checkpoint(model, optimizer, epoch, loss, mAP, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "mAP": mAP,
    }, path)
    print(f"    [Checkpoint] Saved → {path}")


# ======================================================================
# Main training loop
# ======================================================================
def train(config: dict):
    """
    Full training pipeline.

    Args:
        config: dict with all training hyperparameters
    """
    print("=" * 60)
    print("  Phase 3 — Object Detection Training")
    print("=" * 60)
    start = time.time()

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── 1. Dataset ──
    print("\n[1/4] Loading dataset...")
    train_ds = OrigamiMultimodalDataset(
        hdf5_path=config.get("hdf5_path"),
        config_path=config.get("config_path"),
        split="train",
        max_episodes=config.get("max_episodes", 50),
        image_size=(config["image_size"], config["image_size"]),
        synthetic_mode=True,
    )
    val_ds = OrigamiMultimodalDataset(
        hdf5_path=config.get("hdf5_path"),
        config_path=config.get("config_path"),
        split="val",
        max_episodes=config.get("max_episodes", 50),
        image_size=(config["image_size"], config["image_size"]),
        synthetic_mode=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=OrigamiMultimodalDataset.collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        collate_fn=OrigamiMultimodalDataset.collate_fn, num_workers=0,
    )
    print(f"  Train: {len(train_ds)} frames, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} frames, {len(val_loader)} batches")

    # ── 2. Model ──
    print("\n[2/4] Building model...")
    model = build_model(
        num_classes=config.get("num_classes", 5),
        use_depth=config.get("use_depth", True),
        pretrained=config.get("pretrained", True),
        device=device,
    )

    # ── 3. Optimizer + scheduler ──
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=config.get("weight_decay", 0.0005),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("lr_step_size", 10),
        gamma=0.1,
    )
    early_stopping = EarlyStopping(patience=config.get("patience", 7))

    # TensorBoard
    log_dir = config.get("log_dir", "runs/detection")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"  TensorBoard → {log_dir}")

    # Resume
    start_epoch = 0
    if config.get("resume") and os.path.exists(config["resume"]):
        ckpt = torch.load(config["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {start_epoch}")

    # ── 4. Train ──
    print(f"\n[3/4] Training for {config['epochs']} epochs...\n")

    best_mAP = 0.0
    history = {"train_loss": [], "val_loss": [], "val_mAP": [], "lr": []}
    ckpt_dir = config.get("checkpoint_dir", "checkpoints")

    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()
        print(f"  Epoch {epoch}/{config['epochs'] - 1}")
        print(f"  {'─' * 40}")

        # Train
        train_loss, train_comp = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, mAP = validate(model, val_loader, device, epoch, writer)

        # Scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mAP"].append(mAP)
        history["lr"].append(lr)
        writer.add_scalar("train/lr", lr, epoch)

        dt = time.time() - t0
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    mAP:        {mAP:.4f}")
        print(f"    LR:         {lr:.6f}")
        print(f"    Time:       {dt:.1f}s\n")

        # Save best
        if mAP > best_mAP:
            best_mAP = mAP
            save_checkpoint(model, optimizer, epoch, val_loss, mAP,
                            os.path.join(ckpt_dir, "best_model.pth"))

        # Save periodic
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, mAP,
                            os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        # Early stopping
        if early_stopping.step(val_loss):
            print(f"  Early stopping at epoch {epoch} (patience={early_stopping.patience})")
            break

    # ── Save final ──
    save_checkpoint(model, optimizer, epoch, val_loss, mAP,
                    os.path.join(ckpt_dir, "last_model.pth"))

    # Save history
    history_path = os.path.join(ckpt_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    writer.close()

    total_time = time.time() - start
    print(f"{'=' * 60}")
    print(f"  Training Complete")
    print(f"{'=' * 60}")
    print(f"  Time:           {total_time:.1f}s")
    print(f"  Best mAP:       {best_mAP:.4f}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Checkpoints:    {ckpt_dir}/")
    print(f"  History:        {history_path}")
    print(f"  TensorBoard:    {log_dir}/")
    print(f"{'=' * 60}\n")

    return model, history


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--use-depth", action="store_true", default=True)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/detection")
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    config = {
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "image_size":     args.image_size,
        "max_episodes":   args.max_episodes,
        "use_depth":      not args.no_depth,
        "pretrained":     True,
        "num_classes":    5,
        "hdf5_path":     args.hdf5_path,
        "config_path":   args.config_path,
        "resume":        args.resume,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir":       args.log_dir,
        "patience":      args.patience,
        "weight_decay":  0.0005,
        "lr_step_size":  10,
    }

    train(config)
