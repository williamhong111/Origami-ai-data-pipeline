"""
train_transformer.py — Train the Bonus Transformer Fusion Model
=================================================================
Run from phase3_ml/:
    source ~/masterEnv/bin/activate
    pip install tensorboard  (if not installed)
    python3 train_transformer.py
"""

import sys, os, time, json
sys.path.insert(0, ".")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.origami_dataset import OrigamiMultimodalDataset
from models.fusion_transformer import build_transformer_model
from training.train import train_one_epoch, validate, save_checkpoint

# ── Device ──
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}\n")

# ── Dataset ──
train_ds = OrigamiMultimodalDataset(split="train", max_episodes=5, image_size=(256, 256))
val_ds = OrigamiMultimodalDataset(split="val", max_episodes=5, image_size=(256, 256))

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=OrigamiMultimodalDataset.collate_fn)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=OrigamiMultimodalDataset.collate_fn)

# ── Model ──
model = build_transformer_model(num_classes=5, device=device)

# ── Optimizer ──
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
writer = SummaryWriter("runs/transformer")

# ── Train ──
history = {"train_loss": [], "val_loss": [], "val_mAP": [], "lr": []}
best_mAP = 0

for epoch in range(3):
    t0 = time.time()
    print(f"Epoch {epoch}/2")
    print("-" * 40)

    train_loss, _ = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
    val_loss, mAP = validate(model, val_loader, device, epoch, writer)
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_mAP"].append(mAP)
    history["lr"].append(lr)

    dt = time.time() - t0
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  mAP:        {mAP:.4f}")
    print(f"  Time:       {dt:.0f}s\n")

    if mAP > best_mAP:
        best_mAP = mAP
        save_checkpoint(model, optimizer, epoch, val_loss, mAP, "checkpoints_transformer/best_model.pth")

# ── Save history ──
os.makedirs("checkpoints_transformer", exist_ok=True)
with open("checkpoints_transformer/training_history.json", "w") as f:
    json.dump(history, f, indent=2)
writer.close()

print("=" * 50)
print(f"  Transformer Training Complete")
print(f"  Best mAP: {best_mAP:.4f}")
print(f"  Checkpoints: checkpoints_transformer/")
print("=" * 50)
