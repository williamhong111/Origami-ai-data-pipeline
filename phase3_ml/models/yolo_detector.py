"""
yolo_detector.py — Object Detector with Multi-Modal Fusion (RGB + Depth)
==========================================================================
Part 2 of Phase 3: Baseline ML Model (Option A — Object Detection)

Architecture:
    - Backbone: ResNet50-FPN v2 (pretrained on COCO)
    - Detection head: Faster R-CNN
    - Fusion: Early fusion — RGB + Depth concatenated then projected

Two modes:
    1. RGB-only  (use_depth=False) — standard Faster R-CNN
    2. RGB+Depth (use_depth=True)  — multi-modal fusion

Fusion approach (per assignment requirement):
    fused = torch.cat([rgb_features, depth_features], dim=1)

Usage:
    from models.yolo_detector import build_model

    # RGB + Depth fusion
    model = build_model(num_classes=5, use_depth=True)

    # Training: returns loss dict
    losses = model(rgb, depth, targets)

    # Inference: returns prediction list
    predictions = model(rgb, depth)
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Optional


# ======================================================================
# Depth Encoder
# ======================================================================
class DepthEncoder(nn.Module):
    """
    Lightweight CNN to encode single-channel depth → 3-channel features.
    This allows depth information to be fused with RGB before the backbone.

    Architecture:
        [1] → Conv3x3 → BN → ReLU → [16]
        [16] → Conv3x3 → BN → ReLU → [32]
        [32] → Conv1x1 → BN → ReLU → [3]
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: [B, 1, H, W]
        Returns:
            depth_features: [B, 3, H, W]
        """
        return self.encoder(depth)


# ======================================================================
# Multi-Modal Fusion Module
# ======================================================================
class MultiModalFusion(nn.Module):
    """
    Early fusion: concatenate RGB (3ch) + Depth features (3ch) → 6ch,
    then project back to 3ch for the backbone.

    This is the baseline fusion method required by the assignment:
        fused = torch.cat([rgb_features, depth_features], dim=1)
    """

    def __init__(self):
        super().__init__()
        self.depth_encoder = DepthEncoder()
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:   [B, 3, H, W]
            depth: [B, 1, H, W]
        Returns:
            fused: [B, 3, H, W] — ready for backbone
        """
        depth_features = self.depth_encoder(depth)                    # [B, 3, H, W]
        fused = torch.cat([rgb, depth_features], dim=1)               # [B, 6, H, W]
        return self.fusion(fused)                                     # [B, 3, H, W]


# ======================================================================
# Main Detector
# ======================================================================
class YOLODetector(nn.Module):
    """
    Object detection model for Isaac Sim robotics scenes.

    Uses Faster R-CNN with ResNet50-FPN v2 backbone (pretrained on COCO),
    with optional multi-modal fusion of RGB + Depth.

    Why Faster R-CNN instead of raw YOLOv8:
        - Clean integration with PyTorch and our custom fusion module
        - torchvision provides well-tested, production-ready implementation
        - ResNet50-FPN backbone uses same multi-scale feature extraction
          philosophy as YOLO architectures
        - Easy to swap classification head for our 5 classes
    """

    def __init__(
        self,
        num_classes: int = 5,
        use_depth: bool = True,
        pretrained_backbone: bool = True,
    ):
        """
        Args:
            num_classes:         Number of classes (including background)
            use_depth:           Enable RGB + Depth fusion
            pretrained_backbone: Use ImageNet/COCO pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_depth = use_depth

        # ── Multi-modal fusion (only if depth enabled) ──
        if use_depth:
            self.fusion = MultiModalFusion()

        # ── Faster R-CNN with pretrained ResNet50-FPN v2 ──
        if pretrained_backbone:
            self.detector = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
        else:
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=None)

        # ── Replace classification head for our classes ──
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass.

        Args:
            rgb:     [B, 3, H, W] — normalized RGB images
            depth:   [B, 1, H, W] — depth maps (optional)
            targets: list of dicts with 'boxes' [N,4] and 'labels' [N]
                     (provide for training, omit for inference)

        Returns:
            Training mode:  dict of losses
                            {"loss_classifier", "loss_box_reg",
                             "loss_objectness", "loss_rpn_box_reg"}
            Inference mode: list of prediction dicts
                            [{"boxes", "labels", "scores"}, ...]
        """
        # ── Fuse RGB + Depth ──
        if self.use_depth and depth is not None:
            images = self.fusion(rgb, depth)
        else:
            images = rgb

        # ── Faster R-CNN expects list of images ──
        image_list = [img for img in images]

        if targets is not None:
            # Training → returns losses
            self.detector.train()
            return self.detector(image_list, targets)
        else:
            # Inference → returns predictions
            self.detector.eval()
            with torch.no_grad():
                return self.detector(image_list)


# ======================================================================
# Factory function
# ======================================================================
def build_model(
    num_classes: int = 5,
    use_depth: bool = True,
    pretrained: bool = True,
    device: str = "cpu",
) -> YOLODetector:
    """
    Build and configure the detection model.

    Args:
        num_classes: Detection classes (5 = bg + 3 cubes + robot_arm)
        use_depth:   Enable RGB + Depth fusion
        pretrained:  Use pretrained backbone
        device:      Target device ('cpu' or 'cuda')

    Returns:
        Configured YOLODetector on the specified device
    """
    model = YOLODetector(
        num_classes=num_classes,
        use_depth=use_depth,
        pretrained_backbone=pretrained,
    )
    model = model.to(device)

    # ── Print model summary ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in model.fusion.parameters()) if use_depth else 0

    print(f"\n[Model] YOLODetector")
    print(f"  Classes:          {num_classes}")
    print(f"  Depth fusion:     {use_depth}")
    print(f"  Pretrained:       {pretrained}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    if use_depth:
        print(f"  Fusion params:    {fusion_params:,}")
    print(f"  Device:           {device}\n")

    return model


# ======================================================================
# CLI — quick test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Testing YOLODetector")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Build model ──
    model = build_model(num_classes=5, use_depth=True, device=device)

    # ── Test inference ──
    print("Testing inference...")
    model.eval()
    rgb = torch.randn(2, 3, 256, 256).to(device)
    depth = torch.randn(2, 1, 256, 256).to(device)

    preds = model(rgb, depth)
    for i, p in enumerate(preds):
        print(f"  Image {i}: {p['boxes'].shape[0]} detections")
        if p["boxes"].shape[0] > 0:
            print(f"    Top box:   {p['boxes'][0].tolist()}")
            print(f"    Top label: {p['labels'][0].item()}")
            print(f"    Top score: {p['scores'][0].item():.4f}")

    # ── Test training ──
    print("\nTesting training...")
    model.train()
    targets = [
        {
            "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32).to(device),
            "labels": torch.tensor([1], dtype=torch.int64).to(device),
        },
        {
            "boxes": torch.tensor([[30, 30, 120, 120], [100, 100, 200, 200]], dtype=torch.float32).to(device),
            "labels": torch.tensor([2, 3], dtype=torch.int64).to(device),
        },
    ]

    losses = model(rgb, depth, targets)
    print(f"  Losses:")
    total_loss = 0
    for k, v in losses.items():
        print(f"    {k}: {v.item():.4f}")
        total_loss += v.item()
    print(f"    TOTAL: {total_loss:.4f}")

    print("\n  All tests passed.")
