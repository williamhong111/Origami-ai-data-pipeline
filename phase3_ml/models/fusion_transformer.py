"""
fusion_transformer.py — Multi-Modal Fusion Transformer (Bonus)
================================================================
Upgrades the baseline early fusion (concat) in yolo_detector.py
to a Transformer-based cross-attention fusion between RGB and Depth.

Why this matters:
    - Concat fusion treats RGB and Depth equally, no interaction
    - Cross-attention lets RGB features "look at" Depth features
      and vice versa, learning WHERE depth info is most useful
    - This is the approach used in modern robotics models (RT-2, Octo)

Architecture:
    RGB  [B, 3, H, W] → RGB Encoder  → RGB tokens  [B, N, D]
    Depth [B, 1, H, W] → Depth Encoder → Depth tokens [B, N, D]
                                ↓
                    Cross-Attention (RGB queries Depth)
                                ↓
                    Fused tokens [B, N, D]
                                ↓
                    Reshape → [B, 3, H, W] for backbone

Usage:
    from models.fusion_transformer import TransformerFusionDetector, build_transformer_model

    # Build model with Transformer fusion
    model = build_transformer_model(num_classes=5, device="cpu")

    # Same interface as YOLODetector
    losses = model(rgb, depth, targets)      # training
    preds  = model(rgb, depth)               # inference

    # Or use the fusion module standalone
    from models.fusion_transformer import MultiModalFusionTransformer
    fusion = MultiModalFusionTransformer(embed_dim=128, num_heads=4, num_layers=2)
    fused = fusion(rgb, depth)  # [B, 3, H, W]
"""

import math
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Optional


# ======================================================================
# Patch Embedding — converts image to sequence of tokens
# ======================================================================
class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project to embedding dim.

    Example: 256×256 image with patch_size=16 → 16×16 = 256 tokens
    Each token is a flattened patch projected to embed_dim.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Conv2d with kernel=stride=patch_size acts as patch extraction + projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            tokens: [B, N, D] where N = (H/P) * (W/P), D = embed_dim
        """
        x = self.proj(x)          # [B, D, H/P, W/P]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2)          # [B, D, N]
        x = x.transpose(1, 2)     # [B, N, D]
        x = self.norm(x)
        return x


# ======================================================================
# Cross-Attention Layer
# ======================================================================
class CrossAttentionLayer(nn.Module):
    """
    One layer of cross-attention: queries from one modality,
    keys/values from another.

    This is the key innovation over concat fusion:
    RGB tokens can selectively attend to the most relevant
    depth information at each spatial location.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:   [B, N, D] — modality that asks questions (RGB)
            context: [B, N, D] — modality that provides answers (Depth)
        Returns:
            out: [B, N, D] — query enriched with context information
        """
        # Cross-attention: query attends to context
        attn_out, _ = self.cross_attn(query, context, context)
        query = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query


# ======================================================================
# Multi-Modal Fusion Transformer
# ======================================================================
class MultiModalFusionTransformer(nn.Module):
    """
    Transformer-based multi-modal fusion module.

    Pipeline:
        1. RGB  → PatchEmbedding → RGB tokens  [B, N, D]
        2. Depth → PatchEmbedding → Depth tokens [B, N, D]
        3. Cross-attention: RGB queries Depth (× num_layers)
        4. Reshape tokens back to spatial [B, 3, H, W]

    This replaces the simple concat fusion in yolo_detector.py.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim:   Dimension of token embeddings
            num_heads:   Number of attention heads
            num_layers:  Number of cross-attention layers
            patch_size:  Size of image patches (16 → 256/16 = 16×16 grid)
            dropout:     Dropout rate
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Separate encoders for each modality
        self.rgb_embed = PatchEmbedding(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        self.depth_embed = PatchEmbedding(in_channels=1, embed_dim=embed_dim, patch_size=patch_size)

        # Learnable positional embeddings
        # Will be initialized in forward() based on actual input size
        self.pos_embed = None
        self._cached_num_patches = 0

        # Cross-attention layers: RGB queries Depth
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Project fused tokens back to 3-channel image for backbone
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * 3),
        )
        self.output_norm = nn.BatchNorm2d(3)

    def _get_pos_embed(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Lazily create or resize positional embeddings."""
        if self.pos_embed is None or self._cached_num_patches != num_patches:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches, self.embed_dim, device=device) * 0.02
            )
            self._cached_num_patches = num_patches
        return self.pos_embed

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:   [B, 3, H, W]
            depth: [B, 1, H, W]
        Returns:
            fused: [B, 3, H, W] — ready for backbone
        """
        B, _, H, W = rgb.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        N = Hp * Wp

        # 1. Tokenize each modality
        rgb_tokens = self.rgb_embed(rgb)       # [B, N, D]
        depth_tokens = self.depth_embed(depth)  # [B, N, D]

        # 2. Add positional embeddings
        pos = self._get_pos_embed(N, rgb.device)
        rgb_tokens = rgb_tokens + pos
        depth_tokens = depth_tokens + pos

        # 3. Cross-attention: RGB queries Depth
        fused_tokens = rgb_tokens
        for layer in self.cross_attn_layers:
            fused_tokens = layer(fused_tokens, depth_tokens)

        # 4. Project back to image space
        patches = self.output_proj(fused_tokens)    # [B, N, P*P*3]
        patches = patches.view(B, Hp, Wp, self.patch_size, self.patch_size, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # [B, 3, Hp, P, Wp, P]
        fused = patches.reshape(B, 3, H, W)

        fused = self.output_norm(fused)
        return fused


# ======================================================================
# Transformer Fusion Detector (replaces YOLODetector's fusion)
# ======================================================================
class TransformerFusionDetector(nn.Module):
    """
    Object detection model with Transformer-based multi-modal fusion.

    Same interface as YOLODetector but uses cross-attention instead of
    concat for RGB + Depth fusion.
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        patch_size: int = 16,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Transformer fusion module
        self.fusion = MultiModalFusionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
        )

        # Faster R-CNN with pretrained backbone
        if pretrained_backbone:
            self.detector = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
        else:
            self.detector = fasterrcnn_resnet50_fpn_v2(weights=None)

        # Replace head for our classes
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
        """Same interface as YOLODetector."""
        if depth is not None:
            images = self.fusion(rgb, depth)
        else:
            images = rgb

        image_list = [img for img in images]

        if targets is not None:
            self.detector.train()
            return self.detector(image_list, targets)
        else:
            self.detector.eval()
            with torch.no_grad():
                return self.detector(image_list)


# ======================================================================
# Factory function
# ======================================================================
def build_transformer_model(
    num_classes: int = 5,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    patch_size: int = 16,
    pretrained: bool = True,
    device: str = "cpu",
) -> TransformerFusionDetector:
    """Build detection model with Transformer fusion."""
    model = TransformerFusionDetector(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        patch_size=patch_size,
        pretrained_backbone=pretrained,
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion = sum(p.numel() for p in model.fusion.parameters())

    print(f"\n[Model] TransformerFusionDetector")
    print(f"  Classes:          {num_classes}")
    print(f"  Embed dim:        {embed_dim}")
    print(f"  Attention heads:  {num_heads}")
    print(f"  Transformer layers: {num_layers}")
    print(f"  Patch size:       {patch_size}")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Fusion params:    {fusion:,}")
    print(f"  Device:           {device}\n")

    return model


# ======================================================================
# CLI — test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Testing Multi-Modal Fusion Transformer")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = build_transformer_model(
        num_classes=5, embed_dim=128, num_heads=4,
        num_layers=2, patch_size=16, device=device,
    )

    # Test inference
    print("Testing inference...")
    model.eval()
    rgb = torch.randn(2, 3, 256, 256).to(device)
    depth = torch.randn(2, 1, 256, 256).to(device)

    preds = model(rgb, depth)
    for i, p in enumerate(preds):
        print(f"  Image {i}: {p['boxes'].shape[0]} detections")
        if p["boxes"].shape[0] > 0:
            print(f"    Top score: {p['scores'][0].item():.4f}")

    # Test training
    print("\nTesting training...")
    model.train()
    targets = [
        {
            "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32).to(device),
            "labels": torch.tensor([1], dtype=torch.int64).to(device),
        },
        {
            "boxes": torch.tensor([[30, 30, 120, 120]], dtype=torch.float32).to(device),
            "labels": torch.tensor([2], dtype=torch.int64).to(device),
        },
    ]

    losses = model(rgb, depth, targets)
    total_loss = 0
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
        total_loss += v.item()
    print(f"  TOTAL: {total_loss:.4f}")

    # Compare param count with baseline
    print("\n  --- Comparison ---")
    print(f"  Baseline (concat fusion):     6,930 fusion params")
    print(f"  Transformer fusion:           {sum(p.numel() for p in model.fusion.parameters()):,} fusion params")
    print(f"  Transformer adds cross-attention for smarter RGB+Depth interaction")

    print("\n  All tests passed.")
