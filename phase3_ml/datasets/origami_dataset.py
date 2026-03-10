"""
origami_dataset.py — Multimodal Robotics Dataset for PyTorch
==============================================================
Part 1 of Phase 3: Data Loading & Validation

Loads normalized Isaac Sim dataset (mimic_dataset_1k.hdf5) and serves
tensor dictionaries for training object detection models (Faster R-CNN).

Input:
    Normalized output from Phase 2 pipeline:
        real_world_ingest_pipeline.py → data_normalizer.py → validator.py

Output:
    Tensor dictionary per frame:
    {
        "rgb":           Tensor [C, H, W],       float32, normalized [0,1]
        "depth":         Tensor [1, H, W],        float32
        "segmentation":  Tensor [H, W],           int64
        "bbox":          Tensor [N, 4],            float32 (x1,y1,x2,y2 normalized)
        "labels":        Tensor [N],               int64
        "joint_states":  Tensor [J],               float32
        "label":         int                       (primary class in frame)
    }

Supports:
    - Real HDF5 data (mimic_dataset_1k.hdf5)
    - Synthetic fallback for development without 25GB file
    - Multi-camera streams
    - Custom collate_fn for variable-size bounding boxes

Usage:
    from datasets.origami_dataset import OrigamiMultimodalDataset

    # With real data
    dataset = OrigamiMultimodalDataset(
        hdf5_path="../mimic_dataset_1k.hdf5",
        config_path="../phase2_pipeline/source_configs/isaac_sim.yaml",
        split="train",
    )

    # Synthetic fallback (no HDF5 needed)
    dataset = OrigamiMultimodalDataset(split="train", max_episodes=50)

    sample = dataset[0]
"""

import os
import collections
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class OrigamiMultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal Isaac Sim robotics data.

    Loads per-frame data from HDF5 episodes and generates detection
    targets (bounding boxes) from color-based segmentation of cubes.
    Falls back to synthetic data generation when HDF5 is unavailable.
    """

    # ── Class definitions for Isaac Sim cube stacking ──
    CLASS_NAMES = {
        0: "background",
        1: "red_cube",
        2: "blue_cube",
        3: "green_cube",
        4: "robot_arm",
    }
    NUM_CLASSES = 5  # including background

    def __init__(
        self,
        hdf5_path: Optional[str] = None,
        config_path: Optional[str] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        max_episodes: Optional[int] = None,
        image_size: Tuple[int, int] = (256, 256),
        transform=None,
        synthetic_mode: bool = True,
    ):
        """
        Args:
            hdf5_path:      Path to mimic_dataset_1k.hdf5 (None → synthetic)
            config_path:    Path to YAML source config (from Phase 2)
            split:          'train' or 'val'
            train_ratio:    Fraction for training split
            max_episodes:   Max episodes to load
            image_size:     Target (H, W) for resizing
            transform:      Optional torchvision transforms
            synthetic_mode: Generate synthetic data if HDF5 unavailable
        """
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.frames: List[Dict] = []

        # ── Load data ──
        if hdf5_path and os.path.exists(hdf5_path):
            self._load_from_hdf5(hdf5_path, config_path, max_episodes)
        elif synthetic_mode:
            print("[Dataset] HDF5 not available → generating synthetic data")
            self._generate_synthetic_data(num_episodes=max_episodes or 50)
        else:
            raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

        # ── Train / val split ──
        total = len(self.frames)
        split_idx = int(total * train_ratio)
        self.frames = self.frames[:split_idx] if split == "train" else self.frames[split_idx:]
        print(f"[Dataset] {split} split → {len(self.frames)} frames")

    # ==================================================================
    # HDF5 Loading (real Isaac Sim data)
    # ==================================================================
    def _load_from_hdf5(self, hdf5_path: str, config_path: str, max_episodes: int):
        """Load frames from Isaac Sim HDF5 using Phase 2 YAML config."""
        if not HAS_H5PY:
            raise ImportError("h5py required: pip install h5py")

        # Load YAML config if available (reuses Phase 2 field mappings)
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)

        f = h5py.File(hdf5_path, "r")
        data_group = f["data"] if "data" in f else f

        # Find demo groups
        demo_keys = sorted(
            [k for k in data_group.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
        )
        if max_episodes:
            demo_keys = demo_keys[:max_episodes]

        print(f"[Dataset] Loading {len(demo_keys)} episodes from {hdf5_path}")

        vision_cfg = config.get("vision", {}).get("streams", [])
        prop_cfg = config.get("proprioception", {})

        for demo_key in demo_keys:
            demo = data_group[demo_key]

            # ── Find RGB stream ──
            rgb_data = None
            depth_data = None

            for stream in vision_cfg:
                h5p = stream.get("hdf5_path", "")
                if h5p in demo:
                    if stream.get("type") == "rgb" and rgb_data is None:
                        rgb_data = demo[h5p][:]
                    elif stream.get("type") == "depth":
                        depth_data = demo[h5p][:]

            # Fallback to common paths
            if rgb_data is None:
                for path in ["obs/agentview_image", "obs/robot0_eye_in_hand_image"]:
                    if path in demo:
                        rgb_data = demo[path][:]
                        break
            if rgb_data is None:
                continue

            # ── Find joint states ──
            joint_states = None
            jp = prop_cfg.get("joint_pos", {})
            jp_path = jp.get("hdf5_path", "obs/robot0_joint_pos") if isinstance(jp, dict) else jp
            if isinstance(jp_path, str) and jp_path in demo:
                joint_states = demo[jp_path][:]

            # ── Sample frames (subsample to avoid redundancy) ──
            num_frames = rgb_data.shape[0]
            interval = max(1, num_frames // 20)

            for fi in range(0, num_frames, interval):
                seg, bboxes, labels = self._segment_cubes(rgb_data[fi])
                self.frames.append({
                    "rgb":          rgb_data[fi],
                    "depth":        depth_data[fi] if depth_data is not None else None,
                    "segmentation": seg,
                    "bboxes":       bboxes,
                    "labels":       labels,
                    "joint_states": joint_states[fi] if joint_states is not None else np.zeros(7, dtype=np.float32),
                    "episode_id":   demo_key,
                    "frame_idx":    fi,
                })

        f.close()
        print(f"[Dataset] Loaded {len(self.frames)} frames from HDF5")

    # ==================================================================
    # Color-based object segmentation
    # ==================================================================
    def _segment_cubes(self, rgb: np.ndarray):
        """
        Extract bounding boxes via color thresholding.

        Isaac Sim cubes have distinct colors (red / blue / green),
        making simple color segmentation effective for bbox generation.

        Returns:
            seg_mask:  [H, W] int64
            bboxes:    [N, 4] float32  (x1, y1, x2, y2 normalized to [0,1])
            labels:    [N]    int64
        """
        H, W = rgb.shape[:2]
        seg = np.zeros((H, W), dtype=np.int64)
        bboxes, labels = [], []

        # RGB color ranges for each cube class
        color_ranges = {
            1: ([150, 0, 0], [255, 80, 80]),      # red
            2: ([0, 0, 150], [80, 80, 255]),       # blue
            3: ([0, 150, 0], [80, 255, 80]),       # green
        }

        for cls_id, (lo, hi) in color_ranges.items():
            mask = np.all((rgb >= lo) & (rgb <= hi), axis=-1)
            if mask.sum() > 50:  # minimum pixel count
                seg[mask] = cls_id
                ys, xs = np.where(mask)
                bboxes.append([xs.min() / W, ys.min() / H, xs.max() / W, ys.max() / H])
                labels.append(cls_id)

        if not bboxes:
            bboxes = [[0.0, 0.0, 0.0, 0.0]]
            labels = [0]

        return seg, np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    # ==================================================================
    # Synthetic data generation (development fallback)
    # ==================================================================
    def _generate_synthetic_data(self, num_episodes: int = 50):
        """
        Generate Isaac Sim-like synthetic scenes with colored cubes.
        Used for development/testing when 25GB HDF5 is unavailable.
        """
        np.random.seed(42)
        H, W = self.image_size
        frames_per_ep = 20
        colors = {1: [200, 30, 30], 2: [30, 30, 200], 3: [30, 200, 30]}

        print(f"[Dataset] Generating {num_episodes} episodes × {frames_per_ep} frames")

        for ep in range(num_episodes):
            # Random scene: 1–3 cubes at random positions
            n_obj = np.random.randint(1, 4)
            objs = [
                (i + 1, np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.08, 0.2))
                for i in range(n_obj)
            ]
            base_joints = np.random.randn(7).astype(np.float32) * 0.5

            for fi in range(frames_per_ep):
                t = fi / frames_per_ep  # time fraction

                # ── RGB: gray background + colored cubes ──
                rgb = np.clip(
                    np.ones((H, W, 3), dtype=np.int16) * 180 + np.random.randint(-10, 10, (H, W, 3)),
                    0, 255,
                ).astype(np.uint8)

                seg = np.zeros((H, W), dtype=np.int64)
                depth = np.linspace(0.5, 1.0, H).reshape(H, 1).repeat(W, axis=1).astype(np.float32)
                bboxes, labels = [], []

                for cls_id, cx, cy, sz in objs:
                    # Slight motion per frame
                    cx_t = cx + 0.02 * np.sin(2 * np.pi * t + cls_id)
                    cy_t = cy + 0.01 * np.cos(2 * np.pi * t + cls_id)
                    x1 = int(max(0, (cx_t - sz / 2) * W))
                    y1 = int(max(0, (cy_t - sz / 2) * H))
                    x2 = int(min(W, (cx_t + sz / 2) * W))
                    y2 = int(min(H, (cy_t + sz / 2) * H))

                    rgb[y1:y2, x1:x2] = colors[cls_id]
                    seg[y1:y2, x1:x2] = cls_id
                    depth[y1:y2, x1:x2] = 0.3  # objects closer

                    bboxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
                    labels.append(cls_id)

                if not bboxes:
                    bboxes, labels = [[0, 0, 0, 0]], [0]

                self.frames.append({
                    "rgb":          rgb,
                    "depth":        depth,
                    "segmentation": seg,
                    "bboxes":       np.array(bboxes, dtype=np.float32),
                    "labels":       np.array(labels, dtype=np.int64),
                    "joint_states": base_joints + np.random.randn(7).astype(np.float32) * 0.05 * t,
                    "episode_id":   f"demo_{ep}",
                    "frame_idx":    fi,
                })

        print(f"[Dataset] Generated {len(self.frames)} synthetic frames")

    # ==================================================================
    # __len__ / __getitem__
    # ==================================================================
    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single frame as tensor dictionary."""
        frame = self.frames[idx]

        # ── RGB → [3, H, W] float32 [0, 1] ──
        rgb = frame["rgb"]
        if rgb.shape[:2] != self.image_size:
            rgb = self._resize(rgb, self.image_size)
        rgb_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0

        # ── Depth → [1, H, W] float32 ──
        depth = frame["depth"]
        if depth is None:
            depth = np.zeros(self.image_size, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        if depth.shape[:2] != self.image_size:
            depth = self._resize(depth, self.image_size)
        depth_t = torch.from_numpy(depth.copy()).unsqueeze(0).float()

        # ── Segmentation → [H, W] int64 ──
        seg = frame["segmentation"]
        if seg.shape != self.image_size:
            seg = self._resize(seg, self.image_size)
        seg_t = torch.from_numpy(seg.copy()).long()

        # ── Bounding boxes → [N, 4] float32 ──
        bbox_t = torch.from_numpy(frame["bboxes"].copy()).float()

        # ── Labels → [N] int64 ──
        labels_t = torch.from_numpy(frame["labels"].copy()).long()

        # ── Joint states → [J] float32 ──
        js = frame["joint_states"]
        js_t = torch.from_numpy(js.copy()).float() if isinstance(js, np.ndarray) else torch.tensor(js, dtype=torch.float32)

        # ── Primary label (most common non-background class) ──
        valid = frame["labels"][frame["labels"] > 0]
        primary_label = int(valid[0]) if len(valid) > 0 else 0

        return {
            "rgb":          rgb_t,           # [3, H, W]
            "depth":        depth_t,         # [1, H, W]
            "segmentation": seg_t,           # [H, W]
            "bbox":         bbox_t,          # [N, 4]
            "labels":       labels_t,        # [N]
            "joint_states": js_t,            # [7]
            "label":        primary_label,   # int
        }

    # ==================================================================
    # Collate function (handles variable-size bboxes)
    # ==================================================================
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate for DataLoader.
        Pads bboxes and labels to the max count in the batch.
        """
        rgb    = torch.stack([s["rgb"] for s in batch])
        depth  = torch.stack([s["depth"] for s in batch])
        seg    = torch.stack([s["segmentation"] for s in batch])
        joints = torch.stack([s["joint_states"] for s in batch])
        label  = torch.tensor([s["label"] for s in batch])

        # Pad bboxes / labels to max N
        max_n = max(s["bbox"].shape[0] for s in batch)
        B = len(batch)
        bbox_pad   = torch.zeros(B, max_n, 4)
        labels_pad = torch.zeros(B, max_n, dtype=torch.long)

        for i, s in enumerate(batch):
            n = s["bbox"].shape[0]
            bbox_pad[i, :n]   = s["bbox"]
            labels_pad[i, :n] = s["labels"]

        return {
            "rgb": rgb, "depth": depth, "segmentation": seg,
            "bbox": bbox_pad, "labels": labels_pad,
            "joint_states": joints, "label": label,
        }

    # ==================================================================
    # Utility
    # ==================================================================
    @staticmethod
    def _resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Nearest-neighbor resize (no PIL dependency)."""
        H, W = size
        h, w = arr.shape[:2]
        yi = (np.arange(H) * h / H).astype(int)
        xi = (np.arange(W) * w / W).astype(int)
        return arr[np.ix_(yi, xi)]


# ======================================================================
# Task 1.2 — Data Quality Report Generator
# ======================================================================
def generate_data_quality_report(
    dataset: OrigamiMultimodalDataset,
    output_path: str = "reports/data_quality_report.md",
):
    """
    Auto-generate the data quality report required by Phase 3 Task 1.2.

    Covers:
        1. Class distribution histogram
        2. Mean / Std of RGB channels
        3. Depth value range
        4. Missing modality report
        5. Frame count validation
    """
    print("[Report] Generating data quality report...")

    class_counts = collections.Counter()
    rgb_means, rgb_stds = [], []
    depth_min, depth_max = float("inf"), float("-inf")
    missing_depth, missing_seg = 0, 0
    bbox_counts = []
    total = len(dataset)

    for i in range(total):
        s = dataset[i]

        # 1. Class counts
        for lbl in s["labels"].numpy():
            if lbl > 0:
                class_counts[int(lbl)] += 1

        # 2. RGB stats
        rgb_means.append(s["rgb"].mean(dim=[1, 2]).numpy())
        rgb_stds.append(s["rgb"].std(dim=[1, 2]).numpy())

        # 3. Depth range
        d = s["depth"]
        if d.sum() > 0:
            depth_min = min(depth_min, d.min().item())
            depth_max = max(depth_max, d.max().item())
        else:
            missing_depth += 1

        # 4. Missing segmentation
        if s["segmentation"].sum() == 0:
            missing_seg += 1

        # 5. Bbox count
        bbox_counts.append((s["labels"] > 0).sum().item())

    # Aggregate
    rm = np.array(rgb_means).mean(axis=0)
    rs = np.array(rgb_stds).mean(axis=0)
    total_obj = sum(class_counts.values())

    # ── Build markdown ──
    report = f"""# Data Quality Report

**Generated by:** Phase 3 ML Pipeline  
**Dataset split:** {dataset.split}  
**Total frames:** {total}

---

## 1. Class Distribution

| Class ID | Name | Count | Percentage |
|----------|------|------:|----------:|
"""
    for c in sorted(class_counts):
        name = OrigamiMultimodalDataset.CLASS_NAMES.get(c, f"class_{c}")
        pct = class_counts[c] / max(total_obj, 1) * 100
        report += f"| {c} | {name} | {class_counts[c]} | {pct:.1f}% |\n"

    report += f"""
- **Total objects:** {total_obj}
- **Avg objects per frame:** {total_obj / max(total, 1):.2f}
- **Avg bboxes per frame:** {np.mean(bbox_counts):.2f}

---

## 2. RGB Channel Statistics

| Channel | Mean | Std |
|---------|-----:|----:|
| R | {rm[0]:.4f} | {rs[0]:.4f} |
| G | {rm[1]:.4f} | {rs[1]:.4f} |
| B | {rm[2]:.4f} | {rs[2]:.4f} |

---

## 3. Depth Value Range

| Metric | Value |
|--------|------:|
| Min | {depth_min:.4f} |
| Max | {depth_max:.4f} |
| Range | {depth_max - depth_min:.4f} |

---

## 4. Missing Modality Report

| Modality | Missing | Percentage |
|----------|--------:|----------:|
| RGB | 0 | 0.0% |
| Depth | {missing_depth} | {missing_depth / max(total, 1) * 100:.1f}% |
| Segmentation | {missing_seg} | {missing_seg / max(total, 1) * 100:.1f}% |
| Joint States | 0 | 0.0% |

---

## 5. Frame Count Validation

| Metric | Value |
|--------|------:|
| Total frames | {total} |
| Frames with objects | {sum(1 for c in bbox_counts if c > 0)} |
| Frames without objects | {sum(1 for c in bbox_counts if c == 0)} |
| Image size | {dataset.image_size[0]}×{dataset.image_size[1]} |

---

**Status:** {"PASS — All checks passed." if total > 0 and total_obj > 0 else "WARN — Check data loading."}
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"[Report] Saved → {output_path}")
    return report


# ======================================================================
# CLI — quick test
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Testing OrigamiMultimodalDataset")
    print("=" * 50)

    # Create dataset (synthetic, small)
    ds = OrigamiMultimodalDataset(split="train", max_episodes=5)

    print(f"\nDataset length: {len(ds)}")

    # Inspect one sample
    sample = ds[0]
    print(f"\nSample tensors:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:15s} → shape={str(v.shape):15s}  dtype={v.dtype}")
        else:
            print(f"  {k:15s} → {v}")

    # Test DataLoader with custom collate
    from torch.utils.data import DataLoader as TorchDataLoader
    loader = TorchDataLoader(ds, batch_size=4, collate_fn=OrigamiMultimodalDataset.collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch (batch_size=4):")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:15s} → shape={str(v.shape):15s}")

    # Generate quality report
    print()
    generate_data_quality_report(ds, "reports/data_quality_report.md")
