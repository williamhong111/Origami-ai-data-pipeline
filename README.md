# 🤖 Origami AI — Multimodal Robotics Learning Pipeline

An end-to-end ML pipeline that ingests multimodal robot sensor data, trains object detection models, and evaluates sim-to-real transfer readiness. Built for robotics foundation model development using NVIDIA Isaac Sim simulation data.

**Built at InGen Dynamics**  
**Author:** William Hong — ML Engineering Intern

---

## Results

| Metric | Value |
|--------|-------|
| **Best mAP** | 0.930 |
| **Baseline Accuracy** | 98.8% |
| **Stress Test (worst case)** | 95.2% (camera angle shift) |
| **Inference Confidence** | 0.99–1.00 |
| **Training Data** | NVIDIA Isaac Sim (Franka Panda cube stacking) |

<p align="center">
  <img src="docs/inference_visualization.png" width="80%" alt="Inference Demo">
</p>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Origami AI Pipeline                       │
│                                                                   │
│  Phase 1              Phase 2                 Phase 3             │
│  ────────             ────────                ────────            │
│  Schema               Ingestion               ML Training         │
│  Definition           Pipeline                Pipeline            │
│                                                                   │
│  data_schema_v1.json  HDF5 → Normalize       Dataset → Model     │
│  metadata specs       → Validate → JSON      → Train → Evaluate  │
│                       YAML-driven            Faster R-CNN +       │
│                       Zero-code onboarding   Depth Fusion         │
└─────────────────────────────────────────────────────────────────┘

Data Flow:
  mimic_dataset_1k.hdf5 ──→ Phase 2 (validate) ──→ Phase 3 (train)
                              ↓                       ↓
                         output/*.json          best_model.pth
                         (metadata)             (mAP 0.93)
```

---

## Project Structure

```
origami-ai/
│
├── schema/                              # Phase 1 — Data Schema
│   ├── data_schema_v1.json              # Canonical multimodal schema
│   ├── metadata_annotations_spec.md     # Field semantics & rules
│   └── examples/
│       └── sample_example_v1.json
│
├── pipeline/                            # Phase 2 — Data Ingestion
│   ├── real_world_ingest_pipeline.py    # Main orchestrator
│   ├── dataloader.py                    # Universal config-driven reader
│   ├── data_normalizer.py              # Timestamp & sampling normalization
│   ├── schema_packer.py                # Maps data → schema JSON
│   ├── validator.py                    # Schema compliance checker
│   └── source_configs/
│       ├── isaac_sim.yaml              # NVIDIA Isaac Sim config
│       └── rt1.yaml                    # Google RT-1 config
│
├── ml/                                  # Phase 3 — ML Training
│   ├── datasets/
│   │   └── origami_dataset.py          # PyTorch Dataset + quality report
│   ├── models/
│   │   ├── yolo_detector.py            # Faster R-CNN + Depth fusion
│   │   └── fusion_transformer.py       # Bonus: Cross-attention fusion
│   ├── training/
│   │   └── train.py                    # Train loop + early stopping
│   ├── evaluation/
│   │   └── evaluate.py                 # Stress test + visualization
│   └── reports/
│       ├── data_quality_report.md      # Auto-generated
│       └── sim2real_analysis.md        # Sim-to-real gap analysis
│
├── docs/                                # Documentation & visuals
│   ├── workflow_results.pptx
│   ├── inference_visualization.png
│   ├── batch_visualization.png
│   └── training_curves.png
│
├── .gitignore
└── README.md
```

---

## Phase 1 — Unified Data Schema

Defines a canonical JSON schema that can represent any robot interaction episode regardless of source. Supports 5 modalities: vision (RGB/depth), proprioception (joints/IMU), audio, language, and tactile.

## Phase 2 — Data Ingestion Pipeline

Config-driven pipeline that converts raw HDF5/TFRecord data into validated schema records. Adding a new data source requires only a YAML config — no code changes.

| Source | Type | Episodes |
|--------|------|----------|
| NVIDIA Isaac Sim | Simulation | 1,000 |
| Google RT-1 | Real-world | 130,000 |

## Phase 3 — ML Training Pipeline

Trains a Faster R-CNN object detector with multi-modal fusion (RGB + Depth) on Isaac Sim data. Includes domain randomization stress testing and sim-to-real readiness analysis.

### Baseline Model

| Component | Details |
|-----------|---------|
| Backbone | ResNet50-FPN v2 (pretrained COCO) |
| Fusion | Early concat: RGB + Depth → 3ch |
| Classes | 5 (background, red/blue/green cube, robot arm) |
| Total Params | 43.3M |
| Fusion Params | 6,930 |

### Bonus: Multi-Modal Fusion Transformer

Upgrades the baseline concat fusion to a **cross-attention Transformer** — inspired by modern robotics models (RT-2, Octo).

| Component | Details |
|-----------|---------|
| Architecture | Patch Embedding → Cross-Attention → Reconstruct |
| Mechanism | RGB tokens query Depth tokens via multi-head attention |
| Layers | 2 cross-attention layers, 4 heads, 128-dim embeddings |
| Fusion Params | 627,462 (vs 6,930 baseline) |
| Total Params | 43.9M |

**Why Transformer over concat?** Concat just stacks RGB and Depth together — no interaction between modalities. Cross-attention lets RGB features selectively attend to relevant depth information at each spatial location, learning *where* depth matters most for detection. This is critical for distinguishing objects with similar colors but different depths (e.g., a red sticker vs a red cube).

### Training Results

| Metric | Value |
|--------|-------|
| Best mAP | 0.930 |
| Final Train Loss | 0.010 |
| Final Val Loss | 0.088 |
| Epochs | 10 (early stopping) |

### Domain Randomization Stress Test

| Scenario | Accuracy | Drop |
|----------|:--------:|:----:|
| Baseline (clean) | 98.8% | 0.0% |
| Lighting Changes | 98.8% | 0.0% |
| Camera Angle Shift | 95.2% | 3.6% |
| Texture Randomization | 98.8% | 0.0% |
| Combined (all) | 98.8% | 0.0% |

---

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/origami-ai.git
cd origami-ai

# Setup
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision h5py numpy pyyaml tensorboard matplotlib

# Phase 2 — Run ingestion pipeline
cd pipeline
python real_world_ingest_pipeline.py \
    --config source_configs/isaac_sim.yaml \
    --data /path/to/mimic_dataset_1k.hdf5 \
    --max-episodes 3

# Phase 3 — Train detection model
cd ../ml
python training/train.py \
    --hdf5-path /path/to/mimic_dataset_1k.hdf5 \
    --config-path ../pipeline/source_configs/isaac_sim.yaml \
    --max-episodes 20 --epochs 10

# Phase 3 — Evaluate
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --hdf5-path /path/to/mimic_dataset_1k.hdf5 \
    --config-path ../pipeline/source_configs/isaac_sim.yaml
```

---

## Data

| Dataset | Size | Description |
|---------|------|-------------|
| NVIDIA Isaac Sim Mimic | 25.5 GB | Franka Panda cube stacking, 1000 demos |
| Google RT-1 Fractal | 106 MB (shard) | Multi-task real-world robot data |

Data files are not included in this repo (too large). Download from:
- Isaac Sim: [HuggingFace — NVIDIA PhysicalAI-Robotics-Manipulation-Augmented](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Manipulation-Augmented)
- RT-1: [TensorFlow Datasets — fractal20220817_data](https://www.tensorflow.org/datasets/catalog/fractal20220817_data)

---

## Tech Stack

- **ML:** PyTorch, torchvision (Faster R-CNN), TensorBoard
- **Data:** HDF5 (h5py), TFRecord (tensorflow), NumPy
- **Simulation:** NVIDIA Isaac Sim, Omniverse Replicator
- **Robot:** Franka Panda (7-DOF + 2 gripper)

---

## License

CC-BY-4.0
