# Phase 3 — ML Training Pipeline

**Status:** ✅ Complete  
**Intern:** William Hong — InGen Dynamics  
**Objective:** Convert the completed ingestion pipeline into a training-ready multimodal ML system for object detection.

---

## What This Does

Trains an object detection model on Isaac Sim robotics data. The model takes a camera frame (RGB + Depth) and outputs bounding boxes around objects (red/blue/green cubes), telling the robot what's in the scene and where.

```
Camera frame → Model → "red_cube at [x1,y1,x2,y2], confidence 0.98"
```

---

## Architecture

```
mimic_dataset_1k.hdf5 (Phase 2 YAML config for field mapping)
        │
        ▼
┌─────────────────────────┐
│  OrigamiMultimodalDataset│   datasets/origami_dataset.py
│  HDF5 → Tensor dict     │   RGB, Depth, Seg, BBox, Joints
│  Train/Val split         │   Custom collate_fn
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  YOLODetector            │   models/yolo_detector.py
│  Faster R-CNN + ResNet50 │   DepthEncoder + MultiModalFusion
│  RGB + Depth fusion      │   Pretrained on COCO
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Training Pipeline       │   training/train.py
│  SGD + StepLR            │   Early stopping, checkpointing
│  TensorBoard logging     │   mAP evaluation per epoch
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Evaluation              │   evaluation/evaluate.py
│  Domain randomization    │   Stress test table
│  Inference visualization │   Training curves
│  Batch visualization     │   GT vs Predictions
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Reports                 │   reports/
│  data_quality_report.md  │   Auto-generated
│  sim2real_analysis.md    │   Gap analysis + strategy
└─────────────────────────┘
```

---

## Project Structure

```
phase3_ml/
│
├── datasets/
│   └── origami_dataset.py            # PyTorch Dataset + data quality report
│
├── models/
│   └── yolo_detector.py              # Faster R-CNN with RGB+Depth fusion
│
├── training/
│   └── train.py                      # Training loop + early stopping
│
├── evaluation/
│   ├── evaluate.py                   # Stress test + all visualizations
│   ├── batch_visualization.png       # RGB / Depth / Seg / BBox
│   ├── inference_visualization.png   # GT vs Predictions
│   ├── training_curves.png           # Loss / mAP / LR
│   ├── stress_test_table.md          # Domain randomization results
│   └── stress_test_results.json
│
├── reports/
│   ├── data_quality_report.md        # Auto-generated data report
│   └── sim2real_analysis.md          # Sim-to-real readiness analysis
│
├── checkpoints/
│   ├── best_model.pth                # Best mAP checkpoint
│   ├── last_model.pth                # Final epoch checkpoint
│   └── training_history.json         # Loss/mAP/LR per epoch
│
├── runs/                             # TensorBoard logs
│   └── detection/
│
└── README_ML_PHASE3.md               # This file
```

---

## Quick Start

```bash
# Setup
cd phase3_ml
python3 -m venv origami_env
source origami_env/bin/activate
pip install torch torchvision pyyaml h5py numpy tensorboard matplotlib

# Fix macOS SSL (if needed)
pip install certifi
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")

# 1. Test dataset
python3 datasets/origami_dataset.py

# 2. Test model
python3 models/yolo_detector.py

# 3. Train (synthetic data, quick test)
python3 training/train.py --epochs 3 --max-episodes 5 --batch-size 2

# 4. Train with real Isaac Sim data
python3 training/train.py \
    --hdf5-path ../mimic_dataset_1k.hdf5 \
    --config-path ../phase2_pipeline/source_configs/isaac_sim.yaml \
    --epochs 30 --batch-size 4

# 5. Evaluate
python3 evaluation/evaluate.py --checkpoint checkpoints/best_model.pth

# 6. View TensorBoard
tensorboard --logdir runs/
```

---

## Model Details

| Component | Details |
|-----------|---------|
| Backbone | ResNet50-FPN v2 (pretrained on COCO) |
| Detection head | FastRCNNPredictor |
| Depth encoder | 3-layer CNN (1→16→32→3 channels) |
| Fusion | Early fusion: concat RGB + Depth features → project to 3ch |
| Total params | 43,278,458 |
| Fusion params | 6,930 |
| Optimizer | SGD (lr=0.005, momentum=0.9, weight_decay=5e-4) |
| Scheduler | StepLR (step=10, gamma=0.1) |
| Early stopping | Patience=7 |

---

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | background | — |
| 1 | red_cube | Red stacking cube |
| 2 | blue_cube | Blue stacking cube |
| 3 | green_cube | Green stacking cube |
| 4 | robot_arm | Franka Panda arm |

---

## Training Results (Synthetic Data)

| Epoch | Train Loss | Val Loss | mAP |
|-------|-----------|---------|-----|
| 0 | 0.5110 | 0.1560 | 0.667 |
| 1 | 0.1015 | 0.0636 | 1.000 |
| 2 | 0.0547 | 0.0379 | 1.000 |

---

## Domain Randomization Stress Test

| Scenario | Accuracy | Drop |
|----------|:--------:|:----:|
| Baseline (clean) | 100.0% | 0.0% |
| Lighting Changes | 100.0% | 0.0% |
| Camera Angle Shift | 100.0% | 0.0% |
| Texture Randomization | 52.5% | 47.5% |
| Combined (all) | 57.5% | 42.5% |

**Key finding:** Model over-relies on color features. Texture augmentation is critical for real-world deployment. See `reports/sim2real_analysis.md` for full analysis.

---

## Connection to Phase 1 & 2

| Phase | What it does | What Phase 3 uses from it |
|-------|-------------|--------------------------|
| Phase 1 | Defines data schema | Schema structure for validation |
| Phase 2 | Ingests + normalizes HDF5 data | `isaac_sim.yaml` config for field mapping |
| **Phase 3** | **Trains detection model** | Reads same HDF5, reuses YAML config |

Phase 3 reads `mimic_dataset_1k.hdf5` directly for pixel data, but reuses Phase 2's YAML config to know which HDF5 paths map to RGB, depth, and joint states.

---

## Dependencies

```
torch >= 2.0
torchvision >= 0.15
h5py
numpy
pyyaml
tensorboard
matplotlib
certifi (macOS SSL fix)
```
