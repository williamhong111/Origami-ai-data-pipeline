# 🤖 Multimodal Robotics Data Pipeline

A unified data schema and ingestion pipeline for multimodal robot learning. This project standardizes heterogeneous sensor data from simulation and real-world sources into a single canonical format, enabling scalable training of robotics foundation models.

---

## Problem

Modern robot learning requires data from many modalities — RGB cameras, depth sensors, joint encoders, IMUs, microphones, tactile sensors, and language instructions. Each data source uses different formats, sampling rates, coordinate frames, and storage layouts. This fragmentation makes it difficult to train general-purpose models across diverse robot platforms and environments.

## Solution

This project provides:

- **A unified multimodal schema** (`data_schema_v1.json`) that represents any robot interaction episode in a single, versioned JSON structure
- **A config-driven ingestion pipeline** that converts raw data from any source into the canonical format — add a new data source by writing a YAML config, no code changes needed
- **Validation tooling** that enforces schema compliance, timestamp consistency, and sensor ID integrity

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │     │   YAML Config   │     │   Canonical     │
│                 │     │                 │     │   Schema v1     │
│  • Isaac Sim    │────▶│  isaac_sim.yaml │────▶│                 │
│  • RT-1         │     │  rt1.yaml       │     │  Normalized,    │
│  • Real robots  │     │  bridge.yaml    │     │  validated JSON │
│  • RoboNet      │     │  ...            │     │  records        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
               DataLoader          Validator
               Normalizer         Schema Packer
```

---

## Project Structure

```
multimodal-robotics-pipeline/
│
├── README.md                          # This file
│
├── phase1_schema/                     # Phase 1: Schema & Metadata Spec
│   ├── README.md
│   ├── data_schema_v1.json            # Canonical multimodal schema
│   ├── metadata_annotations_spec.md   # Field semantics & annotation rules
│   └── examples/
│       └── sample_example_v1.json     # Example populated record
│
└── phase2_pipeline/                   # Phase 2: Ingestion Pipeline
    ├── README.md
    ├── real_world_ingest_pipeline.py   # Main pipeline orchestrator
    ├── dataloader.py                   # Universal config-driven data reader
    ├── data_normalizer.py              # Timestamp & sampling normalization
    ├── schema_packer.py                # Maps normalized data → schema JSON
    ├── validator.py                    # Schema compliance checker
    ├── source_configs/
    │   └── isaac_sim.yaml              # NVIDIA Isaac Sim field mapping
    └── output/                         # Generated schema records
        ├── nvidia_mimic_franka_stack_demo_0.json
        ├── nvidia_mimic_franka_stack_demo_1.json
        └── nvidia_mimic_franka_stack_demo_2.json
```

---

## Quick Start

### Requirements

- Python 3.10+
- Dependencies: `h5py`, `numpy`, `pyyaml`

```bash
pip install h5py numpy pyyaml
```

### Run the Pipeline

```bash
cd phase2_pipeline

# Process 3 episodes from Isaac Sim dataset
python real_world_ingest_pipeline.py \
    --config source_configs/isaac_sim.yaml \
    --data /path/to/mimic_dataset_1k.hdf5 \
    --max-episodes 3 \
    --output-dir output/
```

### Expected Output

```
============================================================
  Multimodal Robotics Data Ingestion Pipeline
  Phase 2: Real-World Data Pipeline
============================================================
[Step 1/4] Loading raw data...
[Step 2/4] Normalizing data...
[Step 3/4] Packing into canonical schema...
[Step 4/4] Validating records...

Validation Report: nvidia_mimic_franka_stack_demo_0
Status: PASS — All checks passed.

  Pipeline Summary
  Episodes processed:  3
  Valid records:       3
  Total errors:        0
  Total warnings:      0
  Time elapsed:        1.08s
============================================================
```

---

## Supported Modalities

| Modality | Fields | Normalization Target |
|---|---|---|
| **Vision** (RGB / Depth) | sensor_id, frame_rate, resolution, timestamps, data_ref | 30 Hz, 1280×720 |
| **Proprioception** (Joint / IMU) | joint_states, imu.acc, imu.gyro, timestamps | 100 Hz, radians |
| **Audio** | sampling_rate, timestamps, data_ref | 16,000 Hz |
| **Language** | instruction text, language, source | Raw text preserved |
| **Tactile** | force, pressure, timestamps | unix_epoch_ms |

Missing modalities are handled gracefully — required keys are preserved with empty arrays, following the "do not fabricate data" policy.

---

## Adding a New Data Source

1. Create a YAML config in `source_configs/`:

```yaml
source_name: "my_new_source"
format: "hdf5"

vision:
  streams:
    - name: "front_cam"
      type: "rgb"
      hdf5_path: "observations/image"
      sensor_id: "cam_front"

proprioception:
  joint_pos:
    hdf5_path: "observations/joint_positions"
```

2. Run the pipeline with your config:

```bash
python real_world_ingest_pipeline.py \
    --config source_configs/my_new_source.yaml \
    --data /path/to/data.hdf5
```

No Python code changes required.

---

## Data Sources

| Source | Type | Status | Description |
|---|---|---|---|
| **NVIDIA Isaac Sim** | Simulation | ✅ Integrated | 1,000 Franka Panda manipulation demos (cube stacking) |
| **RT-1** | Real-world | 🔜 Planned | Google's 130K episode multi-task robot dataset |

---

## Roadmap

- [x] **Phase 1** — Unified multimodal schema & metadata spec
- [x] **Phase 2** — Config-driven ingestion pipeline with validation
- [ ] **Phase 3** — RT-1 integration, HDF5 batch export, multi-source benchmarking

---

## License

CC-BY-4.0
