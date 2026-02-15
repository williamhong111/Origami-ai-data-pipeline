# 🤖 Multimodal Robotics Data Pipeline

A production-ready data ingestion pipeline that standardizes heterogeneous robot sensor data into a unified canonical schema. Designed to support scalable training of robotics foundation models across simulation and real-world environments.

**Built at InGen Dynamics**  
**Author:** William Hong — ML Engineering Intern  

---

## Problem

Modern robot learning requires data from many modalities — RGB cameras, depth sensors, joint encoders, IMUs, microphones, tactile sensors, and language instructions. Each source uses different formats, sampling rates, coordinate frames, and storage layouts. This fragmentation creates a major bottleneck for training general-purpose robotics models at scale.

## Solution

This pipeline provides:

- **A unified multimodal schema** (`data_schema_v1.json`) — a single versioned JSON structure that can represent any robot interaction episode regardless of source
- **Config-driven ingestion** — add a new data source by writing a YAML config; no code changes needed
- **Automated validation** — enforces schema compliance, timestamp consistency, and sensor ID integrity before data enters the training pipeline

**Result:** Reduced data onboarding time from ~2 days of custom scripting per source to <1 hour of YAML configuration.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │          Ingestion Pipeline              │
                        │                                          │
 ┌───────────────┐      │  ┌────────────┐    ┌────────────────┐   │    ┌──────────────┐
 │  Isaac Sim    │──┐   │  │            │    │                │   │    │              │
 │  (HDF5)      │  │   │  │ DataLoader │───▶│  Normalizer    │   │    │  Canonical   │
 └───────────────┘  │   │  │            │    │                │   │    │  Schema v1   │
                    │   │  │ • Reads    │    │ • Resamples to │   │    │  (JSON)      │
 ┌───────────────┐  │   │  │   YAML     │    │   target Hz    │   │    │              │
 │  RT-1         │──┼──▶│  │   config   │    │ • Aligns       │   │──▶ │ • Validated  │
 │  (TFRecord)   │  │   │  │ • Maps     │    │   timestamps   │   │    │ • Normalized │
 └───────────────┘  │   │  │   fields   │    │ • Converts     │   │    │ • Ready for  │
                    │   │  └────────────┘    │   units        │   │    │   training   │
 ┌───────────────┐  │   │                    └───────┬────────┘   │    │              │
 │  Real Robot   │──┘   │                            │            │    └──────────────┘
 │  (ROS bags)   │      │                    ┌───────▼────────┐   │           │
 └───────────────┘      │                    │ Schema Packer  │   │           │
                        │                    │ + Validator    │   │    ┌──────▼───────┐
 ┌───────────────┐      │                    │               │   │    │   Output/     │
 │  YAML Config  │─────▶│                    │ • Pack → JSON  │   │    │   *.json      │
 │  per source   │      │                    │ • Validate     │   │    └──────────────┘
 └───────────────┘      │                    │   schema       │   │
                        │                    │ • Check        │   │
                        │                    │   timestamps   │   │
                        │                    └────────────────┘   │
                        └──────────────────────────────────────────┘
```

---

## Project Structure

```
multimodal-robotics-pipeline/
│
├── README.md
│
├── schema/
│   ├── data_schema_v1.json            # Canonical multimodal schema
│   ├── metadata_annotations_spec.md   # Field semantics & annotation rules
│   └── examples/
│       └── sample_example_v1.json     # Example populated record
│
└── pipeline/
    ├── real_world_ingest_pipeline.py   # Main pipeline orchestrator
    ├── dataloader.py                   # Universal config-driven data reader
    ├── data_normalizer.py              # Timestamp & sampling normalization
    ├── schema_packer.py                # Maps normalized data → schema JSON
    ├── validator.py                    # Schema compliance checker
    ├── source_configs/
    │   └── isaac_sim.yaml              # NVIDIA Isaac Sim field mapping
    └── output/
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
cd pipeline

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

2. Run the pipeline:

```bash
python real_world_ingest_pipeline.py \
    --config source_configs/my_new_source.yaml \
    --data /path/to/data.hdf5
```

No Python code changes required.

---

## Integrated Data Sources

| Source | Type | Episodes | Description |
|---|---|---|---|
| **NVIDIA Isaac Sim** | Simulation | 1,000 | Franka Panda manipulation demos (cube stacking) |
| **RT-1** | Real-world | 130,000 | Google's multi-task robot dataset |

---

## Technical Highlights

- **Zero-code onboarding** — YAML-driven architecture means new data sources require configuration only, not pipeline modifications
- **Strict validation** — every output record passes schema compliance, timestamp monotonicity, and sensor ID integrity checks before entering downstream training
- **Modality-agnostic normalization** — handles variable sampling rates across sensors with configurable target frequencies
- **HDF5 native** — reads large-scale robotics datasets directly without intermediate conversion

---

## Presentation

📊 [Project Slide Deck](Multimodal_Robotics_Data_Pipeline_v2.pptx)
---

## License

CC-BY-4.0
