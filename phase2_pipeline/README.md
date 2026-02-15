# Phase 2 — Real-World Data Ingestion Pipeline

**Status:** ✅ Complete

Phase 2 implements a config-driven pipeline that converts raw multimodal robot data into the canonical schema defined in Phase 1. The pipeline is designed so that adding a new data source requires only a YAML config file — no Python code changes.

---

## Pipeline Architecture

```
Raw Data (HDF5)
      │
      ▼
┌──────────────┐    source_configs/
│  DataLoader  │◄── isaac_sim.yaml      Config-driven field mapping
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  DataNormalizer   │                    Timestamps, sampling rates, units
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ SchemaPacker │                        Maps to data_schema_v1.json structure
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Validator   │                        Schema compliance + sensor ID checks
└──────┬───────┘
       │
       ▼
   output/*.json                         Validated canonical records
```

---

## Module Reference

| Module | Purpose |
|---|---|
| `real_world_ingest_pipeline.py` | Main orchestrator — chains all modules and runs end-to-end |
| `dataloader.py` | Universal data reader — reads HDF5/JSON using YAML config mappings |
| `data_normalizer.py` | Normalizes timestamps to `unix_epoch_ms`, records actual vs. target sampling rates |
| `schema_packer.py` | Packs normalized data into `data_schema_v1.json` structure |
| `validator.py` | Validates required fields, timestamp bounds, enum values, sensor ID consistency |
| `source_configs/isaac_sim.yaml` | Field mapping for NVIDIA Isaac Sim Mimic HDF5 dataset |

---

## Quick Start

```bash
# Install dependencies
pip install h5py numpy pyyaml

# Run pipeline on Isaac Sim data
python real_world_ingest_pipeline.py \
    --config source_configs/isaac_sim.yaml \
    --data /path/to/mimic_dataset_1k.hdf5 \
    --max-episodes 3 \
    --output-dir output/
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config`, `-c` | `source_configs/isaac_sim.yaml` | Path to source config YAML |
| `--data`, `-d` | Auto-detect | Path to raw data file |
| `--max-episodes`, `-n` | `5` | Max episodes to process |
| `--output-dir`, `-o` | `output/` | Output directory for JSON records |

---

## Validation Checks

The validator performs 9 checks on every output record:

| # | Check | Severity |
|---|---|---|
| 1 | Required top-level fields present | Error |
| 2 | `global_timestamp.time_base` = `unix_epoch_ms` | Error |
| 3 | All modality timestamps within episode bounds | Error |
| 4 | `task_type` is valid enum value | Error |
| 5 | `language` and `source` are valid enum values | Error |
| 6 | Vision streams have `sensor_id` and `data_ref` | Warning |
| 7 | Proprioception `timestamps` and `joint_states` length match | Warning |
| 8 | `metadata.source`, `environment`, `frame_of_reference` valid | Error |
| 9 | `sensor_id` consistency between modalities and `metadata.sensors[]` | Error |

---

## Normalization Standards

From `normalization_rules.md`:

| Modality | Target | Actual (Isaac Sim) | Notes |
|---|---|---|---|
| Vision frame rate | 30 Hz | 20 Hz | Recorded as-is, not upsampled |
| Vision resolution | 1280×720 | 200×200 | Recorded as-is, not upscaled |
| Proprioception | 100 Hz | 20 Hz | Recorded as-is, not interpolated |
| Audio | 16,000 Hz | N/A | Empty arrays (missing data policy) |
| Timestamps | unix_epoch_ms | ✅ Generated | Synthetic from control rate |

Deviations are logged in `metadata.notes` for full transparency.

---

## Test Results

Tested on NVIDIA Isaac Sim Mimic dataset (`mimic_dataset_1k.hdf5`, 25.5 GB):

```
Episodes processed:  3
Records generated:   3
Valid records:       3
Total errors:        0
Total warnings:      0
Time elapsed:        1.08s
```

All 3 episodes passed validation with zero errors.

---

## Adding a New Data Source

1. **Create a YAML config** in `source_configs/` that maps the new source's field paths to unified modality names
2. **Run the pipeline** with `--config source_configs/your_source.yaml`
3. The pipeline handles the rest — no code modifications needed

See `source_configs/isaac_sim.yaml` for a complete example.

---

## Data Source: NVIDIA Isaac Sim Mimic

| Property | Value |
|---|---|
| Dataset | `mimic_dataset_1k.hdf5` |
| Robot | Franka Panda (7-DOF + 2 gripper joints) |
| Task | Cube stacking (blue → red → green) |
| Episodes | 1,000 demonstrations |
| Modalities | 2× RGB camera, 1× depth camera, joint pos/vel, end-effector pose, gripper, actions |
| Size | 25.5 GB |
| License | CC-BY-4.0 |
| Source | [Hugging Face — NVIDIA PhysicalAI-Robotics-Manipulation-Augmented](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Manipulation-Augmented) |
