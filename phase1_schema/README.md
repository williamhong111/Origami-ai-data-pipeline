# Phase 1 — Unified Multimodal Data Schema

**Status:** ✅ Complete

Phase 1 defines the canonical data representation for multimodal robot interaction episodes. The schema serves as a stable data contract between upstream data sources (simulators, real robots) and downstream consumers (model training, policy learning).

---

## Deliverables

| File | Description |
|---|---|
| `data_schema_v1.json` | Machine-readable schema definition (JSON) |
| `metadata_annotations_spec.md` | Human-readable field semantics, constraints, and annotation rules |
| `examples/sample_example_v1.json` | Fully populated example record |

---

## Schema Overview

Each record represents a single robot interaction episode with the following top-level structure:

```json
{
  "schema_version": "v1.0",
  "sample_id": "unique_episode_identifier",
  "global_timestamp": { "time_base": "unix_epoch_ms", "start_ts": ..., "end_ts": ... },
  "task_context": { "task_type": "manipulation", "language_instruction": {...}, ... },
  "modalities": {
    "vision":         { "streams": [...] },
    "proprioception": { "joint_states": [...], "imu": {...} },
    "audio":          { "data_ref": "..." },
    "language":       { "transcript": "..." },
    "tactile":        { "force": [...], "pressure": [...] }
  },
  "metadata": { "source": "simulation", "robot_platform": "...", "sensors": [...] }
}
```

---

## Design Principles

1. **Source-agnostic** — The same schema represents data from simulation (Isaac Sim) and real-world robots without modification
2. **Modality-extensible** — New sensor types can be added without breaking existing records
3. **Timestamp-first** — Every modality carries explicit `unix_epoch_ms` timestamps for cross-modal alignment
4. **Missing data safe** — Absent modalities use empty arrays rather than null values or omitted keys
5. **Sensor-traceable** — Every data stream links to a `sensor_id` registered in `metadata.sensors[]`

---

## Supported Modalities

| Modality | Key Fields | Use Case |
|---|---|---|
| **Vision** | RGB, depth, frame rate, resolution | Visual perception, object detection |
| **Proprioception** | Joint states, IMU (acc, gyro) | Motor control, state estimation |
| **Audio** | Waveform reference, sampling rate | Speech commands, environmental sound |
| **Language** | Instruction text, language, source | Task grounding, NLP-driven control |
| **Tactile** | Force, pressure arrays | Grasping, contact-rich manipulation |

---

## Validation Rules

From `metadata_annotations_spec.md`:

- All timestamps must be `unix_epoch_ms` (integer milliseconds)
- Modality timestamps must fall within `global_timestamp.start_ts` and `end_ts`
- `sensor_id` values in modalities must match entries in `metadata.sensors[]`
- Enum fields (`task_type`, `language`, `source`, `environment`) must use defined values
- Missing modalities must retain required keys with empty arrays

---

## Usage

This schema is consumed by the Phase 2 ingestion pipeline. See [`../phase2_pipeline/README.md`](../phase2_pipeline/README.md) for implementation details.
