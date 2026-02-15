# Metadata Annotation Specification (v1.0)

## 1. Scope
This document specifies the semantics and annotation rules for `data_schema_v1.json`.
Phase 1 is **specification-only**: it does not include ingestion pipelines, normalization, validators, or models.

## 2. Required Fields
A sample MUST include:
- `schema_version`
- `sample_id`
- `global_timestamp`
- `task_context`
- `modalities`
- `metadata`

## 3. Time Standards
- Time base: `unix_epoch_ms`
- `global_timestamp.start_ts` and `global_timestamp.end_ts` define the episode bounds.
- Each modality MUST provide explicit `timestamps` in ms.
- No implicit alignment is assumed; alignment is performed via timestamps only.

## 4. Task Context
### task_context.task_id
Unique identifier for the task instance.

### task_context.task_type
Coarse category used for filtering and analytics. Allowed values:
`manipulation | navigation | interaction | other`

### task_context.goal_description
Short natural-language statement of the objective.

### task_context.language_instruction
- `text`: raw instruction text
- `language`: `en | zh | mixed`
- `source`: `human | llm | scripted`

### task_context.user_intent
Free-text intent label (high-level). Example: "pick_and_place", "follow_me", "go_to_kitchen".

### task_context.success_criteria
Definition of what counts as success (human- or program-defined).

## 5. Modalities

### 5.1 Vision
`modalities.vision.streams[]` defines one or more visual streams.
Each stream MUST include:
- `sensor_id`
- `type` (`rgb | depth`)
- `frame_rate_hz`
- `resolution` (e.g., `1280x720`)
- `timestamps` (ms)
- `data_ref` (path/URI to stored payload)

Notes:
- Vision payloads are stored externally; the schema stores only references.

### 5.2 Proprioception
`modalities.proprioception` captures internal robot state.
Required:
- `sensor_id`
- `sampling_rate_hz`
- `timestamps`
- `joint_states` (ordered joint vector; ordering documented at dataset level)
- `imu.acc`, `imu.gyro` (3D vectors aligned to timestamps)

### 5.3 Audio
Required:
- `sensor_id`
- `sampling_rate_hz`
- `timestamps`
- `data_ref`

### 5.4 Language
`modalities.language.transcript` is optional and may be empty if no speech or transcript is available.
If timestamps are provided, they should correspond to utterance boundaries or alignment points.

### 5.5 Tactile
Required:
- `sensor_id`
- `timestamps`
- `force` and/or `pressure` arrays aligned to timestamps.
If a sensor does not produce a field, keep the array empty.

## 6. Metadata

### metadata.source
Data origin: `simulation | real_world`

### metadata.dataset_id
Logical dataset identifier (e.g., project name, collection batch).

### metadata.robot_platform
Robot identifier or platform name (e.g., `aido_unit_v2`).

### metadata.environment
Collection environment label: `lab | home | warehouse | outdoor | sim`

### metadata.frame_of_reference
Canonical coordinate frame for interpretation: `base_link | world`

### metadata.sensors[]
Per-sensor attribution records. Each entry MUST include:
- `sensor_id` (must match modality sensor_id values)
- `sensor_type` (`camera | imu | joint | mic | tactile | other`)
- `source_id` (hardware id, sim node id, or recording id)
- `calibration_ref` (path/URI/hash to calibration package)

### metadata.notes
Free-form notes for edge cases and collection context.

## 7. Annotation Rules (Minimum)
1. All timestamps must be `unix_epoch_ms`.
2. `metadata.sensors[].sensor_id` must match any `sensor_id` used in `modalities`.
3. Use external storage references via `data_ref`; do not embed large payloads in the sample record.
4. If a modality is present but has no data (e.g., tactile not used), keep required keys and use empty arrays.
5. Any schema changes must be additive and require a version bump in `schema_version`.
