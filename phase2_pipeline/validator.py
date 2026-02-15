"""
validator.py — Schema Record Validator
========================================
Validates packed schema records against:
  - data_schema_v1.json structure
  - metadata_annotations_spec.md rules

Validation checks:
  1. Required fields present
  2. Timestamps are unix_epoch_ms and within episode bounds
  3. sensor_id consistency between modalities and metadata.sensors[]
  4. Enum values are valid (task_type, source, environment, etc.)
  5. data_ref fields are non-empty for vision/audio if data exists

Usage:
    validator = Validator()
    result = validator.validate(record)
    result.print_report()
"""


class ValidationResult:
    """Holds validation results with errors and warnings."""

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.errors = []       # Must fix
        self.warnings = []     # Should fix

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str):
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def print_report(self):
        status = "PASS" if self.is_valid else "FAIL"
        print(f"\n{'='*50}")
        print(f"Validation Report: {self.sample_id}")
        print(f"Status: {status}")
        print(f"{'='*50}")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for i, e in enumerate(self.errors, 1):
                print(f"  {i}. {e}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for i, w in enumerate(self.warnings, 1):
                print(f"  {i}. {w}")

        if self.is_valid and not self.warnings:
            print("\n  All checks passed.")

        print(f"{'='*50}\n")


class Validator:
    """Validate schema records against data_schema_v1.json rules."""

    # Valid enum values from schema
    VALID_TASK_TYPES = {"manipulation", "navigation", "interaction", "other"}
    VALID_LANGUAGES = {"en", "zh", "mixed"}
    VALID_SOURCES = {"human", "llm", "scripted"}
    VALID_DATA_SOURCES = {"simulation", "real_world"}
    VALID_ENVIRONMENTS = {"lab", "home", "warehouse", "outdoor", "sim"}
    VALID_FRAMES = {"base_link", "world"}
    VALID_SENSOR_TYPES = {"camera", "imu", "joint", "mic", "tactile", "other"}
    VALID_VISION_TYPES = {"rgb", "depth"}

    def validate(self, record: dict) -> ValidationResult:
        """
        Run all validation checks on a schema record.

        Args:
            record: A dict from SchemaPacker.pack()

        Returns:
            ValidationResult with errors and warnings.
        """
        sample_id = record.get("sample_id", "unknown")
        result = ValidationResult(sample_id)

        self._check_required_fields(record, result)
        self._check_timestamps(record, result)
        self._check_task_context(record, result)
        self._check_vision(record, result)
        self._check_proprioception(record, result)
        self._check_audio(record, result)
        self._check_tactile(record, result)
        self._check_metadata(record, result)
        self._check_sensor_id_consistency(record, result)

        return result

    # ------------------------------------------------------------------
    # Check 1: Required top-level fields
    # ------------------------------------------------------------------
    def _check_required_fields(self, record: dict, result: ValidationResult):
        """Per metadata_annotations_spec.md Section 2."""
        required = [
            "schema_version", "sample_id", "global_timestamp",
            "task_context", "modalities", "metadata"
        ]
        for field in required:
            if field not in record:
                result.add_error(f"Missing required field: '{field}'")

        if record.get("schema_version") != "v1.0":
            result.add_warning(
                f"schema_version is '{record.get('schema_version')}', expected 'v1.0'"
            )

    # ------------------------------------------------------------------
    # Check 2: Timestamps
    # ------------------------------------------------------------------
    def _check_timestamps(self, record: dict, result: ValidationResult):
        """
        Per normalization_rules.md Section 1:
        - All timestamps must be unix_epoch_ms
        - Modality timestamps must fall within episode bounds
        """
        gt = record.get("global_timestamp", {})

        if gt.get("time_base") != "unix_epoch_ms":
            result.add_error("global_timestamp.time_base must be 'unix_epoch_ms'")

        start_ts = gt.get("start_ts")
        end_ts = gt.get("end_ts")

        if start_ts is None or end_ts is None:
            result.add_error("global_timestamp must have start_ts and end_ts")
            return

        if not isinstance(start_ts, int) or not isinstance(end_ts, int):
            result.add_error("start_ts and end_ts must be integers (unix_epoch_ms)")
            return

        if end_ts < start_ts:
            result.add_error(f"end_ts ({end_ts}) < start_ts ({start_ts})")

        # Check modality timestamps are within bounds
        modalities = record.get("modalities", {})

        # Vision timestamps
        for stream in modalities.get("vision", {}).get("streams", []):
            ts_list = stream.get("timestamps", [])
            if ts_list:
                self._check_ts_bounds(
                    ts_list, start_ts, end_ts,
                    f"vision.{stream.get('sensor_id', '?')}", result
                )

        # Proprioception timestamps
        prop_ts = modalities.get("proprioception", {}).get("timestamps", [])
        if prop_ts:
            self._check_ts_bounds(prop_ts, start_ts, end_ts, "proprioception", result)

        # Audio timestamps
        audio_ts = modalities.get("audio", {}).get("timestamps", [])
        if audio_ts:
            self._check_ts_bounds(audio_ts, start_ts, end_ts, "audio", result)

        # Tactile timestamps
        tactile_ts = modalities.get("tactile", {}).get("timestamps", [])
        if tactile_ts:
            self._check_ts_bounds(tactile_ts, start_ts, end_ts, "tactile", result)

    def _check_ts_bounds(self, timestamps, start_ts, end_ts, label, result):
        """Check that all timestamps fall within episode bounds."""
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            if min_ts < start_ts:
                result.add_error(
                    f"{label} timestamps start ({min_ts}) before episode start ({start_ts})"
                )
            if max_ts > end_ts:
                result.add_error(
                    f"{label} timestamps end ({max_ts}) after episode end ({end_ts})"
                )

    # ------------------------------------------------------------------
    # Check 3: Task context enums
    # ------------------------------------------------------------------
    def _check_task_context(self, record: dict, result: ValidationResult):
        tc = record.get("task_context", {})

        task_type = tc.get("task_type", "")
        if task_type and task_type not in self.VALID_TASK_TYPES:
            result.add_error(f"Invalid task_type: '{task_type}'")

        lang = tc.get("language_instruction", {})
        lang_val = lang.get("language", "")
        if lang_val and lang_val not in self.VALID_LANGUAGES:
            result.add_error(f"Invalid language: '{lang_val}'")

        source_val = lang.get("source", "")
        if source_val and source_val not in self.VALID_SOURCES:
            result.add_error(f"Invalid language source: '{source_val}'")

    # ------------------------------------------------------------------
    # Check 4: Vision streams
    # ------------------------------------------------------------------
    def _check_vision(self, record: dict, result: ValidationResult):
        vision = record.get("modalities", {}).get("vision", {})
        streams = vision.get("streams", [])

        for stream in streams:
            sid = stream.get("sensor_id", "")
            if not sid:
                result.add_error("Vision stream missing sensor_id")

            vtype = stream.get("type", "")
            if vtype and vtype not in self.VALID_VISION_TYPES:
                result.add_error(f"Invalid vision type: '{vtype}'")

            if stream.get("timestamps") and not stream.get("data_ref"):
                result.add_warning(
                    f"Vision stream '{sid}' has timestamps but no data_ref"
                )

    # ------------------------------------------------------------------
    # Check 5: Proprioception
    # ------------------------------------------------------------------
    def _check_proprioception(self, record: dict, result: ValidationResult):
        prop = record.get("modalities", {}).get("proprioception", {})

        if not prop.get("sensor_id"):
            result.add_warning("Proprioception missing sensor_id")

        ts = prop.get("timestamps", [])
        js = prop.get("joint_states", [])
        if ts and js and len(ts) != len(js):
            result.add_warning(
                f"Proprioception: timestamps ({len(ts)}) != joint_states ({len(js)}) length"
            )

    # ------------------------------------------------------------------
    # Check 6: Audio
    # ------------------------------------------------------------------
    def _check_audio(self, record: dict, result: ValidationResult):
        """Audio can be empty but must have required keys."""
        audio = record.get("modalities", {}).get("audio", {})
        required_keys = ["sensor_id", "sampling_rate_hz", "timestamps", "data_ref"]
        for key in required_keys:
            if key not in audio:
                result.add_warning(f"Audio missing key: '{key}'")

    # ------------------------------------------------------------------
    # Check 7: Tactile
    # ------------------------------------------------------------------
    def _check_tactile(self, record: dict, result: ValidationResult):
        """Tactile can be empty but must have required keys."""
        tactile = record.get("modalities", {}).get("tactile", {})
        required_keys = ["sensor_id", "timestamps", "force", "pressure"]
        for key in required_keys:
            if key not in tactile:
                result.add_warning(f"Tactile missing key: '{key}'")

    # ------------------------------------------------------------------
    # Check 8: Metadata
    # ------------------------------------------------------------------
    def _check_metadata(self, record: dict, result: ValidationResult):
        meta = record.get("metadata", {})

        source = meta.get("source", "")
        if source and source not in self.VALID_DATA_SOURCES:
            result.add_error(f"Invalid metadata.source: '{source}'")

        env = meta.get("environment", "")
        if env and env not in self.VALID_ENVIRONMENTS:
            result.add_error(f"Invalid metadata.environment: '{env}'")

        frame = meta.get("frame_of_reference", "")
        if frame and frame not in self.VALID_FRAMES:
            result.add_error(f"Invalid frame_of_reference: '{frame}'")

        for sensor in meta.get("sensors", []):
            stype = sensor.get("sensor_type", "")
            if stype and stype not in self.VALID_SENSOR_TYPES:
                result.add_error(f"Invalid sensor_type: '{stype}'")

    # ------------------------------------------------------------------
    # Check 9: sensor_id consistency
    # ------------------------------------------------------------------
    def _check_sensor_id_consistency(self, record: dict, result: ValidationResult):
        """
        Per metadata_annotations_spec.md Rule #2:
        metadata.sensors[].sensor_id must match modality sensor_ids.
        """
        modalities = record.get("modalities", {})
        metadata_sensors = record.get("metadata", {}).get("sensors", [])

        # Collect all sensor_ids used in modalities
        modality_ids = set()

        for stream in modalities.get("vision", {}).get("streams", []):
            sid = stream.get("sensor_id")
            if sid:
                modality_ids.add(sid)

        prop_sid = modalities.get("proprioception", {}).get("sensor_id")
        if prop_sid:
            modality_ids.add(prop_sid)

        audio_sid = modalities.get("audio", {}).get("sensor_id")
        if audio_sid:
            modality_ids.add(audio_sid)

        tactile_sid = modalities.get("tactile", {}).get("sensor_id")
        if tactile_sid:
            modality_ids.add(tactile_sid)

        # Collect sensor_ids from metadata
        metadata_ids = {s.get("sensor_id") for s in metadata_sensors if s.get("sensor_id")}

        # Check consistency
        missing_in_metadata = modality_ids - metadata_ids
        if missing_in_metadata:
            result.add_error(
                f"sensor_ids in modalities but not in metadata.sensors: {missing_in_metadata}"
            )

        extra_in_metadata = metadata_ids - modality_ids
        if extra_in_metadata:
            result.add_warning(
                f"sensor_ids in metadata.sensors but not in modalities: {extra_in_metadata}"
            )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            record = json.load(f)
        v = Validator()
        result = v.validate(record)
        result.print_report()
    else:
        print("Usage: python validator.py <schema_record.json>")
        print("Validates a schema record against data_schema_v1.json rules.")
