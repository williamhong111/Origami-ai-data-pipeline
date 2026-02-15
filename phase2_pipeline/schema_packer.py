"""
schema_packer.py — Pack Normalized Data into Canonical Schema
==============================================================
Maps normalized episode data into the structure defined by
data_schema_v1.json (Phase 1).

Follows raw_to_schema_mapping.md for field placement.

Usage:
    packer = SchemaPacker()
    record = packer.pack(normalized_episode)
    packer.save(record, "output/demo_0.json")
"""

import json
import os
import numpy as np


class SchemaPacker:
    """Pack normalized data into data_schema_v1.json format."""

    SCHEMA_VERSION = "v1.0"

    def pack(self, normalized: dict) -> dict:
        """
        Pack a normalized episode into the canonical schema.

        Args:
            normalized: Output from DataNormalizer.normalize()

        Returns:
            A dict matching data_schema_v1.json structure.
        """
        config = normalized.get("config", {})
        task_cfg = config.get("task_context", {})
        meta_cfg = config.get("metadata", {})
        demo_id = normalized["demo_id"]

        record = {
            "schema_version": self.SCHEMA_VERSION,
            "sample_id": f"{meta_cfg.get('dataset_id', 'unknown')}_{demo_id}",

            # --- Global timestamp ---
            "global_timestamp": normalized["global_timestamp"],

            # --- Task context ---
            "task_context": self._pack_task_context(task_cfg, demo_id),

            # --- Modalities ---
            "modalities": {
                "vision":         self._pack_vision(normalized.get("vision", {})),
                "proprioception": self._pack_proprioception(normalized.get("proprioception", {})),
                "audio":          self._pack_audio(normalized.get("audio", {})),
                "language":       self._pack_language(),
                "tactile":        self._pack_tactile(normalized.get("tactile", {})),
            },

            # --- Metadata ---
            "metadata": self._pack_metadata(normalized),
        }

        return record

    # ------------------------------------------------------------------
    # Task context
    # ------------------------------------------------------------------
    def _pack_task_context(self, task_cfg: dict, demo_id: str) -> dict:
        """Map task context from config into schema."""
        lang_cfg = task_cfg.get("language_instruction", {})

        return {
            "task_id": f"{task_cfg.get('task_type', 'unknown')}_{demo_id}",
            "task_type": task_cfg.get("task_type", "other"),
            "goal_description": task_cfg.get("goal_description", ""),
            "language_instruction": {
                "text": lang_cfg.get("text", ""),
                "language": lang_cfg.get("language", "en"),
                "source": lang_cfg.get("source", "scripted"),
            },
            "user_intent": task_cfg.get("user_intent", ""),
            "success_criteria": task_cfg.get("success_criteria", ""),
        }

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def _pack_vision(self, vision_data: dict) -> dict:
        """
        Map vision streams into schema.

        Schema path: modalities.vision.streams[]
        Each stream gets: sensor_id, type, frame_rate_hz, resolution,
                          timestamps, data_ref
        """
        streams = []

        for stream_name, stream_info in vision_data.items():
            # Vision payloads stored externally, referenced via data_ref
            # In this implementation, we record the reference path
            data_ref = f"data/vision/{stream_name}.npy"

            stream_record = {
                "sensor_id": stream_info.get("sensor_id", stream_name),
                "type": stream_info.get("type", "rgb"),
                "frame_rate_hz": stream_info.get("frame_rate_hz", 0.0),
                "resolution": stream_info.get("resolution", "0x0"),
                "timestamps": stream_info.get("timestamps", []),
                "data_ref": data_ref,
            }
            streams.append(stream_record)

        return {"streams": streams}

    # ------------------------------------------------------------------
    # Proprioception
    # ------------------------------------------------------------------
    def _pack_proprioception(self, prop_data: dict) -> dict:
        """
        Map proprioception into schema.

        Schema path: modalities.proprioception
        Fields: sensor_id, sampling_rate_hz, timestamps,
                joint_states, imu.acc, imu.gyro
        """
        return {
            "sensor_id": prop_data.get("sensor_id", ""),
            "sampling_rate_hz": prop_data.get("sampling_rate_hz", 0.0),
            "timestamps": prop_data.get("timestamps", []),
            "joint_states": prop_data.get("joint_states", []),
            "imu": {
                "acc": prop_data.get("imu", {}).get("acc", []),
                "gyro": prop_data.get("imu", {}).get("gyro", []),
            },
        }

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def _pack_audio(self, audio_data: dict) -> dict:
        """
        Map audio into schema.

        Per normalization_rules.md: if absent, keep block with empty timestamps.
        """
        return {
            "sensor_id": audio_data.get("sensor_id", ""),
            "sampling_rate_hz": audio_data.get("sampling_rate_hz", 16000.0),
            "timestamps": audio_data.get("timestamps", []),
            "data_ref": audio_data.get("data_ref", ""),
        }

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    def _pack_language(self) -> dict:
        """
        Map language modality into schema.

        Language transcript is optional per metadata_annotations_spec.md.
        """
        return {
            "transcript": "",
            "timestamps": [],
        }

    # ------------------------------------------------------------------
    # Tactile
    # ------------------------------------------------------------------
    def _pack_tactile(self, tactile_data: dict) -> dict:
        """
        Map tactile into schema.

        Per normalization_rules.md: if only force or pressure exists,
        keep the missing array empty.
        """
        return {
            "sensor_id": tactile_data.get("sensor_id", ""),
            "timestamps": tactile_data.get("timestamps", []),
            "force": tactile_data.get("force", []),
            "pressure": tactile_data.get("pressure", []),
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _pack_metadata(self, normalized: dict) -> dict:
        """
        Build metadata block from config and normalized data.

        Includes sensor registry with sensor_id matching per
        metadata_annotations_spec.md Rule #2.
        """
        config = normalized.get("config", {})
        meta_cfg = config.get("metadata", {})

        # Build sensor list from all modalities that have sensor_ids
        sensors = []

        # Vision sensors
        for stream_name, stream_info in normalized.get("vision", {}).items():
            sensors.append({
                "sensor_id": stream_info.get("sensor_id", stream_name),
                "sensor_type": "camera",
                "source_id": stream_info.get("source_id", ""),
                "calibration_ref": "",
            })

        # Proprioception sensor
        prop = normalized.get("proprioception", {})
        if prop.get("sensor_id"):
            sensors.append({
                "sensor_id": prop["sensor_id"],
                "sensor_type": "joint",
                "source_id": prop.get("source_id", ""),
                "calibration_ref": "",
            })

        # Audio sensor
        audio = normalized.get("audio", {})
        if audio.get("sensor_id"):
            sensors.append({
                "sensor_id": audio["sensor_id"],
                "sensor_type": "mic",
                "source_id": "",
                "calibration_ref": "",
            })

        # Tactile sensor
        tactile = normalized.get("tactile", {})
        if tactile.get("sensor_id"):
            sensors.append({
                "sensor_id": tactile["sensor_id"],
                "sensor_type": "tactile",
                "source_id": "",
                "calibration_ref": "",
            })

        notes_parts = []
        # Record normalization gaps
        for stream_name, stream_info in normalized.get("vision", {}).items():
            actual_res = stream_info.get("resolution", "")
            target_res = stream_info.get("target_resolution", "")
            if actual_res and target_res and actual_res != target_res:
                notes_parts.append(
                    f"Vision stream '{stream_name}' is {actual_res}, "
                    f"target is {target_res} (not resized)."
                )

        actual_hz = prop.get("sampling_rate_hz", 0)
        target_hz = prop.get("target_sampling_rate_hz", 0)
        if actual_hz and target_hz and actual_hz != target_hz:
            notes_parts.append(
                f"Proprioception at {actual_hz} Hz, target is {target_hz} Hz (not resampled)."
            )

        return {
            "source": meta_cfg.get("source", "simulation"),
            "dataset_id": meta_cfg.get("dataset_id", ""),
            "robot_platform": meta_cfg.get("robot_platform", ""),
            "environment": meta_cfg.get("environment", "sim"),
            "frame_of_reference": meta_cfg.get("frame_of_reference", "world"),
            "sensors": sensors,
            "notes": " | ".join(notes_parts) if notes_parts else "",
        }

    # ------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------
    def save(self, record: dict, output_path: str):
        """Save a packed record as JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"[SchemaPacker] Saved: {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("SchemaPacker ready. Use with DataNormalizer output.")
    print(f"  Schema version: {SchemaPacker.SCHEMA_VERSION}")
