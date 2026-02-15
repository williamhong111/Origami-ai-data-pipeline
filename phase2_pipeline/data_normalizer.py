"""
data_normalizer.py — Multimodal Data Normalization
====================================================
Normalizes raw episode data from dataloader into standardized format
following the rules defined in normalization_rules.md:

  - Timestamps:      unix_epoch_ms for all modalities
  - Vision:          30 Hz, 1280x720 (record actual if can't resize)
  - Proprioception:  100 Hz, radians
  - Audio:           16,000 Hz
  - Tactile:         unix_epoch_ms timestamps, consistent units
  - Missing data:    keep required keys, use empty arrays

Usage:
    normalizer = DataNormalizer()
    normalized = normalizer.normalize(episode)
"""

import time
import numpy as np


class DataNormalizer:
    """Normalize raw episode data per normalization_rules.md."""

    # Target standards from normalization_rules.md
    VISION_TARGET_FPS = 30.0
    VISION_TARGET_RESOLUTION = "1280x720"
    PROPRIOCEPTION_TARGET_HZ = 100.0
    AUDIO_TARGET_HZ = 16000.0

    def __init__(self, base_timestamp_ms: int = None):
        """
        Args:
            base_timestamp_ms: Starting timestamp in unix_epoch_ms.
                               If None, uses current time as base.
        """
        if base_timestamp_ms is None:
            self.base_timestamp_ms = int(time.time() * 1000)
        else:
            self.base_timestamp_ms = base_timestamp_ms

    def normalize(self, episode: dict) -> dict:
        """
        Normalize a single episode from dataloader output.

        Args:
            episode: Raw episode dict from DataLoader.

        Returns:
            Normalized episode dict with timestamps and standardized fields.
        """
        num_steps = episode["num_steps"]
        config = episode.get("config", {})
        control_rate = config.get("sampling", {}).get("control_rate_hz", 20.0)

        # Calculate episode time bounds
        step_duration_ms = int(1000.0 / control_rate)
        start_ts = self.base_timestamp_ms
        end_ts = start_ts + (num_steps - 1) * step_duration_ms

        normalized = {
            "demo_id": episode["demo_id"],
            "num_steps": num_steps,
            "global_timestamp": {
                "time_base": "unix_epoch_ms",
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
            "vision": self._normalize_vision(
                episode.get("vision", {}), num_steps, start_ts, step_duration_ms
            ),
            "proprioception": self._normalize_proprioception(
                episode.get("proprioception", {}), num_steps, start_ts, step_duration_ms
            ),
            "actions": self._normalize_actions(episode.get("actions")),
            "audio": self._normalize_audio(episode.get("audio", {}), start_ts),
            "tactile": self._normalize_tactile(episode.get("tactile", {}), start_ts),
            "config": config,
        }

        # Increment base timestamp for next episode to avoid overlap
        self.base_timestamp_ms = end_ts + step_duration_ms

        return normalized

    # ------------------------------------------------------------------
    # Vision normalization
    # ------------------------------------------------------------------
    def _normalize_vision(self, vision_data: dict, num_steps: int,
                          start_ts: int, step_duration_ms: int) -> dict:
        """
        Normalize vision streams.

        Per normalization_rules.md:
        - Target 30 Hz, 1280x720
        - Store externally, reference via data_ref
        - Generate timestamps aligned to frames
        - Do NOT fabricate frames if missing
        """
        normalized_streams = {}

        for stream_name, stream_info in vision_data.items():
            data = stream_info.get("data")
            if data is None:
                continue

            T = data.shape[0]
            # Actual resolution from data
            if len(data.shape) == 4:  # (T, H, W, C)
                actual_h, actual_w = data.shape[1], data.shape[2]
            elif len(data.shape) == 3:  # (T, H, W)
                actual_h, actual_w = data.shape[1], data.shape[2]
            else:
                actual_h, actual_w = 0, 0

            actual_resolution = f"{actual_w}x{actual_h}"

            # Compute actual frame rate from control rate
            actual_fps = 1000.0 / step_duration_ms if step_duration_ms > 0 else 0.0

            # Generate timestamps for each frame
            timestamps = [start_ts + i * step_duration_ms for i in range(T)]

            normalized_streams[stream_name] = {
                "data": data,
                "type": stream_info.get("type", "rgb"),
                "sensor_id": stream_info.get("sensor_id", stream_name),
                "source_id": stream_info.get("source_id", ""),
                "frame_rate_hz": actual_fps,
                "target_frame_rate_hz": self.VISION_TARGET_FPS,
                "resolution": actual_resolution,
                "target_resolution": self.VISION_TARGET_RESOLUTION,
                "timestamps": timestamps,
                "num_frames": T,
            }

        return normalized_streams

    # ------------------------------------------------------------------
    # Proprioception normalization
    # ------------------------------------------------------------------
    def _normalize_proprioception(self, prop_data: dict, num_steps: int,
                                  start_ts: int, step_duration_ms: int) -> dict:
        """
        Normalize proprioception data.

        Per normalization_rules.md:
        - Target 100 Hz
        - Joint states in radians
        - IMU: acc in m/s^2, gyro in rad/s
        - Missing fields → empty arrays
        """
        # Generate timestamps at control rate
        timestamps = [start_ts + i * step_duration_ms for i in range(num_steps)]
        actual_hz = 1000.0 / step_duration_ms if step_duration_ms > 0 else 0.0

        normalized = {
            "sensor_id": prop_data.get("sensor_id", ""),
            "source_id": prop_data.get("source_id", ""),
            "sampling_rate_hz": actual_hz,
            "target_sampling_rate_hz": self.PROPRIOCEPTION_TARGET_HZ,
            "timestamps": timestamps,
            "joint_states": [],
            "joint_vel": [],
            "eef_pos": [],
            "eef_quat": [],
            "gripper_pos": [],
            "imu": {
                "acc": [],
                "gyro": [],
            },
        }

        # Map available fields
        if "joint_pos" in prop_data and isinstance(prop_data["joint_pos"], np.ndarray):
            normalized["joint_states"] = prop_data["joint_pos"].tolist()

        if "joint_vel" in prop_data and isinstance(prop_data["joint_vel"], np.ndarray):
            normalized["joint_vel"] = prop_data["joint_vel"].tolist()

        if "eef_pos" in prop_data and isinstance(prop_data["eef_pos"], np.ndarray):
            normalized["eef_pos"] = prop_data["eef_pos"].tolist()

        if "eef_quat" in prop_data and isinstance(prop_data["eef_quat"], np.ndarray):
            normalized["eef_quat"] = prop_data["eef_quat"].tolist()

        if "gripper_pos" in prop_data and isinstance(prop_data["gripper_pos"], np.ndarray):
            normalized["gripper_pos"] = prop_data["gripper_pos"].tolist()

        # IMU — not present in Isaac Sim Mimic, keep empty per missing data policy
        if "imu_acc" in prop_data and isinstance(prop_data["imu_acc"], np.ndarray):
            normalized["imu"]["acc"] = prop_data["imu_acc"].tolist()

        if "imu_gyro" in prop_data and isinstance(prop_data["imu_gyro"], np.ndarray):
            normalized["imu"]["gyro"] = prop_data["imu_gyro"].tolist()

        return normalized

    # ------------------------------------------------------------------
    # Actions normalization
    # ------------------------------------------------------------------
    def _normalize_actions(self, actions) -> list:
        """Convert actions to list format."""
        if actions is not None and isinstance(actions, np.ndarray):
            return actions.tolist()
        return []

    # ------------------------------------------------------------------
    # Audio normalization
    # ------------------------------------------------------------------
    def _normalize_audio(self, audio_data: dict, start_ts: int) -> dict:
        """
        Normalize audio data.

        Per normalization_rules.md:
        - Target 16,000 Hz
        - If absent, keep block but leave timestamps empty
        """
        if not audio_data:
            return {
                "sensor_id": "",
                "sampling_rate_hz": self.AUDIO_TARGET_HZ,
                "timestamps": [],
                "data_ref": "",
            }

        return {
            "sensor_id": audio_data.get("sensor_id", ""),
            "sampling_rate_hz": audio_data.get("sampling_rate_hz", self.AUDIO_TARGET_HZ),
            "timestamps": audio_data.get("timestamps", []),
            "data_ref": audio_data.get("data_ref", ""),
        }

    # ------------------------------------------------------------------
    # Tactile normalization
    # ------------------------------------------------------------------
    def _normalize_tactile(self, tactile_data: dict, start_ts: int) -> dict:
        """
        Normalize tactile data.

        Per normalization_rules.md:
        - Timestamps in unix_epoch_ms
        - Keep missing arrays empty
        """
        if not tactile_data:
            return {
                "sensor_id": "",
                "timestamps": [],
                "force": [],
                "pressure": [],
            }

        return {
            "sensor_id": tactile_data.get("sensor_id", ""),
            "timestamps": tactile_data.get("timestamps", []),
            "force": tactile_data.get("force", []),
            "pressure": tactile_data.get("pressure", []),
        }


# ------------------------------------------------------------------
# CLI entry point for testing
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("DataNormalizer ready. Use with DataLoader output.")
    print(f"  Vision target:          {DataNormalizer.VISION_TARGET_FPS} Hz, {DataNormalizer.VISION_TARGET_RESOLUTION}")
    print(f"  Proprioception target:  {DataNormalizer.PROPRIOCEPTION_TARGET_HZ} Hz")
    print(f"  Audio target:           {DataNormalizer.AUDIO_TARGET_HZ} Hz")
