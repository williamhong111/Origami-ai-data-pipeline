"""
dataloader.py — Universal Multimodal Data Loader
=================================================
Reads raw data from any source (HDF5, JSON, TFRecord) using a YAML config
that maps source-specific fields to a unified intermediate format.

Supported formats:
    - HDF5  (NVIDIA Isaac Sim, RoboNet, etc.)
    - TFRecord (Google RT-1, Open X-Embodiment, Bridge, etc.)
    - JSON  (custom datasets)

Usage:
    loader = DataLoader(config_path="source_configs/isaac_sim.yaml")
    episodes = loader.load("mimic_dataset_1k.hdf5", max_episodes=5)

    loader = DataLoader(config_path="source_configs/rt1.yaml")
    episodes = loader.load("fractal20220817_data-train.tfrecord-00000-of-01024", max_episodes=3)
"""

import os
import json
import yaml
import numpy as np

# Optional: h5py for HDF5 files
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# Optional: tensorflow for TFRecord files
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    HAS_TF = True
except ImportError:
    HAS_TF = False


class DataLoader:
    """Universal data loader driven by YAML source configs."""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to a YAML config file that defines
                         how to read a specific data source.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.source_name = self.config.get("source_name", "unknown")
        self.file_format = self.config.get("format", "hdf5")

        print(f"[DataLoader] Loaded config for source: {self.source_name}")
        print(f"[DataLoader] Expected file format: {self.file_format}")

    def load(self, file_path: str, max_episodes: int = None) -> list:
        """
        Load episodes from a data file.

        Args:
            file_path:    Path to the raw data file.
            max_episodes: Maximum number of episodes to load (None = all).

        Returns:
            List of episode dicts in unified intermediate format.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        if self.file_format == "hdf5":
            return self._load_hdf5(file_path, max_episodes)
        elif self.file_format == "tfrecord":
            return self._load_tfrecord(file_path, max_episodes)
        elif self.file_format == "json":
            return self._load_json(file_path, max_episodes)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")

    # ------------------------------------------------------------------
    # HDF5 loader (Isaac Sim, RoboNet, etc.)
    # ------------------------------------------------------------------
    def _load_hdf5(self, file_path: str, max_episodes: int) -> list:
        """Load episodes from an HDF5 file."""
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 files. Run: pip install h5py")

        f = h5py.File(file_path, "r")
        episodes = []

        # Find all demo groups
        demo_prefix = self.config.get("demo_prefix", "data/demo_")
        parent_path = "/".join(demo_prefix.rstrip("/").rstrip("_").split("/")[:-1])
        if not parent_path:
            parent_path = "/"

        parent_group = f[parent_path] if parent_path != "/" else f
        demo_keys = sorted(
            [k for k in parent_group.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        )

        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]

        print(f"[DataLoader] Found {len(demo_keys)} episodes to load from {file_path}")

        for demo_key in demo_keys:
            demo_path = f"{parent_path}/{demo_key}" if parent_path != "/" else demo_key
            demo_group = f[demo_path]
            episode = self._parse_hdf5_episode(demo_group, demo_key)
            episodes.append(episode)

        f.close()
        print(f"[DataLoader] Successfully loaded {len(episodes)} episodes")
        return episodes

    def _parse_hdf5_episode(self, demo_group, demo_id: str) -> dict:
        """Parse a single episode from an HDF5 group into unified format."""

        episode = {
            "demo_id": demo_id,
            "num_steps": 0,
            "vision": {},
            "proprioception": {},
            "actions": None,
            "audio": {},
            "tactile": {},
            "language": {},
            "config": self.config,
        }

        # --- Vision streams ---
        vision_cfg = self.config.get("vision", {})
        if vision_cfg and "streams" in vision_cfg:
            for stream_cfg in vision_cfg["streams"]:
                h5_path = stream_cfg["hdf5_path"]
                name = stream_cfg["name"]
                if h5_path in demo_group:
                    data = demo_group[h5_path][:]
                    episode["vision"][name] = {
                        "data": data,
                        "type": stream_cfg["type"],
                        "sensor_id": stream_cfg["sensor_id"],
                        "source_id": stream_cfg["source_id"],
                        "shape": data.shape,
                    }
                    if episode["num_steps"] == 0:
                        episode["num_steps"] = data.shape[0]

        # --- Proprioception ---
        prop_cfg = self.config.get("proprioception", {})
        if prop_cfg:
            for field_name in ["joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_pos"]:
                field_cfg = prop_cfg.get(field_name)
                if field_cfg and isinstance(field_cfg, dict):
                    h5_path = field_cfg.get("hdf5_path")
                    if h5_path and h5_path in demo_group:
                        episode["proprioception"][field_name] = demo_group[h5_path][:]

            episode["proprioception"]["sensor_id"] = prop_cfg.get("sensor_id", "")
            episode["proprioception"]["source_id"] = prop_cfg.get("source_id", "")

            if episode["num_steps"] == 0 and "joint_pos" in episode["proprioception"]:
                episode["num_steps"] = episode["proprioception"]["joint_pos"].shape[0]

        # --- Actions ---
        actions_cfg = self.config.get("actions", {})
        if actions_cfg and isinstance(actions_cfg, dict):
            h5_path = actions_cfg.get("hdf5_path")
            if h5_path and h5_path in demo_group:
                episode["actions"] = demo_group[h5_path][:]

        # --- Audio (empty if not present) ---
        if not self.config.get("audio"):
            episode["audio"] = {}

        # --- Tactile (empty if not present) ---
        if not self.config.get("tactile"):
            episode["tactile"] = {}

        return episode

    # ------------------------------------------------------------------
    # TFRecord loader (RT-1, Open X-Embodiment, Bridge, etc.)
    # ------------------------------------------------------------------
    def _load_tfrecord(self, file_path: str, max_episodes: int) -> list:
        """Load episodes from a TFRecord file using tensorflow_datasets."""
        if not HAS_TF:
            raise ImportError(
                "tensorflow and tensorflow_datasets are required for TFRecord files.\n"
                "Run: pip install tensorflow tensorflow_datasets"
            )

        dataset_name = self.config.get("tfrecord_dataset_name", "")

        print(f"[DataLoader] Loading TFRecord: {dataset_name}")
        print(f"[DataLoader] File: {file_path}")
        print(f"[DataLoader] Loading via tensorflow_datasets (streaming from GCS)...")

        # Use tfds.load with try_gcs=True
        # This streams data from Google Cloud Storage (no auth needed for public datasets)
        ds = tfds.load(
            dataset_name,
            split="train",
            data_dir="gs://gresearch/robotics",
            try_gcs=True,
        )

        if max_episodes:
            ds = ds.take(max_episodes)

        episodes = []
        for i, example in enumerate(ds):
            episode = self._parse_tfrecord_episode(example, f"episode_{i}")
            episodes.append(episode)
            print(f"  → Parsed episode_{i}: {episode['num_steps']} steps"
                  f"{', instruction: ' + episode['language'].get('instruction', '')[:50] if episode.get('language') else ''}")

        print(f"[DataLoader] Successfully loaded {len(episodes)} episodes")
        return episodes

    def _parse_tfrecord_episode(self, example, demo_id: str) -> dict:
        """Parse a single TFRecord episode into unified intermediate format."""

        episode = {
            "demo_id": demo_id,
            "num_steps": 0,
            "vision": {},
            "proprioception": {},
            "actions": None,
            "audio": {},
            "tactile": {},
            "language": {},
            "config": self.config,
        }

        # Collect all steps into lists
        steps = list(example["steps"])
        num_steps = len(steps)
        episode["num_steps"] = num_steps

        if num_steps == 0:
            return episode

        # --- Vision ---
        vision_cfg = self.config.get("vision", {})
        if vision_cfg and "streams" in vision_cfg:
            for stream_cfg in vision_cfg["streams"]:
                tf_path = stream_cfg.get("tfrecord_path", "")
                name = stream_cfg["name"]

                # Parse dot-separated path: "observation.image"
                parts = tf_path.split(".")
                try:
                    frames = []
                    for step in steps:
                        val = step
                        for part in parts:
                            val = val[part]
                        frames.append(val.numpy())

                    data = np.array(frames)
                    episode["vision"][name] = {
                        "data": data,
                        "type": stream_cfg["type"],
                        "sensor_id": stream_cfg["sensor_id"],
                        "source_id": stream_cfg["source_id"],
                        "shape": data.shape,
                    }
                except (KeyError, AttributeError) as e:
                    print(f"[DataLoader] Warning: Could not extract vision '{name}': {e}")

        # --- Proprioception ---
        prop_cfg = self.config.get("proprioception", {})
        if prop_cfg:
            for field_name, field_cfg in prop_cfg.items():
                if field_name in ("sensor_id", "source_id"):
                    continue
                if not isinstance(field_cfg, dict):
                    continue

                tf_path = field_cfg.get("tfrecord_path", "")
                parts = tf_path.split(".")

                try:
                    values = []
                    for step in steps:
                        val = step
                        for part in parts:
                            val = val[part]
                        values.append(val.numpy())

                    episode["proprioception"][field_name] = np.array(values)
                except (KeyError, AttributeError) as e:
                    print(f"[DataLoader] Warning: Could not extract prop '{field_name}': {e}")

            episode["proprioception"]["sensor_id"] = prop_cfg.get("sensor_id", "")
            episode["proprioception"]["source_id"] = prop_cfg.get("source_id", "")

        # --- Actions ---
        actions_cfg = self.config.get("actions", {})
        if actions_cfg:
            action_fields = actions_cfg.get("fields", [])
            if action_fields:
                # Concatenate all action fields into a single array
                all_actions = []
                for step in steps:
                    step_action = []
                    for field in action_fields:
                        tf_path = field.get("tfrecord_path", "")
                        parts = tf_path.split(".")
                        try:
                            val = step
                            for part in parts:
                                val = val[part]
                            step_action.append(val.numpy().flatten())
                        except (KeyError, AttributeError):
                            pass
                    if step_action:
                        all_actions.append(np.concatenate(step_action))

                if all_actions:
                    episode["actions"] = np.array(all_actions)
            else:
                # Single action path (like HDF5)
                h5_path = actions_cfg.get("hdf5_path")
                if h5_path:
                    pass  # handled by HDF5 loader

        # --- Language ---
        lang_cfg = self.config.get("language", {})
        if lang_cfg and isinstance(lang_cfg, dict):
            tf_path = lang_cfg.get("tfrecord_path", "")
            if tf_path:
                parts = tf_path.split(".")
                try:
                    # Get instruction from first step
                    val = steps[0]
                    for part in parts:
                        val = val[part]
                    instruction = val.numpy()
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode("utf-8")
                    episode["language"] = {
                        "instruction": instruction,
                        "language": "en",
                        "source": "human",
                    }
                except (KeyError, AttributeError) as e:
                    print(f"[DataLoader] Warning: Could not extract language: {e}")

        # --- Attributes (RT-1 specific metadata) ---
        if "attributes" in example:
            attrs = example["attributes"]
            episode["rt1_attributes"] = {}
            for k, v in attrs.items():
                try:
                    val = v.numpy()
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    episode["rt1_attributes"][k] = val
                except:
                    pass

        # --- Audio (empty if not present) ---
        if not self.config.get("audio"):
            episode["audio"] = {}

        # --- Tactile (empty if not present) ---
        if not self.config.get("tactile"):
            episode["tactile"] = {}

        return episode

    # ------------------------------------------------------------------
    # JSON loader (for future data sources)
    # ------------------------------------------------------------------
    def _load_json(self, file_path: str, max_episodes: int) -> list:
        """Load episodes from a JSON file."""
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            episodes_raw = raw_data[:max_episodes] if max_episodes else raw_data
        elif isinstance(raw_data, dict) and "episodes" in raw_data:
            episodes_raw = raw_data["episodes"][:max_episodes] if max_episodes else raw_data["episodes"]
        else:
            raise ValueError("JSON format not recognized. Expected list or {episodes: [...]}.")

        episodes = []
        for i, ep_raw in enumerate(episodes_raw):
            episode = {
                "demo_id": ep_raw.get("id", f"demo_{i}"),
                "num_steps": ep_raw.get("num_steps", 0),
                "vision": ep_raw.get("vision", {}),
                "proprioception": ep_raw.get("proprioception", {}),
                "actions": ep_raw.get("actions"),
                "audio": ep_raw.get("audio", {}),
                "tactile": ep_raw.get("tactile", {}),
                "language": ep_raw.get("language", {}),
                "config": self.config,
            }
            episodes.append(episode)

        print(f"[DataLoader] Loaded {len(episodes)} episodes from JSON")
        return episodes

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        """Return the loaded source config."""
        return self.config

    def describe(self):
        """Print a summary of what this loader expects."""
        print(f"\n{'='*50}")
        print(f"DataLoader Config Summary")
        print(f"{'='*50}")
        print(f"Source:  {self.source_name}")
        print(f"Format:  {self.file_format}")

        vision_cfg = self.config.get("vision", {})
        if vision_cfg and "streams" in vision_cfg:
            print(f"Vision streams: {len(vision_cfg['streams'])}")
            for s in vision_cfg["streams"]:
                path_key = "hdf5_path" if "hdf5_path" in s else "tfrecord_path"
                print(f"  - {s['name']} ({s['type']}): {s.get(path_key, 'N/A')}")

        prop_cfg = self.config.get("proprioception", {})
        if prop_cfg:
            fields = [k for k in prop_cfg if isinstance(prop_cfg.get(k), dict)]
            print(f"Proprioception fields: {fields}")

        lang_cfg = self.config.get("language", {})
        if lang_cfg and isinstance(lang_cfg, dict):
            print(f"Language: present (from {lang_cfg.get('tfrecord_path', 'config')})")
        else:
            print(f"Language: not available")

        for mod in ["audio", "tactile", "imu"]:
            val = self.config.get(mod)
            print(f"{mod.capitalize()}: {'present' if val else 'not available'}")

        print(f"{'='*50}\n")


# ------------------------------------------------------------------
# CLI entry point for testing
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "source_configs/isaac_sim.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else None

    loader = DataLoader(config_path)
    loader.describe()

    if data_path:
        episodes = loader.load(data_path, max_episodes=2)
        for ep in episodes:
            print(f"\n--- {ep['demo_id']} ({ep['num_steps']} steps) ---")
            print(f"  Vision keys:         {list(ep['vision'].keys())}")
            print(f"  Proprioception keys: {list(ep['proprioception'].keys())}")
            print(f"  Actions shape:       {ep['actions'].shape if ep['actions'] is not None else 'None'}")
            print(f"  Audio:               {'present' if ep['audio'] else 'empty'}")
            print(f"  Tactile:             {'present' if ep['tactile'] else 'empty'}")
            if ep.get("language"):
                print(f"  Language:            {ep['language'].get('instruction', 'N/A')[:80]}")
