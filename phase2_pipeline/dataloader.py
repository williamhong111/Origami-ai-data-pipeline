"""
dataloader.py — Universal Multimodal Data Loader
=================================================
Reads raw data from any source (HDF5, JSON, etc.) using a YAML config
that maps source-specific fields to a unified intermediate format.

Usage:
    loader = DataLoader(config_path="source_configs/isaac_sim.yaml")
    episodes = loader.load("mimic_dataset_1k.hdf5", max_episodes=5)

    # Each episode is a dict with unified keys:
    # {
    #     "demo_id": "demo_0",
    #     "num_steps": 206,
    #     "vision": { "table_cam_rgb": np.array(...), ... },
    #     "proprioception": { "joint_pos": np.array(...), ... },
    #     "actions": np.array(...),
    #     "audio": {},
    #     "tactile": {},
    #     "config": { ... }   # source config for downstream use
    # }
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
        elif self.file_format == "json":
            return self._load_json(file_path, max_episodes)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")

    # ------------------------------------------------------------------
    # HDF5 loader
    # ------------------------------------------------------------------
    def _load_hdf5(self, file_path: str, max_episodes: int) -> list:
        """Load episodes from an HDF5 file."""
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 files. Run: pip install h5py")

        f = h5py.File(file_path, "r")
        episodes = []

        # Find all demo groups
        demo_prefix = self.config.get("demo_prefix", "data/demo_")
        # Navigate to the parent group
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
                    # Use first vision stream to determine num_steps
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

            # Set num_steps from proprioception if vision didn't set it
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
    # JSON loader (for future data sources)
    # ------------------------------------------------------------------
    def _load_json(self, file_path: str, max_episodes: int) -> list:
        """Load episodes from a JSON file."""
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        # Assumes JSON is a list of episode records
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
                print(f"  - {s['name']} ({s['type']}): {s['hdf5_path']}")

        prop_cfg = self.config.get("proprioception", {})
        if prop_cfg:
            fields = [k for k in prop_cfg if isinstance(prop_cfg.get(k), dict)]
            print(f"Proprioception fields: {fields}")

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
