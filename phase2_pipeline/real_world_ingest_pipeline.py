"""
real_world_ingest_pipeline.py — Main Ingestion Pipeline
=========================================================
Chains all modules together:
    DataLoader → DataNormalizer → SchemaPacker → Validator

Supports multiple data sources:
    - NVIDIA Isaac Sim (HDF5)
    - Google RT-1 (TFRecord)
    - Any future source with a YAML config

Usage:
    # Process Isaac Sim data (default):
    python real_world_ingest_pipeline.py

    # Process RT-1 data:
    python real_world_ingest_pipeline.py \
        --config source_configs/rt1.yaml \
        --data ../fractal20220817_data-train.tfrecord-00000-of-01024 \
        --max-episodes 3 \
        --output-dir output_rt1/

    # Custom source and file:
    python real_world_ingest_pipeline.py \
        --config source_configs/isaac_sim.yaml \
        --data mimic_dataset_1k.hdf5 \
        --max-episodes 5 \
        --output-dir output/
"""

import os
import sys
import json
import argparse
import time

from dataloader import DataLoader
from data_normalizer import DataNormalizer
from schema_packer import SchemaPacker
from validator import Validator


def run_pipeline(config_path: str, data_path: str,
                 max_episodes: int = 5, output_dir: str = "output"):
    """
    Run the full ingestion pipeline.

    Steps:
        1. Load raw data using DataLoader + source config
        2. Normalize each episode using DataNormalizer
        3. Pack into canonical schema using SchemaPacker
        4. Validate each record using Validator
        5. Save valid records to output directory

    Args:
        config_path:   Path to source YAML config
        data_path:     Path to raw data file
        max_episodes:  Max episodes to process (None = all)
        output_dir:    Directory for output JSON files
    """
    print("=" * 60)
    print("  Multimodal Data Ingestion Pipeline")
    print("  Phase 2: Real-World Data Pipeline")
    print("=" * 60)
    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load raw data
    # ------------------------------------------------------------------
    print("\n[Step 1/4] Loading raw data...")
    loader = DataLoader(config_path)
    loader.describe()
    episodes = loader.load(data_path, max_episodes=max_episodes)
    print(f"  → Loaded {len(episodes)} episodes")

    # ------------------------------------------------------------------
    # Step 2: Normalize
    # ------------------------------------------------------------------
    print("\n[Step 2/4] Normalizing data...")
    normalizer = DataNormalizer()
    normalized_episodes = []

    for ep in episodes:
        normalized = normalizer.normalize(ep)
        normalized_episodes.append(normalized)
        lang_info = ""
        if normalized.get("language", {}).get("instruction"):
            lang_info = f", instruction: '{normalized['language']['instruction'][:50]}'"
        print(f"  → Normalized {ep['demo_id']}: "
              f"{ep['num_steps']} steps, "
              f"timestamps {normalized['global_timestamp']['start_ts']}"
              f"..{normalized['global_timestamp']['end_ts']}"
              f"{lang_info}")

    # ------------------------------------------------------------------
    # Step 3: Pack into schema
    # ------------------------------------------------------------------
    print("\n[Step 3/4] Packing into canonical schema...")
    packer = SchemaPacker()
    records = []

    for norm_ep in normalized_episodes:
        record = packer.pack(norm_ep)
        records.append(record)
        lang_text = record.get("task_context", {}).get("language_instruction", {}).get("text", "")
        lang_info = f", lang: '{lang_text[:40]}'" if lang_text else ""
        print(f"  → Packed {record['sample_id']}: "
              f"{len(record['modalities']['vision']['streams'])} vision streams, "
              f"{len(record['modalities']['proprioception']['timestamps'])} prop steps"
              f"{lang_info}")

    # ------------------------------------------------------------------
    # Step 4: Validate
    # ------------------------------------------------------------------
    print("\n[Step 4/4] Validating records...")
    validator = Validator()
    valid_records = []
    total_errors = 0
    total_warnings = 0

    for record in records:
        result = validator.validate(record)
        result.print_report()
        total_errors += len(result.errors)
        total_warnings += len(result.warnings)

        if result.is_valid:
            valid_records.append(record)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Saving outputs to: {output_dir}/")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    for record in valid_records:
        output_record = _strip_large_payloads(record)

        filename = f"{record['sample_id']}.json"
        filepath = os.path.join(output_dir, filename)
        packer.save(output_record, filepath)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline Summary")
    print(f"{'='*60}")
    print(f"  Episodes processed:  {len(episodes)}")
    print(f"  Records generated:   {len(records)}")
    print(f"  Valid records:       {len(valid_records)}")
    print(f"  Total errors:        {total_errors}")
    print(f"  Total warnings:      {total_warnings}")
    print(f"  Output directory:    {output_dir}/")
    print(f"  Time elapsed:        {elapsed:.2f}s")
    print(f"{'='*60}\n")

    return valid_records


def _strip_large_payloads(record: dict) -> dict:
    """
    Remove large numpy/binary data from record before saving as JSON.
    Vision pixel data is too large for JSON; only keep metadata and refs.
    """
    import copy
    output = copy.deepcopy(record)
    return output


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Data Ingestion Pipeline (Phase 2)"
    )
    parser.add_argument(
        "--config", "-c",
        default="source_configs/isaac_sim.yaml",
        help="Path to source config YAML (default: source_configs/isaac_sim.yaml)"
    )
    parser.add_argument(
        "--data", "-d",
        default=None,
        help="Path to raw data file"
    )
    parser.add_argument(
        "--max-episodes", "-n",
        type=int, default=5,
        help="Max episodes to process (default: 5)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for schema records (default: output/)"
    )

    args = parser.parse_args()

    if args.data is None:
        # Auto-detect data file based on config
        if "rt1" in args.config:
            default_paths = [
                "../fractal20220817_data-train.tfrecord-00000-of-01024",
                os.path.expanduser("~/Desktop/files/fractal20220817_data-train.tfrecord-00000-of-01024"),
            ]
        else:
            default_paths = [
                "mimic_dataset_1k.hdf5",
                "../mimic_dataset_1k.hdf5",
                os.path.expanduser("~/data/mimic_dataset_1k.hdf5"),
                os.path.expanduser("~/Desktop/files/mimic_dataset_1k.hdf5"),
            ]

        for p in default_paths:
            if os.path.exists(p):
                args.data = p
                break

        if args.data is None:
            print("ERROR: No data file found. Use --data to specify path.")
            print(f"  Tried: {default_paths}")
            sys.exit(1)

    run_pipeline(
        config_path=args.config,
        data_path=args.data,
        max_episodes=args.max_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
