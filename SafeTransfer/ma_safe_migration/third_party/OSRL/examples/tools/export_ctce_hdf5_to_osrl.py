"""Export FSRL TrajectoryBuffer HDF5 dataset into OSRL-readable NPZ format."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export CTCE HDF5 dataset to OSRL NPZ format")
    parser.add_argument("--input-hdf5", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--policy-tag", type=str, default="fsrl_ctce")
    parser.add_argument("--checkpoint-id", type=str, default="unknown")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schema-version", type=str, default="ctce_v1")
    return parser.parse_args()


def _read_hdf5_group(group: h5py.Group, out: Dict[str, np.ndarray]):
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            out[key] = value[()]
        elif isinstance(value, h5py.Group):
            _read_hdf5_group(value, out)


def _normalize_transition_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def _as_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(arr.shape[0], -1)[:, 0]


def main():
    args = parse_args()

    if not os.path.isfile(args.input_hdf5):
        raise FileNotFoundError(f"Input file not found: {args.input_hdf5}")

    raw: Dict[str, np.ndarray] = {}
    with h5py.File(args.input_hdf5, "r") as f:
        _read_hdf5_group(f, raw)

    required = [
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "costs",
        "terminals",
        "timeouts",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing keys in HDF5 dataset: {missing}. Found keys: {sorted(raw.keys())}")

    dataset = {
        "observations": _normalize_transition_shape(raw["observations"]).astype(np.float32),
        "next_observations": _normalize_transition_shape(raw["next_observations"]).astype(np.float32),
        "actions": _normalize_transition_shape(raw["actions"]).astype(np.float32),
        "rewards": _as_1d(raw["rewards"]).astype(np.float32),
        "costs": _as_1d(raw["costs"]).astype(np.float32),
        "terminals": _as_1d(raw["terminals"]).astype(np.float32),
        "timeouts": _as_1d(raw["timeouts"]).astype(np.float32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **dataset)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ctce_schema_version": args.schema_version,
        "num_agents": int(args.num_agents),
        "policy_tag": args.policy_tag,
        "checkpoint_id": args.checkpoint_id,
        "seed": int(args.seed),
        "num_transitions": int(dataset["observations"].shape[0]),
        "obs_dim": int(dataset["observations"].shape[-1]),
        "act_dim": int(dataset["actions"].shape[-1]),
    }

    meta_path = os.path.splitext(args.output_npz)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Export completed:")
    print(f"  npz: {args.output_npz}")
    print(f"  meta: {meta_path}")
    print(f"  transitions: {metadata['num_transitions']}")
    print(f"  obs_dim: {metadata['obs_dim']}, act_dim: {metadata['act_dim']}")


if __name__ == "__main__":
    main()
