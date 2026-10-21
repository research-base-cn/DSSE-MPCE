"""Generate and store power-system trajectory datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dsse.feeder import IEEE123BusFeeder
from dsse.reproducibility import (
    assert_disjoint_splits,
    build_run_manifest,
    canonical_sha256,
    expand_seed_spec,
    git_state,
    sha256_file,
    verify_data_manifest,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_KEYS = (
    "scada",
    "scada_mask",
    "pmu",
    "pmu_mask",
    "v_true",
    "angle_true",
)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


def formal_splits(config: dict[str, Any]) -> dict[str, list[int]]:
    data = config["data"]
    splits = {
        "train": expand_seed_spec(data["train_seeds"]),
        "validation": expand_seed_spec(data["validation_seeds"]),
        "sealed_test": expand_seed_spec(data["sealed_test_seeds"]),
    }
    assert_disjoint_splits(splits)
    forbidden = {
        seed
        for spec in data["pilot_seed_ranges_forbidden_in_formal_study"]
        for seed in expand_seed_spec(spec)
    }
    overlap = forbidden.intersection(
        seed for split_seeds in splits.values() for seed in split_seeds
    )
    if overlap:
        raise ValueError(f"Formal split reuses pilot trajectories: {sorted(overlap)[:10]}")
    return splits


def validate_formal_data_config(config: dict[str, Any]) -> None:
    if config.get("status") != "PREDECLARED_NOT_YET_RUN":
        raise ValueError("Formal data config is not in its predeclared state")
    cache = config["data"]["cache"]
    if not cache.get("required", False):
        raise ValueError("Formal data cache must be required")
    if cache.get("format") != "compressed_npz_shards":
        raise ValueError("Only the checked compressed_npz_shards format is supported")
    if int(cache["trajectories_per_shard"]) <= 0:
        raise ValueError("trajectories_per_shard must be positive")
    formal_splits(config)


class ChannelStatistics:
    def __init__(self, width: int) -> None:
        self.count = np.zeros(width, dtype=np.float64)
        self.total = np.zeros(width, dtype=np.float64)
        self.total_square = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        if values.shape != mask.shape or values.shape[-1] != self.count.size:
            raise ValueError("Normalizer statistics received an incompatible array")
        available = mask > 0.5
        axes = tuple(range(values.ndim - 1))
        self.count += available.sum(axis=axes, dtype=np.float64)
        self.total += np.where(available, values, 0.0).sum(axis=axes, dtype=np.float64)
        self.total_square += np.where(available, values**2, 0.0).sum(
            axis=axes, dtype=np.float64
        )

    def finalize(self) -> tuple[list[float], list[float], list[int]]:
        mean = np.divide(
            self.total,
            self.count,
            out=np.zeros_like(self.total),
            where=self.count > 0,
        )
        second = np.divide(
            self.total_square,
            self.count,
            out=np.zeros_like(self.total_square),
            where=self.count > 0,
        )
        variance = np.maximum(second - mean**2, 0.0)
        scale = np.sqrt(variance)
        scale = np.where(self.count > 0, np.maximum(scale, 1e-6), 1.0)
        return mean.tolist(), scale.tolist(), self.count.astype(np.int64).tolist()


def stack_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("Cannot write an empty shard")
    payload = {
        key: np.stack([sample[key] for sample in samples], axis=0).astype(
            np.float32, copy=False
        )
        for key in ARRAY_KEYS
    }
    payload["trajectory_seed"] = np.asarray(
        [sample["trajectory_seed"] for sample in samples], dtype=np.int64
    )
    return payload


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def read_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        expected = set(ARRAY_KEYS).union({"trajectory_seed"})
        if keys != expected:
            raise ValueError(f"Unexpected arrays in {path}: {sorted(keys)}")
        return {key: archive[key] for key in archive.files}


def verify_cache(output_dir: Path, require_formal_admissible: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError("Cache manifest is incomplete")
    if int(manifest.get("cache_schema_version", -1)) != 1:
        raise ValueError("Unsupported formal cache schema")
    if require_formal_admissible and not manifest.get("admissible_for_formal_study"):
        raise ValueError("Smoke-limited cache is not admissible for the formal study")

    config = manifest["config"]
    validate_formal_data_config(config)
    if canonical_sha256(config) != manifest["config_sha256"]:
        raise RuntimeError("Manifest configuration hash mismatch")
    resolved_config_path = output_dir / manifest["resolved_config"]["relative_path"]
    if sha256_file(resolved_config_path) != manifest["resolved_config"]["sha256"]:
        raise RuntimeError("Resolved configuration file hash mismatch")
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if resolved_config != config:
        raise RuntimeError("Resolved configuration differs from the manifest configuration")

    expected_split_names = {"train", "validation", "sealed_test"}
    if set(manifest["shards"]) != expected_split_names:
        raise RuntimeError("Cache does not contain the three required formal splits")
    embedded_splits = {
        name: [int(seed) for seed in seeds]
        for name, seeds in manifest["split_seeds"].items()
    }
    if set(embedded_splits) != expected_split_names:
        raise RuntimeError("Manifest split metadata is incomplete")
    assert_disjoint_splits(embedded_splits)
    if canonical_sha256(embedded_splits) != manifest["split_sha256"]:
        raise RuntimeError("Manifest split seed hash mismatch")

    sequence_length = int(config["data"]["sequence_length"])
    num_buses = int(manifest["num_opendss_buses"])
    pmu_count = int(manifest["pmu_count"])
    mapping = manifest["bus_mapping"]
    if len(mapping) != num_buses:
        raise RuntimeError("Bus mapping length differs from the cached bus count")
    if [int(row["model_index"]) for row in mapping] != list(range(num_buses)):
        raise RuntimeError("Bus mapping indices are not contiguous and ordered")
    bus_names = [str(row["opendss_bus"]) for row in mapping]
    if len(set(bus_names)) != num_buses:
        raise RuntimeError("Bus mapping contains duplicate OpenDSS names")
    expected_pmu_indices = [
        bus_names.index(str(name)) for name in config["data"]["pmu_bus_names"]
    ]
    if [int(index) for index in manifest["pmu_bus_indices"]] != expected_pmu_indices:
        raise RuntimeError("Ordered PMU indices differ from the configured PMU names")

    topology = manifest["topology"]
    if canonical_sha256(topology["edge_index"]) != topology["sha256"]:
        raise RuntimeError("Topology hash mismatch")
    edge_index = np.asarray(topology["edge_index"], dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
        raise RuntimeError("Malformed cached edge index")
    if edge_index.min() < 0 or edge_index.max() >= num_buses:
        raise RuntimeError("Cached edge index references an unknown bus")
    directed_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if any((right, left) not in directed_edges for left, right in directed_edges):
        raise RuntimeError("Cached topology is missing a reverse directed edge")

    sensitivity = manifest["nominal_voltage_influence"]
    if canonical_sha256(sensitivity["records"]) != sensitivity["sha256"]:
        raise RuntimeError("Nominal voltage influence hash mismatch")
    if len(sensitivity["records"]) != num_buses:
        raise RuntimeError("Nominal voltage influence does not cover every bus")
    for expected_index, record in enumerate(sensitivity["records"]):
        if int(record["model_index"]) != expected_index:
            raise RuntimeError("Nominal voltage influence indices are misaligned")
        if str(record["opendss_bus"]) != bus_names[expected_index]:
            raise RuntimeError("Nominal voltage influence bus mapping is misaligned")
        values = np.asarray(
            [
                record["mean_abs_voltage_change_per_relative_load_step"],
                record["max_abs_voltage_change_per_relative_load_step"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError("Invalid nominal voltage influence value")
    observed_seeds: dict[str, list[int]] = {}
    for split, records in manifest["shards"].items():
        observed_seeds[split] = []
        for record in records:
            path = output_dir / record["relative_path"]
            observed_hash = sha256_file(path)
            if observed_hash != record["sha256"]:
                raise RuntimeError(
                    f"Cache shard hash mismatch: {path}, "
                    f"expected={record['sha256']}, observed={observed_hash}"
                )
            payload = read_shard(path)
            seeds = payload["trajectory_seed"].astype(np.int64).tolist()
            if seeds != record["trajectory_seeds"]:
                raise RuntimeError(f"Trajectory seed mismatch inside {path}")
            count = len(seeds)
            if count != int(record["trajectory_count"]):
                raise RuntimeError(f"Trajectory count mismatch inside {path}")
            if payload["trajectory_seed"].shape != (count,):
                raise RuntimeError(f"Trajectory seed shape mismatch inside {path}")
            if payload["trajectory_seed"].dtype != np.dtype(np.int64):
                raise RuntimeError(f"Trajectory seed dtype mismatch inside {path}")
            expected_shapes = {
                "scada": [count, sequence_length, num_buses, 3],
                "scada_mask": [count, sequence_length, num_buses, 3],
                "pmu": [count, sequence_length, pmu_count, 2],
                "pmu_mask": [count, sequence_length, pmu_count, 2],
                "v_true": [count, sequence_length, num_buses],
                "angle_true": [count, sequence_length, num_buses],
            }
            for key in ARRAY_KEYS:
                if list(payload[key].shape) != record["array_shapes"][key]:
                    raise RuntimeError(f"Array shape mismatch for {key} inside {path}")
                if list(payload[key].shape) != expected_shapes[key]:
                    raise RuntimeError(f"Unexpected formal shape for {key} inside {path}")
                if str(payload[key].dtype) != record["array_dtypes"][key]:
                    raise RuntimeError(f"Array dtype mismatch for {key} inside {path}")
                if payload[key].dtype != np.dtype(np.float32):
                    raise RuntimeError(f"Unexpected formal dtype for {key} inside {path}")
                if not np.isfinite(payload[key]).all():
                    raise RuntimeError(f"Non-finite values in {key} inside {path}")
            for key in ("scada_mask", "pmu_mask"):
                if not np.isin(payload[key], (0.0, 1.0)).all():
                    raise RuntimeError(f"Non-binary mask in {key} inside {path}")
            observed_seeds[split].extend(seeds)
    if observed_seeds != embedded_splits:
        raise RuntimeError("Cached trajectory order differs from manifest split metadata")
    if canonical_sha256(observed_seeds) != manifest["cached_split_sha256"]:
        raise RuntimeError("Cached split seed hash mismatch")
    full_splits = formal_splits(config)
    is_full_cache = observed_seeds == full_splits
    if bool(manifest["admissible_for_formal_study"]) != is_full_cache:
        raise RuntimeError("Formal-admissibility flag is inconsistent with cached splits")

    normalizer_path = output_dir / manifest["normalizer"]["relative_path"]
    if sha256_file(normalizer_path) != manifest["normalizer"]["sha256"]:
        raise RuntimeError("Training-only normalizer hash mismatch")
    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if normalizer.get("fit_split") != "train":
        raise RuntimeError("Normalizer was not declared as training-only")
    for prefix, width in (("scada", 3), ("pmu", 2)):
        for suffix in ("mean", "scale", "available_count"):
            values = np.asarray(normalizer[f"{prefix}_{suffix}"])
            if values.shape != (width,) or not np.isfinite(values).all():
                raise RuntimeError(f"Malformed {prefix}_{suffix} normalizer field")
        if np.any(np.asarray(normalizer[f"{prefix}_scale"]) <= 0.0):
            raise RuntimeError(f"Non-positive {prefix} normalizer scale")
    return manifest


def generate_cache(
    config_path: Path,
    output_dir: Path,
    smoke_limit_per_split: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_formal_data_config(config)
    full_splits = formal_splits(config)
    if smoke_limit_per_split is not None:
        if int(smoke_limit_per_split) <= 0:
            raise ValueError("smoke_limit_per_split must be positive")
        splits = {
            name: seeds[: int(smoke_limit_per_split)]
            for name, seeds in full_splits.items()
        }
        admissible = False
    else:
        splits = full_splits
        admissible = True
        if git_state(REPO_ROOT)["dirty"]:
            raise ValueError("Full formal cache generation requires a clean worktree")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Formal cache output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    resolved_config_path = output_dir / "config.resolved.json"
    write_json(resolved_config_path, config)

    data_config = config["data"]
    data_dir = (REPO_ROOT / data_config["dss_dir"]).resolve()
    data_hashes = verify_data_manifest(data_dir)
    feeder = IEEE123BusFeeder(data_dir, data_config["pmu_bus_names"])
    topology = {
        "edge_index": feeder.edge_index.astype(np.int64).tolist(),
        "edge_index_orientation": "directed_pairs_with_both_directions",
    }
    topology["sha256"] = canonical_sha256(topology["edge_index"])
    sensitivity_config = config["targeted_scenarios"][
        "sensitivity_targeted_outliers_15pct"
    ]
    sensitivity_records = feeder.nominal_bus_voltage_influence(
        float(sensitivity_config["relative_load_step"])
    )
    sensitivity = {
        "definition": sensitivity_config["sensitivity_definition"],
        "relative_load_step": float(sensitivity_config["relative_load_step"]),
        "records": sensitivity_records,
    }
    sensitivity["sha256"] = canonical_sha256(sensitivity_records)
    sequence_length = int(data_config["sequence_length"])
    shard_size = int(data_config["cache"]["trajectories_per_shard"])
    shard_records: dict[str, list[dict[str, Any]]] = {name: [] for name in splits}
    scada_statistics = ChannelStatistics(width=3)
    pmu_statistics = ChannelStatistics(width=2)

    append_event(
        progress_path,
        {
            "event": "cache_start",
            "admissible_for_formal_study": admissible,
            "split_sizes": {name: len(seeds) for name, seeds in splits.items()},
        },
    )
    for split, seeds in splits.items():
        buffer: list[dict[str, Any]] = []
        shard_index = 0
        for index, seed in enumerate(seeds, start=1):
            sample = feeder.generate_sequence(sequence_length, int(seed))
            buffer.append(sample)
            append_event(
                progress_path,
                {
                    "event": "trajectory_generated",
                    "split": split,
                    "index": index,
                    "count": len(seeds),
                    "trajectory_seed": int(seed),
                },
            )
            if len(buffer) == shard_size or index == len(seeds):
                payload = stack_samples(buffer)
                if split == "train":
                    # Fit on the exact float32 arrays that downstream jobs read.
                    scada_statistics.update(payload["scada"], payload["scada_mask"])
                    pmu_statistics.update(payload["pmu"], payload["pmu_mask"])
                relative_path = Path(split) / f"shard-{shard_index:04d}.npz"
                shard_path = output_dir / relative_path
                write_npz_atomic(shard_path, payload)
                record = {
                    "relative_path": relative_path.as_posix(),
                    "trajectory_count": len(buffer),
                    "trajectory_seeds": payload["trajectory_seed"].tolist(),
                    "array_shapes": {
                        key: list(payload[key].shape) for key in ARRAY_KEYS
                    },
                    "array_dtypes": {
                        key: str(payload[key].dtype) for key in ARRAY_KEYS
                    },
                    "sha256": sha256_file(shard_path),
                }
                shard_records[split].append(record)
                append_event(
                    progress_path,
                    {
                        "event": "shard_written",
                        "split": split,
                        "shard_index": shard_index,
                        "trajectory_count": len(buffer),
                        "sha256": record["sha256"],
                    },
                )
                buffer = []
                shard_index += 1

    scada_mean, scada_scale, scada_count = scada_statistics.finalize()
    pmu_mean, pmu_scale, pmu_count = pmu_statistics.finalize()
    normalizer = {
        "fit_split": "train",
        "scada_mean": scada_mean,
        "scada_scale": scada_scale,
        "scada_available_count": scada_count,
        "pmu_mean": pmu_mean,
        "pmu_scale": pmu_scale,
        "pmu_available_count": pmu_count,
    }
    normalizer_path = output_dir / "normalizer.json"
    write_json(normalizer_path, normalizer)

    run_manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        command=sys.argv,
        config=config,
        data_hashes=data_hashes,
        splits=splits,
        mamba_backend="not_loaded_data_generation_only",
    )
    manifest = {
        **run_manifest,
        "cache_schema_version": 1,
        "status": "COMPLETED",
        "admissible_for_formal_study": admissible,
        "smoke_limit_per_split": smoke_limit_per_split,
        "full_predeclared_split_sha256": canonical_sha256(full_splits),
        "cached_split_sha256": canonical_sha256(splits),
        "num_opendss_buses": feeder.num_buses,
        "pmu_count": len(feeder.pmu_bus_indices),
        "pmu_bus_indices": [int(index) for index in feeder.pmu_bus_indices],
        "pmu_coverage": len(feeder.pmu_bus_indices) / feeder.num_buses,
        "bus_mapping": feeder.mapping_records(),
        "topology": topology,
        "nominal_voltage_influence": sensitivity,
        "resolved_config": {
            "relative_path": "config.resolved.json",
            "sha256": sha256_file(resolved_config_path),
        },
        "shards": shard_records,
        "normalizer": {
            "relative_path": "normalizer.json",
            "sha256": sha256_file(normalizer_path),
            "fit_split": "train",
        },
    }
    write_json(output_dir / "cache_manifest.json", manifest)
    verify_cache(output_dir, require_formal_admissible=admissible)
    append_event(
        progress_path,
        {
            "event": "cache_complete",
            "admissible_for_formal_study": admissible,
            "cached_split_sha256": manifest["cached_split_sha256"],
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-limit-per-split", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-formal-admissible", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_cache(args.output, args.require_formal_admissible)
    else:
        if args.config is None:
            parser.error("--config is required unless --verify-only is used")
        manifest = generate_cache(
            args.config, args.output, args.smoke_limit_per_split
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "admissible_for_formal_study": manifest[
                    "admissible_for_formal_study"
                ],
                "cached_split_sha256": manifest["cached_split_sha256"],
                "num_opendss_buses": manifest["num_opendss_buses"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
