"""Trajectory access, observation scenarios, and evaluation metrics."""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dsse.datasets import ARRAY_KEYS, read_shard
from dsse.feeder import apply_normalizer, corrupt_sample
from dsse.reproducibility import canonical_sha256, portable_config, sha256_file


NON_SEALED_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class SealedTestAccess:
    """Recorded acknowledgment required before any sealed arrays are loaded."""

    study_id: str
    acknowledgment: str = "I_ACKNOWLEDGE_THIS_IS_THE_FIRST_SEALED_TEST_ACCESS"


class TrajectoryCache:
    """Read selected immutable cache splits without implicitly opening the test set."""

    def __init__(
        self, cache_dir: Path, *, allow_inadmissible_smoke: bool = False
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest_path = self.cache_dir / "cache_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "COMPLETED":
            raise ValueError("Formal cache is incomplete")
        self.formal_admissible = bool(
            self.manifest.get("admissible_for_formal_study", False)
        )
        if not self.formal_admissible and not allow_inadmissible_smoke:
            raise ValueError(
                "Inadmissible smoke cache requires --allow-inadmissible-smoke"
            )
        self.config = self.manifest["config"]
        if canonical_sha256(self.config) != self.manifest["config_sha256"]:
            raise RuntimeError("Cache configuration hash mismatch")
        self._verify_metadata()
        normalizer_path = self.cache_dir / self.manifest["normalizer"]["relative_path"]
        if sha256_file(normalizer_path) != self.manifest["normalizer"]["sha256"]:
            raise RuntimeError("Cache normalizer hash mismatch")
        self.normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
        if self.normalizer.get("fit_split") != "train":
            raise RuntimeError("Cache normalizer is not training-only")
        self.config = portable_config(self.config)

    def _verify_metadata(self) -> None:
        resolved = self.cache_dir / self.manifest["resolved_config"]["relative_path"]
        if sha256_file(resolved) != self.manifest["resolved_config"]["sha256"]:
            raise RuntimeError("Resolved cache configuration hash mismatch")
        if json.loads(resolved.read_text(encoding="utf-8")) != self.manifest["config"]:
            raise RuntimeError("Resolved cache configuration content mismatch")
        topology = self.manifest["topology"]
        if canonical_sha256(topology["edge_index"]) != topology["sha256"]:
            raise RuntimeError("Cached topology hash mismatch")
        sensitivity = self.manifest["nominal_voltage_influence"]
        if canonical_sha256(sensitivity["records"]) != sensitivity["sha256"]:
            raise RuntimeError("Cached nominal voltage influence hash mismatch")
        if canonical_sha256(self.manifest["split_seeds"]) != self.manifest["split_sha256"]:
            raise RuntimeError("Cached split metadata hash mismatch")

    @property
    def study_id(self) -> str:
        return str(self.config["study_id"])

    @property
    def num_buses(self) -> int:
        return int(self.manifest["num_opendss_buses"])

    @property
    def pmu_count(self) -> int:
        return int(self.manifest["pmu_count"])

    @property
    def pmu_bus_indices(self) -> list[int]:
        return [int(index) for index in self.manifest["pmu_bus_indices"]]

    @property
    def edge_index(self) -> np.ndarray:
        return np.asarray(self.manifest["topology"]["edge_index"], dtype=np.int64)

    def _authorize_split(
        self, split: str, sealed_access: SealedTestAccess | None
    ) -> None:
        if split in NON_SEALED_SPLITS:
            return
        if split != "sealed_test":
            raise ValueError(f"Unknown formal cache split: {split}")
        if sealed_access is None:
            raise PermissionError("Sealed test arrays require explicit first-access unlock")
        if sealed_access.study_id != self.study_id:
            raise PermissionError("Sealed test unlock study ID does not match the cache")
        expected = "I_ACKNOWLEDGE_THIS_IS_THE_FIRST_SEALED_TEST_ACCESS"
        if sealed_access.acknowledgment != expected:
            raise PermissionError("Sealed test acknowledgment text is invalid")

    def load_split(
        self, split: str, *, sealed_access: SealedTestAccess | None = None
    ) -> dict[str, np.ndarray]:
        self._authorize_split(split, sealed_access)
        records = self.manifest["shards"][split]
        pieces: dict[str, list[np.ndarray]] = {
            key: [] for key in (*ARRAY_KEYS, "trajectory_seed")
        }
        observed_seeds: list[int] = []
        for record in records:
            path = self.cache_dir / record["relative_path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Formal cache shard hash mismatch: {path}")
            payload = read_shard(path)
            seeds = payload["trajectory_seed"].astype(np.int64).tolist()
            if seeds != [int(seed) for seed in record["trajectory_seeds"]]:
                raise RuntimeError(f"Formal cache shard seed mismatch: {path}")
            observed_seeds.extend(seeds)
            for key in pieces:
                pieces[key].append(payload[key])
        expected_seeds = [int(seed) for seed in self.manifest["split_seeds"][split]]
        if observed_seeds != expected_seeds:
            raise RuntimeError(f"Formal cache split ordering mismatch: {split}")
        return {key: np.concatenate(values, axis=0) for key, values in pieces.items()}


def sample_at(arrays: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        **{key: arrays[key][index] for key in ARRAY_KEYS},
        "trajectory_seed": int(arrays["trajectory_seed"][index]),
    }


def tensorize_last_state(
    sample: dict[str, Any], normalizer: dict[str, Any]
) -> dict[str, torch.Tensor]:
    normalized = apply_normalizer(sample, normalizer)
    sequence_length = normalized["scada"].shape[0]
    return {
        "scada": torch.as_tensor(
            normalized["scada"].reshape(sequence_length, -1), dtype=torch.float32
        ),
        "scada_mask": torch.as_tensor(
            normalized["scada_mask"].reshape(sequence_length, -1), dtype=torch.float32
        ),
        "pmu": torch.as_tensor(
            normalized["pmu"].reshape(sequence_length, -1), dtype=torch.float32
        ),
        "pmu_mask": torch.as_tensor(
            normalized["pmu_mask"].reshape(sequence_length, -1), dtype=torch.float32
        ),
        "v_true": torch.as_tensor(normalized["v_true"][-1], dtype=torch.float32),
        "angle_true": torch.as_tensor(
            normalized["angle_true"][-1], dtype=torch.float32
        ),
        "trajectory_seed": torch.as_tensor(
            int(normalized["trajectory_seed"]), dtype=torch.int64
        ),
    }


def scenario_seed(trajectory_seed: int, scenario_name: str) -> int:
    return int(
        (int(trajectory_seed) + zlib.crc32(scenario_name.encode("utf-8")))
        % (2**32)
    )


def training_corruption_seed(
    model_seed: int, epoch: int, trajectory_seed: int, scenario_name: str
) -> int:
    return int(
        (
            int(model_seed) * 1_000_003
            + int(epoch) * 10_007
            + int(trajectory_seed) * 101
            + zlib.crc32(scenario_name.encode("utf-8"))
        )
        % (2**32)
    )


def _assert_noise_contract(config: dict[str, Any]) -> None:
    noise = config["corruption_randomization"]["measurement_noise"]
    expected = {
        "scada_relative_sigma": 0.03,
        "scada_minimum_sigma_pu": 0.001,
        "pmu_voltage_sigma_pu": 0.002,
        "pmu_angle_sigma_rad": 0.0005,
    }
    if noise != expected:
        raise ValueError("Formal noise config drifted from the checked corruption code")


def _bus_metadata(manifest: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    names = [str(row["opendss_bus"]) for row in manifest["bus_mapping"]]
    return names, {name: index for index, name in enumerate(names)}


def regional_subtree_indices(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> list[int]:
    names, name_to_index = _bus_metadata(manifest)
    root = name_to_index[str(scenario["root_bus_name"])]
    edge_index = np.asarray(manifest["topology"]["edge_index"], dtype=np.int64)
    neighbors = {index: set() for index in range(len(names))}
    for left, right in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        neighbors[left].add(right)
        neighbors[right].add(left)

    parent: dict[int, int | None] = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(neighbors[node], key=lambda index: names[index]):
            if neighbor not in parent:
                parent[neighbor] = node
                order.append(neighbor)
    if len(order) != len(names):
        raise RuntimeError("Cached feeder topology is disconnected")

    children = {index: [] for index in range(len(names))}
    for node, parent_node in parent.items():
        if parent_node is not None:
            children[parent_node].append(node)
    descendants: dict[int, set[int]] = {}
    for node in reversed(order):
        subtree = {node}
        for child in children[node]:
            subtree.update(descendants[child])
        descendants[node] = subtree
    rate = float(scenario["rate"])
    candidates = [node for node in order if node != root]
    selected = min(
        candidates,
        key=lambda node: (
            abs(len(descendants[node]) / len(names) - rate),
            -len(descendants[node]),
            names[node],
        ),
    )
    return sorted(descendants[selected])


def sensitivity_target_bus_indices(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> list[int]:
    records = manifest["nominal_voltage_influence"]["records"]
    score_key = str(scenario["sensitivity_definition"])
    ranked = sorted(
        records,
        key=lambda row: (-float(row[score_key]), str(row["opendss_bus"])),
    )
    count = max(1, int(math.floor(len(records) * float(scenario["rate"]) + 0.5)))
    return sorted(int(row["model_index"]) for row in ranked[:count])


def _scada_sigma(sample: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    noise = config["corruption_randomization"]["measurement_noise"]
    sigma = np.maximum(
        np.abs(sample["scada"]) * float(noise["scada_relative_sigma"]),
        float(noise["scada_minimum_sigma_pu"]),
    )
    sigma[..., 2] = 0.005
    return sigma


def corrupt_formal_sample(
    sample: dict[str, Any],
    scenario_name: str,
    scenario: dict[str, Any],
    rng: np.random.Generator,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one frozen scenario without consulting any state labels."""

    _assert_noise_contract(config)
    scenario_type = str(scenario["type"])
    if scenario_type in {"normal", "missing", "outliers"}:
        return corrupt_sample(sample, scenario, rng)

    output, _ = corrupt_sample(sample, {"type": "normal"}, rng)
    if scenario_type == "regional_missing":
        bus_indices = regional_subtree_indices(manifest, scenario)
        available = output["scada_mask"][:, bus_indices, :] > 0.5
        effective = int(available.sum())
        target = output["scada"][:, bus_indices, :].copy()
        target_mask = output["scada_mask"][:, bus_indices, :].copy()
        target[available] = 0.0
        target_mask[available] = 0.0
        output["scada"][:, bus_indices, :] = target
        output["scada_mask"][:, bus_indices, :] = target_mask
        eligible = int((sample["scada_mask"] > 0.5).sum())
        diagnostics = {
            "type": scenario_type,
            "scenario": scenario_name,
            "selected_bus_indices": bus_indices,
            "selected_bus_names": [
                manifest["bus_mapping"][index]["opendss_bus"] for index in bus_indices
            ],
            "eligible_scada_count": eligible,
            "requested_corruption_count": int(math.floor(eligible * float(scenario["rate"]))),
            "effective_corruption_count": effective,
            "effective_rate": float(effective / max(eligible, 1)),
        }
        return output, diagnostics

    if scenario_type == "sensitivity_targeted_outliers":
        bus_indices = sensitivity_target_bus_indices(manifest, scenario)
        available = output["scada_mask"][:, bus_indices, :] > 0.5
        sigma = _scada_sigma(sample, config)[:, bus_indices, :]
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=int(available.sum()))
        target = output["scada"][:, bus_indices, :].copy()
        target[available] += (
            float(scenario["sigma_multiplier"]) * sigma[available] * signs
        )
        output["scada"][:, bus_indices, :] = target
        eligible = int((sample["scada_mask"] > 0.5).sum())
        effective = int(available.sum())
        diagnostics = {
            "type": scenario_type,
            "scenario": scenario_name,
            "selected_bus_indices": bus_indices,
            "selected_bus_names": [
                manifest["bus_mapping"][index]["opendss_bus"] for index in bus_indices
            ],
            "eligible_scada_count": eligible,
            "requested_corruption_count": int(math.floor(eligible * float(scenario["rate"]))),
            "effective_corruption_count": effective,
            "effective_rate": float(effective / max(eligible, 1)),
        }
        return output, diagnostics

    if scenario_type == "pmu_missing":
        configured_names = [str(name) for name in config["data"]["pmu_bus_names"]]
        dropped_names = [str(name) for name in scenario["pmu_bus_names"]]
        dropped = [configured_names.index(name) for name in dropped_names]
        available = output["pmu_mask"][:, dropped, :] > 0.5
        effective = int(available.sum())
        target = output["pmu"][:, dropped, :].copy()
        target_mask = output["pmu_mask"][:, dropped, :].copy()
        target[available] = 0.0
        target_mask[available] = 0.0
        output["pmu"][:, dropped, :] = target
        output["pmu_mask"][:, dropped, :] = target_mask
        eligible = int((sample["pmu_mask"] > 0.5).sum())
        diagnostics = {
            "type": scenario_type,
            "scenario": scenario_name,
            "selected_pmu_indices": dropped,
            "selected_pmu_bus_names": dropped_names,
            "eligible_pmu_count": eligible,
            "requested_corruption_count": int(math.floor(eligible * float(scenario["rate"]))),
            "effective_corruption_count": effective,
            "effective_rate": float(effective / max(eligible, 1)),
        }
        return output, diagnostics
    raise ValueError(f"Unknown formal scenario type: {scenario_type}")


def all_formal_scenarios(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = {**config["primary_scenarios"], **config["targeted_scenarios"]}
    if len(scenarios) != len(config["primary_scenarios"]) + len(
        config["targeted_scenarios"]
    ):
        raise ValueError("Duplicate formal scenario name")
    return scenarios


def trajectory_metrics(
    voltage_prediction: np.ndarray,
    angle_prediction: np.ndarray,
    voltage_truth: np.ndarray,
    angle_truth: np.ndarray,
) -> dict[str, float]:
    voltage_error = np.asarray(voltage_prediction) - np.asarray(voltage_truth)
    raw_angle_error = np.asarray(angle_prediction) - np.asarray(angle_truth)
    angle_error = np.angle(np.exp(1j * raw_angle_error))
    return {
        "voltage_rmse": float(np.sqrt(np.mean(voltage_error**2))),
        "angle_circular_rmse": float(np.sqrt(np.mean(angle_error**2))),
        "voltage_mae": float(np.mean(np.abs(voltage_error))),
        "angle_circular_mae": float(np.mean(np.abs(angle_error))),
        "voltage_absolute_error_p95": float(
            np.percentile(np.abs(voltage_error), 95)
        ),
        "angle_absolute_error_p95": float(np.percentile(np.abs(angle_error), 95)),
        "voltage_error_over_0_01pu_rate": float(
            np.mean(np.abs(voltage_error) > 0.01)
        ),
        "angle_error_over_0_01rad_rate": float(
            np.mean(np.abs(angle_error) > 0.01)
        ),
    }
