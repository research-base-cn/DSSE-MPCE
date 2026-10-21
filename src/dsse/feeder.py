"""IEEE 123-node feeder simulation and measurement generation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import opendssdirect as dss
except ImportError as exc:  # pragma: no cover - exercised by fail-closed smoke test
    raise RuntimeError(
        "opendssdirect.py is required for feeder simulation"
    ) from exc


DEFAULT_PMU_BUS_NAMES = (
    "150",
    "13",
    "18",
    "25",
    "35",
    "47",
    "60",
    "67",
    "76",
    "89",
    "101",
    "114",
)


def _base_bus(name: str) -> str:
    return str(name).split(".")[0].strip().lower()


def _reference_aligned_voltage(pu_voltage: Iterable[float], nodes: Iterable[int]) -> complex:
    """Rotate phase phasors to a common phase-A reference before averaging."""

    values = list(pu_voltage)
    bus_nodes = list(nodes)
    aligned = []
    for index, node in enumerate(bus_nodes):
        if node not in (1, 2, 3) or 2 * index + 1 >= len(values):
            continue
        phasor = complex(values[2 * index], values[2 * index + 1])
        rotation = np.exp(1j * 2.0 * np.pi * (node - 1) / 3.0)
        aligned.append(phasor * rotation)
    if not aligned:
        return complex(np.nan, np.nan)
    return sum(aligned) / len(aligned)


class IEEE123BusFeeder:
    """Bus-level adapter for the vendored IEEE 123-node OpenDSS case."""

    def __init__(
        self,
        dss_dir: Path,
        pmu_bus_names: Iterable[str] = DEFAULT_PMU_BUS_NAMES,
        s_base_mva: float = 1.0,
    ) -> None:
        self.dss_dir = Path(dss_dir).resolve()
        self.master_path = self.dss_dir / "IEEE123Master.dss"
        if not self.master_path.is_file():
            raise FileNotFoundError(f"Missing OpenDSS master file: {self.master_path}")
        self.s_base_mva = float(s_base_mva)
        self._compile()
        self._capture_static_case(tuple(str(name).lower() for name in pmu_bus_names))

    def _compile(self) -> None:
        original = Path.cwd()
        os.chdir(self.dss_dir)
        try:
            dss.Basic.ClearAll()
            dss.Text.Command("Redirect IEEE123Master.dss")
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                raise RuntimeError("The vendored IEEE 123-node base power flow did not converge")
        finally:
            os.chdir(original)

    def _capture_static_case(self, requested_pmus: tuple[str, ...]) -> None:
        self.bus_names = [_base_bus(name) for name in dss.Circuit.AllBusNames()]
        if len(self.bus_names) != len(set(self.bus_names)):
            raise RuntimeError("OpenDSS returned duplicate normalized bus names")
        self.bus_to_index = {name: index for index, name in enumerate(self.bus_names)}
        missing_pmus = sorted(set(requested_pmus).difference(self.bus_to_index))
        if missing_pmus:
            raise ValueError(f"Configured PMU buses are absent from the feeder: {missing_pmus}")
        self.pmu_bus_names = list(requested_pmus)
        self.pmu_bus_indices = [self.bus_to_index[name] for name in requested_pmus]

        edges: set[tuple[int, int]] = set()

        def add_terminal_pair(bus_names: Iterable[str]) -> None:
            terminals = [_base_bus(name) for name in bus_names]
            terminals = [name for name in terminals if name in self.bus_to_index]
            for left, right in zip(terminals, terminals[1:]):
                if left == right:
                    continue
                i, j = self.bus_to_index[left], self.bus_to_index[right]
                edges.add((i, j))
                edges.add((j, i))

        for name in dss.Lines.AllNames():
            dss.Lines.Name(name)
            dss.Circuit.SetActiveElement(f"Line.{name}")
            add_terminal_pair(dss.CktElement.BusNames())
        for name in dss.Transformers.AllNames():
            dss.Transformers.Name(name)
            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            add_terminal_pair(dss.CktElement.BusNames())
        if not edges:
            raise RuntimeError("No graph edges were extracted from the OpenDSS case")
        self.edge_index = np.asarray(sorted(edges), dtype=np.int64).T

        self.loads = []
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            dss.Circuit.SetActiveElement(f"Load.{name}")
            terminals = dss.CktElement.BusNames()
            if not terminals:
                continue
            bus = _base_bus(terminals[0])
            if bus not in self.bus_to_index:
                continue
            self.loads.append(
                {
                    "name": str(name),
                    "bus": bus,
                    "bus_index": self.bus_to_index[bus],
                    "kw": float(dss.Loads.kW()),
                    "kvar": float(dss.Loads.kvar()),
                }
            )
        if not self.loads:
            raise RuntimeError("No feeder loads were extracted from OpenDSS")

    @property
    def num_buses(self) -> int:
        return len(self.bus_names)

    def mapping_records(self) -> list[dict[str, Any]]:
        pmu_set = set(self.pmu_bus_names)
        return [
            {
                "model_index": index,
                "opendss_bus": name,
                "is_pmu": name in pmu_set,
            }
            for index, name in enumerate(self.bus_names)
        ]

    def nominal_bus_voltage_influence(
        self, relative_load_step: float = 0.01
    ) -> list[dict[str, Any]]:
        """Rank buses by a training-independent nominal load perturbation.

        The score is empirical context for a targeted stress test, not a
        measurement-Jacobian criticality index. Every load at one bus is scaled
        together, and the mean absolute voltage-magnitude response is divided by
        the relative load step.
        """

        step = float(relative_load_step)
        if not 0.0 < step <= 0.1:
            raise ValueError("relative_load_step must be in (0, 0.1]")
        self._compile()
        nominal_scales = {load["name"]: 1.0 for load in self.loads}
        nominal_voltage = self._solve_snapshot(nominal_scales)["v_true"]
        loads_by_bus: dict[str, list[str]] = {name: [] for name in self.bus_names}
        for load in self.loads:
            loads_by_bus[load["bus"]].append(load["name"])

        records = []
        for bus_name in self.bus_names:
            bus_loads = sorted(loads_by_bus[bus_name])
            if bus_loads:
                perturbed_scales = dict(nominal_scales)
                for load_name in bus_loads:
                    perturbed_scales[load_name] = 1.0 + step
                perturbed_voltage = self._solve_snapshot(perturbed_scales)["v_true"]
                absolute_change = np.abs(perturbed_voltage - nominal_voltage)
                mean_influence = float(absolute_change.mean() / step)
                maximum_influence = float(absolute_change.max() / step)
            else:
                mean_influence = 0.0
                maximum_influence = 0.0
            records.append(
                {
                    "opendss_bus": bus_name,
                    "model_index": int(self.bus_to_index[bus_name]),
                    "load_names": bus_loads,
                    "mean_abs_voltage_change_per_relative_load_step": mean_influence,
                    "max_abs_voltage_change_per_relative_load_step": maximum_influence,
                }
            )
        self._solve_snapshot(nominal_scales)
        return records

    def _solve_snapshot(self, load_scales: dict[str, float]) -> dict[str, np.ndarray]:
        for load in self.loads:
            scale = float(load_scales[load["name"]])
            dss.Loads.Name(load["name"])
            dss.Loads.kW(load["kw"] * scale)
            dss.Loads.kvar(load["kvar"] * scale)
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS failed to converge for a generated operating point")

        voltage = np.empty(self.num_buses, dtype=np.complex128)
        for index, bus_name in enumerate(self.bus_names):
            dss.Circuit.SetActiveBus(bus_name)
            voltage[index] = _reference_aligned_voltage(
                dss.Bus.PuVoltage(), dss.Bus.Nodes()
            )
        if not np.isfinite(voltage.real).all() or not np.isfinite(voltage.imag).all():
            raise RuntimeError("OpenDSS returned a bus without an energized phase voltage")

        p_load = np.zeros(self.num_buses, dtype=np.float64)
        q_load = np.zeros(self.num_buses, dtype=np.float64)
        for load in self.loads:
            scale = float(load_scales[load["name"]])
            p_load[load["bus_index"]] += load["kw"] * scale / (1000.0 * self.s_base_mva)
            q_load[load["bus_index"]] += load["kvar"] * scale / (1000.0 * self.s_base_mva)

        scada = np.stack(
            [p_load, q_load, np.zeros(self.num_buses, dtype=np.float64)], axis=-1
        )
        scada_mask = np.ones_like(scada)
        scada_mask[:, 2] = 0.0

        v_mag = np.abs(voltage)
        v_angle = np.angle(voltage)
        pmu = np.stack(
            [v_mag[self.pmu_bus_indices], v_angle[self.pmu_bus_indices]], axis=-1
        )
        return {
            "scada": scada,
            "scada_mask": scada_mask,
            "pmu": pmu,
            "pmu_mask": np.ones_like(pmu),
            "v_true": v_mag,
            "angle_true": v_angle,
        }

    def generate_sequence(self, seq_len: int, trajectory_seed: int) -> dict[str, Any]:
        """Generate a real temporal sequence from one independently seeded trajectory."""

        self._compile()
        rng = np.random.default_rng(int(trajectory_seed))
        base_level = float(rng.uniform(0.78, 1.22))
        common = float(rng.normal(0.0, 0.03))
        local = {load["name"]: float(rng.normal(0.0, 0.04)) for load in self.loads}
        snapshots = []
        for _ in range(int(seq_len)):
            common = 0.90 * common + float(rng.normal(0.0, 0.035))
            load_scales = {}
            for load in self.loads:
                name = load["name"]
                local[name] = 0.82 * local[name] + float(rng.normal(0.0, 0.045))
                load_scales[name] = float(
                    np.clip(base_level * (1.0 + common + local[name]), 0.55, 1.45)
                )
            snapshots.append(self._solve_snapshot(load_scales))

        sample = {
            key: np.stack([snapshot[key] for snapshot in snapshots], axis=0)
            for key in ("scada", "scada_mask", "pmu", "pmu_mask", "v_true", "angle_true")
        }
        sample.update(
            {
                "trajectory_seed": int(trajectory_seed),
                "bus_names": list(self.bus_names),
                "pmu_bus_names": list(self.pmu_bus_names),
                "pmu_bus_indices": list(self.pmu_bus_indices),
            }
        )
        return sample


def clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in sample.items()
    }


def fit_normalizer(samples: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot fit a normalizer on an empty sample set")

    def statistics(value_key: str, mask_key: str, width: int) -> tuple[list[float], list[float]]:
        means, scales = [], []
        for feature in range(width):
            values = np.concatenate(
                [
                    sample[value_key][..., feature][sample[mask_key][..., feature] > 0.5]
                    for sample in samples
                ]
            )
            if values.size == 0:
                means.append(0.0)
                scales.append(1.0)
            else:
                means.append(float(values.mean()))
                scales.append(float(max(values.std(), 1e-6)))
        return means, scales

    scada_mean, scada_scale = statistics("scada", "scada_mask", 3)
    pmu_mean, pmu_scale = statistics("pmu", "pmu_mask", 2)
    return {
        "scada_mean": scada_mean,
        "scada_scale": scada_scale,
        "pmu_mean": pmu_mean,
        "pmu_scale": pmu_scale,
    }


def apply_normalizer(sample: dict[str, Any], normalizer: dict[str, list[float]]) -> dict[str, Any]:
    output = clone_sample(sample)
    for value_key, mask_key, mean_key, scale_key in (
        ("scada", "scada_mask", "scada_mean", "scada_scale"),
        ("pmu", "pmu_mask", "pmu_mean", "pmu_scale"),
    ):
        mean = np.asarray(normalizer[mean_key], dtype=np.float64)
        scale = np.asarray(normalizer[scale_key], dtype=np.float64)
        mask = output[mask_key] > 0.5
        normalized = (output[value_key] - mean) / scale
        output[value_key] = np.where(mask, normalized, 0.0).astype(np.float32)
        output[mask_key] = output[mask_key].astype(np.float32)
    return output


def _measurement_sigma_scada(scada: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.abs(scada) * 0.03, 0.001)
    sigma[..., 2] = 0.005
    return sigma


def _measurement_sigma_pmu(pmu: np.ndarray) -> np.ndarray:
    sigma = np.empty_like(pmu)
    sigma[..., 0] = 0.002
    sigma[..., 1] = 0.0005
    return sigma


def corrupt_sample(
    sample: dict[str, Any], scenario: dict[str, Any], rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a scenario only to genuinely available measurements."""

    output = clone_sample(sample)
    scada_sigma = _measurement_sigma_scada(output["scada"])
    pmu_sigma = _measurement_sigma_pmu(output["pmu"])
    scada_available = output["scada_mask"] > 0.5
    pmu_available = output["pmu_mask"] > 0.5
    output["scada"][scada_available] += rng.normal(
        0.0, scada_sigma[scada_available]
    )
    output["pmu"][pmu_available] += rng.normal(0.0, pmu_sigma[pmu_available])

    scenario_type = str(scenario["type"])
    eligible = np.flatnonzero(scada_available.reshape(-1))
    requested = 0
    changed = 0
    if scenario_type in {"missing", "outliers"}:
        rate = float(scenario.get("rate", 0.0))
        requested = int(math.floor(len(eligible) * rate))
        chosen = (
            rng.choice(eligible, size=requested, replace=False)
            if requested > 0
            else np.asarray([], dtype=np.int64)
        )
        if scenario_type == "missing":
            flat_value = output["scada"].reshape(-1)
            flat_mask = output["scada_mask"].reshape(-1)
            flat_value[chosen] = 0.0
            flat_mask[chosen] = 0.0
            changed = int(np.count_nonzero(flat_mask[chosen] == 0.0))
        else:
            multiplier = float(scenario.get("sigma_multiplier", 10.0))
            flat_value = output["scada"].reshape(-1)
            flat_sigma = scada_sigma.reshape(-1)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=requested)
            flat_value[chosen] += multiplier * flat_sigma[chosen] * signs
            changed = int(chosen.size)
    elif scenario_type != "normal":
        raise ValueError(f"Unknown corruption scenario type: {scenario_type}")

    diagnostics = {
        "type": scenario_type,
        "eligible_scada_count": int(eligible.size),
        "requested_corruption_count": int(requested),
        "effective_corruption_count": int(changed),
        "effective_rate": float(changed / max(eligible.size, 1)),
    }
    return output, diagnostics
