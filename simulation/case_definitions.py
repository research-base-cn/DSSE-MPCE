"""Training and evaluation observation scenarios."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from simulation.corruption_models import make_cases as _make_gaussian_cases


SENSOR_KEYS = ("scada", "scada_mask", "pmu", "pmu_mask")
MAIN_KEYS = ("S0", "S1", "S2", "S3", "S4", "S5", "S7", "A1", "A2")
TRAINING_VIEW_KEYS = {
    "normal": "S0", "drop50": "S5", "pmu_out30": "S3", "joint_out30": "S4",
}
STREAMS = {
    "pq_outliers": "r11_pq_signed_outliers",
    "pmu_outliers": "r11_pmu_signed_outliers",
    "pq_missing": "r11_pq_missing",
}
GENERATOR_CONTRACT = {
    "gaussian": "frozen R9 make_cases(sigma=.03,pmu_mode='full')",
    "pq_eligibility": "all available SCADA channels0:2, including zero-load slots",
    "pmu_eligibility": "all available PMU voltage/angle channel entries",
    "index_order": "C-order actual SCADA[30,132,3] and PMU[30,12,2]",
    "outliers": "one permutation and sign vector per modality; prefix floor(rate*N); signed10sigma",
    "missing": "separate PQ permutation; prefix floor(rate*N); values and masks zero",
    "targets": "complete original sample retained byte-for-byte; never used for eligibility",
    "training": "S0/S5/S3/S4 using this same generator, complete30frames",
}


def sensor_digest(sample):
    """SHA256 of the four raw sensor arrays, using the existing R9 convention.

    Sorted key, numpy dtype string, tuple-shape string and C-contiguous bytes are
    appended without separators, as in R9 hash_arrays.  Targets are excluded.
    """
    digest = hashlib.sha256()
    for key in sorted(SENSOR_KEYS):
        value = np.ascontiguousarray(sample[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_count(arrays, count):
    if isinstance(count, (bool, np.bool_)) or int(count) != count:
        raise ValueError("count must be a positive integer")
    count = int(count)
    if not 0 < count <= len(arrays["trajectory_seed"]):
        raise ValueError("count is outside the caller-supplied arrays")
    return count


def _same_bits(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def _check_sample(sample):
    expected = {"scada": (30, 132, 3), "pmu": (30, 12, 2)}
    for key, shape in expected.items():
        if sample[key].shape != shape or sample[key + "_mask"].shape != shape:
            raise ValueError(f"Unexpected sensor/mask shape: {key}")
        for candidate in (key, key + "_mask"):
            if not np.issubdtype(sample[candidate].dtype, np.floating):
                raise ValueError(f"Floating raw sensor/mask required: {candidate}")
            if not np.isfinite(sample[candidate]).all():
                raise ValueError(f"Nonfinite raw sensor/mask: {candidate}")


def _known_dropped_slots(cache):
    all_names = [str(x) for x in cache.config["data"]["pmu_bus_names"]]
    names = [str(x) for x in cache.config["targeted_scenarios"]
             ["pmu_dropout_50pct"]["pmu_bus_names"]]
    if len(all_names) != 12 or len(set(all_names)) != 12:
        raise ValueError("The model requires twelve distinct configured PMU stations")
    if len(names) != 6 or len(set(names)) != 6 or not set(names) <= set(all_names):
        raise ValueError("Known PMU50 must declare six configured stations")
    return [all_names.index(x) for x in names]


def _layout_rows(layouts, cache):
    rows = layouts["heldout_layouts"] if isinstance(layouts, dict) else layouts
    if not isinstance(rows, (list, tuple)) or len(rows) != 8:
        raise ValueError("Exactly eight explicitly supplied PMU layouts required")
    names = [str(x) for x in cache.config["data"]["pmu_bus_names"]]
    if isinstance(layouts, dict) and [str(x) for x in layouts["pmu_bus_names"]] != names:
        raise ValueError("Layout PMU ordering differs from the actual cache")
    result = []
    for i, row in enumerate(rows):
        if row["id"] != f"L{i}":
            raise ValueError("Layout rows must have ordered ids L0 through L7")
        slots = row["missing_slot_indices"]
        if (len(slots) != 6 or len(set(slots)) != 6
                or any(isinstance(x, (bool, np.bool_)) or int(x) != x or not 0 <= x < 12
                       for x in slots)):
            raise ValueError(f"Invalid six-station layout: {row['id']}")
        slots = [int(x) for x in slots]
        if "missing_bus_names" in row and [names[x] for x in slots] != [str(x) for x in row["missing_bus_names"]]:
            raise ValueError(f"Layout bus-name mapping differs: {row['id']}")
        result.append((f"S6_L{i}", slots))
    masks = [frozenset(slots) for _, slots in result]
    if len(set(masks)) != 8:
        raise ValueError("Repeated PMU layout")
    if [sum(j in slots for slots in masks) for j in range(12)] != [4] * 12:
        raise ValueError("PMU layouts must remove every station exactly four times")
    for i in range(0, 8, 2):
        if masks[i] & masks[i + 1] or masks[i] | masks[i + 1] != frozenset(range(12)):
            raise ValueError("PMU layout rows must be paired complements")
    known = frozenset(_known_dropped_slots(cache))
    if known in masks or frozenset(range(12)) - known in masks:
        raise ValueError("Known PMU50 or its historical complement cannot be S6")
    # Whether these layouts appeared in any other historical metadata is checked
    # by the separate exposure check, not inferred from the supplied arrays.
    return result


def _outlier_prefixes(eligible, sigma, shared, trajectory_seed, stream, rates):
    indices = np.flatnonzero(eligible.ravel(order="C"))
    rng = np.random.default_rng(shared.scenario_seed(trajectory_seed, stream))
    order = rng.permutation(indices)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=order.size)
    flat_sigma = sigma.ravel(order="C")
    return {
        rate: (order[:int(np.floor(rate * indices.size))],
               10.0 * flat_sigma[order[:int(np.floor(rate * indices.size))]]
               * signs[:int(np.floor(rate * indices.size))])
        for rate in rates
    }


def _add_at_c(array, indices, delta):
    # Flatiter addresses C-order positions even if the caller's original array is
    # Fortran-contiguous.  The temporary preserves dtype's in-place += behavior;
    # reshape(-1)[indices] += can silently update a copy on F-order arrays.
    values = array.flat[indices].copy()
    values += delta
    array.flat[indices] = values


def _make_views(arrays, cache, shared, count, requested, layout_rows):
    count = _validate_count(arrays, count)
    known_slots = _known_dropped_slots(cache)
    base_cases = _make_gaussian_cases(arrays, cache, shared, .03, count=count, pmu_mode="full")
    output = {key: [] for key in requested}
    layout_lookup = dict(layout_rows)
    need_pq_out = any(key in requested for key in ("S1", "S4", "A1"))
    need_pmu_out = any(key in requested for key in ("S3", "S4"))
    need_missing = any(key in requested for key in ("S2", "A2"))

    for index, base in enumerate(base_cases):
        _check_sample(base)
        clean = shared.sample_at(arrays, index)
        seed = int(clean["trajectory_seed"])
        eligible_pq = np.zeros_like(base["scada_mask"], dtype=bool)
        eligible_pq[..., :2] = base["scada_mask"][..., :2] > .5
        eligible_pmu = base["pmu_mask"] > .5
        sigma_pq = np.maximum(np.abs(clean["scada"]) * .03, .001)
        sigma_pq[..., 2] = .005
        sigma_pmu = np.empty_like(clean["pmu"])
        sigma_pmu[..., 0] = .002
        sigma_pmu[..., 1] = .0005
        pq_out = (_outlier_prefixes(eligible_pq, sigma_pq, shared, seed,
                                   STREAMS["pq_outliers"], (.15, .30))
                  if need_pq_out else {})
        pmu_out = (_outlier_prefixes(eligible_pmu, sigma_pmu, shared, seed,
                                    STREAMS["pmu_outliers"], (.30,))
                   if need_pmu_out else {})
        missing_order = None
        if need_missing:
            positions = np.flatnonzero(eligible_pq.ravel(order="C"))
            missing_order = np.random.default_rng(
                shared.scenario_seed(seed, STREAMS["pq_missing"])).permutation(positions)

        one = {}
        for key in requested:
            item = base if key == "S0" else copy.deepcopy(base)
            if key in ("S1", "S4", "A1"):
                chosen, delta = pq_out[.15 if key == "A1" else .30]
                _add_at_c(item["scada"], chosen, delta)
            if key in ("S3", "S4"):
                chosen, delta = pmu_out[.30]
                _add_at_c(item["pmu"], chosen, delta)
            if key in ("S2", "A2"):
                rate = .60 if key == "S2" else .40
                chosen = missing_order[:int(np.floor(rate * len(missing_order)))]
                item["scada"].flat[chosen] = 0.0
                item["scada_mask"].flat[chosen] = 0.0
            if key == "S5" or key in layout_lookup:
                slots = known_slots if key == "S5" else layout_lookup[key]
                item["pmu"][:, slots, :] = 0.0
                item["pmu_mask"][:, slots, :] = 0.0
            if key == "S7":
                item["pmu"][-5:] = 0.0
                item["pmu_mask"][-5:] = 0.0
            for target in ("v_true", "angle_true"):
                if not _same_bits(item[target], clean[target]):
                    raise RuntimeError(f"Case generation changed target bits: {index}/{key}/{target}")
            if not _same_bits(item["scada"][..., 2], base["scada"][..., 2]):
                raise RuntimeError("P/Q corruption changed the third SCADA channel")
            if not _same_bits(item["scada_mask"][..., 2], base["scada_mask"][..., 2]):
                raise RuntimeError("P/Q corruption changed the third SCADA mask")
            if key in ("S0", "S3", "S5", "S7") or key in layout_lookup:
                if any(not _same_bits(item[k], base[k]) for k in ("scada", "scada_mask")):
                    raise RuntimeError("PMU-only view changed P/Q")
            one[key] = item
            output[key].append(item)

        if "S1" in one and "S4" in one:
            if any(not _same_bits(one["S1"][k], one["S4"][k]) for k in ("scada", "scada_mask")):
                raise RuntimeError("Single/joint PQ perturbations differ")
        if "S3" in one and "S4" in one:
            if any(not _same_bits(one["S3"][k], one["S4"][k]) for k in ("pmu", "pmu_mask")):
                raise RuntimeError("Single/joint PMU perturbations differ")
    return output


def make_case_map(arrays, cache, shared, count, layouts):
    """Return S0/S1/S2/S3/S4/S5/S7/A1/A2 and S6_L0...S6_L7.

    ``layouts`` is the parsed planned-layout dict (including heldout_layouts and
    pmu_bus_names), or its ordered eight row dictionaries.  It is not a path.
    S6 remains eight separate samples; aggregation is the analysis caller's job.
    """
    rows = _layout_rows(layouts, cache)
    return _make_views(arrays, cache, shared, count,
                       (*MAIN_KEYS, *(key for key, _ in rows)), rows)


def make_training_views(arrays, cache, shared, count):
    """Return only normal/drop50/pmu_out30/joint_out30, complete30frames.

    Applies the same S0/S5/S3/S4 generator as validation/confirmation.  The full
    window is corrupted before any count-KF caller extracts its final2frames.
    No S6, unused SCADA-missing, or extra outlier case copies are constructed.
    """
    selected = _make_views(arrays, cache, shared, count,
                           tuple(TRAINING_VIEW_KEYS.values()), ())
    return {name: selected[key] for name, key in TRAINING_VIEW_KEYS.items()}
