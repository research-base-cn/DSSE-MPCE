"""Noise, outlier and missing-measurement scenarios."""
import copy

import numpy as np


def make_cases(arrays, cache, shared, sigma, count=128, pmu_mode='full'):
    """Return complete corrupted sample dictionaries, retaining clean targets.

    ``sigma`` is the relative P/Q noise standard deviation, not a multiplier
    applied to the minimum floor. PMU noise and the SCADA floor remain fixed.
    Both layouts use the canonical ``normal`` random stream. The drop50 layout
    removes only the six configured PMU stations after the common noise draw.
    Predictors must consume only scada/scada_mask/pmu/pmu_mask, never targets.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be a finite positive relative standard deviation')
    if int(count) != count or not 0 < count <= len(arrays['trajectory_seed']):
        raise ValueError('count must be a positive integer within caller-supplied arrays')
    if pmu_mode not in ('full', 'drop50'):
        raise ValueError("pmu_mode must be 'full' or 'drop50'")

    dropped = []
    if pmu_mode == 'drop50':
        configured_names = [str(name) for name in cache.config['data']['pmu_bus_names']]
        dropped_names = [str(name) for name in cache.config['targeted_scenarios']['pmu_dropout_50pct']['pmu_bus_names']]
        if len(dropped_names) != 6 or len(set(dropped_names)) != 6:
            raise ValueError('The frozen drop50 declaration must name six unique PMU stations')
        dropped = [configured_names.index(name) for name in dropped_names]

    cases = []
    for index in range(int(count)):
        clean = shared.sample_at(arrays, index)
        output = copy.deepcopy(clean)
        rng = np.random.default_rng(shared.scenario_seed(clean['trajectory_seed'], 'normal'))

        # Match canonical operation order, precision, masks, and in-place +=.
        # Do not promote the cache's float32 values before calculating sigma.
        scada_sigma = np.maximum(np.abs(output['scada']) * sigma, 0.001)
        scada_sigma[..., 2] = 0.005
        pmu_sigma = np.empty_like(output['pmu'])
        pmu_sigma[..., 0] = 0.002
        pmu_sigma[..., 1] = 0.0005
        scada_available = output['scada_mask'] > 0.5
        pmu_available = output['pmu_mask'] > 0.5
        output['scada'][scada_available] += rng.normal(0.0, scada_sigma[scada_available])
        output['pmu'][pmu_available] += rng.normal(0.0, pmu_sigma[pmu_available])

        if sigma == 0.03:
            canonical, _ = shared.corrupt_formal_sample(
                clean, 'normal', {'type': 'normal'},
                np.random.default_rng(shared.scenario_seed(clean['trajectory_seed'], 'normal')),
                cache.config, cache.manifest,
            )
            if output.keys() != canonical.keys():
                raise RuntimeError('3% normal case keys differ from canonical output')
            for key, value in output.items():
                expected = canonical[key]
                if isinstance(value, np.ndarray):
                    if value.dtype != expected.dtype or value.shape != expected.shape or value.tobytes() != expected.tobytes():
                        raise RuntimeError(f'3% normal case is not canonical bitexact: {index}/{key}')
                elif value != expected:
                    raise RuntimeError(f'3% normal case metadata differs: {index}/{key}')

        if pmu_mode == 'drop50':
            # Preserve all P/Q arrays and the exact noise on overlapping PMUs.
            before_scada = output['scada'].copy()
            before_scada_mask = output['scada_mask'].copy()
            before_pmu = output['pmu'].copy()
            before_pmu_mask = output['pmu_mask'].copy()
            output['pmu'][:, dropped, :] = 0.0
            output['pmu_mask'][:, dropped, :] = 0.0
            kept = np.ones(output['pmu'].shape[1], dtype=bool)
            kept[dropped] = False
            if not (np.array_equal(before_scada, output['scada'])
                    and np.array_equal(before_scada_mask, output['scada_mask'])
                    and np.array_equal(before_pmu[:, kept], output['pmu'][:, kept])
                    and np.array_equal(before_pmu_mask[:, kept], output['pmu_mask'][:, kept])):
                raise RuntimeError('PMU-dropout construction altered retained observations')

        for key in ('v_true', 'angle_true'):
            if output[key].dtype != clean[key].dtype or output[key].tobytes() != clean[key].tobytes():
                raise RuntimeError(f'Noise construction altered clean targets: {index}/{key}')
        cases.append(output)
    return cases
