"""Export exact S6 layouts and bus/phase mappings from the supplied final inputs.

S6 is read from all saved masks, never guessed from station names. Phase data
come from a nominal OpenDSS reconstruction, with no generated trajectory or fit.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHYSICS_SHA = '4f7a72a20553000f661b9d7144354238b9f931ae3286db09b5eaed543ba13597'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def export_layouts(root):
    cache = read(root / 'data/training/cache_manifest.json')
    names = cache['config']['data']['pmu_bus_names']
    manifest_path = root / 'data/cases/bundle_manifest.json'
    manifest = read(manifest_path)
    records = {r['case_key']: r for r in manifest['cases']}
    rows = []
    for layout in range(8):
        key = 'S6_L' + str(layout)
        record = records[key]
        path = (manifest_path.parent / record['relative_path']).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or sha(path) != record['sha256']:
            raise ValueError('S6 input identity/path mismatch')
        with np.load(path, allow_pickle=False) as f:
            mask, pmu = f['pmu_mask'], f['pmu']
        if mask.shape != (512, 30, 12, 2) or set(np.unique(mask)) != {0., 1.}:
            raise ValueError('Unexpected S6 mask schema')
        availability = mask[0, 0, :, 0]
        if not np.array_equal(mask, np.broadcast_to(availability[None, None, :, None], mask.shape)):
            raise ValueError('S6 station mask is not constant across trajectories/time/channels')
        missing = np.flatnonzero(availability == 0).tolist()
        if len(missing) != 6 or not np.all(pmu[:, :, missing, :] == 0):
            raise ValueError('S6 must remove exactly six stations, values and masks')
        rows.append(dict(id='L'+str(layout), missing_slot_indices=missing,
            missing_bus_names=[names[j] for j in missing], availability=availability.astype(int).tolist(),
            source_case=key, case_sha256=record['sha256'], sensor_sha256=record['sensor_sha256']))
    if [sum(j in row['missing_slot_indices'] for row in rows) for j in range(12)] != [4]*12:
        raise ValueError('S6 station balance differs from the final design')
    for i in range(0, 8, 2):
        if set(rows[i]['missing_slot_indices']) | set(rows[i+1]['missing_slot_indices']) != set(range(12)):
            raise ValueError('S6 rows are not paired complements')
    return dict(schema_version=1, status='EXPORTED_FROM_ALL_FINAL_S6_MASKS', pmu_bus_names=names,
        heldout_layouts=rows, source_bundle_sha256=sha(manifest_path),
        full_grid_aggregation='8 layout-specific trajectory RMSEs averaged within the same trajectory before case means',
        physical128_assignment='trajectory_index modulo 8, one layout per trajectory, 16 indices per layout',
        case_mask_shape=[512, 30, 12, 2], all_512_trajectories_and_30_frames_checked=True)


def static_asset(root):
    source = root / 'src/physical/phase_physics_gn.py'
    if sha(source) != PHYSICS_SHA:
        raise ValueError('Physical source identity differs')
    spec = importlib.util.spec_from_file_location('static_phase_mapping', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = read(root / 'data/training/cache_manifest.json')
    pmus = cache['config']['data']['pmu_bus_names']
    dss = root / 'src/dsse/data/ieee123'
    with tempfile.TemporaryDirectory(prefix='static_bus_phase_mapping_') as work:
        model = module.PhasePhysicsModel(dss, Path(work)/'dss', pmus)
        rows = []
        for i, name in enumerate(model.bus_names):
            model.dss.Circuit.SetActiveBus(name)
            phases = [int(p) for p in model.dss.Bus.Nodes() if int(p) in (1, 2, 3)]
            nodes = []
            for phase in phases:
                key = name + '.' + str(phase)
                j = model.node_to_index[key]
                value = model.projection[i, j]
                nodes.append(dict(phase=phase, node_name=key, node_index=j,
                    base_volts=float(model.node_base_volts[j]),
                    projection_coefficient_real=float(value.real), projection_coefficient_imag=float(value.imag)))
            rows.append(dict(model_index=i, opendss_bus=name, phases=phases, nodes=nodes,
                pmu_slot=pmus.index(name) if name in pmus else None))
        return dict(source_sha256=PHYSICS_SHA,
            dss_asset_sha256={p.name: sha(p) for p in sorted(dss.iterdir()) if p.is_file()},
            bus_count=len(model.bus_names), node_count=len(model.node_names), node_order=model.node_names,
            dimension=model.dimension, loads=model.loads, regulators=model.regulators, buses=rows,
            asset_nominal_voltage=model.asset_nominal_state['v'].tolist(),
            asset_nominal_angle=model.asset_nominal_state['angle'].tolist())


def export_mapping(root, static):
    cache_path = root / 'data/training/cache_manifest.json'
    cache = read(cache_path)
    source = root / 'src/physical/phase_physics_gn.py'
    if static['source_sha256'] != sha(source) or sha(source) != PHYSICS_SHA:
        raise ValueError('Static mapping source mismatch')
    dss = root / 'src/dsse/data/ieee123'
    provided_assets = {p.name: sha(p) for p in sorted(dss.iterdir()) if p.is_file()}
    if any(static['dss_asset_sha256'].get(name) != digest for name, digest in provided_assets.items()):
        raise ValueError('Static mapping feeder assets differ')
    names = [row['opendss_bus'] for row in cache['bus_mapping']]
    if names != [row['opendss_bus'] for row in static['buses']] or len(names) != 132:
        raise ValueError('Cached model order differs from actual OpenDSS bus order')
    fallback = read(root / 'assets/physical/fallback_asset_projection.json')
    if names != fallback['bus_names'] or not np.allclose(fallback['v'], static['asset_nominal_voltage'], atol=1e-12, rtol=0):
        raise ValueError('Nominal fallback projection is inconsistent')
    rows = []
    for original in static['buses']:
        row = dict(original)
        i, slot = row['model_index'], row['pmu_slot']
        row.update(voltage_target_index=i, angle_target_index=132+i,
            scada_tensor_bus_index=i, kalman_P_index=2*i, kalman_Q_index=2*i+1,
            pmu_tensor_station_index=slot,
            kalman_pmu_voltage_index=None if slot is None else 264+2*slot,
            kalman_pmu_angle_index=None if slot is None else 265+2*slot,
            load_names=[load['name'] for load in static['loads'] if load['bus_index']==i])
        rows.append(row)
    parameters = [dict(parameter_index=i, kind='load_multiplier', **load) for i, load in enumerate(static['loads'])]
    parameters += [dict(parameter_index=len(static['loads'])+i, kind='continuous_transformer_tap', **reg)
        for i, reg in enumerate(static['regulators'])]
    return dict(schema_version=1, status='EXPORTED_FROM_HASH_BOUND_OPENDSS_AND_CACHE',
        bus_count=static['bus_count'], energized_phase_node_count=static['node_count'],
        cache_manifest_sha256=sha(cache_path), source_sha256=PHYSICS_SHA,
        dss_asset_sha256=provided_assets, node_order=static['node_order'], buses=rows,
        physical_parameter_dimension=static['dimension'], physical_parameters=parameters,
        bus_projection='w_b = mean_{p in energized phases}(V_{b,p}/base_volts_b * exp(j*2*pi*(p-1)/3)); target=(abs(w_b),arg(w_b))',
        angle_unit='radian; phase-A-aligned bus phasor angle, not reference-bus-subtracted',
        power_base_mva=1.0, scada_semantics='sum nominal load-setting kW/kvar times load multiplier, divided by 1000; not AC net injections',
        scada_third_channel='dummy zero with mask zero', nominal_static_asset_only=True,
        trajectories_generated=0, training_performed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True, help='New directory')
    p.add_argument('--static-json', type=Path, help='Optional prior static export; its actual source/assets are reverified')
    a = p.parse_args()
    layout = export_layouts(a.root)
    phase = export_mapping(a.root, read(a.static_json) if a.static_json else static_asset(a.root))
    a.output.mkdir(parents=True, exist_ok=False)
    for name, value in (('pmu_layouts.json', layout), ('bus_phase_mapping.json', phase)):
        (a.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='EXPORTED', bus_count=phase['bus_count'], phase_nodes=phase['energized_phase_node_count'], layouts=8)))
