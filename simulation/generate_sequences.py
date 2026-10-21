"""Generate simulated training, validation and confirmation trajectories."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation.runtime import ROOT,load_cache,load_sequences,sha
def main(a):
    started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    cache=load_cache();saved=load_sequences(a.split);count=1 if a.smoke else len(saved['trajectory_seed'])
    from dsse.feeder import IEEE123BusFeeder
    from dsse.datasets import ARRAY_KEYS,stack_samples,write_npz_atomic
    feeder=IEEE123BusFeeder(ROOT/'src'/cache.config['data']['dss_dir'],cache.config['data']['pmu_bus_names'])
    if feeder.mapping_records()!=cache.manifest['bus_mapping']:raise ValueError('Feeder bus order differs')
    if not np.array_equal(feeder.edge_index,cache.edge_index):raise ValueError('Feeder edges differ')
    checks=[];shards=[];buffer=[]
    for index in range(count):
        seed=int(saved['trajectory_seed'][index]);sample=feeder.generate_sequence(30,seed)
        same=all(np.array_equal(np.asarray(sample[k],dtype=np.float32),saved[k][index]) for k in ARRAY_KEYS)
        checks.append(dict(trajectory_seed=seed,same_arrays_as_final=bool(same)))
        if not same:raise ValueError('Simulation replay differs from original data at seed '+str(seed))
        buffer.append(sample)
        if len(buffer)==64 or index==count-1:
            target=a.output/f'shard_{len(shards):03}.npz';write_npz_atomic(target,stack_samples(buffer))
            shards.append(dict(relative_path=target.name,sha256=sha(target),trajectory_count=len(buffer)));buffer=[]
    report=dict(status='PASS_SIMULATION_REPLAY_SMOKE' if a.smoke else 'PASS_SIMULATION_REPLAY',split=a.split,
        trajectory_count=count,checks=checks,shards=shards,model_training=False,model_selection=False,elapsed_seconds=time.perf_counter()-started)
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','split','trajectory_count','elapsed_seconds')}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['train','validation','confirmation'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true',help='Verify only the first saved trajectory; not a full-data reproduction claim')
    main(p.parse_args())
