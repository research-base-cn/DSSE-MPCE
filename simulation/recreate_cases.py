"""Recreate the final 17 observation cases from saved clean sequences."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation.runtime import ROOT,load_cache,load_sequences,scientific_runtime,read,sha
from simulation.case_definitions import make_case_map,SENSOR_KEYS
def main(a):
    tick=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    cache=load_cache();arrays=load_sequences('confirmation');layouts=read(ROOT/'configs/pmu_layouts.json')
    cases=make_case_map(arrays,cache,scientific_runtime(),count=512,layouts=layouts)
    records=[];checks=[]
    reference={r['case_key']:r for r in read(ROOT/'data/cases/bundle_manifest.json')['cases']}
    for case,items in cases.items():
        payload={k:np.stack([i[k] for i in items]) for k in SENSOR_KEYS}
        payload.update(targets=np.concatenate([np.stack([i['v_true'][-1] for i in items]),np.stack([i['angle_true'][-1] for i in items])],axis=1),
            trajectory_seed=np.asarray([i['trajectory_seed'] for i in items],dtype=np.int64),trajectory_index=np.arange(512,dtype=np.int64),
            layout_index=np.full(512,int(case.split('_L')[1]) if case.startswith('S6_L') else -1,dtype=np.int64))
        path=ROOT/'data/cases'/reference[case]['relative_path']
        if sha(path)!=reference[case]['sha256']:raise ValueError('Reference case identity changed')
        with np.load(path,allow_pickle=False) as f:
            same=set(f.files)==set(payload) and all(payload[k].dtype==f[k].dtype and np.array_equal(payload[k],f[k]) for k in payload)
        checks.append(dict(case=case,all_arrays_dtype_and_values_identical=bool(same)))
        if not same:raise ValueError('Reconstructed case differs: '+case)
        if not a.verify_only:
            target=a.output/(case+'.npz');np.savez_compressed(target,**payload)
            records.append(dict(case_key=case,relative_path=target.name,sha256=sha(target)))
    report=dict(status='PASS_EXACT_FINAL_OBSERVATION_RECONSTRUCTION',cases=len(cases),checks=checks,
        generation_of_new_clean_sequences=False,model_training=False,model_selection=False,elapsed_seconds=time.perf_counter()-tick)
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    if records:(a.output/'case_inventory.json').write_text(json.dumps(dict(cases=records),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','elapsed_seconds')}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-only',action='store_true')
    main(p.parse_args())
