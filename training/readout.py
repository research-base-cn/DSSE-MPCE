"""Fit residual readouts from training observations."""
from pathlib import Path
import argparse,json,time
import numpy as np
from training.common import load_training_context,input_bindings,training_views,HERE
from training.ridge_math import fit_view
from neural.model_loader import configure_torch,get_head_linears,state_hash,sha,write,sensor_hash,numpy_readout,set_heads,SEEDS

def collect(runtime,model,items):
    torch=configure_torch();_,vh,ah=get_head_linears(model);capture={}
    def hV(m,inputs,out):capture.update(H_voltage=inputs[0].detach().cpu().numpy().copy(),voltage_logit=out.detach().cpu().numpy().copy())
    def hA(m,inputs,out):capture.update(H_angle=inputs[0].detach().cpu().numpy().copy(),angle_logit=out.detach().cpu().numpy().copy())
    hooks=[vh.register_forward_hook(hV),ah.register_forward_hook(hA)]
    pieces={k:[] for k in ('raw_final','H_voltage','H_angle','voltage_logit','angle_logit','baseline')}
    edges=torch.as_tensor(runtime.edge_index,dtype=torch.long,device=runtime.device);model.eval();model.requires_grad_(False)
    try:
        for start in range(0,len(items),64):
            sensors,raw=runtime.batch([{k:x[k] for k in ('scada','scada_mask','pmu','pmu_mask')} for x in items[start:start+64]])
            with torch.no_grad():out=model(sensors['scada'],sensors['pmu'],edges,scada_mask=sensors['scada_mask'],pmu_mask=sensors['pmu_mask'])
            prediction=torch.cat([out['V_pred'],out['delta_pred']],dim=-1).cpu().numpy().copy()
            if prediction.shape!=(len(raw),264) or not np.isfinite(prediction).all():raise RuntimeError('Nonfinite/malformed native training prediction')
            pieces['baseline'].append(prediction);pieces['raw_final'].append(raw.cpu().numpy().copy())
            for k,v in capture.items():pieces[k].append(v)
    finally:
        for h in hooks:h.remove()
    return dict(**{k:np.concatenate(v) for k,v in pieces.items()},
        targets=np.c_[np.stack([x['v_true'][-1] for x in items]),np.stack([x['angle_true'][-1] for x in items])],
        trajectory_seed=np.asarray([x['trajectory_seed'] for x in items],dtype=np.int64))

def run(a):
    started=time.perf_counter();runtime,shared,source,cache,train,_=load_training_context(a.catalog,a.device)
    torch=configure_torch();shared.seed_everything(0)
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Actual CUDA/Mamba backend required')
    if a.seed not in SEEDS:raise ValueError('Use an original model seed')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    count=16 if a.smoke else 2048;views=training_views(train,cache,shared,count)
    if list(views)!=['normal','drop50','pmu_out30','joint_out30']:raise RuntimeError('Training view order differs')
    entry=runtime.parent(a.variant,a.seed);model,identity=runtime.load_parent(entry)
    if a.parent_checkpoint:
        checkpoint=torch.load(a.parent_checkpoint,map_location=a.device,weights_only=True)
        if checkpoint['model_seed']!=a.seed:raise ValueError('Explicit new parent seed differs')
        if checkpoint.get('variant')!=a.variant:raise ValueError('Explicit new parent variant differs')
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        identity.update(explicit_retrained_checkpoint={'path':str(a.parent_checkpoint.resolve()),'sha256':sha(a.parent_checkpoint)},full_state_sha256=state_hash(model))
    initial=state_hash(model);packs={}
    for name,items in views.items():
        packs[name]=collect(runtime,model,items);np.savez_compressed(a.output/f'training_features_{name}.npz',**packs[name])
    if any(p['targets'].tobytes()!=packs['normal']['targets'].tobytes() or p['trajectory_seed'].tobytes()!=packs['normal']['trajectory_seed'].tobytes() for p in packs.values()):raise RuntimeError('Training views alter endpoint labels/order')
    arms={'main':('raw_plus_H',list(views))}
    if a.variant=='r4_adaptive':arms.update(raw_final=('raw_final',list(views)),H_only=('H_only',list(views)),normal_only=('raw_plus_H',['normal']))
    statistics={};weights={}
    for arm,(view,names) in arms.items():
        arrays={};statistics[arm]={}
        for kind in ('voltage','angle'):
            fit,stat=fit_view([packs[n] for n in names],kind,view,1e-4,1e-6)
            stat['unique_endpoint_labels']=count
            statistics[arm][kind]=stat
            arrays.update({kind+'__'+k:v for k,v in fit.items()})
        np.savez_compressed(a.output/f'fit_{arm}.npz',**arrays)
        weights[arm]={kind+'__'+key:arrays[kind+'__'+key].astype(np.float64) for kind in ('voltage','angle') for key in ('W','b')}
        np.savez_compressed(a.output/f'native_weights_{arm}.npz',**weights[arm])
    # Native arithmetic check uses features from THIS wrapped forward, rather
    # than a separately collected H from another graph execution.
    from neural.residual_readout import StableResidualReadout
    owner,vh,ah=get_head_linears(model);wv=StableResidualReadout(vh);wa=StableResidualReadout(ah)
    for kind,w in (('voltage',wv),('angle',wa)):
        w.load_fitted_raw_affine(torch.as_tensor(weights['main'][kind+'__W'],device=a.device),torch.as_tensor(weights['main'][kind+'__b'],device=a.device));w.eval();w.requires_grad_(False)
    sensors,raw=runtime.batch([{k:x[k] for k in ('scada','scada_mask','pmu','pmu_mask')} for x in views['normal'][:64]])
    capture={}
    def hv(m,args,z):capture.update(H_voltage=args[0].detach().cpu().numpy().copy(),voltage_logit=z.detach().cpu().numpy().copy())
    def ha(m,args,z):capture.update(H_angle=args[0].detach().cpu().numpy().copy(),angle_logit=z.detach().cpu().numpy().copy())
    hooks=[vh.register_forward_hook(hv),ah.register_forward_hook(ha)];set_heads(owner,wv,wa)
    try:
        wv.set_raw_measurements(raw);wa.set_raw_measurements(raw);edges=torch.as_tensor(runtime.edge_index,dtype=torch.long,device=a.device)
        with torch.no_grad():out=model(sensors['scada'],sensors['pmu'],edges,scada_mask=sensors['scada_mask'],pmu_mask=sensors['pmu_mask'])
        actual=torch.cat([out['V_pred'],out['delta_pred']],dim=-1).cpu().numpy().copy()
        actual_features=dict(capture,raw_final=raw.cpu().numpy().copy());reference=numpy_readout(actual_features,weights['main'])
        difference=float(np.abs(actual.astype(np.float64)-reference).max())
    finally:
        set_heads(owner,vh,ah)
        for h in hooks:h.remove()
    if state_hash(model)!=initial:raise RuntimeError('Calibration changed parent state')
    if difference>1e-6:raise RuntimeError('Same-forward native/FP64 reconstruction exceeds unchanged1e-6')
    np.savez_compressed(a.output/'same_forward_native_check.npz',actual=actual,mathematical=reference,**actual_features)
    write(a.output/'fit_statistics.json',statistics)
    manifest=dict(status='COMPLETED_READOUT_SMOKE' if a.smoke else 'COMPLETED_P1_READOUT_FIT',variant=a.variant,model_seed=a.seed,smoke_only=a.smoke,
        source_files=input_bindings(runtime,cache),parent=identity,training_sensor_sha256={k:sensor_hash(v) for k,v in views.items()},training_views=list(views),
        unique_endpoint_labels=count,augmented_rows=count*4,validation_labels_used=False,body_training=False,
        alpha=1e-4,std_floor=1e-6,batch_size=64,numerical_policy='FP64_FIT_COEFFICIENTS_AND_AFFINE__FP32_BACKBONE_AND_NATIVE_LINK',
        same_forward_native_FP64_max_abs_difference=difference,parent_state_restored=True,runtime_seconds=time.perf_counter()-started,
        outputs={p.relative_to(a.output).as_posix():sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()})
    write(a.output/'run_manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],variant=a.variant,seed=a.seed)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--parent-checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
