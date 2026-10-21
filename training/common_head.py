"""Training-only balanced common-head fit and constant gate-gain moments.

The original order is adaptive four views followed by no-gate four views.
No validation target, confirmation case, or selection statistic enters fitting.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from training.common import load_training_context,input_bindings,training_views
from training.readout import collect
from training.ridge_math import fit_view
from neural.model_loader import configure_torch,get_head_linears,set_heads,state_hash,sensor_hash,numpy_readout,sha,write,SEEDS

def native_check(runtime,model,items,weights):
    torch=configure_torch()
    from neural.residual_readout import StableResidualReadout
    owner,vh,ah=get_head_linears(model);before=state_hash(model);capture={}
    def hv(m,args,z):capture.update(H_voltage=args[0].detach().cpu().numpy().copy(),voltage_logit=z.detach().cpu().numpy().copy())
    def ha(m,args,z):capture.update(H_angle=args[0].detach().cpu().numpy().copy(),angle_logit=z.detach().cpu().numpy().copy())
    hooks=[vh.register_forward_hook(hv),ah.register_forward_hook(ha)]
    sensors,raw=runtime.batch([{k:x[k] for k in ('scada','scada_mask','pmu','pmu_mask')} for x in items[:64]])
    edges=torch.as_tensor(runtime.edge_index,dtype=torch.long,device=runtime.device)
    def forward():
        with torch.no_grad():o=model(sensors['scada'],sensors['pmu'],edges,scada_mask=sensors['scada_mask'],pmu_mask=sensors['pmu_mask'])
        return torch.cat([o['V_pred'],o['delta_pred']],dim=-1).cpu().numpy().copy()
    try:
        baseline=forward();wv=StableResidualReadout(vh);wa=StableResidualReadout(ah)
        wv.eval();wa.eval();wv.requires_grad_(False);wa.requires_grad_(False)
        set_heads(owner,wv,wa);wv.set_raw_measurements(raw);wa.set_raw_measurements(raw);zero=forward()
        for kind,w in (('voltage',wv),('angle',wa)):
            w.load_fitted_raw_affine(torch.as_tensor(weights[kind+'__W'],device=runtime.device),torch.as_tensor(weights[kind+'__b'],device=runtime.device))
        wv.set_raw_measurements(raw);wa.set_raw_measurements(raw);actual=forward()
        features=dict(capture,raw_final=raw.cpu().numpy().copy());math=numpy_readout(features,weights)
        difference=float(np.abs(actual.astype(np.float64)-math).max())
    finally:
        set_heads(owner,vh,ah)
        for h in hooks:h.remove()
    restored=state_hash(model)==before;identical=baseline.dtype==zero.dtype and baseline.tobytes()==zero.tobytes()
    return dict(native_FP64_max_abs_difference=difference,native_FP64_passed=difference<=1e-6,zero_bitexact=identical,parent_state_restored=restored),dict(actual=actual,mathematical=math,zero=zero,baseline=baseline,**features)

def run(a):
    started=time.perf_counter();runtime,shared,source,cache,train,_=load_training_context(a.catalog,a.device)
    torch=configure_torch();shared.seed_everything(0)
    if a.seed not in SEEDS:raise ValueError('Use an original model seed')
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Actual CUDA/Mamba required')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);count=16 if a.smoke else 2048
    views=training_views(train,cache,shared,count)
    models={};origins={};packs=[];parent_packs={}
    for variant,explicit in (('r4_adaptive',a.adaptive_checkpoint),('gat_mamba_no_gate',a.no_gate_checkpoint)):
        model,identity=runtime.load_parent(runtime.parent(variant,a.seed))
        if explicit:
            path=explicit.resolve()
            if not path.is_relative_to(runtime.root):raise ValueError('Explicit checkpoint must be inside the repository')
            payload=torch.load(path,map_location=a.device,weights_only=True)
            if payload['model_seed']!=a.seed:raise ValueError('Explicit parent checkpoint seed differs')
            if payload.get('variant')!=variant:raise ValueError('Explicit parent checkpoint variant differs')
            model.load_state_dict(payload['model_state_dict'],strict=True)
            identity['explicit_retrained_checkpoint']=dict(path=path.relative_to(runtime.root).as_posix(),sha256=sha(path))
        model.eval();model.requires_grad_(False);models[variant]=model;origins[variant]=identity;parent_packs[variant]={}
        for name,items in views.items():
            pack=collect(runtime,model,items);packs.append(pack);parent_packs[variant][name]=pack
            np.savez_compressed(a.output/f'training_features_{variant}_{name}.npz',**pack)
    body=[state_hash(m,exclude_r3=True) for m in models.values()]
    if len(set(body))!=1:raise RuntimeError('Balanced common-head parents must share the same inherited body')
    if any(p['targets'].tobytes()!=packs[0]['targets'].tobytes() or p['trajectory_seed'].tobytes()!=packs[0]['trajectory_seed'].tobytes() for p in packs):raise RuntimeError('Common-head parent/view endpoint targets differ')
    arrays={};stats={}
    for kind in ('voltage','angle'):
        fit,stat=fit_view(packs,kind,'raw_plus_H',1e-4,1e-6);stat['unique_endpoint_labels']=count
        arrays.update({kind+'__'+k:v for k,v in fit.items()});stats[kind]=stat
    weights={kind+'__'+key:arrays[kind+'__'+key].astype(np.float64) for kind in ('voltage','angle') for key in ('W','b')}
    np.savez_compressed(a.output/'common_head_fit.npz',**arrays);np.savez_compressed(a.output/'common_head_native.npz',**weights)
    adaptive=models['r4_adaptive'];backbone=adaptive.paper_backbone;before=state_hash(adaptive);capture=backbone._r3_capture
    qsum=np.zeros((3,4),np.float64);applied_sum=qsum.copy();total=0;viewstats={};edges=torch.as_tensor(runtime.edge_index,dtype=torch.long,device=a.device)
    try:
        backbone._r3_capture=True
        for name,items in views.items():
            part=np.zeros((3,4),np.float64);safe_part=part.copy();n=0
            for start in range(0,len(items),64):
                sensors,_=runtime.batch([{k:x[k] for k in ('scada','scada_mask','pmu','pmu_mask')} for x in items[start:start+64]])
                with torch.no_grad():shared.forward_model(adaptive,sensors,edges)
                q=backbone._last_r4_unanchored_gains;safe=backbone._last_r3_gains
                n+=int(np.prod(q[0].shape[:-1]))
                for layer in range(3):
                    part[layer]+=q[layer].detach().cpu().numpy().astype(np.float64).sum(axis=(0,1))
                    safe_part[layer]+=safe[layer].detach().cpu().numpy().astype(np.float64).sum(axis=(0,1))
            qsum+=part;applied_sum+=safe_part;total+=n
            viewstats[name]=dict(count=n,unanchored_mean=(part/n).tolist(),applied_mean=(safe_part/n).tolist())
    finally:
        backbone._r3_capture=capture
    if state_hash(adaptive)!=before:raise RuntimeError('Gain-moment capture changed parent state')
    mean=qsum/total
    if not np.all((mean>0)&(mean<2)):raise RuntimeError('Unanchored gain means outside their open native bounds')
    write(a.output/'training_gate_gain_mean.json',dict(unanchored_q_mean=mean.tolist(),applied_gain_mean=(applied_sum/total).tolist(),
        count_per_layer_head=total,views=viewstats,no_target_values_used=True,anchor_retained=True))
    checks=[]
    for variant,model in models.items():
        check,values=native_check(runtime,model,views['normal'],weights);checks.append(dict(parent=variant,**check))
        np.savez_compressed(a.output/f'common_native_check_{variant}.npz',**values)
    write(a.output/'common_native_checks.json',checks);write(a.output/'fit_statistics.json',stats)
    passed=all(x['native_FP64_passed'] and x['zero_bitexact'] and x['parent_state_restored'] for x in checks)
    manifest=dict(status=('COMPLETED_COMMON_HEAD_SMOKE' if a.smoke else 'COMPLETED_COMMON_HEAD_AND_GAIN_FIT') if passed else 'FAILED_COMMON_HEAD_NATIVE_CONTRACT',
        model_seed=a.seed,smoke_only=a.smoke,input_files=input_bindings(runtime,cache),parents=origins,shared_body_sha256=body[0],
        training_sensor_sha256={k:sensor_hash(v) for k,v in views.items()},training_view_order=list(views),parent_order=list(models),
        augmented_parent_view_rows=count*8,unique_endpoint_labels=count,validation_labels_used=False,confirmation_accessed=False,body_training=False,
        alpha=1e-4,std_floor=1e-6,numerical_policy='FP64_FIT_COEFFICIENTS_AND_AFFINE__FP32_BACKBONE_AND_NATIVE_LINK',
        runtime_seconds=time.perf_counter()-started,output_sha256={p.relative_to(a.output).as_posix():sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()})
    write(a.output/'run_manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],model_seed=a.seed,native_checks=len(checks))))
    if not passed:raise RuntimeError('Common-head native contract failed; all outputs retained')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--adaptive-checkpoint',type=Path);p.add_argument('--no-gate-checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
