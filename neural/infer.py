"""Run neural estimation and gate-control experiments from the model catalog."""
from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural.model_loader import NeuralRuntime,CASE_KEYS,SEEDS,sensor_hash,numpy_readout,sha,write

def metrics(prediction,target):
    e=prediction.astype(np.float64)-target.astype(np.float64)
    return np.sqrt(np.mean(e[:,:132]**2,axis=1)),np.sqrt(np.mean(np.angle(np.exp(1j*e[:,132:]))**2,axis=1))

def compare_reference(runtime,entry,case,predictions,*,gate_label=None):
    refs=[r for r in runtime.catalog.get('reference_predictions',[]) if (gate_label is not None or r['variant']==entry['variant']) and int(r['seed'])==int(entry['seed']) and r['case_key']==case and r.get('gate_label')==gate_label]
    results=[]
    for ref in refs:
        policy=ref.get('comparison_policy','strict_reference')
        legacy=policy=='legacy_nondeterministic_S0_roundoff_diagnostic'
        if policy not in ('strict_reference','legacy_nondeterministic_S0_roundoff_diagnostic'):
            raise ValueError('Unknown archive-reference comparison policy')
        if legacy:
            historical=('gat_mamba_no_gate','gat_mamba_full_gate','r4_adaptive','r4_quality_only','r4_constant','gnn_se_snapshot','sequence_mlp')
            if case!='S0' or gate_label is not None or entry['variant'] not in historical or int(entry['seed'])!=131:
                raise ValueError('Legacy archive policy is limited to documented seed131 historical S0 references')
            if float(ref.get('diagnostic_roundoff_limit',0.))!=2e-6 or not ref.get('policy_reason'):
                raise ValueError('Legacy archive policy requires its documented fixed 2e-6 bound and reason')
            if ref.get('comparison_count')!=64:
                raise ValueError('Legacy archive diagnostic is bound to its observed first64 rows')
        with np.load(runtime.asset(ref),allow_pickle=False) as f:
            for arm,key in ref['prediction_keys'].items():
                if arm not in predictions:continue
                count=ref.get('comparison_count',len(predictions[arm]))
                if not isinstance(count,int) or isinstance(count,bool) or count<=0:raise ValueError('Reference comparison_count must be a positive integer')
                count=min(count,len(predictions[arm]));p=predictions[arm][:count];expected=f[key][:count]
                if count==0 or p.shape!=expected.shape:raise ValueError('Archive reference and replay comparison shapes differ')
                difference=float(np.abs(p.astype(np.float64)-expected.astype(np.float64)).max())
                strict=difference<=1e-6;documented=legacy and not strict and difference<=2e-6
                classification='STRICT_REFERENCE_AGREEMENT' if strict else 'DOCUMENTED_LEGACY_BACKEND_ROUNDOFF' if documented else 'LEGACY_BACKEND_ROUNDOFF_LIMIT_EXCEEDED' if legacy else 'STRICT_REFERENCE_MISMATCH'
                results.append(dict(arm=arm,reference_path=ref['path'],comparison_count=count,comparison_policy=policy,
                    bitexact=p.dtype==expected.dtype and p.tobytes()==expected.tobytes(),max_abs_difference=difference,
                    strict_tolerance=1e-6,strict_passed=strict,diagnostic_roundoff_limit=2e-6 if legacy else None,
                    classification=classification,passed=strict or documented))
    return results

def save_prediction(output,runtime,entry,case,loaded,predictions,features,identity,rows,records,*,gate_label=None,arm_specs=None):
    a=loaded['arrays'];n=len(next(iter(predictions.values())));target=a['targets'][:n];seed=int(entry['seed'])
    directory=output/('gate_controls' if gate_label else entry['variant'])/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(f'{gate_label}_{case}.npz' if gate_label else f'predictions_{case}.npz')
    arrays=dict(targets=target,trajectory_seed=a['trajectory_seed'][:n],trajectory_index=a['trajectory_index'][:n],layout_index=a['layout_index'][:n],sensor_sha256=np.asarray(sensor_hash(loaded['sensors'][:n])))
    for arm,pred in predictions.items():
        if arm=='zero':continue
        key='prediction__'+arm;arrays[key]=pred;vr,ar=metrics(pred,target)
        arrays['voltage_rmse__'+arm]=vr;arrays['angle_rmse__'+arm]=ar
        method='r4_adaptive__'+gate_label if gate_label else entry['variant']+'__original' if arm=='baseline' else entry['variant'] if arm=='main' else entry['variant']+'__'+arm
        # Gate main is the published diagnostic method. Its baseline is saved
        # for inspection but not duplicated as a second diagnostic method.
        if gate_label and arm=='baseline':continue
        valid=True if arm=='baseline' else (arm_specs or entry['arms'])[arm].get('valid_for_scientific_comparison',True)
        records.append(dict(case_key=case,method=method,variant=entry['variant'],model_seed=seed,arm=gate_label or arm,
            relative_path=path.relative_to(output).as_posix(),prediction_key=key,sensor_sha256=str(arrays['sensor_sha256'].item()),count=n,valid_for_scientific_comparison=valid))
        rows.append(dict(case_key=case,method=method,model_seed=seed,voltage_mean_trajectory_RMSE=float(vr.mean()),angle_mean_trajectory_RMSE=float(ar.mean())))
    np.savez_compressed(path,**arrays)
    if features is not None:np.savez_compressed(directory/(f'{gate_label}_{case}_features.npz' if gate_label else f'features_{case}.npz'),**features)
    write(directory/(f'{gate_label}_{case}_identity.json' if gate_label else f'identity_{case}.json'),identity)

def gate_configurations(runtime,seed):
    fitted=next(x for x in runtime.catalog['gate_controls']['seeds'] if int(x['seed'])==seed)
    adaptive=runtime.parent('r4_adaptive',seed);main=adaptive['arms']['main']
    configurations=[('samehead_adaptive','r4_adaptive',None,main),
        ('samehead_noGate','gat_mamba_no_gate',None,main),
        ('samehead_context_parent','r4_quality_only',None,main),
        ('samehead_learned_constant_parent','r4_constant',None,main),
        ('samehead_entropy_centered','r4_adaptive',dict(kind='entropy_centered'),main),
        ('samehead_identity','r4_adaptive',dict(kind='fixed_unanchored_gain',q=1.),main),
        ('samehead_unanchored_q0p5','r4_adaptive',dict(kind='fixed_unanchored_gain',q=.5),main),
        ('samehead_trainmean_q','r4_adaptive',dict(kind='fixed_unanchored_gain',q=fitted['trainmean_q']),main)]
    common=dict(view='raw_plus_H',weights=fitted['common_head'],valid_for_scientific_comparison=fitted.get('valid_for_scientific_comparison',True))
    configurations.extend([('commonhead_adaptive','r4_adaptive',None,common),('commonhead_noGate','gat_mamba_no_gate',None,common)])
    return configurations

def run(args):
    runtime=NeuralRuntime(args.catalog,root=args.root,device=args.device)
    if args.check_manifest:
        for e in runtime.parents:
            for spec in e['arms'].values():runtime.coefficients(spec['weights'])
        for c in runtime.catalog.get('gate_controls',{}).get('seeds',[]):runtime.coefficients(c['common_head'])
        print(json.dumps(dict(status='PASS_FINAL_CATALOG_LEAF_IDENTITIES_AND_FP64_COEFFICIENTS',parents=len(runtime.parents),verified_leaves=len(runtime.verified),models_constructed=False)))
        return
    if args.output is None:raise ValueError('--output required for inference')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    selected=[e for e in runtime.parents if (not args.variant or e['variant'] in args.variant) and (not args.seed or int(e['seed']) in args.seed)]
    if not selected:raise ValueError('No selected parent')
    casekeys=args.case or list(runtime.cases)
    if any(k not in runtime.cases or k not in CASE_KEYS for k in casekeys):raise ValueError('Unexpected/missing explicit case')
    rows=[];records=[];checks=[];all_reference_checks=[];started=time.monotonic()
    manifest=dict(status='STARTED_NEURAL_INFERENCE',catalog_sha256=sha(args.catalog),source='Saved neural models and fitted readouts',batch_size=64,
        FP64_residual_FP32_native_links=True,training=False,case_generation=False,smoke_only=args.smoke,prediction_records=records)
    write(output/'run_manifest.json',manifest)
    try:
        for case in casekeys:
            loaded=runtime.load_case(case);sensors=loaded['sensors'][:args.limit] if args.limit else loaded['sensors']
            for entry in selected:
                collect=args.save_features or args.smoke
                predictions,features,identity=runtime.predict(entry,sensors,save_features=collect,include_zero=args.smoke)
                identity['final_reference_checks']=compare_reference(runtime,entry,case,predictions)
                all_reference_checks.extend(identity['final_reference_checks'])
                if args.smoke:
                    mathematical=numpy_readout(features,runtime.coefficients(entry['arms']['main']['weights']),entry['arms']['main']['view'])
                    delta=float(np.abs(predictions['main'].astype(np.float64)-mathematical).max())
                    check=dict(variant=entry['variant'],seed=entry['seed'],case_key=case,n=len(sensors),parent_restored=identity['parent_restored'],
                        zero_bitexact=identity['whole_output_zero_bitexact'],native_FP64_max_abs_difference=delta,native_FP64_passed=delta<=1e-6,reference=identity['final_reference_checks'])
                    checks.append(check)
                save_prediction(output,runtime,entry,case,loaded,predictions,features if collect else None,identity,rows,records)
                write(output/'run_manifest.json',manifest)
            if args.gate_controls and (case in ('S0','S5','S7') or case.startswith('S6_L')):
                if 'gate_controls' not in runtime.catalog:raise ValueError('Missing final common-head/train-mean assets')
                seeds=args.seed or list(SEEDS)
                for seed in seeds:
                    adaptive_body=None
                    for label,parent_variant,intervention,spec in gate_configurations(runtime,seed):
                        entry=runtime.parent(parent_variant,seed)
                        predictions,features,identity=runtime.predict(entry,sensors,save_features=True,arms_override={'main':spec},intervention=intervention)
                        if adaptive_body is None:adaptive_body=identity['body_state_sha256']
                        if identity['body_state_sha256']!=adaptive_body:raise RuntimeError('Fixed-head parents do not share the original body')
                        identity['final_reference_checks']=compare_reference(runtime,entry,case,predictions,gate_label=label)
                        all_reference_checks.extend(identity['final_reference_checks'])
                        mathematical=numpy_readout(features,runtime.coefficients(spec['weights']),spec['view'])
                        delta=float(np.abs(predictions['main'].astype(np.float64)-mathematical).max())
                        identity['same_forward_native_FP64_max_abs_difference']=delta
                        checks.append(dict(variant=parent_variant,seed=seed,case_key=case,gate_label=label,parent_restored=identity['parent_restored'],native_FP64_passed=delta<=1e-6,native_FP64_max_abs_difference=delta,reference=identity['final_reference_checks']))
                        save_prediction(output,runtime,entry,case,loaded,predictions,features if args.save_features or args.smoke else None,identity,rows,records,gate_label=label,arm_specs={'main':spec})
        write(output/'summary.json',rows);write(output/'smoke_contract_checks.json',checks)
        with (output/'summary.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for r in records:r['archive_sha256']=sha(output/r['relative_path'])
        failures=[c for c in checks if not c.get('parent_restored',False) or not c.get('native_FP64_passed',True) or not c.get('zero_bitexact',True) or any(not r['passed'] for r in c.get('reference',[]))]
        failures.extend(c for c in all_reference_checks if not c['passed'])
        roundoff=[c for c in all_reference_checks if c['classification']=='DOCUMENTED_LEGACY_BACKEND_ROUNDOFF']
        status='FAIL_NEURAL_SMOKE' if args.smoke and failures else 'FAILED_NEURAL_CONTRACT' if failures else 'PASS_NEURAL_SMOKE_WITH_DOCUMENTED_ARCHIVE_ROUNDOFF' if args.smoke and roundoff else 'PASS_NEURAL_SMOKE' if args.smoke else 'COMPLETED_NEURAL_INFERENCE_WITH_DOCUMENTED_ARCHIVE_ROUNDOFF' if roundoff else 'COMPLETED_NEURAL_INFERENCE'
        manifest.update(status=status,
            runtime_seconds=time.monotonic()-started,case_count=len(casekeys),parent_count=len(selected),contract_failures=failures,
            final_reference_comparisons=len(all_reference_checks),roundoff_count=len(roundoff),documented_archive_roundoff=roundoff,
            archive_reference_strict_failures=[c for c in all_reference_checks if not c['strict_passed']],
            output_sha256={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='run_manifest.json'})
        write(output/'run_manifest.json',manifest)
        print(json.dumps(dict(status=manifest['status'],parents=len(selected),cases=len(casekeys),records=len(records),failures=len(failures),roundoff_count=len(roundoff),runtime_seconds=manifest['runtime_seconds'])))
        if failures:raise SystemExit(2)
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        manifest.update(status='FAILED_NEURAL_INFERENCE',exception_type=type(exc).__name__,exception=str(exc));write(output/'run_manifest.json',manifest);raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--root',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path)
    p.add_argument('--case',action='append');p.add_argument('--variant',action='append');p.add_argument('--seed',action='append',type=int)
    p.add_argument('--limit',type=int);p.add_argument('--gate-controls',action='store_true');p.add_argument('--save-features',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--check-manifest',action='store_true')
    a=p.parse_args()
    if a.limit is not None and (a.limit<=0 or not a.smoke):p.error('--limit is positive and smoke-only; full replay always keeps all final cases')
    run(a)
if __name__=='__main__':main()
