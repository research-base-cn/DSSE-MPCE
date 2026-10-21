"""Train graph-temporal backbone models."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,math,time
from training.common import load_training_context,input_bindings,HERE
from neural.model_loader import configure_torch,state_hash,module_from_path,sha,write,SEEDS

def run(a):
    runtime,shared,source,cache,train,validation=load_training_context(a.catalog,a.device)
    torch=configure_torch();recipe=json.loads((HERE.parent/'configs/backbone.json').read_text())
    if a.seed not in SEEDS:raise ValueError('Use one of the five actual training seeds')
    if a.smoke:
        recipe.update(epochs=1,train_trajectory_limit=16,validation_trajectory_limit=16)
        train={k:v[:16] for k,v in train.items()};validation={k:v[:16] for k,v in validation.items()}
    structural=a.variant in ('corrected_gcn_mamba','matched_gat_bilstm')
    if structural:
        from training.structural_model import build_structural_model
        width=json.loads((HERE.parent/'configs/structural_controls.json').read_text())
        model,construction=build_structural_model(source,cache,a.variant,a.seed,'cpu' if a.contract_only else a.device,width)
    else:
        shared.seed_everything(a.seed);preserve=module_from_path(runtime.asset(runtime.catalog['helpers']['preserve_init']),'preserve_mamba_init')
        with preserve.preserve_mamba_initialization():model=shared.build_formal_model(cache.config,num_buses=cache.num_buses,num_pmu_buses=cache.pmu_count,pmu_bus_indices=cache.pmu_bus_indices,variant=a.variant,device=torch.device('cpu' if a.contract_only else a.device))
        if a.variant=='gat_mamba_full_gate':shared.install_pre_dropout_entropy(model)
        construction=dict(initial_state_sha256=state_hash(model),registered_parameters=sum(v.numel() for v in model.parameters()))
    if a.contract_only:
        print(json.dumps(dict(status='PASS_NATIVE_CONSTRUCTION_NO_OPTIMIZATION',variant=a.variant,seed=a.seed,construction=construction)));return
    if a.output is None:raise ValueError('--output required')
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Actual CUDA/Mamba backend required')
    if a.output.exists():raise FileExistsError(a.output)
    # The original v3 routine performs its own same-seed construction again.
    # The temporary contract construction above is discarded; seed_everything
    # inside that routine restores the exact original initialization sequence.
    if not structural:
        del model
        if a.variant in ('gnn_se_snapshot','sequence_mlp'):
            from dsse import training as formal_train
            from training.registry import training_time_config
            formal_train.TrajectoryCache=shared.TrajectoryCache
            legacy_config,compatibility=training_time_config(runtime,a.variant)
            formal_train.validate_job_request(legacy_config,a.variant,a.seed)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            legacy_config_path=a.output.parent/(a.output.name+'.training_config.json')
            if legacy_config_path.exists():raise FileExistsError(legacy_config_path)
            write(legacy_config_path,legacy_config)
            compatibility['training_time_config_file']=dict(path=str(legacy_config_path.resolve()),sha256=sha(legacy_config_path))
            result=formal_train.run_training_job(config_path=legacy_config_path,cache_dir=cache.cache_dir,output_dir=a.output,
                variant=a.variant,model_seed=a.seed,device_name=a.device,allow_inadmissible_smoke=a.smoke,smoke_epochs=1 if a.smoke else None)
            write(a.output/'registry_order.json',compatibility)
        else:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            rp=a.output.parent/(a.output.name+'.recipe.json')
            if rp.exists():raise FileExistsError(rp)
            write(rp,recipe)
            result=shared.run(SimpleNamespace(source_root=source,base_config=runtime.asset(runtime.catalog['config']),protocol=rp,cache=cache.cache_dir,
                output=a.output,arm='no_gate' if a.variant=='gat_mamba_no_gate' else 'corrected_gate',seed=a.seed,device=a.device),recipe)
        write(a.output/'input_bindings.json',dict(inputs=input_bindings(runtime,cache),smoke_only=a.smoke,recipe_source_sha256=sha(HERE.parent/'configs/backbone.json'),
            contract='Graph-temporal training with the configured initialization and data splits'))
        print(json.dumps(dict(status='COMPLETED_TRAINING_SMOKE' if a.smoke else 'COMPLETED_BACKBONE_TRAINING',variant=a.variant,seed=a.seed)));return
    a.output.mkdir(parents=True)
    manifest=dict(status='RUNNING',variant=a.variant,model_seed=a.seed,smoke_only=a.smoke,split_access=['train','validation'],
        training_recipe=recipe,construction=construction,input_files=input_bindings(runtime,cache),body_training=True,heldout_accessed=False)
    write(a.output/'run_manifest.json',manifest)
    device=torch.device(a.device);edges=torch.as_tensor(cache.edge_index,dtype=torch.long,device=device)
    dataset=shared.PairedDataset(train,cache,cache.config,recipe,a.seed)
    generator=torch.Generator().manual_seed(a.seed)
    loader=shared.DataLoader(dataset,batch_size=recipe['batch_size'],shuffle=True,num_workers=0,generator=generator)
    normal_loader=shared.DataLoader(shared.PairedDataset(validation,cache,cache.config,recipe,a.seed,'normal'),batch_size=recipe['batch_size'],shuffle=False,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=recipe['scheduler_horizon_epochs'],eta_min=1e-6)
    best_score,best_epoch,stale=math.inf,0,0;history=[];torch.cuda.synchronize(device);started=time.perf_counter()
    for epoch in range(1,recipe['epochs']+1):
        tick=time.perf_counter();dataset.epoch=epoch;model.train();rows=[];order=hashlib.sha256();inputs=hashlib.sha256()
        for raw in loader:
            order.update(raw['trajectory_seed'].numpy().tobytes())
            for key in ('scada','scada_mask','pmu','pmu_mask','v_true','angle_true'):inputs.update(key.encode('ascii'));inputs.update(raw[key].contiguous().numpy().tobytes())
            batch=shared.move_batch(raw,device);optimizer.zero_grad(set_to_none=True)
            outputs=shared.forward_model(model,batch,edges);loss,lv,la=shared.endpoint_loss(outputs,batch,(1.,1.))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip']);optimizer.step()
            rows.append([float(loss.detach().cpu()),float(lv.detach().cpu()),float(la.detach().cpu())])
        scheduler.step();normal=shared.evaluate(model,normal_loader,edges,device,(1.,1.));score=normal['voltage_rmse']+normal['angle_circular_rmse']
        record=dict(epoch=epoch,training_loss=shared.np.mean(rows,axis=0).tolist(),normal_validation=normal,selection_score=score,
            epoch_wall_seconds=time.perf_counter()-tick,train_data_order_sha256=order.hexdigest(),train_corrupted_inputs_and_targets_sha256=inputs.hexdigest())
        history.append(record)
        with (a.output/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        payload=dict(model_state_dict=model.state_dict(),variant=a.variant,model_seed=a.seed,epoch=epoch,construction=construction,smoke_only=a.smoke)
        if score<best_score:
            best_score,best_epoch,stale=score,epoch,0;torch.save(payload,a.output/'checkpoint.pt')
        else:
            stale+=1
            if stale>=recipe['early_stopping_patience']:break
    torch.cuda.synchronize(device);wall=time.perf_counter()-started;torch.save(payload,a.output/'terminal_checkpoint.pt')
    model.load_state_dict(torch.load(a.output/'checkpoint.pt',map_location=device,weights_only=True)['model_state_dict'],strict=True)
    final={}
    for name in recipe['final_validation_report_scenarios']:
        vl=shared.DataLoader(shared.PairedDataset(validation,cache,cache.config,recipe,a.seed,name),batch_size=recipe['batch_size'],shuffle=False,num_workers=0)
        final[name]=shared.evaluate(model,vl,edges,device,(1.,1.))
    write(a.output/'training_result.json',dict(history=history,final_validation=final))
    manifest.update(status='COMPLETED_TRAINING_SMOKE' if a.smoke else 'COMPLETED_BACKBONE_TRAINING',epochs_completed=len(history),best_epoch=best_epoch,
        training_wall_time_seconds=wall,checkpoint_sha256=sha(a.output/'checkpoint.pt'),terminal_checkpoint_sha256=sha(a.output/'terminal_checkpoint.pt'))
    write(a.output/'run_manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],variant=a.variant,seed=a.seed)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--variant',choices=['gat_mamba_no_gate','gat_mamba_full_gate','corrected_gcn_mamba','matched_gat_bilstm','gnn_se_snapshot','sequence_mlp'],required=True)
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--contract-only',action='store_true');p.add_argument('--smoke',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
