"""Backbone optimization, validation and checkpoint selection."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import sys
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def bootstrap(source_root):
    global np, torch, F, DataLoader, TrajectoryCache, sample_at, tensorize_last_state
    global corrupt_formal_sample, scenario_seed, training_corruption_seed
    global build_formal_model, parameter_count, seed_everything, build_run_manifest, canonical_sha256
    sys.path.insert(0, str(Path(source_root).resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from dsse.runtime import (
        TrajectoryCache, sample_at, tensorize_last_state, corrupt_formal_sample,
        scenario_seed, training_corruption_seed,
    )
    from dsse.models import build_formal_model, parameter_count
    from dsse.reproducibility import seed_everything, build_run_manifest, canonical_sha256


def install_pre_dropout_entropy(model):
    """Use the single reviewed repair implementation; no alternate gate code."""
    from normalized_entropy_gate import install_normalized_entropy_repair
    return install_normalized_entropy_repair(model, slope_mode='free')


def state_hash(model, exclude_gate=False):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if exclude_gate and ('entropy_gate_w' in name or 'entropy_gate_b' in name):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


class PairedDataset:
    """Same trajectory ordering and corrupted arrays for each arm/seed."""
    def __init__(self, arrays, cache, base_config, protocol, model_seed, validation_scenario=None):
        self.arrays = arrays
        self.cache = cache
        self.config = base_config
        self.protocol = protocol
        self.seed = model_seed
        self.validation_scenario = validation_scenario
        self.epoch = 0

    def __len__(self):
        return len(self.arrays['trajectory_seed'])

    def __getitem__(self, index):
        sample = sample_at(self.arrays, index)
        if self.validation_scenario is not None:
            scenario_name = self.validation_scenario
            seed = scenario_seed(sample['trajectory_seed'], scenario_name)
        else:
            names = self.protocol['augmentation_scenarios']
            scenario_name = names[(self.epoch + index) % len(names)]
            seed = training_corruption_seed(self.seed, self.epoch, sample['trajectory_seed'], scenario_name)
        scenario = self.config['primary_scenarios'].get(scenario_name, self.config['targeted_scenarios'].get(scenario_name))
        if scenario is None:
            raise ValueError(f'Unknown scenario: {scenario_name}')
        corrupted, _ = corrupt_formal_sample(sample, scenario_name, scenario, np.random.default_rng(seed), self.config, self.cache.manifest)
        return tensorize_last_state(corrupted, self.cache.normalizer)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def forward_model(model, batch, edges):
    return model(batch['scada'], batch['pmu'], edges,
                 scada_mask=batch['scada_mask'], pmu_mask=batch['pmu_mask'])


def endpoint_loss(outputs, batch, loss_scales):
    voltage = F.mse_loss(outputs['V_pred'], batch['v_true'])
    angle = F.mse_loss(outputs['delta_pred'], batch['angle_true'])
    loss = voltage / loss_scales[0] ** 2 + angle / loss_scales[1] ** 2
    return loss, voltage, angle


def evaluate(model, loader, edges, device, loss_scales):
    model.eval()
    v_errors, a_errors, losses, v_losses, a_losses = [], [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            output = forward_model(model, batch, edges)
            loss, lv, la = endpoint_loss(output, batch, loss_scales)
            losses.append(float(loss.cpu()))
            v_losses.append(float(lv.cpu()))
            a_losses.append(float(la.cpu()))
            v_errors.append((output['V_pred'] - batch['v_true']).cpu().numpy())
            difference = (output['delta_pred'] - batch['angle_true']).cpu().numpy()
            a_errors.append(np.angle(np.exp(1j * difference)))
    ve, ae = np.concatenate(v_errors), np.concatenate(a_errors)
    return {
        'voltage_rmse': float(np.sqrt(np.mean(ve ** 2))),
        'angle_circular_rmse': float(np.sqrt(np.mean(ae ** 2))),
        'mean_trajectory_voltage_rmse': float(np.sqrt(np.mean(ve ** 2, axis=1)).mean()),
        'mean_trajectory_angle_circular_rmse': float(np.sqrt(np.mean(ae ** 2, axis=1)).mean()),
        'loss': float(np.mean(losses)), 'raw_voltage_mse': float(np.mean(v_losses)),
        'raw_angle_mse': float(np.mean(a_losses)),
        'trajectory_voltage_rmse': np.sqrt(np.mean(ve ** 2, axis=1)).tolist(),
        'trajectory_angle_circular_rmse': np.sqrt(np.mean(ae ** 2, axis=1)).tolist(),
    }


def run(args, protocol):
    bootstrap(args.source_root)
    config = json.loads(args.base_config.read_text(encoding='utf-8'))
    arm = protocol['paired_arms'][args.arm]
    if set(protocol['paired_arms']) != {'corrected_gate','no_gate'}:
        raise ValueError('V3 requires exactly the predefined corrected_gate/no_gate pair')
    if protocol.get('initialization_policy') != 'respect_no_reinit_bias_preserve_original_xavier_rng':
        raise ValueError('V3 protected-bias policy was not predefined')
    expected = {'corrected_gate':('gat_mamba_full_gate','pre_message_dropout'),
                'no_gate':('gat_mamba_no_gate','none')}
    if (arm['variant'],arm['entropy_source']) != expected[args.arm]:
        raise ValueError('Arm variant/entropy mapping changed')
    if protocol['checkpoint_selection'] != 'normal_voltage_rmse_plus_angle_circular_rmse':
        raise ValueError('Checkpoint selection must retain the predefined normal-only rule')
    if arm['supervision'] != 'endpoint_only':
        raise ValueError('Minimal runner deliberately excludes the costly prefix grid')
    if args.seed not in protocol['model_seeds']:
        raise ValueError('Seed was not predefined')
    if protocol['sequence_length'] != config['data']['sequence_length']:
        raise ValueError('The study must retain the original T=30 window')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    cache = TrajectoryCache(args.cache)
    if cache.config != config:
        raise ValueError('Base config must match the immutable data cache exactly')
    train_arrays = cache.load_split('train')
    validation_arrays = cache.load_split('validation')
    # Deliberately no sealed access API or sealed cache load in this runner.
    for split, arrays in [('train', train_arrays), ('validation', validation_arrays)]:
        limit = protocol[f'{split}_trajectory_limit']
        if limit is not None:
            if int(limit) < 1:
                raise ValueError('Trajectory limits must be positive')
            subset = {key: value[:int(limit)] for key, value in arrays.items()}
            if split == 'train':
                train_arrays = subset
            else:
                validation_arrays = subset
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required but unavailable')
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    from preserve_mamba_init import preserve_mamba_initialization,summarize_preserved_biases
    initialization_repair_path = Path(inspect.getfile(preserve_mamba_initialization)).resolve()
    # The guard is installed before construction, when native dt biases exist.
    # It performs the complete old callback then restores only marked biases.
    with preserve_mamba_initialization() as initialization_check:
        model = build_formal_model(config, num_buses=cache.num_buses, num_pmu_buses=cache.pmu_count,
                                  pmu_bus_indices=cache.pmu_bus_indices, variant=arm['variant'], device=device)
    initialization_report = summarize_preserved_biases(model,initialization_check)
    if initialization_report['protected_bias_count'] < 2:
        raise RuntimeError('Bidirectional Mamba protected biases were not checked')
    initial_state_sha = state_hash(model)
    initial_common_state_sha = state_hash(model, exclude_gate=True)
    if arm['entropy_source'] == 'pre_message_dropout':
        install_pre_dropout_entropy(model)
    elif arm['entropy_source'] not in ('post_message_dropout', 'none'):
        raise ValueError('Unknown entropy source')
    if state_hash(model) != initial_state_sha:
        raise RuntimeError('Installing the entropy repair changed initial parameter values')
    if arm['loss_scaling'] == 'raw_units':
        loss_scales = (1.0, 1.0)
    else:
        raise ValueError('A loss-scaling stage needs a separate predefined protocol and implementation')
    edges = torch.as_tensor(cache.edge_index, dtype=torch.long, device=device)
    dataset = PairedDataset(train_arrays, cache, config, protocol, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(dataset, batch_size=protocol['batch_size'], shuffle=True, num_workers=0, generator=generator)
    validation_loaders = {
        name: DataLoader(PairedDataset(validation_arrays, cache, config, protocol, args.seed, name),
                         batch_size=protocol['batch_size'], shuffle=False, num_workers=0)
        for name in protocol['validation_scenarios']
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol['scheduler_horizon_epochs'], eta_min=1e-6)
    from normalized_entropy_gate import install_normalized_entropy_repair
    repair_path = Path(inspect.getfile(install_normalized_entropy_repair)).resolve()
    manifest = build_run_manifest(
        repo_root=args.source_root.resolve(), command=sys.argv,
        config={'base_cache_config': config, 'new_confirmatory_training_protocol': protocol},
        data_hashes=cache.manifest['data_hashes'],
        splits={'train':train_arrays['trajectory_seed'].astype(int).tolist(),
                'validation':validation_arrays['trajectory_seed'].astype(int).tolist()},
        mamba_backend='mamba_ssm',
    )
    manifest.update({
        'study_id': protocol['study_id'], 'status': 'RUNNING', 'arm': args.arm,
        'variant': arm['variant'],
        'model_seed': args.seed, 'source_root': str(args.source_root.resolve()),
        'base_config_sha256': sha256_file(args.base_config), 'protocol': protocol,
        'protocol_sha256': sha256_file(args.protocol), 'runner_sha256': sha256_file(__file__),
        'normalized_entropy_repair_path': str(repair_path),
        'normalized_entropy_repair_sha256': sha256_file(repair_path),
        'initialization_policy': protocol['initialization_policy'],
        'initialization_repair_path': str(initialization_repair_path),
        'initialization_repair_sha256': sha256_file(initialization_repair_path),
        'initialization_check': initialization_report,
        'cache_manifest_sha256': sha256_file(cache.manifest_path),
        'initial_state_sha256': initial_state_sha,
        'initial_non_gate_state_sha256': initial_common_state_sha,
        'parameter_count': parameter_count(model), 'device': str(device),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'unique_train_trajectories': len(dataset), 'unique_supervised_trajectory_time_pairs': len(dataset),
        'supervised_time_index': 29,
        'validation_trajectories': len(validation_arrays['trajectory_seed']),
        'sealed_test_accessed': False, 'result_type': protocol.get('result_type','validation_only_pilot'),
        'original_data_generation_git': cache.manifest.get('git'),
        'determinism': {
            'algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'bitwise_reproducibility_established': False,
            'note': 'Fixed seeds and paired data do not guarantee bitwise GPU reproducibility; warn-only kernel warnings must be retained in the job log.'
        },
        'source_hashes': {
            path: sha256_file(args.source_root/path) for path in (
                'graphmamba/graph_mamba_state_estimator.py', 'dsse/models.py',
                'dsse/runtime.py', 'dsse/feeder.py')
        },
    })
    write_json(output/'run_manifest.json', manifest)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    best_score, best_epoch, stale = math.inf, 0, 0
    history = []
    for epoch in range(1, protocol['epochs']+1):
        epoch_started = time.perf_counter()
        dataset.epoch = epoch
        model.train()
        records = []
        gate_grads = []
        order_digest = hashlib.sha256()
        input_digest = hashlib.sha256()
        for raw_batch in train_loader:
            order_digest.update(raw_batch['trajectory_seed'].numpy().tobytes())
            for key in ['scada','scada_mask','pmu','pmu_mask','v_true','angle_true']:
                input_digest.update(key.encode('ascii'))
                input_digest.update(raw_batch[key].contiguous().numpy().tobytes())
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(model, batch, edges)
            loss, lv, la = endpoint_loss(outputs, batch, loss_scales)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite paired training loss')
            loss.backward()
            backbone = getattr(model, 'paper_backbone', None)
            if backbone is not None and getattr(backbone, 'entropy_gate', False):
                gradient = backbone.entropy_gate_w.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError('Entropy slope has no finite gradient')
                gate_grads.append(gradient.detach().abs().cpu().tolist())
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
            optimizer.step()
            records.append([float(loss.detach().cpu()), float(lv.detach().cpu()), float(la.detach().cpu())])
        scheduler.step()
        validation = {name:evaluate(model, loader, edges, device, loss_scales) for name,loader in validation_loaders.items()}
        normal = validation['normal']
        score = normal['voltage_rmse'] + normal['angle_circular_rmse']
        backbone = getattr(model, 'paper_backbone', None)
        record = {
            'epoch': epoch, 'training_loss': np.mean(records, axis=0).tolist(),
            'training_loss_fields': ['optimized_loss','raw_voltage_mse','raw_angle_mse'],
            'validation': {name:{key:value for key,value in metrics.items() if not key.startswith('trajectory_')} for name,metrics in validation.items()},
            'selection_score': score, 'epoch_wall_seconds': time.perf_counter()-epoch_started,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'processed_label_presentations': len(dataset)*epoch,
            'unique_supervised_trajectory_time_pairs': len(dataset),
            'train_data_order_sha256': order_digest.hexdigest(),
            'train_corrupted_inputs_and_targets_sha256': input_digest.hexdigest(),
            'mean_abs_gate_slope_gradients': np.mean(gate_grads, axis=0).tolist() if gate_grads else None,
        }
        if backbone is not None and getattr(backbone,'entropy_gate',False):
            record.update(gate_w=backbone.entropy_gate_w.detach().cpu().tolist(), gate_b=backbone.entropy_gate_b.detach().cpu().tolist(),
                          entropy=backbone._last_entropy_stats, gate=backbone._last_gate_stats,
                          repaired_diagnostics=getattr(backbone,'_fresh_entropy_diagnostics',None))
        history.append(record)
        with (output/'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps({'arm':args.arm,'seed':args.seed,'epoch':epoch,'score':score,'seconds':record['epoch_wall_seconds']}), flush=True)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({'model_state_dict':model.state_dict(), 'arm':args.arm, 'entropy_source':arm['entropy_source'],
                        'variant':arm['variant'],
                        'config_sha256':canonical_sha256(cache.config),
                        'cache_manifest_sha256':sha256_file(cache.manifest_path),
                        'normalizer_sha256':cache.manifest['normalizer']['sha256'],
                        'topology_sha256':cache.manifest['topology']['sha256'],
                        'parameter_count':manifest['parameter_count'],
                        'model_seed':args.seed, 'best_epoch':epoch, 'protocol_sha256':manifest['protocol_sha256'],
                        'study_id':protocol['study_id'],
                        'initialization_policy':protocol['initialization_policy'],
                        'initialization_repair_sha256':sha256_file(initialization_repair_path),
                        'normalized_entropy_repair_sha256':sha256_file(repair_path),
                        'initial_non_gate_state_sha256':initial_common_state_sha,
                        'result_type':protocol.get('result_type','validation_only_pilot'),
                        'sealed_test_accessed':False}, output/'checkpoint.pt')
            write_json(output/'best_validation.json', validation)
        else:
            stale += 1
            if stale >= protocol['early_stopping_patience']:
                break
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_wall_time = time.perf_counter()-started
    final_report = {}
    if protocol.get('final_validation_report_scenarios'):
        best_checkpoint = torch.load(output/'checkpoint.pt',map_location=device,weights_only=True)
        model.load_state_dict(best_checkpoint['model_state_dict'],strict=True)
        for name in protocol['final_validation_report_scenarios']:
            loader = DataLoader(PairedDataset(validation_arrays,cache,config,protocol,args.seed,name),
                                batch_size=protocol['batch_size'],shuffle=False,num_workers=0)
            final_report[name] = evaluate(model,loader,edges,device,loss_scales)
        write_json(output/'final_checkpoint_validation.json',final_report)
    manifest.update(status='COMPLETED', epochs_completed=len(history), best_epoch=best_epoch, best_validation_score=best_score,
                    training_wall_time_seconds=training_wall_time,
                    final_checkpoint_validation_reported=bool(final_report),
                    peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None,
                    checkpoint_sha256=sha256_file(output/'checkpoint.pt'))
    write_json(output/'run_manifest.json', manifest)
    write_json(output/'training_result.json', dict(manifest=manifest, history=history))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('final_config_v3.json'))
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', choices=['corrected_gate','no_gate'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--print-plan', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if args.print_plan:
        print(json.dumps(protocol, indent=2))
        return
    if any(getattr(args,key) is None for key in ['source_root','base_config','cache','output','arm','seed']):
        parser.error('Training requires source-root, base-config, cache, output, arm, and seed')
    try:
        run(args, protocol)
    except BaseException as exc:
        # Preserve failures as study records. No seed replacement or retry.
        # Do not touch an existing nonempty job directory that was refused.
        output=args.output.resolve()
        manifest_path=output/'run_manifest.json'
        if output.exists() and not isinstance(exc,FileExistsError):
            failure={'status':'FAILED','arm':args.arm,'model_seed':args.seed,
                     'study_id':protocol['study_id'],'protocol_sha256':sha256_file(args.protocol),
                     'exception_type':type(exc).__name__,'exception':str(exc),
                     'sealed_test_accessed':False,'automatic_retry_allowed':False}
            write_json(output/'FAILED.json',failure)
            if manifest_path.exists():
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest.update(status='FAILED',failure=failure)
                write_json(manifest_path,manifest)
        raise


if __name__ == '__main__':
    main()
