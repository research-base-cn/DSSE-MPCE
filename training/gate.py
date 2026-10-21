"""Train the coverage-anchored gate with a fixed backbone."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys
import time

ARMS = ('adaptive', 'quality_only', 'constant')
SEEDS = (131, 197, 269, 349, 443)
METRICS = ('mean_trajectory_voltage_rmse', 'mean_trajectory_angle_circular_rmse')
TRAINING = dict(epochs=30, batch_size=16, learning_rate=.003,
                minimum_learning_rate=.0001, gradient_clip=1.,
                validation_every_epochs=5, train_trajectories=2048,
                validation_trajectories=128,
                augmentation_scenarios=['normal', 'missing_40pct', 'outliers_15pct',
                                        'regional_missing_40pct', 'pmu_dropout_50pct'])
SCENARIOS = ('normal', 'outliers_15pct', 'outliers_30pct', 'missing_40pct',
             'missing_60pct', 'regional_missing_40pct',
             'sensitivity_targeted_outliers_15pct', 'pmu_dropout_50pct')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def body_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if any(part.startswith('r3_') for part in key.split('.')):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def score(metrics, reference, scenarios):
    values = [metrics[s][m] / reference[s][m] for s in scenarios for m in METRICS]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Invalid selection metrics')
    return sum(values) / len(values)


def run(args):
    recipe_path = Path(__file__).resolve().parents[1]/'configs/gate.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf8'))
    if (any(recipe['training'].get(k) != v for k, v in TRAINING.items())
            or tuple(recipe['scenarios']) != SCENARIOS
            or tuple(recipe['model_seeds']) != SEEDS):
        raise RuntimeError('Delivered gate recipe differs from the original fixed scientific rule')
    # common exposes AST-preserved PairedDataset/evaluate/endpoint_loss and
    # loads only the supplied train/validation cache, never final cases.
    from training.common import load_training_context, input_bindings
    from neural.model_loader import configure_torch, module_from_path, state_hash
    runtime, shared, source, cache, train, validation = load_training_context(args.catalog, args.device)
    import numpy as np
    torch = configure_torch()
    from torch.utils.data import DataLoader
    shared.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Gate training requires the actual compatible CUDA/Mamba backend')
    torch.cuda.set_device(device)
    if args.output.exists():
        raise FileExistsError('Never overwrite a training output directory')
    args.output.mkdir(parents=True)
    if cache.config != runtime.config:
        raise RuntimeError('Delivered cache and model configuration differ')
    train = {k: v[:2048] for k, v in train.items()}
    validation = {k: v[:128] for k, v in validation.items()}
    if train['trajectory_seed'].tolist() != list(range(1100000, 1102048)):
        raise RuntimeError('The original 2048 training trajectory identities are required')
    if validation['trajectory_seed'].tolist() != list(range(2100000, 2100128)):
        raise RuntimeError('The original first128 development trajectory identities are required')
    entry = runtime.parent('gat_mamba_no_gate', args.seed)
    warm = runtime.asset(entry['checkpoint'])
    base, parent_identity = runtime.load_parent(entry)
    if args.warmstart is not None:
        warm = args.warmstart.resolve()
        if not warm.is_file() or not warm.is_relative_to(runtime.root):
            raise ValueError('--warmstart must be an existing checkpoint inside the repository')
        payload = torch.load(warm, map_location=device, weights_only=True)
        if payload.get('model_seed', payload.get('seed')) != args.seed:
            raise ValueError('New no-gate warmstart has a different model seed')
        if payload.get('variant') != 'gat_mamba_no_gate':
            raise ValueError('Gate warmstart must be a newly trained no-gate backbone')
        base.load_state_dict(payload['model_state_dict'], strict=True)
        parent_identity.update(checkpoint_sha256=sha(warm), full_state_sha256=state_hash(base),
                               body_state_sha256=state_hash(base, True),
                               origin='explicit_new_no_gate_checkpoint')
    else:
        parent_identity['origin'] = 'supplied_final_no_gate_checkpoint'
    hp = runtime.catalog['helpers']
    module_from_path(runtime.asset(hp['normalized_gate']), 'normalized_entropy_gate')
    r3 = module_from_path(runtime.asset(hp['r3_gate']), 'quality_entropy_group_gate')
    install_quality_entropy_gate = r3.install_quality_entropy_gate
    quality_calibration_from_samples = r3.quality_calibration_from_samples
    QUALITY_FEATURE_NAMES = r3.QUALITY_FEATURE_NAMES
    initial_body = body_sha(base)
    edges = torch.as_tensor(cache.edge_index, dtype=torch.long, device=device)
    t = dict(TRAINING)
    helper = {'augmentation_scenarios': t['augmentation_scenarios']}

    def loader(arrays, scenario, shuffle=False, generator=None):
        return DataLoader(shared.PairedDataset(arrays, cache, cache.config, helper, args.seed, scenario),
                          batch_size=t['batch_size'], shuffle=shuffle, generator=generator, num_workers=0)

    # These operations retain the original R3 train64 calibration and dtype
    # order; existing fitted gate calibration is deliberately NOT reused.
    calibration_arrays = {k: v[:64] for k, v in train.items()}
    calibration_loader = loader(calibration_arrays, 'normal')
    scale_metrics = shared.evaluate(base, calibration_loader, edges, device, (1., 1.))
    loss_scales = (max(math.sqrt(scale_metrics['raw_voltage_mse']), 1e-6),
                   max(math.sqrt(scale_metrics['raw_angle_mse']), 1e-6))
    probe = copy.deepcopy(base)
    install_quality_entropy_gate(probe, mode='adaptive', capture=True)
    probe.eval(); pieces = [[] for _ in probe.paper_backbone.gnn_layers]; qualities = []
    calibration_inputs = hashlib.sha256()
    with torch.inference_mode():
        for raw in calibration_loader:
            for k in ('scada', 'scada_mask', 'pmu', 'pmu_mask'):
                calibration_inputs.update(k.encode()); calibration_inputs.update(raw[k].contiguous().numpy().tobytes())
            shared.forward_model(probe, shared.move_batch(raw, device), edges)
            b = probe.paper_backbone
            qualities.append(b._last_r3_quality)
            for i, value in enumerate(b._last_r3_entropies):
                pieces[i].append(value)
    entropy_samples = [torch.cat(v).double() for v in pieces]
    ec = torch.stack([v.mean(0) for v in entropy_samples]).float()
    es = torch.stack([v.std(0, unbiased=False).clamp_min(.001) for v in entropy_samples]).float()
    qc, qs = quality_calibration_from_samples(torch.cat(qualities), std_floor=.05)
    calibration = {'train_seeds': calibration_arrays['trajectory_seed'].tolist(),
                   'input_sha256': calibration_inputs.hexdigest(),
                   'entropy_center': ec.tolist(), 'entropy_scale': es.tolist(),
                   'quality_center': qc.tolist(), 'quality_scale': qs.tolist(),
                   'quality_features': list(QUALITY_FEATURE_NAMES),
                   'loss_scales': list(loss_scales),
                   'loss_scale_source': 'train64_normal_warmstart_residual_RMS',
                   'warmstart_checkpoint_sha256': sha(warm)}
    write(args.output / 'calibration.json', calibration)
    del probe, pieces, qualities, entropy_samples
    val_loaders = {s: loader(validation, s) for s in SCENARIOS}
    reference = {s: shared.evaluate(base, l, edges, device, loss_scales) for s, l in val_loaders.items()}
    first = shared.move_batch(next(iter(val_loaders['normal'])), device)
    base.eval()
    with torch.inference_mode():
        original = shared.forward_model(base, first, edges)
    model = base
    install_quality_entropy_gate(model, mode=args.arm, entropy_center=ec, entropy_scale=es,
                                 quality_center=qc, quality_scale=qs, capture=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    b = model.paper_backbone
    b.r3_gate_bias.requires_grad_(True)
    b.r3_gate_weight.requires_grad_(args.arm != 'constant')
    if body_sha(model) != initial_body:
        raise RuntimeError('Gate installation changed inherited body tensors')
    if getattr(b, '_coverage_anchor_installed', False):
        raise RuntimeError('Training must use the unanchored R3 gate')
    model.eval()
    with torch.inference_mode():
        initial = shared.forward_model(model, first, edges)
    exact = all(torch.equal(original[k], initial[k]) for k in ('V_pred', 'delta_pred'))
    if not exact:
        raise RuntimeError('Zero initialized gate is not identical to the no-gate parent')
    trainable = [v for v in model.parameters() if v.requires_grad]
    trainable_names = [k for k, v in model.named_parameters() if v.requires_grad]
    if any(not k.endswith(('r3_gate_weight', 'r3_gate_bias')) for k in trainable_names):
        raise RuntimeError('Optimizer contains an inherited body parameter')
    registered_count = sum(v.numel() for v in trainable)
    if registered_count != (12 if args.arm == 'constant' else 84):
        raise RuntimeError('Unexpected gate optimizer size')
    manifest = dict(status='RUNNING_GATE_TRAINING', seed=args.seed, arm=args.arm,
                    epochs_requested=args.epochs, smoke=args.smoke,
                    scientific_recipe=dict(TRAINING, optimizer='AdamW', betas=[.9, .999],
                                           eps=1e-8, weight_decay=0., loss='V_MSE/sV^2+angle_MSE/sAngle^2',
                                           calibration_windows=64, dropout=False,
                                           anchor_during_training=False, checkpoint_selection='terminal'),
                    parent_identity=parent_identity, parent_checkpoint_sha256=sha(warm),
                    warmstart_input=dict(path=warm.relative_to(runtime.root).as_posix(), sha256=sha(warm)),
                    catalog_sha256=sha(args.catalog), runner_sha256=sha(__file__),
                    cache_manifest_sha256=sha(cache.manifest_path),
                    source_files=runtime.catalog['source_files'], helper_files=hp,
                    input_files=input_bindings(runtime, cache),
                    gate_recipe_sha256=sha(recipe_path),
                    training_entry_files=[dict(path=p.relative_to(runtime.root).as_posix(), sha256=sha(p))
                                          for p in (Path(__file__).resolve(), recipe_path.resolve(),
                                                    Path(shared.__file__).resolve(),
                                                    Path(__file__).with_name('common.py').resolve())],
                    calibration_sha256=sha(args.output / 'calibration.json'),
                    initial_body_sha256=initial_body, zero_gate_bitexact=exact,
                    train_seeds=train['trajectory_seed'].tolist(),
                    validation_seeds=validation['trajectory_seed'].tolist(),
                    optimizer_parameter_names=trainable_names,
                    registered_trainable_parameters=registered_count,
                    effective_trainable_parameters={'adaptive': 84, 'quality_only': 72, 'constant': 12}[args.arm],
                    gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                    no_confirmation_data_loaded=True, epochs_completed=0)
    write(args.output / 'run_manifest.json', manifest)
    optimizer = torch.optim.AdamW(trainable, lr=t['learning_rate'], betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    # Smoke takes only the first epoch of the unchanged 30-epoch schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'], eta_min=t['minimum_learning_rate'])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(train, None, shuffle=True, generator=generator)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter(); model.eval(); train_loader.dataset.epoch = epoch
        order, input_hash = hashlib.sha256(), hashlib.sha256(); losses = []
        for raw in train_loader:
            order.update(raw['trajectory_seed'].numpy().tobytes())
            for key in ('scada', 'scada_mask', 'pmu', 'pmu_mask', 'v_true', 'angle_true'):
                input_hash.update(key.encode()); input_hash.update(raw[key].contiguous().numpy().tobytes())
            batch = shared.move_batch(raw, device); optimizer.zero_grad(set_to_none=True)
            out = shared.forward_model(model, batch, edges)
            loss, lv, la = shared.endpoint_loss(out, batch, loss_scales)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if args.arm == 'quality_only' and bool((b.r3_gate_weight.grad[..., -1] != 0).any()):
                raise RuntimeError('Quality-only entropy acquired gradient')
            if any(v.grad is not None for k, v in model.named_parameters() if k not in trainable_names):
                raise RuntimeError('Inherited body gradient')
            torch.nn.utils.clip_grad_norm_(trainable, t['gradient_clip']); optimizer.step()
            losses.append([float(loss.detach().cpu()), float(lv.detach().cpu()), float(la.detach().cpu())])
        scheduler.step()
        if body_sha(model) != initial_body:
            raise RuntimeError('Frozen body bytes changed')
        record = {'epoch': epoch, 'seconds': time.perf_counter() - tick,
                  'losses': np.mean(losses, axis=0).tolist(),
                  'order_sha256': order.hexdigest(), 'inputs_sha256': input_hash.hexdigest(),
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'gate_weight': b.r3_gate_weight.detach().cpu().tolist(),
                  'gate_bias': b.r3_gate_bias.detach().cpu().tolist()}
        if epoch % t['validation_every_epochs'] == 0:
            metrics = {s: shared.evaluate(model, l, edges, device, loss_scales) for s, l in val_loaders.items()}
            record.update(validation=metrics, descriptive_reference_ratio=score(metrics, reference, SCENARIOS))
        with (args.output / 'progress.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        manifest['epochs_completed'] = epoch
        write(args.output / 'run_manifest.json', manifest)
        print(json.dumps({'epoch': epoch, 'arm': args.arm, 'seed': args.seed,
                          'seconds': record['seconds']}), flush=True)
    torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - started
    terminal_state = state_hash(model)
    anchor = module_from_path(runtime.asset(hp['coverage_anchor']), 'coverage_anchored_gate')
    anchor.install_coverage_anchor(model)
    if state_hash(model) != terminal_state or body_sha(model) != initial_body:
        raise RuntimeError('Parameter-free anchor changed fitted tensors')
    model.requires_grad_(False); model.eval()
    checkpoint = dict(model_state_dict=model.state_dict(), model_seed=args.seed,
                      variant='r4_' + args.arm, arm=args.arm, epoch=args.epochs,
                      coverage_anchor_applied_after_training=True,
                      calibration_sha256=sha(args.output / 'calibration.json'),
                      scientific_recipe=manifest['scientific_recipe'],
                      warmstart_checkpoint_sha256=sha(warm), smoke=args.smoke)
    torch.save(checkpoint, args.output / 'terminal_checkpoint.pt')
    final = {s: shared.evaluate(model, l, edges, device, loss_scales) for s, l in val_loaders.items()}
    write(args.output / 'final_validation.json', final)
    manifest.update(status='COMPLETED_GATE_SMOKE' if args.smoke else 'COMPLETED_GATE_TRAINING',
                    checkpoint_sha256=sha(args.output / 'terminal_checkpoint.pt'),
                    final_validation_sha256=sha(args.output / 'final_validation.json'),
                    terminal_state_sha256=terminal_state, frozen_body_unchanged=True,
                    coverage_anchor_trained=False,
                    train_loop_seconds=loop_seconds,
                    train_plus_postprocessing_seconds=time.perf_counter() - started,
                    timer_scope='Loop includes periodic validation, hashing and progress writes; excludes loading/calibration. Postprocessing includes anchor installation, terminal save and anchored final development validation.')
    write(args.output / 'run_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('metadata/models.json'))
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--mode', '--arm', dest='arm', choices=ARMS, default='adaptive')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmstart', type=Path,
                        help='New same-seed no-gate checkpoint inside the repository; default uses supplied final parent')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        if args.epochs == 30:
            args.epochs = 1
        if args.epochs != 1:
            parser.error('--smoke permits exactly one epoch, not a changed training recipe')
    elif args.epochs != 30:
        parser.error('The final gate recipe requires 30 epochs; only --smoke permits one')
    args.catalog = args.catalog.resolve()
    args.output = args.output.resolve()
    run(args)


if __name__ == '__main__':
    main()
