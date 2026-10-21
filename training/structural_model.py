"""Build capacity-matched structural controls with paired initialization."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from neural.model_loader import sha,module_from_path,state_hash,get_head_linears,SEEDS
SOURCE_HASHES={
    'graphmamba/graph_mamba_state_estimator.py':'a5dd8575d5d8131f9390b7050c0f1e01200946bdc9eea6836d8c7bdfc9bbcb99',
    'dsse/models.py':'84482cf3f9e1af037274b7998174eda3f023639931188824179cd7b13f4c3028',
    'dsse/runtime.py':'fe923b70bdbd9b0f062a06246afcf94149890c760e650479789f803e19a43c63',
    'dsse/feeder.py':'035e21053c623f7599cdaa036da310bb6ce6fa18138760f09f390d1b84637a0b',
}
import sys
def build_structural_model(source,cache,variant,seed,device,width_policy):
    """Only replace the declared module; pair all unchanged initial tensors.

    Width selection is arithmetic on registered parameter counts, without any
    input/target access. Native full-window endpoint methods remain unchanged.
    The historical stable parent snapshot deliberately has no such constructor.
    """
    source=Path(source).resolve()
    for relative,expected in SOURCE_HASHES.items():
        if sha(source/relative)!=expected:raise RuntimeError('Structural source changed: '+relative)
    sys.path.insert(0,str(source))
    from dsse.models import build_formal_model
    from dsse.reproducibility import seed_everything
    init_path=Path(__file__).resolve().parents[1]/'src/gating/preserve_mamba_init.py'
    if sha(init_path)!='368defe0aa2487fb34e2fae3ae9a718bd80251e7dffb6b64bd238d9b0b07e94d':
        raise RuntimeError('Protected initialization helper changed')
    # Historical registry helpers may be an identical frozen copy under
    # project/source while actual modules live under extension/source.
    if 'preserve_mamba_init' in sys.modules:
        init=sys.modules['preserve_mamba_init']
        if sha(init.__file__)!=sha(init_path):raise RuntimeError('Conflicting protected initializer bytes')
    else:init=module_from_path(init_path,'preserve_mamba_init')
    if variant not in ('corrected_gcn_mamba','matched_gat_bilstm') or seed not in SEEDS:raise ValueError('Unplanned structural variant/seed')
    cfg=cache.config;modelcfg=cfg['model'];d=int(modelcfg['d_model'])
    if d!=160 or modelcfg['readout_mode']!='graph' or int(modelcfg['n_gnn_layers'])!=3:raise RuntimeError('Original structural dimensions changed')
    kwargs=dict(num_buses=cache.num_buses,num_pmu_buses=cache.pmu_count,pmu_bus_indices=cache.pmu_bus_indices,device=torch.device('cpu'))
    seed_everything(seed)
    with init.preserve_mamba_initialization() as reference_check:
        reference=build_formal_model(cfg,variant='gat_mamba_no_gate',**kwargs)
    reference_init=init.summarize_preserved_biases(reference,reference_check)
    if reference_init['protected_bias_count']<2:raise RuntimeError('Reference bidirectional Mamba bias protection failed')
    reference_count=sum(v.numel() for v in reference.parameters())
    if reference_count!=883466:raise RuntimeError('Original noGate registered parameter count changed')
    rng_cpu=torch.get_rng_state().clone()
    rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_everything(seed)
    native_variant='gcn_mamba' if variant=='corrected_gcn_mamba' else 'gat_lstm'
    with init.preserve_mamba_initialization() as target_check:
        model=build_formal_model(cfg,variant=native_variant,**kwargs)
    target_init=init.summarize_preserved_biases(model,target_check)
    backbone=model.paper_backbone
    excluded=('paper_backbone.gnn_layers.',) if variant=='corrected_gcn_mamba' else (
        'paper_backbone.mamba_layers.','paper_backbone.backward_mamba_layers.',
        'paper_backbone.temporal_norms.','paper_backbone.backward_temporal_norms.',
        'paper_backbone.temporal_lstm.','paper_backbone.temporal_proj.')
    width_rows=[];selected_width=None;removed_entropy_parameters=0
    if variant=='corrected_gcn_mamba':
        if backbone.gnn_type!='gcn' or not backbone.use_mamba or not backbone.bidirectional_mamba or backbone.entropy_gate:
            raise RuntimeError('GCN control native structure differs')
        if target_init['protected_bias_count']<2:raise RuntimeError('GCN Mamba protected biases absent')
    else:
        if backbone.gnn_type!='gatv2' or backbone.use_mamba or backbone.use_transformer:raise RuntimeError('Native GAT-LSTM construction changed')
        backbone.entropy_gate=False
        for name in ('entropy_gate_w','entropy_gate_b'):
            if hasattr(backbone,name):
                removed_entropy_parameters+=getattr(backbone,name).numel();delattr(backbone,name)
        if removed_entropy_parameters!=6:raise RuntimeError('Expected exactly six unused original gate scalars removed')
        if width_policy['minimum_width']!=1 or width_policy['maximum_width']!=320 or width_policy['tie_break']!='smaller_width':raise ValueError('Frozen LSTM width policy changed')
        fixed_count=sum(v.numel() for v in model.parameters())-sum(v.numel() for v in backbone.temporal_lstm.parameters())-sum(v.numel() for v in backbone.temporal_proj.parameters())
        for h in range(width_policy['minimum_width'],width_policy['maximum_width']+1):
            # Per direction:4h*d+4h*h+8h; two directions, plus Linear(2h,d).
            count=fixed_count+8*h*(d+h+2)+2*h*d+d
            width_rows.append(dict(width=h,registered_parameters=count,absolute_parameter_gap=abs(count-reference_count)))
        choice=min(width_rows,key=lambda r:(r['absolute_parameter_gap'],r['width']));selected_width=choice['width']
        if selected_width!=154 or choice['registered_parameters']!=884298:raise RuntimeError('Arithmetic LSTM capacity contract changed')
        if choice['absolute_parameter_gap']/reference_count>width_policy['maximum_reference_relative_gap']:raise RuntimeError('No permitted near-capacity one-layer LSTM')
        backbone.temporal_lstm=nn.LSTM(d,selected_width,num_layers=1,batch_first=True,dropout=0.,bidirectional=True)
        backbone.temporal_proj=nn.Linear(2*selected_width,d)
        backbone._init_weights(backbone.temporal_proj)
        backbone.n_temporal_layers=1
    target_state=model.state_dict();reference_state=reference.state_dict()
    target_common={k for k in target_state if not k.startswith(excluded)}
    reference_common={k for k in reference_state if not k.startswith(excluded)}
    if target_common!=reference_common:raise RuntimeError('Unchanged module state sets differ')
    with torch.no_grad():
        for name in sorted(target_common):
            if target_state[name].shape!=reference_state[name].shape:raise RuntimeError('Unchanged module tensor shape differs: '+name)
            target_state[name].copy_(reference_state[name])
    if any(not torch.equal(target_state[k],reference_state[k]) for k in target_common):raise RuntimeError('Paired unchanged initialization failed')
    common_digest=hashlib.sha256()
    for key in sorted(target_common):
        value=target_state[key].detach().cpu().contiguous().numpy()
        common_digest.update(key.encode());common_digest.update(str(value.dtype).encode());common_digest.update(str(value.shape).encode());common_digest.update(value.tobytes())
    count=sum(v.numel() for v in model.parameters())
    expected_count=805226 if variant=='corrected_gcn_mamba' else 884298
    if count!=expected_count:raise RuntimeError('Actual structural registered count differs from arithmetic contract')
    contract=dict(variant=variant,encoder_mode=backbone.encoder_mode,readout_mode=backbone.readout_mode,d_model=d,gnn_type=backbone.gnn_type,
        gnn_layers=len(backbone.gnn_layers),entropy_gate=backbone.entropy_gate,use_mamba=backbone.use_mamba,
        bidirectional_mamba=backbone.bidirectional_mamba,temporal_layers=backbone.n_temporal_layers,
        lstm_hidden_size=selected_width,lstm_bidirectional=True if selected_width else None,
        registered_parameters=count,reference_noGate_parameters=reference_count,relative_parameter_gap=abs(count-reference_count)/reference_count,
        voltage_H_dimension=get_head_linears(model)[1].in_features,angle_H_dimension=get_head_linears(model)[2].in_features)
    # Match the reference construction's subsequent dropout RNG, not the number
    # of random draws in the alternative module's temporary constructor.
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
    construction=dict(architecture_contract=contract,initial_state_sha256=state_hash(model),
        reference_initial_state_sha256=state_hash(reference),paired_common_state_sha256=common_digest.hexdigest(),
        paired_common_tensor_count=len(target_common),paired_common_tensors_bitexact=True,changed_module_prefixes=list(excluded),
        reference_initialization=reference_init,target_native_initialization_before_pairing=target_init,
        removed_unused_entropy_parameters=removed_entropy_parameters,lstm_width_candidates=width_rows,
        initialization_policy='native target construction; unchanged state tensors copied from same-seed actual protected noGate initialization; training RNG restored to post-reference construction',
        no_input_or_target_access=True,native_endpoint_methods_unchanged=True)
    del reference
    model=model.to(torch.device(device))
    return model,construction
