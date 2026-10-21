"""Respect marked Mamba Linear biases during the paper's original initializer.

This is an isolated runtime construction guard. Frozen source is not edited.
The original initializer still performs every Xavier/other operation and uses
the identical RNG sequence. Only a Linear.bias marked `_no_reinit` is restored
in place to the value produced by that module's native constructor.
"""
from contextlib import contextmanager
import hashlib
import inspect


@contextmanager
def preserve_mamba_initialization(backbone_class=None):
    """Construct models inside this context; the class callback is restored.

    Accepting an explicit class permits meaningful CPU-only tests without a
    Mamba CUDA forward. The actual runner uses PaperGATMambaBackbone. This
    context must be installed before build_formal_model, never after its bias
    values have already been overwritten.
    """
    import torch
    import torch.nn as nn
    if backbone_class is None:
        from graphmamba.graph_mamba_state_estimator import PaperGATMambaBackbone
        backbone_class=PaperGATMambaBackbone
    descriptor=inspect.getattr_static(backbone_class,'_init_weights')
    if not isinstance(descriptor,staticmethod):
        raise TypeError('The original paper initializer must be a staticmethod')
    original=backbone_class._init_weights
    check={'initializer_call_count':0,'protected_bias_events':[],
           'frozen_source_mutated':False,'original_initializer_called':True,
           'scope':'Restore marked Linear biases only; retain all original Xavier/other operations.'}

    def guarded(module):
        check['initializer_call_count']+=1
        bias=module.bias if isinstance(module,nn.Linear) else None
        protected=bias is not None and bool(getattr(bias,'_no_reinit',False))
        snapshot=bias.detach().clone() if protected else None
        # Exactly the original callback and therefore the same random draws.
        original(module)
        if protected:
            with torch.no_grad():
                bias.copy_(snapshot)
            check['protected_bias_events'].append({'module_id':id(module),'native_bias':snapshot})

    backbone_class._init_weights=staticmethod(guarded)
    try:
        yield check
    finally:
        # Restore the exact descriptor even on constructor exceptions or nesting.
        backbone_class._init_weights=descriptor


def tensor_sha256(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    digest=hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def summarize_preserved_biases(model,check):
    """Check native snapshot equality and report the initial softplus scale."""
    import torch
    import torch.nn.functional as F
    names={id(module):name for name,module in model.named_modules()}
    modules={id(module):module for module in model.modules()}
    rows=[]
    for event in check['protected_bias_events']:
        identity=event['module_id']
        if identity not in modules:
            raise RuntimeError('A protected constructor module disappeared before check')
        bias=modules[identity].bias
        original=event['native_bias']
        same=torch.equal(bias.detach().cpu(),original.detach().cpu())
        if not same:
            raise RuntimeError('Protected native bias did not survive model construction')
        values=bias.detach()
        steps=F.softplus(values)
        rows.append({'name':names[identity]+'.bias','numel':values.numel(),
                     'native_snapshot_preserved_exactly':same,
                     'native_bias_sha256':tensor_sha256(original),'constructed_bias_sha256':tensor_sha256(values),
                     'bias_min':float(values.min().cpu()),'bias_max':float(values.max().cpu()),
                     'softplus_bias_min':float(steps.min().cpu()),'softplus_bias_max':float(steps.max().cpu()),
                     'requires_grad':bool(bias.requires_grad)})
    return {'initializer_call_count':check['initializer_call_count'],
            'protected_bias_count':len(rows),'protected_bias_scalar_count':sum(r['numel'] for r in rows),
            'protected_biases':rows,'frozen_source_mutated':False,
            'scope':check['scope'],'native_bias_convention_respected':True}
