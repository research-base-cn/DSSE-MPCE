"""Ordered model registration for snapshot and sequence baselines."""
from copy import deepcopy
import hashlib,json
from neural.model_loader import sha

CONFIG_CONTENT_SHA256='5cd1801e8e5c4783f3f90bbb7977585b003f66f0a1e36bbfed853d9aabb0075f'
SOURCE_SHA256={
    'dsse/models.py':'84482cf3f9e1af037274b7998174eda3f023639931188824179cd7b13f4c3028',
    'dsse/training.py':'96c369aba0524a72d8d82ac40811011f85cba30426083e0814cd01d9204af02c',
}
REGISTRY_ORDER=('gat_mamba_full_gate','gat_mamba_no_gate','gcn_mamba','gat_lstm','gnn_se_snapshot','sequence_mlp')

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()

def training_time_config(runtime,variant):
    if variant not in ('gnn_se_snapshot','sequence_mlp'):raise ValueError('This adapter only applies to the two unchanged historical controls')
    for relative,expected in SOURCE_SHA256.items():
        if sha(runtime.source/relative)!=expected:raise RuntimeError('Original formal source identity differs: '+relative)
    original=runtime.config
    if canonical_sha(original)!=CONFIG_CONTENT_SHA256:raise RuntimeError('Canonical training configuration differs from actual historical controls')
    if set(original['neural_variants'])!=set(REGISTRY_ORDER):raise RuntimeError('Registry membership changed; order-only compatibility cannot apply')
    ordered=deepcopy(original)
    ordered['neural_variants']={name:ordered['neural_variants'][name] for name in REGISTRY_ORDER}
    if ordered!=original or canonical_sha(ordered)!=CONFIG_CONTENT_SHA256:raise RuntimeError('Order compatibility changed scientific configuration')
    return ordered,dict(scope='snapshot/MLP formal registry insertion order only; original validator remains enabled',
        original_registry_key_order=list(original['neural_variants']),training_registry_key_order=list(REGISTRY_ORDER),
        original_canonical_config_sha256=CONFIG_CONTENT_SHA256,training_canonical_config_sha256=canonical_sha(ordered),
        all_configuration_values_unchanged=True,source_sha256=SOURCE_SHA256)
