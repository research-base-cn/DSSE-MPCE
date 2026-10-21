"""Model loading, sensor normalization and residual-readout inference."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,os,sys
import numpy as np

SEEDS=(131,197,269,349,443)
VARIANTS=('gat_mamba_no_gate','gat_mamba_full_gate','r4_adaptive','r4_quality_only','r4_constant',
          'gnn_se_snapshot','sequence_mlp','corrected_gcn_mamba','matched_gat_bilstm')
SENSORS=('scada','scada_mask','pmu','pmu_mask')
CASE_KEYS=('S0','S1','S2','S3','S4','S5','S7','A1','A2')+tuple('S6_L'+str(i) for i in range(8))
_CONFIGURED=False

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
def configure_torch():
    global _CONFIGURED
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name,'1')
    import torch
    if not _CONFIGURED:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        _CONFIGURED=True
    return torch

def state_hash(model,exclude_r3=False):
    d=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if exclude_r3 and any(part.startswith('r3_') for part in key.split('.')):continue
        a=value.detach().cpu().contiguous().numpy()
        d.update(key.encode());d.update(str(a.dtype).encode());d.update(str(a.shape).encode());d.update(a.tobytes())
    return d.hexdigest()

def module_from_path(path,name):
    path=Path(path).resolve()
    if name in sys.modules:
        m=sys.modules[name]
        if Path(m.__file__).resolve()!=path:raise RuntimeError('Conflicting source module: '+name)
        return m
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def get_head_linears(model):
    from torch import nn
    owner=getattr(model,'paper_backbone',model)
    def last(head):
        if isinstance(head,nn.Linear):return head
        if isinstance(head,nn.Sequential) and isinstance(head[-1],nn.Linear):return head[-1]
        raise ValueError('Expected actual bare or Sequential final Linear')
    v,a=last(owner.voltage_head),last(owner.angle_head)
    if v.out_features!=132 or a.out_features!=132:raise ValueError('Native output dimension changed')
    return owner,v,a

def set_heads(owner,voltage,angle):
    from torch import nn
    for name,value in (('voltage_head',voltage),('angle_head',angle)):
        head=getattr(owner,name)
        if isinstance(head,nn.Sequential):head[-1]=value
        else:setattr(owner,name,value)

def sensor_hash(items):
    d=hashlib.sha256()
    for key in ('pmu','pmu_mask','scada','scada_mask'):
        first=items[0][key];d.update(key.encode());d.update(str(first.dtype).encode());d.update(str((len(items),*first.shape)).encode())
        for item in items:d.update(np.ascontiguousarray(item[key]).tobytes())
    return d.hexdigest()

def numpy_readout(features,weights,view='raw_plus_H'):
    """Independent FP64 mathematical reference using SAME actual forward H/z."""
    result=[]
    for kind,hkey,zkey in (('voltage','H_voltage','voltage_logit'),('angle','H_angle','angle_logit')):
        h=features[hkey];raw=features['raw_final']
        x=h if view=='H_only' else raw if view=='raw_final' else np.c_[raw,h]
        z=features[zkey].astype(np.float64)+x.astype(np.float64)@weights[kind+'__W'].T+weights[kind+'__b']
        result.append(1.2/(1+np.exp(-z)) if kind=='voltage' else np.pi*np.tanh(z))
    return np.c_[result[0],result[1]]

class NeuralRuntime:
    def __init__(self,manifest,*,root=None,device='cuda:0',require_complete=True):
        self.manifest_path=Path(manifest).resolve();self.root=Path(root).resolve() if root else self.manifest_path.parent.parent
        self.catalog=read(self.manifest_path);self.device=device;self.verified=set()
        if self.catalog['schema_version']!=1:raise ValueError('Unsupported catalog version')
        self.source=self.directory(self.catalog['source_root'])
        for record in self.catalog['source_files']:self.asset(record)
        self.config=read(self.asset(self.catalog['config']));self.normalizer=read(self.asset(self.catalog['normalizer']))
        t=self.catalog['topology'];self.num_buses=int(t['num_buses']);self.pmu_bus_indices=list(map(int,t['pmu_bus_indices']))
        self.edge_index=np.asarray(t['edge_index'],dtype=np.int64)
        if self.num_buses!=132 or len(self.pmu_bus_indices)!=12 or len(set(self.pmu_bus_indices))!=12:
            raise ValueError('Final132-bus/12-station topology required')
        if self.edge_index.ndim!=2 or self.edge_index.shape[0]!=2 or self.edge_index.min()<0 or self.edge_index.max()>=132:
            raise ValueError('Invalid final directed edge array')
        for name in ('scada_mean','scada_scale','pmu_mean','pmu_scale'):
            a=np.asarray(self.normalizer[name]);expected=3 if name.startswith('scada') else 2
            if a.shape!=(expected,) or not np.isfinite(a).all() or ('scale' in name and np.any(a<=0)):
                raise ValueError('Bad frozen channel normalization: '+name)
        im=self.catalog['imputation_mean']
        with np.load(self.asset(im),allow_pickle=False) as f:self.mean=f[im.get('key','training_measurement_channel_mean')].copy()
        if self.mean.shape!=(288,) or not np.isfinite(self.mean).all():raise ValueError('Invalid raw288 training mean')
        self.parents=self.catalog['parents'];keys=[(x['variant'],int(x['seed'])) for x in self.parents]
        if len(set(keys))!=len(keys) or any(v not in VARIANTS or s not in SEEDS for v,s in keys):raise ValueError('Unexpected/duplicate parent identity')
        if require_complete and set(keys)!={(v,s) for v in VARIANTS for s in SEEDS}:raise ValueError('Require all final45 parents')
        for entry in self.parents:
            self.asset(entry['checkpoint'])
            if not re_sha(entry['expected_state_sha256']):raise ValueError('Missing full native parent state identity')
            gate=entry.get('gate',{'kind':'none'})
            expected_gate=('r4',entry['variant'].removeprefix('r4_')) if entry['variant'].startswith('r4_') else ('global_corrected',None) if entry['variant']=='gat_mamba_full_gate' else ('none',None)
            if (gate['kind'],gate.get('mode') if gate['kind']=='r4' else None)!=expected_gate:
                raise ValueError('Variant and final non-state gate mode differ: '+entry['variant'])
            if gate['kind']=='r4':self.asset(gate['calibration'])
            for arm,spec in entry['arms'].items():
                if spec['view'] not in ('raw_plus_H','raw_final','H_only'):raise ValueError('Invalid residual route')
                self.asset(spec['weights'])
            if 'main' not in entry['arms']:raise ValueError('Every final parent requires main readout')
        for record in self.catalog['helpers'].values():self.asset(record)
        self.cases={c['case_key']:c for c in self.catalog.get('cases',[])}
        if len(self.cases)!=len(self.catalog.get('cases',[])):raise ValueError('Duplicate case key')
        self.complete=set(keys)=={(v,s) for v in VARIANTS for s in SEEDS}

    def directory(self,path):
        p=(self.root/path).resolve()
        if Path(path).is_absolute() or not p.is_relative_to(self.root) or not p.is_dir():raise ValueError('Invalid root-relative source directory')
        return p

    def asset(self,record):
        name=record['path'];p=(self.root/name).resolve()
        if Path(name).is_absolute() or not p.is_relative_to(self.root) or not p.is_file():raise ValueError('Missing/unsafe final payload asset: '+name)
        if not re_sha(record['sha256']):raise ValueError('Invalid asset SHA256')
        identity=(str(p),record['sha256'])
        if identity not in self.verified:
            if sha(p)!=record['sha256']:raise RuntimeError('Final payload SHA mismatch: '+name)
            self.verified.add(identity)
        return p

    def parent(self,variant,seed):
        return next(x for x in self.parents if x['variant']==variant and int(x['seed'])==int(seed))

    def coefficients(self,record):
        with np.load(self.asset(record),allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        for kind in ('voltage','angle'):
            for suffix in ('W','b'):
                value=a[kind+'__'+suffix]
                if value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Final coefficients must be finite FP64')
        return a

    def load_parent(self,entry):
        torch=configure_torch()
        if self.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Compatible actual CUDA required')
        sys.path.insert(0,str(self.source))
        from dsse.models import build_formal_model
        import dsse.models as fm
        if not Path(fm.__file__).resolve().is_relative_to(self.source):raise RuntimeError('Conflicting native model source')
        hp=self.catalog['helpers'];preserve=module_from_path(self.asset(hp['preserve_init']),'preserve_mamba_init')
        variant=entry['variant'];gate=entry.get('gate',{'kind':'none'})
        native='gat_mamba_no_gate' if variant.startswith('r4_') else 'gcn_mamba' if variant=='corrected_gcn_mamba' else 'gat_lstm' if variant=='matched_gat_bilstm' else variant
        torch.manual_seed(int(entry['seed']))
        with preserve.preserve_mamba_initialization():
            model=build_formal_model(self.config,num_buses=132,num_pmu_buses=12,pmu_bus_indices=self.pmu_bus_indices,variant=native,device=torch.device('cpu'))
        if variant=='matched_gat_bilstm':
            b=model.paper_backbone;b.entropy_gate=False
            for name in ('entropy_gate_w','entropy_gate_b'):
                if hasattr(b,name):delattr(b,name)
            b.temporal_lstm=torch.nn.LSTM(160,154,num_layers=1,batch_first=True,dropout=0.,bidirectional=True)
            b.temporal_proj=torch.nn.Linear(308,160);b.n_temporal_layers=1
            if b.gnn_type!='gatv2' or b.use_mamba or b.use_transformer or sum(p.numel() for p in model.parameters())!=884298:
                raise RuntimeError('Final one-layer gate-off h154 BiLSTM structure differs')
        if variant=='corrected_gcn_mamba':
            b=model.paper_backbone
            if b.gnn_type!='gcn' or not b.use_mamba or not b.bidirectional_mamba or b.entropy_gate or sum(p.numel() for p in model.parameters())!=805226:
                raise RuntimeError('Final three-layer gate-off GCN-Mamba structure differs')
        if gate['kind'] in ('global_corrected','r4'):
            normal=module_from_path(self.asset(hp['normalized_gate']),'normalized_entropy_gate')
            if gate['kind']=='global_corrected':normal.install_normalized_entropy_repair(model,slope_mode='free')
            else:
                calibration=read(self.asset(gate['calibration']))
                r3=module_from_path(self.asset(hp['r3_gate']),'quality_entropy_group_gate')
                arrays={k:torch.tensor(calibration[k],dtype=torch.float32) for k in ('entropy_center','entropy_scale','quality_center','quality_scale')}
                r3.install_quality_entropy_gate(model,mode=gate['mode'],capture=False,**arrays)
                anchor=module_from_path(self.asset(hp['coverage_anchor']),'coverage_anchored_gate');anchor.install_coverage_anchor(model)
        elif gate['kind']!='none':raise ValueError('Unknown final gate kind')
        checkpoint=torch.load(self.asset(entry['checkpoint']),map_location='cpu',weights_only=True)
        if checkpoint.get('model_seed')!=int(entry['seed']):raise RuntimeError('Final checkpoint seed differs from catalog')
        state=checkpoint[entry['checkpoint'].get('state_key','model_state_dict')]
        model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
        if state_hash(model)!=entry['expected_state_sha256']:raise RuntimeError('Final native parent state differs')
        owner,v,a=get_head_linears(model)
        model=model.to(torch.device(self.device));model.eval()
        if state_hash(model)!=entry['expected_state_sha256']:raise RuntimeError('Device transfer changed parent state')
        return model,dict(variant=variant,seed=entry['seed'],checkpoint_sha256=entry['checkpoint']['sha256'],
            full_state_sha256=state_hash(model),body_state_sha256=state_hash(model,True),voltage_H_dimension=v.in_features,angle_H_dimension=a.in_features,
            original_parameter_count=sum(p.numel() for p in model.parameters()))

    def batch(self,items):
        torch=configure_torch();normalized=[];raw=[]
        for item in items:
            one={}
            for value,mask,mean,scale in (('scada','scada_mask','scada_mean','scada_scale'),('pmu','pmu_mask','pmu_mean','pmu_scale')):
                # Exact original apply_normalizer order, without importing the
                # dataset generator or opening any train/test cache.
                z=(item[value]-np.asarray(self.normalizer[mean],dtype=np.float64))/np.asarray(self.normalizer[scale],dtype=np.float64)
                one[value]=np.where(item[mask]>.5,z,0.).astype(np.float32)
                one[mask]=item[mask].astype(np.float32)
            steps=one['scada'].shape[0]
            normalized.append({k:torch.as_tensor(one[k].reshape(steps,-1),dtype=torch.float32) for k in SENSORS})
            y=np.r_[item['scada'][-1,:,:2].reshape(-1),item['pmu'][-1].reshape(-1)].astype(np.float64)
            m=np.r_[item['scada_mask'][-1,:,:2].reshape(-1),item['pmu_mask'][-1].reshape(-1)]>.5
            raw.append(np.where(m,y,self.mean).astype(np.float32))
        return {k:torch.stack([x[k] for x in normalized]).to(self.device) for k in SENSORS},torch.as_tensor(np.stack(raw),device=self.device)

    def load_case(self,key):
        record=self.cases[key]
        with np.load(self.asset(record),allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        if set(a)!=set(SENSORS)|{'targets','trajectory_seed','trajectory_index','layout_index'}:raise ValueError('Unexpected final raw-case keys')
        n=len(a['trajectory_seed'])
        for name,tail in (('scada',(30,132,3)),('scada_mask',(30,132,3)),('pmu',(30,12,2)),('pmu_mask',(30,12,2))):
            if a[name].shape!=(n,*tail) or not np.issubdtype(a[name].dtype,np.floating) or not np.isfinite(a[name]).all():raise ValueError('Invalid final sensor array')
            if name.endswith('_mask') and not np.isin(a[name],(0,1)).all():raise ValueError('Nonbinary mask')
        if a['targets'].shape!=(n,264) or not np.isfinite(a['targets']).all():raise ValueError('Invalid scoring targets')
        sensors=[{k:a[k][i] for k in SENSORS} for i in range(n)];digest=sensor_hash(sensors)
        if record.get('sensor_sha256') and digest!=record['sensor_sha256']:raise RuntimeError('Final case sensor identity differs')
        return dict(arrays=a,sensors=sensors,sensor_sha256=digest)

    def predict(self,entry,sensor_items,*,save_features=False,arms_override=None,intervention=None,include_zero=False):
        torch=configure_torch()
        from .residual_readout import StableResidualReadout
        if not sensor_items or any(set(x)!=set(SENSORS) for x in sensor_items):raise ValueError('Only four sensor arrays may enter prediction')
        model,identity=self.load_parent(entry);owner,originalV,originalA=get_head_linears(model)
        initial=state_hash(model);restore=None;hooks=[];capture={};features=None
        if intervention is not None:restore=apply_gate_intervention(model,intervention)
        before=state_hash(model);edges=torch.as_tensor(self.edge_index,dtype=torch.long,device=self.device)
        if save_features:
            def vh(module,args,value):capture.update(H_voltage=args[0].detach().cpu().numpy().copy(),voltage_logit=value.detach().cpu().numpy().copy())
            def ah(module,args,value):capture.update(H_angle=args[0].detach().cpu().numpy().copy(),angle_logit=value.detach().cpu().numpy().copy())
            hooks=[originalV.register_forward_hook(vh),originalA.register_forward_hook(ah)]
        def execute(wv=None,wa=None,collect=False):
            values=[];pieces={k:[] for k in ('raw_final','H_voltage','H_angle','voltage_logit','angle_logit')}
            for start in range(0,len(sensor_items),64):
                sensors,raw=self.batch(sensor_items[start:start+64])
                if wv is not None:wv.set_raw_measurements(raw)
                if wa is not None:wa.set_raw_measurements(raw)
                with torch.no_grad():out=model(sensors['scada'],sensors['pmu'],edges,scada_mask=sensors['scada_mask'],pmu_mask=sensors['pmu_mask'])
                prediction=torch.cat([out['V_pred'],out['delta_pred']],dim=-1).cpu().numpy().copy()
                if prediction.shape!=(len(raw),264) or prediction.dtype!=np.float32 or not np.isfinite(prediction).all():raise RuntimeError('Invalid actual native output')
                values.append(prediction)
                if collect:
                    pieces['raw_final'].append(raw.cpu().numpy().copy())
                    for k in capture:pieces[k].append(capture[k])
            return np.concatenate(values),{k:np.concatenate(v) for k,v in pieces.items()} if collect else None
        predictions={}
        try:
            predictions['baseline'],_=execute()
            specs=entry['arms'] if arms_override is None else arms_override
            specs=dict(specs)
            if include_zero:specs['zero']=dict(view='raw_plus_H',zero=True)
            for arm,spec in specs.items():
                wv=StableResidualReadout(originalV,view=spec['view']);wa=StableResidualReadout(originalA,view=spec['view'])
                wv.eval();wa.eval();wv.requires_grad_(False);wa.requires_grad_(False)
                if not spec.get('zero'):
                    coefficients=self.coefficients(spec['weights'])
                    for kind,wrapper in (('voltage',wv),('angle',wa)):
                        wrapper.load_fitted_raw_affine(torch.as_tensor(coefficients[kind+'__W'],device=self.device),torch.as_tensor(coefficients[kind+'__b'],device=self.device))
                set_heads(owner,wv,wa)
                try:predictions[arm],same_forward=execute(wv,wa,collect=save_features and arm=='main')
                finally:set_heads(owner,originalV,originalA)
                if save_features and arm=='main':features=same_forward
                if state_hash(model)!=before:raise RuntimeError('Readout failed to restore original parent')
            identity.update(parent_restored=True,native_batch_size=64,autocast=False,TF32=False)
            if include_zero:identity['whole_output_zero_bitexact']=predictions['zero'].tobytes()==predictions['baseline'].tobytes()
            return predictions,features,identity
        finally:
            set_heads(owner,originalV,originalA)
            for hook in hooks:hook.remove()
            if restore is not None:restore()
            if state_hash(model)!=initial:raise RuntimeError('Final model/gate restoration failed')
            del model
            if self.device.startswith('cuda'):torch.cuda.empty_cache()

def re_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def apply_gate_intervention(model,spec):
    """Same final fixed-head gate switches; the coverage anchor stays installed."""
    import torch
    b=model.paper_backbone
    if not getattr(b,'_coverage_anchor_installed',False):raise ValueError('Gate intervention requires an R4 parent')
    mode=b._r3_mode;w=b.r3_gate_weight.detach().clone();bias=b.r3_gate_bias.detach().clone()
    if spec['kind']=='entropy_centered':b._r3_mode='quality_only'
    elif spec['kind']=='fixed_unanchored_gain':
        q=torch.as_tensor(spec['q'],device=b.r3_gate_bias.device,dtype=b.r3_gate_bias.dtype).expand_as(b.r3_gate_bias)
        if not bool(torch.isfinite(q).all() and (q>0).all() and (q<2).all()):raise ValueError('Fixed q must be in(0,2)')
        with torch.no_grad():b.r3_gate_weight.zero_();b.r3_gate_bias.copy_(torch.log(q/(2-q)))
        b._r3_mode='constant'
    else:raise ValueError('Unknown fixed gate intervention')
    def restore():
        with torch.no_grad():b.r3_gate_weight.copy_(w);b.r3_gate_bias.copy_(bias)
        b._r3_mode=mode
    return restore
