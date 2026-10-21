"""Shared data loading and training initialization."""
from pathlib import Path
import subprocess,sys
from neural.model_loader import NeuralRuntime,configure_torch,module_from_path,sha,write

HERE=Path(__file__).resolve().parent

def source_release_git_state(native_git_state,repo_root):
    """Use file identities only for an explicitly non-Git source release."""
    try:
        return native_git_state(repo_root)
    except subprocess.CalledProcessError as exc:
        message=((exc.stderr or '')+' '+(exc.stdout or '')).lower()
        if exc.returncode!=128 or 'not a git repository' not in message:
            raise
        source=Path(repo_root).resolve()
        files={p.relative_to(source).as_posix():sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and p.suffix.lower() in ('.py','.json','.dss','.dat','.txt')}
        return dict(available=False,identity_kind='SOURCE_RELEASE_FILE_SHA256',commit=None,branch=None,dirty=None,
            status_porcelain=None,source_files_sha256=files,
            qualification='No Git repository in the source release; unavailable Git metadata is replaced by exact file identities, without initializing a repository.')

def install_portable_bootstrap(shared):
    """Reapply reporting-only cache compatibility after EVERY native bootstrap.

    The v3 run() bootstraps a second time. Its original bootstrap is retained
    exactly, including all scientific imports; the thin cache adapter is then
    installed again. No source module/file or cached array is changed.
    """
    native_bootstrap=getattr(shared,'_original_bootstrap',shared.bootstrap)
    shared._original_bootstrap=native_bootstrap
    def portable_bootstrap(source_root):
        native_bootstrap(source_root)
        from dsse import reproducibility as repro
        native_git=getattr(repro,'_original_git_state',repro.git_state)
        repro._original_git_state=native_git
        repro.git_state=lambda repo_root:source_release_git_state(native_git,repo_root)
        native_cache=shared.TrajectoryCache
        class PortableTrajectoryCache(native_cache):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                if 'data_hashes' not in self.manifest:
                    self.manifest['data_hashes']={record['relative_path']:record['sha256']
                        for split in ('train','validation') for record in self.manifest['shards'][split]}
        shared.TrajectoryCache=PortableTrajectoryCache
    shared.bootstrap=portable_bootstrap

def load_training_context(catalog,device='cuda:0'):
    runtime=NeuralRuntime(catalog,device=device)
    configure_torch()
    sys.path[:0]=[str(HERE/'engines'),str(runtime.root/'src/gating'),str(runtime.source)]
    shared=module_from_path(HERE/'engines/backbone_trainer.py','training_engine')
    install_portable_bootstrap(shared)
    shared.bootstrap(runtime.source)
    # The portable cache manifest keeps the verified shard leaves rather than
    # the historical duplicate data_hashes reporting map. Restore that map in
    # memory for the unchanged original runner's provenance writer. Scientific
    # arrays/configuration and the delivered manifest bytes stay unchanged.
    cache=shared.TrajectoryCache(runtime.root/'data/training')
    if cache.config!=runtime.config:raise RuntimeError('Training cache and inference configuration differ')
    train=cache.load_split('train');validation=cache.load_split('validation')
    if train['trajectory_seed'].tolist()!=list(range(1100000,1102048)) or validation['trajectory_seed'].tolist()!=list(range(2100000,2100256)):
        raise RuntimeError('Original train2048/validation256 identities differ')
    return runtime,shared,runtime.source,cache,train,validation

def input_bindings(runtime,cache):
    paths=[cache.manifest_path,cache.cache_dir/'config.resolved.json',cache.cache_dir/'normalizer.json']
    for split in ('train','validation'):
        paths += [cache.cache_dir/r['relative_path'] for r in cache.manifest['shards'][split]]
    paths += [runtime.asset(x) for x in runtime.catalog['source_files']]
    paths += [runtime.asset(x) for x in runtime.catalog['helpers'].values()]
    paths += sorted(HERE.rglob('*.py'))+sorted((runtime.root/'configs').glob('*.json'))
    return [{'path':p.relative_to(runtime.root).as_posix(),'sha256':sha(p)} for p in paths]

def training_views(arrays,cache,shared,count):
    from simulation.case_definitions import make_training_views
    return make_training_views(arrays,cache,shared,count)
