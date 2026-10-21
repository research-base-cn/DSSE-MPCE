"""Load simulation sequences and their normalization parameters."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simulation'))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def scientific_runtime(root=ROOT):
    root=Path(root).resolve();sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(root/'simulation'))
    from dsse import runtime as formal_runtime
    return formal_runtime
def load_cache(root=ROOT):
    return scientific_runtime(root).TrajectoryCache(Path(root)/'data/training')
def load_sequences(split,root=ROOT):
    root=Path(root).resolve()
    if split in ('train','validation'):return load_cache(root).load_split(split)
    if split!='confirmation':raise ValueError('Choose train, validation or confirmation')
    manifest=read(root/'data/confirmation_clean/manifest.json');chunks=[]
    sys.path.insert(0,str(root/'src'))
    from dsse.datasets import read_shard
    for r in manifest['shards']:
        path=root/'data/confirmation_clean'/r['relative_path']
        if sha(path)!=r['sha256']:raise ValueError('Confirmation shard checksum mismatch')
        chunks.append(read_shard(path))
    arrays={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if arrays['trajectory_seed'].tolist()!=list(range(19000000,19000512)):raise ValueError('Confirmation identity changed')
    return arrays
