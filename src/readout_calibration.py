"""Unchanged final train-only residual ridge fit; no selection or evaluation."""

def fit_residual(packs,kind,alpha=1e-4,std_floor=1e-6):
    import numpy as np
    feature='H_voltage' if kind=='voltage' else 'H_angle'
    logit='voltage_logit' if kind=='voltage' else 'angle_logit'
    x=np.concatenate([np.concatenate([p['raw_final'],p[feature]],axis=1).astype(np.float64) for p in packs])
    y=np.concatenate([p['targets'][:,:132] if kind=='voltage' else p['targets'][:,132:] for p in packs]).astype(np.float64)
    if kind=='voltage':
        if not np.all((y>0)&(y<1.2)):
            raise ValueError('True training V outside preserved sigmoid link')
        y=np.log(y/(1.2-y))
    else:
        if not np.all(np.abs(y/np.pi)<1):
            raise ValueError('True training theta outside preserved tanh link')
        y=np.arctanh(y/np.pi)
    y-=np.concatenate([p[logit].astype(np.float64) for p in packs])
    mu=x.mean(axis=0);sd_raw=x.std(axis=0,ddof=0);sd=np.maximum(sd_raw,std_floor);xn=(x-mu)/sd;ym=y.mean(axis=0)
    gram=xn.T@xn/len(x)+alpha*np.eye(x.shape[1]);wstd=np.linalg.solve(gram,xn.T@(y-ym)/len(x))
    w=(wstd/sd[:,None]).T;b=ym-mu@w.T
    return dict(W=w,b=b,mean=mu,sd_raw=sd_raw,sd_used=sd,
        gram_eigenvalues=np.linalg.eigvalsh(gram),training_residual_mse=np.asarray(np.mean((x@w.T+b-y)**2)))
