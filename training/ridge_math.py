"""Unchanged fixed-ridge function extracted from the actual final fitting runner."""
def fit_view(packs,kind,view,alpha=1e-4,std_floor=1e-6):
    """Mean-loss ridge with unpenalized intercept, only caller train packs."""
    import numpy as np
    hkey='H_voltage' if kind=='voltage' else 'H_angle';zkey='voltage_logit' if kind=='voltage' else 'angle_logit'
    def features(pack):
        return pack['raw_final'] if view=='raw_final' else pack[hkey] if view=='H_only' else np.concatenate([pack['raw_final'],pack[hkey]],axis=1)
    x=np.concatenate([features(p).astype(np.float64) for p in packs])
    true=np.concatenate([p['targets'][:,:132] if kind=='voltage' else p['targets'][:,132:] for p in packs]).astype(np.float64)
    if kind=='voltage':
        if not np.all((true>0)&(true<1.2)):raise RuntimeError('Training V outside native sigmoid link')
        target=np.log(true/(1.2-true))
    else:
        if not np.all(np.abs(true/np.pi)<1):raise RuntimeError('Training theta outside native tanh link')
        target=np.arctanh(true/np.pi)
    target-=np.concatenate([p[zkey].astype(np.float64) for p in packs])
    mu=x.mean(0);sd0=x.std(0);sd=np.maximum(sd0,std_floor);xn=(x-mu)/sd;ym=target.mean(0)
    gram=xn.T@xn/len(x)+alpha*np.eye(x.shape[1]);stdw=np.linalg.solve(gram,xn.T@(target-ym)/len(x))
    w=(stdw/sd[:,None]).T;b=ym-mu@w.T;spectrum=np.linalg.eigvalsh(gram)
    return dict(W=w,b=b,feature_mean=mu,feature_sd_raw=sd0,feature_sd_used=sd,gram_eigenvalues=spectrum,
        training_residual_mse=np.asarray(np.mean((x@w.T+b-target)**2))),dict(rows=len(x),unique_endpoint_labels=2048,feature_dimension=x.shape[1],
            alpha=alpha,std_floor=std_floor,gram_condition_number=float(spectrum[-1]/spectrum[0]),coefficient_frobenius_norm=float(np.linalg.norm(w)),
            validation_labels_used=False)
