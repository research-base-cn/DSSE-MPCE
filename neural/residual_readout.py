"""Final native-logit residual: FP64 affine/addition, then native FP32 link.

The parent still owns its original single sigmoid/tanh link. These modules
return logits only, with the same operation order as the final experiment.
"""
import torch
from torch import nn


class StableResidualReadout(nn.Module):
    def __init__(self,original,raw_dim=288,view='raw_plus_H'):
        super().__init__()
        if not isinstance(original,nn.Linear) or original.out_features!=132:
            raise ValueError('Require the actual native132-output Linear')
        if view not in ('raw_plus_H','raw_final','H_only'):
            raise ValueError('Unknown frozen readout feature route')
        self.original=original;self.raw_dim=raw_dim;self.view=view
        dimension={'raw_plus_H':raw_dim+original.in_features,'raw_final':raw_dim,'H_only':original.in_features}[view]
        self.residual=nn.Linear(dimension,132,bias=True).to(device=original.weight.device,dtype=torch.float64)
        self.residual.requires_grad_(False);self.raw_measurements=None
        self.reset_zero()

    def reset_zero(self):
        with torch.no_grad():self.residual.weight.zero_();self.residual.bias.zero_()
        self.raw_measurements=None;self._coefficients_are_zero=True

    def set_raw_measurements(self,raw):
        if raw.ndim!=2 or raw.shape[1]!=self.raw_dim or raw.dtype!=self.original.weight.dtype or raw.device!=self.original.weight.device:
            raise ValueError('Require current native-dtype physical raw [batch,288]')
        if not torch.isfinite(raw).all():raise ValueError('Nonfinite raw measurements')
        self.raw_measurements=raw

    def load_fitted_raw_affine(self,weight,bias):
        if weight.dtype!=torch.float64 or bias.dtype!=torch.float64:
            raise ValueError('Use final original FP64 coefficients, never FP32 exports')
        if weight.shape!=self.residual.weight.shape or bias.shape!=self.residual.bias.shape:
            raise ValueError('Native hidden width/readout shape mismatch')
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():raise ValueError('Nonfinite coefficients')
        with torch.no_grad():self.residual.weight.copy_(weight);self.residual.bias.copy_(bias)
        self._coefficients_are_zero=bool(torch.count_nonzero(weight).item()==0 and torch.count_nonzero(bias).item()==0)

    def forward(self,h):
        if h.ndim!=2 or h.shape[1]!=self.original.in_features:raise RuntimeError('Native H shape changed')
        raw=self.raw_measurements;self.raw_measurements=None
        if self.view!='H_only' and (raw is None or len(raw)!=len(h)):
            raise RuntimeError('Set current batch raw measurements before every forward')
        z=self.original(h)
        if self._coefficients_are_zero:return z
        x=h if self.view=='H_only' else raw if self.view=='raw_final' else torch.cat([raw,h],dim=-1)
        residual=self.residual(x.to(dtype=torch.float64))
        return (z.to(dtype=torch.float64)+residual).to(dtype=z.dtype)
