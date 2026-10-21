"""R4 safety activation sidecar for the immutable R3 grouped message gate.

No parameters, buffers, training recipe, or R3 source files are changed. The
only message-path change is q_safe = 1 + (1-c_raw) * (q_R3-1), where c_raw is
the observed per-snapshot availability over deployed PMU channels. This is a
new candidate motivated by R3 development results, not a reinterpretation of
R3's original joint SCADA robustness decision.
"""
from types import MethodType

import torch
import torch.nn.functional as F

from quality_entropy_group_gate import (
    GROUP_COUNT, QUALITY_FEATURE_COUNT, QUALITY_FEATURE_NAMES,
    _degree_normalized_entropy, _r3_spatial_encode,
)

PMU_COVERAGE_COLUMN = QUALITY_FEATURE_NAMES.index("global_pmu_availability")


def coverage_anchored_gain(gain, raw_coverage):
    """Shrink a (0,2) message gain toward one with observed coverage c in [0,1].

    Shapes: gain [...,groups], raw_coverage [...]. The sensor-mask contract
    supplies coverage in [0,1]. No standardization or threshold is used.
    For finite q, c=1 produces exactly one; c=0 recovers q in exact arithmetic.
    At intermediate c, q_safe belongs to (c,2-c).
    """
    if tuple(gain.shape[:-1]) != tuple(raw_coverage.shape):
        raise ValueError("Coverage dimensions must match gain except its group axis")
    return 1 + (1 - raw_coverage).unsqueeze(-1) * (gain - 1)


def _coverage_anchored_spatial_encode(self, node_feat, edge_index):
    # This follows the frozen R3 spatial operation order. Only the gain line
    # changes; the optional capture records the safe gain actually applied.
    batch, steps, nodes, width = node_feat.shape
    snapshots = batch * steps
    raw_quality = self._r3_quality_current
    if raw_quality is None or tuple(raw_quality.shape) != (batch, steps, nodes, QUALITY_FEATURE_COUNT):
        raise RuntimeError("Quality features are absent or stale for this forward")
    self._r3_quality_current = None
    quality = raw_quality.reshape(snapshots, nodes, QUALITY_FEATURE_COUNT)
    raw_coverage = quality[..., PMU_COVERAGE_COLUMN]
    z_quality = ((quality - self.r3_quality_center) / self.r3_quality_scale).clamp(-self._r3_clip, self._r3_clip)
    x = node_feat.reshape(snapshots * nodes, width)
    edges = self._batched_edge_index(edge_index, snapshots)
    entropies, gains, unanchored_gains = [], [], []
    for layer_index, (layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
        message, (returned_edges, _) = layer(x, edges, return_attention_weights=True)
        probabilities = layer.last_normalized_attention
        if probabilities is None:
            raise RuntimeError("Same-forward pre-dropout attention cache absent")
        entropy = _degree_normalized_entropy(probabilities, returned_edges[1], snapshots * nodes).reshape(snapshots, nodes)
        z_entropy = ((entropy - self.r3_entropy_center[layer_index]) / self.r3_entropy_scale[layer_index]).clamp(-self._r3_clip, self._r3_clip)
        if self._r3_mode == "quality_only":
            z = torch.cat((z_quality, torch.zeros_like(z_entropy).unsqueeze(-1)), dim=-1)
        elif self._r3_mode == "entropy_only":
            z = torch.cat((torch.zeros_like(z_quality), z_entropy.unsqueeze(-1)), dim=-1)
        elif self._r3_mode == "constant":
            z = torch.zeros_like(torch.cat((z_quality, z_entropy.unsqueeze(-1)), dim=-1))
        elif self._r3_mode == "adaptive":
            z = torch.cat((z_quality, z_entropy.unsqueeze(-1)), dim=-1)
        else:
            raise RuntimeError("Unknown R3 gate control")
        logits = torch.einsum("snk,gk->sng", z, self.r3_gate_weight[layer_index]) + self.r3_gate_bias[layer_index]
        unanchored = 2 * torch.sigmoid(logits)
        gain = coverage_anchored_gain(unanchored, raw_coverage)
        message = F.gelu(message).reshape(snapshots * nodes, GROUP_COUNT, width // GROUP_COUNT)
        message = (message * gain.reshape(snapshots * nodes, GROUP_COUNT, 1)).reshape(snapshots * nodes, width)
        x = norm(x + message)
        if self._r3_capture:
            entropies.append(entropy.detach().cpu())
            gains.append(gain.detach().cpu())
            unanchored_gains.append(unanchored.detach().cpu())
    self._last_r3_entropies = entropies if self._r3_capture else None
    self._last_r3_quality = quality.detach().cpu() if self._r3_capture else None
    self._last_r3_gains = gains if self._r3_capture else None
    self._last_r4_unanchored_gains = unanchored_gains if self._r3_capture else None
    self._last_r4_coverage = raw_coverage.detach().cpu() if self._r3_capture else None
    self._last_entropy_stats = None
    self._last_gate_stats = None
    self._last_gate_weights = None
    self._last_gate_biases = None
    return x.reshape(batch, steps, nodes, width)


def install_coverage_anchor(model):
    """Apply R4 after R3 installation, before/after loading R3 state tensors.

    Registered state_dict entries remain identical. A checkpoint loaded after
    this call must still be the separately declared terminal-epoch R3 state;
    this function never selects an epoch, fits a parameter, or changes a mode.
    """
    backbone = getattr(model, "paper_backbone", model)
    if getattr(backbone, "_coverage_anchor_installed", False):
        raise ValueError("Coverage anchor already installed")
    if not hasattr(backbone, "r3_gate_weight") or getattr(backbone._spatial_encode, "__func__", None) is not _r3_spatial_encode:
        raise ValueError("Install the unmodified R3 grouped gate before its R4 anchor")
    backbone._spatial_encode = MethodType(_coverage_anchored_spatial_encode, backbone)
    backbone._coverage_anchor_installed = True
    backbone._last_r4_unanchored_gains = None
    backbone._last_r4_coverage = None
    return model
