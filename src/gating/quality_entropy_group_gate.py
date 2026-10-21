"""Prospective R3 measurement-context and local-entropy grouped message gate.

This is a new methodology candidate. It neither edits the submitted modules
nor repairs or relabels the preceding negative R1/R2 experiments. No labels,
corruption oracle, held-out statistics, or state corrections enter this gate.
Install before optimizer creation and freeze inherited parameters AFTER layer
conversion, because GAT conversion creates fresh Parameters.
"""
from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from normalized_entropy_gate import EntropyAwareGATv2Conv, node_attention_entropy


QUALITY_FEATURE_NAMES = (
    "local_scada_availability",
    "local_pmu_availability",
    "valid_pair_scada_log1p_absolute_temporal_jump",
    "valid_pair_pmu_log1p_absolute_temporal_jump",
    "global_pmu_availability",
)
QUALITY_FEATURE_COUNT = len(QUALITY_FEATURE_NAMES)
QUALITY_SURPRISE_COLUMNS = (2, 3)
GROUP_COUNT = 4


def _valid_pair_log_jump(values: Tensor, mask: Tensor) -> Tensor:
    """Observed temporal change, not an outlier label or physical residual.

    Inputs [B,T,N,C] already use the original training-only measurement
    normalization. A channel contributes only if both consecutive masks are
    positive. t=0 and nodes with no valid pair return exactly zero. Division
    uses the number of valid channels, avoiding attenuation from missing ones.
    """
    if values.ndim != 4 or values.shape != mask.shape:
        raise ValueError("Expected matching [batch,time,nodes,channels] tensors")
    result = values.new_zeros(values.shape[:-1])
    if values.shape[1] > 1:
        valid = (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
        # where occurs before reduction, so a masked value cannot create a
        # temporal-change feature simply by changing from/to the fill value.
        difference = torch.where(valid, values[:, 1:] - values[:, :-1],
                                 torch.zeros_like(values[:, 1:]))
        numerator = difference.abs().log1p().sum(dim=-1)
        denominator = valid.sum(dim=-1).clamp_min(1).to(values.dtype)
        result[:, 1:] = numerator / denominator
    return result


def quality_features_from_measurements(
    scada: Tensor, pmu: Tensor, scada_mask: Tensor, pmu_mask: Tensor,
    *, num_buses: int, scada_features: int, pmu_bus_indices: Tensor,
) -> Tensor:
    """Five observable context features, shape [B,T,N,K].

    Explicit masks are mandatory. Absent PMU locations have local coverage and
    jump zero. With fixed per-node normal-training centers they differ from a
    deployed PMU that becomes unavailable (normal local coverage is one).
    Global PMU coverage averages deployed PMU channels only, then broadcasts.
    A real operating change can also cause a large jump; no reliability truth
    is claimed for this observable statistic.
    """
    if scada.ndim != 3 or pmu.ndim != 3:
        raise ValueError("Expected [batch,time,flattened channels] measurements")
    batch, steps, _ = scada.shape
    if tuple(pmu.shape[:2]) != (batch, steps):
        raise ValueError("SCADA and PMU windows differ")
    if scada_mask is None or pmu_mask is None:
        raise ValueError("Explicit measurement masks are required by the R3 gate")
    if scada.shape != scada_mask.shape or pmu.shape != pmu_mask.shape:
        raise ValueError("Measurement/mask dimensions differ")
    count = int(pmu_bus_indices.numel())
    if count <= 0 or num_buses <= 0 or scada_features <= 0:
        raise ValueError("Nonempty SCADA and PMU designs are required")
    if pmu.shape[-1] != 2 * count or scada.shape[-1] != num_buses * scada_features:
        raise ValueError("Measurement dimensions differ from the fixed sensor design")
    # Sensor-design values are validated once at installation, avoiding host
    # synchronization for GPU unique/range checks in every training forward.
    if pmu_bus_indices.dtype != torch.long:
        raise ValueError("PMU design indices must be integer-long tensors")
    s = scada.reshape(batch, steps, num_buses, scada_features)
    sm = scada_mask.reshape_as(s)
    p = pmu.reshape(batch, steps, count, 2)
    pm = pmu_mask.reshape_as(p)
    local_s = sm.mean(dim=-1)
    local_p = s.new_zeros(batch, steps, num_buses)
    local_p[:, :, pmu_bus_indices] = pm.mean(dim=-1)
    jump_s = _valid_pair_log_jump(s, sm)
    jump_p = s.new_zeros(batch, steps, num_buses)
    jump_p[:, :, pmu_bus_indices] = _valid_pair_log_jump(p, pm)
    global_p = pm.mean(dim=(-1, -2)).unsqueeze(-1).expand(-1, -1, num_buses)
    return torch.stack((local_s, local_p, jump_s, jump_p, global_p), dim=-1)


def quality_calibration_from_samples(
    samples: Tensor, *, std_floor: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Train-normal moments for [snapshots,nodes,K] raw context features.

    Availability features are centered at per-node normal-training means and
    use scale exactly one; their vanishing normal-training variance is never
    amplified. Only temporal-jump columns use population SD with a fixed floor.
    t=0 zero entries remain in these moments, as they also enter the model.
    """
    if samples.ndim != 3 or samples.shape[-1] != QUALITY_FEATURE_COUNT or samples.shape[0] < 2:
        raise ValueError("Expected at least two [snapshot,node,K] samples")
    if not bool(torch.isfinite(samples).all()) or not 0 < std_floor <= 1:
        raise ValueError("Finite samples and positive declared SD floor required")
    center = samples.double().mean(dim=0)
    scale = torch.ones_like(center)
    for column in QUALITY_SURPRISE_COLUMNS:
        scale[:, column] = samples[:, :, column].double().std(dim=0, unbiased=False).clamp_min(std_floor)
    return center.to(samples.dtype), scale.to(samples.dtype)


def entropy_calibration_from_samples(
    samples: list[Tensor], *, std_floor: float = 0.001,
) -> tuple[Tensor, Tensor]:
    if not samples or not 0 < std_floor <= 1:
        raise ValueError("Nonempty layer samples and positive SD floor required")
    if any(part.ndim != 2 or part.shape != samples[0].shape or part.shape[0] < 2 or not bool(torch.isfinite(part).all()) for part in samples):
        raise ValueError("Expected finite matching [snapshot,node] layer samples")
    center = torch.stack([part.double().mean(dim=0) for part in samples])
    scale = torch.stack([part.double().std(dim=0, unbiased=False).clamp_min(std_floor) for part in samples])
    return center.to(samples[0].dtype), scale.to(samples[0].dtype)


def _degree_normalized_entropy(probabilities: Tensor, target_nodes: Tensor, total_nodes: int) -> Tensor:
    entropy = node_attention_entropy(probabilities, target_nodes, total_nodes)
    degree = torch.zeros(total_nodes, dtype=torch.long, device=target_nodes.device)
    degree.index_add_(0, target_nodes, torch.ones_like(target_nodes))
    denominator = degree.clamp_min(2).to(probabilities.dtype).log()
    return torch.where(degree > 1, entropy / denominator, torch.zeros_like(entropy))


def _r3_encode_measurements(self, measurements_scada, measurements_pmu,
                            scada_mask=None, pmu_mask=None):
    self._r3_quality_current = quality_features_from_measurements(
        measurements_scada, measurements_pmu, scada_mask, pmu_mask,
        num_buses=self.num_buses, scada_features=self.scada_features,
        pmu_bus_indices=self.pmu_bus_indices,
    )
    return self._r3_original_encode_measurements_function(
        self, measurements_scada, measurements_pmu, scada_mask, pmu_mask)


def _r3_spatial_encode(self, node_feat: Tensor, edge_index: Tensor) -> Tensor:
    batch, steps, nodes, width = node_feat.shape
    snapshots = batch * steps
    raw_quality = self._r3_quality_current
    if raw_quality is None or tuple(raw_quality.shape) != (batch, steps, nodes, QUALITY_FEATURE_COUNT):
        raise RuntimeError("Quality features are absent or stale for this forward")
    # Clear the temporary reference on use: direct repeated spatial calls must
    # not accidentally reuse quality from an earlier measurement window.
    self._r3_quality_current = None
    quality = raw_quality.reshape(snapshots, nodes, QUALITY_FEATURE_COUNT)
    z_quality = ((quality - self.r3_quality_center) / self.r3_quality_scale).clamp(-self._r3_clip, self._r3_clip)
    x = node_feat.reshape(snapshots * nodes, width)
    edges = self._batched_edge_index(edge_index, snapshots)
    entropies, gains = [], []
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
        gain = 2 * torch.sigmoid(logits)
        # The four contiguous groups are concatenated GAT output-head blocks.
        # GELU is elementwise; LayerNorm then mixes their normalization scale.
        message = F.gelu(message).reshape(snapshots * nodes, GROUP_COUNT, width // GROUP_COUNT)
        message = (message * gain.reshape(snapshots * nodes, GROUP_COUNT, 1)).reshape(snapshots * nodes, width)
        x = norm(x + message)
        if self._r3_capture:
            entropies.append(entropy.detach().cpu())
            gains.append(gain.detach().cpu())
    self._last_r3_entropies = entropies if self._r3_capture else None
    self._last_r3_quality = quality.detach().cpu() if self._r3_capture else None
    self._last_r3_gains = gains if self._r3_capture else None
    self._last_entropy_stats = None
    self._last_gate_stats = None
    self._last_gate_weights = None
    self._last_gate_biases = None
    return x.reshape(batch, steps, nodes, width)


def install_quality_entropy_gate(
    model, *, mode: str = "adaptive", entropy_center: Optional[Tensor] = None,
    entropy_scale: Optional[Tensor] = None, quality_center: Optional[Tensor] = None,
    quality_scale: Optional[Tensor] = None, capture: bool = False, clip: float = 3.0,
    reset_gate: bool = True,
):
    """Install R3 or a declared independently trained control.

    W=[layers,4,K+1], b=[layers,4]. K quality features precede one head-mean
    local-entropy feature. At zero W and b, q=1 and the entire forward function
    equals noGate, including LayerNorm epsilon and dropout RNG sequence.
    Effective trainable counts for three layers: adaptive 84, quality-only 72,
    entropy-only 24, constant 12. Inactive W entries must stay zero (wd=0);
    freeze all W for constant and inherited body after this installation.
    """
    if mode not in {"adaptive", "quality_only", "entropy_only", "constant"} or not 0 < clip <= 10:
        raise ValueError("Invalid declared mode or clip interval")
    backbone = getattr(model, "paper_backbone", model)
    if getattr(backbone, "gnn_type", None) != "gatv2" or getattr(backbone, "encoder_mode", None) != "nodewise":
        raise ValueError("Only original nodewise GAT paper backbone supported")
    if hasattr(backbone, "r3_gate_weight"):
        raise ValueError("R3 gate already installed; do not silently reset an existing study")
    layers, nodes = len(backbone.gnn_layers), int(backbone.num_buses)
    design = backbone.pmu_bus_indices
    if design.dtype != torch.long or design.ndim != 1 or not design.numel() or design.unique().numel() != design.numel() or bool((design < 0).any()) or bool((design >= nodes).any()):
        raise ValueError("Invalid fixed PMU sensor design")
    if int(backbone.d_model) % GROUP_COUNT:
        raise ValueError("GAT output width must divide into four concatenated head groups")
    reference = next(backbone.parameters())
    for index, layer in enumerate(backbone.gnn_layers):
        if layer.heads != GROUP_COUNT or not layer.concat:
            raise ValueError("Expected four concatenated GAT heads")
        if not isinstance(layer, EntropyAwareGATv2Conv):
            backbone.gnn_layers[index] = EntropyAwareGATv2Conv.from_layer(layer)
    backbone.r3_gate_weight = nn.Parameter(reference.new_zeros(layers, GROUP_COUNT, QUALITY_FEATURE_COUNT + 1))
    backbone.r3_gate_bias = nn.Parameter(reference.new_zeros(layers, GROUP_COUNT))
    # reset_gate exists only to make the installation API explicit; no previous
    # R3 state is accepted, so both settings start from the neutral function.
    if not isinstance(reset_gate, bool):
        raise ValueError("reset_gate must be a boolean")
    specifications = (
        ("r3_entropy_center", entropy_center, (layers, nodes), 0.0),
        ("r3_entropy_scale", entropy_scale, (layers, nodes), 1.0),
        ("r3_quality_center", quality_center, (nodes, QUALITY_FEATURE_COUNT), 0.0),
        ("r3_quality_scale", quality_scale, (nodes, QUALITY_FEATURE_COUNT), 1.0),
    )
    for name, value, shape, fill in specifications:
        value = reference.new_full(shape, fill) if value is None else value.to(reference)
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid {name} calibration")
        if name.endswith("scale") and bool((value <= 0).any()):
            raise ValueError("Every frozen calibration scale must be positive")
        backbone.register_buffer(name, value.detach().clone())
    backbone._r3_original_encode_measurements_function = type(backbone)._encode_measurements
    backbone._encode_measurements = MethodType(_r3_encode_measurements, backbone)
    backbone._spatial_encode = MethodType(_r3_spatial_encode, backbone)
    backbone._r3_mode, backbone._r3_clip, backbone._r3_capture = mode, float(clip), bool(capture)
    backbone._r3_quality_current = None
    backbone._last_r3_quality = backbone._last_r3_entropies = backbone._last_r3_gains = None
    backbone.entropy_gate = True
    return model
