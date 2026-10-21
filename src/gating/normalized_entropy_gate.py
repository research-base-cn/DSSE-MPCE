"""Isolated repair of the submitted global attention-entropy gate.

No frozen source is edited. The GATv2 message still uses its ordinary attention
dropout. Only the gate statistic reads the pre-dropout softmax probabilities.
The default affine slope remains unconstrained, with the original parameters.
The optional ``monotone`` mode is a separately declared variant, not the default.
"""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax


class EntropyAwareGATv2Conv(GATv2Conv):
    """GATv2 with a differentiable cache of normalized attention.

    ``last_normalized_attention`` is a temporary tensor, not a new parameter or
    buffer. It refers to the exact softmax used for the subsequent dropout and
    message in the same forward pass. Its edge order is the edge order returned
    by inherited ``forward(..., return_attention_weights=True)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_normalized_attention: Optional[Tensor] = None

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Optional[Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tensor:
        # This preserves the public GATv2 operator, including optional edge
        # attributes, source projection reuse, and attention-dropout scaling.
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if self.lin_edge is None:
                raise ValueError("edge_attr requires a configured edge_dim")
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        logits = (x * self.att).sum(dim=-1)
        probabilities = softmax(logits, index, ptr, dim_size)
        self.last_normalized_attention = probabilities
        return F.dropout(probabilities, p=self.dropout, training=self.training)

    @classmethod
    def from_layer(cls, layer: GATv2Conv) -> "EntropyAwareGATv2Conv":
        """Copy the original settings, weights, dtype, device, and mode.

        The paper uses default message-passing flow/aggregation. Nonstandard
        operators are rejected rather than silently converted.
        """
        if not isinstance(layer, GATv2Conv):
            raise TypeError("The entropy repair requires a GATv2Conv layer")
        if layer.flow != "source_to_target" or layer.aggr not in {"add", "sum"}:
            raise ValueError("Only the submitted source-to-target sum operator is supported")
        kwargs = {
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "heads": layer.heads,
            "concat": layer.concat,
            "negative_slope": layer.negative_slope,
            "dropout": layer.dropout,
            "add_self_loops": layer.add_self_loops,
            "edge_dim": layer.edge_dim,
            "fill_value": layer.fill_value,
            "bias": layer.bias is not None,
            "share_weights": layer.share_weights,
        }
        # PyG added this option after early GATv2 releases. Preserve it when the
        # installed base class supports it instead of requiring a new version.
        if "residual" in inspect.signature(GATv2Conv.__init__).parameters:
            kwargs["residual"] = bool(getattr(layer, "residual", False))
        reference = next(layer.parameters())
        # GAT construction initializes weights on CPU before moving them; keep
        # that initialization from changing future CPU random draws.
        with torch.random.fork_rng(devices=[]):
            converted = cls(**kwargs)
        converted = converted.to(device=reference.device, dtype=reference.dtype)
        converted.load_state_dict(layer.state_dict(), strict=True)
        converted.train(layer.training)
        return converted


def node_attention_entropy(
    probabilities: Tensor,
    target_nodes: Tensor,
    total_nodes: int,
    epsilon: float = 1e-8,
) -> Tensor:
    """Head-mean Shannon entropy summed over each node's incoming edges.

    Values must be pre-dropout softmax probabilities. The probability itself
    is not floored: ``0 * log(epsilon)`` is zero, so underflowed edges contribute
    zero. Only the logarithm is stabilized. No detach or degree normalization
    is applied.
    """
    if probabilities.ndim != 2:
        raise ValueError("Expected [edges, heads] attention probabilities")
    if target_nodes.ndim != 1 or target_nodes.numel() != probabilities.shape[0]:
        raise ValueError("One target node is required for every attention edge")
    if total_nodes <= 0 or not 0.0 < epsilon < 1.0:
        raise ValueError("Invalid node count or logarithm stabilizer")
    edge_entropy = -(
        probabilities * probabilities.clamp_min(epsilon).log()
    ).mean(dim=-1)
    return probabilities.new_zeros(total_nodes).index_add(
        0, target_nodes, edge_entropy
    )


def snapshot_mean_attention_entropy(
    probabilities: Tensor,
    target_nodes: Tensor,
    num_nodes: int,
    num_snapshots: int,
    epsilon: float = 1e-8,
) -> Tensor:
    """Return one gate statistic per independent snapshot, as submitted."""
    if num_nodes <= 0 or num_snapshots <= 0:
        raise ValueError("Node and snapshot counts must be positive")
    entropy = node_attention_entropy(
        probabilities, target_nodes, num_nodes * num_snapshots, epsilon
    )
    return entropy.reshape(num_snapshots, num_nodes).mean(dim=1)


def effective_affine_slope(parameter: Tensor, mode: str = "free") -> Tensor:
    """Original free slope by default; explicitly optional negative slope."""
    if mode == "free":
        return parameter
    if mode == "monotone":
        return -F.softplus(parameter)
    raise ValueError("slope_mode must be 'free' or 'monotone'")


def _normalized_entropy_spatial_encode(self, node_feat: Tensor, edge_index: Tensor) -> Tensor:
    """Paper spatial forward with the statistic's sole repair isolated."""
    batch_size, sequence_length, num_nodes, d_model = node_feat.shape
    num_snapshots = batch_size * sequence_length
    x_flat = node_feat.reshape(num_snapshots * num_nodes, d_model)
    batched_edges = self._batched_edge_index(edge_index, num_snapshots)
    entropy_stats = []
    gate_stats = []
    gate_weights = []
    for layer_index, (gnn_layer, norm_layer) in enumerate(
        zip(self.gnn_layers, self.gnn_norms)
    ):
        if not isinstance(gnn_layer, EntropyAwareGATv2Conv):
            raise TypeError("A repaired entropy-gated layer must be EntropyAwareGATv2Conv")
        message, (returned_edges, dropped_attention) = gnn_layer(
            x_flat, batched_edges, return_attention_weights=True
        )
        probabilities = gnn_layer.last_normalized_attention
        if probabilities is None or probabilities.shape != dropped_attention.shape:
            raise RuntimeError("The normalized-attention cache is absent or misaligned")
        message = F.gelu(message)
        graph_entropy = snapshot_mean_attention_entropy(
            probabilities, returned_edges[1], num_nodes, num_snapshots
        )
        slope = effective_affine_slope(
            self.entropy_gate_w[layer_index], self._entropy_slope_mode
        )
        gate = torch.sigmoid(slope * graph_entropy + self.entropy_gate_b[layer_index])
        broadcast_gate = gate.repeat_interleave(num_nodes).unsqueeze(-1)
        x_flat = norm_layer(broadcast_gate * message + (1.0 - broadcast_gate) * x_flat)
        # These detach calls reproduce diagnostic logging; the forward gate
        # above remains differentiable through the normalized attention.
        entropy_stats.append(float(graph_entropy.detach().mean().cpu()))
        gate_stats.append(float(gate.detach().mean().cpu()))
        gate_weights.append(float(slope.detach().cpu()))
    self._last_entropy_stats = entropy_stats or None
    self._last_gate_stats = gate_stats or None
    self._last_gate_weights = gate_weights or None
    self._last_gate_biases = [float(value.detach().cpu()) for value in self.entropy_gate_b]
    return x_flat.reshape(batch_size, sequence_length, num_nodes, d_model)


def install_normalized_entropy_repair(model, *, slope_mode: str = "free"):
    """Install the isolated repair on the paper backbone and return ``model``.

    All existing trainable parameters retain their values and state-dict names.
    Call before constructing the optimizer, since GAT layer objects are copied.
    No temporal block, encoder, readout, output head, or source gate is changed.

    ``slope_mode='monotone'`` explicitly reinterprets existing slope parameters
    as raw rho with effective w=-softplus(rho). It is an optional experiment
    with changed slope semantics and requires its own manifest/study identity.
    Log raw rho separately from effective w if this optional mode is selected.
    The default ``free`` mode preserves the original affine gate and its zero
    initialization; no performance claim is implied by this installation.
    """
    if slope_mode not in {"free", "monotone"}:
        raise ValueError("slope_mode must be 'free' or 'monotone'")
    backbone = getattr(model, "paper_backbone", model)
    if not bool(getattr(backbone, "entropy_gate", False)):
        raise ValueError("The repair applies only to the original entropy-gated paper backbone")
    for index, layer in enumerate(backbone.gnn_layers):
        if not isinstance(layer, EntropyAwareGATv2Conv):
            backbone.gnn_layers[index] = EntropyAwareGATv2Conv.from_layer(layer)
    backbone._entropy_slope_mode = slope_mode
    backbone._entropy_definition = "pre_dropout_softmax_global_mean"
    backbone._spatial_encode = MethodType(_normalized_entropy_spatial_encode, backbone)
    return model
