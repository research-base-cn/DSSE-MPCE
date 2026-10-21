"""Neural model architectures for distribution system state estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from graphmamba.graph_mamba_state_estimator import (
    HAS_MAMBA,
    GraphMambaStateEstimator,
)


NEURAL_VARIANTS = (
    "gat_mamba_full_gate",
    "gat_mamba_no_gate",
    "gcn_mamba",
    "gat_lstm",
    "gnn_se_snapshot",
    "sequence_mlp",
)


class SnapshotGNNStateEstimator(nn.Module):
    """GATv2 snapshot baseline using only the final measurement frame."""

    def __init__(
        self,
        num_buses: int,
        num_pmu_buses: int,
        pmu_bus_indices: list[int],
        d_model: int,
        n_gnn_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_buses = int(num_buses)
        self.num_pmu_buses = int(num_pmu_buses)
        self.d_model = int(d_model)
        self.register_buffer(
            "pmu_bus_indices",
            torch.as_tensor(pmu_bus_indices, dtype=torch.long),
            persistent=True,
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(10, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.node_embedding = nn.Parameter(torch.randn(num_buses, d_model) * 0.02)
        heads = 4
        self.gnn_layers = nn.ModuleList(
            [
                GATv2Conv(
                    d_model,
                    d_model // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
                for _ in range(n_gnn_layers)
            ]
        )
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_gnn_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.voltage_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_buses),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_buses),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _batched_edges(self, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        num_edges = edge_index.shape[1]
        offsets = torch.arange(batch_size, device=edge_index.device) * self.num_buses
        return edge_index.repeat(1, batch_size) + offsets.repeat_interleave(
            num_edges
        ).unsqueeze(0)

    def forward(
        self,
        measurements_scada: torch.Tensor,
        measurements_pmu: torch.Tensor,
        edge_index: torch.Tensor,
        scada_mask: torch.Tensor | None = None,
        pmu_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch_size, sequence_length, _ = measurements_scada.shape
        scada = measurements_scada.reshape(
            batch_size, sequence_length, self.num_buses, 3
        )[:, -1]
        pmu = measurements_pmu.reshape(
            batch_size, sequence_length, self.num_pmu_buses, 2
        )[:, -1]
        if scada_mask is None:
            scada_available = (scada.abs() > 1e-8).to(scada.dtype)
        else:
            scada_available = scada_mask.reshape(
                batch_size, sequence_length, self.num_buses, 3
            )[:, -1]
        if pmu_mask is None:
            pmu_available = (pmu.abs() > 1e-8).to(pmu.dtype)
        else:
            pmu_available = pmu_mask.reshape(
                batch_size, sequence_length, self.num_pmu_buses, 2
            )[:, -1]
        pmu_slots = torch.zeros(
            batch_size,
            self.num_buses,
            2,
            dtype=measurements_scada.dtype,
            device=measurements_scada.device,
        )
        pmu_mask_slots = torch.zeros_like(pmu_slots)
        pmu_slots[:, self.pmu_bus_indices] = pmu
        pmu_mask_slots[:, self.pmu_bus_indices] = pmu_available
        node_input = torch.cat(
            [scada, pmu_slots, scada_available, pmu_mask_slots], dim=-1
        )
        x = self.node_encoder(node_input) + self.node_embedding.unsqueeze(0)
        batched_edges = self._batched_edges(edge_index, batch_size)
        x = x.reshape(batch_size * self.num_buses, self.d_model)
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            x = norm(x + F.gelu(layer(x, batched_edges)))
        node_features = x.reshape(batch_size, self.num_buses, self.d_model)
        features = self.readout(node_features.mean(dim=1))
        return {
            "V_pred": torch.sigmoid(self.voltage_head(features)) * 1.2,
            "delta_pred": torch.tanh(self.angle_head(features)) * np.pi,
            "features": features,
            "node_features": node_features,
            "entropy_stats": None,
            "gate_stats": None,
            "gate_weights": None,
            "gate_biases": None,
        }


class SequenceMLPStateEstimator(nn.Module):
    """Non-graph, non-recurrent MLP over the complete fixed-length sequence."""

    def __init__(
        self,
        num_buses: int,
        num_pmu_buses: int,
        sequence_length: int,
        step_hidden_dimension: int,
        sequence_hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_buses = int(num_buses)
        self.num_pmu_buses = int(num_pmu_buses)
        self.sequence_length = int(sequence_length)
        input_per_step = self.num_buses * 6 + self.num_pmu_buses * 4
        self.step_encoder = nn.Sequential(
            nn.Linear(input_per_step, step_hidden_dimension),
            nn.LayerNorm(step_hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(step_hidden_dimension, step_hidden_dimension),
            nn.LayerNorm(step_hidden_dimension),
            nn.GELU(),
        )
        self.sequence_encoder = nn.Sequential(
            nn.Linear(
                self.sequence_length * step_hidden_dimension,
                sequence_hidden_dimension,
            ),
            nn.LayerNorm(sequence_hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dimension, sequence_hidden_dimension),
            nn.LayerNorm(sequence_hidden_dimension),
            nn.GELU(),
        )
        self.voltage_head = nn.Linear(sequence_hidden_dimension, num_buses)
        self.angle_head = nn.Linear(sequence_hidden_dimension, num_buses)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        measurements_scada: torch.Tensor,
        measurements_pmu: torch.Tensor,
        edge_index: torch.Tensor,
        scada_mask: torch.Tensor | None = None,
        pmu_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del edge_index
        if scada_mask is None or pmu_mask is None:
            raise ValueError("The formal sequence MLP requires explicit masks")
        batch_size, sequence_length, _ = measurements_scada.shape
        if sequence_length != self.sequence_length:
            raise ValueError("Sequence MLP received an unexpected sequence length")
        step_input = torch.cat(
            [measurements_scada, measurements_pmu, scada_mask, pmu_mask], dim=-1
        )
        step_features = self.step_encoder(step_input)
        features = self.sequence_encoder(step_features.reshape(batch_size, -1))
        return {
            "V_pred": torch.sigmoid(self.voltage_head(features)) * 1.2,
            "delta_pred": torch.tanh(self.angle_head(features)) * np.pi,
            "features": features,
            "node_features": None,
            "entropy_stats": None,
            "gate_stats": None,
            "gate_weights": None,
            "gate_biases": None,
        }


def build_formal_model(
    config: dict[str, Any],
    *,
    num_buses: int,
    num_pmu_buses: int,
    pmu_bus_indices: list[int],
    variant: str,
    device: torch.device,
) -> nn.Module:
    if variant not in NEURAL_VARIANTS:
        raise ValueError(f"Unknown formal neural variant: {variant}")
    if variant not in config["neural_variants"]:
        raise ValueError(f"Variant is absent from the formal config: {variant}")
    model_config = config["model"]
    if variant == "gnn_se_snapshot":
        return SnapshotGNNStateEstimator(
            num_buses=num_buses,
            num_pmu_buses=num_pmu_buses,
            pmu_bus_indices=pmu_bus_indices,
            d_model=int(model_config["d_model"]),
            n_gnn_layers=int(model_config["n_gnn_layers"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
    if variant == "sequence_mlp":
        mlp = model_config["sequence_mlp"]
        return SequenceMLPStateEstimator(
            num_buses=num_buses,
            num_pmu_buses=num_pmu_buses,
            sequence_length=int(config["data"]["sequence_length"]),
            step_hidden_dimension=int(mlp["step_hidden_dimension"]),
            sequence_hidden_dimension=int(mlp["sequence_hidden_dimension"]),
            dropout=float(model_config["dropout"]),
        ).to(device)

    if bool(model_config["require_real_mamba"]) and variant in {
        "gat_mamba_full_gate",
        "gat_mamba_no_gate",
        "gcn_mamba",
    } and not HAS_MAMBA:
        raise RuntimeError("A formal Mamba variant was requested without mamba_ssm")
    architecture = {
        "gat_mamba_full_gate": "paper_nodewise_gat_mamba",
        "gat_mamba_no_gate": "paper_nodewise_gat_mamba_nogate",
        "gcn_mamba": "paper_nodewise_gcn_mamba",
        "gat_lstm": "paper_nodewise_gat_lstm",
    }[variant]
    model = GraphMambaStateEstimator(
        num_buses=num_buses,
        num_pmu_buses=num_pmu_buses,
        d_model=int(model_config["d_model"]),
        d_state=int(model_config["d_state"]),
        n_gnn_layers=int(model_config["n_gnn_layers"]),
        use_mamba=True,
        dropout=float(model_config["dropout"]),
        n_temporal_layers=int(model_config["n_temporal_layers"]),
        bidirectional_mamba=True,
        architecture=architecture,
        pmu_bus_indices=pmu_bus_indices,
        mamba_clamp=float(model_config["mamba_clamp"]),
        readout_mode=str(model_config["readout_mode"]),
        scada_features=int(model_config["scada_features"]),
        require_mamba=bool(model_config["require_real_mamba"]),
        entropy_gate_w_init=float(model_config["gate_w_init"]),
        entropy_gate_b_init=float(model_config["gate_b_init"]),
    ).to(device)
    if variant in {"gat_mamba_full_gate", "gat_mamba_no_gate", "gcn_mamba"}:
        if not model.paper_backbone.use_mamba:
            raise RuntimeError("Formal Mamba variant instantiated a fallback backbone")
    if variant == "gat_lstm" and model.paper_backbone.use_mamba:
        raise RuntimeError("Formal GAT-LSTM unexpectedly instantiated Mamba")
    return model


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
