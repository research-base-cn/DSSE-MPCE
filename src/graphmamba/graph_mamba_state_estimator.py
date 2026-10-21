"""
Graph-Mamba for Distribution Network State Estimation
======================================================

Graph-Mamba architecture combining:
- Graph Neural Networks (GNN) for spatial topology encoding
- Mamba State Space Model for temporal dynamics
- Applied to distribution network state estimation

State Vector: x = [V_1, ..., V_n, delta_1, ..., delta_n]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, global_mean_pool
import numpy as np

try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not available, using LSTM fallback")


class NodeAwareGraphMambaBackbone(nn.Module):
    """Node-aware DSSE backbone that preserves bus-local measurements."""

    def __init__(
        self,
        num_buses,
        num_pmu_buses,
        d_model=128,
        d_state=32,
        n_gnn_layers=3,
        use_mamba=True,
        dropout=0.1,
        n_temporal_layers=2,
        pmu_bus_indices=None,
    ):
        super().__init__()
        self.num_buses = num_buses
        self.num_pmu_buses = num_pmu_buses
        self.d_model = d_model
        self.use_mamba = use_mamba and HAS_MAMBA
        self.n_temporal_layers = n_temporal_layers

        if pmu_bus_indices is None:
            pmu_bus_indices = list(range(min(num_buses, num_pmu_buses)))
        pmu_tensor = torch.tensor(pmu_bus_indices, dtype=torch.long)
        self.register_buffer("pmu_bus_indices", pmu_tensor, persistent=False)

        self.scada_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pmu_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pmu_quality_encoder = nn.Sequential(
            nn.Linear(7, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.pmu_input_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.node_embedding = nn.Parameter(torch.randn(num_buses, d_model) * 0.02)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(d_model, d_model) for _ in range(n_gnn_layers)]
        )
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_gnn_layers)]
        )

        if self.use_mamba:
            self.local_mamba_layers = nn.ModuleList(
                [
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    for _ in range(n_temporal_layers)
                ]
            )
            self.local_mamba_norms = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(n_temporal_layers)]
            )
            self.global_mamba_layers = nn.ModuleList(
                [
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    for _ in range(n_temporal_layers)
                ]
            )
            self.global_mamba_norms = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(n_temporal_layers)]
            )
        else:
            lstm_layers = max(2, n_temporal_layers)
            lstm_dropout = dropout if lstm_layers > 1 else 0.0
            self.local_lstm = nn.LSTM(
                d_model,
                d_model,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=True,
            )
            self.local_lstm_proj = nn.Linear(d_model * 2, d_model)
            self.global_lstm = nn.LSTM(
                d_model,
                d_model,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=True,
            )
            self.global_lstm_proj = nn.Linear(d_model * 2, d_model)

        self.global_context_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_context_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.voltage_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.pmu_output_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def gnn_encode(self, x, edge_index):
        batch_seq, num_nodes, _ = x.shape
        x_flat = x.reshape(batch_seq * num_nodes, -1)
        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_seq)
        offsets = torch.arange(batch_seq, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(num_edges)
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)

        for gcn_layer, norm_layer in zip(self.gnn_layers, self.gnn_norms):
            x_gcn = gcn_layer(x_flat, edge_index_batch)
            x_gcn = F.gelu(x_gcn)
            x_flat = norm_layer(x_flat + x_gcn)

        return x_flat.reshape(batch_seq, num_nodes, -1)

    def temporal_encode(self, x, branch):
        if self.use_mamba:
            if branch == "local":
                layers = self.local_mamba_layers
                norms = self.local_mamba_norms
            else:
                layers = self.global_mamba_layers
                norms = self.global_mamba_norms

            for layer, norm in zip(layers, norms):
                x_layer = layer(x)
                x = norm(x + x_layer)
            return x[:, -1, :]

        if branch == "local":
            x_lstm, _ = self.local_lstm(x)
            return self.local_lstm_proj(x_lstm[:, -1, :])

        x_lstm, _ = self.global_lstm(x)
        return self.global_lstm_proj(x_lstm[:, -1, :])

    def forward(
        self,
        measurements_scada,
        measurements_pmu,
        edge_index,
        scada_mask=None,
        pmu_mask=None,
    ):
        batch_size, seq_len, _ = measurements_scada.shape
        scada_nodes = measurements_scada.reshape(batch_size, seq_len, self.num_buses, 3)
        pmu_seq = measurements_pmu.reshape(batch_size, seq_len, self.num_pmu_buses, 2)

        scada_feat = self.scada_encoder(scada_nodes)

        pmu_available = (pmu_seq[..., 0].abs() > 1e-6).float()
        prev_pmu = torch.roll(pmu_seq, shifts=1, dims=1)
        prev_pmu[:, 0] = pmu_seq[:, 0]
        scada_v_at_pmu = scada_nodes[:, :, self.pmu_bus_indices, 2]
        pmu_quality = torch.stack(
            [
                scada_v_at_pmu,
                pmu_seq[..., 0],
                pmu_seq[..., 1],
                torch.abs(pmu_seq[..., 0] - scada_v_at_pmu),
                torch.abs(pmu_seq[..., 0] - prev_pmu[..., 0]),
                torch.abs(pmu_seq[..., 1] - prev_pmu[..., 1]),
                pmu_available,
            ],
            dim=-1,
        )
        pmu_quality_emb = self.pmu_quality_encoder(pmu_quality)

        pmu_slots = torch.zeros(
            batch_size,
            seq_len,
            self.num_buses,
            3,
            device=measurements_pmu.device,
            dtype=measurements_pmu.dtype,
        )
        pmu_slots[:, :, self.pmu_bus_indices, :2] = pmu_seq
        pmu_slots[:, :, self.pmu_bus_indices, 2] = pmu_available
        pmu_feat_raw = self.pmu_encoder(pmu_slots)

        pmu_quality_slots = torch.zeros_like(pmu_feat_raw)
        pmu_quality_slots[:, :, self.pmu_bus_indices, :] = pmu_quality_emb
        pmu_mask_slots = torch.zeros(
            batch_size,
            seq_len,
            self.num_buses,
            1,
            device=measurements_pmu.device,
            dtype=measurements_pmu.dtype,
        )
        pmu_mask_slots[:, :, self.pmu_bus_indices, 0] = pmu_available
        pmu_gate_logits = self.pmu_input_gate(
            torch.cat([scada_feat, pmu_feat_raw, pmu_quality_slots], dim=-1)
        )
        pmu_gate = torch.sigmoid(pmu_gate_logits) * pmu_mask_slots
        pmu_feat = pmu_feat_raw * pmu_gate + pmu_quality_slots * pmu_gate

        node_input = (
            scada_feat + pmu_feat + self.node_embedding.unsqueeze(0).unsqueeze(0)
        )
        spatial_input = node_input.reshape(
            batch_size * seq_len, self.num_buses, self.d_model
        )
        spatial_feat = self.gnn_encode(spatial_input, edge_index).reshape(
            batch_size, seq_len, self.num_buses, self.d_model
        )

        local_seq = spatial_feat.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_buses, seq_len, self.d_model
        )
        local_temp = self.temporal_encode(local_seq, branch="local").reshape(
            batch_size, self.num_buses, self.d_model
        )

        global_seq = spatial_feat.mean(dim=2)
        global_temp = self.temporal_encode(global_seq, branch="global")
        global_snapshot = spatial_feat[:, -1].mean(dim=1)
        global_context = self.global_context_proj(
            torch.cat([global_temp, global_snapshot], dim=-1)
        )
        global_context = global_context.unsqueeze(1).expand(-1, self.num_buses, -1)

        last_input = node_input[:, -1]
        last_spatial = spatial_feat[:, -1]
        node_state = self.node_context_fusion(
            torch.cat([local_temp, last_spatial, last_input, global_context], dim=-1)
        )

        V_pred = torch.sigmoid(self.voltage_head(node_state).squeeze(-1)) * 0.4 + 0.8
        delta_pred = torch.tanh(self.angle_head(node_state).squeeze(-1)) * (np.pi / 4)

        last_pmu_feat = pmu_feat[:, -1]
        last_quality_feat = pmu_quality_slots[:, -1]
        skip_gate = torch.sigmoid(
            self.pmu_output_gate(
                torch.cat([node_state, last_pmu_feat, last_quality_feat], dim=-1)
            )
        ).squeeze(-1)
        pmu_v = torch.clamp(pmu_seq[:, -1, :, 0], 0.8, 1.2)
        pmu_delta = torch.clamp(pmu_seq[:, -1, :, 1], -np.pi / 4, np.pi / 4)
        pmu_skip_gate = skip_gate[:, self.pmu_bus_indices] * pmu_available[:, -1]
        V_pred = V_pred.clone()
        delta_pred = delta_pred.clone()
        V_pred[:, self.pmu_bus_indices] = (1.0 - pmu_skip_gate) * V_pred[
            :, self.pmu_bus_indices
        ] + pmu_skip_gate * pmu_v
        delta_pred[:, self.pmu_bus_indices] = (1.0 - pmu_skip_gate) * delta_pred[
            :, self.pmu_bus_indices
        ] + pmu_skip_gate * pmu_delta

        return {
            "V_pred": V_pred,
            "delta_pred": delta_pred,
            "features": node_state.mean(dim=1),
            "node_features": node_state,
        }


class RobustGraphMambaBackbone(nn.Module):
    """Robust DSSE backbone with dual-head output, configurable GNN and temporal."""

    def __init__(
        self,
        num_buses,
        num_pmu_buses,
        d_model=128,
        d_state=32,
        n_gnn_layers=3,
        use_mamba=True,
        use_transformer=False,
        gnn_type="gatv2",
        dropout=0.1,
        n_temporal_layers=2,
        pmu_bus_indices=None,
    ):
        super().__init__()
        self.num_buses = num_buses
        self.num_pmu_buses = num_pmu_buses
        self.d_model = d_model
        self.use_mamba = use_mamba and HAS_MAMBA
        self.use_transformer = use_transformer and not self.use_mamba
        self.gnn_type = gnn_type

        if pmu_bus_indices is None:
            pmu_bus_indices = list(range(min(num_buses, num_pmu_buses)))
        self.register_buffer(
            "pmu_bus_indices",
            torch.tensor(pmu_bus_indices, dtype=torch.long),
            persistent=False,
        )

        scada_in = 6
        self.scada_encoder = nn.Sequential(
            nn.Linear(scada_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pmu_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.node_embedding = nn.Parameter(torch.randn(num_buses, d_model) * 0.02)

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(n_gnn_layers):
            if gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(d_model, d_model))
            else:
                gat_heads = 4
                self.gnn_layers.append(
                    GATv2Conv(
                        d_model,
                        d_model // gat_heads,
                        heads=gat_heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            self.gnn_norms.append(nn.LayerNorm(d_model))

        if self.use_mamba:
            self.temporal_layers = nn.ModuleList(
                [
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    for _ in range(n_temporal_layers)
                ]
            )
            self.temporal_norms = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(n_temporal_layers)]
            )
        elif self.use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(
                enc_layer, num_layers=n_temporal_layers
            )
        else:
            self.temporal_lstm = nn.LSTM(
                d_model,
                d_model,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(d_model * 2, d_model)

        self.node_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.voltage_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.voltage_head_abs = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.availability_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self.pmu_blend_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _gnn_encode(self, x, edge_index):
        bt, n, d = x.shape
        x_flat = x.reshape(bt * n, d)
        num_edges = edge_index.size(1)
        ei_batch = edge_index.repeat(1, bt)
        offsets = torch.arange(bt, device=edge_index.device) * n
        ei_batch = ei_batch + offsets.repeat_interleave(num_edges).unsqueeze(0)
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            x_gnn = gnn(x_flat, ei_batch)
            x_gnn = F.gelu(x_gnn)
            if x_flat.shape[-1] != x_gnn.shape[-1]:
                x_flat = norm(x_gnn)
            else:
                x_flat = norm(x_flat + x_gnn)
        return x_flat.reshape(bt, n, -1)

    def _temporal_encode(self, x):
        if self.use_mamba:
            for layer, norm in zip(self.temporal_layers, self.temporal_norms):
                x_l = layer(x)
                x = norm(x + x_l)
            return x[:, -1, :]
        if self.use_transformer:
            x = self.temporal_transformer(x)
            return x[:, -1, :]
        out, _ = self.temporal_lstm(x)
        return self.temporal_proj(out[:, -1, :])

    def forward(
        self,
        measurements_scada,
        measurements_pmu,
        edge_index,
        scada_mask=None,
        pmu_mask=None,
    ):
        batch_size, seq_len, _ = measurements_scada.shape
        scada = measurements_scada.reshape(batch_size, seq_len, self.num_buses, 3)
        pmu_seq = measurements_pmu.reshape(batch_size, seq_len, self.num_pmu_buses, 2)

        prev_scada = torch.roll(scada, shifts=1, dims=1)
        prev_scada[:, 0] = scada[:, 0]
        temporal_jump = torch.abs(scada - prev_scada)
        availability = (scada[..., 0].abs() > 1e-6).float().unsqueeze(-1)
        scada_input = torch.cat([scada, temporal_jump[..., :2], availability], dim=-1)
        scada_feat = self.scada_encoder(scada_input)

        pmu_slots = torch.zeros(
            batch_size,
            seq_len,
            self.num_buses,
            self.d_model,
            device=measurements_scada.device,
            dtype=measurements_scada.dtype,
        )
        pmu_emb = self.pmu_encoder(pmu_seq)
        pmu_slots[:, :, self.pmu_bus_indices, :] = pmu_emb

        node_input = (
            scada_feat + pmu_slots + self.node_embedding.unsqueeze(0).unsqueeze(0)
        )

        spatial_in = node_input.reshape(batch_size * seq_len, self.num_buses, -1)
        spatial_feat = self._gnn_encode(spatial_in, edge_index).reshape(
            batch_size, seq_len, self.num_buses, self.d_model
        )

        global_seq = spatial_feat.mean(dim=2)
        global_ctx = self._temporal_encode(global_seq)
        global_ctx = global_ctx.unsqueeze(1).expand(-1, self.num_buses, -1)

        last_spatial = spatial_feat[:, -1]
        node_state = self.node_fusion(torch.cat([last_spatial, global_ctx], dim=-1))

        V_raw = torch.clamp(scada[:, -1, :, 2], 0.8, 1.2)
        delta_V = self.voltage_head(node_state).squeeze(-1)
        V_residual = V_raw + delta_V
        V_absolute = (
            torch.sigmoid(self.voltage_head_abs(node_state).squeeze(-1)) * 0.4 + 0.8
        )
        hard_avail = (scada[:, -1, :, 2].abs() > 1e-6).float()
        learned_avail = self.availability_head(node_state).squeeze(-1)
        avail_v = hard_avail * learned_avail
        V_pred = avail_v * V_residual + (1.0 - avail_v) * V_absolute

        delta_pred = torch.tanh(self.angle_head(node_state).squeeze(-1)) * (np.pi / 4)

        pmu_v = torch.clamp(pmu_seq[:, -1, :, 0], 0.8, 1.2)
        pmu_delta = torch.clamp(pmu_seq[:, -1, :, 1], -np.pi / 4, np.pi / 4)
        pmu_available = (pmu_seq[:, -1, :, 0].abs() > 1e-6).float()

        node_state_pmu = node_state[:, self.pmu_bus_indices]
        pmu_feat_last = pmu_slots[:, -1, self.pmu_bus_indices]
        blend_logits = self.pmu_blend_head(
            torch.cat([node_state_pmu, pmu_feat_last], dim=-1)
        ).squeeze(-1)
        alpha = torch.sigmoid(blend_logits) * pmu_available

        V_pred = V_pred.clone()
        delta_pred = delta_pred.clone()
        V_pred[:, self.pmu_bus_indices] = (1.0 - alpha) * V_pred[
            :, self.pmu_bus_indices
        ] + alpha * pmu_v
        delta_pred[:, self.pmu_bus_indices] = (1.0 - alpha) * delta_pred[
            :, self.pmu_bus_indices
        ] + alpha * pmu_delta

        return {
            "V_pred": V_pred,
            "delta_pred": delta_pred,
            "features": node_state.mean(dim=1),
            "node_features": node_state,
        }


class PaperGATMambaBackbone(nn.Module):
    """Paper-facing DSSE backbone with entropy-gated spatial aggregation.

    The backbone supports two measurement encoders for the paper experiments:
    node-wise local features, and a global encoder/broadcast variant. Both keep
    the same entropy-gated spatial module, graph-level readout, temporal block,
    and bounded voltage/angle heads so the encoder choice can be selected by
    validation performance.
    """

    def __init__(
        self,
        num_buses,
        num_pmu_buses,
        d_model=160,
        d_state=32,
        n_gnn_layers=3,
        use_mamba=True,
        use_transformer=False,
        bidirectional_mamba=True,
        gnn_type="gatv2",
        entropy_gate=True,
        encoder_mode="nodewise",
        readout_mode="graph",
        dropout=0.1,
        n_temporal_layers=1,
        pmu_bus_indices=None,
        mamba_clamp=0.0,
        scada_features=3,
        require_mamba=False,
        entropy_gate_w_init=0.0,
        entropy_gate_b_init=0.0,
    ):
        super().__init__()
        self.num_buses = num_buses
        self.num_pmu_buses = num_pmu_buses
        self.d_model = d_model
        self.gnn_type = gnn_type
        self.entropy_gate = entropy_gate and gnn_type == "gatv2"
        self.encoder_mode = encoder_mode
        self.readout_mode = readout_mode
        if require_mamba and use_mamba and not HAS_MAMBA:
            raise RuntimeError(
                "The paper experiment requires mamba_ssm, but it is unavailable. "
                "Refusing the legacy LSTM fallback."
            )
        self.use_mamba = use_mamba and HAS_MAMBA
        self.use_transformer = use_transformer and not self.use_mamba
        self.bidirectional_mamba = bidirectional_mamba and self.use_mamba
        self.n_temporal_layers = max(1, n_temporal_layers)
        self.mamba_clamp = mamba_clamp

        if pmu_bus_indices is None:
            pmu_bus_indices = list(range(min(num_buses, num_pmu_buses)))
        self.register_buffer(
            "pmu_bus_indices",
            torch.tensor(pmu_bus_indices, dtype=torch.long),
            persistent=False,
        )

        self.scada_features = scada_features

        # Per-node vector: SCADA P/Q/[V], PMU V/theta slots, and explicit masks.
        self.node_feature_dim = 2 * scada_features + 4
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(scada_features * num_buses + 2 * num_pmu_buses, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.node_embedding = nn.Parameter(torch.randn(num_buses, d_model) * 0.02)

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        gat_heads = 4
        for _ in range(n_gnn_layers):
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(d_model, d_model))
            elif gnn_type == "sage":
                self.gnn_layers.append(SAGEConv(d_model, d_model))
            else:
                self.gnn_layers.append(
                    GATv2Conv(
                        d_model,
                        d_model // gat_heads,
                        heads=gat_heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            self.gnn_norms.append(nn.LayerNorm(d_model))

        if self.entropy_gate:
            # Keep the submitted affine entropy calibration fully learnable.
            # A neutral initialization makes g=sigmoid(0)=0.5, which is
            # functionally aligned with the no-gate residual after LayerNorm.
            self.entropy_gate_w = nn.Parameter(
                torch.full((n_gnn_layers,), float(entropy_gate_w_init))
            )
            self.entropy_gate_b = nn.Parameter(
                torch.full((n_gnn_layers,), float(entropy_gate_b_init))
            )

        if self.use_mamba:
            self.mamba_layers = nn.ModuleList(
                [
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    for _ in range(self.n_temporal_layers)
                ]
            )
            self.temporal_norms = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(self.n_temporal_layers)]
            )
            if self.bidirectional_mamba:
                self.backward_mamba_layers = nn.ModuleList(
                    [
                        Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                        for _ in range(self.n_temporal_layers)
                    ]
                )
                self.backward_temporal_norms = nn.ModuleList(
                    [nn.LayerNorm(d_model) for _ in range(self.n_temporal_layers)]
                )
                self.temporal_proj = nn.Linear(d_model * 2, d_model)
        elif self.use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(
                enc_layer, num_layers=max(1, self.n_temporal_layers)
            )
        else:
            lstm_layers = max(2, self.n_temporal_layers)
            self.temporal_lstm = nn.LSTM(
                d_model,
                d_model,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(d_model * 2, d_model)

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
        self.node_voltage_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.node_angle_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._last_entropy_stats = None
        self._last_gate_stats = None
        self._last_gate_weights = None
        self._last_gate_biases = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _build_node_features(
        self, measurements_scada, measurements_pmu, scada_mask=None, pmu_mask=None
    ):
        batch_size, seq_len, _ = measurements_scada.shape
        scada = measurements_scada.reshape(
            batch_size, seq_len, self.num_buses, self.scada_features
        )
        pmu = measurements_pmu.reshape(batch_size, seq_len, self.num_pmu_buses, 2)

        if scada_mask is None:
            scada_mask = (scada.abs() > 1e-8).float()
        else:
            scada_mask = scada_mask.reshape(
                batch_size, seq_len, self.num_buses, self.scada_features
            )

        if pmu_mask is None:
            pmu_mask = (pmu.abs() > 1e-8).float()
        else:
            pmu_mask = pmu_mask.reshape(batch_size, seq_len, self.num_pmu_buses, 2)

        pmu_slots = torch.zeros(
            batch_size,
            seq_len,
            self.num_buses,
            2,
            device=measurements_scada.device,
            dtype=measurements_scada.dtype,
        )
        pmu_mask_slots = torch.zeros_like(pmu_slots)
        pmu_slots[:, :, self.pmu_bus_indices, :] = pmu
        pmu_mask_slots[:, :, self.pmu_bus_indices, :] = pmu_mask

        return torch.cat([scada, pmu_slots, scada_mask, pmu_mask_slots], dim=-1)

    def _encode_measurements(
        self, measurements_scada, measurements_pmu, scada_mask=None, pmu_mask=None
    ):
        batch_size, seq_len, _ = measurements_scada.shape
        if self.encoder_mode == "global":
            meas = torch.cat([measurements_scada, measurements_pmu], dim=-1)
            graph_feat = self.global_encoder(meas)
            node_feat = graph_feat.unsqueeze(2).expand(-1, -1, self.num_buses, -1)
        else:
            node_inputs = self._build_node_features(
                measurements_scada, measurements_pmu, scada_mask, pmu_mask
            )
            node_feat = self.node_encoder(node_inputs)

        return node_feat + self.node_embedding.view(1, 1, self.num_buses, self.d_model)

    def _batched_edge_index(self, edge_index, batch_seq):
        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_seq)
        offsets = torch.arange(batch_seq, device=edge_index.device) * self.num_buses
        return edge_index_batch + offsets.repeat_interleave(num_edges).unsqueeze(0)

    def _spatial_encode(self, node_feat, edge_index):
        batch_size, seq_len, num_nodes, d_model = node_feat.shape
        batch_seq = batch_size * seq_len
        x_flat = node_feat.reshape(batch_seq * num_nodes, d_model)
        edge_index_batch = self._batched_edge_index(edge_index, batch_seq)

        entropy_stats = []
        gate_stats = []
        gate_weights = []
        for layer_idx, (gnn_layer, norm_layer) in enumerate(
            zip(self.gnn_layers, self.gnn_norms)
        ):
            if self.entropy_gate:
                x_msg, (ei_out, alpha) = gnn_layer(
                    x_flat, edge_index_batch, return_attention_weights=True
                )
                x_msg = F.gelu(x_msg)
                target_nodes = ei_out[1]
                alpha_clamped = torch.clamp(alpha, min=1e-8)
                edge_entropy = -(alpha_clamped * torch.log(alpha_clamped)).mean(dim=-1)
                node_entropy = torch.zeros_like(x_flat[:, 0])
                node_entropy.scatter_add_(0, target_nodes, edge_entropy)
                graph_entropy = node_entropy.reshape(batch_seq, num_nodes).mean(dim=1)
                gate_weight = self.entropy_gate_w[layer_idx]
                gate = torch.sigmoid(
                    gate_weight * graph_entropy + self.entropy_gate_b[layer_idx]
                )
                gate_node = gate.repeat_interleave(num_nodes).unsqueeze(-1)
                x_flat = norm_layer(gate_node * x_msg + (1.0 - gate_node) * x_flat)
                entropy_stats.append(float(graph_entropy.detach().mean().cpu()))
                gate_stats.append(float(gate.detach().mean().cpu()))
                gate_weights.append(float(gate_weight.detach().cpu()))
            else:
                x_msg = F.gelu(gnn_layer(x_flat, edge_index_batch))
                x_flat = norm_layer(x_flat + x_msg)

        self._last_entropy_stats = entropy_stats if entropy_stats else None
        self._last_gate_stats = gate_stats if gate_stats else None
        self._last_gate_weights = gate_weights if gate_weights else None
        self._last_gate_biases = (
            [float(value.detach().cpu()) for value in self.entropy_gate_b]
            if self.entropy_gate
            else None
        )
        return x_flat.reshape(batch_size, seq_len, num_nodes, d_model)

    @staticmethod
    def _bidirectional_mamba_summary(x_forward, x_backward_reversed):
        """Return full-window endpoint states from both directional scans.

        ``x_backward_reversed`` remains in reverse-sequence order. Its final
        position has consumed the complete reversed window. Flipping it back
        before selecting the last original-time position would instead select
        the state that has consumed only the original final snapshot.
        """
        return torch.cat(
            [x_forward[:, -1], x_backward_reversed[:, -1]], dim=-1
        )

    @staticmethod
    def _bidirectional_lstm_summary(hidden_state):
        """Return the last-layer forward and backward LSTM hidden states."""
        if hidden_state.shape[0] < 2 or hidden_state.shape[0] % 2 != 0:
            raise ValueError("Expected interleaved forward/backward LSTM states")
        return torch.cat([hidden_state[-2], hidden_state[-1]], dim=-1)

    def _temporal_encode(self, graph_seq):
        if self.use_mamba:
            x_forward = graph_seq
            for layer, norm in zip(self.mamba_layers, self.temporal_norms):
                x_in = (
                    torch.clamp(x_forward, -self.mamba_clamp, self.mamba_clamp)
                    if self.mamba_clamp > 0
                    else x_forward
                )
                x_forward = norm(x_forward + layer(x_in))

            if self.bidirectional_mamba:
                x_backward = torch.flip(graph_seq, dims=[1])
                for layer, norm in zip(
                    self.backward_mamba_layers, self.backward_temporal_norms
                ):
                    x_in = (
                        torch.clamp(x_backward, -self.mamba_clamp, self.mamba_clamp)
                        if self.mamba_clamp > 0
                        else x_backward
                    )
                    x_backward = norm(x_backward + layer(x_in))
                return self.temporal_proj(
                    self._bidirectional_mamba_summary(x_forward, x_backward)
                )
            return x_forward[:, -1]

        if self.use_transformer:
            return self.temporal_transformer(graph_seq)[:, -1]

        _, (hidden_state, _) = self.temporal_lstm(graph_seq)
        return self.temporal_proj(
            self._bidirectional_lstm_summary(hidden_state)
        )

    def forward(
        self,
        measurements_scada,
        measurements_pmu,
        edge_index,
        scada_mask=None,
        pmu_mask=None,
    ):
        node_feat = self._encode_measurements(
            measurements_scada, measurements_pmu, scada_mask, pmu_mask
        )
        spatial_feat = self._spatial_encode(node_feat, edge_index)
        if self.readout_mode == "nodewise":
            batch_size, seq_len, num_nodes, d_model = spatial_feat.shape
            node_seq = spatial_feat.permute(0, 2, 1, 3).reshape(
                batch_size * num_nodes, seq_len, d_model
            )
            node_temporal = self._temporal_encode(node_seq)
            node_features = self.readout(node_temporal).reshape(
                batch_size, num_nodes, d_model
            )
            V_pred = (
                torch.sigmoid(self.node_voltage_head(node_features)).squeeze(-1) * 1.2
            )
            delta_pred = (
                torch.tanh(self.node_angle_head(node_features)).squeeze(-1) * np.pi
            )
            features = node_features.mean(dim=1)
        else:
            graph_seq = spatial_feat.mean(dim=2)
            temporal_feat = self._temporal_encode(graph_seq)
            features = self.readout(temporal_feat)
            V_pred = torch.sigmoid(self.voltage_head(features)) * 1.2
            delta_pred = torch.tanh(self.angle_head(features)) * np.pi
        return {
            "V_pred": V_pred,
            "delta_pred": delta_pred,
            "features": features,
            "node_features": spatial_feat[:, -1],
            "entropy_stats": self._last_entropy_stats,
            "gate_stats": self._last_gate_stats,
            "gate_weights": self._last_gate_weights,
            "gate_biases": self._last_gate_biases,
        }


class GraphMambaStateEstimator(nn.Module):
    """
    Graph-Mamba for Distribution Network State Estimation.

    Architecture:
    1. GNN Encoder: Captures spatial dependencies from power network topology
    2. Mamba Temporal Encoder: Models time-varying dynamics via SSM
    3. Prediction Head: Outputs voltage magnitudes and angles
    """

    def __init__(
        self,
        num_buses,
        num_pmu_buses,
        d_model=128,
        d_state=32,
        n_gnn_layers=3,
        use_mamba=True,
        dropout=0.1,
        n_temporal_layers=1,
        bidirectional_mamba=False,
        architecture="legacy",
        pmu_bus_indices=None,
        gnn_type="gatv2",
        use_transformer=False,
        mamba_clamp=0.0,
        readout_mode="graph",
        scada_features=3,
        require_mamba=False,
        entropy_gate_w_init=0.0,
        entropy_gate_b_init=0.0,
    ):
        super().__init__()
        self.num_buses = num_buses
        self.num_pmu_buses = num_pmu_buses
        self.d_model = d_model
        self.use_mamba = use_mamba and HAS_MAMBA
        self.n_temporal_layers = n_temporal_layers
        self.bidirectional_mamba = bidirectional_mamba and self.use_mamba
        self.architecture = architecture
        self.mamba_clamp = mamba_clamp

        paper_architectures = {
            "paper_nodewise_gat_mamba",
            "paper_global_gat_mamba",
            "paper_nodewise_gcn_mamba",
            "paper_global_gcn_mamba",
            "paper_nodewise_gat_lstm",
            "paper_global_gat_lstm",
            "paper_nodewise_gat_transformer",
            "paper_global_gat_transformer",
            "paper_nodewise_gat_mamba_nogate",
            "paper_global_gat_mamba_nogate",
        }
        if self.architecture in paper_architectures:
            encoder_mode = "global" if "global" in self.architecture else "nodewise"
            paper_gnn_type = "gcn" if "gcn" in self.architecture else gnn_type
            paper_use_transformer = "transformer" in self.architecture
            paper_use_mamba = (
                use_mamba
                and "lstm" not in self.architecture
                and "transformer" not in self.architecture
            )
            paper_entropy_gate = (
                "gat" in self.architecture and "nogate" not in self.architecture
            )
            self.paper_backbone = PaperGATMambaBackbone(
                num_buses=num_buses,
                num_pmu_buses=num_pmu_buses,
                d_model=d_model,
                d_state=d_state,
                n_gnn_layers=n_gnn_layers,
                use_mamba=paper_use_mamba,
                use_transformer=paper_use_transformer,
                bidirectional_mamba=bidirectional_mamba
                or ("mamba" in self.architecture),
                gnn_type=paper_gnn_type,
                entropy_gate=paper_entropy_gate,
                encoder_mode=encoder_mode,
                dropout=dropout,
                n_temporal_layers=n_temporal_layers,
                pmu_bus_indices=pmu_bus_indices,
                mamba_clamp=mamba_clamp,
                readout_mode=readout_mode,
                scada_features=scada_features,
                require_mamba=require_mamba,
                entropy_gate_w_init=entropy_gate_w_init,
                entropy_gate_b_init=entropy_gate_b_init,
            )
            return

        if self.architecture == "node_temporal":
            self.redesign = NodeAwareGraphMambaBackbone(
                num_buses=num_buses,
                num_pmu_buses=num_pmu_buses,
                d_model=d_model,
                d_state=d_state,
                n_gnn_layers=n_gnn_layers,
                use_mamba=use_mamba,
                dropout=dropout,
                n_temporal_layers=max(2, n_temporal_layers),
                pmu_bus_indices=pmu_bus_indices,
            )
            return

        if self.architecture == "robust_v3":
            self.robust_backbone = RobustGraphMambaBackbone(
                num_buses=num_buses,
                num_pmu_buses=num_pmu_buses,
                d_model=d_model,
                d_state=d_state,
                n_gnn_layers=n_gnn_layers,
                use_mamba=use_mamba,
                use_transformer=use_transformer,
                gnn_type=gnn_type,
                dropout=dropout,
                n_temporal_layers=max(2, n_temporal_layers),
                pmu_bus_indices=pmu_bus_indices,
            )
            return

        # Measurement input dimensions
        self.n_scada = scada_features * num_buses
        self.n_pmu = 2 * num_pmu_buses  # V, delta for PMU buses

        # Feature encoder: raw measurements -> d_model
        self.measurement_encoder = nn.Sequential(
            nn.Linear(self.n_scada + self.n_pmu, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Graph Encoder (GNN layers)
        if architecture == "apig_mamba":
            self.gnn_type_legacy = "gatv2"
            self.entropy_monitoring = True
        elif architecture == "legacy":
            self.gnn_type_legacy = gnn_type
            self.entropy_monitoring = False
        else:
            self.gnn_type_legacy = "gcn"
            self.entropy_monitoring = False

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(n_gnn_layers):
            if self.gnn_type_legacy == "sage":
                self.gnn_layers.append(SAGEConv(d_model, d_model))
            elif self.gnn_type_legacy == "gatv2":
                gat_heads = 4
                self.gnn_layers.append(
                    GATv2Conv(
                        d_model,
                        d_model // gat_heads,
                        heads=gat_heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            else:
                self.gnn_layers.append(GCNConv(d_model, d_model))
            self.gnn_norms.append(nn.LayerNorm(d_model))

        if self.entropy_monitoring:
            self.entropy_gate_w = nn.Parameter(torch.full((n_gnn_layers,), -3.0))
            self.entropy_gate_b = nn.Parameter(torch.full((n_gnn_layers,), 2.0))
            self._last_entropy_stats = None

        self.use_transformer_legacy = (
            use_transformer and not self.use_mamba and architecture == "legacy"
        )

        # Temporal Encoder (Mamba, Transformer, or LSTM fallback)
        if self.use_mamba:
            if self.n_temporal_layers == 1 and not self.bidirectional_mamba:
                # Preserve the original single-layer checkpoint-compatible path.
                self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                self.temporal_norm = nn.LayerNorm(d_model)
            else:
                self.mamba_layers = nn.ModuleList(
                    [
                        Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                        for _ in range(self.n_temporal_layers)
                    ]
                )
                self.temporal_norms = nn.ModuleList(
                    [nn.LayerNorm(d_model) for _ in range(self.n_temporal_layers)]
                )
                if self.bidirectional_mamba:
                    self.backward_mamba_layers = nn.ModuleList(
                        [
                            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                            for _ in range(self.n_temporal_layers)
                        ]
                    )
                    self.backward_temporal_norms = nn.ModuleList(
                        [nn.LayerNorm(d_model) for _ in range(self.n_temporal_layers)]
                    )
                    self.mamba_proj = nn.Linear(d_model * 2, d_model)
        elif self.use_transformer_legacy:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        else:
            # LSTM fallback
            self.lstm = nn.LSTM(
                d_model,
                d_model,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(d_model * 2, d_model)

        # Graph pooling and prediction
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )

        # State prediction heads
        # Voltage magnitude: [0, ~1.5] -> use sigmoid scaled
        self.voltage_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_buses),
            nn.Sigmoid(),  # Output in [0, 1], will scale to typical voltage range
        )

        # Voltage angle: [-pi, pi] -> use tanh scaled
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_buses),
            nn.Tanh(),  # Output in [-1, 1], will scale to angle range
        )

        # Physics-informed loss weight
        self.lambda_phy = nn.Parameter(torch.tensor(0.1))

    def gnn_encode(self, x, edge_index):
        """
        Apply GNN layers to encode spatial dependencies.

        Args:
            x: [batch * seq, nodes, d_model]
            edge_index: [2, num_edges]
        Returns:
            x: [batch * seq, nodes, d_model] encoded features
        """
        batch_seq, num_nodes, _ = x.shape

        # Reshape for GCN: [batch_seq * nodes, d_model]
        x_flat = x.reshape(batch_seq * num_nodes, -1)

        # Build batched edge index so each (batch, time) snapshot has its own graph
        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_seq)
        offsets = torch.arange(batch_seq, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(num_edges)
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)

        # Apply GNN layers
        for gcn_layer, norm_layer in zip(self.gnn_layers, self.gnn_norms):
            x_gcn = gcn_layer(x_flat, edge_index_batch)
            x_gcn = F.gelu(x_gcn)
            x_flat = norm_layer(x_flat + x_gcn)  # Residual connection

        # Reshape back
        x = x_flat.reshape(batch_seq, num_nodes, -1)
        return x

    def gnn_encode_apig(self, x, edge_index):
        """
        Entropy-aware GNN encoding for APIG-Mamba.

        Extracts attention weights from GATv2Conv layers, computes per-node
        Shannon entropy, and uses a learned gate to adaptively modulate GNN
        influence. When attention entropy is high (uncertain aggregation due to
        corrupted inputs), the gate reduces graph influence and preserves
        node-local features.

        Args:
            x: [batch_seq, nodes, d_model]
            edge_index: [2, num_edges]
        Returns:
            x: [batch_seq, nodes, d_model] entropy-gated encoded features
        """
        batch_seq, num_nodes, _ = x.shape
        num_nodes_bt = batch_seq * num_nodes
        x_flat = x.reshape(num_nodes_bt, -1)

        num_edges = edge_index.size(1)
        edge_index_batch = edge_index.repeat(1, batch_seq)
        offsets = torch.arange(batch_seq, device=edge_index.device) * num_nodes
        offsets = offsets.repeat_interleave(num_edges)
        edge_index_batch = edge_index_batch + offsets.unsqueeze(0)

        entropy_stats = []

        for layer_idx, (gat_layer, norm_layer) in enumerate(
            zip(self.gnn_layers, self.gnn_norms)
        ):
            x_gnn, (ei_out, alpha) = gat_layer(
                x_flat, edge_index_batch, return_attention_weights=True
            )
            x_gnn = F.gelu(x_gnn)

            target_nodes = ei_out[1]
            alpha_clamped = torch.clamp(alpha, min=1e-8)
            edge_entropy = -(alpha_clamped * torch.log(alpha_clamped)).mean(dim=-1)

            node_entropy = torch.zeros(num_nodes_bt, device=x_flat.device)
            node_count = torch.zeros(num_nodes_bt, device=x_flat.device)
            node_entropy.scatter_add_(0, target_nodes, edge_entropy)
            node_count.scatter_add_(0, target_nodes, torch.ones_like(edge_entropy))
            node_entropy = node_entropy / (node_count + 1e-8)

            entropy_stats.append(node_entropy.mean().item())

            gate = torch.sigmoid(
                self.entropy_gate_w[layer_idx] * node_entropy.unsqueeze(-1)
                + self.entropy_gate_b[layer_idx]
            )
            x_flat = norm_layer(gate * x_gnn + (1.0 - gate) * x_flat)

        self._last_entropy_stats = entropy_stats
        x = x_flat.reshape(batch_seq, num_nodes, -1)
        return x

    def temporal_encode(self, x):
        """
        Apply temporal encoding (Mamba or LSTM).

        Args:
            x: [batch, seq, d_model]
        Returns:
            x: [batch, d_model] temporal features
        """
        if self.use_mamba:
            if self.n_temporal_layers == 1 and not self.bidirectional_mamba:
                x_in = (
                    torch.clamp(x, -self.mamba_clamp, self.mamba_clamp)
                    if self.mamba_clamp > 0
                    else x
                )
                x_mamba = self.mamba(x_in)
                x = self.temporal_norm(x + x_mamba)
                x = x[:, -1, :]
            else:
                x_forward = x
                for mamba_layer, norm_layer in zip(
                    self.mamba_layers, self.temporal_norms
                ):
                    x_in = (
                        torch.clamp(x_forward, -self.mamba_clamp, self.mamba_clamp)
                        if self.mamba_clamp > 0
                        else x_forward
                    )
                    x_layer = mamba_layer(x_in)
                    x_forward = norm_layer(x_forward + x_layer)

                if self.bidirectional_mamba:
                    x_backward = torch.flip(x, dims=[1])
                    for mamba_layer, norm_layer in zip(
                        self.backward_mamba_layers, self.backward_temporal_norms
                    ):
                        x_in = (
                            torch.clamp(x_backward, -self.mamba_clamp, self.mamba_clamp)
                            if self.mamba_clamp > 0
                            else x_backward
                        )
                        x_layer = mamba_layer(x_in)
                        x_backward = norm_layer(x_backward + x_layer)
                    x_backward = torch.flip(x_backward, dims=[1])
                    x = torch.cat([x_forward[:, -1, :], x_backward[:, -1, :]], dim=-1)
                    x = self.mamba_proj(x)
                else:
                    x = x_forward[:, -1, :]
        elif self.use_transformer_legacy:
            x = self.temporal_transformer(x)
            x = x[:, -1, :]
        else:
            # LSTM fallback
            x_lstm, _ = self.lstm(x)
            x = self.temporal_proj(x_lstm[:, -1, :])

        return x

    def forward(
        self,
        measurements_scada,
        measurements_pmu,
        edge_index,
        scada_mask=None,
        pmu_mask=None,
    ):
        """
        Forward pass for state estimation.

        Args:
            measurements_scada: [batch, seq, 3 * n_buses] - P, Q, V for all buses
            measurements_pmu: [batch, seq, 2 * n_pmu_buses] - V, delta for PMU buses
            edge_index: [2, num_edges] - Graph connectivity

        Returns:
            Dictionary with:
            - V_pred: [batch, n_buses] predicted voltage magnitudes (pu)
            - delta_pred: [batch, n_buses] predicted voltage angles (rad)
            - features: intermediate features
        """
        if hasattr(self, "paper_backbone"):
            return self.paper_backbone(
                measurements_scada,
                measurements_pmu,
                edge_index,
                scada_mask=scada_mask,
                pmu_mask=pmu_mask,
            )

        if self.architecture == "node_temporal":
            return self.redesign(measurements_scada, measurements_pmu, edge_index)

        if self.architecture == "robust_v3":
            return self.robust_backbone(
                measurements_scada, measurements_pmu, edge_index
            )

        batch_size, seq_len, _ = measurements_scada.shape

        # Concatenate measurements
        measurements = torch.cat([measurements_scada, measurements_pmu], dim=-1)

        # Encode measurements: [batch, seq, n_meas] -> [batch, seq, d_model]
        x_encoded = []
        for t in range(seq_len):
            x_t = self.measurement_encoder(measurements[:, t])
            x_encoded.append(x_t)
        x = torch.stack(x_encoded, dim=1)  # [batch, seq, d_model]

        # Expand to per-node features: [batch, seq, nodes, d_model]
        x = x.unsqueeze(2).expand(-1, -1, self.num_buses, -1)

        # Spatial encoding (GNN)
        # Reshape: [batch * seq, nodes, d_model]
        x = x.reshape(batch_size * seq_len, self.num_buses, self.d_model)
        if self.entropy_monitoring:
            x = self.gnn_encode_apig(x, edge_index)
        else:
            x = self.gnn_encode(x, edge_index)

        # Reshape for temporal: [batch, seq, nodes, d_model] -> [batch, seq, d_model]
        # Pool across nodes for temporal processing
        x_pooled = x.mean(dim=1)  # [batch * seq, d_model]
        x_pooled = x_pooled.reshape(batch_size, seq_len, self.d_model)

        # Temporal encoding (Mamba/LSTM)
        x_temp = self.temporal_encode(x_pooled)  # [batch, d_model]

        # Final pooling
        x_out = self.pool(x_temp)

        # Predict states
        V_pred = self.voltage_head(x_out)  # [batch, n_buses], in [0, 1]
        V_pred = V_pred * 1.2  # Scale to typical voltage range [0, 1.2] pu

        delta_pred = self.angle_head(x_out) * np.pi  # [batch, n_buses], in [-pi, pi]

        return {"V_pred": V_pred, "delta_pred": delta_pred, "features": x_out}

    def compute_physics_loss(self, V_pred, delta_pred, measurements_scada, Y_bus):
        """
        Physics-informed loss based on power flow equations.

        P_i = V_i * sum_j(V_j * (G_ij * cos(delta_ij) + B_ij * sin(delta_ij)))
        Q_i = V_i * sum_j(V_j * (G_ij * sin(delta_ij) - B_ij * cos(delta_ij)))
        """
        batch_size = V_pred.shape[0]
        n_buses = self.num_buses

        G = torch.tensor(Y_bus.real, dtype=torch.float32, device=V_pred.device)
        B = torch.tensor(Y_bus.imag, dtype=torch.float32, device=V_pred.device)

        # Compute power injections from predicted states
        P_calc = torch.zeros(batch_size, n_buses, device=V_pred.device)
        Q_calc = torch.zeros(batch_size, n_buses, device=V_pred.device)

        for b in range(batch_size):
            V = V_pred[b]
            delta = delta_pred[b]

            for i in range(n_buses):
                for j in range(n_buses):
                    delta_ij = delta[i] - delta[j]
                    P_calc[b, i] += (
                        V[i]
                        * V[j]
                        * (
                            G[i, j] * torch.cos(delta_ij)
                            + B[i, j] * torch.sin(delta_ij)
                        )
                    )
                    Q_calc[b, i] += (
                        V[i]
                        * V[j]
                        * (
                            G[i, j] * torch.sin(delta_ij)
                            - B[i, j] * torch.cos(delta_ij)
                        )
                    )

        # Compare with measured P, Q
        P_meas = measurements_scada[:, :, :n_buses].mean(dim=1)  # Average over time
        Q_meas = measurements_scada[:, :, n_buses : 2 * n_buses].mean(dim=1)

        loss_p = F.mse_loss(P_calc, P_meas)
        loss_q = F.mse_loss(Q_calc, Q_meas)

        return loss_p + loss_q


class GraphMambaStateEstimatorV2(nn.Module):
    """
    Enhanced Graph-Mamba with multi-scale temporal modeling.
    Uses multiple Mamba layers with different time constants.
    """

    def __init__(
        self,
        num_buses,
        num_pmu_buses,
        d_model=128,
        d_state=32,
        n_gnn_layers=3,
        n_mamba_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_buses = num_buses
        self.num_pmu_buses = num_pmu_buses
        self.d_model = d_model
        self.use_mamba = HAS_MAMBA

        # Input encoding
        self.n_scada = 3 * num_buses
        self.n_pmu = 2 * num_pmu_buses

        self.encoder = nn.Sequential(
            nn.Linear(self.n_scada + self.n_pmu, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Node embedding (learnable per-bus parameters)
        self.node_embedding = nn.Parameter(torch.randn(num_buses, d_model // 4))

        # GNN layers
        self.gnn_layers = nn.ModuleList(
            [
                GCNConv(d_model + d_model // 4 if i == 0 else d_model, d_model)
                for i in range(n_gnn_layers)
            ]
        )
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_gnn_layers)]
        )

        # Multi-scale Mamba layers
        if self.use_mamba:
            self.mamba_layers = nn.ModuleList(
                [
                    Mamba(
                        d_model=d_model, d_state=d_state * (i + 1), d_conv=4, expand=2
                    )
                    for i in range(n_mamba_layers)
                ]
            )
            self.mamba_norms = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(n_mamba_layers)]
            )
        else:
            self.lstm = nn.LSTM(
                d_model, d_model, n_mamba_layers, batch_first=True, dropout=dropout
            )

        # Prediction heads with skip connections
        self.voltage_head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_buses),
            nn.Sigmoid(),
        )

        self.angle_head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_buses),
            nn.Tanh(),
        )

    def forward(self, measurements_scada, measurements_pmu, edge_index):
        batch_size, seq_len, _ = measurements_scada.shape

        # Encode measurements
        measurements = torch.cat([measurements_scada, measurements_pmu], dim=-1)
        x = self.encoder(measurements)  # [batch, seq, d_model]

        # Expand and add node embeddings
        x = x.unsqueeze(2).expand(
            -1, -1, self.num_buses, -1
        )  # [batch, seq, nodes, d_model]
        node_emb = (
            self.node_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, seq_len, -1, -1)
        )
        x = torch.cat([x, node_emb], dim=-1)  # [batch, seq, nodes, d_model + node_emb]

        # GNN processing
        x = x.reshape(batch_size * seq_len, self.num_buses, -1)
        x_flat = x.reshape(batch_size * seq_len * self.num_buses, -1)

        for gcn_layer, norm_layer in zip(self.gnn_layers, self.gnn_norms):
            x_gcn = gcn_layer(x_flat, edge_index)
            x_gcn = F.gelu(x_gcn)
            x_flat = x_flat[:, : self.d_model]  # Trim to match dimensions
            x_flat = norm_layer(x_flat + x_gcn)

        x = x_flat.reshape(batch_size, seq_len, self.num_buses, self.d_model)

        # Pool across nodes for temporal processing
        x = x.mean(dim=2)  # [batch, seq, d_model]

        # Temporal encoding
        if self.use_mamba:
            for mamba_layer, norm_layer in zip(self.mamba_layers, self.mamba_norms):
                x_mamba = mamba_layer(x)
                x = norm_layer(x + x_mamba)
        else:
            x, _ = self.lstm(x)

        # Use last timestep
        x = x[:, -1, :]  # [batch, d_model]

        # Concatenate with pooled node embedding for prediction
        node_emb_pooled = (
            self.node_embedding.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        )
        x = torch.cat([x, node_emb_pooled], dim=-1)

        # Predict
        V_pred = self.voltage_head(x) * 1.2
        delta_pred = self.angle_head(x) * np.pi

        return {"V_pred": V_pred, "delta_pred": delta_pred}


if __name__ == "__main__":
    # Test models
    num_buses = 123
    num_pmu = 12
    batch_size = 4
    seq_len = 20

    # Test Graph-Mamba State Estimator
    model = GraphMambaStateEstimator(num_buses, num_pmu, d_model=64)

    measurements_scada = torch.randn(batch_size, seq_len, 3 * num_buses)
    measurements_pmu = torch.randn(batch_size, seq_len, 2 * num_pmu)
    edge_index = torch.randint(0, num_buses, (2, 200))

    output = model(measurements_scada, measurements_pmu, edge_index)

    print("Graph-Mamba State Estimator Test:")
    print(f"  V_pred shape: {output['V_pred'].shape}")
    print(f"  delta_pred shape: {output['delta_pred'].shape}")
    print(f"  Using Mamba: {model.use_mamba}")
