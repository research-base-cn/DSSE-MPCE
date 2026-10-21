# Training

The training code implements three stages: backbone optimization, gate fitting, and residual-readout calibration. Datasets, pretrained checkpoints, and fitted calibration files are not included. The catalog-based entry points require a local catalog with valid paths and hashes for those inputs; `metadata/models.json` documents the study catalog, rather than providing its numerical assets.

## Backbone, gate, and readout

`training/backbone.py` trains the selected backbone. The backbone uses up to 200 epochs, batch size 16, AdamW learning rate `5e-4`, weight decay `1e-5`, gradient clipping at 1, and cosine decay to `1e-6`. The selected checkpoint minimizes normal-validation voltage-plus-angle RMSE, with patience 30. See [backbone settings](../configs/backbone.json) for the complete recipe.

`training/gate.py` freezes the selected backbone, disables dropout, and trains the gate for 30 epochs. It uses batch size 16, AdamW learning rate 0.003, zero weight decay, gradient clipping at 1, and cosine decay to 0.0001. Gate normalization and loss scales are computed from the first 64 normal training trajectories. The terminal epoch-30 state is used, followed by the fixed coverage anchor. See [gate settings](../configs/gate.json).

`training/readout.py` calibrates inverse-link residuals using raw measurements and the parent's hidden representation. It uses four observation views of 2,048 training trajectories: normal, 50% PMU dropout, 30% PMU outliers, and joint 30% P/Q–PMU outliers. The mean-loss ridge strength is `1e-4`, the standard-deviation floor is `1e-6`, and the intercept is unpenalized. Coefficients and residual accumulation use FP64; native neural logits and output links use FP32. The adaptive readout also emits raw-only, hidden-only, and normal-only calibration variants.

The study uses model seeds `131`, `197`, `269`, `349`, and `443`. Training uses training labels, and checkpoint selection uses validation labels; confirmation targets are excluded from fitting and selection. Newly generated parameters need their own catalog entries and hashes before catalog-based inference.

## Structural and gate ablations

| Variant | Model |
|---|---|
| `gat_mamba_no_gate` | GAT–Mamba without the adaptive gate |
| `gat_mamba_full_gate` | Global-gate GAT–Mamba |
| `corrected_gcn_mamba` | GCN–Mamba |
| `matched_gat_bilstm` | Capacity-matched GAT–BiLSTM |
| `gnn_se_snapshot` | Snapshot graph estimator |
| `sequence_mlp` | Sequence MLP |

Structural controls retain paired initialization and their specified recipes in [structural_controls.json](../configs/structural_controls.json). Snapshot and MLP variants use the recipes in the base configuration.

The gate modes `adaptive`, `quality_only`, and `constant` correspond to inference variants `r4_adaptive`, `r4_quality_only`, and `r4_constant`. Each starts from the same-seed no-gate backbone.

## Common-head gate controls

`training/common_head.py` fits a common head using the four views of each parent, giving 16,384 feature rows from 2,048 distinct labels. It uses the same ridge settings and records training-only mean gate gains. Native and fixed-head controls have different interpretations from independently calibrated-head comparisons.

## Short execution checks

After preparing the local input artifacts, append `--smoke` to a training entry point to run a reduced computation. These outputs are unsuitable for producing comparison tables. `training.backbone --contract-only` constructs the paired network without optimization, but still requires its configured local inputs.
