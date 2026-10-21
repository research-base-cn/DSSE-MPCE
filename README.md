# DSSE-MPCE

GAT–Mamba for distribution system state estimation with sparse, missing, and corrupted measurements.

The model combines graph attention, temporal Mamba blocks, an adaptive measurement-quality gate, and a calibrated residual readout. This repository contains the model, training routines, IEEE 123-node feeder simulation, and comparison-estimator implementations.

## Installation

```sh
git clone https://github.com/research-base-cn/DSSE-MPCE.git
cd DSSE-MPCE
python -m pip install -r requirements-data.txt
```

Data generation requires NumPy, OpenDSSDirect.py, and PyTorch; a CPU build of PyTorch is sufficient. Neural training additionally requires PyTorch Geometric, Mamba, and a compatible CUDA installation. Numerical dependencies and the recorded experimental environment are listed in [requirements-numerical.txt](requirements-numerical.txt) and [environment.json](environment.json).

## Data

The included OpenDSS feeder files and simulation code can be used to generate new datasets locally.

## Generate data from the IEEE 123-node feeder

The feeder assets are in [src/dsse/data/ieee123](src/dsse/data/ieee123), and [configs/model.json](configs/model.json) specifies the PMU locations, trajectory length, and random seeds. For each trajectory, the simulator perturbs nominal load P/Q using correlated common and per-load variations, clips the load multipliers to `[0.55, 1.45]`, and solves the OpenDSS power flow at each time step. It records bus-load P/Q pseudo-measurements, voltage magnitude and angle at 12 PMU locations, availability masks, and voltage/angle targets. The bus-level targets use phase phasors aligned to phase A and averaged before taking magnitude and angle; angles are in radians.

From the repository root, run:

```sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m dsse.datasets --config configs/model.json --output data/training
python -m dsse.datasets --output data/training --verify-only
```

In PowerShell, set the import path with `$env:PYTHONPATH = "$PWD/src"` instead of the `export` line. Use a clean Git worktree and an empty output directory for the full run.

The configuration generates 2,048 training, 256 validation, and 512 `sealed_test` trajectories, each with 30 time steps. The generator saves compressed NumPy shards, training-only normalization statistics, topology, bus/PMU mappings, seeds, and checksums. Measurement noise, missingness, and outliers are applied after clean trajectory generation. Their implementations are in [simulation/case_definitions.py](simulation/case_definitions.py), [simulation/corruption_models.py](simulation/corruption_models.py), and [src/dsse/runtime.py](src/dsse/runtime.py). 

## Model and training

The network and gate implementations are in `src/graphmamba/` and `src/gating/`. Optimization settings are in `configs/backbone.json` and `configs/gate.json`; [training/README.md](training/README.md) describes the training stages and ablations.

The study-specific `training/` and `neural/` entry points require local datasets, calibration files, and a model catalog pointing to locally available parameters. `metadata/models.json` records the study's catalog structure and file identities. Its referenced numerical assets are not included. 
## Repository structure

| Directory | Contents |
|---|---|
| `src/` | Network, gate, readout, classical-estimator, and physical-estimator implementations |
| `simulation/` | Observation scenarios and simulation utilities |
| `training/` | Backbone training, gate fitting, and readout calibration |
| `neural/` | Catalog-based model loading and inference |
| `configs/` | Model, optimization, structural-control, and sensor-layout settings |
| `metadata/` | Study model catalog and bus-phase mapping |

## License and third-party material

A license for the original research code has not yet been specified. The included IEEE 123-node/OpenDSS feeder files retain the Electric Power Research Institute copyright and license notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [the feeder license](src/dsse/data/ieee123/LICENSE.opendss.txt).
