"""Train neural state estimators and select validation checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dsse.models import (
    HAS_MAMBA,
    NEURAL_VARIANTS,
    build_formal_model,
    parameter_count,
)
from dsse.runtime import (
    TrajectoryCache,
    corrupt_formal_sample,
    sample_at,
    scenario_seed,
    tensorize_last_state,
    training_corruption_seed,
)
from dsse.reproducibility import (
    build_run_manifest,
    canonical_sha256,
    git_state,
    portable_config,
    seed_everything,
    sha256_file,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


class AugmentedTrainingDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        normalizer: dict[str, Any],
        scenario_names: list[str],
        config: dict[str, Any],
        manifest: dict[str, Any],
        model_seed: int,
    ) -> None:
        self.arrays = arrays
        self.normalizer = normalizer
        self.scenario_names = list(scenario_names)
        self.config = config
        self.manifest = manifest
        self.model_seed = int(model_seed)
        self.epoch = 0
        if not self.scenario_names:
            raise ValueError("Formal training requires at least one augmentation scenario")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.arrays["trajectory_seed"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = sample_at(self.arrays, index)
        scenario_name = self.scenario_names[(self.epoch + index) % len(self.scenario_names)]
        scenario = self.config["primary_scenarios"][scenario_name]
        seed = training_corruption_seed(
            self.model_seed,
            self.epoch,
            sample["trajectory_seed"],
            scenario_name,
        )
        corrupted, _ = corrupt_formal_sample(
            sample,
            scenario_name,
            scenario,
            np.random.default_rng(seed),
            self.config,
            self.manifest,
        )
        return tensorize_last_state(corrupted, self.normalizer)


class FixedValidationDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        normalizer: dict[str, Any],
        config: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        self.arrays = arrays
        self.normalizer = normalizer
        self.config = config
        self.manifest = manifest

    def __len__(self) -> int:
        return int(self.arrays["trajectory_seed"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = sample_at(self.arrays, index)
        scenario_name = "normal"
        corrupted, _ = corrupt_formal_sample(
            sample,
            scenario_name,
            self.config["primary_scenarios"][scenario_name],
            np.random.default_rng(scenario_seed(sample["trajectory_seed"], scenario_name)),
            self.config,
            self.manifest,
        )
        return tensorize_last_state(corrupted, self.normalizer)


def to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def forward_model(
    model: nn.Module, batch: dict[str, torch.Tensor], edge_index: torch.Tensor
) -> dict[str, Any]:
    return model(
        batch["scada"],
        batch["pmu"],
        edge_index,
        scada_mask=batch["scada_mask"],
        pmu_mask=batch["pmu_mask"],
    )


def forward_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output = forward_model(model, batch, edge_index)
    loss = criterion(output["V_pred"], batch["v_true"]) + criterion(
        output["delta_pred"], batch["angle_true"]
    )
    return loss, output


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    voltage_errors: list[np.ndarray] = []
    angle_errors: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            loss, output = forward_loss(model, batch, edge_index, criterion)
            losses.append(float(loss.detach().cpu()))
            voltage_errors.append(
                (output["V_pred"] - batch["v_true"]).detach().cpu().numpy()
            )
            raw_angle = (
                output["delta_pred"] - batch["angle_true"]
            ).detach().cpu().numpy()
            angle_errors.append(np.angle(np.exp(1j * raw_angle)))
    voltage = np.concatenate(voltage_errors, axis=0)
    angle = np.concatenate(angle_errors, axis=0)
    return {
        "loss": float(np.mean(losses)),
        "voltage_rmse": float(np.sqrt(np.mean(voltage**2))),
        "angle_circular_rmse": float(np.sqrt(np.mean(angle**2))),
    }


def validate_job_request(
    config: dict[str, Any], variant: str, model_seed: int
) -> None:
    if config.get("status") != "PREDECLARED_NOT_YET_RUN":
        raise ValueError("Formal configuration is no longer in its predeclared state")
    configured_variants = tuple(config["neural_variants"])
    if tuple(configured_variants) != NEURAL_VARIANTS:
        raise ValueError("Formal neural variant registry drifted from the implementation")
    if variant not in configured_variants:
        raise ValueError(f"Unregistered formal neural variant: {variant}")
    seeds = [int(seed) for seed in config["training"]["model_seeds"]]
    if int(model_seed) not in seeds:
        raise ValueError(f"Unregistered formal model seed: {model_seed}")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Formal model seeds are not unique")
    training = config["training"]
    if not training.get("no_automatic_retry", False):
        raise ValueError("Formal config must forbid automatic retries")
    if not training.get("no_seed_replacement", False):
        raise ValueError("Formal config must forbid seed replacement")
    if config["model"]["validation_selection_score"] != (
        "voltage_rmse_plus_angle_circular_rmse"
    ):
        raise ValueError("Unknown formal validation selection score")


def make_criterion(config: dict[str, Any]) -> nn.Module:
    if str(config["training"]["loss"]).lower() != "mse":
        raise ValueError("The formal config predeclares MSE training loss")
    return nn.MSELoss()


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@contextmanager
def hard_deadline(seconds: int) -> Iterator[None]:
    if seconds <= 0:
        raise ValueError("Hard deadline must be positive")
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(f"Formal seed job exceeded its {seconds}-second hard deadline")

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _device_from_string(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA index does not exist: {index}")
        torch.cuda.set_device(index)
    return device


def run_training_job(
    *,
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
    variant: str,
    model_seed: int,
    device_name: str,
    allow_inadmissible_smoke: bool = False,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_job_request(config, variant, int(model_seed))
    if smoke_epochs is not None:
        if not allow_inadmissible_smoke:
            raise ValueError("A smoke epoch override requires inadmissible-smoke mode")
        if int(smoke_epochs) <= 0:
            raise ValueError("smoke_epochs must be positive")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"

    cache = TrajectoryCache(
        cache_dir, allow_inadmissible_smoke=allow_inadmissible_smoke
    )
    config = portable_config(config)
    if canonical_sha256(cache.config) != canonical_sha256(config):
        raise RuntimeError("Training config differs from the immutable cache config")
    if cache.formal_admissible and smoke_epochs is None and git_state(REPO_ROOT)["dirty"]:
        raise ValueError("Formal neural training requires a clean worktree")
    train_arrays = cache.load_split("train")
    validation_arrays = cache.load_split("validation")
    device = _device_from_string(device_name)
    seed_everything(int(model_seed))

    training = config["training"]
    epochs = int(smoke_epochs) if smoke_epochs is not None else int(training["epochs"])
    job_admissible = bool(cache.formal_admissible and smoke_epochs is None)
    scenarios = [str(name) for name in training["augmentation_scenarios"]]
    train_dataset = AugmentedTrainingDataset(
        train_arrays,
        cache.normalizer,
        scenarios,
        config,
        cache.manifest,
        int(model_seed),
    )
    validation_dataset = FixedValidationDataset(
        validation_arrays, cache.normalizer, config, cache.manifest
    )
    generator = torch.Generator().manual_seed(int(model_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = build_formal_model(
        config,
        num_buses=cache.num_buses,
        num_pmu_buses=cache.pmu_count,
        pmu_bus_indices=cache.pmu_bus_indices,
        variant=variant,
        device=device,
    )
    edge_index = torch.as_tensor(cache.edge_index, dtype=torch.long, device=device)
    criterion = make_criterion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    cache_manifest_sha = sha256_file(cache.manifest_path)
    model_parameters = parameter_count(model)
    mamba_backend = (
        "mamba_ssm"
        if variant in {"gat_mamba_full_gate", "gat_mamba_no_gate", "gcn_mamba"}
        else "not_applicable"
    )
    if variant == "gat_lstm":
        mamba_backend = "pytorch_lstm"
    run_manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        command=sys.argv,
        config=config,
        data_hashes=cache.manifest["data_hashes"],
        splits={
            "train": cache.manifest["split_seeds"]["train"],
            "validation": cache.manifest["split_seeds"]["validation"],
        },
        mamba_backend=mamba_backend,
    )
    run_manifest.update(
        {
            "job_schema_version": 1,
            "status": "RUNNING",
            "study_id": cache.study_id,
            "variant": variant,
            "model_seed": int(model_seed),
            "device_requested": device_name,
            "device_resolved": str(device),
            "cache_manifest_sha256": cache_manifest_sha,
            "cache_formal_admissible": cache.formal_admissible,
            "job_formal_admissible": job_admissible,
            "smoke_epochs": smoke_epochs,
            "epochs_effective": epochs,
            "parameter_count": model_parameters,
            "sealed_test_accessed": False,
            "has_mamba_package": bool(HAS_MAMBA),
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    append_event(
        progress_path,
        {
            "event": "training_start",
            "variant": variant,
            "model_seed": int(model_seed),
            "epochs": epochs,
            "job_formal_admissible": job_admissible,
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    best_score = math.inf
    best_epoch = 0
    best_validation: dict[str, float] | None = None
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        batch_losses: list[float] = []
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = forward_loss(model, batch, edge_index, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite formal training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation = evaluate_validation(
            model, validation_loader, edge_index, criterion, device
        )
        score = validation["voltage_rmse"] + validation["angle_circular_rmse"]
        epoch_record: dict[str, Any] = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "validation": validation,
            "selection_score": float(score),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        paper_backbone = getattr(model, "paper_backbone", None)
        if paper_backbone is not None and getattr(
            paper_backbone, "entropy_gate", False
        ):
            epoch_record["gate_w"] = (
                paper_backbone.entropy_gate_w.detach().cpu().tolist()
            )
            epoch_record["gate_b"] = (
                paper_backbone.entropy_gate_b.detach().cpu().tolist()
            )
        history.append(epoch_record)
        append_event(progress_path, epoch_record)
        if score < best_score:
            best_score = float(score)
            best_epoch = epoch
            best_validation = validation
            patience = 0
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "checkpoint_schema_version": 1,
                    "model_state_dict": model.state_dict(),
                    "variant": variant,
                    "model_seed": int(model_seed),
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "selection_score": best_score,
                    "config_sha256": canonical_sha256(config),
                    "cache_manifest_sha256": cache_manifest_sha,
                    "normalizer_sha256": cache.manifest["normalizer"]["sha256"],
                    "topology_sha256": cache.manifest["topology"]["sha256"],
                    "parameter_count": model_parameters,
                    "sealed_test_accessed": False,
                    "job_formal_admissible": job_admissible,
                },
            )
        else:
            patience += 1
            if patience >= int(training["early_stopping_patience"]):
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = float(time.perf_counter() - started)
    if best_validation is None or not checkpoint_path.is_file():
        raise RuntimeError("Formal training ended without a valid checkpoint")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    result = {
        "result_schema_version": 1,
        "status": "COMPLETED",
        "study_id": cache.study_id,
        "variant": variant,
        "model_seed": int(model_seed),
        "job_formal_admissible": job_admissible,
        "sealed_test_accessed": False,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "selection_score": best_score,
        "epochs_completed": len(history),
        "training_wall_time_seconds": wall_time,
        "peak_gpu_memory_bytes": peak_memory,
        "parameter_count": model_parameters,
        "checkpoint": {
            "relative_path": "checkpoint.pt",
            "sha256": checkpoint_sha,
        },
        "history": history,
    }
    write_json(output_dir / "training_result.json", result)
    run_manifest["status"] = "COMPLETED"
    run_manifest["checkpoint_sha256"] = checkpoint_sha
    run_manifest["training_result_sha256"] = sha256_file(
        output_dir / "training_result.json"
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    append_event(
        progress_path,
        {
            "event": "training_complete",
            "best_epoch": best_epoch,
            "checkpoint_sha256": checkpoint_sha,
            "training_wall_time_seconds": wall_time,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=NEURAL_VARIANTS, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--allow-inadmissible-smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    deadline = int(config["training"]["maximum_wall_time_seconds_per_seed_job"])
    with hard_deadline(deadline):
        result = run_training_job(
            config_path=args.config,
            cache_dir=args.cache,
            output_dir=args.output,
            variant=args.variant,
            model_seed=args.model_seed,
            device_name=args.device,
            allow_inadmissible_smoke=args.allow_inadmissible_smoke,
            smoke_epochs=args.smoke_epochs,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "variant": result["variant"],
                "model_seed": result["model_seed"],
                "job_formal_admissible": result["job_formal_admissible"],
                "best_epoch": result["best_epoch"],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
