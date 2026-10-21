"""Experiment configuration, random seeds and environment information."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np
import torch


PACKAGE_NAMES = (
    "numpy",
    "opendssdirect.py",
    "torch",
    "torch-geometric",
    "mamba-ssm",
    "tqdm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def portable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the packaged feeder without changing experiment settings.

    Archived configurations may record a different package prefix. Their
    original content and hashes are verified before this in-memory copy is
    used; only the location of the same IEEE 123-node feeder is normalized.
    """
    location = config.get("data", {}).get("dss_dir")
    if not isinstance(location, str) or not location:
        raise ValueError("A complete relative IEEE 123-node feeder path is required")
    normalized = location.replace("\\", "/")
    parts = normalized.split("/")
    if (PureWindowsPath(location).drive or normalized.startswith("/")
            or any(part in ("", ".", "..") for part in parts)
            or parts[-2:] != ["data", "ieee123"]):
        raise ValueError("A complete relative IEEE 123-node feeder path is required")
    feeder_dir = Path(__file__).resolve().parent / "data" / "ieee123"
    if not (feeder_dir / "IEEE123Master.dss").is_file():
        raise FileNotFoundError("The packaged IEEE 123-node feeder is incomplete")
    verify_data_manifest(feeder_dir)
    result = deepcopy(config)
    result["data"]["dss_dir"] = "dsse/data/ieee123"
    return result


def public_path(path: str | Path, repo_root: Path) -> str:
    """Describe a path without publishing its machine-specific parent folders.

    These labels are provenance only; they are never used to locate inputs.
    File hashes and package-relative paths retain the reproducibility identity.
    Windows paths are also recognized when a report is replayed on Linux.
    """
    value = str(path)
    if value == "~" or value.startswith(("~/", "~\\")):
        return "<external_path>"
    if PureWindowsPath(value).is_absolute():
        try:
            return PureWindowsPath(value).relative_to(PureWindowsPath(repo_root)).as_posix() or "."
        except ValueError:
            return "<external_path>"
    candidate = Path(value)
    if value.startswith("/") and not candidate.is_absolute():
        return "<external_path>"
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(Path(repo_root).resolve()).as_posix() or "."
    except ValueError:
        return "<external_path>"


def public_command(command: Iterable[str], repo_root: Path) -> list[str]:
    """Keep CLI options and package paths, without absolute user/home paths."""
    result = []
    for argument in command:
        argument = str(argument)
        option, separator, value = argument.partition("=")
        if separator and option.startswith("-"):
            result.append(option + "=" + public_path(value, repo_root))
        else:
            result.append(public_path(argument, repo_root))
    return result


def expand_seed_spec(spec: dict[str, int]) -> list[int]:
    start = int(spec["start"])
    count = int(spec["count"])
    return list(range(start, start + count))


def assert_disjoint_splits(split_groups: dict[str, Iterable[int]]) -> None:
    names = list(split_groups)
    for index, name in enumerate(names):
        left = set(int(seed) for seed in split_groups[name])
        if len(left) != len(list(split_groups[name])):
            raise AssertionError(f"Duplicate trajectory seed inside {name}")
        for other in names[index + 1 :]:
            overlap = left.intersection(int(seed) for seed in split_groups[other])
            if overlap:
                raise AssertionError(
                    f"Trajectory leakage between {name} and {other}: {sorted(overlap)}"
                )


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def git_state(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain")
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def environment_state() -> dict[str, Any]:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def verify_data_manifest(data_dir: Path) -> dict[str, str]:
    manifest_path = Path(data_dir) / "upstream_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {}
    for name, expected in manifest["files"].items():
        path = Path(data_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing vendored feeder file: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Feeder hash mismatch for {name}: expected {expected}, got {observed}"
            )
        actual[name] = observed
    return actual


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_run_manifest(
    repo_root: Path,
    command: list[str],
    config: dict[str, Any],
    data_hashes: dict[str, str],
    splits: dict[str, list[int]],
    mamba_backend: str,
) -> dict[str, Any]:
    public_argv = public_command(command, repo_root)
    return {
        "schema_version": 2,
        "command": public_argv,
        "execution": {
            "argv": public_argv,
            "python_executable": Path(sys.executable).name,
            "working_directory": public_path(Path.cwd().resolve(), repo_root),
            "repo_root": ".",
            "path_policy": "Package-relative paths; external absolute paths omitted. No user/home directory is recorded.",
        },
        "config": config,
        "config_sha256": canonical_sha256(config),
        "data_hashes": data_hashes,
        "split_seeds": splits,
        "split_sha256": canonical_sha256(splits),
        "git": git_state(repo_root),
        "environment": environment_state(),
        "mamba_backend": mamba_backend,
    }
