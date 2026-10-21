"""Training-calibrated static and forecast-aided estimator baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ESTIMATOR_VARIANTS = (
    "per_bus_training_mean",
    "data_calibrated_linear_wls",
    "huber_irls_m_estimator",
    "forecast_aided_kalman",
    "forecast_aided_huber",
)


ARTIFACT_ARRAY_KEYS = (
    "state_mean",
    "state_components",
    "measurement_mean",
    "measurement_matrix",
    "measurement_variance",
    "transition_matrix",
    "transition_covariance",
    "initial_covariance",
    "state_eigenvalues",
)


def measurement_sequence(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed P/Q + PMU V/angle measurements and availability masks."""

    scada = np.asarray(sample["scada"], dtype=np.float64)[..., :2]
    scada_mask = np.asarray(sample["scada_mask"], dtype=np.float64)[..., :2]
    pmu = np.asarray(sample["pmu"], dtype=np.float64)
    pmu_mask = np.asarray(sample["pmu_mask"], dtype=np.float64)
    sequence_length = scada.shape[0]
    measurements = np.concatenate(
        [scada.reshape(sequence_length, -1), pmu.reshape(sequence_length, -1)],
        axis=1,
    )
    masks = np.concatenate(
        [
            scada_mask.reshape(sequence_length, -1),
            pmu_mask.reshape(sequence_length, -1),
        ],
        axis=1,
    )
    return measurements, masks


def training_matrices(
    arrays: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voltage = np.asarray(arrays["v_true"], dtype=np.float64)
    angle = np.asarray(arrays["angle_true"], dtype=np.float64)
    states = np.concatenate([voltage, angle], axis=-1)
    scada = np.asarray(arrays["scada"], dtype=np.float64)[..., :2]
    scada_mask = np.asarray(arrays["scada_mask"], dtype=np.float64)[..., :2]
    pmu = np.asarray(arrays["pmu"], dtype=np.float64)
    pmu_mask = np.asarray(arrays["pmu_mask"], dtype=np.float64)
    if not np.all(scada_mask > 0.5) or not np.all(pmu_mask > 0.5):
        raise ValueError("Estimator calibration requires complete clean training inputs")
    measurements = np.concatenate(
        [
            scada.reshape(*scada.shape[:2], -1),
            pmu.reshape(*pmu.shape[:2], -1),
        ],
        axis=-1,
    )
    noise = config["corruption_randomization"]["measurement_noise"]
    scada_sigma = np.maximum(
        np.abs(scada) * float(noise["scada_relative_sigma"]),
        float(noise["scada_minimum_sigma_pu"]),
    )
    scada_noise_variance = np.mean(scada_sigma**2, axis=(0, 1)).reshape(-1)
    pmu_noise_variance = np.tile(
        np.asarray(
            [noise["pmu_voltage_sigma_pu"] ** 2, noise["pmu_angle_sigma_rad"] ** 2],
            dtype=np.float64,
        ),
        pmu.shape[2],
    )
    nominal_noise_variance = np.concatenate(
        [scada_noise_variance, pmu_noise_variance]
    )
    return states, measurements, nominal_noise_variance


def fit_estimator_arrays(
    arrays: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit every classical baseline parameter from the training split only."""

    states, measurements, nominal_noise_variance = training_matrices(arrays, config)
    num_trajectories, sequence_length, state_dimension = states.shape
    measurement_dimension = measurements.shape[-1]
    hyper = config["estimator_hyperparameters"]
    latent_dimension = int(hyper["latent_dimension"])
    if latent_dimension <= 0 or latent_dimension > state_dimension:
        raise ValueError("Invalid estimator latent dimension")

    flat_states = states.reshape(-1, state_dimension)
    flat_measurements = measurements.reshape(-1, measurement_dimension)
    state_mean = flat_states.mean(axis=0)
    centered_states = flat_states - state_mean
    state_covariance = centered_states.T @ centered_states / max(
        centered_states.shape[0] - 1, 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(state_covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    state_components = eigenvectors[:, order[:latent_dimension]]
    for column in range(state_components.shape[1]):
        pivot = int(np.argmax(np.abs(state_components[:, column])))
        if state_components[pivot, column] < 0.0:
            state_components[:, column] *= -1.0
    latent = centered_states @ state_components

    measurement_mean = flat_measurements.mean(axis=0)
    centered_measurements = flat_measurements - measurement_mean
    measurement_ridge = float(hyper["measurement_model_ridge"])
    latent_gram = latent.T @ latent + measurement_ridge * np.eye(latent_dimension)
    measurement_matrix = np.linalg.solve(
        latent_gram, latent.T @ centered_measurements
    ).T
    measurement_residual = centered_measurements - latent @ measurement_matrix.T
    residual_variance = np.mean(measurement_residual**2, axis=0)
    minimum_variance = float(hyper["minimum_variance"])
    measurement_variance = np.maximum(
        residual_variance + nominal_noise_variance, minimum_variance
    )

    latent_sequences = latent.reshape(num_trajectories, sequence_length, latent_dimension)
    previous = latent_sequences[:, :-1].reshape(-1, latent_dimension)
    following = latent_sequences[:, 1:].reshape(-1, latent_dimension)
    transition_ridge = float(hyper["transition_ridge"])
    transition_matrix = np.linalg.solve(
        previous.T @ previous + transition_ridge * np.eye(latent_dimension),
        previous.T @ following,
    ).T
    transition_residual = following - previous @ transition_matrix.T
    transition_covariance = (
        transition_residual.T @ transition_residual
        / max(transition_residual.shape[0] - 1, 1)
    )
    transition_covariance += minimum_variance * np.eye(latent_dimension)
    initial_latent = latent_sequences[:, 0]
    initial_covariance = (
        initial_latent.T @ initial_latent / max(initial_latent.shape[0] - 1, 1)
    )
    initial_covariance += minimum_variance * np.eye(latent_dimension)

    artifact = {
        "state_mean": state_mean,
        "state_components": state_components,
        "measurement_mean": measurement_mean,
        "measurement_matrix": measurement_matrix,
        "measurement_variance": measurement_variance,
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "initial_covariance": initial_covariance,
        "state_eigenvalues": eigenvalues,
    }
    artifact = {
        key: np.asarray(value, dtype=np.float64) for key, value in artifact.items()
    }
    total_variance = float(np.sum(eigenvalues))
    diagnostics = {
        "num_training_trajectories": int(num_trajectories),
        "sequence_length": int(sequence_length),
        "state_dimension": int(state_dimension),
        "measurement_dimension": int(measurement_dimension),
        "latent_dimension": int(latent_dimension),
        "state_variance_explained_ratio": float(
            np.sum(eigenvalues[:latent_dimension]) / max(total_variance, 1e-30)
        ),
        "measurement_residual_rmse": float(
            np.sqrt(np.mean(measurement_residual**2))
        ),
        "transition_residual_rmse": float(
            np.sqrt(np.mean(transition_residual**2))
        ),
        "fit_split": "train",
        "previous_ground_truth_used": False,
    }
    return artifact, diagnostics


def write_estimator_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(ARTIFACT_ARRAY_KEYS):
        raise ValueError("Estimator artifact arrays do not match the frozen schema")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def read_estimator_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(ARTIFACT_ARRAY_KEYS):
            raise ValueError("Estimator artifact schema mismatch")
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite estimator artifact array: {key}")
    return arrays


@dataclass
class EstimatorPrediction:
    voltage: np.ndarray
    angle: np.ndarray
    converged: bool
    failure_reason: str | None
    iteration_count: int


class LatentLinearEstimators:
    def __init__(self, artifact: dict[str, np.ndarray], config: dict[str, Any]) -> None:
        if set(artifact) != set(ARTIFACT_ARRAY_KEYS):
            raise ValueError("Incomplete estimator artifact")
        self.artifact = artifact
        self.hyper = config["estimator_hyperparameters"]
        self.state_mean = artifact["state_mean"]
        self.components = artifact["state_components"]
        self.measurement_mean = artifact["measurement_mean"]
        self.measurement_matrix = artifact["measurement_matrix"]
        self.measurement_variance = artifact["measurement_variance"]
        self.transition = artifact["transition_matrix"]
        self.transition_covariance = artifact["transition_covariance"]
        self.initial_covariance = artifact["initial_covariance"]
        self.latent_dimension = int(self.components.shape[1])
        self.num_buses = int(self.state_mean.size // 2)
        if self.state_mean.size != self.num_buses * 2:
            raise ValueError("Estimator state vector is not V/angle paired")

    def _prediction(
        self,
        latent: np.ndarray,
        converged: bool = True,
        failure_reason: str | None = None,
        iteration_count: int = 1,
    ) -> EstimatorPrediction:
        state = self.state_mean + self.components @ latent
        return EstimatorPrediction(
            voltage=state[: self.num_buses],
            angle=state[self.num_buses :],
            converged=bool(converged),
            failure_reason=failure_reason,
            iteration_count=int(iteration_count),
        )

    def training_mean(self) -> EstimatorPrediction:
        return self._prediction(
            np.zeros(self.latent_dimension), iteration_count=0
        )

    def _observed(
        self, measurement: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        observed = np.flatnonzero(mask > 0.5)
        if measurement.shape != self.measurement_mean.shape or mask.shape != measurement.shape:
            raise ValueError("Estimator measurement vector shape mismatch")
        return (
            measurement[observed] - self.measurement_mean[observed],
            self.measurement_matrix[observed],
            self.measurement_variance[observed],
        )

    def _weighted_solution(
        self,
        centered_measurement: np.ndarray,
        matrix: np.ndarray,
        variance: np.ndarray,
        robust_weight: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_covariance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if centered_measurement.size == 0:
            if prior_mean is None or prior_covariance is None:
                return np.zeros(self.latent_dimension), np.eye(self.latent_dimension)
            return prior_mean.copy(), prior_covariance.copy()
        weight = 1.0 / variance
        if robust_weight is not None:
            weight = weight * robust_weight
        information = matrix.T @ (matrix * weight[:, None])
        rhs = matrix.T @ (centered_measurement * weight)
        if prior_mean is None:
            information += float(self.hyper["snapshot_information_ridge"]) * np.eye(
                self.latent_dimension
            )
        else:
            if prior_covariance is None:
                raise ValueError("Prior covariance is required with a prior mean")
            prior_information = np.linalg.solve(
                prior_covariance, np.eye(self.latent_dimension)
            )
            information += prior_information
            rhs += prior_information @ prior_mean
        latent = np.linalg.solve(information, rhs)
        covariance = np.linalg.solve(
            information, np.eye(self.latent_dimension)
        )
        return latent, covariance

    def linear_wls(
        self, measurement: np.ndarray, mask: np.ndarray
    ) -> EstimatorPrediction:
        centered, matrix, variance = self._observed(measurement, mask)
        try:
            latent, _ = self._weighted_solution(centered, matrix, variance)
            return self._prediction(latent)
        except np.linalg.LinAlgError as exc:
            return self._prediction(
                np.zeros(self.latent_dimension),
                converged=False,
                failure_reason=f"linear_algebra:{type(exc).__name__}",
            )

    def huber_irls(
        self, measurement: np.ndarray, mask: np.ndarray
    ) -> EstimatorPrediction:
        centered, matrix, variance = self._observed(measurement, mask)
        if centered.size == 0:
            return self.training_mean()
        delta = float(self.hyper["huber_delta_standardized"])
        maximum = int(self.hyper["irls_max_iterations"])
        tolerance = float(self.hyper["irls_scaled_step_tolerance"])
        try:
            latent, _ = self._weighted_solution(centered, matrix, variance)
            for iteration in range(1, maximum + 1):
                standardized = (centered - matrix @ latent) / np.sqrt(variance)
                absolute = np.abs(standardized)
                robust_weight = np.ones_like(absolute)
                large = absolute > delta
                robust_weight[large] = delta / absolute[large]
                updated, _ = self._weighted_solution(
                    centered, matrix, variance, robust_weight=robust_weight
                )
                scaled_step = np.linalg.norm(updated - latent) / max(
                    np.linalg.norm(latent), 1.0
                )
                latent = updated
                if scaled_step <= tolerance:
                    return self._prediction(latent, iteration_count=iteration)
            return self._prediction(
                latent,
                converged=False,
                failure_reason="maximum_irls_iterations",
                iteration_count=maximum,
            )
        except np.linalg.LinAlgError as exc:
            return self._prediction(
                np.zeros(self.latent_dimension),
                converged=False,
                failure_reason=f"linear_algebra:{type(exc).__name__}",
            )

    def _kalman(
        self,
        measurements: np.ndarray,
        masks: np.ndarray,
        robust: bool,
    ) -> EstimatorPrediction:
        measurements = np.asarray(measurements, dtype=np.float64)
        masks = np.asarray(masks, dtype=np.float64)
        if measurements.ndim != 2 or masks.shape != measurements.shape:
            raise ValueError("Kalman estimator requires a measurement sequence")
        latent = np.zeros(self.latent_dimension)
        covariance = self.initial_covariance.copy()
        total_iterations = 0
        all_converged = True
        failure_reason = None
        try:
            for time_index in range(measurements.shape[0]):
                if time_index > 0:
                    latent = self.transition @ latent
                    covariance = (
                        self.transition @ covariance @ self.transition.T
                        + self.transition_covariance
                    )
                centered, matrix, variance = self._observed(
                    measurements[time_index], masks[time_index]
                )
                if centered.size == 0:
                    continue
                if not robust:
                    latent, covariance = self._weighted_solution(
                        centered,
                        matrix,
                        variance,
                        prior_mean=latent,
                        prior_covariance=covariance,
                    )
                    total_iterations += 1
                    continue
                prior_mean = latent.copy()
                prior_covariance = covariance.copy()
                updated, updated_covariance = self._weighted_solution(
                    centered,
                    matrix,
                    variance,
                    prior_mean=prior_mean,
                    prior_covariance=prior_covariance,
                )
                delta = float(self.hyper["huber_delta_standardized"])
                maximum = int(self.hyper["irls_max_iterations"])
                tolerance = float(self.hyper["irls_scaled_step_tolerance"])
                converged = False
                for iteration in range(1, maximum + 1):
                    standardized = (centered - matrix @ updated) / np.sqrt(variance)
                    absolute = np.abs(standardized)
                    robust_weight = np.ones_like(absolute)
                    large = absolute > delta
                    robust_weight[large] = delta / absolute[large]
                    candidate, candidate_covariance = self._weighted_solution(
                        centered,
                        matrix,
                        variance,
                        robust_weight=robust_weight,
                        prior_mean=prior_mean,
                        prior_covariance=prior_covariance,
                    )
                    scaled_step = np.linalg.norm(candidate - updated) / max(
                        np.linalg.norm(updated), 1.0
                    )
                    updated = candidate
                    updated_covariance = candidate_covariance
                    total_iterations += 1
                    if scaled_step <= tolerance:
                        converged = True
                        break
                if not converged:
                    all_converged = False
                    failure_reason = "maximum_irls_iterations"
                latent = updated
                covariance = updated_covariance
            return self._prediction(
                latent,
                converged=all_converged,
                failure_reason=failure_reason,
                iteration_count=total_iterations,
            )
        except np.linalg.LinAlgError as exc:
            return self._prediction(
                latent,
                converged=False,
                failure_reason=f"linear_algebra:{type(exc).__name__}",
                iteration_count=total_iterations,
            )

    def forecast_aided_kalman(
        self, measurements: np.ndarray, masks: np.ndarray
    ) -> EstimatorPrediction:
        return self._kalman(measurements, masks, robust=False)

    def forecast_aided_huber(
        self, measurements: np.ndarray, masks: np.ndarray
    ) -> EstimatorPrediction:
        return self._kalman(measurements, masks, robust=True)

    def predict(
        self,
        variant: str,
        measurements: np.ndarray,
        masks: np.ndarray,
    ) -> EstimatorPrediction:
        if variant == "per_bus_training_mean":
            return self.training_mean()
        if variant == "data_calibrated_linear_wls":
            return self.linear_wls(measurements[-1], masks[-1])
        if variant == "huber_irls_m_estimator":
            return self.huber_irls(measurements[-1], masks[-1])
        if variant == "forecast_aided_kalman":
            return self.forecast_aided_kalman(measurements, masks)
        if variant == "forecast_aided_huber":
            return self.forecast_aided_huber(measurements, masks)
        raise ValueError(f"Unknown estimator variant: {variant}")
