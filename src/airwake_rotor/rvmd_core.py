# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
# RVMD method/port attribution: Zimo Liao et al.; see THIRD_PARTY_NOTICES.md.
"""Reduced-order variational mode decomposition (RVMD) for real data.

This module is a NumPy/SciPy port of the real-valued block-coordinate
algorithm described by Liao et al., J. Fluid Mech. 966, A7 (2023), and
aligned with the authors' public ``rvmd.m`` implementation.  Its input is a
real state-by-time matrix.  The reconstruction convention is::

    q_approx = result.phi @ result.c.T

The implementation deliberately keeps the normalized-frequency convention
of the reference algorithm: the bandwidth penalty uses cycles per sample.
``dt`` is nevertheless mandatory so that every result also carries physical
frequency in hertz and Strouhal number.

Reference
---------
Z. Liao et al. (2023), "Reduced-order variational mode decomposition to
reveal transient and non-stationary dynamics in fluid flows", Journal of
Fluid Mechanics 966, A7. https://doi.org/10.1017/jfm.2023.435
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.fft import fft, ifft


ArrayLike = Union[np.ndarray, Sequence[float]]


RESTART_SCHEMA_VERSION = 1


def _array_sha256(array: np.ndarray, block_rows: int = 4096) -> str:
    """Hash dtype, shape, and values without constructing one large byte copy."""

    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(b"rvmd-array-v1\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value).tobytes())
    else:
        for start in range(0, int(value.shape[0]), block_rows):
            digest.update(np.ascontiguousarray(value[start : start + block_rows]).tobytes())
    return digest.hexdigest()


@dataclass
class RVMDRestartState:
    """Opaque unsorted block-coordinate state required for exact continuation.

    Public modes in :class:`RVMDResult` are sorted for interpretation.  These
    arrays intentionally retain the internal update order, because sorting
    them and then resuming would change the Gauss--Seidel sweep trajectory.
    """

    phi_unsorted: np.ndarray
    coefficient_spectrum_unsorted: np.ndarray
    residual_spectrum: np.ndarray
    weights: np.ndarray
    omega_history_unsorted: np.ndarray
    difference_history: np.ndarray
    data_spectrum_norm: float
    completed_steps: int
    n_state: int
    n_time: int
    k: int
    alpha: float
    dt: float
    dtype: str
    n_dc: int
    init_frequency_type: int
    init_frequency_max_hz_effective: float
    initial_frequencies_hz: np.ndarray
    input_sha256: str
    maximum_steps_at_save: int
    tolerance_at_save: float
    schema_version: int = RESTART_SCHEMA_VERSION
    array_sha256: Optional[Dict[str, str]] = None

    def current_array_sha256(self) -> Dict[str, str]:
        return {
            "phi_unsorted": _array_sha256(self.phi_unsorted),
            "coefficient_spectrum_unsorted": _array_sha256(
                self.coefficient_spectrum_unsorted
            ),
            "residual_spectrum": _array_sha256(self.residual_spectrum),
            "weights": _array_sha256(self.weights),
            "omega_history_unsorted": _array_sha256(self.omega_history_unsorted),
            "difference_history": _array_sha256(self.difference_history),
            "initial_frequencies_hz": _array_sha256(self.initial_frequencies_hz),
        }

    def metadata(self) -> Dict[str, Any]:
        """Return JSON-safe scalar restart metadata; arrays are stored separately."""

        current_hashes = self.current_array_sha256()
        if self.array_sha256 is not None and self.array_sha256 != current_hashes:
            raise ValueError("RVMD restart state arrays changed after the checkpoint was created")
        return {
            "schema_version": int(self.schema_version),
            "completed_steps": int(self.completed_steps),
            "n_state": int(self.n_state),
            "n_time": int(self.n_time),
            "k": int(self.k),
            "alpha": float(self.alpha),
            "dt": float(self.dt),
            "dtype": str(self.dtype),
            "n_dc": int(self.n_dc),
            "init_frequency_type": int(self.init_frequency_type),
            "init_frequency_max_hz_effective": float(
                self.init_frequency_max_hz_effective
            ),
            "input_sha256": str(self.input_sha256),
            "data_spectrum_norm": float(self.data_spectrum_norm),
            "maximum_steps_at_save": int(self.maximum_steps_at_save),
            "tolerance_at_save": float(self.tolerance_at_save),
            "state_order": "unsorted internal Gauss-Seidel update order",
            "array_sha256": current_hashes,
        }


def _as_jsonable(value: Any) -> Any:
    """Convert nested NumPy values to JSON-compatible Python values."""

    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class RVMDResult:
    """Container returned by :func:`rvmd`.

    ``phi`` has shape ``(n_state, k)`` and ``c`` has shape ``(n_time, k)``.
    Modes are sorted in ascending center frequency.  ``energy`` follows the
    reference implementation and is ``sum(abs(c)**2, axis=0)``.  It must not
    be summed and called total captured energy when modes are non-orthogonal;
    use the explicitly labelled terms in ``audit`` for that purpose.
    """

    phi: np.ndarray
    c: np.ndarray
    omega_cycles_per_sample: np.ndarray
    frequency_hz: np.ndarray
    strouhal: np.ndarray
    energy: np.ndarray
    coefficient_spectrum: np.ndarray
    omega_history: np.ndarray
    difference_history: np.ndarray
    info: Dict[str, Any]
    audit: Dict[str, float]
    restart_state: Optional[RVMDRestartState] = None

    def reconstruct(self, mode_indices: Optional[Iterable[int]] = None) -> np.ndarray:
        """Reconstruct all modes, or a zero-based subset of modes."""

        if mode_indices is None:
            return self.phi @ self.c.T
        indices = np.asarray(list(mode_indices), dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("mode_indices must be a one-dimensional iterable")
        if indices.size == 0:
            return np.zeros((self.phi.shape[0], self.c.shape[0]), dtype=self.phi.dtype)
        if np.any(indices < 0) or np.any(indices >= self.phi.shape[1]):
            raise IndexError("mode index is outside the decomposition")
        return self.phi[:, indices] @ self.c[:, indices].T

    def save(self, path: Union[str, Path], compressed: bool = False) -> Path:
        """Save the portable decomposition to an NPZ file without pickling."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            phi=self.phi,
            c=self.c,
            omega_cycles_per_sample=self.omega_cycles_per_sample,
            frequency_hz=self.frequency_hz,
            strouhal=self.strouhal,
            energy=self.energy,
            coefficient_spectrum=self.coefficient_spectrum,
            omega_history=self.omega_history,
            difference_history=self.difference_history,
            info_json=np.asarray(json.dumps(_as_jsonable(self.info), ensure_ascii=False)),
            audit_json=np.asarray(json.dumps(_as_jsonable(self.audit), ensure_ascii=False)),
        )
        if self.restart_state is not None:
            state = self.restart_state
            payload.update(
                restart_json=np.asarray(
                    json.dumps(_as_jsonable(state.metadata()), ensure_ascii=False)
                ),
                restart_phi_unsorted=state.phi_unsorted,
                restart_coefficient_spectrum_unsorted=state.coefficient_spectrum_unsorted,
                restart_residual_spectrum=state.residual_spectrum,
                restart_weights=state.weights,
                restart_omega_history_unsorted=state.omega_history_unsorted,
                restart_difference_history=state.difference_history,
                restart_initial_frequencies_hz=state.initial_frequencies_hz,
            )
        saver = np.savez_compressed if compressed else np.savez
        saver(str(output), **payload)
        return output

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RVMDResult":
        """Load an :class:`RVMDResult` produced by :meth:`save`."""

        with np.load(str(path), allow_pickle=False) as data:
            restart_state: Optional[RVMDRestartState] = None
            restart_names = {
                "restart_json",
                "restart_phi_unsorted",
                "restart_coefficient_spectrum_unsorted",
                "restart_residual_spectrum",
                "restart_weights",
                "restart_omega_history_unsorted",
                "restart_difference_history",
                "restart_initial_frequencies_hz",
            }
            present = restart_names.intersection(data.files)
            if present and present != restart_names:
                missing = sorted(restart_names - present)
                raise ValueError(
                    "incomplete RVMD restart payload; missing {}".format(", ".join(missing))
                )
            if present:
                try:
                    restart_metadata = json.loads(str(data["restart_json"].item()))
                except (ValueError, TypeError, json.JSONDecodeError) as exception:
                    raise ValueError("invalid restart_json in RVMD result") from exception
                if not isinstance(restart_metadata, dict):
                    raise ValueError("restart_json must encode an object")
                required_metadata = {
                    "schema_version",
                    "completed_steps",
                    "n_state",
                    "n_time",
                    "k",
                    "alpha",
                    "dt",
                    "dtype",
                    "n_dc",
                    "init_frequency_type",
                    "init_frequency_max_hz_effective",
                    "input_sha256",
                    "data_spectrum_norm",
                    "maximum_steps_at_save",
                    "tolerance_at_save",
                    "array_sha256",
                }
                missing_metadata = required_metadata - set(restart_metadata)
                if missing_metadata:
                    raise ValueError(
                        "restart_json is missing {}".format(", ".join(sorted(missing_metadata)))
                    )
                restart_arrays = {
                    "phi_unsorted": data["restart_phi_unsorted"],
                    "coefficient_spectrum_unsorted": data[
                        "restart_coefficient_spectrum_unsorted"
                    ],
                    "residual_spectrum": data["restart_residual_spectrum"],
                    "weights": data["restart_weights"],
                    "omega_history_unsorted": data["restart_omega_history_unsorted"],
                    "difference_history": data["restart_difference_history"],
                    "initial_frequencies_hz": data["restart_initial_frequencies_hz"],
                }
                recorded_hashes = restart_metadata["array_sha256"]
                if not isinstance(recorded_hashes, dict):
                    raise ValueError("restart_json/array_sha256 must encode an object")
                for name, array in restart_arrays.items():
                    if recorded_hashes.get(name) != _array_sha256(array):
                        raise ValueError("RVMD restart array checksum mismatch for {}".format(name))
                restart_state = RVMDRestartState(
                    phi_unsorted=restart_arrays["phi_unsorted"],
                    coefficient_spectrum_unsorted=restart_arrays[
                        "coefficient_spectrum_unsorted"
                    ],
                    residual_spectrum=restart_arrays["residual_spectrum"],
                    weights=restart_arrays["weights"],
                    omega_history_unsorted=restart_arrays["omega_history_unsorted"],
                    difference_history=restart_arrays["difference_history"],
                    initial_frequencies_hz=restart_arrays["initial_frequencies_hz"],
                    data_spectrum_norm=float(restart_metadata["data_spectrum_norm"]),
                    completed_steps=int(restart_metadata["completed_steps"]),
                    n_state=int(restart_metadata["n_state"]),
                    n_time=int(restart_metadata["n_time"]),
                    k=int(restart_metadata["k"]),
                    alpha=float(restart_metadata["alpha"]),
                    dt=float(restart_metadata["dt"]),
                    dtype=str(restart_metadata["dtype"]),
                    n_dc=int(restart_metadata["n_dc"]),
                    init_frequency_type=int(restart_metadata["init_frequency_type"]),
                    init_frequency_max_hz_effective=float(
                        restart_metadata["init_frequency_max_hz_effective"]
                    ),
                    input_sha256=str(restart_metadata["input_sha256"]),
                    maximum_steps_at_save=int(restart_metadata["maximum_steps_at_save"]),
                    tolerance_at_save=float(restart_metadata["tolerance_at_save"]),
                    schema_version=int(restart_metadata["schema_version"]),
                    array_sha256={str(key): str(value) for key, value in recorded_hashes.items()},
                )
            return cls(
                phi=data["phi"],
                c=data["c"],
                omega_cycles_per_sample=data["omega_cycles_per_sample"],
                frequency_hz=data["frequency_hz"],
                strouhal=data["strouhal"],
                energy=data["energy"],
                coefficient_spectrum=data["coefficient_spectrum"],
                omega_history=data["omega_history"],
                difference_history=data["difference_history"],
                info=json.loads(str(data["info_json"].item())),
                audit=json.loads(str(data["audit_json"].item())),
                restart_state=restart_state,
            )


def mirror_extend(q: np.ndarray) -> Tuple[np.ndarray, int]:
    """Mirror a state-by-time matrix to exactly twice its time length.

    The construction matches the authors' implementation.  If ``T`` is odd,
    the left mirror contains ``ceil(T/2)`` samples and the right mirror the
    remaining ``floor(T/2)`` samples.  The returned integer is the starting
    index used to crop the original interval after inverse transformation.
    """

    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError("q must be a two-dimensional matrix with at least two time samples")
    n_time = q.shape[1]
    half = int(math.ceil(n_time / 2.0))
    extended = np.concatenate(
        (q[:, :half][:, ::-1], q, q[:, half:][:, ::-1]), axis=1
    )
    if extended.shape[1] != 2 * n_time:
        raise RuntimeError("internal mirror-extension length error")
    return extended, half


def _normalize_weights(weights: Optional[ArrayLike], n_state: int, dtype: np.dtype) -> np.ndarray:
    if weights is None:
        result = np.ones(n_state, dtype=dtype)
    else:
        raw = np.asarray(weights, dtype=dtype)
        if raw.ndim == 0:
            result = np.full(n_state, raw.item(), dtype=dtype)
        elif raw.ndim == 1 and raw.size == n_state:
            result = raw.copy()
        else:
            raise ValueError("spatial_weights must be a positive scalar or a vector of n_state")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("spatial_weights must contain finite positive values")
    result /= np.mean(result, dtype=np.float64)
    return result


def _require_finite_array(value: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError("restart {} contains NaN or infinite values".format(label))


def _validated_restart_state(
    restart: RVMDResult,
    q_work: np.ndarray,
    k: int,
    alpha: float,
    dt: float,
    real_dtype: np.dtype,
    complex_dtype: np.dtype,
    n_dc: int,
    weights: np.ndarray,
    maximum_steps: int,
) -> RVMDRestartState:
    """Validate immutable settings and return a safe exact continuation state."""

    if not isinstance(restart, RVMDResult):
        raise TypeError("restart must be an RVMDResult")
    state = restart.restart_state
    if state is None:
        raise ValueError(
            "the saved RVMDResult has no internal restart state; legacy NPZ files remain "
            "readable but cannot be continued"
        )
    if state.schema_version != RESTART_SCHEMA_VERSION:
        raise ValueError(
            "unsupported RVMD restart schema version {}; expected {}".format(
                state.schema_version, RESTART_SCHEMA_VERSION
            )
        )
    if state.array_sha256 is None or state.array_sha256 != state.current_array_sha256():
        raise ValueError("restart state checksum validation failed")
    n_state, n_time = q_work.shape
    immutable_checks = (
        (state.n_state, n_state, "state dimension"),
        (state.n_time, n_time, "time dimension"),
        (state.k, int(k), "k"),
        (state.n_dc, int(n_dc), "n_dc"),
    )
    for saved, requested, label in immutable_checks:
        if saved != requested:
            raise ValueError(
                "restart {} mismatch: saved {}, requested {}".format(label, saved, requested)
            )
    if float(state.alpha) != float(alpha):
        raise ValueError(
            "restart alpha mismatch: saved {!r}, requested {!r}".format(state.alpha, alpha)
        )
    if float(state.dt) != float(dt):
        raise ValueError("restart dt mismatch: saved {!r}, requested {!r}".format(state.dt, dt))
    if state.dtype != real_dtype.name:
        raise ValueError(
            "restart dtype mismatch: saved {}, requested {}".format(state.dtype, real_dtype.name)
        )
    if maximum_steps < state.completed_steps:
        raise ValueError(
            "maximum_steps is a total sweep limit and cannot be less than the {} already "
            "completed sweeps".format(state.completed_steps)
        )
    if state.completed_steps < 0:
        raise ValueError("restart completed_steps must be non-negative")
    if state.init_frequency_type not in (-1, 0, 1):
        raise ValueError("restart init_frequency_type must be -1, 0, or 1")
    expected_shapes = (
        (state.phi_unsorted, (n_state, k), "phi_unsorted"),
        (
            state.coefficient_spectrum_unsorted,
            (n_time + 1, k),
            "coefficient_spectrum_unsorted",
        ),
        (state.residual_spectrum, (n_state, n_time + 1), "residual_spectrum"),
        (state.weights, (n_state,), "weights"),
        (
            state.omega_history_unsorted,
            (k, state.completed_steps + 1),
            "omega_history_unsorted",
        ),
        (
            state.difference_history,
            (state.completed_steps,),
            "difference_history",
        ),
        (state.initial_frequencies_hz, (k,), "initial_frequencies_hz"),
    )
    for array, expected, label in expected_shapes:
        if np.asarray(array).shape != expected:
            raise ValueError(
                "restart {} shape mismatch: expected {}, got {}".format(
                    label, expected, np.asarray(array).shape
                )
            )
        _require_finite_array(np.asarray(array), label)
    if np.asarray(state.phi_unsorted).dtype != real_dtype:
        raise ValueError("restart phi dtype does not match the saved precision")
    if np.asarray(state.coefficient_spectrum_unsorted).dtype != complex_dtype:
        raise ValueError("restart coefficient-spectrum dtype does not match the saved precision")
    if np.asarray(state.residual_spectrum).dtype != complex_dtype:
        raise ValueError("restart residual dtype does not match the saved precision")
    for array, label in (
        (state.weights, "weights"),
        (state.omega_history_unsorted, "omega history"),
        (state.difference_history, "difference history"),
    ):
        if np.asarray(array).dtype != real_dtype:
            raise ValueError("restart {} dtype does not match the saved precision".format(label))
    if not np.array_equal(np.asarray(state.weights), weights):
        raise ValueError("restart spatial weights differ from the requested normalized weights")
    if not np.isfinite(state.data_spectrum_norm) or state.data_spectrum_norm < 0.0:
        raise ValueError("restart data_spectrum_norm must be finite and non-negative")
    if state.n_dc and not np.all(
        np.asarray(state.omega_history_unsorted)[: state.n_dc, :] == 0
    ):
        raise ValueError("restart n_dc frequency histories are not identically zero")
    current_input_hash = _array_sha256(q_work)
    if state.input_sha256 != current_input_hash:
        raise ValueError("restart input data differ from the saved state")

    if restart.omega_history.shape != state.omega_history_unsorted.shape or not np.array_equal(
        restart.omega_history, state.omega_history_unsorted
    ):
        raise ValueError("public and restart omega histories disagree")
    if restart.difference_history.shape != state.difference_history.shape or not np.array_equal(
        restart.difference_history, state.difference_history
    ):
        raise ValueError("public and restart difference histories disagree")
    final_internal_omega = np.asarray(state.omega_history_unsorted)[:, -1]
    order = np.argsort(final_internal_omega, kind="mergesort")
    if not np.array_equal(restart.omega_cycles_per_sample, final_internal_omega[order]):
        raise ValueError("public sorted frequencies disagree with the unsorted restart state")
    if not np.array_equal(restart.phi, np.asarray(state.phi_unsorted)[:, order]):
        raise ValueError("public sorted phi disagrees with the unsorted restart state")
    if not np.array_equal(
        restart.coefficient_spectrum,
        np.asarray(state.coefficient_spectrum_unsorted)[:, order],
    ):
        raise ValueError("public sorted spectrum disagrees with the unsorted restart state")
    info_steps = restart.info.get("completed_steps")
    if info_steps is None or int(info_steps) != state.completed_steps:
        raise ValueError("result info completed_steps disagrees with restart state")
    return state


def _initial_frequencies(
    k: int,
    n_dc: int,
    init_type: int,
    maximum_cycles_per_sample: float,
    initial_hz: Optional[ArrayLike],
    dt: float,
    seed: int,
    dtype: np.dtype,
) -> np.ndarray:
    if initial_hz is not None:
        frequencies = np.asarray(initial_hz, dtype=np.float64).reshape(-1)
        if frequencies.size != k or not np.all(np.isfinite(frequencies)):
            raise ValueError("initial_frequencies_hz must contain exactly k finite values")
        nyquist = 0.5 / dt
        if np.any(frequencies < 0) or np.any(frequencies > nyquist):
            raise ValueError("initial_frequencies_hz must lie between zero and Nyquist")
        omega = (frequencies * dt).astype(dtype)
        if n_dc and np.any(omega[:n_dc] != 0):
            raise ValueError("the first n_dc initial frequencies must be zero")
        return omega

    if init_type not in (-1, 0, 1):
        raise ValueError("init_frequency_type must be -1, 0, or 1")
    omega = np.zeros(k, dtype=dtype)
    free = k - n_dc
    if free == 0:
        return omega
    if init_type == -1:
        rng = np.random.RandomState(seed)
        omega[n_dc:] = rng.rand(free).astype(dtype) * maximum_cycles_per_sample
    elif init_type == 1:
        omega[n_dc:] = (
            np.arange(1, free + 1, dtype=dtype) / dtype.type(free)
        ) * maximum_cycles_per_sample
    return omega


def energy_audit(q: np.ndarray, phi: np.ndarray, c: np.ndarray, weights: Optional[ArrayLike] = None) -> Dict[str, float]:
    """Return a labelled weighted reconstruction and energy-closure audit."""

    q_array = np.asarray(q)
    if q_array.ndim != 2:
        raise ValueError("q must be two-dimensional")
    if phi.shape[0] != q_array.shape[0] or c.shape[0] != q_array.shape[1] or phi.shape[1] != c.shape[1]:
        raise ValueError("phi and c dimensions are inconsistent with q")
    weight = _normalize_weights(weights, q_array.shape[0], np.dtype(np.float64))
    reconstruction = np.asarray(phi, dtype=np.float64) @ np.asarray(c, dtype=np.float64).T
    q64 = np.asarray(q_array, dtype=np.float64)
    residual = q64 - reconstruction
    weighted = weight[:, None]
    data_energy = float(np.sum(weighted * q64 * q64, dtype=np.float64))
    reconstruction_energy = float(np.sum(weighted * reconstruction * reconstruction, dtype=np.float64))
    residual_energy = float(np.sum(weighted * residual * residual, dtype=np.float64))
    cross_term = float(2.0 * np.sum(weighted * reconstruction * residual, dtype=np.float64))
    closure_error = data_energy - reconstruction_energy - residual_energy - cross_term
    individual = np.empty(phi.shape[1], dtype=np.float64)
    for mode_index in range(phi.shape[1]):
        spatial_norm_sq = float(np.sum(weight * np.asarray(phi[:, mode_index], dtype=np.float64) ** 2))
        temporal_norm_sq = float(np.sum(np.asarray(c[:, mode_index], dtype=np.float64) ** 2))
        individual[mode_index] = spatial_norm_sq * temporal_norm_sq
    individual_sum = float(np.sum(individual))
    scale = max(data_energy, np.finfo(np.float64).tiny)
    gram = np.asarray(phi, dtype=np.float64).T @ (weight[:, None] * np.asarray(phi, dtype=np.float64))
    off_diagonal = gram - np.diag(np.diag(gram))
    return {
        "weighted_data_energy": data_energy,
        "weighted_reconstruction_energy": reconstruction_energy,
        "weighted_residual_energy": residual_energy,
        "weighted_reconstruction_residual_cross_term": cross_term,
        "energy_identity_closure_error": float(closure_error),
        "energy_identity_closure_error_relative": float(abs(closure_error) / scale),
        "residual_energy_fraction": float(residual_energy / scale),
        "captured_energy_fraction_one_minus_residual": float(1.0 - residual_energy / scale),
        "relative_frobenius_error": float(math.sqrt(residual_energy / scale)),
        "sum_individual_modal_self_energy": individual_sum,
        "modal_cross_energy_reconstruction_minus_self_sum": float(reconstruction_energy - individual_sum),
        "max_abs_spatial_gram_off_diagonal": float(np.max(np.abs(off_diagonal))) if off_diagonal.size else 0.0,
    }


def coefficient_spectrum_moments(
    coefficient_spectrum: np.ndarray,
    n_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return RVMD center frequency and RMS bandwidth in cycles per sample.

    RVMD optimizes the one-sided spectrum of a ``2*n_time`` mirror extension.
    The DC and Nyquist bins therefore carry half weight, matching the update
    equation used inside :func:`rvmd`.  This helper is intentionally public so
    saved modal frequencies can be audited independently from the optimizer.
    """

    spectrum = np.asarray(coefficient_spectrum)
    if not isinstance(n_time, (int, np.integer)) or int(n_time) < 2:
        raise ValueError("n_time must be an integer of at least two")
    expected_rows = int(n_time) + 1
    if spectrum.ndim != 2 or spectrum.shape[0] != expected_rows:
        raise ValueError(
            "coefficient_spectrum must have shape (n_time + 1, k); got {}".format(
                spectrum.shape
            )
        )
    if spectrum.shape[1] < 1 or not np.all(np.isfinite(spectrum)):
        raise ValueError("coefficient_spectrum must contain finite values and at least one mode")
    frequency = np.arange(expected_rows, dtype=np.float64) / float(2 * int(n_time))
    weights = np.ones(expected_rows, dtype=np.float64)
    weights[[0, -1]] = 0.5
    power = weights[:, None] * np.abs(spectrum.astype(np.complex128, copy=False)) ** 2
    denominator = np.sum(power, axis=0, dtype=np.float64)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("every coefficient spectrum must have positive finite energy")
    center = np.sum(power * frequency[:, None], axis=0, dtype=np.float64) / denominator
    variance = (
        np.sum(power * (frequency[:, None] - center[None, :]) ** 2, axis=0, dtype=np.float64)
        / denominator
    )
    return center, np.sqrt(np.maximum(variance, 0.0))


def _iterate_torch_cuda(
    *,
    phi: np.ndarray,
    coefficient_spectrum: np.ndarray,
    residual_spectrum: np.ndarray,
    omega: np.ndarray,
    frequency: np.ndarray,
    frequency_weight: np.ndarray,
    weights: np.ndarray,
    omega_history: np.ndarray,
    difference_history: np.ndarray,
    completed_steps: int,
    restart_from_completed_steps: int,
    maximum_steps: int,
    previous_difference: float,
    tolerance: float,
    alpha: float,
    n_dc: int,
    scale_floor: float,
    dt: float,
    verbose: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    float,
    bool,
    Dict[str, Any],
]:
    """Run RVMD Gauss-Seidel sweeps on CUDA without changing checkpoints."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "compute_backend='torch-cuda' requires a PyTorch installation with CUDA"
        ) from exc
    if not torch.cuda.is_available():  # pragma: no cover - depends on optional hardware
        raise RuntimeError(
            "compute_backend='torch-cuda' was requested but torch.cuda.is_available() is false"
        )

    device_index = 0
    device = torch.device("cuda:0")
    real_torch_dtype = torch.float32 if phi.dtype == np.float32 else torch.float64
    complex_torch_dtype = torch.complex64 if phi.dtype == np.float32 else torch.complex128
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.no_grad():
            phi_gpu = torch.as_tensor(phi, dtype=real_torch_dtype, device=device).clone()
            coefficient_gpu = torch.as_tensor(
                coefficient_spectrum, dtype=complex_torch_dtype, device=device
            ).clone()
            residual_gpu = torch.as_tensor(
                residual_spectrum, dtype=complex_torch_dtype, device=device
            ).clone()
            omega_gpu = torch.as_tensor(omega, dtype=real_torch_dtype, device=device).clone()
            frequency_gpu = torch.as_tensor(
                frequency, dtype=real_torch_dtype, device=device
            )
            frequency_weight_gpu = torch.as_tensor(
                frequency_weight, dtype=real_torch_dtype, device=device
            )
            weights_gpu = torch.as_tensor(weights, dtype=real_torch_dtype, device=device)
            omega_history_gpu = torch.empty(
                omega_history.shape, dtype=real_torch_dtype, device=device
            )
            difference_history_gpu = torch.empty(
                difference_history.shape, dtype=real_torch_dtype, device=device
            )
            omega_history_gpu[:, : completed_steps + 1] = torch.as_tensor(
                omega_history[:, : completed_steps + 1],
                dtype=real_torch_dtype,
                device=device,
            )
            if completed_steps:
                difference_history_gpu[:completed_steps] = torch.as_tensor(
                    difference_history[:completed_steps],
                    dtype=real_torch_dtype,
                    device=device,
                )
            scale_floor_gpu = torch.tensor(
                scale_floor, dtype=real_torch_dtype, device=device
            )
            real_tiny_gpu = torch.tensor(
                np.finfo(phi.dtype).tiny, dtype=real_torch_dtype, device=device
            )
            k = int(phi.shape[1])
            n_state = int(phi.shape[0])
            converged = previous_difference <= tolerance

            for step in range(completed_steps, maximum_steps):
                if previous_difference <= tolerance:
                    break
                difference_gpu = torch.zeros((), dtype=torch.float64, device=device)
                for mode_index in range(k):
                    old_mode = (
                        phi_gpu[:, mode_index, None]
                        * coefficient_gpu[:, mode_index][None, :]
                    )
                    residual_gpu.add_(old_mode)

                    projection = residual_gpu @ (
                        torch.conj(coefficient_gpu[:, mode_index])
                        * frequency_weight_gpu
                    )
                    projection = torch.real(projection)
                    projection_norm = torch.sqrt(
                        torch.sum(
                            torch.abs(projection) ** 2 * weights_gpu,
                            dtype=torch.float64,
                        )
                    )
                    projection_valid = torch.isfinite(projection_norm) & (
                        projection_norm > 0
                    )
                    normalized_projection = projection / torch.clamp(
                        projection_norm.to(real_torch_dtype), min=real_tiny_gpu
                    )
                    fallback = phi_gpu[:, mode_index]
                    fallback_norm = torch.sqrt(
                        torch.sum(
                            torch.abs(fallback) ** 2 * weights_gpu,
                            dtype=torch.float64,
                        )
                    )
                    fallback_valid = torch.isfinite(fallback_norm) & (fallback_norm > 0)
                    normalized_fallback = fallback / torch.clamp(
                        fallback_norm.to(real_torch_dtype), min=real_tiny_gpu
                    )
                    default_fallback = torch.zeros(
                        n_state, dtype=real_torch_dtype, device=device
                    )
                    row = mode_index % n_state
                    default_fallback[row] = torch.rsqrt(weights_gpu[row])
                    safe_fallback = torch.where(
                        fallback_valid, normalized_fallback, default_fallback
                    )
                    updated_phi = torch.where(
                        projection_valid, normalized_projection, safe_fallback
                    )
                    pivot = torch.argmax(torch.abs(updated_phi))
                    orientation = torch.where(
                        updated_phi[pivot] < 0,
                        torch.tensor(-1.0, dtype=real_torch_dtype, device=device),
                        torch.tensor(1.0, dtype=real_torch_dtype, device=device),
                    )
                    phi_gpu[:, mode_index] = updated_phi * orientation

                    denominator = 1.0 + 2.0 * float(alpha) * (
                        frequency_gpu - omega_gpu[mode_index]
                    ) ** 2
                    coefficient_gpu[:, mode_index] = (
                        residual_gpu.T
                        @ (phi_gpu[:, mode_index] * weights_gpu).to(complex_torch_dtype)
                    ) / denominator
                    coefficient_power = torch.abs(
                        coefficient_gpu[:, mode_index]
                    ) ** 2
                    coefficient_energy = torch.sum(
                        frequency_weight_gpu * coefficient_power,
                        dtype=torch.float64,
                    )
                    if mode_index < n_dc:
                        new_omega = torch.zeros(
                            (), dtype=real_torch_dtype, device=device
                        )
                    else:
                        omega_numerator = torch.sum(
                            frequency_weight_gpu * frequency_gpu * coefficient_power,
                            dtype=torch.float64,
                        )
                        energy_valid = torch.isfinite(coefficient_energy) & (
                            coefficient_energy > 0
                        )
                        omega_candidate = (
                            omega_numerator / torch.clamp(coefficient_energy, min=1e-300)
                        ).to(real_torch_dtype)
                        new_omega = torch.where(
                            energy_valid, omega_candidate, omega_gpu[mode_index]
                        )

                    new_mode = (
                        phi_gpu[:, mode_index, None]
                        * coefficient_gpu[:, mode_index][None, :]
                    )
                    residual_gpu.sub_(new_mode)
                    old_norm = torch.linalg.vector_norm(old_mode)
                    change_norm = torch.linalg.vector_norm(new_mode - old_mode)
                    relative_change = change_norm / torch.maximum(
                        old_norm, scale_floor_gpu
                    )
                    difference_gpu += relative_change.to(torch.float64)
                    omega_gpu[mode_index] = new_omega

                completed_steps = step + 1
                difference_history_gpu[step] = difference_gpu.to(real_torch_dtype)
                omega_history_gpu[:, step + 1] = omega_gpu
                previous_difference = float(difference_gpu.item())
                if verbose and (
                    step == restart_from_completed_steps
                    or completed_steps % 20 == 0
                    or previous_difference <= tolerance
                ):
                    frequencies_hz = (
                        torch.sort(omega_gpu).values.detach().cpu().numpy().astype(np.float64)
                        / float(dt)
                    )
                    print(
                        "RVMD step {:d}: difference={:.6e}, f_Hz=[{:.5g}, {:.5g}], "
                        "unique={:d}/{:d}, backend=torch-cuda".format(
                            completed_steps,
                            previous_difference,
                            float(frequencies_hz[0]),
                            float(frequencies_hz[-1]),
                            int(np.unique(frequencies_hz).size),
                            int(frequencies_hz.size),
                        ),
                        flush=True,
                    )
                if previous_difference <= tolerance:
                    converged = True
                    break

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_memory = int(torch.cuda.max_memory_allocated())
            phi_out = phi_gpu.detach().cpu().numpy().copy()
            coefficient_out = coefficient_gpu.detach().cpu().numpy().copy()
            residual_out = residual_gpu.detach().cpu().numpy().copy()
            omega_out = omega_gpu.detach().cpu().numpy().copy()
            omega_history_out = omega_history_gpu.detach().cpu().numpy().copy()
            difference_history_out = difference_history_gpu.detach().cpu().numpy().copy()
            device_properties = torch.cuda.get_device_properties(device_index)
            backend_audit = {
                "compute_backend_requested": "torch-cuda",
                "compute_backend_effective": "torch-cuda",
                "compute_library": "pytorch",
                "compute_library_version": str(torch.__version__),
                "cuda_runtime_version": str(torch.version.cuda),
                "compute_device": str(device_properties.name),
                "compute_capability": [
                    int(device_properties.major),
                    int(device_properties.minor),
                ],
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "iteration_elapsed_seconds_this_call": float(elapsed),
                "peak_gpu_memory_bytes": peak_memory,
            }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.use_deterministic_algorithms(previous_deterministic)

    return (
        phi_out,
        coefficient_out,
        residual_out,
        omega_out,
        omega_history_out,
        difference_history_out,
        completed_steps,
        previous_difference,
        converged,
        backend_audit,
    )


def rvmd(
    q: np.ndarray,
    k: int,
    alpha: float,
    *,
    dt: float,
    spatial_weights: Optional[ArrayLike] = None,
    tolerance: float = 5e-3,
    maximum_steps: int = 500,
    init_frequency_type: Optional[int] = None,
    init_frequency_max_hz: Optional[float] = None,
    initial_frequencies_hz: Optional[ArrayLike] = None,
    n_dc: int = 0,
    random_seed: int = 0,
    dtype: Union[str, np.dtype] = "float32",
    restart: Optional[RVMDResult] = None,
    strouhal_length: float = 13.716,
    strouhal_velocity: float = 20.0,
    verbose: bool = False,
    compute_backend: str = "numpy",
) -> RVMDResult:
    """Decompose a real state-by-time matrix using RVMD.

    Parameters
    ----------
    q:
        Finite real array with shape ``(n_state, n_time)``.
    k:
        Number of modes.  ``k=30`` is fully supported; runtime scales roughly
        linearly with ``k`` and with the number of input rows.
    alpha:
        Non-negative RVMD bandwidth penalty in normalized frequency
        (cycles/sample), exactly as in the reference implementation.
    dt:
        Required sample interval in seconds.  Frequencies are returned as
        ``omega / dt`` in hertz.
    spatial_weights:
        Positive scalar or state vector.  It is normalized to mean one.
    n_dc:
        Number of leading internal modes whose center frequency remains zero.
    restart:
        An earlier :class:`RVMDResult` carrying its unsorted internal state.
        ``maximum_steps`` is the *total* sweep limit, including sweeps already
        completed by that result.  Tolerance may change; dimensions, ``k``,
        ``alpha``, ``dt``, precision, ``n_dc``, weights, and input data may not.
    strouhal_length, strouhal_velocity:
        Defaults implement ``St = f * 13.716 / 20``.
    compute_backend:
        ``"numpy"`` retains the reference CPU implementation.  ``"torch-cuda"``
        moves only the block-coordinate RVMD sweeps to an NVIDIA GPU while
        preserving the NumPy checkpoint schema and CPU-side FFT/final audit.

    Notes
    -----
    For large CFD arrays, call this routine on the deterministic sketch or
    reduced matrix produced by ``nato_gd_rvmd_adapter.py``.  Calling it on a
    full 62 GB state matrix would defeat the adapter's bounded-memory design.
    """

    input_array = np.asarray(q)
    if input_array.ndim != 2 or input_array.size == 0 or input_array.shape[1] < 2:
        raise ValueError("q must be a nonempty state-by-time matrix with at least two samples")
    if np.iscomplexobj(input_array):
        raise ValueError("this implementation currently supports real q only")
    if not np.all(np.isfinite(input_array)):
        raise ValueError("q contains NaN or infinite values")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number of seconds")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not isinstance(maximum_steps, (int, np.integer)) or maximum_steps <= 0:
        raise ValueError("maximum_steps must be a positive integer")
    if not isinstance(n_dc, (int, np.integer)) or n_dc < 0 or n_dc > k:
        raise ValueError("n_dc must be an integer between zero and k")
    if not np.isfinite(strouhal_length) or strouhal_length <= 0:
        raise ValueError("strouhal_length must be finite and positive")
    if not np.isfinite(strouhal_velocity) or strouhal_velocity <= 0:
        raise ValueError("strouhal_velocity must be finite and positive")
    compute_backend = str(compute_backend).strip().lower()
    if compute_backend not in ("numpy", "torch-cuda"):
        raise ValueError("compute_backend must be numpy or torch-cuda")

    real_dtype = np.dtype(dtype)
    if real_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    complex_dtype = np.dtype(np.complex64 if real_dtype == np.dtype(np.float32) else np.complex128)
    q_work = np.asarray(input_array, dtype=real_dtype, order="C")
    n_state, n_time = q_work.shape
    weights = _normalize_weights(spatial_weights, n_state, real_dtype)

    requested_init_max_was_explicit = init_frequency_max_hz is not None
    if not requested_init_max_was_explicit:
        requested_init_max_cycles = 0.5
        requested_init_max_hz_effective = 0.5 / dt
    else:
        if not np.isfinite(init_frequency_max_hz) or init_frequency_max_hz < 0:
            raise ValueError("init_frequency_max_hz must be finite and non-negative")
        requested_init_max_cycles = min(float(init_frequency_max_hz) * dt, 0.5)
        requested_init_max_hz_effective = requested_init_max_cycles / dt

    extended_length = 2 * n_time
    half = int(math.ceil(n_time / 2.0))
    spectrum_length = n_time + 1
    frequency = (np.arange(spectrum_length, dtype=real_dtype) / real_dtype.type(extended_length)).astype(real_dtype)
    frequency_weight = np.ones(spectrum_length, dtype=real_dtype)
    frequency_weight[[0, -1]] = real_dtype.type(0.5)
    eps = np.finfo(real_dtype).eps

    restart_from_completed_steps = 0
    if restart is None:
        effective_init_frequency_type = 1 if init_frequency_type is None else init_frequency_type
        omega = _initial_frequencies(
            k,
            n_dc,
            effective_init_frequency_type,
            requested_init_max_cycles,
            initial_frequencies_hz,
            dt,
            random_seed,
            real_dtype,
        )
        initial_frequencies_effective_hz = omega.astype(np.float64) / float(dt)
        init_max_hz_effective = requested_init_max_hz_effective
        initial_random_seed_effective = int(random_seed)

        extended, observed_half = mirror_extend(q_work)
        if observed_half != half:
            raise RuntimeError("internal mirror-extension index mismatch")
        q_spectrum = fft(extended, axis=1)[:, :spectrum_length].astype(
            complex_dtype, copy=False
        )
        del extended
        data_spectrum_norm = float(np.linalg.norm(q_spectrum))
        phi = np.zeros((n_state, k), dtype=real_dtype)
        for mode_index in range(k):
            row = mode_index % n_state
            phi[row, mode_index] = real_dtype.type(
                1.0 / math.sqrt(float(weights[row]))
            )
        if data_spectrum_norm == 0:
            coefficient_spectrum = np.zeros((spectrum_length, k), dtype=complex_dtype)
        else:
            coefficient_spectrum = np.full(
                (spectrum_length, k), eps, dtype=complex_dtype
            )
        residual_spectrum = q_spectrum - phi @ coefficient_spectrum.T
        del q_spectrum
        omega_history = np.empty((k, maximum_steps + 1), dtype=real_dtype)
        difference_history = np.empty(maximum_steps, dtype=real_dtype)
        omega_history[:, 0] = omega
        completed_steps = 0
    else:
        state = _validated_restart_state(
            restart,
            q_work,
            k,
            alpha,
            dt,
            real_dtype,
            complex_dtype,
            n_dc,
            weights,
            maximum_steps,
        )
        if init_frequency_type is not None and int(init_frequency_type) != state.init_frequency_type:
            raise ValueError(
                "restart init_frequency_type mismatch: saved {}, requested {}".format(
                    state.init_frequency_type, init_frequency_type
                )
            )
        if requested_init_max_was_explicit and float(
            requested_init_max_hz_effective
        ) != float(state.init_frequency_max_hz_effective):
            raise ValueError(
                "restart initialization-frequency maximum differs from the saved state"
            )
        if initial_frequencies_hz is not None:
            requested_initial_omega = _initial_frequencies(
                k,
                n_dc,
                state.init_frequency_type,
                state.init_frequency_max_hz_effective * dt,
                initial_frequencies_hz,
                dt,
                random_seed,
                real_dtype,
            )
            if not np.array_equal(
                requested_initial_omega.astype(np.float64) / float(dt),
                np.asarray(state.initial_frequencies_hz, dtype=np.float64),
            ):
                raise ValueError(
                    "restart initial_frequencies_hz differ from the saved state"
                )
        effective_init_frequency_type = state.init_frequency_type
        init_max_hz_effective = state.init_frequency_max_hz_effective
        initial_frequencies_effective_hz = np.asarray(
            state.initial_frequencies_hz, dtype=np.float64
        ).copy()
        initial_random_seed_effective = int(restart.info.get("random_seed", random_seed))
        phi = np.asarray(state.phi_unsorted).copy()
        coefficient_spectrum = np.asarray(
            state.coefficient_spectrum_unsorted
        ).copy()
        residual_spectrum = np.asarray(state.residual_spectrum).copy()
        data_spectrum_norm = float(state.data_spectrum_norm)
        completed_steps = int(state.completed_steps)
        restart_from_completed_steps = completed_steps
        omega_history = np.empty((k, maximum_steps + 1), dtype=real_dtype)
        difference_history = np.empty(maximum_steps, dtype=real_dtype)
        omega_history[:, : completed_steps + 1] = state.omega_history_unsorted
        if completed_steps:
            difference_history[:completed_steps] = state.difference_history
        omega = omega_history[:, completed_steps].copy()

    scale_floor = float(eps * max(data_spectrum_norm, 1.0))

    previous_difference = (
        float("inf")
        if completed_steps == 0
        else float(difference_history[completed_steps - 1])
    )
    converged = previous_difference <= tolerance

    backend_audit: Dict[str, Any]
    if compute_backend == "torch-cuda":
        (
            phi,
            coefficient_spectrum,
            residual_spectrum,
            omega,
            omega_history,
            difference_history,
            completed_steps,
            previous_difference,
            converged,
            backend_audit,
        ) = _iterate_torch_cuda(
            phi=phi,
            coefficient_spectrum=coefficient_spectrum,
            residual_spectrum=residual_spectrum,
            omega=omega,
            frequency=frequency,
            frequency_weight=frequency_weight,
            weights=weights,
            omega_history=omega_history,
            difference_history=difference_history,
            completed_steps=completed_steps,
            restart_from_completed_steps=restart_from_completed_steps,
            maximum_steps=maximum_steps,
            previous_difference=previous_difference,
            tolerance=tolerance,
            alpha=alpha,
            n_dc=n_dc,
            scale_floor=scale_floor,
            dt=dt,
            verbose=verbose,
        )
    else:
        started = time.perf_counter()
        for step in range(completed_steps, maximum_steps):
            if previous_difference <= tolerance:
                break
            difference = 0.0
            for mode_index in range(k):
                old_mode = phi[:, mode_index, None] * coefficient_spectrum[:, mode_index][None, :]
                residual_spectrum += old_mode

                projection = residual_spectrum @ (
                    np.conj(coefficient_spectrum[:, mode_index]) * frequency_weight
                )
                projection = np.real(projection).astype(real_dtype, copy=False)
                projection_norm = math.sqrt(
                    float(np.sum(np.abs(projection) ** 2 * weights, dtype=np.float64))
                )
                if np.isfinite(projection_norm) and projection_norm > 0:
                    phi[:, mode_index] = projection / projection_norm
                else:
                    fallback = phi[:, mode_index]
                    fallback_norm = math.sqrt(
                        float(np.sum(np.abs(fallback) ** 2 * weights, dtype=np.float64))
                    )
                    if not np.isfinite(fallback_norm) or fallback_norm <= 0:
                        fallback.fill(0)
                        row = mode_index % n_state
                        fallback[row] = real_dtype.type(1.0 / math.sqrt(float(weights[row])))
                    else:
                        fallback /= fallback_norm
                pivot = int(np.argmax(np.abs(phi[:, mode_index])))
                if phi[pivot, mode_index] < 0:
                    phi[:, mode_index] *= -1

                denominator = 1.0 + 2.0 * float(alpha) * (
                    frequency - omega[mode_index]
                ) ** 2
                coefficient_spectrum[:, mode_index] = (
                    residual_spectrum.T @ (phi[:, mode_index] * weights)
                ) / denominator
                coefficient_energy = float(
                    np.sum(
                        frequency_weight * np.abs(coefficient_spectrum[:, mode_index]) ** 2,
                        dtype=np.float64,
                    )
                )
                if mode_index < n_dc:
                    new_omega = 0.0
                elif np.isfinite(coefficient_energy) and coefficient_energy > 0:
                    new_omega = float(
                        np.sum(
                            frequency_weight
                            * frequency
                            * np.abs(coefficient_spectrum[:, mode_index]) ** 2,
                            dtype=np.float64,
                        )
                        / coefficient_energy
                    )
                else:
                    new_omega = float(omega[mode_index])

                new_mode = phi[:, mode_index, None] * coefficient_spectrum[:, mode_index][None, :]
                residual_spectrum -= new_mode
                old_norm = float(np.linalg.norm(old_mode))
                change_norm = float(np.linalg.norm(new_mode - old_mode))
                difference += change_norm / max(old_norm, scale_floor)
                omega[mode_index] = real_dtype.type(new_omega)

            completed_steps = step + 1
            difference_history[step] = real_dtype.type(difference)
            omega_history[:, step + 1] = omega
            previous_difference = float(difference)
            if verbose and (
                step == restart_from_completed_steps
                or completed_steps % 20 == 0
                or difference <= tolerance
            ):
                frequencies_hz = np.sort(np.asarray(omega, dtype=np.float64) / float(dt))
                print(
                    "RVMD step {:d}: difference={:.6e}, f_Hz=[{:.5g}, {:.5g}], "
                    "unique={:d}/{:d}".format(
                        completed_steps,
                        difference,
                        float(frequencies_hz[0]),
                        float(frequencies_hz[-1]),
                        int(np.unique(frequencies_hz).size),
                        int(frequencies_hz.size),
                    ),
                    flush=True,
                )
            if difference <= tolerance:
                converged = True
                break
        backend_audit = {
            "compute_backend_requested": "numpy",
            "compute_backend_effective": "numpy",
            "compute_library": "numpy",
            "compute_library_version": str(np.__version__),
            "compute_device": "cpu",
            "deterministic_algorithms": True,
            "allow_tf32": False,
            "iteration_elapsed_seconds_this_call": float(time.perf_counter() - started),
            "peak_gpu_memory_bytes": 0,
        }

    full_spectrum = np.zeros((extended_length, k), dtype=complex_dtype)
    full_spectrum[:spectrum_length, :] = coefficient_spectrum
    full_spectrum[spectrum_length:, :] = np.conj(coefficient_spectrum[-2:0:-1, :])
    coefficient = np.real(ifft(full_spectrum, axis=0))[half : half + n_time, :]
    coefficient = coefficient.astype(real_dtype, copy=False)

    order = np.argsort(omega, kind="mergesort")
    phi_sorted = phi[:, order]
    coefficient_sorted = coefficient[:, order]
    omega_sorted = omega[order]
    spectrum_sorted = coefficient_spectrum[:, order]
    frequency_hz = omega_sorted.astype(np.float64) / float(dt)
    strouhal = frequency_hz * float(strouhal_length) / float(strouhal_velocity)
    energy = np.sum(np.abs(coefficient_sorted) ** 2, axis=0, dtype=np.float64)
    audit = energy_audit(q_work, phi_sorted, coefficient_sorted, weights)
    audit["final_one_sided_spectral_residual_relative"] = float(
        np.linalg.norm(residual_spectrum) / max(data_spectrum_norm, np.finfo(np.float64).tiny)
    )
    audit["sum_reference_temporal_mode_energy"] = float(np.sum(energy))

    omega_history_output = omega_history[:, : completed_steps + 1].copy()
    difference_history_output = difference_history[:completed_steps].copy()
    restart_state = RVMDRestartState(
        phi_unsorted=phi.copy(),
        coefficient_spectrum_unsorted=coefficient_spectrum.copy(),
        residual_spectrum=residual_spectrum.copy(),
        weights=weights.copy(),
        omega_history_unsorted=omega_history_output.copy(),
        difference_history=difference_history_output.copy(),
        data_spectrum_norm=float(data_spectrum_norm),
        completed_steps=int(completed_steps),
        n_state=int(n_state),
        n_time=int(n_time),
        k=int(k),
        alpha=float(alpha),
        dt=float(dt),
        dtype=real_dtype.name,
        n_dc=int(n_dc),
        init_frequency_type=int(effective_init_frequency_type),
        init_frequency_max_hz_effective=float(init_max_hz_effective),
        initial_frequencies_hz=initial_frequencies_effective_hz.copy(),
        input_sha256=_array_sha256(q_work),
        maximum_steps_at_save=int(maximum_steps),
        tolerance_at_save=float(tolerance),
    )
    restart_state.array_sha256 = restart_state.current_array_sha256()

    info: Dict[str, Any] = {
        "algorithm": "reduced-order variational mode decomposition",
        "reference": "Liao et al., Journal of Fluid Mechanics 966, A7 (2023)",
        "doi": "10.1017/jfm.2023.435",
        "reconstruction_convention": "q_approx = phi @ c.T",
        "real_input": True,
        "n_state": int(n_state),
        "n_time": int(n_time),
        "k": int(k),
        "alpha_normalized_frequency": float(alpha),
        "dt_seconds": float(dt),
        "sample_rate_hz": float(1.0 / dt),
        "strouhal_formula": "St = f_Hz * 13.716 / 20",
        "strouhal_length": float(strouhal_length),
        "strouhal_velocity": float(strouhal_velocity),
        "tolerance": float(tolerance),
        "stopping_metric": (
            "sum_over_modes_of_relative_Frobenius_change_in_each_rank_one_"
            "spatiotemporal_mode_per_completed_sweep"
        ),
        "final_stopping_metric_sum": float(previous_difference),
        "final_stopping_metric_mean_per_mode": float(previous_difference / float(k)),
        "convergence_rule": "final_stopping_metric_sum <= tolerance",
        "maximum_steps": int(maximum_steps),
        "completed_steps": int(completed_steps),
        "converged": bool(converged),
        "stop_reason": "converged" if converged else "maximum_steps",
        "n_dc": int(n_dc),
        "init_frequency_type": int(effective_init_frequency_type),
        "init_frequency_max_hz_effective": float(init_max_hz_effective),
        "initial_frequencies_hz": initial_frequencies_effective_hz.tolist(),
        "random_seed": int(initial_random_seed_effective),
        "dtype": real_dtype.name,
        "mirror_extended_length": int(extended_length),
        "one_sided_spectrum_length": int(spectrum_length),
        "one_sided_endpoint_weights": [0.5, 0.5],
        "real_projection_enforced": True,
        "mode_sort": "ascending center frequency",
        "restart_state_schema_version": RESTART_SCHEMA_VERSION,
        "restart_used_this_call": restart is not None,
        "restart_from_completed_steps": int(restart_from_completed_steps),
        "maximum_steps_semantics": "total sweeps including saved restart sweeps",
        "restart_internal_state_order": "unsorted Gauss-Seidel update order",
    }
    info.update(backend_audit)

    return RVMDResult(
        phi=phi_sorted,
        c=coefficient_sorted,
        omega_cycles_per_sample=omega_sorted,
        frequency_hz=frequency_hz,
        strouhal=strouhal,
        energy=energy,
        coefficient_spectrum=spectrum_sorted,
        omega_history=omega_history_output,
        difference_history=difference_history_output,
        info=info,
        audit=audit,
        restart_state=restart_state,
    )


__all__ = [
    "RVMDRestartState",
    "RVMDResult",
    "coefficient_spectrum_moments",
    "energy_audit",
    "mirror_extend",
    "rvmd",
]
