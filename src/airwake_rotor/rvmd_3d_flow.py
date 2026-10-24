# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Validated three-dimensional velocity access for RVMD and raw CFD fields.

The full-space regression written by :mod:`nato_gd_rvmd_adapter` uses the
physical reconstruction

``u(x, t) = mean(x) + sum_k phi_k(x) * (c_k(t) - mean_t(c_k))``.

This module is the single place where that convention is turned into a
spatially and temporally interpolated velocity field.  It deliberately
rejects spatial clipping, time extrapolation, ambiguous modal selections,
non-Cartesian coordinates, and inconsistent artifacts.  Both
``RVMD3DFlow`` and ``NatoGD3DBaselineFlow`` expose the minimal common method
``velocity_at(relative_time_s, points_xyz)`` used by the dynamics solver.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


class RVMD3DFlowError(ValueError):
    """Base class for invalid RVMD flow artifacts or queries."""


class ArtifactValidationError(RVMD3DFlowError):
    """Raised when a persisted RVMD artifact set is internally inconsistent."""


class SpatialOutOfBoundsError(RVMD3DFlowError):
    """Raised when a requested point is outside the complete Cartesian grid."""


class TimeOutOfBoundsError(RVMD3DFlowError):
    """Raised when a relative time would require temporal extrapolation."""


@dataclass(frozen=True)
class RVMDFlowSample:
    """Detailed RVMD sample returned when modal contributions are requested.

    ``velocity`` and ``mean_velocity`` have shape ``points_xyz.shape``.
    ``mode_contributions`` has shape ``points_xyz.shape[:-1] + (n_modes, 3)``.
    The order along its modal axis is exactly ``mode_indices``.
    """

    velocity: np.ndarray
    mean_velocity: np.ndarray
    mode_contributions: np.ndarray
    mode_indices: np.ndarray
    centered_coefficients: np.ndarray


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError("required artifact is missing: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exception:
        raise ArtifactValidationError("cannot read metadata JSON {}: {}".format(path, exception))
    if not isinstance(value, dict):
        raise ArtifactValidationError("metadata JSON root must be an object")
    return value


def _require_artifact(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise ArtifactValidationError("required artifact is missing: {}".format(path))
    return path


def _positive_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exception:
        raise ArtifactValidationError("{} must be a finite positive number".format(label)) from exception
    if not math.isfinite(result) or result <= 0.0:
        raise ArtifactValidationError("{} must be a finite positive number".format(label))
    return result


def _integer_metadata(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ArtifactValidationError("{} must be an integer".format(label))
    result = int(value)
    if result < minimum:
        raise ArtifactValidationError("{} must be at least {}".format(label, minimum))
    return result


def _assert_finite_in_blocks(array: np.ndarray, label: str, block_rows: int = 65536) -> None:
    if array.ndim == 0:
        if not bool(np.isfinite(array)):
            raise ArtifactValidationError("{} contains a non-finite value".format(label))
        return
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        if not bool(np.isfinite(np.asarray(array[start:stop])).all()):
            raise ArtifactValidationError("{} contains NaN or infinity".format(label))


def _normalise_points(points_xyz: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], bool]:
    points = np.asarray(points_xyz, dtype=np.float64)
    single = points.ndim == 1
    if single:
        if points.shape != (3,):
            raise ValueError("one point must have shape (3,)")
        batch_shape: Tuple[int, ...] = ()
        flat = points.reshape(1, 3)
    else:
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("batch points must have shape (..., 3)")
        batch_shape = tuple(int(value) for value in points.shape[:-1])
        flat = points.reshape(-1, 3)
    if flat.shape[0] == 0:
        raise ValueError("point batch must not be empty")
    if not bool(np.isfinite(flat).all()):
        row, column = np.argwhere(~np.isfinite(flat))[0]
        raise SpatialOutOfBoundsError(
            "point batch index {} has non-finite {} coordinate".format(
                int(row), ("x", "y", "z")[int(column)]
            )
        )
    return flat, batch_shape, single


def _regular_axis(values: np.ndarray, label: str) -> Tuple[np.ndarray, Optional[float]]:
    axis = np.unique(np.asarray(values, dtype=np.float64))
    if axis.size == 1:
        # A complete Cartesian product may legitimately contain one plane in
        # an axis.  Interpolation then degenerates to bilinear/linear sampling
        # and only the exact stored coordinate is in bounds.
        return axis, None
    differences = np.diff(axis)
    if not bool(np.all(differences > 0.0)):
        raise ArtifactValidationError("{} axis is not strictly increasing".format(label))
    spacing = float(np.median(differences))
    # coordinates.npy is normally float32.  This is the same storage-aware
    # tolerance used by the adapter audit; it admits representation noise but
    # still rejects a geometrically nonuniform grid.
    if not bool(np.allclose(differences, spacing, rtol=1.0e-4, atol=2.0e-5)):
        raise ArtifactValidationError("{} axis is not regularly spaced".format(label))
    return axis, spacing


def _cartesian_row_maps(
    coordinates: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[Optional[float], Optional[float], Optional[float]],
]:
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ArtifactValidationError("coordinates.npy must have shape (n_points, 3)")
    _assert_finite_in_blocks(coords, "coordinates.npy")
    coords64 = np.asarray(coords, dtype=np.float64)
    x, dx = _regular_axis(coords64[:, 0], "x")
    y, dy = _regular_axis(coords64[:, 1], "y")
    z, dz = _regular_axis(coords64[:, 2], "z")
    nx, ny, nz = int(x.size), int(y.size), int(z.size)
    expected = nx * ny * nz
    if coords64.shape[0] != expected:
        raise ArtifactValidationError(
            "coordinates do not form a complete Cartesian product: rows={}, nx*ny*nz={}".format(
                coords64.shape[0], expected
            )
        )
    ix = np.searchsorted(x, coords64[:, 0]).astype(np.int64)
    iy = np.searchsorted(y, coords64[:, 1]).astype(np.int64)
    iz = np.searchsorted(z, coords64[:, 2]).astype(np.int64)
    if (
        bool(np.any(ix >= nx))
        or bool(np.any(iy >= ny))
        or bool(np.any(iz >= nz))
        or not bool(np.array_equal(x[ix], coords64[:, 0]))
        or not bool(np.array_equal(y[iy], coords64[:, 1]))
        or not bool(np.array_equal(z[iz], coords64[:, 2]))
    ):
        raise ArtifactValidationError("coordinate-to-axis mapping is inconsistent")
    row_to_grid_flat = (iz * ny + iy) * nx + ix
    counts = np.bincount(row_to_grid_flat, minlength=expected)
    if counts.size != expected or not bool(np.all(counts == 1)):
        raise ArtifactValidationError("Cartesian coordinates contain duplicate or missing cells")
    grid_flat_to_row = np.empty(expected, dtype=np.int64)
    grid_flat_to_row[row_to_grid_flat] = np.arange(expected, dtype=np.int64)
    row_index_zyx = grid_flat_to_row.reshape(nz, ny, nx)
    return x, y, z, row_to_grid_flat, row_index_zyx, (dx, dy, dz)


def _declared_shape(metadata: Mapping[str, Any], key: str) -> Optional[Tuple[int, ...]]:
    output_arrays = metadata.get("output_arrays")
    if not isinstance(output_arrays, dict) or key not in output_arrays:
        return None
    raw = output_arrays[key]
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in raw
    ):
        raise ArtifactValidationError("metadata output_arrays[{!r}] is not an integer shape".format(key))
    return tuple(int(value) for value in raw)


class RVMD3DFlow:
    """Read-only, bounds-safe RVMD reconstruction over a regular 3-D grid."""

    REQUIRED_FILES = (
        "coordinates.npy",
        "mean_field_physical.npy",
        "spatial_modes_physical.npy",
        "coefficient_mean_for_full_regression.npy",
        "temporal_model.npz",
        "metadata.json",
    )

    def __init__(self, artifact_directory: PathLike, mmap_mode: Optional[str] = "r") -> None:
        self.artifact_directory = Path(artifact_directory).expanduser().resolve()
        if not self.artifact_directory.is_dir():
            raise FileNotFoundError(
                "RVMD artifact directory does not exist: {}".format(self.artifact_directory)
            )
        for filename in self.REQUIRED_FILES:
            _require_artifact(self.artifact_directory, filename)

        self.metadata = _load_json_object(self.artifact_directory / "metadata.json")
        variables = self.metadata.get("variables")
        if variables != ["u", "v", "w"] and variables != ("u", "v", "w"):
            raise ArtifactValidationError(
                "RVMD flow variables must be exactly ['u', 'v', 'w'] in that order; got {!r}".format(
                    variables
                )
            )

        self.coordinates = np.load(
            str(self.artifact_directory / "coordinates.npy"), mmap_mode=mmap_mode, allow_pickle=False
        )
        self.mean_field = np.load(
            str(self.artifact_directory / "mean_field_physical.npy"),
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.spatial_modes = np.load(
            str(self.artifact_directory / "spatial_modes_physical.npy"),
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.coefficient_mean = np.asarray(
            np.load(
                str(self.artifact_directory / "coefficient_mean_for_full_regression.npy"),
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        with np.load(
            str(self.artifact_directory / "temporal_model.npz"), allow_pickle=False
        ) as temporal:
            required_temporal = (
                "c",
                "frequency_hz",
                "strouhal",
                "omega_cycles_per_sample",
                "info_json",
            )
            missing = [name for name in required_temporal if name not in temporal.files]
            if missing:
                raise ArtifactValidationError(
                    "temporal_model.npz is missing arrays: {}".format(", ".join(missing))
                )
            self.coefficients = np.asarray(temporal["c"], dtype=np.float64)
            self.frequency_hz = np.asarray(temporal["frequency_hz"], dtype=np.float64)
            self.strouhal = np.asarray(temporal["strouhal"], dtype=np.float64)
            omega = np.asarray(temporal["omega_cycles_per_sample"], dtype=np.float64)
            try:
                info_text = str(np.asarray(temporal["info_json"]).item())
                self.temporal_info = json.loads(info_text)
            except (ValueError, TypeError, json.JSONDecodeError) as exception:
                raise ArtifactValidationError("temporal_model.npz/info_json is invalid") from exception
        if not isinstance(self.temporal_info, dict):
            raise ArtifactValidationError("temporal_model.npz/info_json must encode an object")

        self._validate_array_shapes_and_values(omega)
        (
            self.x,
            self.y,
            self.z,
            self.row_to_grid_flat_zyx,
            self.row_index_zyx,
            self.spacing_xyz_m,
        ) = _cartesian_row_maps(self.coordinates)
        self.shape_zyx = (int(self.z.size), int(self.y.size), int(self.x.size))
        self._validate_metadata_and_grid(omega)

    @property
    def mode_count(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def relative_time_end_s(self) -> float:
        return float((self.frame_count - 1) * self.dt_s)

    @property
    def bounds_xyz_m(self) -> np.ndarray:
        return np.asarray(
            [[self.x[0], self.x[-1]], [self.y[0], self.y[-1]], [self.z[0], self.z[-1]]],
            dtype=np.float64,
        )

    def close(self) -> None:
        """Close memory maps explicitly (important for temporary files on Windows)."""

        for array in (self.coordinates, self.mean_field, self.spatial_modes):
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __enter__(self) -> "RVMD3DFlow":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

    def _validate_array_shapes_and_values(self, omega: np.ndarray) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ArtifactValidationError("coordinates.npy must have shape (n_points, 3)")
        n_points = int(self.coordinates.shape[0])
        if self.mean_field.shape != (n_points, 3):
            raise ArtifactValidationError(
                "mean_field_physical.npy must have shape ({}, 3); got {}".format(
                    n_points, self.mean_field.shape
                )
            )
        if self.spatial_modes.ndim != 3 or self.spatial_modes.shape[:2] != (n_points, 3):
            raise ArtifactValidationError(
                "spatial_modes_physical.npy must have shape (n_points, 3, k); got {}".format(
                    self.spatial_modes.shape
                )
            )
        if self.coefficients.ndim != 2 or self.coefficients.shape[0] < 2:
            raise ArtifactValidationError("temporal coefficients c must have shape (n_time>=2, k)")
        k = int(self.coefficients.shape[1])
        if k < 1 or self.spatial_modes.shape[2] != k:
            raise ArtifactValidationError("spatial-mode and temporal-coefficient mode counts disagree")
        for label, array in (
            ("coefficient_mean_for_full_regression.npy", self.coefficient_mean),
            ("frequency_hz", self.frequency_hz),
            ("strouhal", self.strouhal),
            ("omega_cycles_per_sample", omega),
        ):
            if array.shape != (k,):
                raise ArtifactValidationError("{} must have shape ({},); got {}".format(label, k, array.shape))
        _assert_finite_in_blocks(self.mean_field, "mean_field_physical.npy")
        _assert_finite_in_blocks(self.spatial_modes, "spatial_modes_physical.npy")
        _assert_finite_in_blocks(self.coefficients, "temporal coefficients")
        _assert_finite_in_blocks(self.coefficient_mean, "coefficient mean")
        _assert_finite_in_blocks(self.frequency_hz, "frequency_hz")
        _assert_finite_in_blocks(self.strouhal, "strouhal")
        _assert_finite_in_blocks(omega, "omega_cycles_per_sample")
        actual_mean = np.mean(self.coefficients, axis=0, dtype=np.float64)
        if not bool(np.allclose(self.coefficient_mean, actual_mean, rtol=2.0e-10, atol=2.0e-12)):
            maximum_error = float(np.max(np.abs(self.coefficient_mean - actual_mean)))
            raise ArtifactValidationError(
                "saved coefficient mean disagrees with mean(temporal c); max error={:.6g}".format(
                    maximum_error
                )
            )
        frequency_tolerance = np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(self.frequency_hz)))
        ) * 32.0
        if bool(np.any(self.frequency_hz < -frequency_tolerance)):
            raise ArtifactValidationError("frequency_hz contains a negative center frequency")
        if bool(np.any(np.diff(self.frequency_hz) < -frequency_tolerance)):
            raise ArtifactValidationError(
                "RVMD modes are not stored in ascending center-frequency order"
            )

    def _validate_metadata_and_grid(self, omega: np.ndarray) -> None:
        n_points = int(self.coordinates.shape[0])
        n_time, k = (int(value) for value in self.coefficients.shape)
        for key, expected in (("n_points", n_points), ("n_frames", n_time), ("k", k)):
            actual = _integer_metadata(self.metadata.get(key), "metadata/{}".format(key))
            if actual != expected:
                raise ArtifactValidationError(
                    "metadata/{}={} disagrees with artifact value {}".format(key, actual, expected)
                )
        self.dt_s = _positive_finite(self.metadata.get("dt_seconds"), "metadata/dt_seconds")
        info_dt = _positive_finite(self.temporal_info.get("dt_seconds"), "temporal info dt_seconds")
        if not math.isclose(self.dt_s, info_dt, rel_tol=0.0, abs_tol=1.0e-15):
            raise ArtifactValidationError("metadata and temporal model use different time steps")
        for key, expected in (("n_time", n_time), ("k", k)):
            actual = _integer_metadata(
                self.temporal_info.get(key), "temporal info {}".format(key)
            )
            if actual != expected:
                raise ArtifactValidationError(
                    "temporal info {}={} disagrees with artifact value {}".format(
                        key, actual, expected
                    )
                )
        if not bool(np.allclose(omega, self.frequency_hz * self.dt_s, rtol=2.0e-10, atol=2.0e-12)):
            raise ArtifactValidationError("omega_cycles_per_sample is inconsistent with f_Hz * dt")
        expected_st = self.frequency_hz * 13.716 / 20.0
        if not bool(np.allclose(self.strouhal, expected_st, rtol=2.0e-10, atol=2.0e-12)):
            raise ArtifactValidationError("Strouhal values do not satisfy St = f * 13.716 / 20")
        nyquist = 0.5 / self.dt_s
        if bool(np.any(self.frequency_hz > nyquist * (1.0 + 1.0e-10))):
            raise ArtifactValidationError("an RVMD center frequency exceeds the Nyquist frequency")

        grid = self.metadata.get("grid_audit")
        if not isinstance(grid, dict):
            raise ArtifactValidationError("metadata/grid_audit must be an object")
        if grid.get("is_complete_cartesian_product") is not True:
            raise ArtifactValidationError("metadata does not certify a complete Cartesian grid")
        if grid.get("is_uniform_each_axis") is not True:
            raise ArtifactValidationError("metadata does not certify uniform coordinate axes")
        expected_nxyz = [int(self.x.size), int(self.y.size), int(self.z.size)]
        if grid.get("shape_nx_ny_nz") != expected_nxyz:
            raise ArtifactValidationError(
                "metadata grid shape {!r} disagrees with coordinates {}".format(
                    grid.get("shape_nx_ny_nz"), expected_nxyz
                )
            )
        for key, actual in (
            ("coordinates.npy", self.coordinates.shape),
            ("mean_field_physical.npy", self.mean_field.shape),
            ("spatial_modes_physical.npy", self.spatial_modes.shape),
            ("coefficient_mean_for_full_regression.npy", self.coefficient_mean.shape),
            ("temporal_model.npz/c", self.coefficients.shape),
        ):
            declared = _declared_shape(self.metadata, key)
            if declared is not None and declared != tuple(int(value) for value in actual):
                raise ArtifactValidationError(
                    "metadata declares {} shape {}, but artifact shape is {}".format(
                        key, declared, tuple(actual)
                    )
                )
        time_axis = self.metadata.get("time_axis")
        if isinstance(time_axis, dict):
            relative = time_axis.get("relative_time_from_first_frame_seconds")
            if isinstance(relative, dict):
                try:
                    first = float(relative.get("first"))
                    last = float(relative.get("last"))
                except (TypeError, ValueError) as exception:
                    raise ArtifactValidationError("metadata relative time bounds are invalid") from exception
                if not math.isclose(first, 0.0, rel_tol=0.0, abs_tol=1.0e-15) or not math.isclose(
                    last, self.relative_time_end_s, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ArtifactValidationError("metadata relative time bounds disagree with c and dt")

    def validate(self) -> Dict[str, Any]:
        """Return a compact audit after all constructor-time hard checks pass."""

        return {
            "valid": True,
            "n_points": int(self.coordinates.shape[0]),
            "grid_shape_zyx": list(self.shape_zyx),
            "n_frames": self.frame_count,
            "n_modes": self.mode_count,
            "dt_s": self.dt_s,
            "relative_time_end_s": self.relative_time_end_s,
            "bounds_xyz_m": self.bounds_xyz_m.tolist(),
            "variables": ["u", "v", "w"],
            "mode_order": "ascending center frequency",
            "reconstruction": "mean + sum(phi_k * (c_k - coefficient_mean_k))",
            "spatial_bounds_policy": "hard-fail; no clipping or extrapolation",
            "temporal_bounds_policy": "hard-fail; no extrapolation",
        }

    def _check_spatial_bounds(self, flat_points: np.ndarray) -> None:
        bounds = self.bounds_xyz_m
        invalid = (flat_points < bounds[:, 0]) | (flat_points > bounds[:, 1])
        if bool(invalid.any()):
            row, column = np.argwhere(invalid)[0]
            axis = ("x", "y", "z")[int(column)]
            raise SpatialOutOfBoundsError(
                "point batch index {} is outside the {} domain: value={!r}, bounds=[{!r}, {!r}]; "
                "clipping and extrapolation are disabled".format(
                    int(row),
                    axis,
                    float(flat_points[row, column]),
                    float(bounds[column, 0]),
                    float(bounds[column, 1]),
                )
            )

    @staticmethod
    def _axis_brackets(
        axis: np.ndarray, query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if axis.size == 1:
            index = np.zeros(query.shape, dtype=np.int64)
            return index, index.copy(), np.zeros(query.shape, dtype=np.float64)
        upper = np.searchsorted(axis, query, side="right").astype(np.int64)
        at_upper_endpoint = upper == axis.size
        upper[at_upper_endpoint] = axis.size - 1
        lower = upper - 1
        lower[lower < 0] = 0
        fraction = (query - axis[lower]) / (axis[upper] - axis[lower])
        return lower, upper, fraction

    def _corner_rows_and_weights(
        self, flat_points: np.ndarray
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        ix0, ix1, fx = self._axis_brackets(self.x, flat_points[:, 0])
        iy0, iy1, fy = self._axis_brackets(self.y, flat_points[:, 1])
        iz0, iz1, fz = self._axis_brackets(self.z, flat_points[:, 2])
        return (
            (self.row_index_zyx[iz0, iy0, ix0], (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
            (self.row_index_zyx[iz0, iy0, ix1], fx * (1.0 - fy) * (1.0 - fz)),
            (self.row_index_zyx[iz0, iy1, ix0], (1.0 - fx) * fy * (1.0 - fz)),
            (self.row_index_zyx[iz0, iy1, ix1], fx * fy * (1.0 - fz)),
            (self.row_index_zyx[iz1, iy0, ix0], (1.0 - fx) * (1.0 - fy) * fz),
            (self.row_index_zyx[iz1, iy0, ix1], fx * (1.0 - fy) * fz),
            (self.row_index_zyx[iz1, iy1, ix0], (1.0 - fx) * fy * fz),
            (self.row_index_zyx[iz1, iy1, ix1], fx * fy * fz),
        )

    @staticmethod
    def _weighted_rows(
        field: np.ndarray,
        corners: Sequence[Tuple[np.ndarray, np.ndarray]],
        mode_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result: Optional[np.ndarray] = None
        for rows, weights in corners:
            values = np.asarray(field[rows], dtype=np.float64)
            if mode_indices is not None:
                values = values[..., mode_indices]
            broadcast = weights.reshape((weights.size,) + (1,) * (values.ndim - 1))
            term = values * broadcast
            if result is None:
                result = term
            else:
                result += term
        assert result is not None
        return result

    def _select_modes(
        self,
        retained_mode_count: Optional[int],
        mode_indices: Optional[Iterable[int]],
    ) -> np.ndarray:
        if retained_mode_count is not None and mode_indices is not None:
            raise ValueError("specify retained_mode_count or mode_indices, not both")
        if retained_mode_count is not None:
            if isinstance(retained_mode_count, bool) or not isinstance(retained_mode_count, Integral):
                raise TypeError("retained_mode_count must be an integer")
            count = int(retained_mode_count)
            if count < 0 or count > self.mode_count:
                raise ValueError(
                    "retained_mode_count must be in [0, {}]".format(self.mode_count)
                )
            return np.arange(count, dtype=np.int64)
        if mode_indices is None:
            return np.arange(self.mode_count, dtype=np.int64)
        raw = list(mode_indices)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
            raise TypeError("mode_indices must contain only integers")
        indices = np.asarray([int(value) for value in raw], dtype=np.int64)
        if indices.size and (bool(np.any(indices < 0)) or bool(np.any(indices >= self.mode_count))):
            raise IndexError("mode index is outside [0, {}]".format(self.mode_count - 1))
        if np.unique(indices).size != indices.size:
            raise ValueError("mode_indices must not contain duplicates")
        return indices

    def centered_coefficients_at(
        self,
        relative_time_s: float,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate ``c-c_mean`` without temporal extrapolation."""

        if isinstance(relative_time_s, bool):
            raise TypeError("relative_time_s must be a scalar number")
        try:
            time_value = float(relative_time_s)
        except (TypeError, ValueError) as exception:
            raise TypeError("relative_time_s must be a scalar number") from exception
        if not math.isfinite(time_value):
            raise TimeOutOfBoundsError("relative_time_s must be finite")
        if time_value < 0.0 or time_value > self.relative_time_end_s:
            raise TimeOutOfBoundsError(
                "relative_time_s={!r} outside [0.0, {!r}]; extrapolation is disabled".format(
                    time_value, self.relative_time_end_s
                )
            )
        indices = self._select_modes(retained_mode_count, mode_indices)
        position = time_value / self.dt_s
        lower = int(math.floor(position))
        if lower >= self.frame_count - 1:
            coefficient = self.coefficients[-1]
        else:
            fraction = position - lower
            coefficient = (
                self.coefficients[lower] * (1.0 - fraction)
                + self.coefficients[lower + 1] * fraction
            )
        centered = coefficient[indices] - self.coefficient_mean[indices]
        return centered, indices

    def spatial_modes_at(
        self,
        points_xyz: np.ndarray,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate mean and selected physical modes at arbitrary points.

        Returns ``(mean_velocity, spatial_modes, indices)``.  The modal array
        has shape ``points_xyz.shape[:-1] + (n_selected_modes, 3)``.
        """

        flat, batch_shape, single = _normalise_points(points_xyz)
        self._check_spatial_bounds(flat)
        indices = self._select_modes(retained_mode_count, mode_indices)
        corners = self._corner_rows_and_weights(flat)
        mean_flat = self._weighted_rows(self.mean_field, corners)
        if indices.size:
            # Stored order is [point, component, mode]; public order keeps the
            # vector component last: [..., mode, component].
            modes_flat = np.moveaxis(
                self._weighted_rows(self.spatial_modes, corners, indices), -1, -2
            )
        else:
            modes_flat = np.empty((flat.shape[0], 0, 3), dtype=np.float64)
        if single:
            return mean_flat[0], modes_flat[0], indices.copy()
        return (
            mean_flat.reshape(batch_shape + (3,)),
            modes_flat.reshape(batch_shape + (indices.size, 3)),
            indices.copy(),
        )

    def velocity_at(
        self,
        relative_time_s: float,
        points_xyz: np.ndarray,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
        return_mode_contributions: bool = False,
    ) -> Union[np.ndarray, RVMDFlowSample]:
        """Return reconstructed velocity, optionally with every modal term.

        ``retained_mode_count=r`` keeps the first ``r`` modes in the persisted
        ascending-frequency order.  Hence decreasing ``r`` removes the
        highest-frequency remaining mode at each stage.  ``mode_indices``
        selects any unique zero-based subset and preserves the requested order.
        """

        centered, indices = self.centered_coefficients_at(
            relative_time_s,
            retained_mode_count=retained_mode_count,
            mode_indices=mode_indices,
        )
        mean, modes, spatial_indices = self.spatial_modes_at(
            points_xyz, mode_indices=indices.tolist()
        )
        if not bool(np.array_equal(indices, spatial_indices)):
            raise RuntimeError("internal modal selection mismatch")
        coefficient_shape = (1,) * (modes.ndim - 2) + (indices.size, 1)
        contributions = modes * centered.reshape(coefficient_shape)
        velocity = mean + np.sum(contributions, axis=-2)
        if not return_mode_contributions:
            return velocity
        return RVMDFlowSample(
            velocity=velocity,
            mean_velocity=mean,
            mode_contributions=contributions,
            mode_indices=indices.copy(),
            centered_coefficients=centered.copy(),
        )

    def interpolate_time(
        self,
        relative_time_s: float,
        points_xyz: np.ndarray,
        **kwargs: Any
    ) -> Union[np.ndarray, RVMDFlowSample]:
        """Alias matching :class:`NatoGD3DCache` while retaining modal options."""

        return self.velocity_at(relative_time_s, points_xyz, **kwargs)

    def interpolate_frame(
        self,
        frame_index: int,
        points_xyz: np.ndarray,
        **kwargs: Any
    ) -> Union[np.ndarray, RVMDFlowSample]:
        """Sample at one exact coefficient frame with the same modal options."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
            raise TypeError("frame_index must be an integer")
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError("frame_index {} outside [0, {}]".format(index, self.frame_count - 1))
        return self.velocity_at(index * self.dt_s, points_xyz, **kwargs)


class NatoGD3DBaselineFlow:
    """Small adapter giving the raw CFD cache the common ``velocity_at`` API."""

    def __init__(self, cache_or_directory: Any, expected_dt_s: Optional[float] = None) -> None:
        if isinstance(cache_or_directory, (str, Path)):
            try:
                from .nato_gd_3d_io import NatoGD3DCache
            except ImportError:  # pragma: no cover - package import path
                from .nato_gd_3d_io import NatoGD3DCache

            self.cache = NatoGD3DCache(
                Path(cache_or_directory), expected_dt_s=expected_dt_s
            )
        else:
            self.cache = cache_or_directory
            if not callable(getattr(self.cache, "interpolate_time", None)):
                raise TypeError("cache object must provide interpolate_time(time, points)")
            if expected_dt_s is not None:
                observed = _positive_finite(getattr(self.cache, "dt_s", None), "cache dt_s")
                expected = _positive_finite(expected_dt_s, "expected_dt_s")
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-15):
                    raise ArtifactValidationError(
                        "cache dt_s={!r} differs from expected_dt_s={!r}".format(observed, expected)
                    )

    @staticmethod
    def _reject_modal_options(
        retained_mode_count: Optional[int],
        mode_indices: Optional[Iterable[int]],
        return_mode_contributions: bool,
    ) -> None:
        if retained_mode_count is not None or mode_indices is not None:
            raise ValueError("raw CFD baseline has no RVMD modes to retain or select")
        if return_mode_contributions:
            raise ValueError("raw CFD baseline has no per-mode contributions")

    def velocity_at(
        self,
        relative_time_s: float,
        points_xyz: np.ndarray,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
        return_mode_contributions: bool = False,
    ) -> np.ndarray:
        self._reject_modal_options(
            retained_mode_count, mode_indices, return_mode_contributions
        )
        return self.cache.interpolate_time(relative_time_s, points_xyz)

    def interpolate_time(
        self,
        relative_time_s: float,
        points_xyz: np.ndarray,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
        return_mode_contributions: bool = False,
    ) -> np.ndarray:
        return self.velocity_at(
            relative_time_s,
            points_xyz,
            retained_mode_count=retained_mode_count,
            mode_indices=mode_indices,
            return_mode_contributions=return_mode_contributions,
        )

    def interpolate_frame(
        self,
        frame_index: int,
        points_xyz: np.ndarray,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
        return_mode_contributions: bool = False,
    ) -> np.ndarray:
        self._reject_modal_options(
            retained_mode_count, mode_indices, return_mode_contributions
        )
        return self.cache.interpolate_frame(frame_index, points_xyz)

    def __getattr__(self, name: str) -> Any:
        # Expose dt_s, frame_count, bounds axes, and cache audits without
        # duplicating or weakening the cache implementation's validation.
        return getattr(self.cache, name)


__all__ = [
    "ArtifactValidationError",
    "NatoGD3DBaselineFlow",
    "RVMD3DFlow",
    "RVMD3DFlowError",
    "RVMDFlowSample",
    "SpatialOutOfBoundsError",
    "TimeOutOfBoundsError",
]
