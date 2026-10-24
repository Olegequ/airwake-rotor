#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Bounds-safe reconstruction of grouped physical 3-D DMD modes.

The persisted DMD artifact stores one or two real quadrature components for
each physical mode group.  Public mode indices in this adapter always refer to
the physical groups, never to the individual quadratures.  This is essential
for complex-conjugate DMD pairs: deleting only one quadrature would create a
non-physical flow input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .rvmd_3d_flow import (
    ArtifactValidationError,
    RVMD3DFlow,
    SpatialOutOfBoundsError,
    TimeOutOfBoundsError,
    _assert_finite_in_blocks,
    _cartesian_row_maps,
    _load_json_object,
    _normalise_points,
    _require_artifact,
)


PathLike = Union[str, Path]


@dataclass(frozen=True)
class DMDFlowSample:
    velocity: np.ndarray
    mean_velocity: np.ndarray
    mode_contributions: np.ndarray
    mode_indices: np.ndarray
    quadrature_indices: np.ndarray
    quadrature_coefficients: np.ndarray


class DMD3DFlow(RVMD3DFlow):
    """Read-only 3-D DMD reconstruction with physical-group selection.

    The class intentionally reuses the already audited trilinear interpolation
    and hard bounds checks of :class:`RVMD3DFlow`, but overrides temporal and
    modal selection because a DMD physical mode can contain two real
    quadratures.
    """

    REQUIRED_FILES = (
        "coordinates.npy",
        "mean_field_physical.npy",
        "spatial_modes_physical.npy",
        "temporal_model.npz",
        "metadata.json",
        "full_space_regression_audit.json",
    )

    def __init__(self, artifact_directory: PathLike, mmap_mode: Optional[str] = "r") -> None:
        self.artifact_directory = Path(artifact_directory).expanduser().resolve()
        if not self.artifact_directory.is_dir():
            raise FileNotFoundError(
                "DMD artifact directory does not exist: {}".format(self.artifact_directory)
            )
        for filename in self.REQUIRED_FILES:
            _require_artifact(self.artifact_directory, filename)

        self.metadata = _load_json_object(self.artifact_directory / "metadata.json")
        self.full_space_audit = _load_json_object(
            self.artifact_directory / "full_space_regression_audit.json"
        )
        variables = self.metadata.get("variables")
        if variables != ["u", "v", "w"] and variables != ("u", "v", "w"):
            raise ArtifactValidationError(
                "DMD flow variables must be exactly ['u', 'v', 'w'] in that order"
            )

        self.coordinates = np.load(
            str(self.artifact_directory / "coordinates.npy"), mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        self.mean_field = np.load(
            str(self.artifact_directory / "mean_field_physical.npy"),
            mmap_mode=mmap_mode, allow_pickle=False,
        )
        self.spatial_modes = np.load(
            str(self.artifact_directory / "spatial_modes_physical.npy"),
            mmap_mode=mmap_mode, allow_pickle=False,
        )
        with np.load(
            str(self.artifact_directory / "temporal_model.npz"), allow_pickle=False
        ) as temporal:
            required = (
                "temporal_coefficients",
                "group_component_start",
                "group_component_stop",
                "frequency_hz",
                "strouhal",
            )
            missing = [name for name in required if name not in temporal.files]
            if missing:
                raise ArtifactValidationError(
                    "DMD temporal_model.npz is missing arrays: {}".format(
                        ", ".join(missing)
                    )
                )
            self.coefficients = np.asarray(
                temporal["temporal_coefficients"], dtype=np.float64
            )
            self.group_component_start = np.asarray(
                temporal["group_component_start"], dtype=np.int64
            )
            self.group_component_stop = np.asarray(
                temporal["group_component_stop"], dtype=np.int64
            )
            self.frequency_hz = np.asarray(temporal["frequency_hz"], dtype=np.float64)
            self.strouhal = np.asarray(temporal["strouhal"], dtype=np.float64)
            self.eigenfrequency_hz = (
                np.asarray(temporal["eigenfrequency_hz"], dtype=np.float64)
                if "eigenfrequency_hz" in temporal.files
                else np.full(self.frequency_hz.shape, np.nan, dtype=np.float64)
            )
            self.growth_rate_1ps = (
                np.asarray(temporal["growth_rate_1ps"], dtype=np.float64)
                if "growth_rate_1ps" in temporal.files
                else np.full(self.frequency_hz.shape, np.nan, dtype=np.float64)
            )

        self.dt_s = float(self.metadata.get("dt_seconds", float("nan")))
        (
            self.x,
            self.y,
            self.z,
            self.row_to_grid_flat_zyx,
            self.row_index_zyx,
            self.spacing_xyz_m,
        ) = _cartesian_row_maps(self.coordinates)
        self.shape_zyx = (int(self.z.size), int(self.y.size), int(self.x.size))
        self._validate_dmd_artifact()

    @property
    def mode_count(self) -> int:
        """Number of physical DMD groups (not real quadrature columns)."""

        return int(self.group_component_start.size)

    @property
    def component_count(self) -> int:
        return int(self.coefficients.shape[1])

    def _validate_dmd_artifact(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ArtifactValidationError("DMD metadata/dt_seconds must be finite and positive")
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ArtifactValidationError("coordinates.npy must have shape (n_points, 3)")
        n_points = int(self.coordinates.shape[0])
        if self.mean_field.shape != (n_points, 3):
            raise ArtifactValidationError("DMD mean field has an inconsistent shape")
        if self.coefficients.ndim != 2 or self.coefficients.shape[0] < 2:
            raise ArtifactValidationError(
                "DMD temporal coefficients must have shape (n_time>=2, n_components)"
            )
        if self.spatial_modes.shape != (n_points, 3, self.component_count):
            raise ArtifactValidationError("DMD spatial modes and quadratures disagree")
        if self.mode_count < 1:
            raise ArtifactValidationError("DMD artifact contains no physical mode group")
        if self.group_component_stop.shape != self.group_component_start.shape:
            raise ArtifactValidationError("DMD group start/stop arrays disagree")
        if self.group_component_start[0] != 0:
            raise ArtifactValidationError("first DMD group must start at component zero")
        if self.group_component_stop[-1] != self.component_count:
            raise ArtifactValidationError("DMD group map does not cover every quadrature")
        if not np.array_equal(
            self.group_component_start[1:], self.group_component_stop[:-1]
        ):
            raise ArtifactValidationError("DMD group component map must be contiguous")
        widths = self.group_component_stop - self.group_component_start
        if bool(np.any((widths < 1) | (widths > 2))):
            raise ArtifactValidationError("each DMD physical group must have one or two quadratures")
        for label, values in (
            ("frequency_hz", self.frequency_hz),
            ("strouhal", self.strouhal),
        ):
            if values.shape != (self.mode_count,):
                raise ArtifactValidationError(
                    "{} must have shape ({},)".format(label, self.mode_count)
                )
            _assert_finite_in_blocks(values, label)
        if bool(np.any(self.frequency_hz < 0.0)) or bool(np.any(np.diff(self.frequency_hz) < 0.0)):
            raise ArtifactValidationError(
                "DMD physical groups must be stored in ascending nonnegative centre frequency"
            )
        expected_st = self.frequency_hz * 13.716 / 20.0
        if not bool(np.allclose(self.strouhal, expected_st, rtol=2.0e-10, atol=2.0e-12)):
            raise ArtifactValidationError("DMD St values do not satisfy St=f*13.716/20")
        if float(self.frequency_hz[-1]) > 0.5 / self.dt_s * (1.0 + 1.0e-10):
            raise ArtifactValidationError("a DMD centre frequency exceeds Nyquist")
        if int(self.metadata.get("n_points", -1)) != n_points:
            raise ArtifactValidationError("DMD metadata n_points disagrees with coordinates")
        if int(self.metadata.get("n_frames", -1)) != self.frame_count:
            raise ArtifactValidationError("DMD metadata n_frames disagrees with coefficients")
        if int(self.metadata.get("selected_physical_groups", -1)) != self.mode_count:
            raise ArtifactValidationError("DMD metadata physical-group count disagrees")
        if int(self.metadata.get("selected_real_quadrature_components", -1)) != self.component_count:
            raise ArtifactValidationError("DMD metadata quadrature count disagrees")
        _assert_finite_in_blocks(self.mean_field, "mean_field_physical.npy")
        _assert_finite_in_blocks(self.spatial_modes, "spatial_modes_physical.npy")
        _assert_finite_in_blocks(self.coefficients, "temporal_coefficients")
        captured = float(
            self.full_space_audit.get(
                "full_normalized_captured_energy_fraction_one_minus_residual",
                float("nan"),
            )
        )
        target = float(self.full_space_audit.get("target", float("nan")))
        if (
            not math.isfinite(captured)
            or not math.isfinite(target)
            or captured + 1.0e-12 < target
            or self.full_space_audit.get("passed") is not True
        ):
            raise ArtifactValidationError("DMD full-space energy gate is not satisfied")

    def validate(self) -> Dict[str, Any]:
        return {
            "valid": True,
            "method": "DMD physical groups",
            "n_points": int(self.coordinates.shape[0]),
            "grid_shape_zyx": list(self.shape_zyx),
            "n_frames": self.frame_count,
            "n_physical_modes": self.mode_count,
            "n_real_quadratures": self.component_count,
            "dt_s": self.dt_s,
            "relative_time_end_s": self.relative_time_end_s,
            "bounds_xyz_m": self.bounds_xyz_m.tolist(),
            "mode_order": "ascending temporal-coefficient PSD-centroid frequency",
            "reconstruction": (
                "mean + joint sum of all real quadratures belonging to each selected "
                "physical DMD group"
            ),
            "spatial_bounds_policy": "hard-fail; no clipping or extrapolation",
            "temporal_bounds_policy": "hard-fail; no extrapolation",
        }

    def _component_indices(self, group_indices: np.ndarray) -> np.ndarray:
        if group_indices.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [
                np.arange(
                    int(self.group_component_start[index]),
                    int(self.group_component_stop[index]),
                    dtype=np.int64,
                )
                for index in group_indices
            ]
        )

    def _coefficients_at(
        self, relative_time_s: float, component_indices: np.ndarray
    ) -> np.ndarray:
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
        position = time_value / self.dt_s
        lower = int(math.floor(position))
        if lower >= self.frame_count - 1:
            values = self.coefficients[-1]
        else:
            fraction = position - lower
            values = (
                self.coefficients[lower] * (1.0 - fraction)
                + self.coefficients[lower + 1] * fraction
            )
        # DMD quadrature bases were explicitly centred before full-space
        # regression.  The persisted reconstruction contract therefore uses
        # these coefficients directly, without a second mean subtraction.
        return np.asarray(values[component_indices], dtype=np.float64)

    def velocity_at(
        self,
        relative_time_s: float,
        points_xyz: np.ndarray,
        *,
        retained_mode_count: Optional[int] = None,
        mode_indices: Optional[Iterable[int]] = None,
        return_mode_contributions: bool = False,
    ) -> Union[np.ndarray, DMDFlowSample]:
        """Interpolate the DMD velocity while selecting whole physical groups."""

        # Explicit indices define single/leave-one-out physical-group subsets.
        # The retained count remains an independently checked invariant.
        if retained_mode_count is not None and mode_indices is not None:
            explicit = list(mode_indices)
            if int(retained_mode_count) != len(explicit):
                raise ValueError(
                    "retained_mode_count disagrees with explicit DMD group indices"
                )
            groups = self._select_modes(None, explicit)
        else:
            groups = self._select_modes(retained_mode_count, mode_indices)
        components = self._component_indices(groups)
        coefficients = self._coefficients_at(relative_time_s, components)
        flat, batch_shape, single = _normalise_points(points_xyz)
        self._check_spatial_bounds(flat)
        corners = self._corner_rows_and_weights(flat)
        mean_flat = self._weighted_rows(self.mean_field, corners)
        if components.size:
            modes_flat = self._weighted_rows(self.spatial_modes, corners, components)
            velocity_flat = mean_flat + np.tensordot(
                modes_flat, coefficients, axes=([-1], [0])
            )
        else:
            modes_flat = np.empty((flat.shape[0], 3, 0), dtype=np.float64)
            velocity_flat = mean_flat.copy()

        if single:
            velocity: np.ndarray = velocity_flat[0]
            mean: np.ndarray = mean_flat[0]
        else:
            velocity = velocity_flat.reshape(batch_shape + (3,))
            mean = mean_flat.reshape(batch_shape + (3,))
        if not return_mode_contributions:
            return velocity

        contributions = []
        cursor = 0
        for group in groups:
            width = int(
                self.group_component_stop[group] - self.group_component_start[group]
            )
            contribution = np.tensordot(
                modes_flat[..., cursor : cursor + width],
                coefficients[cursor : cursor + width],
                axes=([-1], [0]),
            )
            contributions.append(contribution)
            cursor += width
        contribution_flat = (
            np.stack(contributions, axis=-2)
            if contributions
            else np.empty((flat.shape[0], 0, 3), dtype=np.float64)
        )
        if single:
            contribution_value = contribution_flat[0]
        else:
            contribution_value = contribution_flat.reshape(batch_shape + (groups.size, 3))
        return DMDFlowSample(
            velocity=velocity,
            mean_velocity=mean,
            mode_contributions=contribution_value,
            mode_indices=groups.copy(),
            quadrature_indices=components.copy(),
            quadrature_coefficients=coefficients.copy(),
        )

    def interpolate_time(self, relative_time_s: float, points_xyz: np.ndarray, **kwargs: Any) -> Any:
        return self.velocity_at(relative_time_s, points_xyz, **kwargs)

    def interpolate_frame(self, frame_index: int, points_xyz: np.ndarray, **kwargs: Any) -> Any:
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
            raise TypeError("frame_index must be an integer")
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError("frame_index is outside the stored DMD time axis")
        return self.velocity_at(index * self.dt_s, points_xyz, **kwargs)


__all__ = [
    "DMD3DFlow",
    "DMDFlowSample",
    "ArtifactValidationError",
    "SpatialOutOfBoundsError",
    "TimeOutOfBoundsError",
]
