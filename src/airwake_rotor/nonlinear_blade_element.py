#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Auditable nonlinear blade-element loads for the NATO-GD coupling.

The implementation is independent of the CFD reader, RVMD solver and rigid-
body integrator.  All vectors are resolved in the world frame.  The adopted
sign convention is explicit:

* ``+n`` is positive rotor thrust;
* positive induced speed is an air velocity through the disk along ``-n``;
* positive rotor rotation is along each supplied tangential unit vector;
* positive shaft torque is the torque that the shaft must supply, hence
  ``Q = -M_hub dot n`` and the signed identity is ``P_shaft = Q * Omega``.

The CFD velocity is the velocity of the undisturbed/resolved air.  At every
blade element the air and blade velocities used by the section model are

``V_air = V_CFD - v_i*n``

``V_blade = V_translation + omega_body x (x-CG) + V_rotor_tangential``.

No tangential-speed floor and no non-negative power/thrust clamp are applied.
Angle of attack, Mach and Reynolds validity limits are checked without
clipping; the default action is to raise :class:`AerodynamicDomainError`.
The simple static ``C_l`` cap retained for compatibility with the original
project is an explicit model feature and every activation is reported.

The transverse distributed blade-root moment is deliberately *not* multiplied
by the former ``flapping_moment_transfer`` tuning coefficient.  Two transparent
CG moment closures are returned instead:

``force_line_only``
    ``(hub-CG) x F + projection_n(M_hub)``.  This is the primary articulated-
    rotor approximation: only the force line and shaft-axis torque reach the
    fuselage.

``rigid_root_bound``
    ``(hub-CG) x F + M_hub``.  This is a deliberately conservative sensitivity
    bound in which the complete distributed blade-root moment is transferred.

The model is quasi-steady and two-dimensional at each section.  Radial relative
flow is diagnosed but does not enter the section polar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


Number = Union[int, float, np.number]


class BladeElementConfigurationError(ValueError):
    """Raised when geometry or a model parameter is internally inconsistent."""


class AerodynamicDomainError(RuntimeError):
    """Raised when the requested state is outside the declared polar domain."""

    def __init__(self, message: str, diagnostics: "BladeElementDiagnostics") -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _positive(value: Number, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BladeElementConfigurationError("%s must be finite and positive" % name)
    return result


def _finite_scalar(value: Number, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise BladeElementConfigurationError("%s must be finite" % name)
    return result


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise BladeElementConfigurationError("%s must be a finite three-vector" % name)
    return result


def _unit_vector(value: Sequence[float], name: str) -> np.ndarray:
    result = _vector(value, name)
    length = float(np.linalg.norm(result))
    if abs(length - 1.0) > 1.0e-6:
        raise BladeElementConfigurationError(
            "%s must be unit length (received norm %.12g)" % (name, length)
        )
    return result / length


def _field_array(
    value: Union[Number, Sequence[float], np.ndarray],
    shape: Tuple[int, int],
    name: str,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    try:
        result = np.broadcast_to(array, shape)
    except ValueError:
        raise BladeElementConfigurationError(
            "%s must be scalar or broadcastable to %s; received %s"
            % (name, shape, array.shape)
        )
    if not np.all(np.isfinite(result)):
        raise BladeElementConfigurationError("%s contains NaN or infinity" % name)
    if positive and np.any(result <= 0.0):
        raise BladeElementConfigurationError("%s must be positive everywhere" % name)
    return np.asarray(result, dtype=float)


def _element_vectors(value: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (shape[0], 3):
        array = array[:, None, :]
    expected = shape + (3,)
    try:
        result = np.broadcast_to(array, expected)
    except ValueError:
        raise BladeElementConfigurationError(
            "%s must be broadcastable to %s; received %s"
            % (name, expected, array.shape)
        )
    if not np.all(np.isfinite(result)):
        raise BladeElementConfigurationError("%s contains NaN or infinity" % name)
    return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class AerodynamicDomainLimits:
    """Declared validity envelope; values are diagnosed, never clipped.

    ``action='raise'`` is the publication/default setting.  ``'report'`` is
    useful for a deliberate sensitivity run: the returned diagnostic masks
    then identify every extrapolated element.  Reverse flow has a separate
    switch because it is a kinematic regime rather than a scalar polar bound.
    """

    alpha_min_deg: Optional[float] = -25.0
    alpha_max_deg: Optional[float] = 25.0
    mach_max: Optional[float] = 0.85
    reynolds_min: Optional[float] = None
    reynolds_max: Optional[float] = None
    action: str = "raise"
    reverse_flow_action: str = "raise"

    def __post_init__(self) -> None:
        if self.action not in ("raise", "report"):
            raise BladeElementConfigurationError("domain action must be 'raise' or 'report'")
        if self.reverse_flow_action not in ("raise", "report"):
            raise BladeElementConfigurationError(
                "reverse_flow_action must be 'raise' or 'report'"
            )
        for name in (
            "alpha_min_deg",
            "alpha_max_deg",
            "mach_max",
            "reynolds_min",
            "reynolds_max",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise BladeElementConfigurationError("%s must be finite or None" % name)
        if (
            self.alpha_min_deg is not None
            and self.alpha_max_deg is not None
            and float(self.alpha_min_deg) >= float(self.alpha_max_deg)
        ):
            raise BladeElementConfigurationError("alpha_min_deg must be below alpha_max_deg")
        if self.mach_max is not None and float(self.mach_max) <= 0.0:
            raise BladeElementConfigurationError("mach_max must be positive")
        if self.reynolds_min is not None and float(self.reynolds_min) < 0.0:
            raise BladeElementConfigurationError("reynolds_min cannot be negative")
        if self.reynolds_max is not None and float(self.reynolds_max) <= 0.0:
            raise BladeElementConfigurationError("reynolds_max must be positive")
        if (
            self.reynolds_min is not None
            and self.reynolds_max is not None
            and float(self.reynolds_min) >= float(self.reynolds_max)
        ):
            raise BladeElementConfigurationError(
                "reynolds_min must be below reynolds_max"
            )


@dataclass(frozen=True)
class BladeElementParameters:
    """Rotor/section parameters, compatible with entries in ``rotor_config.json``."""

    blade_count: int
    radius_m: float
    root_cutout_fraction: float
    chord_m: float
    omega_radps: float
    collective_deg: float
    twist_deg: float
    cd0: float
    cl_alpha_per_rad: float
    cl_max: float
    transmission_efficiency: float = 1.0
    induced_drag_factor: float = 0.012
    cyclic_cos_deg: float = 0.0
    cyclic_sin_deg: float = 0.0
    prandtl_tip_loss: bool = True
    prandtl_root_loss: bool = True
    speed_of_sound_mps: float = 340.294
    dynamic_viscosity_Pas: float = 1.7894e-5
    domain_limits: AerodynamicDomainLimits = field(
        default_factory=AerodynamicDomainLimits
    )

    def __post_init__(self) -> None:
        if int(self.blade_count) != self.blade_count or int(self.blade_count) < 1:
            raise BladeElementConfigurationError("blade_count must be a positive integer")
        object.__setattr__(self, "blade_count", int(self.blade_count))
        for name in (
            "radius_m",
            "chord_m",
            "omega_radps",
            "cl_alpha_per_rad",
            "cl_max",
            "transmission_efficiency",
            "speed_of_sound_mps",
            "dynamic_viscosity_Pas",
        ):
            _positive(getattr(self, name), name)
        root = float(self.root_cutout_fraction)
        if not math.isfinite(root) or not 0.0 <= root < 1.0:
            raise BladeElementConfigurationError(
                "root_cutout_fraction must lie in [0, 1)"
            )
        for name in (
            "collective_deg",
            "twist_deg",
            "cd0",
            "induced_drag_factor",
            "cyclic_cos_deg",
            "cyclic_sin_deg",
        ):
            _finite_scalar(getattr(self, name), name)
        if self.cd0 < 0.0 or self.induced_drag_factor < 0.0:
            raise BladeElementConfigurationError(
                "cd0 and induced_drag_factor cannot be negative"
            )
        if not 0.0 < float(self.transmission_efficiency) <= 1.0:
            raise BladeElementConfigurationError(
                "transmission_efficiency must lie in (0, 1]"
            )
        if not isinstance(self.domain_limits, AerodynamicDomainLimits):
            raise BladeElementConfigurationError(
                "domain_limits must be an AerodynamicDomainLimits instance"
            )

    @classmethod
    def from_mapping(
        cls, rotor: Mapping[str, Any], **overrides: Any
    ) -> "BladeElementParameters":
        """Build from one existing project rotor mapping.

        The obsolete ``flapping_moment_transfer`` key is intentionally ignored;
        it has no pathway into the returned physical loads.
        """

        required = (
            "blades",
            "diameter_m",
            "root_cutout",
            "mean_chord_m",
            "rpm",
            "collective_deg",
            "twist_deg",
            "cd0",
            "cl_alpha",
            "cl_max",
        )
        missing = [name for name in required if name not in rotor]
        if missing:
            raise BladeElementConfigurationError(
                "rotor mapping is missing: %s" % ", ".join(missing)
            )
        values: Dict[str, Any] = {
            "blade_count": rotor["blades"],
            "radius_m": float(rotor["diameter_m"]) / 2.0,
            "root_cutout_fraction": rotor["root_cutout"],
            "chord_m": rotor["mean_chord_m"],
            "omega_radps": float(rotor["rpm"]) * 2.0 * math.pi / 60.0,
            "collective_deg": rotor["collective_deg"],
            "twist_deg": rotor["twist_deg"],
            "cd0": rotor["cd0"],
            "cl_alpha_per_rad": rotor["cl_alpha"],
            "cl_max": rotor["cl_max"],
            "transmission_efficiency": rotor.get("transmission_efficiency", 1.0),
            "induced_drag_factor": rotor.get("induced_drag_factor", 0.012),
            "cyclic_cos_deg": rotor.get("cyclic_cos_deg", 0.0),
            "cyclic_sin_deg": rotor.get("cyclic_sin_deg", 0.0),
            "prandtl_tip_loss": rotor.get("prandtl_tip_loss", True),
            "prandtl_root_loss": rotor.get("prandtl_root_loss", True),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class BladeElementDiagnostics:
    """Per-call ranges, counts and masks used in result QA."""

    element_count: int
    alpha_min_deg: float
    alpha_max_deg: float
    cl_raw_min: float
    cl_raw_max: float
    cl_effective_min: float
    cl_effective_max: float
    mach_min: float
    mach_max: float
    reynolds_min: float
    reynolds_max: float
    radial_relative_speed_max_mps: float
    induced_velocity_min_mps: float
    induced_velocity_max_mps: float
    prandtl_factor_min: float
    prandtl_factor_max: float
    reverse_flow_count: int
    cl_saturation_count: int
    alpha_domain_violation_count: int
    mach_domain_violation_count: int
    reynolds_domain_violation_count: int
    negative_thrust: bool
    negative_shaft_torque: bool
    negative_shaft_power: bool
    reverse_flow_mask: np.ndarray = field(repr=False)
    cl_saturation_mask: np.ndarray = field(repr=False)
    alpha_domain_violation_mask: np.ndarray = field(repr=False)
    mach_domain_violation_mask: np.ndarray = field(repr=False)
    reynolds_domain_violation_mask: np.ndarray = field(repr=False)

    @property
    def extrapolated_element_count(self) -> int:
        combined = (
            self.alpha_domain_violation_mask
            | self.mach_domain_violation_mask
            | self.reynolds_domain_violation_mask
        )
        return int(np.count_nonzero(combined))

    def as_dict(self) -> Dict[str, Union[int, float, bool]]:
        """Return scalar diagnostics suitable for CSV/JSON logging."""

        return {
            "element_count": self.element_count,
            "alpha_min_deg": self.alpha_min_deg,
            "alpha_max_deg": self.alpha_max_deg,
            "cl_raw_min": self.cl_raw_min,
            "cl_raw_max": self.cl_raw_max,
            "cl_effective_min": self.cl_effective_min,
            "cl_effective_max": self.cl_effective_max,
            "mach_min": self.mach_min,
            "mach_max": self.mach_max,
            "reynolds_min": self.reynolds_min,
            "reynolds_max": self.reynolds_max,
            "radial_relative_speed_max_mps": self.radial_relative_speed_max_mps,
            "induced_velocity_min_mps": self.induced_velocity_min_mps,
            "induced_velocity_max_mps": self.induced_velocity_max_mps,
            "prandtl_factor_min": self.prandtl_factor_min,
            "prandtl_factor_max": self.prandtl_factor_max,
            "reverse_flow_count": self.reverse_flow_count,
            "cl_saturation_count": self.cl_saturation_count,
            "alpha_domain_violation_count": self.alpha_domain_violation_count,
            "mach_domain_violation_count": self.mach_domain_violation_count,
            "reynolds_domain_violation_count": self.reynolds_domain_violation_count,
            "extrapolated_element_count": self.extrapolated_element_count,
            "negative_thrust": self.negative_thrust,
            "negative_shaft_torque": self.negative_shaft_torque,
            "negative_shaft_power": self.negative_shaft_power,
        }


@dataclass(frozen=True)
class BladeElementLoads:
    """Integrated loads plus the element fields needed for an audit."""

    hub_force_world_N: np.ndarray
    hub_aerodynamic_moment_world_Nm: np.ndarray
    hub_axis_moment_world_Nm: np.ndarray
    cg_moment_force_line_only_world_Nm: np.ndarray
    cg_moment_rigid_root_bound_world_Nm: np.ndarray
    thrust_N: float
    shaft_torque_Nm: float
    shaft_power_W: float
    engine_power_W: float
    element_force_world_N: np.ndarray
    angle_of_attack_rad: np.ndarray
    lift_coefficient_raw: np.ndarray
    lift_coefficient_effective: np.ndarray
    drag_coefficient: np.ndarray
    mach: np.ndarray
    reynolds: np.ndarray
    prandtl_factor: np.ndarray
    diagnostics: BladeElementDiagnostics

    def cg_moment(self, closure: str = "force_line_only") -> np.ndarray:
        """Return one named, explicit hub-to-CG moment closure."""

        if closure == "force_line_only":
            return self.cg_moment_force_line_only_world_Nm.copy()
        if closure == "rigid_root_bound":
            return self.cg_moment_rigid_root_bound_world_Nm.copy()
        raise BladeElementConfigurationError(
            "closure must be 'force_line_only' or 'rigid_root_bound'"
        )


def radial_element_centres_and_widths(
    parameters: BladeElementParameters, radial_elements: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return midpoint radii and widths matching the project's legacy mesh."""

    if int(radial_elements) != radial_elements or int(radial_elements) < 1:
        raise BladeElementConfigurationError("radial_elements must be a positive integer")
    root = parameters.root_cutout_fraction * parameters.radius_m
    edges = np.linspace(root, parameters.radius_m, int(radial_elements) + 1)
    return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)


def element_relative_velocity_world(
    element_positions_world_m: np.ndarray,
    cfd_velocity_world_mps: np.ndarray,
    cg_world_m: Sequence[float],
    body_translation_velocity_world_mps: Sequence[float],
    body_angular_velocity_world_radps: Sequence[float],
    rotor_tangential_velocity_world_mps: np.ndarray,
    rotor_axis_world: Sequence[float],
    induced_velocity_mps: Union[Number, Sequence[float], np.ndarray],
) -> np.ndarray:
    """Return blade-relative-air velocity at each element.

    This helper exposes the most important coupling sign convention for direct
    verification.  Inputs have shape ``(blade, radial, 3)`` except scalar or
    broadcastable local induced speed.
    """

    positions = np.asarray(element_positions_world_m, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise BladeElementConfigurationError(
            "element_positions_world_m must have shape (blade, radial, 3)"
        )
    if not np.all(np.isfinite(positions)):
        raise BladeElementConfigurationError(
            "element_positions_world_m contains NaN or infinity"
        )
    shape = positions.shape[:2]
    cfd = _element_vectors(cfd_velocity_world_mps, shape, "cfd_velocity_world_mps")
    rotor_velocity = _element_vectors(
        rotor_tangential_velocity_world_mps,
        shape,
        "rotor_tangential_velocity_world_mps",
    )
    cg = _vector(cg_world_m, "cg_world_m")
    translation = _vector(
        body_translation_velocity_world_mps,
        "body_translation_velocity_world_mps",
    )
    angular = _vector(
        body_angular_velocity_world_radps,
        "body_angular_velocity_world_radps",
    )
    normal = _unit_vector(rotor_axis_world, "rotor_axis_world")
    induced = _field_array(induced_velocity_mps, shape, "induced_velocity_mps")
    arms = positions - cg[None, None, :]
    angular_velocity = np.cross(
        np.broadcast_to(angular, positions.shape), arms
    )
    blade_velocity = translation[None, None, :] + angular_velocity + rotor_velocity
    air_velocity = cfd - induced[:, :, None] * normal[None, None, :]
    return blade_velocity - air_velocity


def _prandtl_loss_factor(
    parameters: BladeElementParameters,
    radii_m: np.ndarray,
    inflow_angle_rad: np.ndarray,
) -> np.ndarray:
    """Classical Prandtl root/tip factors on the supplied radial midpoints."""

    shape = inflow_angle_rad.shape
    radii = np.broadcast_to(radii_m[None, :], shape)
    sine = np.abs(np.sin(inflow_angle_rad))
    factor = np.ones(shape, dtype=float)
    # sin(phi)->0 is the analytic F->1 limit.  Treat it explicitly instead of
    # placing an undocumented floor in the denominator.
    active = sine > 1.0e-12
    blades_over_two = 0.5 * parameters.blade_count
    if parameters.prandtl_tip_loss:
        exponent = np.full(shape, np.inf, dtype=float)
        exponent[active] = (
            blades_over_two
            * (parameters.radius_m - radii[active])
            / (radii[active] * sine[active])
        )
        tip = (2.0 / math.pi) * np.arccos(np.exp(-exponent))
        factor *= tip
    if parameters.prandtl_root_loss and parameters.root_cutout_fraction > 0.0:
        root_radius = parameters.root_cutout_fraction * parameters.radius_m
        exponent = np.full(shape, np.inf, dtype=float)
        exponent[active] = (
            blades_over_two
            * (radii[active] - root_radius)
            / (root_radius * sine[active])
        )
        root = (2.0 / math.pi) * np.arccos(np.exp(-exponent))
        factor *= root
    return factor


def _domain_masks(
    alpha_deg: np.ndarray,
    mach: np.ndarray,
    reynolds: np.ndarray,
    limits: AerodynamicDomainLimits,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_bad = np.zeros(alpha_deg.shape, dtype=bool)
    mach_bad = np.zeros(alpha_deg.shape, dtype=bool)
    reynolds_bad = np.zeros(alpha_deg.shape, dtype=bool)
    if limits.alpha_min_deg is not None:
        alpha_bad |= alpha_deg < float(limits.alpha_min_deg)
    if limits.alpha_max_deg is not None:
        alpha_bad |= alpha_deg > float(limits.alpha_max_deg)
    if limits.mach_max is not None:
        mach_bad |= mach > float(limits.mach_max)
    if limits.reynolds_min is not None:
        reynolds_bad |= reynolds < float(limits.reynolds_min)
    if limits.reynolds_max is not None:
        reynolds_bad |= reynolds > float(limits.reynolds_max)
    return alpha_bad, mach_bad, reynolds_bad


def integrate_blade_element_loads(
    parameters: BladeElementParameters,
    element_positions_world_m: np.ndarray,
    cfd_velocity_world_mps: np.ndarray,
    density_kgpm3: Union[Number, Sequence[float], np.ndarray],
    radii_m: Sequence[float],
    element_widths_m: Sequence[float],
    blade_radial_direction_world: np.ndarray,
    blade_tangential_direction_world: np.ndarray,
    rotor_axis_world: Sequence[float],
    hub_world_m: Sequence[float],
    cg_world_m: Sequence[float],
    body_translation_velocity_world_mps: Sequence[float],
    body_angular_velocity_world_radps: Sequence[float],
    rotor_tangential_velocity_world_mps: np.ndarray,
    induced_velocity_mps: Union[Number, Sequence[float], np.ndarray],
    blade_azimuth_rad: Optional[Sequence[float]] = None,
) -> BladeElementLoads:
    """Integrate nonlinear quasi-steady section loads over all blades.

    Parameters containing element quantities use ``(blade, radial, 3)``.  A
    density or induced-speed scalar is broadcast to ``(blade, radial)``.
    ``blade_radial_direction_world`` and ``blade_tangential_direction_world``
    may be ``(blade, 3)`` or ``(blade, radial, 3)``.
    """

    positions = np.asarray(element_positions_world_m, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise BladeElementConfigurationError(
            "element_positions_world_m must have shape (blade, radial, 3)"
        )
    if not np.all(np.isfinite(positions)):
        raise BladeElementConfigurationError(
            "element_positions_world_m contains NaN or infinity"
        )
    blade_count, radial_count = positions.shape[:2]
    if blade_count != parameters.blade_count:
        raise BladeElementConfigurationError(
            "geometry has %d blades but parameters declare %d"
            % (blade_count, parameters.blade_count)
        )
    shape = (blade_count, radial_count)
    radii = np.asarray(radii_m, dtype=float)
    widths = np.asarray(element_widths_m, dtype=float)
    if radii.shape != (radial_count,) or not np.all(np.isfinite(radii)):
        raise BladeElementConfigurationError(
            "radii_m must be a finite vector with one value per radial element"
        )
    if widths.shape != radii.shape or not np.all(np.isfinite(widths)):
        raise BladeElementConfigurationError(
            "element_widths_m must be a finite vector matching radii_m"
        )
    if np.any(widths <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise BladeElementConfigurationError(
            "element widths must be positive and radii strictly increasing"
        )
    root_radius = parameters.root_cutout_fraction * parameters.radius_m
    tolerance = 1.0e-10 * max(parameters.radius_m, 1.0)
    if np.any(radii - 0.5 * widths < root_radius - tolerance) or np.any(
        radii + 0.5 * widths > parameters.radius_m + tolerance
    ):
        raise BladeElementConfigurationError(
            "radial elements extend outside the declared root/tip boundaries"
        )

    normal = _unit_vector(rotor_axis_world, "rotor_axis_world")
    hub = _vector(hub_world_m, "hub_world_m")
    cg = _vector(cg_world_m, "cg_world_m")
    density = _field_array(density_kgpm3, shape, "density_kgpm3", positive=True)
    induced = _field_array(induced_velocity_mps, shape, "induced_velocity_mps")
    radial_direction = _element_vectors(
        blade_radial_direction_world,
        shape,
        "blade_radial_direction_world",
    )
    tangential_direction = _element_vectors(
        blade_tangential_direction_world,
        shape,
        "blade_tangential_direction_world",
    )
    radial_norm = np.linalg.norm(radial_direction, axis=2)
    tangential_norm = np.linalg.norm(tangential_direction, axis=2)
    if np.max(np.abs(radial_norm - 1.0)) > 1.0e-6 or np.max(
        np.abs(tangential_norm - 1.0)
    ) > 1.0e-6:
        raise BladeElementConfigurationError(
            "radial and tangential directions must be unit vectors"
        )
    radial_direction = radial_direction / radial_norm[:, :, None]
    tangential_direction = tangential_direction / tangential_norm[:, :, None]
    if (
        np.max(np.abs(np.sum(radial_direction * tangential_direction, axis=2)))
        > 1.0e-6
        or np.max(np.abs(np.sum(radial_direction * normal, axis=2))) > 1.0e-6
        or np.max(np.abs(np.sum(tangential_direction * normal, axis=2))) > 1.0e-6
    ):
        raise BladeElementConfigurationError(
            "radial, tangential and rotor-axis directions must be orthogonal"
        )
    handed = np.cross(radial_direction, tangential_direction)
    if np.min(np.sum(handed * normal, axis=2)) < 1.0 - 1.0e-6:
        raise BladeElementConfigurationError(
            "directions must obey radial x tangential = rotor axis"
        )

    geometric_arms = positions - hub[None, None, :]
    expected_arms = radii[None, :, None] * radial_direction
    geometry_scale = max(parameters.radius_m, 1.0)
    if np.max(np.abs(geometric_arms - expected_arms)) > 1.0e-7 * geometry_scale:
        raise BladeElementConfigurationError(
            "element positions are inconsistent with hub, radii and radial directions"
        )
    rotor_velocity = _element_vectors(
        rotor_tangential_velocity_world_mps,
        shape,
        "rotor_tangential_velocity_world_mps",
    )
    expected_rotor_velocity = (
        parameters.omega_radps
        * radii[None, :, None]
        * tangential_direction
    )
    velocity_scale = max(parameters.omega_radps * parameters.radius_m, 1.0)
    if np.max(np.abs(rotor_velocity - expected_rotor_velocity)) > 1.0e-7 * velocity_scale:
        raise BladeElementConfigurationError(
            "rotor tangential velocity is inconsistent with Omega*r*e_t"
        )

    relative = element_relative_velocity_world(
        positions,
        cfd_velocity_world_mps,
        cg,
        body_translation_velocity_world_mps,
        body_angular_velocity_world_radps,
        rotor_velocity,
        normal,
        induced,
    )
    tangential_speed = np.sum(relative * tangential_direction, axis=2)
    normal_speed = np.sum(relative * normal[None, None, :], axis=2)
    radial_speed = np.sum(relative * radial_direction, axis=2)
    section_speed = np.hypot(tangential_speed, normal_speed)
    inflow_angle = np.arctan2(normal_speed, tangential_speed)

    if blade_azimuth_rad is None:
        if parameters.cyclic_cos_deg != 0.0 or parameters.cyclic_sin_deg != 0.0:
            raise BladeElementConfigurationError(
                "blade_azimuth_rad is required when cyclic pitch is nonzero"
            )
        azimuth = np.zeros(blade_count, dtype=float)
    else:
        azimuth = np.asarray(blade_azimuth_rad, dtype=float)
        if azimuth.shape != (blade_count,) or not np.all(np.isfinite(azimuth)):
            raise BladeElementConfigurationError(
                "blade_azimuth_rad must contain one finite value per blade"
            )
    span_fraction = (radii - root_radius) / (parameters.radius_m - root_radius)
    pitch_deg = (
        parameters.collective_deg
        + parameters.twist_deg * span_fraction[None, :]
        + parameters.cyclic_cos_deg * np.cos(azimuth)[:, None]
        + parameters.cyclic_sin_deg * np.sin(azimuth)[:, None]
    )
    alpha = np.deg2rad(pitch_deg) - inflow_angle
    alpha_deg = np.rad2deg(alpha)
    cl_raw = parameters.cl_alpha_per_rad * alpha
    cl_saturation = np.abs(cl_raw) > parameters.cl_max
    # This is the declared static-stall model, not an undisclosed domain clamp.
    cl_section = np.clip(cl_raw, -parameters.cl_max, parameters.cl_max)

    loss_factor = _prandtl_loss_factor(parameters, radii, inflow_angle)
    # Apply the finite-blade correction to circulatory loading.  Profile drag
    # cd0 is retained at the physical blade tip/root rather than being tapered
    # artificially to zero.
    cl_effective = loss_factor * cl_section
    cd = parameters.cd0 + parameters.induced_drag_factor * cl_effective ** 2
    mach = section_speed / parameters.speed_of_sound_mps
    reynolds = (
        density * section_speed * parameters.chord_m
        / parameters.dynamic_viscosity_Pas
    )
    reverse = tangential_speed <= 0.0
    alpha_bad, mach_bad, reynolds_bad = _domain_masks(
        alpha_deg, mach, reynolds, parameters.domain_limits
    )

    dynamic_pressure = 0.5 * density * section_speed ** 2
    area_strip = parameters.chord_m * widths[None, :]
    lift = dynamic_pressure * area_strip * cl_effective
    drag = dynamic_pressure * area_strip * cd
    cosine = np.cos(inflow_angle)
    sine = np.sin(inflow_angle)
    normal_force = lift * cosine - drag * sine
    tangential_resisting_force = lift * sine + drag * cosine
    element_force = (
        normal_force[:, :, None] * normal[None, None, :]
        - tangential_resisting_force[:, :, None] * tangential_direction
    )
    hub_force = np.sum(element_force, axis=(0, 1))
    hub_moment = np.sum(
        np.cross(positions - hub[None, None, :], element_force), axis=(0, 1)
    )
    axis_moment = float(np.dot(hub_moment, normal)) * normal
    force_arm_moment = np.cross(hub - cg, hub_force)
    force_line_only = force_arm_moment + axis_moment
    rigid_root_bound = force_arm_moment + hub_moment
    thrust = float(np.dot(hub_force, normal))
    shaft_torque = -float(np.dot(hub_moment, normal))
    shaft_power = shaft_torque * parameters.omega_radps
    engine_power = shaft_power / parameters.transmission_efficiency

    diagnostics = BladeElementDiagnostics(
        element_count=blade_count * radial_count,
        alpha_min_deg=float(np.min(alpha_deg)),
        alpha_max_deg=float(np.max(alpha_deg)),
        cl_raw_min=float(np.min(cl_raw)),
        cl_raw_max=float(np.max(cl_raw)),
        cl_effective_min=float(np.min(cl_effective)),
        cl_effective_max=float(np.max(cl_effective)),
        mach_min=float(np.min(mach)),
        mach_max=float(np.max(mach)),
        reynolds_min=float(np.min(reynolds)),
        reynolds_max=float(np.max(reynolds)),
        radial_relative_speed_max_mps=float(np.max(np.abs(radial_speed))),
        induced_velocity_min_mps=float(np.min(induced)),
        induced_velocity_max_mps=float(np.max(induced)),
        prandtl_factor_min=float(np.min(loss_factor)),
        prandtl_factor_max=float(np.max(loss_factor)),
        reverse_flow_count=int(np.count_nonzero(reverse)),
        cl_saturation_count=int(np.count_nonzero(cl_saturation)),
        alpha_domain_violation_count=int(np.count_nonzero(alpha_bad)),
        mach_domain_violation_count=int(np.count_nonzero(mach_bad)),
        reynolds_domain_violation_count=int(np.count_nonzero(reynolds_bad)),
        negative_thrust=bool(thrust < 0.0),
        negative_shaft_torque=bool(shaft_torque < 0.0),
        negative_shaft_power=bool(shaft_power < 0.0),
        reverse_flow_mask=reverse.copy(),
        cl_saturation_mask=cl_saturation.copy(),
        alpha_domain_violation_mask=alpha_bad.copy(),
        mach_domain_violation_mask=mach_bad.copy(),
        reynolds_domain_violation_mask=reynolds_bad.copy(),
    )

    failures = []
    if diagnostics.alpha_domain_violation_count:
        failures.append("alpha=%d" % diagnostics.alpha_domain_violation_count)
    if diagnostics.mach_domain_violation_count:
        failures.append("Mach=%d" % diagnostics.mach_domain_violation_count)
    if diagnostics.reynolds_domain_violation_count:
        failures.append("Re=%d" % diagnostics.reynolds_domain_violation_count)
    if failures and parameters.domain_limits.action == "raise":
        raise AerodynamicDomainError(
            "blade elements left the declared polar domain (%s)" % ", ".join(failures),
            diagnostics,
        )
    if diagnostics.reverse_flow_count and parameters.domain_limits.reverse_flow_action == "raise":
        raise AerodynamicDomainError(
            "reverse-flow elements detected (%d)" % diagnostics.reverse_flow_count,
            diagnostics,
        )

    return BladeElementLoads(
        hub_force_world_N=hub_force,
        hub_aerodynamic_moment_world_Nm=hub_moment,
        hub_axis_moment_world_Nm=axis_moment,
        cg_moment_force_line_only_world_Nm=force_line_only,
        cg_moment_rigid_root_bound_world_Nm=rigid_root_bound,
        thrust_N=thrust,
        shaft_torque_Nm=shaft_torque,
        shaft_power_W=shaft_power,
        engine_power_W=engine_power,
        element_force_world_N=element_force,
        angle_of_attack_rad=alpha,
        lift_coefficient_raw=cl_raw,
        lift_coefficient_effective=cl_effective,
        drag_coefficient=cd,
        mach=mach,
        reynolds=reynolds,
        prandtl_factor=loss_factor,
        diagnostics=diagnostics,
    )


# A short alias for callers that treat this module as a solver.
solve_blade_elements = integrate_blade_element_loads


__all__ = [
    "AerodynamicDomainError",
    "AerodynamicDomainLimits",
    "BladeElementConfigurationError",
    "BladeElementDiagnostics",
    "BladeElementLoads",
    "BladeElementParameters",
    "element_relative_velocity_world",
    "integrate_blade_element_loads",
    "radial_element_centres_and_widths",
    "solve_blade_elements",
]
