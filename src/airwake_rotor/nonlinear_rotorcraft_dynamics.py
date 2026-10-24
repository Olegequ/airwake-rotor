#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Nonlinear dynamic-inflow and attitude/heave building blocks.

The module is deliberately independent of CFD/RVMD input and output.  It uses
one explicit set of conventions throughout:

* world axes are right handed and ``+z`` is upward;
* body axes are right handed and coincide with world axes at zero attitude;
* the rotor disk basis ``(e1, e2, n)`` is right handed;
* positive rotor thrust is along ``+n``;
* positive induced velocity passes through the disk along ``-n``;
* the axial external-flow perturbation is
  ``delta_mu_z=-mean(delta_u dot n)/(Omega*R)``, hence positive
  ``delta_mu_z`` is also along the downwash direction ``-n``;
* dynamic-inflow states are perturbations ordered
  ``[delta_lambda_0, delta_lambda_c, delta_lambda_s]`` and total inflow is
  ``lambda_star + delta_lambda_0 + r/R*(delta_lambda_c*cos(psi) +
  delta_lambda_s*sin(psi))``;
* aerodynamic inflow forcing is ``[C_T, -C_My, +C_Mx]``.  Moments used here
  are aerodynamic moments about the hub, resolved in disk axes.

The implemented inflow equation is deliberately the auditable *near-hover,
frozen-coefficient perturbation* form of the Pitt--Peters finite-state model.
It does not claim to be a nonlinear, skew-wake, all-flight-envelope inflow
model.  The state derivative is with respect to physical time, so the
nondimensional equation is multiplied by rotor speed ``Omega``.  Nonlinearity
in a coupled calculation comes from blade-element aerodynamics, moving/tilted
sampling geometry, load arms and attitude projection--not from this linear
local inflow equation.  Leaving its configured near-hover domain raises
``DynamicsDomainError`` rather than silently changing equations.

References
----------
D. M. Pitt and D. A. Peters, "Theoretical prediction of dynamic-inflow
derivatives," Sixth European Rotorcraft and Powered Lift Aircraft Forum, 1980.
D. A. Peters and N. HaQuang, "Dynamic inflow for practical applications,"
Journal of the American Helicopter Society 33(4), 64--68, 1988.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


INFLOW_STATE_ORDER = ("delta_lambda_0", "delta_lambda_c", "delta_lambda_s")
BODY_STATE_ORDER = (
    "heave_z_m",
    "heave_velocity_mps",
    "roll_rad",
    "pitch_rad",
    "roll_rate_radps",
    "pitch_rate_radps",
)


class DynamicsConfigurationError(ValueError):
    """Raised when a physical/configuration parameter is missing or invalid."""


class DynamicsDomainError(RuntimeError):
    """Raised when a state leaves the declared validity domain of the model."""


def _finite_array(value: Sequence[float], shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise DynamicsConfigurationError(
            "%s must have shape %s; received %s" % (name, shape, array.shape)
        )
    if not np.all(np.isfinite(array)):
        raise DynamicsConfigurationError("%s contains NaN or infinity" % name)
    return array


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise DynamicsConfigurationError("%s must be finite and positive" % name)
    return value


def unit_vector(vector: Sequence[float], name: str = "vector") -> np.ndarray:
    """Return a normalized three-vector, failing on a degenerate direction."""

    vector_array = _finite_array(vector, (3,), name)
    norm = float(np.linalg.norm(vector_array))
    if norm <= 1.0e-12:
        raise DynamicsConfigurationError("%s has zero length" % name)
    return vector_array / norm


def disk_basis(axis_body: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a deterministic right-handed disk basis in body axes."""

    normal = unit_vector(axis_body, "rotor_axis_body")
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(normal[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    e1 = reference - normal * float(np.dot(reference, normal))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return normal, e1, e2


def resolve_vector_in_basis(
    vector: Sequence[float],
    e1: Sequence[float],
    e2: Sequence[float],
    normal: Sequence[float],
) -> np.ndarray:
    """Resolve a vector into an explicitly supplied right-handed disk basis."""

    value = _finite_array(vector, (3,), "vector")
    basis = np.column_stack(
        [
            unit_vector(e1, "e1"),
            unit_vector(e2, "e2"),
            unit_vector(normal, "normal"),
        ]
    )
    if not np.allclose(np.dot(basis.T, basis), np.eye(3), atol=1.0e-10, rtol=0.0):
        raise DynamicsConfigurationError("supplied basis must be orthonormal")
    if np.linalg.det(basis) <= 0.0:
        raise DynamicsConfigurationError("supplied basis must be right handed")
    return np.dot(basis.T, value)


def assemble_vector_from_basis(
    components: Sequence[float],
    e1: Sequence[float],
    e2: Sequence[float],
    normal: Sequence[float],
) -> np.ndarray:
    """Assemble a vector from components in a right-handed disk basis."""

    values = _finite_array(components, (3,), "components")
    # Reuse the strict basis checks performed by resolve_vector_in_basis.
    basis_vectors = [
        unit_vector(e1, "e1"), unit_vector(e2, "e2"), unit_vector(normal, "normal")
    ]
    basis = np.column_stack(basis_vectors)
    resolve_vector_in_basis(np.zeros(3), *basis_vectors)
    return np.dot(basis, values)


def rotation_body_to_world(roll_rad: float, pitch_rad: float) -> np.ndarray:
    """Body-to-world matrix for the yaw-constrained ZYX attitude ``(0,theta,phi)``."""

    phi = float(roll_rad)
    theta = float(pitch_rad)
    if not math.isfinite(phi) or not math.isfinite(theta):
        raise DynamicsDomainError("roll and pitch must be finite")
    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    # R_wb = R_y(theta) R_x(phi); yaw is constrained to zero.
    return np.array(
        [
            [ct, st * sp, st * cp],
            [0.0, cp, -sp],
            [-st, ct * sp, ct * cp],
        ],
        dtype=float,
    )


def euler_rate_projection(roll_rad: float) -> np.ndarray:
    """Map constrained generalized rates ``[phi_dot, theta_dot]`` to body omega."""

    phi = float(roll_rad)
    return np.array(
        [[1.0, 0.0], [0.0, math.cos(phi)], [0.0, -math.sin(phi)]], dtype=float
    )


def body_angular_velocity(
    roll_rad: float, roll_rate_radps: float, pitch_rate_radps: float
) -> np.ndarray:
    """Return body angular velocity for zero-yaw constrained attitude motion."""

    rates = np.array([roll_rate_radps, pitch_rate_radps], dtype=float)
    if not np.all(np.isfinite(rates)):
        raise DynamicsDomainError("attitude rates contain NaN or infinity")
    return np.dot(euler_rate_projection(roll_rad), rates)


@dataclass
class DynamicsDiagnostics:
    """Mutable counters/extrema that make numerical-domain events auditable."""

    inflow_rhs_calls: int = 0
    rigid_body_rhs_calls: int = 0
    hard_failures: int = 0
    maximum_abs_delta_inflow_ratio: float = 0.0
    maximum_abs_delta_mu_z: float = 0.0
    maximum_abs_attitude_deg: float = 0.0

    def as_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "inflow_rhs_calls": self.inflow_rhs_calls,
            "rigid_body_rhs_calls": self.rigid_body_rhs_calls,
            "hard_failures": self.hard_failures,
            "maximum_abs_delta_inflow_ratio": self.maximum_abs_delta_inflow_ratio,
            "maximum_abs_delta_mu_z": self.maximum_abs_delta_mu_z,
            "maximum_abs_attitude_deg": self.maximum_abs_attitude_deg,
        }


@dataclass(frozen=True)
class DynamicInflowParameters:
    """Settings and frozen reference for near-hover Pitt--Peters inflow."""

    radius_m: float
    omega_radps: float
    density_kgpm3: float
    reference_thrust_N: float
    reference_hub_aero_moment_disk_Nm: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    maximum_delta_lambda_fraction: float = 0.75
    maximum_delta_mu_z_fraction: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.radius_m, "radius_m")
        _positive(self.omega_radps, "omega_radps")
        _positive(self.density_kgpm3, "density_kgpm3")
        _positive(self.reference_thrust_N, "reference_thrust_N")
        object.__setattr__(
            self,
            "reference_hub_aero_moment_disk_Nm",
            _finite_array(
                self.reference_hub_aero_moment_disk_Nm,
                (3,),
                "reference_hub_aero_moment_disk_Nm",
            ),
        )
        if not (0.0 < float(self.maximum_delta_lambda_fraction) <= 2.0):
            raise DynamicsConfigurationError(
                "maximum_delta_lambda_fraction must lie in (0, 2]"
            )
        if not (0.0 < float(self.maximum_delta_mu_z_fraction) <= 2.0):
            raise DynamicsConfigurationError(
                "maximum_delta_mu_z_fraction must lie in (0, 2]"
            )


def physical_hover_induced_velocity(
    thrust_N: float, density_kgpm3: float, radius_m: float
) -> float:
    """Ideal actuator-disk hover induced speed ``sqrt(T/(2 rho A))``."""

    thrust = _positive(thrust_N, "thrust_N")
    density = _positive(density_kgpm3, "density_kgpm3")
    radius = _positive(radius_m, "radius_m")
    return math.sqrt(thrust / (2.0 * density * math.pi * radius * radius))


def disk_average_delta_mu_z(
    delta_air_velocity: np.ndarray,
    disk_normal: Sequence[float],
    tip_speed_mps: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Return disk-mean axial external-flow perturbation with downwash-positive sign.

    ``delta_air_velocity`` may have any leading disk dimensions and must end in
    three Cartesian components.  Optional nonnegative weights must match those
    leading dimensions (for example, blade-element area weights).
    """

    velocity = np.asarray(delta_air_velocity, dtype=float)
    if velocity.ndim < 2 or velocity.shape[-1] != 3:
        raise DynamicsConfigurationError("delta_air_velocity must have shape (..., 3)")
    if not np.all(np.isfinite(velocity)):
        raise DynamicsDomainError("delta_air_velocity contains NaN or infinity")
    normal = unit_vector(disk_normal, "disk_normal")
    tip_speed = _positive(tip_speed_mps, "tip_speed_mps")
    axial = np.dot(velocity.reshape((-1, 3)), normal)
    if weights is None:
        mean_axial = float(np.mean(axial))
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != velocity.shape[:-1]:
            raise DynamicsConfigurationError(
                "weights must match the leading dimensions of delta_air_velocity"
            )
        flat_weights = weight_array.reshape(-1)
        if (not np.all(np.isfinite(flat_weights)) or np.any(flat_weights < 0.0)
                or float(np.sum(flat_weights)) <= 0.0):
            raise DynamicsConfigurationError("weights must be finite, nonnegative and nonzero")
        mean_axial = float(np.dot(flat_weights, axial) / np.sum(flat_weights))
    return -mean_axial / tip_speed


class NearHoverPittPetersInflow:
    """Three-state frozen-coefficient Pitt--Peters hover perturbation model.

    ``M delta_lambda_dot = Omega (delta_c_eff - K_star delta_lambda)``

    with ``M=diag(128/(75*pi),64/(45*pi),64/(45*pi))`` and
    ``K_star=diag(4*lambda_star,lambda_star,lambda_star)``.  The first forcing
    component includes the axial external-flow correction
    ``delta_CT - 2*lambda_star*delta_mu_z``.  The API accepts perturbation load
    coefficients so total and perturbation quantities cannot be mixed.
    """

    def __init__(
        self,
        parameters: DynamicInflowParameters,
        diagnostics: Optional[DynamicsDiagnostics] = None,
    ) -> None:
        self.parameters = parameters
        self.diagnostics = diagnostics if diagnostics is not None else DynamicsDiagnostics()

    @property
    def tip_speed_mps(self) -> float:
        return self.parameters.radius_m * self.parameters.omega_radps

    @property
    def disk_area_m2(self) -> float:
        return math.pi * self.parameters.radius_m ** 2

    @property
    def apparent_mass_matrix(self) -> np.ndarray:
        return np.diag(
            [128.0 / (75.0 * math.pi), 64.0 / (45.0 * math.pi),
             64.0 / (45.0 * math.pi)]
        )

    @property
    def reference_thrust_coefficient(self) -> float:
        return float(self.load_coefficients(
            self.parameters.reference_thrust_N,
            self.parameters.reference_hub_aero_moment_disk_Nm,
        )[0])

    @property
    def lambda_star(self) -> float:
        return math.sqrt(self.reference_thrust_coefficient / 2.0)

    @property
    def gain_matrix(self) -> np.ndarray:
        value = self.lambda_star
        return np.diag([4.0 * value, value, value])

    @property
    def time_constants_s(self) -> np.ndarray:
        return np.diag(self.apparent_mass_matrix) / (
            self.parameters.omega_radps * np.diag(self.gain_matrix)
        )

    @property
    def hover_total_state(self) -> np.ndarray:
        return np.array([self.lambda_star, 0.0, 0.0], dtype=float)

    @property
    def initial_perturbation_state(self) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def metadata(self) -> Dict[str, object]:
        """Return serializable equations, conventions and frozen coefficients."""

        return {
            "model": "Pitt-Peters near-hover frozen-coefficient perturbation",
            "validity": "normal-working-state, near-axial-hover perturbations",
            "state_order": list(INFLOW_STATE_ORDER),
            "forcing_order": ["delta_CT", "-delta_CMy_hub", "+delta_CMx_hub"],
            "time_variable": "physical seconds; RHS includes Omega",
            "reference_thrust_N": self.parameters.reference_thrust_N,
            "reference_lambda": self.lambda_star,
            "reference_induced_velocity_mps": self.lambda_star * self.tip_speed_mps,
            "apparent_mass_diagonal": np.diag(self.apparent_mass_matrix).tolist(),
            "gain_diagonal": np.diag(self.gain_matrix).tolist(),
            "time_constants_s": self.time_constants_s.tolist(),
            "axial_external_flow_correction": "delta_CT_eff=delta_CT-2*lambda_star*delta_mu_z",
            "maximum_delta_lambda_fraction": self.parameters.maximum_delta_lambda_fraction,
            "maximum_delta_mu_z_fraction": self.parameters.maximum_delta_mu_z_fraction,
        }

    def _validate_delta_state(self, value: Sequence[float]) -> np.ndarray:
        state = _finite_array(value, (3,), "delta_inflow_state")
        self.diagnostics.maximum_abs_delta_inflow_ratio = max(
            self.diagnostics.maximum_abs_delta_inflow_ratio,
            float(np.max(np.abs(state))),
        )
        state_limit = self.parameters.maximum_delta_lambda_fraction * self.lambda_star
        if np.any(np.abs(state) > state_limit):
            self.diagnostics.hard_failures += 1
            raise DynamicsDomainError(
                "inflow perturbation exceeds the configured near-hover limit"
            )
        if self.lambda_star + state[0] <= 0.0:
            self.diagnostics.hard_failures += 1
            raise DynamicsDomainError("total uniform induced inflow became nonpositive")
        return state

    def _validate_delta_mu_z(self, value: float) -> float:
        delta_mu = float(value)
        if not math.isfinite(delta_mu):
            raise DynamicsDomainError("delta_mu_z must be finite")
        self.diagnostics.maximum_abs_delta_mu_z = max(
            self.diagnostics.maximum_abs_delta_mu_z, abs(delta_mu)
        )
        mu_limit = self.parameters.maximum_delta_mu_z_fraction * self.lambda_star
        if abs(delta_mu) > mu_limit:
            self.diagnostics.hard_failures += 1
            raise DynamicsDomainError(
                "axial external-flow perturbation exceeds the configured near-hover limit"
            )
        return delta_mu

    def state_from_hover_thrust(self, thrust_N: float) -> np.ndarray:
        """Return zero perturbation after validating the declared hover thrust."""

        thrust = _positive(thrust_N, "thrust_N")
        tolerance = max(1.0e-9 * self.parameters.reference_thrust_N, 1.0e-9)
        if abs(thrust - self.parameters.reference_thrust_N) > tolerance:
            raise DynamicsConfigurationError(
                "hover initialization thrust differs from frozen reference thrust"
            )
        return self.initial_perturbation_state

    def state_from_hover_mass(self, mass_kg: float, gravity_mps2: float = 9.80665) -> np.ndarray:
        return self.state_from_hover_thrust(
            _positive(mass_kg, "mass_kg") * _positive(gravity_mps2, "gravity_mps2")
        )

    def dimensional_induced_velocity(
        self,
        delta_inflow_state: Sequence[float],
        radius_ratio: Union[float, Sequence[float], np.ndarray],
        azimuth_rad: Union[float, Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Evaluate positive-through-``-n`` induced velocity at disk locations."""

        state = self._validate_delta_state(delta_inflow_state)
        rbar = np.asarray(radius_ratio, dtype=float)
        psi = np.asarray(azimuth_rad, dtype=float)
        if not np.all(np.isfinite(rbar)) or not np.all(np.isfinite(psi)):
            raise DynamicsDomainError("radius ratio and azimuth must be finite")
        if np.any(rbar < 0.0) or np.any(rbar > 1.0):
            raise DynamicsDomainError("radius_ratio must remain within [0, 1]")
        local_ratio = self.lambda_star + state[0] + rbar * (
            state[1] * np.cos(psi) + state[2] * np.sin(psi)
        )
        if np.any(local_ratio <= 0.0):
            self.diagnostics.hard_failures += 1
            raise DynamicsDomainError(
                "local total induced inflow became nonpositive; near-hover model invalid"
            )
        return self.tip_speed_mps * local_ratio

    def load_coefficients(
        self, thrust_N: float, hub_aero_moment_disk_Nm: Sequence[float]
    ) -> np.ndarray:
        """Return forcing ``[C_T, -C_My, +C_Mx]`` from hub aerodynamic loads."""

        thrust = float(thrust_N)
        moment = _finite_array(hub_aero_moment_disk_Nm, (3,), "hub_aero_moment_disk_Nm")
        if not math.isfinite(thrust):
            raise DynamicsDomainError("thrust_N must be finite")
        force_scale = (
            self.parameters.density_kgpm3
            * self.disk_area_m2
            * self.tip_speed_mps ** 2
        )
        moment_scale = force_scale * self.parameters.radius_m
        return np.array(
            [thrust / force_scale, -moment[1] / moment_scale, moment[0] / moment_scale],
            dtype=float,
        )

    def rhs(
        self,
        time_s: float,
        delta_inflow_state: Sequence[float],
        delta_load_coefficients: Sequence[float],
        delta_mu_z: float = 0.0,
    ) -> np.ndarray:
        """Return the perturbation-state derivative in physical seconds.

        ``delta_load_coefficients`` must be ``[delta_CT, -delta_CMy,
        +delta_CMx]``.  Supplying total coefficients here is a contract error.
        Use :meth:`coefficient_perturbation_from_total_loads` first when loads
        are available as dimensional totals.  ``delta_mu_z`` is positive for
        external flow along ``-n`` and can be computed with
        :func:`disk_average_delta_mu_z`.
        """

        del time_s
        state = self._validate_delta_state(delta_inflow_state)
        delta_coefficients = _finite_array(
            delta_load_coefficients, (3,), "delta_load_coefficients"
        )
        delta_mu = self._validate_delta_mu_z(delta_mu_z)
        self.diagnostics.inflow_rhs_calls += 1

        effective_forcing = delta_coefficients.copy()
        effective_forcing[0] -= 2.0 * self.lambda_star * delta_mu
        derivative = self.parameters.omega_radps * np.linalg.solve(
            self.apparent_mass_matrix,
            effective_forcing - np.dot(self.gain_matrix, state),
        )
        if not np.all(np.isfinite(derivative)):
            self.diagnostics.hard_failures += 1
            raise DynamicsDomainError("dynamic-inflow derivative became non-finite")
        return derivative

    def coefficient_perturbation_from_total_loads(
        self, thrust_N: float, hub_aero_moment_disk_Nm: Sequence[float]
    ) -> np.ndarray:
        """Convert dimensional total hub loads to coefficients about the reference."""

        total = self.load_coefficients(thrust_N, hub_aero_moment_disk_Nm)
        reference = self.load_coefficients(
            self.parameters.reference_thrust_N,
            self.parameters.reference_hub_aero_moment_disk_Nm,
        )
        return total - reference

    def exact_frozen_forcing_step(
        self,
        delta_inflow_state: Sequence[float],
        delta_load_coefficients: Sequence[float],
        delta_mu_z: float,
        dt_s: float,
    ) -> np.ndarray:
        """Exact update of the isolated linear inflow subsystem for frozen forcing."""

        state = self._validate_delta_state(delta_inflow_state)
        forcing = _finite_array(
            delta_load_coefficients, (3,), "delta_load_coefficients"
        ).copy()
        forcing[0] -= 2.0 * self.lambda_star * self._validate_delta_mu_z(delta_mu_z)
        dt = _positive(dt_s, "dt_s")
        gain_diagonal = np.diag(self.gain_matrix)
        decay = np.exp(-dt / self.time_constants_s)
        equilibrium = forcing / gain_diagonal
        result = decay * state + (1.0 - decay) * equilibrium
        return self._validate_delta_state(result)


# Short backward-compatible name for callers written while this module was
# being integrated.  Both names refer to the same explicitly near-hover model.
ThreeStateDynamicInflow = NearHoverPittPetersInflow


@dataclass(frozen=True)
class BodyLoads:
    """Rotor and balance loads with aerodynamic/transmitted moments separated.

    ``hub_aero_moment_body_Nm`` is the unscaled disk-load moment used by the
    dynamic-inflow model.  It is deliberately *not* applied to the fuselage.
    ``transmitted_hub_moment_body_Nm`` is the moment delivered through a
    separately declared hub/structural model and is the only hub moment used
    in the rigid-body equation.  This required field prevents accidental
    transmission of the complete disk bending moment.  ``other_force_body_N``
    and ``other_moment_body_Nm`` are defined at the CG.
    """

    rotor_force_body_N: np.ndarray
    hub_aero_moment_body_Nm: np.ndarray
    transmitted_hub_moment_body_Nm: np.ndarray
    other_force_body_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    other_moment_body_Nm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotor_force_body_N",
            _finite_array(self.rotor_force_body_N, (3,), "rotor_force_body_N"),
        )
        object.__setattr__(
            self,
            "hub_aero_moment_body_Nm",
            _finite_array(
                self.hub_aero_moment_body_Nm, (3,), "hub_aero_moment_body_Nm"
            ),
        )
        object.__setattr__(
            self,
            "transmitted_hub_moment_body_Nm",
            _finite_array(
                self.transmitted_hub_moment_body_Nm,
                (3,),
                "transmitted_hub_moment_body_Nm",
            ),
        )
        object.__setattr__(
            self,
            "other_force_body_N",
            _finite_array(self.other_force_body_N, (3,), "other_force_body_N"),
        )
        object.__setattr__(
            self,
            "other_moment_body_Nm",
            _finite_array(self.other_moment_body_Nm, (3,), "other_moment_body_Nm"),
        )


@dataclass(frozen=True)
class RigidBody3DOFParameters:
    """Parameters for yaw-constrained roll/pitch/heave rigid-body dynamics."""

    mass_kg: float
    inertia_body_kgm2: np.ndarray
    hub_from_cg_body_m: np.ndarray
    gravity_mps2: float = 9.80665
    vertical_damping_N_per_mps: float = 0.0
    angular_damping_body_Nms: np.ndarray = field(default_factory=lambda: np.zeros(3))
    maximum_abs_attitude_deg: float = 80.0
    maximum_abs_heave_m: float = math.inf
    maximum_abs_heave_speed_mps: float = math.inf

    def __post_init__(self) -> None:
        _positive(self.mass_kg, "mass_kg")
        _positive(self.gravity_mps2, "gravity_mps2")
        inertia = _finite_array(self.inertia_body_kgm2, (3, 3), "inertia_body_kgm2")
        if not np.allclose(inertia, inertia.T, rtol=1.0e-10, atol=1.0e-10):
            raise DynamicsConfigurationError("inertia_body_kgm2 must be symmetric")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise DynamicsConfigurationError("inertia_body_kgm2 must be positive definite")
        object.__setattr__(self, "inertia_body_kgm2", inertia)
        object.__setattr__(
            self,
            "hub_from_cg_body_m",
            _finite_array(self.hub_from_cg_body_m, (3,), "hub_from_cg_body_m"),
        )
        damping = _finite_array(
            self.angular_damping_body_Nms, (3,), "angular_damping_body_Nms"
        )
        if np.any(damping < 0.0) or self.vertical_damping_N_per_mps < 0.0:
            raise DynamicsConfigurationError("damping coefficients cannot be negative")
        object.__setattr__(self, "angular_damping_body_Nms", damping)
        if not (0.0 < float(self.maximum_abs_attitude_deg) < 90.0):
            raise DynamicsConfigurationError(
                "maximum_abs_attitude_deg must lie between 0 and 90 degrees"
            )
        if self.maximum_abs_heave_m <= 0.0 or self.maximum_abs_heave_speed_mps <= 0.0:
            raise DynamicsConfigurationError("heave safety limits must be positive")

    def metadata(self) -> Dict[str, object]:
        return {
            "model": "yaw-constrained nonlinear rigid-body roll/pitch/heave",
            "state_order": list(BODY_STATE_ORDER),
            "world_vertical": "+z upward",
            "attitude_sequence": "R_wb=Ry(pitch)Rx(roll), yaw constrained to zero",
            "hub_moment_contract": (
                "rigid body uses only explicitly supplied transmitted hub moment; "
                "unscaled aerodynamic disk moment is retained separately for inflow"
            ),
            "mass_kg": self.mass_kg,
            "inertia_body_kgm2": self.inertia_body_kgm2.tolist(),
            "hub_from_cg_body_m": self.hub_from_cg_body_m.tolist(),
            "vertical_damping_N_per_mps": self.vertical_damping_N_per_mps,
            "angular_damping_body_Nms": self.angular_damping_body_Nms.tolist(),
            "maximum_abs_attitude_deg": self.maximum_abs_attitude_deg,
            "maximum_abs_heave_m": self.maximum_abs_heave_m,
            "maximum_abs_heave_speed_mps": self.maximum_abs_heave_speed_mps,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, float],
        hub_from_cg_body_m: Optional[Sequence[float]] = None,
        **kwargs: object
    ) -> "RigidBody3DOFParameters":
        """Build from configuration without silently inventing missing inertia.

        Required keys are ``mass_kg``, ``Ixx_kgm2``, ``Iyy_kgm2`` and
        ``Izz_kgm2``.  Products of inertia default to zero only when absent,
        because zero products are an explicit principal-axis assumption.
        """

        required = ("mass_kg", "Ixx_kgm2", "Iyy_kgm2", "Izz_kgm2")
        missing = [key for key in required if key not in mapping]
        if missing:
            raise DynamicsConfigurationError(
                "missing rigid-body configuration keys: %s" % ", ".join(missing)
            )
        ixx = float(mapping["Ixx_kgm2"])
        iyy = float(mapping["Iyy_kgm2"])
        izz = float(mapping["Izz_kgm2"])
        ixy = float(mapping.get("Ixy_kgm2", 0.0))
        ixz = float(mapping.get("Ixz_kgm2", 0.0))
        iyz = float(mapping.get("Iyz_kgm2", 0.0))
        # Tensor convention: supplied Ixy/Ixz/Iyz are products of inertia and
        # therefore enter the inertia tensor with negative off-diagonal signs.
        inertia = np.array(
            [[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]], dtype=float
        )
        if hub_from_cg_body_m is None:
            if "hub_above_cg_m" not in mapping:
                raise DynamicsConfigurationError(
                    "hub_from_cg_body_m or hub_above_cg_m is required"
                )
            hub_from_cg_body_m = (0.0, 0.0, float(mapping["hub_above_cg_m"]))
        return cls(
            mass_kg=float(mapping["mass_kg"]),
            inertia_body_kgm2=inertia,
            hub_from_cg_body_m=np.asarray(hub_from_cg_body_m, dtype=float),
            **kwargs
        )


@dataclass(frozen=True)
class HoverTrimReference:
    """Fixed non-rotor loads that balance one declared uniform-hover state."""

    other_force_body_N: np.ndarray
    other_moment_body_Nm: np.ndarray
    rotor_thrust_N: float
    induced_velocity_mps: float
    reference_lambda: float
    delta_inflow_state: np.ndarray


def moment_about_cg_components(
    rotor_force_body_N: Sequence[float],
    transmitted_hub_moment_body_Nm: Sequence[float],
    hub_from_cg_body_m: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(transmitted hub, force-arm, total)`` moments about the CG."""

    force = _finite_array(rotor_force_body_N, (3,), "rotor_force_body_N")
    transmitted_moment = _finite_array(
        transmitted_hub_moment_body_Nm, (3,), "transmitted_hub_moment_body_Nm"
    )
    arm = _finite_array(hub_from_cg_body_m, (3,), "hub_from_cg_body_m")
    arm_moment = np.cross(arm, force)
    return transmitted_moment.copy(), arm_moment, transmitted_moment + arm_moment


def integrate_element_loads_about_hub(
    element_points: np.ndarray,
    element_forces: np.ndarray,
    hub_position: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate resultant force and the unscaled aerodynamic moment at the hub.

    Inputs may have any leading blade/radial dimensions but must end in length
    three and share the same shape.  All quantities must be in the same frame.
    """

    points = np.asarray(element_points, dtype=float)
    forces = np.asarray(element_forces, dtype=float)
    hub = _finite_array(hub_position, (3,), "hub_position")
    if points.shape != forces.shape or points.ndim < 2 or points.shape[-1] != 3:
        raise DynamicsConfigurationError(
            "element_points and element_forces must share shape (..., 3)"
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(forces)):
        raise DynamicsDomainError("element loads contain NaN or infinity")
    flat_points = points.reshape((-1, 3))
    flat_forces = forces.reshape((-1, 3))
    resultant_force = np.sum(flat_forces, axis=0)
    hub_aero_moment = np.sum(np.cross(flat_points - hub, flat_forces), axis=0)
    return resultant_force, hub_aero_moment


def physical_hover_trim_reference(
    body_parameters: RigidBody3DOFParameters,
    inflow_model: ThreeStateDynamicInflow,
    rotor_force_body_N: Sequence[float],
    hub_aero_moment_body_Nm: Sequence[float],
    transmitted_hub_moment_body_Nm: Sequence[float],
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    rotor_axis_body: Sequence[float] = (0.0, 0.0, 1.0),
) -> HoverTrimReference:
    """Create fixed balance loads and momentum-theory inflow for physical hover.

    The returned balance is computed once from the declared uniform-hover
    reference; it must be reused unchanged for every modal case.  This is not
    per-case mean-load removal.
    """

    force = _finite_array(rotor_force_body_N, (3,), "rotor_force_body_N")
    hub_moment = _finite_array(
        hub_aero_moment_body_Nm, (3,), "hub_aero_moment_body_Nm"
    )
    transmitted_moment = _finite_array(
        transmitted_hub_moment_body_Nm, (3,), "transmitted_hub_moment_body_Nm"
    )
    rotation = rotation_body_to_world(roll_rad, pitch_rad)
    required_force_body = np.dot(
        rotation.T, np.array([0.0, 0.0, body_parameters.mass_kg * body_parameters.gravity_mps2])
    )
    other_force = required_force_body - force
    _, arm_moment, cg_rotor_moment = moment_about_cg_components(
        force, transmitted_moment, body_parameters.hub_from_cg_body_m
    )
    del arm_moment
    other_moment = -cg_rotor_moment
    axis = unit_vector(rotor_axis_body, "rotor_axis_body")
    rotor_thrust = float(np.dot(force, axis))
    if rotor_thrust <= 0.0:
        raise DynamicsConfigurationError(
            "uniform-hover rotor force must have positive thrust along rotor axis"
        )
    inflow_state = inflow_model.state_from_hover_thrust(rotor_thrust)
    coefficient_delta = inflow_model.coefficient_perturbation_from_total_loads(
        rotor_thrust, hub_moment
    )
    if not np.allclose(coefficient_delta, 0.0, atol=1.0e-12, rtol=1.0e-9):
        raise DynamicsConfigurationError(
            "hover hub aerodynamic loads differ from the frozen inflow reference"
        )
    induced = float(inflow_model.lambda_star * inflow_model.tip_speed_mps)
    return HoverTrimReference(
        other_force_body_N=other_force,
        other_moment_body_Nm=other_moment,
        rotor_thrust_N=rotor_thrust,
        induced_velocity_mps=induced,
        reference_lambda=inflow_model.lambda_star,
        delta_inflow_state=inflow_state,
    )


LoadProvider = Union[BodyLoads, Callable[[float, np.ndarray], BodyLoads]]


def _resolve_loads(loads: LoadProvider, time_s: float, state: np.ndarray) -> BodyLoads:
    resolved = loads(time_s, state) if callable(loads) else loads
    if not isinstance(resolved, BodyLoads):
        raise DynamicsConfigurationError("load provider must return BodyLoads")
    return resolved


def rigid_body_3dof_rhs(
    time_s: float,
    state: Sequence[float],
    parameters: RigidBody3DOFParameters,
    loads: LoadProvider,
    diagnostics: Optional[DynamicsDiagnostics] = None,
) -> np.ndarray:
    """Yaw-constrained nonlinear roll/pitch/heave equations of motion.

    The attitude equations are obtained by projecting Euler's rigid-body
    equation onto the zero-yaw tangent basis ``J``:

    ``J.T I J qdd = J.T (M - I Jdot qdot - omega x I omega)``.

    Thus the implementation retains nonlinear kinematics, products of inertia,
    gyroscopic coupling and the constraint reaction without integrating yaw.
    """

    values = _finite_array(state, (6,), "rigid_body_state")
    z, vz, phi, theta, phi_dot, theta_dot = values
    diag = diagnostics if diagnostics is not None else DynamicsDiagnostics()
    diag.rigid_body_rhs_calls += 1
    attitude_deg = max(abs(math.degrees(phi)), abs(math.degrees(theta)))
    diag.maximum_abs_attitude_deg = max(diag.maximum_abs_attitude_deg, attitude_deg)
    if attitude_deg > parameters.maximum_abs_attitude_deg:
        diag.hard_failures += 1
        raise DynamicsDomainError(
            "attitude %.3f deg exceeds configured limit %.3f deg"
            % (attitude_deg, parameters.maximum_abs_attitude_deg)
        )
    if abs(z) > parameters.maximum_abs_heave_m:
        diag.hard_failures += 1
        raise DynamicsDomainError("heave displacement exceeded configured limit")
    if abs(vz) > parameters.maximum_abs_heave_speed_mps:
        diag.hard_failures += 1
        raise DynamicsDomainError("heave speed exceeded configured limit")

    resolved = _resolve_loads(loads, float(time_s), values)
    force_body = resolved.rotor_force_body_N + resolved.other_force_body_N
    transmitted_moment, arm_moment, rotor_cg_moment = moment_about_cg_components(
        resolved.rotor_force_body_N,
        resolved.transmitted_hub_moment_body_Nm,
        parameters.hub_from_cg_body_m,
    )
    del transmitted_moment, arm_moment

    rotation = rotation_body_to_world(phi, theta)
    force_world = np.dot(rotation, force_body)
    vertical_force = (
        force_world[2]
        - parameters.mass_kg * parameters.gravity_mps2
        - parameters.vertical_damping_N_per_mps * vz
    )
    vertical_acceleration = vertical_force / parameters.mass_kg

    generalized_rates = np.array([phi_dot, theta_dot], dtype=float)
    projection = euler_rate_projection(phi)
    omega_body = np.dot(projection, generalized_rates)
    projection_dot = np.array(
        [
            [0.0, 0.0],
            [0.0, -math.sin(phi) * phi_dot],
            [0.0, -math.cos(phi) * phi_dot],
        ],
        dtype=float,
    )
    inertia = parameters.inertia_body_kgm2
    total_moment_body = (
        rotor_cg_moment
        + resolved.other_moment_body_Nm
        - parameters.angular_damping_body_Nms * omega_body
    )
    gyroscopic_moment = np.cross(omega_body, np.dot(inertia, omega_body))
    projected_inertia = np.dot(projection.T, np.dot(inertia, projection))
    projected_rhs = np.dot(
        projection.T,
        total_moment_body
        - np.dot(inertia, np.dot(projection_dot, generalized_rates))
        - gyroscopic_moment,
    )
    try:
        generalized_acceleration = np.linalg.solve(projected_inertia, projected_rhs)
    except np.linalg.LinAlgError as exc:
        diag.hard_failures += 1
        raise DynamicsDomainError("projected attitude inertia is singular") from exc

    derivative = np.array(
        [
            vz,
            vertical_acceleration,
            phi_dot,
            theta_dot,
            generalized_acceleration[0],
            generalized_acceleration[1],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(derivative)):
        diag.hard_failures += 1
        raise DynamicsDomainError("rigid-body derivative became non-finite")
    return derivative


@dataclass(frozen=True)
class RotorDiskGeometry:
    """Moving, tilted rotor sampling geometry in world coordinates."""

    hub_world_m: np.ndarray
    normal_world: np.ndarray
    e1_world: np.ndarray
    e2_world: np.ndarray
    points_world_m: np.ndarray
    blade_point_velocity_world_mps: np.ndarray
    radial_world: np.ndarray
    tangential_world: np.ndarray


def moving_tilted_disk_geometry(
    rigid_body_state: Sequence[float],
    nominal_cg_world_m: Sequence[float],
    hub_from_cg_body_m: Sequence[float],
    rotor_axis_body: Sequence[float],
    radii_m: Sequence[float],
    blade_count: int,
    phase_rad: float,
    rotor_omega_radps: float,
) -> RotorDiskGeometry:
    """Return blade-element coordinates and velocities for the current state."""

    state = _finite_array(rigid_body_state, (6,), "rigid_body_state")
    nominal_cg = _finite_array(nominal_cg_world_m, (3,), "nominal_cg_world_m")
    hub_offset = _finite_array(hub_from_cg_body_m, (3,), "hub_from_cg_body_m")
    radii = np.asarray(radii_m, dtype=float)
    if radii.ndim != 1 or len(radii) == 0 or not np.all(np.isfinite(radii)):
        raise DynamicsConfigurationError("radii_m must be a nonempty finite 1-D array")
    if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise DynamicsConfigurationError("radii_m must be strictly increasing and positive")
    if int(blade_count) != blade_count or blade_count < 1:
        raise DynamicsConfigurationError("blade_count must be a positive integer")
    if not math.isfinite(float(phase_rad)):
        raise DynamicsDomainError("phase_rad must be finite")
    omega_rotor = _positive(rotor_omega_radps, "rotor_omega_radps")

    z, vz, phi, theta, phi_dot, theta_dot = state
    rotation = rotation_body_to_world(phi, theta)
    cg_world = nominal_cg + np.array([0.0, 0.0, z])
    hub_world = cg_world + np.dot(rotation, hub_offset)
    normal_body, e1_body, e2_body = disk_basis(rotor_axis_body)
    normal_world = np.dot(rotation, normal_body)
    e1_world = np.dot(rotation, e1_body)
    e2_world = np.dot(rotation, e2_body)
    omega_body = body_angular_velocity(phi, phi_dot, theta_dot)
    omega_world = np.dot(rotation, omega_body)
    cg_velocity_world = np.array([0.0, 0.0, vz])

    points = np.empty((int(blade_count), len(radii), 3), dtype=float)
    velocities = np.empty_like(points)
    radial_directions = np.empty((int(blade_count), 3), dtype=float)
    tangential_directions = np.empty_like(radial_directions)
    for blade in range(int(blade_count)):
        azimuth = float(phase_rad) + 2.0 * math.pi * blade / int(blade_count)
        radial = math.cos(azimuth) * e1_world + math.sin(azimuth) * e2_world
        tangential = -math.sin(azimuth) * e1_world + math.cos(azimuth) * e2_world
        radial_directions[blade] = radial
        tangential_directions[blade] = tangential
        relative_to_cg = (
            np.dot(rotation, hub_offset)[None, :] + radii[:, None] * radial[None, :]
        )
        points[blade] = cg_world[None, :] + relative_to_cg
        velocities[blade] = (
            cg_velocity_world[None, :]
            + np.cross(np.broadcast_to(omega_world, relative_to_cg.shape), relative_to_cg)
            + omega_rotor * radii[:, None] * tangential[None, :]
        )
    return RotorDiskGeometry(
        hub_world_m=hub_world,
        normal_world=normal_world,
        e1_world=e1_world,
        e2_world=e2_world,
        points_world_m=points,
        blade_point_velocity_world_mps=velocities,
        radial_world=radial_directions,
        tangential_world=tangential_directions,
    )


RhsFunction = Callable[[float, np.ndarray], np.ndarray]


def heun_step(rhs: RhsFunction, time_s: float, state: Sequence[float], dt_s: float) -> np.ndarray:
    """Advance an autonomous or non-autonomous ODE by explicit Heun RK2."""

    state_array = np.asarray(state, dtype=float)
    dt = _positive(dt_s, "dt_s")
    k1 = np.asarray(rhs(float(time_s), state_array.copy()), dtype=float)
    if k1.shape != state_array.shape:
        raise DynamicsConfigurationError("rhs returned a state derivative with wrong shape")
    predictor = state_array + dt * k1
    k2 = np.asarray(rhs(float(time_s) + dt, predictor), dtype=float)
    if k2.shape != state_array.shape:
        raise DynamicsConfigurationError("rhs returned a state derivative with wrong shape")
    result = state_array + 0.5 * dt * (k1 + k2)
    if not np.all(np.isfinite(result)):
        raise DynamicsDomainError("Heun step produced a non-finite state")
    return result


def rk4_step(rhs: RhsFunction, time_s: float, state: Sequence[float], dt_s: float) -> np.ndarray:
    """Advance an ODE by classical fourth-order Runge--Kutta."""

    state_array = np.asarray(state, dtype=float)
    dt = _positive(dt_s, "dt_s")
    t = float(time_s)
    k1 = np.asarray(rhs(t, state_array.copy()), dtype=float)
    k2 = np.asarray(rhs(t + 0.5 * dt, state_array + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(rhs(t + 0.5 * dt, state_array + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(rhs(t + dt, state_array + dt * k3), dtype=float)
    for derivative in (k1, k2, k3, k4):
        if derivative.shape != state_array.shape:
            raise DynamicsConfigurationError("rhs returned a state derivative with wrong shape")
    result = state_array + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not np.all(np.isfinite(result)):
        raise DynamicsDomainError("RK4 step produced a non-finite state")
    return result


def integrate_step(
    rhs: RhsFunction,
    time_s: float,
    state: Sequence[float],
    dt_s: float,
    method: str = "rk4",
) -> np.ndarray:
    """Dispatch to ``rk4`` or ``heun`` with an explicit method name."""

    if method == "rk4":
        return rk4_step(rhs, time_s, state, dt_s)
    if method == "heun":
        return heun_step(rhs, time_s, state, dt_s)
    raise DynamicsConfigurationError("integration method must be 'rk4' or 'heun'")
