#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""One-way 3-D NATO-GD/RVMD coupling to nonlinear rotorcraft dynamics.

The solver first obtains a still-air momentum/BEM hover solution as a numerical
seed.  It then solves a fixed-body, azimuth-mean equilibrium in the stored
NATO-GD time-mean field for collective and all three Pitt--Peters inflow
states.  Every cumulative high-frequency-deletion case starts from those same
controls and states.  No per-case load mean is subtracted.  A single constant
reaction balancing the remaining mean fixed-body force/moment is computed once
per rotor and reused unchanged for all ``K+1`` RVMD stages and the original-CFD
baseline.  This is an azimuth-mean equilibrium, not periodic-orbit shooting.

Stage semantics are fixed for the energy-selected mode count ``K``: stage 0
retains all frequency-sorted modes; stage ``j`` retains the first ``K-j``
low-frequency modes, so stage 1 deletes mode ``K`` and stage ``K`` contains
the time-mean field only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .nonlinear_blade_element import (
    AerodynamicDomainLimits,
    BladeElementLoads,
    BladeElementParameters,
    integrate_blade_element_loads,
    radial_element_centres_and_widths,
)
from .nonlinear_rotorcraft_dynamics import (
    BodyLoads,
    DynamicInflowParameters,
    DynamicsDiagnostics,
    NearHoverPittPetersInflow,
    RigidBody3DOFParameters,
    body_angular_velocity,
    disk_average_delta_mu_z,
    heun_step,
    moving_tilted_disk_geometry,
    physical_hover_induced_velocity,
    resolve_vector_in_basis,
    rigid_body_3dof_rhs,
    rotation_body_to_world,
)
from .nato_gd_study import (
    StudyConfigurationError,
    StudyContext,
    build_stage_rows,
    initialise_output_tree,
    load_study_context,
    validate_grid_and_time,
    write_context_provenance,
    write_stage_manifest,
)
from .rvmd_3d_flow import NatoGD3DBaselineFlow, RVMD3DFlow


CORE_METRICS = ("lift_N", "engine_power_W", "roll_deg", "pitch_deg", "heave_m")
STATISTICS = ("mean", "std", "rms", "peak_to_peak", "max_abs")


class CouplingConfigurationError(ValueError):
    pass


class CouplingRunError(RuntimeError):
    pass


_WORKER_RVMD_FLOW: Optional[RVMD3DFlow] = None
_WORKER_BASELINE_FLOW: Optional[NatoGD3DBaselineFlow] = None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CouplingConfigurationError("JSON root must be an object: {}".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
    return path


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "rotor"


def _artifact_signature(paths: Sequence[Path]) -> str:
    """Hash the exact configuration and modal artifacts that define a run."""

    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise CouplingConfigurationError("signature input is missing: {}".format(path))
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_full_space_energy_gate(
    metadata: Mapping[str, Any], target_fraction: Optional[float]
) -> Dict[str, Any]:
    """Validate the mean-removed full-space reconstruction-energy gate.

    The gate deliberately uses ``1 - residual_energy/original_energy``.  RVMD
    modes are not generally orthogonal, so a sum of individual modal
    self-energies is not a valid substitute for this check.
    """

    full_audit = metadata.get("full_space_regression_audit")
    if not isinstance(full_audit, dict):
        raise CouplingConfigurationError("full-space RVMD regression audit is missing")
    residual_fraction = float(
        full_audit.get("full_normalized_residual_energy_fraction", float("nan"))
    )
    captured_fraction = float(
        full_audit.get(
            "full_normalized_captured_energy_fraction_one_minus_residual",
            float("nan"),
        )
    )
    if (
        not math.isfinite(residual_fraction)
        or not math.isfinite(captured_fraction)
        or residual_fraction < 0.0
        or residual_fraction > 1.0 + 1.0e-12
    ):
        raise CouplingConfigurationError("full-space RVMD residual audit is invalid")
    if not math.isclose(
        captured_fraction,
        1.0 - residual_fraction,
        rel_tol=0.0,
        abs_tol=2.0e-10,
    ):
        raise CouplingConfigurationError(
            "full-space captured energy is inconsistent with one minus residual energy"
        )

    if target_fraction is None:
        return {
            "target_fraction": None,
            "captured_fraction": captured_fraction,
            "passed": True,
            "metric": "mean-removed full-space residual reconstruction energy",
        }

    target = float(target_fraction)
    if not math.isfinite(target) or target <= 0.0 or target > 1.0:
        raise CouplingConfigurationError("full-space energy target must lie in (0, 1]")
    gate = metadata.get("full_space_energy_gate")
    if not isinstance(gate, dict):
        raise CouplingConfigurationError(
            "the configured full-space energy target requires a persisted energy gate"
        )
    try:
        persisted_target = float(gate["target_fraction"])
        persisted_captured = float(gate["captured_fraction"])
    except (KeyError, TypeError, ValueError) as exception:
        raise CouplingConfigurationError("persisted full-space energy gate is invalid") from exception
    if not math.isclose(persisted_target, target, rel_tol=0.0, abs_tol=1.0e-12):
        raise CouplingConfigurationError(
            "persisted full-space energy target disagrees with the study configuration"
        )
    if not math.isclose(
        persisted_captured, captured_fraction, rel_tol=0.0, abs_tol=2.0e-10
    ):
        raise CouplingConfigurationError(
            "persisted full-space energy gate disagrees with the regression audit"
        )
    if gate.get("modal_self_energy_sum_prohibited_for_gate") is not True:
        raise CouplingConfigurationError(
            "energy gate does not explicitly prohibit modal self-energy summation"
        )
    passed = captured_fraction + 1.0e-12 >= target
    if gate.get("passed") is not True or not passed:
        raise CouplingConfigurationError(
            "full-space captured fluctuation energy {:.9f} is below required {:.9f}".format(
                captured_fraction, target
            )
        )
    return {
        "target_fraction": target,
        "captured_fraction": captured_fraction,
        "passed": True,
        "metric": "mean-removed full-space residual reconstruction energy",
    }


def _world_to_body(vector_world: Sequence[float], body_state: np.ndarray) -> np.ndarray:
    rotation = rotation_body_to_world(float(body_state[2]), float(body_state[3]))
    return np.dot(rotation.T, np.asarray(vector_world, dtype=float))


def _disk_components(vector_world: Sequence[float], geometry: Any) -> np.ndarray:
    return resolve_vector_in_basis(
        vector_world, geometry.e1_world, geometry.e2_world, geometry.normal_world
    )


@dataclass(frozen=True)
class HoverTrim:
    rotor_name: str
    collective_input_deg: float
    collective_trim_deg: float
    weight_N: float
    induced_velocity_mps: float
    reference_lambda: float
    disk_loading_Npm2: float
    loads: BladeElementLoads
    alpha_domain_violation_count: int
    cl_saturation_count: int
    relative_thrust_residual: float

    def as_row(self) -> Dict[str, Any]:
        return {
            "rotor_name": self.rotor_name,
            "collective_input_deg": self.collective_input_deg,
            "collective_trim_deg": self.collective_trim_deg,
            "weight_N": self.weight_N,
            "induced_velocity_trim_mps": self.induced_velocity_mps,
            "reference_lambda": self.reference_lambda,
            "disk_loading_Npm2": self.disk_loading_Npm2,
            "trim_thrust_N": self.loads.thrust_N,
            "trim_Fx_N": self.loads.hub_force_world_N[0],
            "trim_Fy_N": self.loads.hub_force_world_N[1],
            "trim_Fz_N": self.loads.hub_force_world_N[2],
            "trim_hub_Mx_Nm": self.loads.hub_aerodynamic_moment_world_Nm[0],
            "trim_hub_My_Nm": self.loads.hub_aerodynamic_moment_world_Nm[1],
            "trim_hub_Mz_Nm": self.loads.hub_aerodynamic_moment_world_Nm[2],
            "trim_shaft_torque_Nm": self.loads.shaft_torque_Nm,
            "trim_shaft_power_W": self.loads.shaft_power_W,
            "trim_engine_power_W": self.loads.engine_power_W,
            "alpha_domain_violation_count": self.alpha_domain_violation_count,
            "cl_saturation_count": self.cl_saturation_count,
            "relative_thrust_residual": self.relative_thrust_residual,
        }


@dataclass
class RotorStudyModel:
    rotor: Dict[str, Any]
    blade_parameters: BladeElementParameters
    body_parameters: RigidBody3DOFParameters
    inflow_model: NearHoverPittPetersInflow
    radii_m: np.ndarray
    widths_m: np.ndarray
    rotor_axis_body: np.ndarray
    nominal_cg_world_m: np.ndarray
    initial_phase_rad: float
    density_kgpm3: float
    trim: HoverTrim
    balance_other_force_body_N: np.ndarray
    balance_other_moment_body_Nm: np.ndarray
    balance_definition: str
    initial_inflow_state: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    mean_flow_trim_audit: Dict[str, Any] = field(default_factory=dict)
    hover_hold_vertical_stiffness_Npm: float = 0.0
    hover_hold_vertical_rate_gain_Ns_pm: float = 0.0
    hover_hold_angular_stiffness_Nm_per_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    hover_hold_angular_rate_gain_Nms: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    hover_hold_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    stage_index: Optional[int]
    retained_mode_count: Optional[int]
    flow_kind: str
    cutoff_frequency_hz: Optional[float]
    cutoff_St: Optional[float]
    mode_indices_zero_based: Optional[Tuple[int, ...]] = None
    sensitivity_case_kind: Optional[str] = None
    sensitivity_mode_index_one_based: Optional[int] = None


@dataclass
class CaseResult:
    definition: CaseDefinition
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def load_rotor_mappings(context: StudyContext) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base = _read_json(context.rotor_config_path)
    simulation = base.get("simulation")
    rotors = base.get("rotors")
    if not isinstance(simulation, dict) or not isinstance(rotors, list) or not rotors:
        raise CouplingConfigurationError("rotor_config.json requires simulation and rotors")
    overrides_root = _read_json(context.dynamic_overrides_path)
    overrides = overrides_root.get("rotors")
    if not isinstance(overrides, dict):
        raise CouplingConfigurationError("dynamic overrides require a rotors object")
    merged: List[Dict[str, Any]] = []
    for raw in rotors:
        if not isinstance(raw, dict) or "name" not in raw:
            raise CouplingConfigurationError("each rotor entry must be a named object")
        name = str(raw["name"])
        if name not in overrides or not isinstance(overrides[name], dict):
            raise CouplingConfigurationError("missing dynamic override for {}".format(name))
        item = dict(raw)
        item.update(overrides[name])
        # The old empirical transfer factors remain in the legacy file but are
        # not copied into any physical calculation.
        item.pop("flapping_moment_transfer", None)
        merged.append(item)
    return dict(simulation), merged


def _hover_geometry(
    rotor: Mapping[str, Any], radii_m: np.ndarray, rotor_axis_body: np.ndarray
) -> Tuple[Any, np.ndarray]:
    body_state = np.zeros(6, dtype=float)
    hub_offset = np.asarray(
        [0.0, 0.0, float(rotor["hub_above_cg_m"])], dtype=float
    )
    nominal_cg = -hub_offset
    geometry = moving_tilted_disk_geometry(
        body_state,
        nominal_cg,
        hub_offset,
        rotor_axis_body,
        radii_m,
        int(rotor["blades"]),
        0.0,
        float(rotor["rpm"]) * 2.0 * math.pi / 60.0,
    )
    return geometry, nominal_cg


def solve_physical_hover_trim(
    rotor: Mapping[str, Any],
    *,
    density_kgpm3: float,
    gravity_mps2: float,
    radial_elements: int,
    rotor_axis_body: Sequence[float] = (0.0, 0.0, 1.0),
    collective_bounds_deg: Tuple[float, float] = (-5.0, 35.0),
) -> HoverTrim:
    """Solve T=mg using the same nonlinear blade-element model as coupling."""

    radius = 0.5 * float(rotor["diameter_m"])
    area = math.pi * radius ** 2
    weight = float(rotor["mass_kg"]) * float(gravity_mps2)
    induced = physical_hover_induced_velocity(weight, density_kgpm3, radius)
    axis = np.asarray(rotor_axis_body, dtype=float)
    axis /= np.linalg.norm(axis)
    trial_base = BladeElementParameters.from_mapping(
        rotor,
        prandtl_tip_loss=True,
        prandtl_root_loss=True,
        domain_limits=AerodynamicDomainLimits(action="report", reverse_flow_action="report"),
    )
    radii, widths = radial_element_centres_and_widths(trial_base, radial_elements)
    geometry, nominal_cg_relative = _hover_geometry(rotor, radii, axis)
    blade_azimuth = 2.0 * math.pi * np.arange(trial_base.blade_count) / trial_base.blade_count
    tangential = (
        trial_base.omega_radps
        * radii[None, :, None]
        * geometry.tangential_world[:, None, :]
    )
    zeros = np.zeros_like(geometry.points_world_m)

    def evaluate(collective_deg: float) -> Tuple[BladeElementParameters, BladeElementLoads]:
        parameters = BladeElementParameters.from_mapping(
            rotor,
            collective_deg=float(collective_deg),
            prandtl_tip_loss=True,
            prandtl_root_loss=True,
            domain_limits=AerodynamicDomainLimits(action="report", reverse_flow_action="report"),
        )
        loads = integrate_blade_element_loads(
            parameters,
            geometry.points_world_m,
            zeros,
            density_kgpm3,
            radii,
            widths,
            geometry.radial_world,
            geometry.tangential_world,
            geometry.normal_world,
            geometry.hub_world_m,
            nominal_cg_relative,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            tangential,
            induced,
            blade_azimuth,
        )
        return parameters, loads

    low, high = (float(collective_bounds_deg[0]), float(collective_bounds_deg[1]))
    _, low_loads = evaluate(low)
    _, high_loads = evaluate(high)
    low_residual = low_loads.thrust_N - weight
    high_residual = high_loads.thrust_N - weight
    if low_residual > 0.0 or high_residual < 0.0:
        raise CouplingConfigurationError(
            "hover collective bracket failed for {}: residuals {:.6g}, {:.6g} N".format(
                rotor["name"], low_residual, high_residual
            )
        )
    final_parameters = trial_base
    final_loads = low_loads
    for _ in range(80):
        middle = 0.5 * (low + high)
        parameters, loads = evaluate(middle)
        residual = loads.thrust_N - weight
        final_parameters, final_loads = parameters, loads
        if abs(residual) <= max(weight * 1e-10, 1e-6):
            low = high = middle
            break
        if residual >= 0.0:
            high = middle
        else:
            low = middle
    collective = 0.5 * (low + high)
    final_parameters, final_loads = evaluate(collective)
    reference_lambda = induced / (final_parameters.omega_radps * radius)
    return HoverTrim(
        rotor_name=str(rotor["name"]),
        collective_input_deg=float(rotor["collective_deg"]),
        collective_trim_deg=collective,
        weight_N=weight,
        induced_velocity_mps=induced,
        reference_lambda=reference_lambda,
        disk_loading_Npm2=weight / area,
        loads=final_loads,
        alpha_domain_violation_count=final_loads.diagnostics.alpha_domain_violation_count,
        cl_saturation_count=final_loads.diagnostics.cl_saturation_count,
        relative_thrust_residual=(final_loads.thrust_N - weight) / weight,
    )


def _body_parameters(
    rotor: Mapping[str, Any], simulation: Mapping[str, Any], body_config: Mapping[str, Any]
) -> RigidBody3DOFParameters:
    tau_z = float(simulation.get("heave_damping_time_s", 3.0))
    tau_roll = float(simulation.get("roll_damping_time_s", 1.5))
    tau_pitch = float(simulation.get("pitch_damping_time_s", 1.8))
    if min(tau_z, tau_roll, tau_pitch) <= 0.0:
        raise CouplingConfigurationError("legacy damping time constants must be positive")
    return RigidBody3DOFParameters.from_mapping(
        rotor,
        vertical_damping_N_per_mps=float(rotor["mass_kg"]) / tau_z,
        angular_damping_body_Nms=np.asarray(
            [float(rotor["Ixx_kgm2"]) / tau_roll, float(rotor["Iyy_kgm2"]) / tau_pitch, 0.0]
        ),
        gravity_mps2=float(body_config["gravity_mps2"]),
        maximum_abs_attitude_deg=float(body_config.get("maximum_abs_attitude_deg", 12.0)),
        maximum_abs_heave_m=float(body_config.get("maximum_abs_heave_m", 1.0)),
        maximum_abs_heave_speed_mps=float(body_config.get("maximum_abs_heave_speed_mps", 5.0)),
    )


def _evaluate_aerodynamics(
    model: RotorStudyModel,
    flow: Any,
    relative_time_s: float,
    combined_state: np.ndarray,
    retained_mode_count: Optional[int],
    mode_indices_zero_based: Optional[Sequence[int]] = None,
) -> Tuple[BladeElementLoads, Any, np.ndarray, float, np.ndarray, np.ndarray]:
    body_state = np.asarray(combined_state[:6], dtype=float)
    delta_inflow = np.asarray(combined_state[6:9], dtype=float)
    phase = model.initial_phase_rad + model.blade_parameters.omega_radps * float(relative_time_s)
    geometry = moving_tilted_disk_geometry(
        body_state,
        model.nominal_cg_world_m,
        model.body_parameters.hub_from_cg_body_m,
        model.rotor_axis_body,
        model.radii_m,
        model.blade_parameters.blade_count,
        phase,
        model.blade_parameters.omega_radps,
    )
    if retained_mode_count is None and mode_indices_zero_based is None:
        cfd = flow.velocity_at(relative_time_s, geometry.points_world_m)
    else:
        cfd = flow.velocity_at(
            relative_time_s,
            geometry.points_world_m,
            retained_mode_count=retained_mode_count,
            mode_indices=mode_indices_zero_based,
        )
    blade_azimuth = phase + 2.0 * math.pi * np.arange(
        model.blade_parameters.blade_count
    ) / model.blade_parameters.blade_count
    azimuth_grid = np.broadcast_to(blade_azimuth[:, None], geometry.points_world_m.shape[:2])
    radius_ratio = np.broadcast_to(
        model.radii_m[None, :] / model.blade_parameters.radius_m,
        geometry.points_world_m.shape[:2],
    )
    induced = model.inflow_model.dimensional_induced_velocity(
        delta_inflow, radius_ratio, azimuth_grid
    )
    rotation = rotation_body_to_world(float(body_state[2]), float(body_state[3]))
    omega_body = body_angular_velocity(
        float(body_state[2]), float(body_state[4]), float(body_state[5])
    )
    omega_world = np.dot(rotation, omega_body)
    cg_world = model.nominal_cg_world_m + np.array([0.0, 0.0, body_state[0]])
    translation_world = np.array([0.0, 0.0, body_state[1]])
    tangential = (
        model.blade_parameters.omega_radps
        * model.radii_m[None, :, None]
        * geometry.tangential_world[:, None, :]
    )
    loads = integrate_blade_element_loads(
        model.blade_parameters,
        geometry.points_world_m,
        cfd,
        model.density_kgpm3,
        model.radii_m,
        model.widths_m,
        geometry.radial_world,
        geometry.tangential_world,
        geometry.normal_world,
        geometry.hub_world_m,
        cg_world,
        translation_world,
        omega_world,
        tangential,
        induced,
        blade_azimuth,
    )
    point_body_motion = translation_world[None, None, :] + np.cross(
        np.broadcast_to(omega_world, geometry.points_world_m.shape),
        geometry.points_world_m - cg_world[None, None, :],
    )
    relative_external_air = cfd - point_body_motion
    area_weights = np.broadcast_to(
        (model.radii_m * model.widths_m)[None, :],
        geometry.points_world_m.shape[:2],
    )
    delta_mu_z = disk_average_delta_mu_z(
        relative_external_air,
        geometry.normal_world,
        model.inflow_model.tip_speed_mps,
        weights=area_weights,
    )
    hub_moment_disk = _disk_components(
        loads.hub_aerodynamic_moment_world_Nm, geometry
    )
    return loads, geometry, cfd, delta_mu_z, hub_moment_disk, induced


def _periodic_mean_flow_samples(
    model: RotorStudyModel,
    mean_flow: Any,
    phase_samples: int,
    delta_inflow: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Sample the frozen mean field over one blade-passage arrangement cycle."""

    samples = int(phase_samples)
    if samples < 8:
        raise CouplingConfigurationError("mean-flow trim requires at least 8 phase samples")
    blade_count = model.blade_parameters.blade_count
    passage_angle = 2.0 * math.pi / blade_count
    phase_offsets = (np.arange(samples, dtype=float) + 0.5) * passage_angle / samples
    state = np.zeros(9, dtype=float)
    state[6:9] = np.asarray(delta_inflow, dtype=float)
    forces_body = np.empty((samples, 3), dtype=float)
    transmitted_body = np.empty((samples, 3), dtype=float)
    hub_aero_body = np.empty((samples, 3), dtype=float)
    cg_moments_body = np.empty((samples, 3), dtype=float)
    inflow_equation_residual = np.empty((samples, 3), dtype=float)
    inflow_derivative = np.empty((samples, 3), dtype=float)
    thrust = np.empty(samples, dtype=float)
    shaft_torque = np.empty(samples, dtype=float)
    shaft_power = np.empty(samples, dtype=float)
    engine_power = np.empty(samples, dtype=float)
    delta_mu_z_values = np.empty(samples, dtype=float)
    alpha_violations = np.empty(samples, dtype=int)
    cl_saturations = np.empty(samples, dtype=int)

    for index, phase_offset in enumerate(phase_offsets):
        relative_time_s = phase_offset / model.blade_parameters.omega_radps
        loads, _, _, delta_mu_z, moment_disk, _ = _evaluate_aerodynamics(
            model, mean_flow, relative_time_s, state, retained_mode_count=0
        )
        force_body = _world_to_body(loads.hub_force_world_N, state[:6])
        transmitted = _world_to_body(loads.hub_axis_moment_world_Nm, state[:6])
        hub_aero = _world_to_body(loads.hub_aerodynamic_moment_world_Nm, state[:6])
        derivative = model.inflow_model.rhs(
            relative_time_s,
            state[6:9],
            model.inflow_model.coefficient_perturbation_from_total_loads(
                loads.thrust_N, moment_disk
            ),
            delta_mu_z,
        )
        forces_body[index] = force_body
        transmitted_body[index] = transmitted
        hub_aero_body[index] = hub_aero
        cg_moments_body[index] = (
            np.cross(model.body_parameters.hub_from_cg_body_m, force_body)
            + transmitted
        )
        inflow_derivative[index] = derivative
        # This is c_eff-K*delta_lambda, recovered from M*lambda_dot/Omega.
        inflow_equation_residual[index] = (
            model.inflow_model.apparent_mass_matrix @ derivative
            / model.blade_parameters.omega_radps
        )
        thrust[index] = loads.thrust_N
        shaft_torque[index] = loads.shaft_torque_Nm
        shaft_power[index] = loads.shaft_power_W
        engine_power[index] = loads.engine_power_W
        delta_mu_z_values[index] = delta_mu_z
        alpha_violations[index] = loads.diagnostics.alpha_domain_violation_count
        cl_saturations[index] = loads.diagnostics.cl_saturation_count

    return {
        "force_body_N": forces_body,
        "transmitted_hub_moment_body_Nm": transmitted_body,
        "hub_aero_moment_body_Nm": hub_aero_body,
        "moment_cg_body_Nm": cg_moments_body,
        "inflow_equation_residual": inflow_equation_residual,
        "inflow_derivative_per_s": inflow_derivative,
        "thrust_N": thrust,
        "shaft_torque_Nm": shaft_torque,
        "shaft_power_W": shaft_power,
        "engine_power_W": engine_power,
        "delta_mu_z": delta_mu_z_values,
        "alpha_domain_violation_count": alpha_violations,
        "cl_saturation_count": cl_saturations,
    }


def solve_azimuth_mean_hover_trim(
    model: RotorStudyModel,
    mean_flow: Any,
    *,
    phase_samples: int = 72,
    collective_bounds_deg: Tuple[float, float] = (-5.0, 35.0),
    maximum_delta_lambda_fraction: float = 0.75,
    scaled_tolerance: float = 1.0e-8,
) -> Dict[str, Any]:
    """Solve mean(Fz)=weight and mean Pitt--Peters forcing=0.

    Collective and the three inflow perturbation coefficients are unknown.
    The body remains fixed; horizontal translation and yaw are constrained.
    The still-air physical trim stored on ``model`` is only the initial guess.
    """

    samples = int(phase_samples)
    if samples < 8:
        raise CouplingConfigurationError("mean-flow trim requires at least 8 phase samples")
    fraction = float(maximum_delta_lambda_fraction)
    if not 0.0 < fraction < 1.0:
        raise CouplingConfigurationError(
            "maximum_delta_lambda_fraction must lie in (0,1)"
        )
    lambda_star = model.inflow_model.lambda_star
    state_limit = min(
        fraction,
        float(model.inflow_model.parameters.maximum_delta_lambda_fraction),
    ) * lambda_star
    lower = np.array(
        [float(collective_bounds_deg[0])] + [-state_limit] * 3, dtype=float
    )
    upper = np.array(
        [float(collective_bounds_deg[1])] + [state_limit] * 3, dtype=float
    )
    unknown = np.array(
        [model.trim.collective_trim_deg, 0.0, 0.0, 0.0], dtype=float
    )
    weight = model.trim.weight_N
    ct_reference = model.inflow_model.reference_thrust_coefficient
    evaluations = 0

    def evaluate_unknown(value: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        nonlocal evaluations
        evaluations += 1
        blade = replace(
            model.blade_parameters,
            collective_deg=float(value[0]),
            cyclic_cos_deg=0.0,
            cyclic_sin_deg=0.0,
        )
        candidate = replace(model, blade_parameters=blade)
        sampled = _periodic_mean_flow_samples(
            candidate, mean_flow, samples, value[1:4]
        )
        mean_force = np.mean(sampled["force_body_N"], axis=0)
        mean_inflow = np.mean(sampled["inflow_equation_residual"], axis=0)
        scaled = np.concatenate(
            ([float((mean_force[2] - weight) / weight)], mean_inflow / ct_reference)
        )
        return np.asarray(scaled, dtype=float), sampled

    residual, sampled = evaluate_unknown(unknown)
    success = False
    message = "maximum iterations reached"
    iterations = 0
    for iteration in range(40):
        iterations = iteration + 1
        if float(np.max(np.abs(residual))) <= float(scaled_tolerance):
            success = True
            message = "scaled residual tolerance satisfied"
            break
        increments = np.array(
            [1.0e-3, 2.0e-5 * lambda_star, 2.0e-5 * lambda_star, 2.0e-5 * lambda_star],
            dtype=float,
        )
        jacobian = np.empty((4, 4), dtype=float)
        for column in range(4):
            plus = unknown.copy()
            minus = unknown.copy()
            plus[column] = min(upper[column], plus[column] + increments[column])
            minus[column] = max(lower[column], minus[column] - increments[column])
            if plus[column] == minus[column]:
                raise CouplingRunError("mean-flow trim variable is fixed at both bounds")
            plus_residual, _ = evaluate_unknown(plus)
            minus_residual, _ = evaluate_unknown(minus)
            jacobian[:, column] = (
                (plus_residual - minus_residual) / (plus[column] - minus[column])
            )
        step, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=1.0e-10)
        if rank < 4 or not np.all(np.isfinite(step)):
            message = "finite-difference trim Jacobian is rank deficient"
            break
        step[0] = float(np.clip(step[0], -3.0, 3.0))
        step[1:] = np.clip(step[1:], -0.20 * lambda_star, 0.20 * lambda_star)
        old_norm = float(np.linalg.norm(residual))
        accepted = False
        for exponent in range(12):
            trial = np.clip(unknown + (0.5 ** exponent) * step, lower, upper)
            trial_residual, trial_sampled = evaluate_unknown(trial)
            if float(np.linalg.norm(trial_residual)) < old_norm:
                unknown = trial
                residual = trial_residual
                sampled = trial_sampled
                accepted = True
                break
        if not accepted:
            message = "bounded line search could not reduce the trim residual"
            break

    # Re-evaluate exactly at the reported state, including all physical outputs.
    residual, sampled = evaluate_unknown(unknown)
    if float(np.max(np.abs(residual))) <= float(scaled_tolerance):
        success = True
        message = "scaled residual tolerance satisfied"
    mean_force = np.mean(sampled["force_body_N"], axis=0)
    mean_moment = np.mean(sampled["moment_cg_body_Nm"], axis=0)
    mean_inflow_residual = np.mean(sampled["inflow_equation_residual"], axis=0)
    mean_inflow_derivative = np.mean(sampled["inflow_derivative_per_s"], axis=0)
    return {
        "rotor_name": str(model.rotor["name"]),
        "method": "azimuth-mean fixed-body collective/inflow equilibrium",
        "periodic_orbit_shooting": False,
        "constrained_body_coordinates": ["x_translation", "y_translation", "yaw"],
        "phase_samples": samples,
        "blade_passage_angle_rad": 2.0 * math.pi / model.blade_parameters.blade_count,
        "success": bool(success),
        "solver_message": message,
        "iterations": iterations,
        "function_evaluations": evaluations,
        "scaled_tolerance": float(scaled_tolerance),
        "maximum_abs_scaled_residual": float(np.max(np.abs(residual))),
        "collective_deg": float(unknown[0]),
        "cyclic_cos_deg": 0.0,
        "cyclic_sin_deg": 0.0,
        "delta_inflow_state": unknown[1:4].tolist(),
        "reference_lambda": float(lambda_star),
        "uniform_total_lambda": float(lambda_star + unknown[1]),
        "uniform_induced_velocity_mps": float(
            model.inflow_model.tip_speed_mps * (lambda_star + unknown[1])
        ),
        "mean_force_body_N": mean_force.tolist(),
        "force_minus_weight_body_N": (
            mean_force - np.array([0.0, 0.0, weight], dtype=float)
        ).tolist(),
        "force_peak_to_peak_body_N": np.ptp(sampled["force_body_N"], axis=0).tolist(),
        "mean_hub_aero_moment_body_Nm": np.mean(
            sampled["hub_aero_moment_body_Nm"], axis=0
        ).tolist(),
        "mean_transmitted_hub_moment_body_Nm": np.mean(
            sampled["transmitted_hub_moment_body_Nm"], axis=0
        ).tolist(),
        "mean_cg_moment_body_Nm": mean_moment.tolist(),
        "cg_moment_peak_to_peak_body_Nm": np.ptp(
            sampled["moment_cg_body_Nm"], axis=0
        ).tolist(),
        "mean_inflow_equation_residual": mean_inflow_residual.tolist(),
        "mean_inflow_derivative_per_s": mean_inflow_derivative.tolist(),
        "mean_thrust_N": float(np.mean(sampled["thrust_N"])),
        "mean_shaft_torque_Nm": float(np.mean(sampled["shaft_torque_Nm"])),
        "mean_shaft_power_W": float(np.mean(sampled["shaft_power_W"])),
        "mean_engine_power_W": float(np.mean(sampled["engine_power_W"])),
        "mean_delta_mu_z": float(np.mean(sampled["delta_mu_z"])),
        "delta_mu_z_peak_to_peak": float(np.ptp(sampled["delta_mu_z"])),
        "maximum_alpha_domain_violation_count_per_phase": int(
            np.max(sampled["alpha_domain_violation_count"])
        ),
        "maximum_cl_saturation_count_per_phase": int(
            np.max(sampled["cl_saturation_count"])
        ),
        "collective_at_bound": bool(
            np.isclose(unknown[0], lower[0], atol=1.0e-7)
            or np.isclose(unknown[0], upper[0], atol=1.0e-7)
        ),
        "delta_inflow_at_bound": bool(
            np.any(np.isclose(unknown[1:], lower[1:], atol=1.0e-9))
            or np.any(np.isclose(unknown[1:], upper[1:], atol=1.0e-9))
        ),
    }


def prepare_rotor_model(
    rotor: Mapping[str, Any],
    simulation: Mapping[str, Any],
    context: StudyContext,
    mean_flow: Any,
) -> RotorStudyModel:
    body_config = context.config["body_response"]
    rotor_config = context.config["rotor_model"]
    inflow_config = context.config["dynamic_inflow"]
    density = float(context.config["flow_model"]["air_density_kgpm3"])
    axis = np.asarray(context.config["coordinates"]["rotor_axis_body"], dtype=float)
    axis /= np.linalg.norm(axis)
    radial_elements = int(rotor_config["radial_elements"])
    trim = solve_physical_hover_trim(
        rotor,
        density_kgpm3=density,
        gravity_mps2=float(body_config["gravity_mps2"]),
        radial_elements=radial_elements,
        rotor_axis_body=axis,
    )
    blade = BladeElementParameters.from_mapping(
        rotor,
        collective_deg=trim.collective_trim_deg,
        prandtl_tip_loss=bool(rotor_config.get("prandtl_root_tip_loss", True)),
        prandtl_root_loss=bool(rotor_config.get("prandtl_root_tip_loss", True)),
        # The static Cl saturation defines the extrapolative section model.
        # Every out-of-envelope element is recorded in the CSV; no clipping of
        # alpha, Mach or Reynolds takes place.
        domain_limits=AerodynamicDomainLimits(action="report", reverse_flow_action="report"),
    )
    radii, widths = radial_element_centres_and_widths(blade, radial_elements)
    body_parameters = _body_parameters(rotor, simulation, body_config)
    controller_config = body_config.get("hover_hold_controller", {})
    controller_enabled = bool(controller_config.get("enabled", False))
    if controller_enabled:
        heave_omega = 2.0 * math.pi * float(
            controller_config["heave_natural_frequency_hz"]
        )
        roll_omega = 2.0 * math.pi * float(
            controller_config["roll_natural_frequency_hz"]
        )
        pitch_omega = 2.0 * math.pi * float(
            controller_config["pitch_natural_frequency_hz"]
        )
        heave_zeta = float(controller_config["heave_damping_ratio"])
        attitude_zeta = float(controller_config["attitude_damping_ratio"])
        if min(heave_omega, roll_omega, pitch_omega, heave_zeta, attitude_zeta) <= 0.0:
            raise CouplingConfigurationError("hover-hold frequencies and damping ratios must be positive")
        vertical_stiffness = body_parameters.mass_kg * heave_omega ** 2
        target_vertical_rate = 2.0 * heave_zeta * body_parameters.mass_kg * heave_omega
        vertical_rate_gain = max(
            0.0, target_vertical_rate - body_parameters.vertical_damping_N_per_mps
        )
        inertia_diagonal = np.diag(body_parameters.inertia_body_kgm2)
        angular_stiffness = np.array(
            [
                inertia_diagonal[0] * roll_omega ** 2,
                inertia_diagonal[1] * pitch_omega ** 2,
                0.0,
            ],
            dtype=float,
        )
        target_angular_rate = np.array(
            [
                2.0 * attitude_zeta * inertia_diagonal[0] * roll_omega,
                2.0 * attitude_zeta * inertia_diagonal[1] * pitch_omega,
                0.0,
            ],
            dtype=float,
        )
        angular_rate_gain = np.maximum(
            0.0, target_angular_rate - body_parameters.angular_damping_body_Nms
        )
    else:
        heave_omega = roll_omega = pitch_omega = 0.0
        heave_zeta = attitude_zeta = 0.0
        vertical_stiffness = 0.0
        vertical_rate_gain = 0.0
        angular_stiffness = np.zeros(3, dtype=float)
        angular_rate_gain = np.zeros(3, dtype=float)
    controller_metadata = {
        "enabled": controller_enabled,
        "model": controller_config.get("model", "none"),
        "parameter_status": controller_config.get("parameter_status", "not_applicable"),
        "implementation": (
            "idealized generalized-force PD perturbation hover hold; the constant mean-flow "
            "trim reaction remains separate and no case-specific mean is removed"
            if controller_enabled
            else "disabled"
        ),
        "heave_natural_frequency_radps": heave_omega,
        "roll_natural_frequency_radps": roll_omega,
        "pitch_natural_frequency_radps": pitch_omega,
        "heave_damping_ratio": heave_zeta,
        "attitude_damping_ratio": attitude_zeta,
        "vertical_stiffness_Npm": vertical_stiffness,
        "additional_vertical_rate_gain_Ns_pm": vertical_rate_gain,
        "angular_stiffness_Nm_per_rad": angular_stiffness.tolist(),
        "additional_angular_rate_gain_Nms": angular_rate_gain.tolist(),
        "actuator_saturation": controller_config.get("actuator_saturation", "not_modelled"),
    }
    nominal_cg = context.hub_world_m - body_parameters.hub_from_cg_body_m
    hover_geometry, _ = _hover_geometry(rotor, radii, axis)
    hover_moment_disk = _disk_components(
        trim.loads.hub_aerodynamic_moment_world_Nm, hover_geometry
    )
    inflow = NearHoverPittPetersInflow(
        DynamicInflowParameters(
            radius_m=blade.radius_m,
            omega_radps=blade.omega_radps,
            density_kgpm3=density,
            reference_thrust_N=trim.weight_N,
            reference_hub_aero_moment_disk_Nm=hover_moment_disk,
            maximum_delta_lambda_fraction=float(
                inflow_config.get("maximum_delta_lambda_fraction", 0.75)
            ),
            maximum_delta_mu_z_fraction=float(
                inflow_config.get("maximum_delta_mu_z_fraction", 1.0)
            ),
        )
    )
    provisional_model = RotorStudyModel(
        rotor=dict(rotor),
        blade_parameters=blade,
        body_parameters=body_parameters,
        inflow_model=inflow,
        radii_m=radii,
        widths_m=widths,
        rotor_axis_body=axis,
        nominal_cg_world_m=nominal_cg,
        initial_phase_rad=math.radians(float(simulation.get("initial_phase_deg", 0.0))),
        density_kgpm3=density,
        trim=trim,
        balance_other_force_body_N=np.zeros(3),
        balance_other_moment_body_Nm=np.zeros(3),
        balance_definition="pending",
        initial_inflow_state=np.zeros(3, dtype=float),
        mean_flow_trim_audit={},
        hover_hold_vertical_stiffness_Npm=vertical_stiffness,
        hover_hold_vertical_rate_gain_Ns_pm=vertical_rate_gain,
        hover_hold_angular_stiffness_Nm_per_rad=angular_stiffness,
        hover_hold_angular_rate_gain_Nms=angular_rate_gain,
        hover_hold_metadata=controller_metadata,
    )
    mean_trim = solve_azimuth_mean_hover_trim(
        provisional_model,
        mean_flow,
        phase_samples=int(body_config.get("mean_flow_trim_phase_samples", 72)),
        maximum_delta_lambda_fraction=float(
            inflow_config.get("maximum_delta_lambda_fraction", 0.75)
        ),
    )
    if not bool(mean_trim["success"]):
        raise CouplingRunError(
            "azimuth-mean NATO-GD hover trim failed for {}: {}".format(
                rotor["name"], mean_trim["solver_message"]
            )
        )
    if bool(mean_trim["collective_at_bound"]) or bool(mean_trim["delta_inflow_at_bound"]):
        raise CouplingRunError(
            "azimuth-mean NATO-GD hover trim reached a validity bound for {}".format(
                rotor["name"]
            )
        )
    provisional_model.blade_parameters = replace(
        blade,
        collective_deg=float(mean_trim["collective_deg"]),
        cyclic_cos_deg=0.0,
        cyclic_sin_deg=0.0,
    )
    provisional_model.initial_inflow_state = np.asarray(
        mean_trim["delta_inflow_state"], dtype=float
    )
    force_body = np.asarray(mean_trim["mean_force_body_N"], dtype=float)
    mean_cg_moment = np.asarray(mean_trim["mean_cg_moment_body_Nm"], dtype=float)
    required_force_body = np.array([0.0, 0.0, trim.weight_N])
    other_force = required_force_body - force_body
    other_moment = -mean_cg_moment
    provisional_model.balance_other_force_body_N = other_force
    provisional_model.balance_other_moment_body_Nm = other_moment
    provisional_model.mean_flow_trim_audit = mean_trim
    provisional_model.balance_definition = (
        "constant fixed-body reaction balancing the residual azimuth-mean force/moment "
        "after solving collective plus three Pitt-Peters states in the RVMD time-mean "
        "NATO-GD field; computed once and reused unchanged in every case"
    )
    return provisional_model


def _hover_hold_generalized_loads(
    model: RotorStudyModel, body_state: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the explicitly declared perturbation hover-hold PD loads."""

    state = np.asarray(body_state, dtype=float)
    if state.shape != (6,):
        raise CouplingRunError("body_state must have six entries")
    force = np.zeros(3, dtype=float)
    force[2] = -(
        model.hover_hold_vertical_stiffness_Npm * state[0]
        + model.hover_hold_vertical_rate_gain_Ns_pm * state[1]
    )
    angles = np.array([state[2], state[3], 0.0], dtype=float)
    rates = np.array([state[4], state[5], 0.0], dtype=float)
    moment = -(
        model.hover_hold_angular_stiffness_Nm_per_rad * angles
        + model.hover_hold_angular_rate_gain_Nms * rates
    )
    return force, moment


def combined_rhs(
    model: RotorStudyModel,
    flow: Any,
    retained_mode_count: Optional[int],
    diagnostics: DynamicsDiagnostics,
    mode_indices_zero_based: Optional[Sequence[int]] = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(relative_time_s: float, state: np.ndarray) -> np.ndarray:
        loads, geometry, _, delta_mu_z, hub_moment_disk, _ = _evaluate_aerodynamics(
            model,
            flow,
            relative_time_s,
            state,
            retained_mode_count,
            mode_indices_zero_based,
        )
        body_state = state[:6]
        force_body = _world_to_body(loads.hub_force_world_N, body_state)
        hub_aero_body = _world_to_body(loads.hub_aerodynamic_moment_world_Nm, body_state)
        transmitted_body = _world_to_body(loads.hub_axis_moment_world_Nm, body_state)
        controller_force, controller_moment = _hover_hold_generalized_loads(
            model, body_state
        )
        body_loads = BodyLoads(
            rotor_force_body_N=force_body,
            hub_aero_moment_body_Nm=hub_aero_body,
            transmitted_hub_moment_body_Nm=transmitted_body,
            other_force_body_N=model.balance_other_force_body_N + controller_force,
            other_moment_body_Nm=model.balance_other_moment_body_Nm + controller_moment,
        )
        body_derivative = rigid_body_3dof_rhs(
            relative_time_s, body_state, model.body_parameters, body_loads, diagnostics
        )
        delta_coefficients = model.inflow_model.coefficient_perturbation_from_total_loads(
            loads.thrust_N, hub_moment_disk
        )
        inflow_derivative = model.inflow_model.rhs(
            relative_time_s, state[6:9], delta_coefficients, delta_mu_z
        )
        return np.concatenate((body_derivative, inflow_derivative))

    return rhs


def _row_from_state_and_loads(
    model: RotorStudyModel,
    definition: CaseDefinition,
    relative_time_s: float,
    state: np.ndarray,
    loads: BladeElementLoads,
    geometry: Any,
    delta_mu_z: float,
    hub_moment_disk: np.ndarray,
    induced: np.ndarray,
) -> Dict[str, Any]:
    body = state[:6]
    delta = state[6:9]
    diagnostics = loads.diagnostics
    controller_force, controller_moment = _hover_hold_generalized_loads(model, body)
    return {
        "rotor_name": model.rotor["name"],
        "rotor_diameter_m": model.rotor["diameter_m"],
        "case_id": definition.case_id,
        "flow_kind": definition.flow_kind,
        "stage_index": "" if definition.stage_index is None else definition.stage_index,
        "retained_mode_count": "" if definition.retained_mode_count is None else definition.retained_mode_count,
        "cutoff_frequency_hz": "" if definition.cutoff_frequency_hz is None else definition.cutoff_frequency_hz,
        "cutoff_St": "" if definition.cutoff_St is None else definition.cutoff_St,
        "sensitivity_case_kind": (
            "" if definition.sensitivity_case_kind is None else definition.sensitivity_case_kind
        ),
        "sensitivity_mode_index_one_based": (
            ""
            if definition.sensitivity_mode_index_one_based is None
            else definition.sensitivity_mode_index_one_based
        ),
        "time_s": relative_time_s,
        "phase_deg": math.degrees(
            model.initial_phase_rad + model.blade_parameters.omega_radps * relative_time_s
        ) % 360.0,
        "Fx_N": loads.hub_force_world_N[0],
        "Fy_N": loads.hub_force_world_N[1],
        "Fz_N": loads.hub_force_world_N[2],
        "lift_N": loads.hub_force_world_N[2],
        "thrust_N": loads.thrust_N,
        "hub_aero_Mx_Nm": loads.hub_aerodynamic_moment_world_Nm[0],
        "hub_aero_My_Nm": loads.hub_aerodynamic_moment_world_Nm[1],
        "hub_aero_Mz_Nm": loads.hub_aerodynamic_moment_world_Nm[2],
        "hub_disk_M1_Nm": hub_moment_disk[0],
        "hub_disk_M2_Nm": hub_moment_disk[1],
        "hub_disk_Mn_Nm": hub_moment_disk[2],
        "transmitted_hub_Mx_Nm": loads.hub_axis_moment_world_Nm[0],
        "transmitted_hub_My_Nm": loads.hub_axis_moment_world_Nm[1],
        "transmitted_hub_Mz_Nm": loads.hub_axis_moment_world_Nm[2],
        "shaft_torque_Nm": loads.shaft_torque_Nm,
        "shaft_power_W": loads.shaft_power_W,
        "engine_power_W": loads.engine_power_W,
        "hover_hold_Fz_body_N": controller_force[2],
        "hover_hold_Mx_body_Nm": controller_moment[0],
        "hover_hold_My_body_Nm": controller_moment[1],
        "heave_m": body[0],
        "heave_velocity_mps": body[1],
        "roll_deg": math.degrees(body[2]),
        "pitch_deg": math.degrees(body[3]),
        "roll_rate_degps": math.degrees(body[4]),
        "pitch_rate_degps": math.degrees(body[5]),
        "delta_lambda_0": delta[0],
        "delta_lambda_c": delta[1],
        "delta_lambda_s": delta[2],
        "lambda_total_uniform": model.inflow_model.lambda_star + delta[0],
        "induced_velocity_mean_mps": float(np.mean(induced)),
        "induced_velocity_min_mps": float(np.min(induced)),
        "induced_velocity_max_mps": float(np.max(induced)),
        "delta_mu_z": delta_mu_z,
        "hub_x_m": geometry.hub_world_m[0],
        "hub_y_m": geometry.hub_world_m[1],
        "hub_z_m": geometry.hub_world_m[2],
        "rotor_normal_x": geometry.normal_world[0],
        "rotor_normal_y": geometry.normal_world[1],
        "rotor_normal_z": geometry.normal_world[2],
        "alpha_min_deg": diagnostics.alpha_min_deg,
        "alpha_max_deg": diagnostics.alpha_max_deg,
        "mach_max": diagnostics.mach_max,
        "reynolds_min": diagnostics.reynolds_min,
        "reynolds_max": diagnostics.reynolds_max,
        "prandtl_factor_min": diagnostics.prandtl_factor_min,
        "cl_saturation_count": diagnostics.cl_saturation_count,
        "reverse_flow_count": diagnostics.reverse_flow_count,
        "alpha_domain_violation_count": diagnostics.alpha_domain_violation_count,
        "mach_domain_violation_count": diagnostics.mach_domain_violation_count,
        "reynolds_domain_violation_count": diagnostics.reynolds_domain_violation_count,
    }


def output_times(end_time_s: float, interval_s: float) -> np.ndarray:
    if end_time_s <= 0.0 or interval_s <= 0.0:
        raise CouplingConfigurationError("time limits must be positive")
    count = int(math.floor(end_time_s / interval_s + 1e-12))
    values = np.arange(count + 1, dtype=float) * interval_s
    if values[-1] < end_time_s - 1e-12:
        values = np.concatenate((values, [float(end_time_s)]))
    else:
        values[-1] = float(end_time_s)
    return values


def simulate_case(
    model: RotorStudyModel,
    flow: Any,
    definition: CaseDefinition,
    *,
    end_time_s: float,
    output_interval_s: float,
    maximum_azimuth_step_deg: float,
) -> CaseResult:
    """Run one case from the identical azimuth-mean hover initial state."""

    if maximum_azimuth_step_deg <= 0.0 or maximum_azimuth_step_deg > 30.0:
        raise CouplingConfigurationError("maximum azimuth step must lie in (0,30] deg")
    maximum_step_s = math.radians(maximum_azimuth_step_deg) / model.blade_parameters.omega_radps
    times = output_times(end_time_s, output_interval_s)
    state = np.zeros(9, dtype=float)
    state[6:9] = np.asarray(model.initial_inflow_state, dtype=float)
    diagnostics = DynamicsDiagnostics()
    model.inflow_model.diagnostics = diagnostics
    rhs = combined_rhs(
        model,
        flow,
        definition.retained_mode_count,
        diagnostics,
        definition.mode_indices_zero_based,
    )
    rows: List[Dict[str, Any]] = []
    current_time = 0.0
    internal_steps = 0
    started = time.monotonic()
    for target_time in times:
        while current_time < target_time - 1e-14:
            step = min(maximum_step_s, target_time - current_time)
            state = heun_step(rhs, current_time, state, step)
            current_time += step
            internal_steps += 1
        current_time = float(target_time)
        loads, geometry, _, delta_mu_z, hub_moment_disk, induced = _evaluate_aerodynamics(
            model,
            flow,
            current_time,
            state,
            definition.retained_mode_count,
            definition.mode_indices_zero_based,
        )
        rows.append(
            _row_from_state_and_loads(
                model,
                definition,
                current_time,
                state,
                loads,
                geometry,
                delta_mu_z,
                hub_moment_disk,
                induced,
            )
        )
    return CaseResult(
        definition=definition,
        rows=rows,
        metadata={
            "initial_state_body": [0.0] * 6,
            "initial_state_delta_inflow": model.initial_inflow_state.tolist(),
            "integrator": "explicit Heun predictor-corrector",
            "maximum_internal_step_s": maximum_step_s,
            "maximum_azimuth_step_deg": maximum_azimuth_step_deg,
            "internal_step_count": internal_steps,
            "output_interval_s": output_interval_s,
            "end_time_s": end_time_s,
            "elapsed_wall_s": time.monotonic() - started,
            "dynamics_diagnostics": diagnostics.as_dict(),
            "inflow_metadata": model.inflow_model.metadata(),
            "body_metadata": model.body_parameters.metadata(),
            "balance_definition": model.balance_definition,
            "balance_other_force_body_N": model.balance_other_force_body_N.tolist(),
            "balance_other_moment_body_Nm": model.balance_other_moment_body_Nm.tolist(),
            "mean_flow_trim_audit": model.mean_flow_trim_audit,
            "hover_hold_controller": model.hover_hold_metadata,
        },
    )


def write_case_csv(path: Path, result: CaseResult) -> Path:
    if not result.rows:
        raise CouplingRunError("cannot write an empty case")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.rows[0].keys()))
        writer.writeheader()
        writer.writerows(result.rows)
    temporary.replace(path)
    metadata = dict(result.metadata)
    metadata["case_definition"] = {
        "case_id": result.definition.case_id,
        "stage_index": result.definition.stage_index,
        "retained_mode_count": result.definition.retained_mode_count,
        "flow_kind": result.definition.flow_kind,
        "cutoff_frequency_hz": result.definition.cutoff_frequency_hz,
        "cutoff_St": result.definition.cutoff_St,
        "mode_indices_zero_based": (
            None
            if result.definition.mode_indices_zero_based is None
            else list(result.definition.mode_indices_zero_based)
        ),
        "sensitivity_case_kind": result.definition.sensitivity_case_kind,
        "sensitivity_mode_index_one_based": (
            result.definition.sensitivity_mode_index_one_based
        ),
    }
    _write_json(path.with_suffix(".metadata.json"), metadata)
    return path


def read_case_csv(path: Path, definition: CaseDefinition) -> CaseResult:
    """Reload one atomic case output so resumed runs rebuild complete statistics."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise CouplingRunError("existing case CSV is empty: {}".format(path))
    metadata = _read_json(path.with_suffix(".metadata.json"))
    saved = metadata.get("case_definition")
    if not isinstance(saved, dict) or saved.get("case_id") != definition.case_id:
        raise CouplingRunError("existing case metadata disagrees with requested case: {}".format(path))
    return CaseResult(definition=definition, rows=rows, metadata=metadata)


def statistics_rows(
    result: CaseResult,
    analysis_start_s: float,
    analysis_end_s: float,
    rotor_diameter_m: float,
) -> List[Dict[str, Any]]:
    selected = [
        row for row in result.rows
        if analysis_start_s - 1e-12 <= float(row["time_s"]) <= analysis_end_s + 1e-12
    ]
    if len(selected) < 2:
        raise CouplingRunError("analysis window contains fewer than two samples")
    output: List[Dict[str, Any]] = []
    definition = result.definition
    for metric in CORE_METRICS:
        values = np.asarray([float(row[metric]) for row in selected], dtype=float)
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "rms": float(math.sqrt(float(np.mean(values ** 2)))),
            "peak_to_peak": float(np.ptp(values)),
            "max_abs": float(np.max(np.abs(values))),
        }
        for statistic in STATISTICS:
            output.append(
                {
                    "rotor_name": selected[0]["rotor_name"],
                    "rotor_diameter_m": rotor_diameter_m,
                    "case_id": definition.case_id,
                    "flow_kind": definition.flow_kind,
                    "stage_index": "" if definition.stage_index is None else definition.stage_index,
                    "retained_mode_count": "" if definition.retained_mode_count is None else definition.retained_mode_count,
                    "cutoff_frequency_hz": "" if definition.cutoff_frequency_hz is None else definition.cutoff_frequency_hz,
                    "cutoff_St": "" if definition.cutoff_St is None else definition.cutoff_St,
                    "sensitivity_case_kind": (
                        ""
                        if definition.sensitivity_case_kind is None
                        else definition.sensitivity_case_kind
                    ),
                    "sensitivity_mode_index_one_based": (
                        ""
                        if definition.sensitivity_mode_index_one_based is None
                        else definition.sensitivity_mode_index_one_based
                    ),
                    "analysis_start_s": analysis_start_s,
                    "analysis_end_s": analysis_end_s,
                    "sample_count": len(selected),
                    "metric": metric,
                    "statistic": statistic,
                    "value": stats[statistic],
                }
            )
    return output


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise CouplingRunError("cannot write empty table {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _case_definitions(flow: RVMD3DFlow, stages: Iterable[int]) -> List[CaseDefinition]:
    stage_rows = build_stage_rows(flow.frequency_hz, flow.strouhal)
    definitions = []
    for stage in stages:
        if stage < 0 or stage > flow.mode_count:
            raise CouplingConfigurationError(
                "stage index must be in [0,{}]".format(flow.mode_count)
            )
        row = stage_rows[stage]
        definitions.append(
            CaseDefinition(
                case_id=str(row["case_id"]),
                stage_index=stage,
                retained_mode_count=int(row["retained_mode_count"]),
                flow_kind="RVMD",
                cutoff_frequency_hz=row["cutoff_frequency_hz"],
                cutoff_St=row["cutoff_St"],
            )
        )
    return definitions


def _modal_sensitivity_case_definitions(
    flow: RVMD3DFlow, mode_indices_one_based: Optional[Iterable[int]] = None
) -> List[CaseDefinition]:
    """Build full/mean/single/leave-one-out cases for modal sensitivity.

    Mode subsets are explicit and therefore independent of the cumulative
    high-to-low deletion path.  This distinction is required for a normalized
    per-mode sensitivity spectrum in a nonlinear dynamics model.
    """

    if mode_indices_one_based is None:
        selected = list(range(1, flow.mode_count + 1))
    else:
        selected = sorted(set(int(value) for value in mode_indices_one_based))
    if not selected or selected[0] < 1 or selected[-1] > flow.mode_count:
        raise CouplingConfigurationError(
            "sensitivity mode indices must lie in [1,{}]".format(flow.mode_count)
        )
    width = max(3, len(str(flow.mode_count)))
    all_modes = tuple(range(flow.mode_count))
    definitions = [
        CaseDefinition(
            "sensitivity_full",
            None,
            flow.mode_count,
            "RVMD",
            float(flow.frequency_hz[-1]),
            float(flow.strouhal[-1]),
            sensitivity_case_kind="full",
        ),
        CaseDefinition(
            "sensitivity_mean",
            None,
            0,
            "RVMD",
            None,
            None,
            sensitivity_case_kind="mean",
        ),
    ]
    for one_based in selected:
        zero_based = one_based - 1
        tag = "M{:0{width}d}".format(one_based, width=width)
        frequency = float(flow.frequency_hz[zero_based])
        st = float(flow.strouhal[zero_based])
        definitions.append(
            CaseDefinition(
                "sensitivity_single_{}".format(tag),
                None,
                None,
                "RVMD",
                frequency,
                st,
                mode_indices_zero_based=(zero_based,),
                sensitivity_case_kind="single",
                sensitivity_mode_index_one_based=one_based,
            )
        )
        definitions.append(
            CaseDefinition(
                "sensitivity_loo_{}".format(tag),
                None,
                None,
                "RVMD",
                frequency,
                st,
                mode_indices_zero_based=tuple(
                    index for index in all_modes if index != zero_based
                ),
                sensitivity_case_kind="leave_one_out",
                sensitivity_mode_index_one_based=one_based,
            )
        )
    return definitions


def _resume_output_matches_request(
    metadata_path: Path,
    definition: CaseDefinition,
    model: RotorStudyModel,
    *,
    end_time_s: float,
    output_interval_s: float,
    maximum_azimuth_step_deg: float,
    run_signature_sha256: str,
) -> bool:
    """Reject stale partial/smoke outputs instead of silently reusing them."""

    try:
        metadata = _read_json(metadata_path)
        saved_case = metadata["case_definition"]
        if not isinstance(saved_case, dict):
            return False
        requested_case = {
            "case_id": definition.case_id,
            "stage_index": definition.stage_index,
            "retained_mode_count": definition.retained_mode_count,
            "flow_kind": definition.flow_kind,
            "cutoff_frequency_hz": definition.cutoff_frequency_hz,
            "cutoff_St": definition.cutoff_St,
            "mode_indices_zero_based": (
                None
                if definition.mode_indices_zero_based is None
                else list(definition.mode_indices_zero_based)
            ),
            "sensitivity_case_kind": definition.sensitivity_case_kind,
            "sensitivity_mode_index_one_based": (
                definition.sensitivity_mode_index_one_based
            ),
        }
        if saved_case != requested_case:
            return False
        scalar_checks = (
            (metadata["end_time_s"], end_time_s),
            (metadata["output_interval_s"], output_interval_s),
            (metadata["maximum_azimuth_step_deg"], maximum_azimuth_step_deg),
        )
        if any(
            not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
            for observed, expected in scalar_checks
        ):
            return False
        if metadata.get("run_signature_sha256") != run_signature_sha256:
            return False
        if not np.array_equal(
            np.asarray(metadata["initial_state_delta_inflow"], dtype=float),
            np.asarray(model.initial_inflow_state, dtype=float),
        ):
            return False
        if metadata.get("hover_hold_controller") != model.hover_hold_metadata:
            return False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _initialize_coupling_worker(
    rvmd_directory: str, cache_directory: str, dt_s: float
) -> None:
    """Open one read-only flow pair per worker process."""

    global _WORKER_RVMD_FLOW, _WORKER_BASELINE_FLOW
    _WORKER_RVMD_FLOW = RVMD3DFlow(Path(rvmd_directory))
    _WORKER_BASELINE_FLOW = NatoGD3DBaselineFlow(
        Path(cache_directory), expected_dt_s=float(dt_s)
    )


def _execute_coupling_case_worker(
    task: Mapping[str, Any],
) -> Tuple[CaseDefinition, CaseResult, Path]:
    """Run or reload one atomic case inside an isolated process."""

    definition = task["definition"]
    model = task["model"]
    if not isinstance(definition, CaseDefinition) or not isinstance(model, RotorStudyModel):
        raise CouplingRunError("coupling worker received an invalid task")
    if _WORKER_RVMD_FLOW is None or _WORKER_BASELINE_FLOW is None:
        raise CouplingRunError("coupling worker flows were not initialized")
    active_flow = (
        _WORKER_BASELINE_FLOW
        if definition.flow_kind == "original_CFD"
        else _WORKER_RVMD_FLOW
    )
    output_path = Path(str(task["output_path"]))
    metadata_path = output_path.with_suffix(".metadata.json")
    resume = bool(task["resume"])
    reusable = (
        resume
        and output_path.is_file()
        and metadata_path.is_file()
        and _resume_output_matches_request(
            metadata_path,
            definition,
            model,
            end_time_s=float(task["end_time_s"]),
            output_interval_s=float(task["output_interval_s"]),
            maximum_azimuth_step_deg=float(task["maximum_azimuth_step_deg"]),
            run_signature_sha256=str(task["run_signature_sha256"]),
        )
    )
    if reusable:
        print("skip existing {}".format(output_path.name), flush=True)
        result = read_case_csv(output_path, definition)
    else:
        if resume and (output_path.exists() or metadata_path.exists()):
            print("replace stale or incompatible {}".format(output_path.name), flush=True)
        print(
            "simulate {} {}".format(model.rotor["name"], definition.case_id),
            flush=True,
        )
        result = simulate_case(
            model,
            active_flow,
            definition,
            end_time_s=float(task["end_time_s"]),
            output_interval_s=float(task["output_interval_s"]),
            maximum_azimuth_step_deg=float(task["maximum_azimuth_step_deg"]),
        )
        result.metadata["run_signature_sha256"] = str(task["run_signature_sha256"])
        write_case_csv(output_path, result)
    return definition, result, output_path


def run_study(
    context: StudyContext,
    *,
    rotor_names: Optional[Sequence[str]] = None,
    stages: Optional[Sequence[int]] = None,
    include_modal_sensitivity: bool = False,
    sensitivity_mode_indices_one_based: Optional[Sequence[int]] = None,
    include_original_cfd: bool = True,
    resume: bool = True,
    end_time_override_s: Optional[float] = None,
    maximum_azimuth_step_override_deg: Optional[float] = None,
    allow_nonconverged_rvmd: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    initialise_output_tree(context)
    write_context_provenance(context)
    simulation, rotors = load_rotor_mappings(context)
    if rotor_names:
        requested = set(rotor_names)
        rotors = [rotor for rotor in rotors if rotor["name"] in requested]
        missing = requested - {rotor["name"] for rotor in rotors}
        if missing:
            raise CouplingConfigurationError("unknown rotors: {}".format(sorted(missing)))
    rvmd_directory = context.output_dir / "02_rvmd"
    rvmd_flow = RVMD3DFlow(rvmd_directory)
    if rvmd_flow.mode_count != context.mode_count:
        raise CouplingConfigurationError(
            "configured mode_count={} disagrees with the RVMD artifact mode_count={}".format(
                context.mode_count, rvmd_flow.mode_count
            )
        )
    if not math.isclose(rvmd_flow.dt_s, context.dt_s, rel_tol=0.0, abs_tol=1.0e-15):
        raise CouplingConfigurationError(
            "configured dt={:.17g} s disagrees with the RVMD artifact dt={:.17g} s".format(
                context.dt_s, rvmd_flow.dt_s
            )
        )
    run_signature = _artifact_signature(
        [
            context.config_path,
            context.rotor_config_path,
            context.dynamic_overrides_path,
            Path(__file__),
            Path(__file__).with_name("nonlinear_blade_element.py"),
            Path(__file__).with_name("nonlinear_rotorcraft_dynamics.py"),
            Path(__file__).with_name("rvmd_3d_flow.py"),
            context.cache_dir / "metadata.json",
            context.cache_dir / "manifest.json",
            rvmd_directory / "metadata.json",
            rvmd_directory / "temporal_model.npz",
            rvmd_directory / "mean_field_physical.npy",
            rvmd_directory / "spatial_modes_physical.npy",
        ]
    )
    rvmd_info = rvmd_flow.metadata.get("rvmd_info", {})
    if (
        context.config["production_gate"].get("require_rvmd_convergence_or_explicit_override", True)
        and not bool(rvmd_info.get("converged", False))
        and not allow_nonconverged_rvmd
    ):
        raise CouplingConfigurationError(
            "RVMD did not satisfy its convergence tolerance; an explicit nonconverged override "
            "is required and the output must remain provisional"
        )
    energy_gate_audit = _validate_full_space_energy_gate(
        rvmd_flow.metadata, context.full_space_energy_target
    )
    residual_fraction = 1.0 - float(energy_gate_audit["captured_fraction"])
    baseline_flow = NatoGD3DBaselineFlow(context.cache_dir, expected_dt_s=context.dt_s)
    grid_audit = validate_grid_and_time(
        context,
        rvmd_flow.bounds_xyz_m,
        rvmd_flow.frame_count,
        [float(rotor["diameter_m"]) for rotor in rotors],
    )
    baseline_bounds_at_rvmd_precision = np.asarray(
        baseline_flow.metadata["bounds_xyz_m"], dtype=np.float32
    ).astype(float)
    if not np.array_equal(rvmd_flow.bounds_xyz_m, baseline_bounds_at_rvmd_precision):
        raise CouplingConfigurationError(
            "RVMD and original-CFD spatial bounds disagree after exact float32 normalization"
        )
    stage_indices = (
        list(range(rvmd_flow.mode_count + 1))
        if stages is None
        else sorted(set(int(value) for value in stages))
    )
    definitions = _case_definitions(rvmd_flow, stage_indices)
    if include_modal_sensitivity:
        sensitivity_definitions = _modal_sensitivity_case_definitions(
            rvmd_flow, sensitivity_mode_indices_one_based
        )
        has_stage_full = any(item.stage_index == 0 for item in definitions)
        has_stage_mean = any(item.stage_index == rvmd_flow.mode_count for item in definitions)
        definitions.extend(
            item
            for item in sensitivity_definitions
            if not (
                (item.sensitivity_case_kind == "full" and has_stage_full)
                or (item.sensitivity_case_kind == "mean" and has_stage_mean)
            )
        )
    stage_rows = build_stage_rows(rvmd_flow.frequency_hz, rvmd_flow.strouhal)
    write_stage_manifest(context.output_dir / "00_provenance" / "stage_manifest.csv", stage_rows)
    body_config = context.config["body_response"]
    end_time = (
        float(body_config["analysis_window_end_s"])
        if end_time_override_s is None
        else float(end_time_override_s)
    )
    maximum_azimuth_step = (
        float(context.config["rotor_model"]["maximum_azimuth_step_deg"])
        if maximum_azimuth_step_override_deg is None
        else float(maximum_azimuth_step_override_deg)
    )
    worker_count = int(workers)
    if worker_count < 1 or worker_count > 8:
        raise CouplingConfigurationError("workers must lie in [1,8]")
    trim_rows: List[Dict[str, Any]] = []
    all_statistics: List[Dict[str, Any]] = []
    case_audits: List[Dict[str, Any]] = []
    case_manifest_rows: List[Dict[str, Any]] = []
    for rotor in rotors:
        model = prepare_rotor_model(rotor, simulation, context, rvmd_flow)
        trim_row = model.trim.as_row()
        mean_trim = model.mean_flow_trim_audit
        mean_force = mean_trim["mean_force_body_N"]
        mean_hub_aero = mean_trim["mean_hub_aero_moment_body_Nm"]
        trim_row.update(
            {
                "trim_environment": "NATO-GD RVMD time-mean 3-D field",
                "trim_method": mean_trim["method"],
                "periodic_orbit_shooting": mean_trim["periodic_orbit_shooting"],
                "mean_flow_trim_phase_samples": mean_trim["phase_samples"],
                "still_air_seed_collective_deg": model.trim.collective_trim_deg,
                "still_air_seed_induced_velocity_mps": model.trim.induced_velocity_mps,
                "collective_trim_deg": mean_trim["collective_deg"],
                "induced_velocity_trim_mps": mean_trim["uniform_induced_velocity_mps"],
                "initial_delta_lambda_0": model.initial_inflow_state[0],
                "initial_delta_lambda_c": model.initial_inflow_state[1],
                "initial_delta_lambda_s": model.initial_inflow_state[2],
                "initial_uniform_total_lambda": mean_trim["uniform_total_lambda"],
                "trim_thrust_N": mean_trim["mean_thrust_N"],
                "trim_Fx_N": mean_force[0],
                "trim_Fy_N": mean_force[1],
                "trim_Fz_N": mean_force[2],
                "trim_hub_Mx_Nm": mean_hub_aero[0],
                "trim_hub_My_Nm": mean_hub_aero[1],
                "trim_hub_Mz_Nm": mean_hub_aero[2],
                "trim_shaft_torque_Nm": mean_trim["mean_shaft_torque_Nm"],
                "trim_shaft_power_W": mean_trim["mean_shaft_power_W"],
                "trim_engine_power_W": mean_trim["mean_engine_power_W"],
                "alpha_domain_violation_count": mean_trim[
                    "maximum_alpha_domain_violation_count_per_phase"
                ],
                "cl_saturation_count": mean_trim[
                    "maximum_cl_saturation_count_per_phase"
                ],
                "relative_thrust_residual": (
                    float(mean_trim["mean_thrust_N"]) - model.trim.weight_N
                ) / model.trim.weight_N,
                "mean_flow_trim_maximum_abs_scaled_residual": mean_trim[
                    "maximum_abs_scaled_residual"
                ],
                "parameter_status": rotor.get("parameter_status", "unspecified"),
                "balance_definition": model.balance_definition,
                "balance_Fx_body_N": model.balance_other_force_body_N[0],
                "balance_Fy_body_N": model.balance_other_force_body_N[1],
                "balance_Fz_body_N": model.balance_other_force_body_N[2],
                "balance_Mx_body_Nm": model.balance_other_moment_body_Nm[0],
                "balance_My_body_Nm": model.balance_other_moment_body_Nm[1],
                "balance_Mz_body_Nm": model.balance_other_moment_body_Nm[2],
                "hover_hold_enabled": model.hover_hold_metadata["enabled"],
                "hover_hold_vertical_stiffness_Npm": model.hover_hold_vertical_stiffness_Npm,
                "hover_hold_vertical_rate_gain_Ns_pm": model.hover_hold_vertical_rate_gain_Ns_pm,
                "hover_hold_roll_stiffness_Nm_per_rad": model.hover_hold_angular_stiffness_Nm_per_rad[0],
                "hover_hold_pitch_stiffness_Nm_per_rad": model.hover_hold_angular_stiffness_Nm_per_rad[1],
                "hover_hold_roll_rate_gain_Nms": model.hover_hold_angular_rate_gain_Nms[0],
                "hover_hold_pitch_rate_gain_Nms": model.hover_hold_angular_rate_gain_Nms[1],
            }
        )
        trim_rows.append(trim_row)
        rotor_definitions = list(definitions)
        if include_original_cfd:
            rotor_definitions.append(
                CaseDefinition(
                    case_id="original_cfd",
                    stage_index=None,
                    retained_mode_count=None,
                    flow_kind="original_CFD",
                    cutoff_frequency_hz=None,
                    cutoff_St=None,
                )
            )
        def execute_definition(definition: CaseDefinition) -> Tuple[CaseDefinition, CaseResult, Path]:
            active_flow = baseline_flow if definition.flow_kind == "original_CFD" else rvmd_flow
            output_name = "{}__{}.csv".format(_slug(str(rotor["name"])), definition.case_id)
            output_path = context.output_dir / "04_timeseries_csv" / output_name
            metadata_path = output_path.with_suffix(".metadata.json")
            reusable = (
                resume
                and output_path.is_file()
                and metadata_path.is_file()
                and _resume_output_matches_request(
                    metadata_path,
                    definition,
                    model,
                    end_time_s=end_time,
                    output_interval_s=context.dt_s,
                    maximum_azimuth_step_deg=maximum_azimuth_step,
                    run_signature_sha256=run_signature,
                )
            )
            if reusable:
                print("skip existing {}".format(output_name), flush=True)
                result = read_case_csv(output_path, definition)
            else:
                if resume and (output_path.exists() or metadata_path.exists()):
                    print("replace stale or incompatible {}".format(output_name), flush=True)
                print("simulate {} {}".format(rotor["name"], definition.case_id), flush=True)
                result = simulate_case(
                    copy.deepcopy(model),
                    active_flow,
                    definition,
                    end_time_s=end_time,
                    output_interval_s=context.dt_s,
                    maximum_azimuth_step_deg=maximum_azimuth_step,
                )
                result.metadata["run_signature_sha256"] = run_signature
                write_case_csv(output_path, result)
            return definition, result, output_path

        completed_cases: List[Tuple[CaseDefinition, CaseResult, Path]] = []
        if worker_count == 1 or len(rotor_definitions) == 1:
            completed_cases = [execute_definition(item) for item in rotor_definitions]
        else:
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(rotor_definitions)),
                initializer=_initialize_coupling_worker,
                initargs=(
                    str(rvmd_directory),
                    str(context.cache_dir),
                    float(context.dt_s),
                ),
            ) as executor:
                futures = {}
                for item in rotor_definitions:
                    output_name = "{}__{}.csv".format(
                        _slug(str(rotor["name"])), item.case_id
                    )
                    task = {
                        "definition": item,
                        "model": model,
                        "output_path": str(
                            context.output_dir / "04_timeseries_csv" / output_name
                        ),
                        "resume": resume,
                        "end_time_s": end_time,
                        "output_interval_s": context.dt_s,
                        "maximum_azimuth_step_deg": maximum_azimuth_step,
                        "run_signature_sha256": run_signature,
                    }
                    futures[executor.submit(_execute_coupling_case_worker, task)] = item
                for future in as_completed(futures):
                    completed_cases.append(future.result())
        completed_cases.sort(
            key=lambda item: (
                item[0].stage_index is None,
                999 if item[0].stage_index is None else int(item[0].stage_index),
                item[0].case_id,
            )
        )
        for definition, result, output_path in completed_cases:
            all_statistics.extend(
                statistics_rows(
                    result,
                    float(body_config["analysis_window_start_s"]),
                    float(body_config["analysis_window_end_s"]),
                    float(rotor["diameter_m"]),
                )
            )
            case_audits.append(
                {
                    "rotor_name": rotor["name"],
                    "case_id": definition.case_id,
                    "csv": str(output_path),
                    "metadata": result.metadata,
                }
            )
            manifest_case_kind = definition.sensitivity_case_kind
            if include_modal_sensitivity and definition.stage_index == 0:
                manifest_case_kind = "full"
            elif (
                include_modal_sensitivity
                and definition.stage_index == rvmd_flow.mode_count
            ):
                manifest_case_kind = "mean"
            if manifest_case_kind is not None:
                case_manifest_rows.append(
                    {
                        "rotor_name": rotor["name"],
                        "rotor_diameter_m": rotor["diameter_m"],
                        "case_kind": manifest_case_kind,
                        "mode_index_one_based": (
                            ""
                            if definition.sensitivity_mode_index_one_based is None
                            else definition.sensitivity_mode_index_one_based
                        ),
                        "csv_path": str(output_path.resolve()),
                    }
                )
    if trim_rows:
        write_rows_csv(context.output_dir / "03_hover_trim" / "hover_trim_all_rotors.csv", trim_rows)
    if all_statistics:
        write_rows_csv(context.output_dir / "05_statistics" / "statistics_long.csv", all_statistics)
    if case_manifest_rows:
        write_rows_csv(
            context.output_dir / "05_statistics" / "modal_sensitivity_case_manifest.csv",
            case_manifest_rows,
        )
    summary = {
        "status": (
            "provisional"
            if (not context.production_ready or allow_nonconverged_rvmd)
            else "production_ready"
        ),
        "study_context": context.metadata(),
        "grid_and_time_audit": grid_audit,
        "rotors_completed": [rotor["name"] for rotor in rotors],
        "stage_indices_requested": stage_indices,
        "modal_sensitivity_requested": bool(include_modal_sensitivity),
        "sensitivity_mode_indices_one_based": (
            list(range(1, rvmd_flow.mode_count + 1))
            if include_modal_sensitivity and sensitivity_mode_indices_one_based is None
            else list(sensitivity_mode_indices_one_based or [])
        ),
        "full_space_energy_gate": energy_gate_audit,
        "original_cfd_requested": include_original_cfd,
        "parallel_case_workers": worker_count,
        "rvmd_converged": bool(rvmd_info.get("converged", False)),
        "nonconverged_override_used": bool(allow_nonconverged_rvmd),
        "rvmd_full_space_residual_energy_fraction": residual_fraction,
        "run_signature_sha256": run_signature,
        "case_audits": case_audits,
    }
    _write_json(context.output_dir / "00_provenance" / "coupling_run.json", summary)
    rvmd_flow.close()
    return summary


def _parse_int_ranges(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    output: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            output.extend(range(first, last + 1))
        else:
            output.append(int(token))
    return sorted(set(output))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--rotor", action="append", dest="rotors")
    parser.add_argument("--stages", help="comma/range list, for example 0,1,5-8")
    parser.add_argument(
        "--modal-sensitivity",
        action="store_true",
        help="also run mean/full plus single-mode and leave-one-out cases",
    )
    parser.add_argument(
        "--sensitivity-modes",
        help="optional one-based comma/range subset for modal sensitivity",
    )
    parser.add_argument("--skip-original-cfd", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--end-time", type=float, help="testing override; default common window end")
    parser.add_argument("--maximum-azimuth-step-deg", type=float, help="testing/convergence override")
    parser.add_argument("--allow-nonconverged-rvmd", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel cases per rotor (1-8; each case receives an independent dynamics model)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    context = load_study_context(args.config, allow_provisional=args.allow_provisional)
    summary = run_study(
        context,
        rotor_names=args.rotors,
        stages=_parse_int_ranges(args.stages),
        include_modal_sensitivity=args.modal_sensitivity,
        sensitivity_mode_indices_one_based=_parse_int_ranges(args.sensitivity_modes),
        include_original_cfd=not args.skip_original_cfd,
        resume=not args.no_resume,
        end_time_override_s=args.end_time,
        maximum_azimuth_step_override_deg=args.maximum_azimuth_step_deg,
        allow_nonconverged_rvmd=args.allow_nonconverged_rvmd,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CORE_METRICS",
    "CaseDefinition",
    "CaseResult",
    "CouplingConfigurationError",
    "CouplingRunError",
    "HoverTrim",
    "RotorStudyModel",
    "combined_rhs",
    "load_rotor_mappings",
    "output_times",
    "prepare_rotor_model",
    "read_case_csv",
    "run_study",
    "simulate_case",
    "solve_azimuth_mean_hover_trim",
    "solve_physical_hover_trim",
    "statistics_rows",
    "write_case_csv",
]
