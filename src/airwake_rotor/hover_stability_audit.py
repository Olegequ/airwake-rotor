#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Audit the production NATO-GD azimuth-mean hover closure.

The audit never changes a case, subtracts a case-specific mean, clips a state,
or extrapolates the CFD field.  It independently samples the stored time-mean
flow over one blade-passage cycle at the production collective and initial
Pitt--Peters state, then reports force, moment and inflow-equation closure.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .nato_gd_study import load_study_context
from .rvmd_3d_flow import RVMD3DFlow
from .rvmd_nato_coupling_pipeline import (
    _evaluate_aerodynamics,
    _world_to_body,
    load_rotor_mappings,
    prepare_rotor_model,
    solve_azimuth_mean_hover_trim,
)


def periodic_mean_hover_audit(
    model: Any,
    mean_flow: Any,
    *,
    phase_samples: int = 144,
) -> Dict[str, Any]:
    """Return a one-blade-passage closure audit at the reference body state.

    Equally spaced blades make the total rotor loads periodic in
    ``2*pi/blade_count`` for a frozen spatial field.  Endpoint-free midpoint
    quadrature avoids duplicating the same blade arrangement.
    """

    samples = int(phase_samples)
    if samples < 8:
        raise ValueError("phase_samples must be at least 8")
    blade_count = model.blade_parameters.blade_count
    period_angle = 2.0 * math.pi / blade_count
    phases = (np.arange(samples, dtype=float) + 0.5) * period_angle / samples
    state = np.zeros(9, dtype=float)
    state[6:9] = np.asarray(model.initial_inflow_state, dtype=float)
    forces_body = np.empty((samples, 3), dtype=float)
    transmitted_body = np.empty((samples, 3), dtype=float)
    hub_moments_disk = np.empty((samples, 3), dtype=float)
    inflow_rhs = np.empty((samples, 3), dtype=float)
    thrust = np.empty(samples, dtype=float)
    delta_mu_z = np.empty(samples, dtype=float)

    initial_phase = float(model.initial_phase_rad)
    for index, phase in enumerate(phases):
        # retained_mode_count=0 is exactly the stored RVMD time-mean field.
        time_s = (phase - initial_phase) / model.blade_parameters.omega_radps
        loads, _, _, delta_mu, moment_disk, _ = _evaluate_aerodynamics(
            model, mean_flow, time_s, state, retained_mode_count=0
        )
        forces_body[index] = _world_to_body(loads.hub_force_world_N, state[:6])
        transmitted_body[index] = _world_to_body(
            loads.hub_axis_moment_world_Nm, state[:6]
        )
        hub_moments_disk[index] = moment_disk
        thrust[index] = loads.thrust_N
        delta_mu_z[index] = delta_mu
        delta_coefficients = (
            model.inflow_model.coefficient_perturbation_from_total_loads(
                loads.thrust_N, moment_disk
            )
        )
        inflow_rhs[index] = model.inflow_model.rhs(
            time_s, state[6:9], delta_coefficients, delta_mu
        )

    mean_force = np.mean(forces_body, axis=0)
    mean_transmitted = np.mean(transmitted_body, axis=0)
    mean_rhs = np.mean(inflow_rhs, axis=0)
    mean_fixed_point = np.linalg.solve(
        model.blade_parameters.omega_radps
        * np.linalg.solve(
            model.inflow_model.apparent_mass_matrix,
            model.inflow_model.gain_matrix,
        ),
        mean_rhs,
    )
    # Algebraically this is K^-1 times the mean effective forcing.  Expressing
    # it through the observed derivative makes the audit independent of a
    # duplicated sign convention.
    phase_zero_balance_force = np.array(
        [0.0, 0.0, model.trim.weight_N], dtype=float
    ) - model.balance_other_force_body_N
    phase_zero_balance_moment = -model.balance_other_moment_body_Nm
    mean_cg_moment = (
        np.cross(model.body_parameters.hub_from_cg_body_m, mean_force)
        + mean_transmitted
    )
    residual_force_current_balance = (
        mean_force
        + model.balance_other_force_body_N
        - np.array([0.0, 0.0, model.trim.weight_N], dtype=float)
    )
    residual_moment_current_balance = (
        mean_cg_moment + model.balance_other_moment_body_Nm
    )

    return {
        "rotor_name": str(model.rotor["name"]),
        "phase_samples": samples,
        "sampled_phase_interval_rad": [0.0, period_angle],
        "blade_passage_period_s": period_angle / model.blade_parameters.omega_radps,
        "still_air_seed_collective_trim_deg": model.trim.collective_trim_deg,
        "still_air_seed_induced_velocity_mps": model.trim.induced_velocity_mps,
        "production_mean_flow_collective_trim_deg": model.blade_parameters.collective_deg,
        "production_initial_delta_inflow_state": model.initial_inflow_state.tolist(),
        "phase_zero_force_implied_by_current_balance_body_N": phase_zero_balance_force.tolist(),
        "phase_zero_cg_moment_implied_by_current_balance_body_Nm": phase_zero_balance_moment.tolist(),
        "periodic_mean_force_body_N": mean_force.tolist(),
        "periodic_force_peak_to_peak_body_N": np.ptp(forces_body, axis=0).tolist(),
        "periodic_mean_thrust_N": float(np.mean(thrust)),
        "periodic_thrust_peak_to_peak_N": float(np.ptp(thrust)),
        "periodic_mean_cg_moment_body_Nm": mean_cg_moment.tolist(),
        "periodic_cg_moment_peak_to_peak_body_Nm": np.ptp(
            np.cross(
                np.broadcast_to(model.body_parameters.hub_from_cg_body_m, forces_body.shape),
                forces_body,
            )
            + transmitted_body,
            axis=0,
        ).tolist(),
        "periodic_mean_delta_mu_z": float(np.mean(delta_mu_z)),
        "periodic_delta_mu_z_peak_to_peak": float(np.ptp(delta_mu_z)),
        "periodic_mean_initial_inflow_derivative_per_s": mean_rhs.tolist(),
        "mean_forcing_fixed_point_delta_lambda": mean_fixed_point.tolist(),
        "mean_forcing_fixed_point_uniform_induced_velocity_mps": float(
            model.inflow_model.tip_speed_mps
            * (model.inflow_model.lambda_star + mean_fixed_point[0])
        ),
        "current_single_phase_balance_periodic_mean_force_residual_body_N": (
            residual_force_current_balance.tolist()
        ),
        "current_single_phase_balance_periodic_mean_cg_moment_residual_body_Nm": (
            residual_moment_current_balance.tolist()
        ),
        "current_single_phase_balance_periodic_mean_vertical_acceleration_mps2": float(
            residual_force_current_balance[2] / model.body_parameters.mass_kg
        ),
        "closure_interpretation": (
            "The production trim passes when the periodic-mean vertical force residual "
            "and all three mean inflow derivatives are within their declared numerical "
            "tolerances; the fixed-body reaction closes the remaining mean coordinates."
        ),
    }


def _periodic_samples(
    model: Any,
    mean_flow: Any,
    phase_samples: int,
    delta_inflow: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Evaluate fixed-body loads and inflow residuals over one blade passage."""

    blade_count = model.blade_parameters.blade_count
    period_angle = 2.0 * math.pi / blade_count
    phases = (np.arange(phase_samples, dtype=float) + 0.5) * period_angle / phase_samples
    state = np.zeros(9, dtype=float)
    state[6:9] = np.asarray(delta_inflow, dtype=float)
    forces_body = np.empty((phase_samples, 3), dtype=float)
    moments_cg = np.empty((phase_samples, 3), dtype=float)
    inflow_equation_residual = np.empty((phase_samples, 3), dtype=float)
    thrust = np.empty(phase_samples, dtype=float)
    for index, phase in enumerate(phases):
        time_s = (phase - float(model.initial_phase_rad)) / model.blade_parameters.omega_radps
        loads, _, _, delta_mu, moment_disk, _ = _evaluate_aerodynamics(
            model, mean_flow, time_s, state, retained_mode_count=0
        )
        force = _world_to_body(loads.hub_force_world_N, state[:6])
        transmitted = _world_to_body(loads.hub_axis_moment_world_Nm, state[:6])
        forces_body[index] = force
        moments_cg[index] = (
            np.cross(model.body_parameters.hub_from_cg_body_m, force) + transmitted
        )
        thrust[index] = loads.thrust_N
        derivative = model.inflow_model.rhs(
            time_s,
            state[6:9],
            model.inflow_model.coefficient_perturbation_from_total_loads(
                loads.thrust_N, moment_disk
            ),
            delta_mu,
        )
        # Convert lambda_dot back to the dimensionless forcing residual
        # M*lambda_dot/Omega = c_eff-K*delta_lambda.
        inflow_equation_residual[index] = (
            model.inflow_model.apparent_mass_matrix @ derivative
            / model.blade_parameters.omega_radps
        )
    return {
        "force_body_N": forces_body,
        "moment_cg_body_Nm": moments_cg,
        "thrust_N": thrust,
        "inflow_equation_residual": inflow_equation_residual,
    }


def _legacy_solve_azimuth_mean_hover_trim(
    model: Any,
    mean_flow: Any,
    *,
    phase_samples: int = 48,
    collective_bounds_deg: Sequence[float] = (-5.0, 35.0),
) -> Dict[str, Any]:
    """Solve an azimuth-mean fixed-body rotor/inflow hover equilibrium.

    Unknowns are collective and the three Pitt--Peters perturbation states.
    The four equations are ``mean(Fz)=weight`` and zero mean of each dynamic-
    inflow equation.  Horizontal translation and yaw are constrained by the
    declared body model; the remaining azimuth-mean force/moment reactions are
    reported explicitly and may be balanced only once for every modal case.

    A small bounded Gauss--Newton solver is implemented with NumPy so the
    production trim does not depend on optional SciPy sparse-library DLLs.
    This is a mean fixed point, not a periodic-orbit shooting solution.
    """
    samples = int(phase_samples)
    if samples < 8:
        raise ValueError("phase_samples must be at least 8")
    lambda_star = model.inflow_model.lambda_star
    delta_limit = 0.74 * lambda_star
    lower = np.array(
        [float(collective_bounds_deg[0])] + [-delta_limit] * 3,
        dtype=float,
    )
    upper = np.array(
        [float(collective_bounds_deg[1])] + [delta_limit] * 3,
        dtype=float,
    )
    initial = np.array(
        [model.trim.collective_trim_deg, 0.0, 0.0, 0.0], dtype=float
    )
    weight = model.trim.weight_N
    ct_reference = model.inflow_model.reference_thrust_coefficient

    evaluations = 0

    def evaluate_unknown(unknown: np.ndarray) -> tuple:
        nonlocal evaluations
        evaluations += 1
        blade = replace(
            model.blade_parameters,
            collective_deg=float(unknown[0]),
            cyclic_cos_deg=0.0,
            cyclic_sin_deg=0.0,
        )
        candidate = replace(model, blade_parameters=blade)
        evaluated = _periodic_samples(
            candidate, mean_flow, samples, unknown[1:4]
        )
        mean_force = np.mean(evaluated["force_body_N"], axis=0)
        mean_inflow = np.mean(evaluated["inflow_equation_residual"], axis=0)
        scaled = np.concatenate(
            (np.array([(mean_force[2] - weight) / weight]), mean_inflow / ct_reference)
        )
        return scaled, evaluated, candidate

    unknown = initial.copy()
    scaled_residual, evaluated, candidate = evaluate_unknown(unknown)
    success = False
    message = "maximum iterations reached"
    iterations = 0
    for iteration in range(40):
        iterations = iteration + 1
        residual_norm = float(np.max(np.abs(scaled_residual)))
        if residual_norm <= 1.0e-8:
            success = True
            message = "scaled residual tolerance satisfied"
            break
        increments = np.array(
            [1.0e-3, 2.0e-5 * lambda_star, 2.0e-5 * lambda_star,
             2.0e-5 * lambda_star], dtype=float
        )
        jacobian = np.empty((4, 4), dtype=float)
        for column in range(4):
            plus = unknown.copy()
            minus = unknown.copy()
            plus[column] = min(upper[column], plus[column] + increments[column])
            minus[column] = max(lower[column], minus[column] - increments[column])
            if plus[column] == minus[column]:
                raise RuntimeError("trim variable is immobile at both bounds")
            plus_residual, _, _ = evaluate_unknown(plus)
            minus_residual, _, _ = evaluate_unknown(minus)
            jacobian[:, column] = (
                (plus_residual - minus_residual) / (plus[column] - minus[column])
            )
        step, _, rank, _ = np.linalg.lstsq(jacobian, -scaled_residual, rcond=1.0e-10)
        if rank < 4 or not np.all(np.isfinite(step)):
            message = "finite-difference trim Jacobian is rank deficient"
            break
        step[0] = float(np.clip(step[0], -3.0, 3.0))
        step[1:] = np.clip(step[1:], -0.20 * lambda_star, 0.20 * lambda_star)
        old_norm = float(np.linalg.norm(scaled_residual))
        accepted = False
        for exponent in range(12):
            scale = 0.5 ** exponent
            trial = np.clip(unknown + scale * step, lower, upper)
            trial_residual, trial_evaluated, trial_candidate = evaluate_unknown(trial)
            if float(np.linalg.norm(trial_residual)) < old_norm:
                unknown = trial
                scaled_residual = trial_residual
                evaluated = trial_evaluated
                candidate = trial_candidate
                accepted = True
                break
        if not accepted:
            message = "bounded line search could not reduce the trim residual"
            break

    blade = replace(
        model.blade_parameters,
        collective_deg=float(unknown[0]),
        cyclic_cos_deg=0.0,
        cyclic_sin_deg=0.0,
    )
    candidate = replace(model, blade_parameters=blade)
    evaluated = _periodic_samples(candidate, mean_flow, samples, unknown[1:4])
    mean_force = np.mean(evaluated["force_body_N"], axis=0)
    mean_moment = np.mean(evaluated["moment_cg_body_Nm"], axis=0)
    mean_inflow = np.mean(evaluated["inflow_equation_residual"], axis=0)
    scaled_residual, _, _ = evaluate_unknown(unknown)
    if float(np.max(np.abs(scaled_residual))) <= 1.0e-8:
        success = True
        message = "scaled residual tolerance satisfied"
    return {
        "rotor_name": str(model.rotor["name"]),
        "method": "azimuth-mean fixed-body collective/inflow equilibrium",
        "periodic_orbit_shooting": False,
        "phase_samples": samples,
        "success": bool(success),
        "optimizer_status": 1 if success else 0,
        "optimizer_message": message,
        "iterations": iterations,
        "function_evaluations": evaluations,
        "collective_deg": float(unknown[0]),
        "cyclic_cos_deg": 0.0,
        "cyclic_sin_deg": 0.0,
        "delta_inflow_state": unknown[1:4].tolist(),
        "uniform_induced_velocity_mps": float(
            model.inflow_model.tip_speed_mps * (lambda_star + unknown[1])
        ),
        "mean_force_body_N": mean_force.tolist(),
        "force_residual_body_N": (
            mean_force - np.array([0.0, 0.0, weight], dtype=float)
        ).tolist(),
        "mean_cg_moment_body_Nm": mean_moment.tolist(),
        "mean_inflow_equation_residual": mean_inflow.tolist(),
        "maximum_abs_scaled_residual": float(np.max(np.abs(scaled_residual))),
        "controls_at_bound": bool(
            np.isclose(unknown[0], lower[0], atol=1.0e-6)
            or np.isclose(unknown[0], upper[0], atol=1.0e-6)
        ),
        "delta_inflow_at_bound": bool(
            np.any(np.isclose(unknown[1:], lower[1:], atol=1.0e-7))
            or np.any(np.isclose(unknown[1:], upper[1:], atol=1.0e-7))
        ),
    }


def run_audit(
    config_path: Path,
    *,
    rotor_names: Optional[Sequence[str]] = None,
    phase_samples: int = 144,
) -> Dict[str, Any]:
    context = load_study_context(config_path, allow_provisional=True)
    simulation, rotors = load_rotor_mappings(context)
    if rotor_names:
        requested = set(rotor_names)
        rotors = [rotor for rotor in rotors if str(rotor["name"]) in requested]
        missing = requested - {str(rotor["name"]) for rotor in rotors}
        if missing:
            raise ValueError("unknown rotors: {}".format(sorted(missing)))
    flow = RVMD3DFlow(context.output_dir / "02_rvmd")
    try:
        audits = []
        for rotor in rotors:
            model = prepare_rotor_model(rotor, simulation, context, flow)
            audit = periodic_mean_hover_audit(
                model, flow, phase_samples=phase_samples
            )
            audit["production_azimuth_mean_trim"] = model.mean_flow_trim_audit
            audits.append(audit)
    finally:
        flow.close()
    return {
        "config": str(config_path.resolve()),
        "flow_used": "RVMD stored time-mean field only (retained_mode_count=0)",
        "case_specific_mean_removal": False,
        "flow_clipping_or_extrapolation": False,
        "rotors": audits,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--rotor", action="append", dest="rotors")
    parser.add_argument("--phase-samples", type=int, default=144)
    parser.add_argument("--output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_audit(
        Path(args.config), rotor_names=args.rotors, phase_samples=args.phase_samples
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized + "\n", encoding="utf-8")
        temporary.replace(output)
    print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "periodic_mean_hover_audit",
    "run_audit",
    "solve_azimuth_mean_hover_trim",
]
