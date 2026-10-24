# Copyright (c) 2026 Olegequ.
"""Deterministic synthetic RVMD -> nonlinear blade/inflow/body demonstration."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .branding import save_figure
from .nonlinear_blade_element import (
    AerodynamicDomainLimits, BladeElementParameters, radial_element_centres_and_widths,
)
from .nonlinear_rotorcraft_dynamics import (
    DynamicInflowParameters, NearHoverPittPetersInflow, RigidBody3DOFParameters,
)
from .rvmd_core import rvmd
from .rvmd_nato_coupling_pipeline import (
    CaseDefinition, RotorStudyModel, simulate_case, solve_physical_hover_trim, write_case_csv,
)


class SyntheticModalFlow:
    """Analytic, unbounded demonstration field driven by recovered RVMD modes.

    This deliberately small example has no CFD geometry or dataset cache.
    Production spatial/time bounds are provided by RVMD3DFlow instead.
    """
    def __init__(self, result, dt):
        self.result = result
        self.times = np.arange(result.c.shape[0]) * dt

    def velocity_at(self, relative_time_s, points_xyz, *, retained_mode_count=None,
                    mode_indices=None, return_mode_contributions=False):
        if return_mode_contributions:
            raise ValueError("Synthetic demo supports total velocity only")
        if relative_time_s < 0 or relative_time_s > self.times[-1] + 1e-12:
            raise ValueError("Synthetic flow time is outside the sampled interval")
        indices = list(range(self.result.phi.shape[1])) if mode_indices is None else list(mode_indices)
        if retained_mode_count is not None:
            indices = indices[:retained_mode_count]
        coefficients = np.array([
            np.interp(relative_time_s, self.times, self.result.c[:, i]) for i in indices
        ])
        value = self.result.phi[:, indices] @ coefficients
        points = np.asarray(points_xyz, dtype=float)
        velocity = np.zeros_like(points)
        velocity[..., 0] = 0.03 * value[0]
        velocity[..., 1] = 0.01 * value[1]
        velocity[..., 2] = 0.04 * value[2] + 0.005 * points[..., 0] * value[3]
        return velocity


def build_synthetic_model():
    rotor = dict(
        name="Synthetic", diameter_m=2.0, blades=2, rpm=600.0,
        root_cutout=0.2, mean_chord_m=0.16, collective_deg=12.0,
        twist_deg=-5.0, cd0=0.012, cl_alpha=5.73, cl_max=1.35,
        transmission_efficiency=0.9, mass_kg=22.0, Ixx_kgm2=6.0,
        Iyy_kgm2=7.0, Izz_kgm2=11.0, hub_above_cg_m=0.2,
    )
    density = 1.225
    trim = solve_physical_hover_trim(
        rotor, density_kgpm3=density, gravity_mps2=9.80665, radial_elements=10,
    )
    blade = BladeElementParameters.from_mapping(
        rotor, collective_deg=trim.collective_trim_deg,
        domain_limits=AerodynamicDomainLimits(action="report", reverse_flow_action="report"),
    )
    radii, widths = radial_element_centres_and_widths(blade, 10)
    body = RigidBody3DOFParameters.from_mapping(
        rotor, vertical_damping_N_per_mps=10.0,
        angular_damping_body_Nms=np.array([2.0, 2.0, 0.0]),
        maximum_abs_attitude_deg=12.0, maximum_abs_heave_m=1.0,
        maximum_abs_heave_speed_mps=5.0,
    )
    inflow = NearHoverPittPetersInflow(DynamicInflowParameters(
        radius_m=blade.radius_m, omega_radps=blade.omega_radps,
        density_kgpm3=density, reference_thrust_N=trim.weight_N,
        reference_hub_aero_moment_disk_Nm=trim.loads.hub_aerodynamic_moment_world_Nm,
    ))
    force = trim.loads.hub_force_world_N
    transmitted = trim.loads.hub_axis_moment_world_Nm
    return RotorStudyModel(
        rotor=rotor, blade_parameters=blade, body_parameters=body,
        inflow_model=inflow, radii_m=radii, widths_m=widths,
        rotor_axis_body=np.array([0.0, 0.0, 1.0]),
        nominal_cg_world_m=np.zeros(3), initial_phase_rad=0.0,
        density_kgpm3=density, trim=trim,
        balance_other_force_body_N=np.array([0.0, 0.0, trim.weight_N]) - force,
        balance_other_moment_body_Nm=-(np.cross(body.hub_from_cg_body_m, force) + transmitted),
        balance_definition="synthetic still-air balance shared across all demo cases",
    )


def run_demo(output):
    """Write a repeatable two-frequency example and three coupled responses."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dt = 0.02
    times = np.arange(256) * dt
    frequencies = [1.5625, 3.125]
    spatial = np.array([[1, 0], [0, 0.3], [0, 1], [0.3, 0], [0.2, 0], [0, 0.1]], dtype=float)
    spatial /= np.linalg.norm(spatial, axis=0)
    temporal = np.array([np.cos(2 * math.pi * frequencies[0] * times),
                         0.6 * np.cos(2 * math.pi * frequencies[1] * times)])
    field = spatial @ temporal
    result = rvmd(field, k=2, alpha=2000, dt=dt, dtype="float64",
                  initial_frequencies_hz=frequencies, tolerance=1e-7, maximum_steps=2000)
    reconstruction = result.reconstruct()
    relative_error = float(np.linalg.norm(field - reconstruction) / np.linalg.norm(field))
    if not np.all(np.isfinite(reconstruction)) or relative_error > 0.04:
        raise RuntimeError("Synthetic RVMD reconstruction did not pass the demo error gate")
    result.save(output / "rvmd_result.npz")
    np.savetxt(output / "modal_coefficients.csv", np.column_stack([times, result.c]),
               delimiter=",", header="time_s,mode_1,mode_2", comments="")
    model = build_synthetic_model()
    flow = SyntheticModalFlow(result, dt)
    responses = {}
    for case_id, retained in [("full", 2), ("delete_highest", 1), ("mean", 0)]:
        definition = CaseDefinition(case_id, 2-retained, retained, "rvmd", None, None)
        case = simulate_case(model, flow, definition, end_time_s=1.0,
                             output_interval_s=dt, maximum_azimuth_step_deg=15.0)
        if not all(np.isfinite(float(row["lift_N"])) for row in case.rows):
            raise RuntimeError("Synthetic coupled response contains nonfinite lift")
        write_case_csv(output / (case_id + "_timeseries.csv"), case)
        responses[case_id] = case
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(times, field[0], label="Synthetic input", linewidth=1.5)
    axes[0].plot(times, reconstruction[0], "--", label="RVMD reconstruction", linewidth=1.0)
    axes[0].set_ylabel("Signal (a.u.)")
    for name, case in responses.items():
        response_times = [row["time_s"] for row in case.rows]
        axes[1].plot(response_times, [row["lift_N"] for row in case.rows], label=name)
        axes[2].plot(response_times, [row["heave_m"] for row in case.rows], label=name)
    axes[1].set_ylabel("Lift (N)")
    axes[2].set_ylabel("Heave (m)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(alpha=0.2)
    figure.suptitle("airwake-rotor | Synthetic demonstration")
    save_figure(figure, output / "demo.png", dpi=160)
    plt.close(figure)
    report = {
        "project": "airwake-rotor", "author": "Olegequ", "synthetic_data": True,
        "seed": "deterministic analytic signals", "dt_s": dt,
        "known_frequency_hz": frequencies, "recovered_frequency_hz": result.frequency_hz.tolist(),
        "relative_reconstruction_error": relative_error,
        "rvmd_converged": bool(result.info.get("converged", result.difference_history[-1] <= 1e-7)),
        "hover_trim_relative_thrust_residual": float(model.trim.relative_thrust_residual),
        "coupled_cases": list(responses),
        "scope": "Small analytic field with synthetic rotor; installation and method demonstration, not validation against measured CFD or aircraft data.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " | Olegequ")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    report = run_demo(args.output)
    print(json.dumps(report, indent=2))
    print("Outputs:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
