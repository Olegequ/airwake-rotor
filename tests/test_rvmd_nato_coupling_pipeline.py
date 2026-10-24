# Copyright (c) 2026 Olegequ.
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airwake_rotor.nonlinear_blade_element import (
    AerodynamicDomainLimits,
    BladeElementParameters,
    radial_element_centres_and_widths,
)
from airwake_rotor.nonlinear_rotorcraft_dynamics import (
    DynamicInflowParameters,
    NearHoverPittPetersInflow,
    RigidBody3DOFParameters,
)
from airwake_rotor.rvmd_nato_coupling_pipeline import (
    _modal_sensitivity_case_definitions,
    _validate_full_space_energy_gate,
    _hover_hold_generalized_loads,
    CaseDefinition,
    RotorStudyModel,
    output_times,
    read_case_csv,
    simulate_case,
    solve_azimuth_mean_hover_trim,
    solve_physical_hover_trim,
    statistics_rows,
    write_case_csv,
)


class ZeroModalFlow:
    def __init__(self):
        self.retained_requests = []
        self.mode_requests = []

    def velocity_at(
        self,
        relative_time_s,
        points_xyz,
        *,
        retained_mode_count=None,
        mode_indices=None,
        return_mode_contributions=False
    ):
        del relative_time_s, return_mode_contributions
        self.retained_requests.append(retained_mode_count)
        self.mode_requests.append(None if mode_indices is None else tuple(mode_indices))
        return np.zeros_like(np.asarray(points_xyz, dtype=float))


class UniformModalFlow(ZeroModalFlow):
    def __init__(self, velocity_xyz_mps):
        super().__init__()
        self.velocity_xyz_mps = np.asarray(velocity_xyz_mps, dtype=float)

    def velocity_at(
        self,
        relative_time_s,
        points_xyz,
        *,
        retained_mode_count=None,
        mode_indices=None,
        return_mode_contributions=False,
    ):
        del relative_time_s, return_mode_contributions
        self.retained_requests.append(retained_mode_count)
        self.mode_requests.append(None if mode_indices is None else tuple(mode_indices))
        points = np.asarray(points_xyz, dtype=float)
        return np.broadcast_to(self.velocity_xyz_mps, points.shape).copy()


def synthetic_rotor():
    return {
        "name": "Synthetic",
        "diameter_m": 2.0,
        "blades": 2,
        "rpm": 600.0,
        "root_cutout": 0.2,
        "mean_chord_m": 0.16,
        "collective_deg": 12.0,
        "twist_deg": -5.0,
        "cd0": 0.012,
        "cl_alpha": 5.73,
        "cl_max": 1.35,
        "transmission_efficiency": 0.9,
        "mass_kg": 22.0,
        "Ixx_kgm2": 6.0,
        "Iyy_kgm2": 7.0,
        "Izz_kgm2": 11.0,
        "hub_above_cg_m": 0.2,
    }


def build_equilibrium_model():
    rotor = synthetic_rotor()
    density = 1.225
    gravity = 9.80665
    trim = solve_physical_hover_trim(
        rotor,
        density_kgpm3=density,
        gravity_mps2=gravity,
        radial_elements=10,
    )
    blade = BladeElementParameters.from_mapping(
        rotor,
        collective_deg=trim.collective_trim_deg,
        domain_limits=AerodynamicDomainLimits(action="report", reverse_flow_action="report"),
    )
    radii, widths = radial_element_centres_and_widths(blade, 10)
    body = RigidBody3DOFParameters.from_mapping(
        rotor,
        vertical_damping_N_per_mps=10.0,
        angular_damping_body_Nms=np.array([2.0, 2.0, 0.0]),
        maximum_abs_attitude_deg=12.0,
        maximum_abs_heave_m=1.0,
        maximum_abs_heave_speed_mps=5.0,
    )
    inflow = NearHoverPittPetersInflow(
        DynamicInflowParameters(
            radius_m=blade.radius_m,
            omega_radps=blade.omega_radps,
            density_kgpm3=density,
            reference_thrust_N=trim.weight_N,
            reference_hub_aero_moment_disk_Nm=trim.loads.hub_aerodynamic_moment_world_Nm,
        )
    )
    force = trim.loads.hub_force_world_N
    transmitted = trim.loads.hub_axis_moment_world_Nm
    other_force = np.array([0.0, 0.0, trim.weight_N]) - force
    other_moment = -(np.cross(body.hub_from_cg_body_m, force) + transmitted)
    return RotorStudyModel(
        rotor=rotor,
        blade_parameters=blade,
        body_parameters=body,
        inflow_model=inflow,
        radii_m=radii,
        widths_m=widths,
        rotor_axis_body=np.array([0.0, 0.0, 1.0]),
        nominal_cg_world_m=np.array([0.0, 0.0, 0.0]),
        initial_phase_rad=0.0,
        density_kgpm3=density,
        trim=trim,
        balance_other_force_body_N=other_force,
        balance_other_moment_body_Nm=other_moment,
        balance_definition="synthetic still-air balance",
    )


class CouplingPipelineTests(unittest.TestCase):
    def test_modal_sensitivity_cases_are_explicit_single_and_leave_one_out(self):
        class FlowDescriptor:
            mode_count = 3
            frequency_hz = np.asarray([0.1, 0.2, 0.3])
            strouhal = frequency_hz * 13.716 / 20.0

        definitions = _modal_sensitivity_case_definitions(FlowDescriptor(), [2])
        self.assertEqual([item.sensitivity_case_kind for item in definitions], [
            "full", "mean", "single", "leave_one_out"
        ])
        self.assertEqual(definitions[2].mode_indices_zero_based, (1,))
        self.assertEqual(definitions[3].mode_indices_zero_based, (0, 2))

        model = build_equilibrium_model()
        flow = ZeroModalFlow()
        simulate_case(
            model,
            flow,
            definitions[2],
            end_time_s=0.01,
            output_interval_s=0.01,
            maximum_azimuth_step_deg=5.0,
        )
        self.assertEqual(set(flow.mode_requests), {(1,)})
        self.assertEqual(set(flow.retained_requests), {None})

    def test_full_space_energy_gate_uses_one_minus_residual(self):
        metadata = {
            "full_space_regression_audit": {
                "full_normalized_residual_energy_fraction": 0.0018,
                "full_normalized_captured_energy_fraction_one_minus_residual": 0.9982,
                "full_normalized_individual_modal_self_energy": [100.0, 200.0],
            },
            "full_space_energy_gate": {
                "target_fraction": 0.998,
                "captured_fraction": 0.9982,
                "passed": True,
                "modal_self_energy_sum_prohibited_for_gate": True,
            },
        }
        audit = _validate_full_space_energy_gate(metadata, 0.998)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["captured_fraction"], 0.9982)

        metadata["full_space_energy_gate"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "below required"):
            _validate_full_space_energy_gate(metadata, 0.998)

    def test_physical_trim_closes_weight_and_initializes_momentum_inflow(self):
        trim = build_equilibrium_model().trim
        self.assertLess(abs(trim.relative_thrust_residual), 1.0e-8)
        expected = math.sqrt(
            trim.weight_N / (2.0 * 1.225 * math.pi * (synthetic_rotor()["diameter_m"] / 2) ** 2)
        )
        self.assertAlmostEqual(trim.induced_velocity_mps, expected, places=12)
        self.assertEqual(trim.alpha_domain_violation_count, 0)

    def test_still_air_trim_is_stationary_and_stage_is_forwarded(self):
        model = build_equilibrium_model()
        flow = ZeroModalFlow()
        definition = CaseDefinition("stage_07_retain_23", 7, 23, "RVMD", 2.0, 1.3716)
        result = simulate_case(
            model,
            flow,
            definition,
            end_time_s=0.02,
            output_interval_s=0.01,
            maximum_azimuth_step_deg=5.0,
        )
        self.assertTrue(flow.retained_requests)
        self.assertEqual(set(flow.retained_requests), {23})
        for row in result.rows:
            self.assertAlmostEqual(float(row["heave_m"]), 0.0, places=10)
            self.assertAlmostEqual(float(row["roll_deg"]), 0.0, places=10)
            self.assertAlmostEqual(float(row["pitch_deg"]), 0.0, places=10)
            self.assertAlmostEqual(float(row["delta_lambda_0"]), 0.0, places=10)

    def test_azimuth_mean_trim_closes_weight_and_dynamic_inflow_equations(self):
        model = build_equilibrium_model()
        report = solve_azimuth_mean_hover_trim(
            model,
            UniformModalFlow([0.8, -0.3, -0.7]),
            phase_samples=24,
        )
        self.assertTrue(report["success"], report["solver_message"])
        self.assertLess(report["maximum_abs_scaled_residual"], 1.0e-8)
        self.assertLess(abs(report["force_minus_weight_body_N"][2]), 1.0e-5)
        self.assertLess(
            max(abs(value) for value in report["mean_inflow_equation_residual"]),
            1.0e-10,
        )
        self.assertGreater(
            float(np.linalg.norm(report["delta_inflow_state"])), 1.0e-8
        )
        self.assertFalse(report["periodic_orbit_shooting"])

    def test_hover_hold_generalized_loads_are_restoring(self):
        model = build_equilibrium_model()
        model.hover_hold_vertical_stiffness_Npm = 100.0
        model.hover_hold_vertical_rate_gain_Ns_pm = 20.0
        model.hover_hold_angular_stiffness_Nm_per_rad = np.array([30.0, 40.0, 0.0])
        model.hover_hold_angular_rate_gain_Nms = np.array([3.0, 4.0, 0.0])
        body = np.array([0.2, 0.3, 0.1, -0.2, 0.4, -0.5])
        force, moment = _hover_hold_generalized_loads(model, body)
        np.testing.assert_allclose(force, [0.0, 0.0, -26.0])
        np.testing.assert_allclose(moment, [-4.2, 10.0, 0.0])

    def test_repeated_cases_start_from_identical_state(self):
        model = build_equilibrium_model()
        definition = CaseDefinition("stage_00_retain_30", 0, 30, "RVMD", 3.0, 2.0574)
        first = simulate_case(model, ZeroModalFlow(), definition, end_time_s=0.02, output_interval_s=0.01, maximum_azimuth_step_deg=5.0)
        second = simulate_case(model, ZeroModalFlow(), definition, end_time_s=0.02, output_interval_s=0.01, maximum_azimuth_step_deg=5.0)
        columns = ("lift_N", "engine_power_W", "roll_deg", "pitch_deg", "heave_m")
        for column in columns:
            np.testing.assert_allclose(
                [float(row[column]) for row in first.rows],
                [float(row[column]) for row in second.rows],
                rtol=0.0,
                atol=0.0,
            )

    def test_output_grid_and_statistics_window_are_explicit(self):
        np.testing.assert_allclose(output_times(0.025, 0.01), [0.0, 0.01, 0.02, 0.025])
        model = build_equilibrium_model()
        definition = CaseDefinition("stage_30_retain_00", 30, 0, "RVMD", None, None)
        result = simulate_case(model, ZeroModalFlow(), definition, end_time_s=0.03, output_interval_s=0.01, maximum_azimuth_step_deg=5.0)
        rows = statistics_rows(result, 0.01, 0.03, 2.0)
        self.assertEqual(len(rows), 25)
        self.assertEqual({row["sample_count"] for row in rows}, {3})

    def test_atomic_csv_roundtrip_preserves_case_identity(self):
        model = build_equilibrium_model()
        definition = CaseDefinition("stage_01_retain_29", 1, 29, "RVMD", 2.9, 1.98882)
        result = simulate_case(model, ZeroModalFlow(), definition, end_time_s=0.01, output_interval_s=0.01, maximum_azimuth_step_deg=5.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.csv"
            write_case_csv(path, result)
            loaded = read_case_csv(path, definition)
            self.assertEqual(len(loaded.rows), len(result.rows))
            self.assertEqual(loaded.definition.retained_mode_count, 29)


if __name__ == "__main__":
    unittest.main()
