# Copyright (c) 2026 Olegequ.
#!/usr/bin/env python
"""Verification tests for the independent nonlinear dynamics core."""

import math
import unittest

import numpy as np

from airwake_rotor.nonlinear_rotorcraft_dynamics import (
    BodyLoads,
    DynamicInflowParameters,
    DynamicsConfigurationError,
    DynamicsDiagnostics,
    DynamicsDomainError,
    RigidBody3DOFParameters,
    ThreeStateDynamicInflow,
    body_angular_velocity,
    disk_average_delta_mu_z,
    euler_rate_projection,
    integrate_step,
    integrate_element_loads_about_hub,
    moment_about_cg_components,
    moving_tilted_disk_geometry,
    physical_hover_induced_velocity,
    physical_hover_trim_reference,
    rigid_body_3dof_rhs,
    rotation_body_to_world,
)


class DynamicInflowTests(unittest.TestCase):
    def setUp(self):
        self.rho = 1.225
        self.radius = 5.0
        self.omega = 30.0
        self.reference_thrust = 55000.0
        self.model = ThreeStateDynamicInflow(
            DynamicInflowParameters(
                radius_m=self.radius,
                omega_radps=self.omega,
                density_kgpm3=self.rho,
                reference_thrust_N=self.reference_thrust,
            )
        )

    def test_uniform_hover_is_an_equilibrium(self):
        thrust = self.reference_thrust
        state = self.model.state_from_hover_thrust(thrust)
        expected_vi = physical_hover_induced_velocity(thrust, self.rho, self.radius)
        self.assertAlmostEqual(self.model.lambda_star * self.omega * self.radius,
                               expected_vi, places=12)
        np.testing.assert_allclose(
            self.model.dimensional_induced_velocity(
                state, np.array([0.2, 0.7, 1.0]), np.array([0.0, 1.2, 4.0])
            ),
            expected_vi,
            rtol=0.0,
            atol=2.0e-13,
        )
        derivative = self.model.rhs(0.0, state, np.zeros(3))
        np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=2.0e-14)

    def test_hover_ct_step_matches_frozen_coefficient_analytic_solution(self):
        # M*d(delta_lambda)/dt = Omega*(delta_c-K*delta_lambda).
        thrust_1 = 60000.0
        state = self.model.initial_perturbation_state
        delta_coefficients = self.model.coefficient_perturbation_from_total_loads(
            thrust_1, np.zeros(3)
        )
        horizon = 0.08
        dt = 1.0e-4
        rhs = lambda t, value: self.model.rhs(t, value, delta_coefficients)
        time = 0.0
        while time < horizon - 0.5 * dt:
            state = integrate_step(rhs, time, state, dt, method="rk4")
            time += dt
        analytic = self.model.exact_frozen_forcing_step(
            self.model.initial_perturbation_state,
            delta_coefficients,
            delta_mu_z=0.0,
            dt_s=horizon,
        )
        np.testing.assert_allclose(state, analytic, rtol=0.0, atol=2.0e-12)

    def test_all_three_channels_match_frozen_forcing_analytic_solution(self):
        initial = np.array([0.001, -0.0005, 0.0003])
        forcing = np.array([0.002, -0.00015, 0.00021])
        delta_mu_z = -0.002
        horizon = 0.025
        dt = 2.5e-5
        state = initial.copy()
        rhs = lambda t, value: self.model.rhs(
            t, value, forcing, delta_mu_z=delta_mu_z
        )
        time = 0.0
        while time < horizon - 0.5 * dt:
            state = integrate_step(rhs, time, state, dt, method="rk4")
            time += dt
        analytic = self.model.exact_frozen_forcing_step(
            initial, forcing, delta_mu_z, horizon
        )
        np.testing.assert_allclose(state, analytic, rtol=0.0, atol=4.0e-13)

    def test_hub_moment_direction_signs(self):
        state = self.model.initial_perturbation_state
        mx_delta = self.model.coefficient_perturbation_from_total_loads(
            self.reference_thrust, [1200.0, 0.0, 0.0]
        )
        my_delta = self.model.coefficient_perturbation_from_total_loads(
            self.reference_thrust, [0.0, 1200.0, 0.0]
        )
        mx_positive = self.model.rhs(0.0, state, mx_delta)
        my_positive = self.model.rhs(0.0, state, my_delta)
        self.assertGreater(mx_positive[2], 0.0)  # +Mx -> +lambda_s
        self.assertLess(my_positive[1], 0.0)     # +My -> -lambda_c

    def test_axial_external_flow_correction_has_required_static_gain(self):
        delta_mu_z = 0.01
        delta_ct = 0.025
        delta_coefficients = np.array([delta_ct, 0.0, 0.0])
        equilibrium = np.array(
            [delta_ct / (4.0 * self.model.lambda_star) - 0.5 * delta_mu_z,
             0.0, 0.0]
        )
        derivative = self.model.rhs(
            0.0, equilibrium, delta_coefficients, delta_mu_z=delta_mu_z
        )
        np.testing.assert_allclose(derivative, 0.0, atol=2.0e-15, rtol=0.0)

    def test_external_axial_flow_direction_sign(self):
        normal = np.array([0.0, 0.0, 1.0])
        downward = np.array([[[0.0, 0.0, -3.0], [0.0, 0.0, -1.0]]])
        upward = -downward
        weights = np.array([[1.0, 3.0]])
        self.assertAlmostEqual(
            disk_average_delta_mu_z(downward, normal, 100.0, weights), 0.015
        )
        self.assertAlmostEqual(
            disk_average_delta_mu_z(upward, normal, 100.0, weights), -0.015
        )

    def test_large_perturbation_hard_fails(self):
        with self.assertRaises(DynamicsDomainError):
            self.model.rhs(
                0.0,
                np.array([self.model.lambda_star, 0.0, 0.0]),
                np.zeros(3),
            )

    def test_metadata_records_matrix_time_scale_and_state_contract(self):
        metadata = self.model.metadata()
        self.assertEqual(
            metadata["state_order"],
            ["delta_lambda_0", "delta_lambda_c", "delta_lambda_s"],
        )
        self.assertIn("Omega", metadata["time_variable"])
        np.testing.assert_allclose(
            metadata["apparent_mass_diagonal"],
            [128.0 / (75.0 * math.pi), 64.0 / (45.0 * math.pi),
             64.0 / (45.0 * math.pi)],
            atol=0.0,
            rtol=0.0,
        )


class RigidBodyTests(unittest.TestCase):
    def setUp(self):
        self.mass = 1000.0
        self.gravity = 9.80665
        self.inertia = np.array(
            [[1200.0, -35.0, -20.0], [-35.0, 1800.0, -15.0], [-20.0, -15.0, 2100.0]]
        )
        self.parameters = RigidBody3DOFParameters(
            mass_kg=self.mass,
            inertia_body_kgm2=self.inertia,
            hub_from_cg_body_m=np.array([0.2, -0.1, 1.5]),
            maximum_abs_attitude_deg=85.0,
        )

    def test_large_angle_force_is_projected_into_world_vertical(self):
        phi, theta = math.radians(60.0), math.radians(45.0)
        thrust = 12000.0
        state = np.array([0.0, 0.0, phi, theta, 0.0, 0.0])
        loads = BodyLoads(
            np.array([0.0, 0.0, thrust]), np.zeros(3), np.zeros(3)
        )
        derivative = rigid_body_3dof_rhs(0.0, state, self.parameters, loads)
        expected = thrust * math.cos(phi) * math.cos(theta) / self.mass - self.gravity
        self.assertAlmostEqual(derivative[1], expected, places=12)

    def test_small_angle_limit_recovers_linear_roll_pitch(self):
        parameters = RigidBody3DOFParameters(
            mass_kg=self.mass,
            inertia_body_kgm2=np.diag([1200.0, 1800.0, 2100.0]),
            hub_from_cg_body_m=np.zeros(3),
        )
        mx, my = 240.0, -360.0
        state = np.zeros(6)
        loads = BodyLoads(
            np.array([0.0, 0.0, self.mass * self.gravity]),
            np.array([mx, my, 0.0]),
            np.array([mx, my, 0.0]),
        )
        derivative = rigid_body_3dof_rhs(0.0, state, parameters, loads)
        self.assertAlmostEqual(derivative[1], 0.0, places=13)
        self.assertAlmostEqual(derivative[4], mx / 1200.0, places=13)
        self.assertAlmostEqual(derivative[5], my / 1800.0, places=13)

    def test_projected_euler_equation_retains_inertia_coupling(self):
        phi, theta = 0.43, -0.31
        phi_dot, theta_dot = 0.8, -0.55
        state = np.array([0.0, 0.0, phi, theta, phi_dot, theta_dot])
        moment = np.array([90.0, -110.0, 35.0])
        loads = BodyLoads(
            np.array([0.0, 0.0, self.mass * self.gravity]), moment, moment
        )
        derivative = rigid_body_3dof_rhs(0.0, state, self.parameters, loads)
        qdot = np.array([phi_dot, theta_dot])
        qddot = derivative[4:6]
        projection = euler_rate_projection(phi)
        projection_dot = np.array(
            [[0.0, 0.0], [0.0, -math.sin(phi) * phi_dot],
             [0.0, -math.cos(phi) * phi_dot]]
        )
        omega = np.dot(projection, qdot)
        omega_dot = np.dot(projection, qddot) + np.dot(projection_dot, qdot)
        _, _, total_rotor_moment = moment_about_cg_components(
            loads.rotor_force_body_N,
            loads.transmitted_hub_moment_body_Nm,
            self.parameters.hub_from_cg_body_m,
        )
        residual = np.dot(
            projection.T,
            np.dot(self.inertia, omega_dot)
            + np.cross(omega, np.dot(self.inertia, omega))
            - total_rotor_moment,
        )
        np.testing.assert_allclose(residual, 0.0, atol=2.0e-12, rtol=0.0)

    def test_hub_and_force_arm_moments_remain_separate(self):
        force = np.array([100.0, -50.0, 9000.0])
        hub_moment = np.array([11.0, 22.0, -33.0])
        hub, arm, total = moment_about_cg_components(
            force, hub_moment, self.parameters.hub_from_cg_body_m
        )
        np.testing.assert_array_equal(hub, hub_moment)
        np.testing.assert_allclose(
            arm, np.cross(self.parameters.hub_from_cg_body_m, force),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(total, hub + arm, atol=0.0, rtol=0.0)

    def test_full_disk_bending_moment_is_not_implicitly_transmitted(self):
        parameters = RigidBody3DOFParameters(
            mass_kg=self.mass,
            inertia_body_kgm2=np.diag([1200.0, 1800.0, 2100.0]),
            hub_from_cg_body_m=np.zeros(3),
        )
        full_disk_moment = np.array([8000.0, -5000.0, 200.0])
        loads = BodyLoads(
            np.array([0.0, 0.0, self.mass * self.gravity]),
            full_disk_moment,
            np.zeros(3),  # explicit ideal hinge: no bending moment transmitted
        )
        derivative = rigid_body_3dof_rhs(0.0, np.zeros(6), parameters, loads)
        np.testing.assert_allclose(derivative, 0.0, atol=0.0, rtol=0.0)

    def test_element_loads_produce_unscaled_hub_moment(self):
        hub = np.array([2.0, -1.0, 0.5])
        points = np.array(
            [[[3.0, -1.0, 0.5], [2.0, 1.0, 0.5]],
             [[1.0, -1.0, 0.5], [2.0, -3.0, 0.5]]]
        )
        forces = np.array(
            [[[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
             [[0.0, 0.0, 30.0], [0.0, 0.0, 40.0]]]
        )
        resultant, hub_moment = integrate_element_loads_about_hub(
            points, forces, hub
        )
        np.testing.assert_array_equal(resultant, [0.0, 0.0, 100.0])
        expected = np.sum(
            np.cross(points.reshape((-1, 3)) - hub, forces.reshape((-1, 3))), axis=0
        )
        np.testing.assert_array_equal(hub_moment, expected)

    def test_physical_hover_trim_is_stationary_and_not_mean_removal(self):
        hub_moment = np.array([30.0, -40.0, 10.0])
        transmitted_moment = np.array([5.0, -7.0, 10.0])
        inflow = ThreeStateDynamicInflow(
            DynamicInflowParameters(
                5.0, 30.0, 1.225,
                reference_thrust_N=self.mass * self.gravity,
                reference_hub_aero_moment_disk_Nm=hub_moment,
            )
        )
        force = np.array([75.0, -120.0, self.mass * self.gravity])
        trim = physical_hover_trim_reference(
            self.parameters, inflow, force, hub_moment, transmitted_moment
        )
        loads = BodyLoads(
            force,
            hub_moment,
            transmitted_moment,
            trim.other_force_body_N,
            trim.other_moment_body_Nm,
        )
        derivative = rigid_body_3dof_rhs(0.0, np.zeros(6), self.parameters, loads)
        np.testing.assert_allclose(derivative, 0.0, atol=2.0e-15, rtol=0.0)
        self.assertGreater(trim.induced_velocity_mps, 0.0)

    def test_configuration_requires_measured_or_declared_izz(self):
        with self.assertRaises(DynamicsConfigurationError):
            RigidBody3DOFParameters.from_mapping(
                {"mass_kg": 1.0, "Ixx_kgm2": 1.0, "Iyy_kgm2": 1.0,
                 "hub_above_cg_m": 1.0}
            )

    def test_attitude_safety_limit_hard_fails(self):
        state = np.array([0.0, 0.0, math.radians(86.0), 0.0, 0.0, 0.0])
        loads = BodyLoads(np.zeros(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(DynamicsDomainError):
            rigid_body_3dof_rhs(0.0, state, self.parameters, loads)


class GeometryAndIntegratorTests(unittest.TestCase):
    def test_moving_tilted_disk_geometry(self):
        state = np.array([1.25, -0.4, math.radians(25.0), math.radians(-15.0), 0.2, -0.1])
        nominal_cg = np.array([150.0, 0.0, 10.0])
        hub_offset = np.array([0.0, 0.0, 1.2])
        radii = np.array([1.0, 2.0, 3.0])
        geometry = moving_tilted_disk_geometry(
            state, nominal_cg, hub_offset, [0.0, 0.0, 1.0], radii,
            blade_count=4, phase_rad=0.3, rotor_omega_radps=20.0
        )
        rotation = rotation_body_to_world(state[2], state[3])
        expected_normal = np.dot(rotation, np.array([0.0, 0.0, 1.0]))
        expected_hub = nominal_cg + np.array([0.0, 0.0, state[0]]) + np.dot(rotation, hub_offset)
        np.testing.assert_allclose(geometry.normal_world, expected_normal, atol=1.0e-14)
        np.testing.assert_allclose(geometry.hub_world_m, expected_hub, atol=1.0e-14)
        relative = geometry.points_world_m - geometry.hub_world_m
        # Every element remains in the tilted disk and at its declared radius.
        np.testing.assert_allclose(np.dot(relative, geometry.normal_world), 0.0, atol=2.0e-14)
        np.testing.assert_allclose(
            np.linalg.norm(relative, axis=2),
            np.broadcast_to(radii, (4, len(radii))),
            atol=2.0e-14,
        )
        self.assertTrue(np.all(np.isfinite(geometry.blade_point_velocity_world_mps)))

    def test_rk4_and_heun_interfaces(self):
        rhs = lambda t, y: -2.0 * y
        exact = math.exp(-0.2)
        rk4 = integrate_step(rhs, 0.0, np.array([1.0]), 0.1, method="rk4")[0]
        heun = integrate_step(rhs, 0.0, np.array([1.0]), 0.1, method="heun")[0]
        self.assertAlmostEqual(rk4, exact, delta=3.0e-6)
        self.assertAlmostEqual(heun, exact, delta=1.5e-3)


if __name__ == "__main__":
    unittest.main()
