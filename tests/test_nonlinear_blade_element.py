# Copyright (c) 2026 Olegequ.
#!/usr/bin/env python
"""Unit tests for the auditable nonlinear blade-element load module."""

import math
import unittest

import numpy as np

from airwake_rotor.nonlinear_blade_element import (
    AerodynamicDomainError,
    AerodynamicDomainLimits,
    BladeElementParameters,
    element_relative_velocity_world,
    integrate_blade_element_loads,
    radial_element_centres_and_widths,
)


class NonlinearBladeElementTests(unittest.TestCase):
    def setUp(self):
        self.parameters = BladeElementParameters(
            blade_count=4,
            radius_m=3.0,
            root_cutout_fraction=0.2,
            chord_m=0.25,
            omega_radps=24.0,
            collective_deg=11.0,
            twist_deg=-4.0,
            cd0=0.012,
            cl_alpha_per_rad=5.73,
            cl_max=1.35,
            transmission_efficiency=0.9,
            prandtl_tip_loss=True,
            prandtl_root_loss=True,
        )
        self.radii, self.widths = radial_element_centres_and_widths(
            self.parameters, 8
        )
        self.axis = np.array([0.0, 0.0, 1.0])
        self.hub = np.array([2.0, -1.0, 3.0])
        self.cg = np.array([2.0, -1.0, 1.5])
        self.azimuth = 2.0 * math.pi * np.arange(4) / 4.0
        self.radial = np.column_stack(
            (np.cos(self.azimuth), np.sin(self.azimuth), np.zeros(4))
        )
        self.tangential = np.column_stack(
            (-np.sin(self.azimuth), np.cos(self.azimuth), np.zeros(4))
        )
        self.positions = (
            self.hub[None, None, :]
            + self.radii[None, :, None] * self.radial[:, None, :]
        )
        self.rotor_velocity = (
            self.parameters.omega_radps
            * self.radii[None, :, None]
            * self.tangential[:, None, :]
        )

    def solve(self, **updates):
        arguments = {
            "parameters": self.parameters,
            "element_positions_world_m": self.positions,
            "cfd_velocity_world_mps": np.zeros_like(self.positions),
            "density_kgpm3": 1.225,
            "radii_m": self.radii,
            "element_widths_m": self.widths,
            "blade_radial_direction_world": self.radial,
            "blade_tangential_direction_world": self.tangential,
            "rotor_axis_world": self.axis,
            "hub_world_m": self.hub,
            "cg_world_m": self.cg,
            "body_translation_velocity_world_mps": np.zeros(3),
            "body_angular_velocity_world_radps": np.zeros(3),
            "rotor_tangential_velocity_world_mps": self.rotor_velocity,
            "induced_velocity_mps": 3.0,
            "blade_azimuth_rad": self.azimuth,
        }
        arguments.update(updates)
        return integrate_blade_element_loads(**arguments)

    def test_uniform_axisymmetric_load_has_no_transverse_hub_moment(self):
        loads = self.solve()
        self.assertGreater(loads.thrust_N, 0.0)
        self.assertGreater(loads.shaft_torque_Nm, 0.0)
        np.testing.assert_allclose(
            loads.hub_force_world_N[:2], np.zeros(2), atol=2.0e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            loads.hub_aerodynamic_moment_world_Nm[:2],
            np.zeros(2),
            atol=2.0e-12,
            rtol=0.0,
        )
        self.assertLess(loads.hub_aerodynamic_moment_world_Nm[2], 0.0)
        np.testing.assert_allclose(
            loads.cg_moment_force_line_only_world_Nm,
            loads.cg_moment_rigid_root_bound_world_Nm,
            atol=2.0e-12,
            rtol=0.0,
        )

    def test_force_line_closure_uses_right_handed_hub_arm_cross_force(self):
        # Weight blade 0 more strongly.  Its drag is along -y; a +z hub arm
        # therefore produces +x roll moment because z cross (-y) = +x.
        density = np.full((4, len(self.radii)), 1.225)
        density[0] *= 1.8
        loads = self.solve(density_kgpm3=density)
        expected = (
            np.cross(self.hub - self.cg, loads.hub_force_world_N)
            + loads.hub_axis_moment_world_Nm
        )
        np.testing.assert_allclose(
            loads.cg_moment_force_line_only_world_Nm,
            expected,
            atol=2.0e-12,
            rtol=0.0,
        )
        self.assertLess(loads.hub_force_world_N[1], 0.0)
        self.assertGreater(loads.cg_moment_force_line_only_world_Nm[0], 0.0)
        # The upper-bound closure differs by precisely the transverse root moment.
        transverse_hub = (
            loads.hub_aerodynamic_moment_world_Nm
            - loads.hub_axis_moment_world_Nm
        )
        np.testing.assert_allclose(
            loads.cg_moment_rigid_root_bound_world_Nm
            - loads.cg_moment_force_line_only_world_Nm,
            transverse_hub,
            atol=2.0e-12,
            rtol=0.0,
        )

    def test_translation_and_rotation_enter_blade_relative_air_with_correct_sign(self):
        positions = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=float
        )
        cfd = np.array(
            [[[4.0, -2.0, 0.5], [-1.0, 3.0, -0.25]]], dtype=float
        )
        cg = np.array([0.0, 0.0, 0.0])
        translation = np.array([0.7, -0.4, 0.2])
        angular = np.array([0.0, 0.0, 1.5])
        rotor = np.array(
            [[[0.0, 5.0, 0.0], [-6.0, 0.0, 0.0]]], dtype=float
        )
        induced = np.array([[2.0, 3.0]])
        actual = element_relative_velocity_world(
            positions,
            cfd,
            cg,
            translation,
            angular,
            rotor,
            np.array([0.0, 0.0, 1.0]),
            induced,
        )
        expected = (
            translation[None, None, :]
            + np.cross(np.broadcast_to(angular, positions.shape), positions - cg)
            + rotor
            - cfd
            + induced[:, :, None] * np.array([0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
        # Translating the aircraft +V is exactly equivalent to changing the CFD
        # velocity by -V, which guards against a common coupling sign error.
        no_translation = element_relative_velocity_world(
            positions,
            cfd - translation[None, None, :],
            cg,
            np.zeros(3),
            angular,
            rotor,
            self.axis,
            induced,
        )
        np.testing.assert_allclose(actual, no_translation, atol=1.0e-14, rtol=0.0)

    def test_signed_main_power_is_exactly_shaft_torque_times_omega(self):
        loads = self.solve()
        self.assertEqual(
            loads.shaft_power_W,
            loads.shaft_torque_Nm * self.parameters.omega_radps,
        )
        self.assertEqual(
            loads.engine_power_W,
            loads.shaft_power_W / self.parameters.transmission_efficiency,
        )
        self.assertEqual(
            loads.shaft_torque_Nm,
            -float(np.dot(loads.hub_aerodynamic_moment_world_Nm, self.axis)),
        )

    def test_domain_extrapolation_raises_with_auditable_counts(self):
        strict = BladeElementParameters(
            blade_count=self.parameters.blade_count,
            radius_m=self.parameters.radius_m,
            root_cutout_fraction=self.parameters.root_cutout_fraction,
            chord_m=self.parameters.chord_m,
            omega_radps=self.parameters.omega_radps,
            collective_deg=self.parameters.collective_deg,
            twist_deg=self.parameters.twist_deg,
            cd0=self.parameters.cd0,
            cl_alpha_per_rad=self.parameters.cl_alpha_per_rad,
            cl_max=self.parameters.cl_max,
            domain_limits=AerodynamicDomainLimits(mach_max=0.01),
        )
        with self.assertRaises(AerodynamicDomainError) as caught:
            self.solve(parameters=strict)
        self.assertGreater(caught.exception.diagnostics.mach_domain_violation_count, 0)
        self.assertEqual(
            caught.exception.diagnostics.extrapolated_element_count,
            caught.exception.diagnostics.mach_domain_violation_count,
        )


if __name__ == "__main__":
    unittest.main()
