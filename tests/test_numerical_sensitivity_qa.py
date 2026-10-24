# Copyright (c) 2026 Olegequ.
import types
import unittest

import numpy as np

from airwake_rotor.run_numerical_sensitivity_qa import (
    _compare_steps,
    _full_case,
    _representative_rotors,
    _variant_model,
)
from tests.test_rvmd_nato_coupling_pipeline import build_equilibrium_model


class NumericalSensitivityQATests(unittest.TestCase):
    def test_full_case_uses_actual_dynamic_mode_count(self):
        flow = types.SimpleNamespace(
            mode_count=160,
            frequency_hz=np.linspace(0.1, 2.0, 160),
            strouhal=np.linspace(0.06858, 1.3716, 160),
        )
        definition = _full_case(flow)
        self.assertEqual(definition.case_id, "stage_000_retain_160")
        self.assertEqual(definition.retained_mode_count, 160)

    def test_representatives_are_small_median_large_in_diameter_order(self):
        rotors = [
            {"name": "large", "diameter_m": 10.0},
            {"name": "small", "diameter_m": 2.0},
            {"name": "mid_low", "diameter_m": 4.0},
            {"name": "median", "diameter_m": 6.0},
            {"name": "mid_high", "diameter_m": 8.0},
        ]
        selected = _representative_rotors(rotors)
        self.assertEqual([item["name"] for item in selected], ["small", "median", "large"])

    def test_identical_step_histories_pass_every_metric(self):
        rows = []
        for index, time_s in enumerate((0.0, 1.0, 2.0)):
            rows.append(
                {
                    "time_s": time_s,
                    "lift_N": 100.0 + index,
                    "engine_power_W": 1000.0 + 2.0 * index,
                    "roll_deg": 0.1 * index,
                    "pitch_deg": -0.2 * index,
                    "heave_m": 0.01 * index,
                }
            )
        result = types.SimpleNamespace(rows=rows)
        model = types.SimpleNamespace(
            trim=types.SimpleNamespace(weight_N=100.0),
            mean_flow_trim_audit={"mean_engine_power_W": 1000.0},
        )
        compared = _compare_steps(
            "rotor", model, result, result, 5.0, 2.5, 0.0, 2.0, 0.02, 0.05
        )
        self.assertEqual(len(compared), 5)
        self.assertTrue(all(row["passes"] for row in compared))

    def test_sensitivity_variants_do_not_mutate_baseline(self):
        model = build_equilibrium_model()
        model.hover_hold_vertical_stiffness_Npm = 10.0
        model.hover_hold_vertical_rate_gain_Ns_pm = 5.0
        model.hover_hold_angular_stiffness_Nm_per_rad = np.array([3.0, 4.0, 0.0])
        model.hover_hold_angular_rate_gain_Nms = np.array([1.0, 2.0, 0.0])
        variant = _variant_model(model, "hover_hold_gains", 1.2)
        self.assertEqual(model.hover_hold_vertical_stiffness_Npm, 10.0)
        self.assertAlmostEqual(variant.hover_hold_vertical_stiffness_Npm, 12.0)
        np.testing.assert_allclose(model.hover_hold_angular_stiffness_Nm_per_rad, [3.0, 4.0, 0.0])
        np.testing.assert_allclose(variant.hover_hold_angular_stiffness_Nm_per_rad, [3.6, 4.8, 0.0])


if __name__ == "__main__":
    unittest.main()
