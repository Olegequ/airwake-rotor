# Copyright (c) 2026 Olegequ.
import unittest

import numpy as np
import pandas as pd

from airwake_rotor.analyze_modal_deletion_stability import analyze_case, summarize_rotor


class ModalDeletionStabilityTests(unittest.TestCase):
    def setUp(self):
        self.limits = {
            "maximum_abs_attitude_deg": 12.0,
            "maximum_abs_heave_m": 1.0,
            "maximum_abs_heave_speed_mps": 5.0,
        }
        self.metadata = {
            "dynamics_diagnostics": {
                "hard_failures": 0,
                "maximum_abs_attitude_deg": 2.0,
                "maximum_abs_delta_inflow_ratio": 0.01,
                "maximum_abs_delta_mu_z": 0.01,
            },
            "inflow_metadata": {
                "reference_lambda": 0.1,
            }
        }

    @staticmethod
    def frame(heave):
        time = np.linspace(0.0, 4.0, 41)
        return pd.DataFrame(
            {
                "time_s": time,
                "heave_m": np.asarray(heave(time), dtype=float),
                "heave_velocity_mps": np.zeros_like(time),
                "roll_deg": np.zeros_like(time),
                "pitch_deg": np.zeros_like(time),
                "delta_lambda_0": np.zeros_like(time),
                "delta_lambda_c": np.zeros_like(time),
                "delta_lambda_s": np.zeros_like(time),
                "delta_mu_z": np.zeros_like(time),
                "lambda_total_uniform": np.full_like(time, 0.1),
                "induced_velocity_min_mps": np.full_like(time, 1.0),
                "cl_saturation_count": np.zeros_like(time),
                "reverse_flow_count": np.zeros_like(time),
                "alpha_domain_violation_count": np.zeros_like(time),
                "mach_domain_violation_count": np.zeros_like(time),
                "reynolds_domain_violation_count": np.zeros_like(time),
            }
        )

    def test_bounded_case_remains_below_hard_limit(self):
        result = analyze_case(
            self.frame(lambda time: 0.25 * np.sin(time)),
            self.metadata,
            self.limits,
            0.0,
            4.0,
        )
        self.assertFalse(result["observed_hard_instability"])
        self.assertLessEqual(result["maximum_hard_limit_utilization"], 0.25 + 1.0e-12)

    def test_limit_crossing_is_observed_hard_instability(self):
        result = analyze_case(
            self.frame(lambda time: 0.3 * time),
            self.metadata,
            self.limits,
            0.0,
            4.0,
        )
        self.assertTrue(result["observed_hard_instability"])
        self.assertEqual(result["classification"], "observed_hard_instability")

    def test_model_domain_failure_is_not_relabeled_as_dynamic_instability(self):
        frame = self.frame(lambda time: 0.1 * np.sin(time))
        frame.loc[10, "reverse_flow_count"] = 1
        result = analyze_case(frame, self.metadata, self.limits, 0.0, 4.0)
        self.assertTrue(result["model_validity_failure"])
        self.assertFalse(result["observed_hard_instability"])
        self.assertEqual(result["classification"], "model_domain_invalid")

    def test_summary_reports_no_cutoff_when_every_stage_is_bounded(self):
        table = pd.DataFrame(
            {
                "rotor_name": ["R"] * 3,
                "rotor_diameter_m": [10.0] * 3,
                "stage_index": [0, 1, 2],
                "deleted_mode_this_stage": ["", 30, 29],
                "retained_mode_count": [30, 29, 28],
                "observed_hard_instability": [False, False, False],
                "terminal_growth_advisory": [False, True, False],
                "terminal_growth_warning_quantity_count": [0, 1, 0],
                "maximum_hard_limit_utilization": [0.1, 0.2, 0.15],
                "limiting_quantity": ["roll_deg", "heave_m", "pitch_deg"],
            }
        )
        result = summarize_rotor(table)
        self.assertEqual(result["hard_instability_cutoff_status"], "not_observed_through_stage_30")
        self.assertEqual(result["worst_stage_index"], 1)
        self.assertEqual(result["first_terminal_growth_advisory_stage"], 1)
        self.assertEqual(result["terminal_growth_advisory_quantity_count"], 1)
        self.assertEqual(result["sustained_hard_instability_cutoff_stage"], "")

    def test_summary_only_calls_a_cutoff_when_all_later_stages_are_unstable(self):
        table = pd.DataFrame(
            {
                "rotor_name": ["R"] * 4,
                "rotor_diameter_m": [10.0] * 4,
                "stage_index": [0, 1, 2, 3],
                "deleted_mode_this_stage": ["", 30, 29, 28],
                "retained_mode_count": [30, 29, 28, 27],
                "observed_hard_instability": [False, True, True, True],
                "terminal_growth_advisory": [False] * 4,
                "terminal_growth_warning_quantity_count": [0] * 4,
                "maximum_hard_limit_utilization": [0.1, 1.1, 0.2, 1.2],
                "limiting_quantity": ["roll_deg"] * 4,
            }
        )
        result = summarize_rotor(table)
        self.assertEqual(result["first_observed_hard_instability_stage"], 1)
        self.assertEqual(result["sustained_hard_instability_cutoff_stage"], 1)
        self.assertEqual(result["sustained_hard_instability_tail_length"], 3)

        table["observed_hard_instability"] = [False, False, False, True]
        boundary_result = summarize_rotor(table)
        self.assertEqual(boundary_result["first_observed_hard_instability_stage"], 3)
        self.assertEqual(boundary_result["sustained_hard_instability_cutoff_stage"], "")
        self.assertEqual(
            boundary_result["hard_instability_cutoff_status"],
            "hard_event_observed_but_no_sustained_cutoff",
        )


if __name__ == "__main__":
    unittest.main()
