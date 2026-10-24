# Copyright (c) 2026 Olegequ.
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from airwake_rotor.analyze_normalized_modal_sensitivity import (
    analyze_normalized_modal_sensitivity,
    frequency_spectrum_metrics,
)
from airwake_rotor.select_rvmd_mode_count import select_mode_count
from airwake_rotor.plot_normalized_modal_sensitivity import plot_normalized_modal_sensitivity


class AdaptiveModeCountTests(unittest.TestCase):
    @staticmethod
    def _candidate(root: Path, k: int, residual: float) -> Path:
        directory = root / ("K{}".format(k))
        directory.mkdir()
        audit = {
            "k": k,
            "full_normalized_centered_data_energy": 100.0,
            "full_normalized_residual_energy": residual * 100.0,
            "full_normalized_residual_energy_fraction": residual,
            "full_normalized_captured_energy_fraction_one_minus_residual": 1.0 - residual,
        }
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "k": k,
                    "rvmd_info": {"converged": True},
                    "full_space_regression_audit": audit,
                }
            ),
            encoding="utf-8",
        )
        (directory / "full_space_regression_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        return directory

    def test_adjacent_fail_pass_demonstrates_strict_energy_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            k5 = self._candidate(root, 5, 0.0021)
            k6 = self._candidate(root, 6, 0.0019)
            report = select_mode_count([k6, k5], root / "selection", target=0.998)
            selection = report["selection"]
            self.assertTrue(selection["strict_minimum_demonstrated"])
            self.assertEqual(selection["strict_minimum_k"], 6)
            self.assertFalse(report["criterion"]["modal_self_energy_sum_used_for_selection"])
            metadata_path = k6 / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["rvmd_info"]["converged"] = False
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            unconverged = select_mode_count(
                [k6], root / "selection_unconverged", target=0.998
            )
            self.assertIsNone(unconverged["selection"]["minimum_evaluated_passing_k"])


class NormalizedSensitivityTests(unittest.TestCase):
    def test_complete_single_and_leave_one_out_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rvmd = root / "rvmd"
            rvmd.mkdir()
            pd.DataFrame(
                {
                    "mode": [1, 2],
                    "frequency_hz": [0.1, 0.2],
                    "strouhal_St_f13.716_over_20": [0.06858, 0.13716],
                    "full_normalized_modal_self_energy": [1.0, 4.0],
                }
            ).to_csv(rvmd / "frequencies.csv", index=False)
            (rvmd / "full_space_regression_audit.json").write_text(
                json.dumps(
                    {
                        "full_normalized_centered_data_energy": 10.0,
                        "full_normalized_captured_energy_fraction_one_minus_residual": 0.999,
                    }
                ),
                encoding="utf-8",
            )
            time = np.linspace(0.0, 2.0, 9)
            mode1 = np.sin(2.0 * np.pi * 0.5 * time)
            mode2 = 0.4 * np.cos(2.0 * np.pi * time)
            signals = {
                ("mean", None): np.zeros_like(time),
                ("full", None): mode1 + mode2,
                ("single", 1): mode1,
                ("single", 2): mode2,
                ("leave_one_out", 1): mode2,
                ("leave_one_out", 2): mode1,
            }
            manifest_rows = []
            for (kind, mode), signal in signals.items():
                path = root / ("{}_{}.csv".format(kind, "all" if mode is None else mode))
                pd.DataFrame(
                    {
                        "time_s": time,
                        "rotor_name": ["R"] * len(time),
                        "rotor_diameter_m": [4.0] * len(time),
                        "roll_deg": signal,
                    }
                ).to_csv(path, index=False)
                manifest_rows.append(
                    {
                        "rotor_name": "R",
                        "rotor_diameter_m": 4.0,
                        "case_kind": kind,
                        "mode_index_one_based": "" if mode is None else mode,
                        "csv_path": str(path),
                    }
                )
            manifest = root / "manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "analysis_window_s": [0.0, 2.0],
                        "reference_length_m": 13.716,
                        "primary_estimator": "leave_one_out",
                        "response_metrics": {
                            "roll_deg": {
                                "unit": "deg",
                                "normalization_strategy": "constant",
                                "normalization_value": 12.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            audit = analyze_normalized_modal_sensitivity(manifest, rvmd, config, output)
            self.assertEqual(audit["mode_count"], 2)
            long = pd.read_csv(output / "mode_sensitivity_long.csv")
            self.assertEqual(len(long), 2)
            self.assertAlmostEqual(float(long["sensitivity_L1_primary"].sum()), 1.0)
            self.assertAlmostEqual(
                float(long["response_sensitivity_L1_primary"].sum()), 1.0
            )
            metrics = pd.read_csv(output / "diameter_frequency_metrics.csv")
            self.assertEqual(set(metrics["estimator"]), {"single", "leave_one_out"})
            self.assertTrue(np.all(np.isfinite(metrics["St90"])))
            self.assertTrue(np.all(np.isfinite(metrics["response_St90"])))
            figure_audit = plot_normalized_modal_sensitivity(
                output / "mode_sensitivity_long.csv",
                output / "diameter_frequency_metrics.csv",
                output / "figures",
                dpi=80,
            )
            self.assertEqual(figure_audit["status"], "pass")
            self.assertTrue((output / "figures" / "diameter_vs_St90.svg").is_file())
            fits = pd.read_csv(
                output / "figures" / "diameter_frequency_power_law_fits.csv"
            )
            self.assertEqual(
                set(fits["spectrum_basis"]),
                {"actual_response_contribution_R", "unit_modal_energy_gain_G"},
            )

    def test_frequency_metrics_report_weighted_upper_band(self):
        result = frequency_spectrum_metrics([0.1, 0.2, 0.4], [0.1, 0.2, 0.7])
        self.assertEqual(result["q50"], 0.4)
        self.assertEqual(result["q90"], 0.4)
        self.assertGreater(result["geometric_centroid"], 0.2)


if __name__ == "__main__":
    unittest.main()
