# Copyright (c) 2026 Olegequ.
from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from airwake_rotor.plot_rvmd_3d_results import (
    METRICS,
    adjacent_effect_table,
    dominant_mode_table,
    plot_rvmd_convergence,
    reconstruction_response_closure,
)


def _statistics(rotor: str = "Rotor A", mode_count: int = 30) -> pd.DataFrame:
    rows = []
    width = max(2, len(str(mode_count)))
    for statistic in ("mean", "peak_to_peak"):
        for stage in range(mode_count + 1):
            for metric_index, metric in enumerate(METRICS):
                value = float(stage + 10 * metric_index)
                if statistic == "peak_to_peak":
                    value += 100.0
                rows.append(
                    {
                        "rotor_name": rotor,
                        "rotor_diameter_m": 10.0,
                        "case_id": "stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
                            stage=stage, retained=mode_count - stage, width=width
                        ),
                        "flow_kind": "RVMD",
                        "stage_index": stage,
                        "retained_mode_count": mode_count - stage,
                        "analysis_start_s": 1.0,
                        "analysis_end_s": 2.0,
                        "metric": metric,
                        "statistic": statistic,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def _trim(rotor: str = "Rotor A") -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "rotor_name": rotor,
            "weight_N": 100.0,
            "trim_engine_power_W": 200.0,
            "collective_trim_deg": 10.0,
            "induced_velocity_trim_mps": 5.0,
        }]
    )


class ResultTableTests(unittest.TestCase):
    def test_adjacent_deletion_maps_stage_one_to_mode_30(self) -> None:
        effects = adjacent_effect_table(_statistics(), _trim(), "mean")
        lift = effects[effects["metric"] == "lift_N"].sort_values("stage_index")
        self.assertEqual(int(lift.iloc[0]["deleted_mode"]), 30)
        self.assertEqual(int(lift.iloc[-1]["deleted_mode"]), 1)
        np.testing.assert_allclose(lift["normalized_adjacent_effect"], 1.0)
        power = effects[effects["metric"] == "engine_power_W"]
        np.testing.assert_allclose(power["normalized_adjacent_effect"], 0.5)

    def test_adjacent_deletion_supports_energy_selected_k160(self) -> None:
        effects = adjacent_effect_table(_statistics(mode_count=160), _trim(), "mean")
        lift = effects[effects["metric"] == "lift_N"].sort_values("stage_index")
        self.assertEqual(len(lift), 160)
        self.assertEqual(int(lift.iloc[0]["deleted_mode"]), 160)
        self.assertEqual(int(lift.iloc[-1]["deleted_mode"]), 1)
        np.testing.assert_allclose(lift["normalized_adjacent_effect"], 1.0)

    def test_cumulative_plots_ignore_single_and_leave_one_out_rows(self) -> None:
        statistics = _statistics()
        extras = []
        for kind in ("single", "leave_one_out"):
            for metric in METRICS:
                extras.append(
                    {
                        "rotor_name": "Rotor A",
                        "rotor_diameter_m": 10.0,
                        "case_id": "sensitivity_{}_M001".format(kind),
                        "flow_kind": "RVMD",
                        "stage_index": "",
                        "retained_mode_count": "",
                        "analysis_start_s": 1.0,
                        "analysis_end_s": 2.0,
                        "metric": metric,
                        "statistic": "mean",
                        "value": 123.0,
                    }
                )
        statistics = pd.concat([statistics, pd.DataFrame(extras)], ignore_index=True)
        effects = adjacent_effect_table(statistics, _trim(), "mean")
        self.assertEqual(len(effects), 30 * len(METRICS))

    def test_dominant_table_uses_largest_absolute_adjacent_change(self) -> None:
        mean_effects = adjacent_effect_table(_statistics(), _trim(), "mean")
        variability_effects = adjacent_effect_table(_statistics(), _trim(), "peak_to_peak")
        mask = (mean_effects["metric"] == "roll_deg") & (mean_effects["deleted_mode"] == 7)
        mean_effects.loc[mask, "normalized_adjacent_effect"] = -99.0
        runner_mask = (mean_effects["metric"] == "roll_deg") & (
            mean_effects["deleted_mode"] == 8
        )
        mean_effects.loc[runner_mask, "normalized_adjacent_effect"] = 95.0
        frequency = pd.DataFrame(
            {
                "mode": np.arange(1, 31),
                "frequency_hz": np.arange(1, 31, dtype=float) / 10.0,
                "strouhal_St_f13.716_over_20": np.arange(1, 31, dtype=float) / 20.0,
            }
        )
        dominant = dominant_mode_table(mean_effects, variability_effects, frequency)
        chosen = dominant[(dominant["effect_kind"] == "time_mean") & (dominant["metric"] == "roll_deg")]
        self.assertEqual(int(chosen.iloc[0]["dominant_deleted_mode"]), 7)
        self.assertEqual(int(chosen.iloc[0]["runner_up_deleted_mode"]), 8)
        self.assertAlmostEqual(float(chosen.iloc[0]["dominant_frequency_hz"]), 0.7)
        self.assertAlmostEqual(float(chosen.iloc[0]["signed_normalized_adjacent_effect"]), -99.0)
        self.assertAlmostEqual(float(chosen.iloc[0]["dominance_ratio_top_over_runner_up"]), 99.0 / 95.0)
        self.assertTrue(bool(chosen.iloc[0]["near_tie_top_over_runner_up_below_1p10"]))

    def test_reconstruction_closure_uses_only_common_analysis_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "04_timeseries_csv"
            output.mkdir(parents=True)
            times = np.asarray([0.0, 1.0, 2.0, 3.0])
            for case_id, offset, outside in (
                ("stage_00_retain_30", 1.0, 1000.0),
                ("original_cfd", 0.0, -1000.0),
            ):
                frame = pd.DataFrame({"time_s": times})
                for metric_index, metric in enumerate(METRICS):
                    frame[metric] = np.asarray(
                        [outside, 1.0 + metric_index + offset, 3.0 + metric_index + offset, outside]
                    )
                frame.to_csv(output / "Rotor_A__{}.csv".format(case_id), index=False, encoding="utf-8-sig")
            closure = reconstruction_response_closure(root, _statistics(), _trim())
            self.assertTrue(np.all(closure["sample_count"] == 2))
            np.testing.assert_allclose(closure["rmse"], 1.0)
            np.testing.assert_allclose(closure["analysis_start_s"], 1.0)
            np.testing.assert_allclose(closure["analysis_end_s"], 2.0)

    def test_convergence_plot_exports_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rvmd = root / "02_rvmd"
            output = root / "figures"
            rvmd.mkdir(parents=True)
            difference = np.asarray([1.0, 0.1, 0.004], dtype=np.float32)
            omega = np.tile(np.linspace(0.001, 0.03, 30)[:, None], (1, 4)).astype(np.float32)
            np.savez(
                rvmd / "temporal_model.npz",
                difference_history=difference,
                omega_history=omega,
                info_json=np.asarray(json.dumps({"dt_seconds": 0.01, "tolerance": 0.005})),
            )
            files, source = plot_rvmd_convergence(root, output, ("png",), 72)
            self.assertEqual(len(files), 1)
            self.assertTrue(Path(files[0]).is_file())
            self.assertEqual(len(source), 4)
            self.assertEqual(source.iloc[0]["difference_after_sweep"], "")
            self.assertAlmostEqual(float(source.iloc[-1]["difference_after_sweep"]), 0.004)


if __name__ == "__main__":
    unittest.main()
