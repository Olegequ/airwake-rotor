# Copyright (c) 2026 Olegequ.
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from airwake_rotor.nato_gd_study import (
    StudyConfigurationError,
    build_stage_rows,
    load_study_context,
    validate_grid_and_time,
)


class NatoGDStudyTests(unittest.TestCase):
    def _config(self, root):
        source = root / "输出20s-45s" / "空间流场"
        source.mkdir(parents=True)
        (root / "rotor_config.json").write_text("{}", encoding="utf-8")
        (root / "overrides.json").write_text("{}", encoding="utf-8")
        value = {
            "schema_version": "nato-gd-rvmd-3d-v1",
            "study": {
                "source_data_dir": str(source),
                "output_dir": "out",
                "cache_dir": "out/01_cache",
                "first_snapshot_id": 2001,
                "last_snapshot_id": 4500,
                "sample_interval_s": 0.01289304,
                "sample_interval_status": "user_specified",
                "mode_count": 30,
                "strouhal_reference_length_m": 13.716,
                "strouhal_reference_velocity_mps": 20.0,
            },
            "coordinates": {
                "initial_hub_world_m": [150.9, 0.0, 11.8],
                "hub_position_status": "confirmed",
                "out_of_bounds_policy": "hard_fail_no_clip_no_extrapolation",
            },
            "rvmd": {"mode_count": 30, "full_space_energy_target": 0.998},
            "rotor_model": {
                "base_rotor_config": "rotor_config.json",
                "dynamic_overrides": "overrides.json",
                "flapping_moment_transfer": "prohibited",
                "hub_moment_policy": "force_line_only",
            },
            "body_response": {
                "pre_roll_s": 1.0,
                "analysis_window_start_s": 1.0,
                "analysis_window_end_s": 11.0,
                "common_window_required": True,
            },
            "production_gate": {
                "require_confirmed_sample_interval": True,
                "require_confirmed_hub_world_position": True,
            },
        }
        path = root / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_context_records_time_label_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = load_study_context(self._config(root), project_root=root)
            self.assertTrue(context.production_ready)
            self.assertEqual(context.dt_s, 0.01289304)
            self.assertEqual(context.full_space_energy_target, 0.998)
            self.assertEqual(len(context.warnings), 1)
            self.assertIn("directory label implies", context.warnings[0])

    def test_inferred_hub_requires_explicit_provisional_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._config(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["coordinates"]["hub_position_status"] = "inferred_pending"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(StudyConfigurationError):
                load_study_context(path, project_root=root)
            context = load_study_context(path, project_root=root, allow_provisional=True)
            self.assertFalse(context.production_ready)

    def test_grid_and_common_window_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = load_study_context(self._config(root), project_root=root)
            audit = validate_grid_and_time(
                context,
                [[110.9, 190.9], [-35.0, 35.0], [9.8, 13.8]],
                2500,
                [18.6, 3.4],
            )
            self.assertTrue(audit["passed"])
            with self.assertRaises(StudyConfigurationError):
                validate_grid_and_time(context, [[145, 155], [-2, 2], [9.8, 13.8]], 2500, [18.6])

    def test_stage_semantics_delete_highest_first(self):
        frequency = np.arange(1.0, 31.0)
        rows = build_stage_rows(frequency, frequency * 13.716 / 20.0)
        self.assertEqual(len(rows), 31)
        self.assertEqual(rows[0]["retained_mode_count"], 30)
        self.assertIsNone(rows[0]["deleted_mode_this_stage"])
        self.assertEqual(rows[1]["deleted_mode_this_stage"], 30)
        self.assertEqual(rows[1]["retained_mode_indices_one_based"][-1], 29)
        self.assertEqual(rows[30]["deleted_mode_this_stage"], 1)
        self.assertEqual(rows[30]["retained_mode_indices_one_based"], [])

    def test_invalid_full_space_energy_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._config(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["rvmd"]["full_space_energy_target"] = 1.01
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(StudyConfigurationError, "energy_target"):
                load_study_context(path, project_root=root)


if __name__ == "__main__":
    unittest.main()
