# Copyright (c) 2026 Olegequ.
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np



from airwake_rotor.dmd_3d_flow import DMD3DFlow, SpatialOutOfBoundsError  # noqa: E402


class TestDMD3DFlow(unittest.TestCase):
    def _write_artifact(self, directory: Path):
        x = np.asarray([0.0, 1.0])
        y = np.asarray([-1.0, 1.0])
        z = np.asarray([5.0, 6.0])
        coordinates = np.asarray(
            [(xx, yy, zz) for zz in z for yy in y for xx in x], dtype=np.float32
        )
        n_points = coordinates.shape[0]
        n_time = 4
        starts = np.asarray([0, 2, 3], dtype=np.int32)
        stops = np.asarray([2, 3, 5], dtype=np.int32)
        coefficients = np.asarray(
            [
                [1.0, 0.0, 2.0, -1.0, 0.5],
                [0.0, 1.0, 1.0, 0.5, -1.0],
                [-1.0, 0.0, 0.0, 1.0, 0.25],
                [0.0, -1.0, -1.0, -0.5, 1.0],
            ],
            dtype=np.float32,
        )
        mean = np.column_stack(
            (1.0 + coordinates[:, 0], 2.0 + coordinates[:, 1], coordinates[:, 2])
        ).astype(np.float32)
        modes = np.empty((n_points, 3, 5), dtype=np.float32)
        for component in range(5):
            modes[:, :, component] = (
                (component + 1) * 0.01 * coordinates
                + np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
            )
        np.save(str(directory / "coordinates.npy"), coordinates)
        np.save(str(directory / "mean_field_physical.npy"), mean)
        np.save(str(directory / "spatial_modes_physical.npy"), modes)
        frequency = np.asarray([0.1, 0.4, 0.9], dtype=np.float64)
        np.savez(
            str(directory / "temporal_model.npz"),
            temporal_coefficients=coefficients,
            group_component_start=starts,
            group_component_stop=stops,
            frequency_hz=frequency,
            strouhal=frequency * 13.716 / 20.0,
        )
        metadata = {
            "variables": ["u", "v", "w"],
            "dt_seconds": 0.5,
            "n_points": n_points,
            "n_frames": n_time,
            "selected_physical_groups": 3,
            "selected_real_quadrature_components": 5,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        audit = {
            "passed": True,
            "target": 0.998,
            "full_normalized_captured_energy_fraction_one_minus_residual": 0.999,
        }
        (directory / "full_space_regression_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        return coordinates, mean, modes, coefficients

    def test_grouped_reconstruction_and_explicit_subset(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            coordinates, mean, modes, coefficients = self._write_artifact(directory)
            flow = DMD3DFlow(directory)
            points = coordinates[[0, 7]].astype(float)
            expected = mean[[0, 7]] + np.tensordot(
                modes[[0, 7]], coefficients[1], axes=([-1], [0])
            )
            actual = flow.velocity_at(0.5, points, retained_mode_count=3)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-7)

            sample = flow.velocity_at(
                0.5, points, retained_mode_count=2, mode_indices=[0, 2],
                return_mode_contributions=True,
            )
            np.testing.assert_allclose(
                sample.velocity,
                sample.mean_velocity + sample.mode_contributions.sum(axis=-2),
                rtol=0.0,
                atol=1e-12,
            )
            self.assertEqual(sample.mode_contributions.shape, (2, 2, 3))
            self.assertEqual(sample.quadrature_indices.tolist(), [0, 1, 3, 4])
            with self.assertRaises(ValueError):
                flow.velocity_at(
                    0.5, points, retained_mode_count=1, mode_indices=[0, 2]
                )
            with self.assertRaises(SpatialOutOfBoundsError):
                flow.velocity_at(0.5, np.asarray([[2.0, 0.0, 5.5]]))
            flow.close()


if __name__ == "__main__":
    unittest.main()
