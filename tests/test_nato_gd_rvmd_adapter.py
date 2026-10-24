# Copyright (c) 2026 Olegequ.
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np



from airwake_rotor.nato_gd_rvmd_adapter import (  # noqa: E402
    COLUMN_NAMES,
    NatoGDCacheSeries,
    NatoGDCsvSeries,
    apply_countsketch_frame,
    choose_variable_scales,
    countsketch_map,
    exact_pod_projection_by_energy,
    identify_temporal_coefficients,
    infer_cartesian_grid,
    randomized_pod_projection,
    regress_full_space_modes,
    resolve_compute_backend,
    solve_spatial_modes_from_statistics,
)


class AdapterUnitTests(unittest.TestCase):
    def test_compute_backend_policy_preserves_explicit_requests(self):
        self.assertEqual(
            resolve_compute_backend("numpy"),
            ("numpy", "explicit backend request"),
        )
        self.assertEqual(
            resolve_compute_backend("torch-cuda"),
            ("torch-cuda", "explicit backend request"),
        )
        with self.assertRaisesRegex(ValueError, "compute_backend"):
            resolve_compute_backend("unknown")

    def test_auto_compute_backend_resolves_to_supported_backend(self):
        effective, reason = resolve_compute_backend("auto")
        self.assertIn(effective, ("numpy", "torch-cuda"))
        self.assertTrue(reason.startswith("auto "))

    def test_validated_cache_series_restores_original_csv_row_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            coordinates = np.asarray(
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            )
            raw_to_grid = np.asarray([2, 0, 1], dtype=np.int64)
            np.save(str(cache / "raw_coordinates.npy"), coordinates)
            np.save(str(cache / "raw_to_grid.npy"), raw_to_grid)
            for offset, name in enumerate(("u", "v", "w")):
                values = np.asarray(
                    [[[[10.0 + offset, 20.0 + offset, 30.0 + offset]]],
                     [[[11.0 + offset, 21.0 + offset, 31.0 + offset]]]],
                    dtype=np.float32,
                )
                np.save(str(cache / (name + ".npy")), values)
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "format": "nato-gd-3d-cache-v1",
                        "complete": True,
                        "source_dir": str(cache / "source"),
                        "dt_s": 0.0125,
                        "frame_count": 2,
                        "row_count_per_frame": 3,
                        "array_shape_tzyx": [2, 1, 1, 3],
                    }
                ),
                encoding="utf-8",
            )
            (cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {"frame_id": 2001, "name": "frame_2001.csv"},
                            {"frame_id": 2002, "name": "frame_2002.csv"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            series = NatoGDCacheSeries(cache, expected_dt=0.0125)
            values, observed_coordinates = series.read_frame(0, include_coordinates=True)
            np.testing.assert_array_equal(observed_coordinates, coordinates)
            np.testing.assert_allclose(values[:, 0], [30.0, 10.0, 20.0])
            np.testing.assert_allclose(values[:, 1], [31.0, 11.0, 21.0])
            series.close()
            with self.assertRaisesRegex(ValueError, "cache dt"):
                NatoGDCacheSeries(cache, expected_dt=0.02)

    def test_exact_pod_projection_selects_smallest_energy_rank(self):
        singular_values = np.asarray([4.0, 2.0, 1.0, 0.5])
        matrix = np.diag(singular_values)
        target = 20.0 / 21.25
        basis, scores, metadata = exact_pod_projection_by_energy(
            matrix, retained_energy_target=target, minimum_rank=1
        )

        self.assertEqual(metadata["energy_target_rank"], 2)
        self.assertEqual(metadata["rank"], 2)
        self.assertGreaterEqual(metadata["retained_energy_fraction"], target)
        self.assertLess(16.0 / 21.25, target)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(basis @ scores, np.diag([4.0, 2.0, 0.0, 0.0]))

        _, _, forced = exact_pod_projection_by_energy(
            matrix, retained_energy_target=0.5, minimum_rank=3
        )
        self.assertEqual(forced["energy_target_rank"], 1)
        self.assertEqual(forced["rank"], 3)
        self.assertGreaterEqual(forced["retained_energy_fraction"], 0.5)

        for invalid_target in (0.0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                exact_pod_projection_by_energy(matrix, invalid_target)

    def test_countsketch_is_deterministic_and_uses_every_input(self):
        values = np.arange(30, dtype=float).reshape(10, 3)
        buckets_a, signs_a, metadata_a = countsketch_map(10, 3, 7, 1234)
        buckets_b, signs_b, metadata_b = countsketch_map(10, 3, 7, 1234)
        np.testing.assert_array_equal(buckets_a, buckets_b)
        np.testing.assert_array_equal(signs_a, signs_b)
        np.testing.assert_allclose(
            apply_countsketch_frame(values, buckets_a, signs_a, 7),
            apply_countsketch_frame(values, buckets_b, signs_b, 7),
        )
        self.assertTrue(metadata_a["all_input_degrees_of_freedom_participate"])
        self.assertEqual(metadata_a, metadata_b)

    def test_default_velocity_scaling_is_one_common_scalar(self):
        component_rms = np.asarray([1.0, 2.0, 5.0])
        component_energy = component_rms ** 2 * 100.0
        scales, metadata = choose_variable_scales(
            component_rms, component_energy, ("u", "v", "w"), 10, 10, "velocity_global_rms"
        )
        self.assertTrue(np.all(scales == scales[0]))
        self.assertTrue(metadata["preserves_relative_velocity_kinetic_energy_weighting"])
        with self.assertRaises(ValueError):
            choose_variable_scales(
                [1.0, 2.0], [100.0, 400.0], ("u", "p"), 10, 10, "velocity_global_rms"
            )

    def test_randomized_pod_and_block_regression(self):
        rng = np.random.RandomState(8)
        spatial = rng.standard_normal((60, 4))
        coefficient = rng.standard_normal((80, 4))
        field = spatial @ coefficient.T
        basis, scores, metadata = randomized_pod_projection(
            field.astype(np.float64), rank=4, seed=2, oversampling=4, power_iterations=1
        )
        np.testing.assert_allclose(basis @ scores, field, atol=1e-9, rtol=1e-9)
        self.assertGreater(metadata["retained_energy_fraction"], 1 - 1e-12)

        centered_cross = field @ coefficient
        recovered, audit = solve_spatial_modes_from_statistics(
            centered_cross, coefficient.T @ coefficient, ridge_relative=0.0
        )
        np.testing.assert_allclose(recovered, spatial, atol=1e-10, rtol=1e-10)
        self.assertEqual(audit["coefficient_gram_rank"], 4)


class AdapterCsvEndToEndTest(unittest.TestCase):
    @staticmethod
    def _write_series(directory: Path, n_time: int, dt: float) -> np.ndarray:
        coordinates = np.asarray(
            [(x, y, z) for x in range(4) for y in range(3) for z in range(2)], dtype=float
        )
        header = [
            COLUMN_NAMES["u"],
            COLUMN_NAMES["v"],
            COLUMN_NAMES["w"],
            COLUMN_NAMES["p"],
            COLUMN_NAMES["x"],
            COLUMN_NAMES["y"],
            COLUMN_NAMES["z"],
        ]
        first_frequency = 0.78125
        second_frequency = 1.5625
        spatial_a = 0.2 + coordinates[:, 0] / 4.0 + coordinates[:, 2] / 5.0
        spatial_b = -0.3 + coordinates[:, 1] / 3.0
        for time_index in range(n_time):
            time_value = time_index * dt
            first = np.cos(2 * np.pi * first_frequency * time_value)
            second = np.cos(2 * np.pi * second_frequency * time_value)
            u = 12.0 + spatial_a * first
            v = -1.0 + 0.7 * spatial_b * second
            w = 0.3 + 0.25 * spatial_a * first + 0.2 * spatial_b * second
            p = np.full(coordinates.shape[0], 101325.0)
            table = np.column_stack((u, v, w, p, coordinates))
            with (directory / "{:04d}.csv".format(2001 + time_index)).open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(table)
        return coordinates

    def test_small_csv_series_writes_declared_shapes_and_full_regression(self):
        n_time = 128
        dt = 0.04
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory = root / "input"
            output_directory = root / "output"
            input_directory.mkdir()
            coordinates = self._write_series(input_directory, n_time, dt)
            series = NatoGDCsvSeries(
                input_directory, variables=("u", "v", "w"), reader_engine="numpy"
            )
            grid = infer_cartesian_grid(series.coordinates)
            self.assertTrue(grid["is_complete_cartesian_product"])
            self.assertTrue(grid["is_uniform_each_axis"])
            self.assertEqual(grid["shape_nx_ny_nz"], [4, 3, 2])

            result, metadata = identify_temporal_coefficients(
                series,
                output_directory,
                dt=dt,
                k=2,
                alpha=1000.0,
                method="countsketch",
                countsketch_rows_per_variable=24,
                sketch_seed=9,
                pod_rank=8,
                pod_seed=10,
                tolerance=1e-4,
                maximum_steps=120,
                initial_frequencies_hz=[0.78125, 1.5625],
                coordinate_check_interval=1,
                progress_interval=0,
                dtype="float64",
            )
            self.assertEqual(result.c.shape, (n_time, 2))
            self.assertTrue(metadata["identification_compression"]["all_input_degrees_of_freedom_participate"])
            self.assertEqual(
                metadata["time_axis"]["absolute_source_time_status"],
                "unknown_not_inferred_from_frame_id_or_filename",
            )
            self.assertFalse(metadata["time_axis"]["frame_id_times_dt_used_as_absolute_time"])
            np.testing.assert_allclose(result.frequency_hz, [0.78125, 1.5625], atol=0.08)

            scales = np.load(str(output_directory / "component_scales.npy"))
            audit = regress_full_space_modes(
                series,
                result,
                scales,
                output_directory,
                time_batch_size=9,
                spatial_block_rows=17,
                ridge_relative=1e-12,
                coordinate_check_interval=1,
                progress_interval=0,
            )
            saved_coordinates = np.load(str(output_directory / "coordinates.npy"))
            mean_field = np.load(str(output_directory / "mean_field_physical.npy"))
            modes = np.load(str(output_directory / "spatial_modes_physical.npy"))
            coefficient_mean = np.load(
                str(output_directory / "coefficient_mean_for_full_regression.npy")
            )
            np.testing.assert_allclose(saved_coordinates, coordinates)
            self.assertEqual(mean_field.shape, (24, 3))
            self.assertEqual(modes.shape, (24, 3, 2))
            np.testing.assert_allclose(
                np.mean(result.c - coefficient_mean[None, :], axis=0), 0.0, atol=1e-14
            )
            self.assertLess(audit["full_normalized_relative_frobenius_error"], 0.15)
            for filename in (
                "temporal_model.npz",
                "metadata.json",
                "audit.json",
                "frequencies.csv",
                "full_space_regression_audit.json",
            ):
                self.assertTrue((output_directory / filename).is_file(), filename)

    def test_restart_directory_reuses_projection_and_matches_direct_total_steps(self):
        n_time = 64
        dt = 0.04
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory = root / "input"
            first_directory = root / "first_2_steps"
            resumed_directory = root / "resumed_to_5_steps"
            direct_directory = root / "direct_5_steps"
            input_directory.mkdir()
            self._write_series(input_directory, n_time, dt)
            series = NatoGDCsvSeries(
                input_directory, variables=("u", "v", "w"), reader_engine="numpy"
            )
            common = dict(
                dt=dt,
                k=2,
                alpha=1000.0,
                method="countsketch",
                countsketch_rows_per_variable=16,
                sketch_seed=19,
                pod_rank=8,
                pod_seed=23,
                pod_oversampling=4,
                pod_power_iterations=1,
                tolerance=0.0,
                initial_frequencies_hz=[0.78125, 1.5625],
                coordinate_check_interval=1,
                progress_interval=0,
                dtype="float64",
            )
            first, first_metadata = identify_temporal_coefficients(
                series, first_directory, maximum_steps=2, **common
            )
            resumed, resumed_metadata = identify_temporal_coefficients(
                series,
                resumed_directory,
                maximum_steps=5,
                restart_directory=first_directory,
                **common,
            )
            direct, _ = identify_temporal_coefficients(
                series, direct_directory, maximum_steps=5, **common
            )

            self.assertEqual(first.info["completed_steps"], 2)
            self.assertEqual(resumed.info["completed_steps"], 5)
            self.assertEqual(resumed.info["restart_from_completed_steps"], 2)
            np.testing.assert_array_equal(resumed.omega_history, direct.omega_history)
            np.testing.assert_array_equal(
                resumed.difference_history, direct.difference_history
            )
            np.testing.assert_array_equal(resumed.phi, direct.phi)
            np.testing.assert_array_equal(resumed.c, direct.c)
            np.testing.assert_array_equal(
                np.load(str(first_directory / "identification_projection_basis.npy")),
                np.load(str(resumed_directory / "identification_projection_basis.npy")),
            )
            np.testing.assert_array_equal(
                np.load(str(first_directory / "component_scales.npy")),
                np.load(str(resumed_directory / "component_scales.npy")),
            )
            np.testing.assert_array_equal(
                np.load(str(first_directory / "rvmd_input.npy")),
                np.load(str(resumed_directory / "rvmd_input.npy")),
            )
            self.assertEqual(first_metadata["restart_artifacts"]["schema_version"], 1)
            self.assertEqual(
                resumed_metadata["restart_provenance"]["source_completed_steps"], 2
            )
            self.assertEqual(
                resumed_metadata["restart_provenance"]["requested_total_maximum_steps"], 5
            )
            self.assertTrue(resumed_metadata["restart_provenance"]["projection_basis_reused"])

            with self.assertRaisesRegex(ValueError, "rows per variable"):
                identify_temporal_coefficients(
                    series,
                    root / "bad_restart",
                    maximum_steps=5,
                    restart_directory=first_directory,
                    countsketch_rows_per_variable=17,
                    **{
                        key: value
                        for key, value in common.items()
                        if key != "countsketch_rows_per_variable"
                    },
                )


if __name__ == "__main__":
    unittest.main()
