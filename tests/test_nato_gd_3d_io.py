# Copyright (c) 2026 Olegequ.
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np



from airwake_rotor import nato_gd_3d_io as io3d
class SyntheticNatoDataset(object):
    def __init__(self, root, frame_count=4, dt_s=0.1):
        self.root = Path(root)
        self.source = self.root / "source"
        self.cache = self.root / "cache"
        self.source.mkdir(parents=True)
        self.frame_count = int(frame_count)
        self.dt_s = float(dt_s)
        self.first_id = 2001
        self.last_id = self.first_id + self.frame_count - 1
        self.x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        self.y = np.array([-1.0, 1.0], dtype=np.float64)
        self.z = np.array([10.0, 11.0], dtype=np.float64)
        zz, yy, xx = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        self.canonical_coordinates = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        # Fixed, deliberately non-monotonic raw order.
        self.permutation = np.array([7, 0, 11, 3, 5, 8, 2, 10, 1, 6, 9, 4], dtype=np.int64)
        self.raw_coordinates = self.canonical_coordinates[self.permutation]
        for frame_index in range(self.frame_count):
            self.write_frame(frame_index)

    @staticmethod
    def velocity(points, relative_time_s):
        points = np.asarray(points, dtype=np.float64)
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        t = float(relative_time_s)
        return np.stack(
            [
                1.0 + 2.0 * x - 3.0 * y + 4.0 * z + 5.0 * t,
                -2.0 + x + 0.5 * y - 2.0 * z - 3.0 * t,
                4.0 - 0.25 * x + 2.0 * y + 3.0 * z + 1.5 * t,
            ],
            axis=-1,
        )

    def frame_path(self, frame_index):
        frame_id = self.first_id + int(frame_index)
        return self.source / "{}{}.csv".format(io3d.DEFAULT_PREFIX, frame_id)

    def write_frame(self, frame_index, coordinates=None, nonfinite_pressure_row=None, header=None):
        coordinates = self.raw_coordinates if coordinates is None else np.asarray(coordinates)
        velocities = self.velocity(coordinates, frame_index * self.dt_s)
        path = self.frame_path(frame_index)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(io3d.EXPECTED_HEADER if header is None else header)
            for row_index, (velocity, point) in enumerate(zip(velocities, coordinates)):
                pressure = 101325.0 + point[0] - point[1] + 0.1 * point[2]
                if row_index == nonfinite_pressure_row:
                    pressure = float("nan")
                writer.writerow(
                    [
                        float(velocity[0]),
                        float(velocity[1]),
                        float(velocity[2]),
                        float(pressure),
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    ]
                )

    def build(self):
        return io3d.build_cache(
            self.source,
            self.cache,
            self.dt_s,
            expected_start_id=self.first_id,
            expected_end_id=self.last_id,
            flush_every=1,
        )


class NatoGD3DIOTests(unittest.TestCase):
    def setUp(self):
        self.temp_context = tempfile.TemporaryDirectory(prefix="nato_gd_3d_test_")
        self.case = SyntheticNatoDataset(Path(self.temp_context.name))

    def tearDown(self):
        self.temp_context.cleanup()

    def test_build_mapping_and_exact_linear_interpolation(self):
        metadata = self.case.build()
        self.assertEqual(metadata["grid_shape_zyx"], [2, 2, 3])
        self.assertEqual(metadata["array_shape_tzyx"], [4, 2, 2, 3])
        self.assertEqual(metadata["array_dtype"], "float32")

        mapping = np.load(str(self.case.cache / "raw_to_grid.npy"))
        np.testing.assert_array_equal(mapping, self.case.permutation)
        raw_coordinates = np.load(str(self.case.cache / "raw_coordinates.npy"))
        np.testing.assert_array_equal(raw_coordinates, self.case.raw_coordinates)

        cache = io3d.NatoGD3DCache(self.case.cache, expected_dt_s=self.case.dt_s)
        points = np.array(
            [
                [[0.25, -0.5, 10.25], [1.5, 0.25, 10.75]],
                [[0.0, -1.0, 10.0], [2.0, 1.0, 11.0]],
            ],
            dtype=np.float64,
        )
        actual = cache.interpolate_frame(2, points)
        expected = self.case.velocity(points, 2 * self.case.dt_s)
        self.assertEqual(actual.shape, (2, 2, 3))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-6)

        single = cache.interpolate_frame(1, points[0, 0])
        self.assertEqual(single.shape, (3,))
        np.testing.assert_allclose(
            single,
            self.case.velocity(points[0, 0], self.case.dt_s),
            rtol=0.0,
            atol=5.0e-6,
        )

    def test_temporal_interpolation_is_exact_for_linear_field(self):
        self.case.build()
        cache = io3d.NatoGD3DCache(self.case.cache)
        point = np.array([0.6, 0.4, 10.4])
        query_time = 0.15
        actual = cache.interpolate_time(query_time, point)
        expected = self.case.velocity(point, query_time)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-6)

        last = cache.interpolate_time(cache.relative_time_end_s, point)
        np.testing.assert_allclose(
            last,
            self.case.velocity(point, cache.relative_time_end_s),
            rtol=0.0,
            atol=5.0e-6,
        )

    def test_out_of_bounds_and_nonfinite_requests_hard_fail(self):
        self.case.build()
        cache = io3d.NatoGD3DCache(self.case.cache)
        for point in (
            [-1.0e-12, 0.0, 10.5],
            [2.0 + 1.0e-12, 0.0, 10.5],
            [1.0, -1.0 - 1.0e-12, 10.5],
            [1.0, 0.0, 11.0 + 1.0e-12],
            [float("nan"), 0.0, 10.5],
        ):
            with self.assertRaises(io3d.CoordinateOutOfBoundsError):
                cache.interpolate_frame(0, point)
        with self.assertRaises(io3d.CoordinateOutOfBoundsError):
            cache.interpolate_time(-1.0e-12, [1.0, 0.0, 10.5])
        with self.assertRaises(io3d.CoordinateOutOfBoundsError):
            cache.interpolate_time(cache.relative_time_end_s + 1.0e-12, [1.0, 0.0, 10.5])

    def test_every_frame_must_preserve_raw_coordinate_order(self):
        changed = self.case.raw_coordinates.copy()
        changed[[0, 1]] = changed[[1, 0]]
        self.case.write_frame(2, coordinates=changed)
        with self.assertRaisesRegex(io3d.DatasetValidationError, "coordinate order/value mismatch"):
            self.case.build()
        state = json.loads((self.case.cache / "build_state.json").read_text(encoding="utf-8"))
        self.assertFalse(state["complete"])
        self.assertIn("coordinate order/value mismatch", state["last_error"])

    def test_nonfinite_in_pressure_is_rejected_even_though_pressure_is_not_cached(self):
        self.case.write_frame(3, nonfinite_pressure_row=4)
        with self.assertRaisesRegex(io3d.DatasetValidationError, "non-finite value"):
            io3d.scan_dataset(
                self.case.source,
                self.case.dt_s,
                expected_start_id=self.case.first_id,
                expected_end_id=self.case.last_id,
            )

    def test_header_and_numbering_are_strict(self):
        bad_header = list(io3d.EXPECTED_HEADER)
        bad_header[0] = "u"
        self.case.write_frame(0, header=bad_header)
        with self.assertRaisesRegex(io3d.DatasetValidationError, "header mismatch"):
            io3d.scan_dataset(
                self.case.source,
                self.case.dt_s,
                expected_start_id=self.case.first_id,
                expected_end_id=self.case.last_id,
            )

        # Restore the file, then add an otherwise plausible but out-of-range CSV.
        self.case.write_frame(0)
        shutil.copyfile(
            str(self.case.frame_path(0)),
            str(self.case.source / "{}9999.csv".format(io3d.DEFAULT_PREFIX)),
        )
        with self.assertRaisesRegex(io3d.DatasetValidationError, "not the exact closed interval"):
            io3d.discover_frames(
                self.case.source,
                expected_start_id=self.case.first_id,
                expected_end_id=self.case.last_id,
            )

    def test_scan_smoke_and_completed_build_are_idempotent(self):
        scan = io3d.scan_dataset(
            self.case.source,
            self.case.dt_s,
            expected_start_id=self.case.first_id,
            expected_end_id=self.case.last_id,
        )
        self.assertTrue(scan["validated"])
        self.assertEqual(scan["row_count_per_frame"], 12)
        first_metadata = self.case.build()
        second_metadata = self.case.build()
        self.assertEqual(first_metadata["manifest_sha256"], second_metadata["manifest_sha256"])
        self.assertFalse(
            first_metadata["time_axis"]["frame_id_times_dt_used_as_absolute_time"]
        )
        self.assertIsNone(
            first_metadata["time_axis"]["absolute_source_time_origin_seconds"]
        )
        smoke = io3d.smoke_cache(self.case.cache, self.case.dt_s)
        self.assertTrue(smoke["smoke_passed"])

        cache = io3d.NatoGD3DCache(self.case.cache, expected_dt_s=self.case.dt_s)
        with self.assertRaisesRegex(io3d.DatasetValidationError, "absolute CFD source-time"):
            _ = cache.source_time_start_s

        with self.assertRaisesRegex(io3d.DatasetValidationError, "differs from required"):
            io3d.NatoGD3DCache(self.case.cache, expected_dt_s=0.2)

        migrated = io3d.normalize_cache_time_metadata(self.case.cache)
        self.assertFalse(migrated["time_axis"]["frame_id_times_dt_used_as_absolute_time"])
        self.assertNotIn("source_time_definition", migrated)

    def test_interrupted_build_resumes_only_with_identical_policy_and_manifest(self):
        original_read_frame = io3d._read_frame

        def fail_once_on_third_frame(path, expected_rows=None):
            if Path(path).name == self.case.frame_path(2).name:
                raise OSError("synthetic transient read failure")
            return original_read_frame(path, expected_rows=expected_rows)

        with mock.patch.object(io3d, "_read_frame", side_effect=fail_once_on_third_frame):
            with self.assertRaisesRegex(OSError, "transient read failure"):
                self.case.build()

        state = json.loads((self.case.cache / "build_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["completed_frames"], 2)
        self.assertIn("transient read failure", state["last_error"])

        with self.assertRaisesRegex(io3d.DatasetValidationError, "coordinate_atol mismatch"):
            io3d.build_cache(
                self.case.source,
                self.case.cache,
                self.case.dt_s,
                expected_start_id=self.case.first_id,
                expected_end_id=self.case.last_id,
                coordinate_atol=1.0e-12,
                flush_every=1,
            )

        metadata = self.case.build()
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["frame_count"], self.case.frame_count)
        self.assertTrue(io3d.smoke_cache(self.case.cache, self.case.dt_s)["smoke_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
