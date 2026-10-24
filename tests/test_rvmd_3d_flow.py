# Copyright (c) 2026 Olegequ.
import json
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np



from airwake_rotor.rvmd_3d_flow import (  # noqa: E402
    ArtifactValidationError,
    NatoGD3DBaselineFlow,
    RVMD3DFlow,
    RVMDFlowSample,
    SpatialOutOfBoundsError,
    TimeOutOfBoundsError,
)


class SyntheticRVMDArtifacts:
    x = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.asarray([-1.0, 1.0], dtype=np.float64)
    z = np.asarray([5.0, 6.5], dtype=np.float64)
    dt = 0.5
    times = np.arange(4, dtype=np.float64) * dt
    frequency_hz = np.asarray([0.0, 0.25, 0.75], dtype=np.float64)

    mean_base = np.asarray([1.0, -2.0, 0.5])
    mean_gradient = np.asarray(
        [[0.4, -0.2, 0.1], [-0.3, 0.7, 0.2], [0.05, -0.1, 0.6]]
    )
    mode_bases = np.asarray(
        [[0.2, -0.1, 0.4], [-0.5, 0.3, 0.1], [0.7, -0.2, -0.4]]
    )
    mode_gradients = np.asarray(
        [
            [[0.10, 0.02, -0.03], [0.04, -0.08, 0.01], [0.02, 0.03, 0.05]],
            [[-0.03, 0.06, 0.02], [0.07, 0.01, -0.04], [0.05, -0.02, 0.03]],
            [[0.02, -0.01, 0.08], [-0.04, 0.05, 0.02], [0.01, 0.07, -0.02]],
        ]
    )

    @classmethod
    def coefficient_function(cls, time_s):
        time_s = np.asarray(time_s, dtype=np.float64)
        return np.stack(
            (1.0 + 2.0 * time_s, -2.0 + 0.5 * time_s, 0.25 - time_s), axis=-1
        )

    @classmethod
    def mean_function(cls, points):
        points = np.asarray(points, dtype=np.float64)
        return cls.mean_base + points @ cls.mean_gradient.T

    @classmethod
    def mode_function(cls, points):
        points = np.asarray(points, dtype=np.float64)
        return np.stack(
            [
                cls.mode_bases[index] + points @ cls.mode_gradients[index].T
                for index in range(3)
            ],
            axis=-2,
        )

    @classmethod
    def write(cls, directory: Path) -> Path:
        canonical = np.asarray(
            [(x, y, z) for z in cls.z for y in cls.y for x in cls.x], dtype=np.float64
        )
        permutation = np.asarray([7, 0, 10, 3, 5, 9, 1, 11, 2, 8, 4, 6], dtype=np.int64)
        coordinates = canonical[permutation]
        mean = cls.mean_function(coordinates)
        # Public analytical order is [point, mode, component], whereas the
        # full-space adapter persists [point, component, mode].
        modes = np.moveaxis(cls.mode_function(coordinates), -2, -1)
        coefficients = cls.coefficient_function(cls.times)
        coefficient_mean = np.mean(coefficients, axis=0)
        k = coefficients.shape[1]

        directory.mkdir(parents=True, exist_ok=True)
        np.save(str(directory / "coordinates.npy"), coordinates.astype(np.float32))
        np.save(str(directory / "mean_field_physical.npy"), mean.astype(np.float32))
        np.save(str(directory / "spatial_modes_physical.npy"), modes.astype(np.float32))
        np.save(
            str(directory / "coefficient_mean_for_full_regression.npy"), coefficient_mean
        )
        info = {
            "algorithm": "reduced-order variational mode decomposition",
            "n_time": int(coefficients.shape[0]),
            "k": int(k),
            "dt_seconds": cls.dt,
            "mode_sort": "ascending center frequency",
        }
        np.savez(
            str(directory / "temporal_model.npz"),
            c=coefficients,
            frequency_hz=cls.frequency_hz,
            strouhal=cls.frequency_hz * 13.716 / 20.0,
            omega_cycles_per_sample=cls.frequency_hz * cls.dt,
            info_json=np.asarray(json.dumps(info)),
        )
        metadata = {
            "schema_version": 1,
            "stage": "full_space_regression_complete",
            "variables": ["u", "v", "w"],
            "n_points": int(coordinates.shape[0]),
            "n_frames": int(coefficients.shape[0]),
            "k": int(k),
            "dt_seconds": cls.dt,
            "grid_audit": {
                "shape_nx_ny_nz": [len(cls.x), len(cls.y), len(cls.z)],
                "is_complete_cartesian_product": True,
                "is_uniform_each_axis": True,
            },
            "time_axis": {
                "relative_time_from_first_frame_seconds": {
                    "first": 0.0,
                    "last": float(cls.times[-1]),
                }
            },
            "output_arrays": {
                "coordinates.npy": list(coordinates.shape),
                "mean_field_physical.npy": list(mean.shape),
                "spatial_modes_physical.npy": list(modes.shape),
                "coefficient_mean_for_full_regression.npy": list(coefficient_mean.shape),
                "temporal_model.npz/c": list(coefficients.shape),
            },
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream)
        return directory


class RVMD3DFlowInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = SyntheticRVMDArtifacts.write(Path(self.temporary.name) / "rvmd")
        self.flow = RVMD3DFlow(self.directory)

    def tearDown(self):
        self.flow.close()
        self.temporary.cleanup()

    def test_affine_space_and_linear_time_are_exact_for_batched_points(self):
        points = np.asarray(
            [
                [[0.25, -0.50, 5.20], [1.25, 0.25, 5.75]],
                [[1.80, -0.20, 6.10], [0.60, 0.80, 6.35]],
            ]
        )
        query_time = 0.75
        sample = self.flow.velocity_at(
            query_time, points, return_mode_contributions=True
        )
        self.assertIsInstance(sample, RVMDFlowSample)
        expected_mean = SyntheticRVMDArtifacts.mean_function(points)
        expected_modes = SyntheticRVMDArtifacts.mode_function(points)
        saved_mean = np.mean(
            SyntheticRVMDArtifacts.coefficient_function(SyntheticRVMDArtifacts.times), axis=0
        )
        expected_centered = (
            SyntheticRVMDArtifacts.coefficient_function(query_time) - saved_mean
        )
        expected_contributions = expected_modes * expected_centered.reshape(1, 1, 3, 1)
        expected_velocity = expected_mean + np.sum(expected_contributions, axis=-2)
        np.testing.assert_allclose(sample.mean_velocity, expected_mean, atol=2e-6, rtol=2e-7)
        np.testing.assert_allclose(
            sample.mode_contributions, expected_contributions, atol=2e-6, rtol=2e-7
        )
        np.testing.assert_allclose(sample.velocity, expected_velocity, atol=3e-6, rtol=2e-7)
        np.testing.assert_array_equal(sample.mode_indices, [0, 1, 2])
        np.testing.assert_allclose(sample.centered_coefficients, expected_centered)

        # The arbitrary original row order was converted into an exact inverse
        # map rather than assumed to already be x/y/z or z/y/x sorted.
        np.testing.assert_array_equal(
            self.flow.row_index_zyx.reshape(-1)[self.flow.row_to_grid_flat_zyx],
            np.arange(self.flow.coordinates.shape[0]),
        )
        self.assertTrue(self.flow.validate()["valid"])

    def test_frequency_order_retained_count_and_arbitrary_indices(self):
        point = np.asarray([0.4, 0.2, 5.9])
        query_time = 0.35
        mean = SyntheticRVMDArtifacts.mean_function(point)
        modes = SyntheticRVMDArtifacts.mode_function(point)
        coefficient_mean = np.mean(
            SyntheticRVMDArtifacts.coefficient_function(SyntheticRVMDArtifacts.times), axis=0
        )
        centered = SyntheticRVMDArtifacts.coefficient_function(query_time) - coefficient_mean

        np.testing.assert_allclose(
            self.flow.velocity_at(query_time, point, retained_mode_count=0), mean, atol=2e-6
        )
        np.testing.assert_allclose(
            self.flow.velocity_at(query_time, point, retained_mode_count=2),
            mean + np.sum(modes[:2] * centered[:2, None], axis=0),
            atol=2e-6,
        )
        selected = self.flow.velocity_at(
            query_time, point, mode_indices=[2, 0], return_mode_contributions=True
        )
        np.testing.assert_array_equal(selected.mode_indices, [2, 0])
        np.testing.assert_allclose(
            selected.mode_contributions,
            modes[[2, 0]] * centered[[2, 0], None],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            selected.velocity,
            mean + np.sum(modes[[2, 0]] * centered[[2, 0], None], axis=0),
            atol=2e-6,
        )

        with self.assertRaises(ValueError):
            self.flow.velocity_at(
                query_time, point, retained_mode_count=2, mode_indices=[0, 1]
            )
        with self.assertRaises(ValueError):
            self.flow.velocity_at(query_time, point, retained_mode_count=4)
        with self.assertRaises(ValueError):
            self.flow.velocity_at(query_time, point, mode_indices=[1, 1])
        with self.assertRaises(IndexError):
            self.flow.velocity_at(query_time, point, mode_indices=[3])
        with self.assertRaises(TypeError):
            self.flow.velocity_at(query_time, point, mode_indices=[0.0])

    def test_exact_endpoints_work_but_space_and_time_never_extrapolate(self):
        corners = np.asarray([[0.0, -1.0, 5.0], [2.0, 1.0, 6.5]])
        observed = self.flow.velocity_at(self.flow.relative_time_end_s, corners)
        self.assertTrue(np.isfinite(observed).all())
        with self.assertRaises(SpatialOutOfBoundsError):
            self.flow.velocity_at(0.0, np.asarray([-1.0e-12, 0.0, 5.5]))
        with self.assertRaises(SpatialOutOfBoundsError):
            self.flow.velocity_at(0.0, np.asarray([1.0, 0.0, np.nan]))
        with self.assertRaises(TimeOutOfBoundsError):
            self.flow.velocity_at(-1.0e-15, corners)
        with self.assertRaises(TimeOutOfBoundsError):
            self.flow.velocity_at(self.flow.relative_time_end_s + 1.0e-15, corners)

    def test_frame_sampling_matches_exact_temporal_sample(self):
        point = np.asarray([1.1, -0.1, 5.6])
        for frame_index in range(self.flow.frame_count):
            np.testing.assert_allclose(
                self.flow.interpolate_frame(frame_index, point),
                self.flow.interpolate_time(frame_index * self.flow.dt_s, point),
            )
        with self.assertRaises(IndexError):
            self.flow.interpolate_frame(self.flow.frame_count, point)


class RVMD3DFlowArtifactFailureTests(unittest.TestCase):
    def _fresh_directory(self, temporary):
        return SyntheticRVMDArtifacts.write(Path(temporary) / "rvmd")

    def test_wrong_variable_contract_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._fresh_directory(temporary)
            path = directory / "metadata.json"
            with path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            metadata["variables"] = ["u", "w", "v"]
            with path.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            with self.assertRaisesRegex(ArtifactValidationError, "exactly"):
                RVMD3DFlow(directory)

    def test_duplicate_coordinate_and_broken_row_map_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._fresh_directory(temporary)
            path = directory / "coordinates.npy"
            coordinates = np.load(str(path))
            coordinates[-1] = coordinates[0]
            np.save(str(path), coordinates)
            with self.assertRaises(ArtifactValidationError):
                RVMD3DFlow(directory)

    def test_wrong_coefficient_centering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._fresh_directory(temporary)
            path = directory / "coefficient_mean_for_full_regression.npy"
            coefficient_mean = np.load(str(path))
            coefficient_mean[1] += 0.1
            np.save(str(path), coefficient_mean)
            with self.assertRaisesRegex(ArtifactValidationError, "mean"):
                RVMD3DFlow(directory)

    def test_unsorted_frequency_and_bad_strouhal_fail(self):
        for mutation in ("unsorted", "bad_st"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = self._fresh_directory(temporary)
                path = directory / "temporal_model.npz"
                with np.load(str(path), allow_pickle=False) as source:
                    payload = {name: source[name] for name in source.files}
                if mutation == "unsorted":
                    payload["frequency_hz"] = np.asarray([0.0, 0.8, 0.4])
                    payload["omega_cycles_per_sample"] = payload["frequency_hz"] * 0.5
                    payload["strouhal"] = payload["frequency_hz"] * 13.716 / 20.0
                else:
                    payload["strouhal"] = np.asarray(payload["strouhal"]) + 0.01
                np.savez(str(path), **payload)
                with self.assertRaises(ArtifactValidationError):
                    RVMD3DFlow(directory)

    def test_singleton_axis_degenerates_cleanly_and_remains_hard_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._fresh_directory(temporary)
            coordinates_path = directory / "coordinates.npy"
            coordinates = np.load(str(coordinates_path))
            keep = coordinates[:, 2] == np.float32(5.0)
            np.save(str(coordinates_path), coordinates[keep])
            for filename in ("mean_field_physical.npy", "spatial_modes_physical.npy"):
                path = directory / filename
                np.save(str(path), np.load(str(path))[keep])
            metadata_path = directory / "metadata.json"
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            metadata["n_points"] = int(np.sum(keep))
            metadata["grid_audit"]["shape_nx_ny_nz"] = [3, 2, 1]
            metadata["output_arrays"]["coordinates.npy"] = [int(np.sum(keep)), 3]
            metadata["output_arrays"]["mean_field_physical.npy"] = [int(np.sum(keep)), 3]
            metadata["output_arrays"]["spatial_modes_physical.npy"] = [
                int(np.sum(keep)),
                3,
                3,
            ]
            with metadata_path.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            with RVMD3DFlow(directory) as flow:
                point = np.asarray([0.4, 0.2, 5.0])
                expected_mean = SyntheticRVMDArtifacts.mean_function(point)
                np.testing.assert_allclose(
                    flow.velocity_at(0.3, point, retained_mode_count=0),
                    expected_mean,
                    atol=2e-6,
                )
                with self.assertRaises(SpatialOutOfBoundsError):
                    flow.velocity_at(0.3, np.asarray([0.4, 0.2, 5.0 + 1.0e-12]))


class BaselineWrapperTests(unittest.TestCase):
    class FakeCache:
        dt_s = 0.25
        frame_count = 3
        relative_time_end_s = 0.5

        @staticmethod
        def interpolate_time(relative_time_s, points_xyz):
            return np.asarray(points_xyz, dtype=np.float64) + float(relative_time_s)

        @staticmethod
        def interpolate_frame(frame_index, points_xyz):
            return np.asarray(points_xyz, dtype=np.float64) + 0.25 * int(frame_index)

    def test_raw_cache_uses_same_minimal_velocity_interface(self):
        wrapper = NatoGD3DBaselineFlow(self.FakeCache(), expected_dt_s=0.25)
        points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(wrapper.velocity_at(0.125, points), points + 0.125)
        np.testing.assert_allclose(wrapper.interpolate_time(0.125, points), points + 0.125)
        np.testing.assert_allclose(wrapper.interpolate_frame(1, points), points + 0.25)
        self.assertEqual(wrapper.frame_count, 3)
        self.assertEqual(
            list(inspect.signature(RVMD3DFlow.velocity_at).parameters),
            list(inspect.signature(NatoGD3DBaselineFlow.velocity_at).parameters),
        )
        with self.assertRaisesRegex(ValueError, "no RVMD modes"):
            wrapper.velocity_at(0.125, points, retained_mode_count=0)
        with self.assertRaisesRegex(ValueError, "no per-mode"):
            wrapper.velocity_at(0.125, points, return_mode_contributions=True)
        with self.assertRaises(ArtifactValidationError):
            NatoGD3DBaselineFlow(self.FakeCache(), expected_dt_s=0.5)


if __name__ == "__main__":
    unittest.main()
