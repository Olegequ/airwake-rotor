# Copyright (c) 2026 Olegequ.
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np



from airwake_rotor.rvmd_core import (  # noqa: E402
    RVMDResult,
    coefficient_spectrum_moments,
    mirror_extend,
    rvmd,
)


class RVMDCoreTests(unittest.TestCase):
    @staticmethod
    def _restart_problem():
        rng = np.random.RandomState(31)
        field = rng.standard_normal((12, 96))
        weights = np.linspace(1.0, 2.0, field.shape[0])
        settings = {
            "k": 4,
            "alpha": 300.0,
            "dt": 0.02,
            "spatial_weights": weights,
            "n_dc": 1,
            # Deliberately unsorted so a test cannot accidentally continue
            # from the public frequency-sorted output state.
            "initial_frequencies_hz": [0.0, 9.0, 2.0, 5.0],
            "tolerance": 0.0,
            "dtype": "float64",
        }
        return field, weights, settings

    def test_mirror_extension_matches_reference_indexing(self):
        even = np.arange(12, dtype=float).reshape(2, 6)
        extended_even, half_even = mirror_extend(even)
        self.assertEqual(half_even, 3)
        self.assertEqual(extended_even.shape, (2, 12))
        np.testing.assert_array_equal(extended_even[:, half_even : half_even + 6], even)
        np.testing.assert_array_equal(extended_even[:, :half_even], even[:, :half_even][:, ::-1])

        odd = np.arange(10, dtype=float).reshape(2, 5)
        extended_odd, half_odd = mirror_extend(odd)
        self.assertEqual(half_odd, 3)
        self.assertEqual(extended_odd.shape, (2, 10))
        np.testing.assert_array_equal(extended_odd[:, half_odd : half_odd + 5], odd)

    def test_two_frequency_field_recovers_centers_and_energy_identity(self):
        n_time = 256
        dt = 0.02
        time = np.arange(n_time) * dt
        frequencies = np.asarray([1.5625, 3.125])
        rng = np.random.RandomState(4)
        first = rng.standard_normal(14)
        first /= np.linalg.norm(first)
        second = rng.standard_normal(14)
        second -= first * np.dot(first, second)
        second /= np.linalg.norm(second)
        field = np.outer(first, np.cos(2 * np.pi * frequencies[0] * time))
        field += 0.65 * np.outer(second, np.cos(2 * np.pi * frequencies[1] * time))

        result = rvmd(
            field,
            2,
            2000.0,
            dt=dt,
            initial_frequencies_hz=frequencies,
            tolerance=1e-5,
            maximum_steps=120,
            dtype="float64",
        )
        np.testing.assert_allclose(result.frequency_hz, frequencies, atol=0.02, rtol=0)
        self.assertLess(result.audit["relative_frobenius_error"], 0.04)
        self.assertLess(result.audit["energy_identity_closure_error_relative"], 1e-12)
        np.testing.assert_allclose(result.strouhal, result.frequency_hz * 13.716 / 20.0)
        self.assertTrue(np.all(np.diff(result.frequency_hz) >= 0))

    def test_saved_centers_are_power_centroids_of_mirror_extended_coefficients(self):
        rng = np.random.RandomState(19)
        field = rng.standard_normal((10, 80))
        result = rvmd(
            field,
            4,
            250.0,
            dt=0.0125,
            maximum_steps=5,
            tolerance=0.0,
            dtype="float64",
        )
        center, bandwidth = coefficient_spectrum_moments(
            result.coefficient_spectrum, field.shape[1]
        )
        np.testing.assert_allclose(center, result.omega_cycles_per_sample, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.frequency_hz, center / 0.0125, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(bandwidth)))
        self.assertTrue(np.all(bandwidth >= 0.0))

    def test_real_rvmd_represents_a_traveling_wave_with_two_modes(self):
        n_space = 32
        n_time = 400
        dt = 0.01
        frequency = 2.0
        x = np.linspace(0, 2 * np.pi, n_space, endpoint=False)
        time = np.arange(n_time) * dt
        field = np.cos(x[:, None] - 2 * np.pi * frequency * time[None, :])
        result = rvmd(
            field,
            2,
            100.0,
            dt=dt,
            initial_frequencies_hz=[1.9, 2.1],
            tolerance=1e-3,
            maximum_steps=220,
            dtype="float64",
        )
        self.assertTrue(np.all(np.abs(result.frequency_hz - frequency) < 0.12))
        self.assertLess(result.audit["relative_frobenius_error"], 0.08)

    def test_k30_shapes_sorting_and_save_roundtrip(self):
        rng = np.random.RandomState(12)
        field = rng.standard_normal((40, 64)).astype(np.float32)
        result = rvmd(
            field,
            30,
            200.0,
            dt=0.01,
            maximum_steps=2,
            tolerance=0.0,
            dtype="float32",
        )
        self.assertEqual(result.phi.shape, (40, 30))
        self.assertEqual(result.c.shape, (64, 30))
        self.assertEqual(result.coefficient_spectrum.shape, (65, 30))
        self.assertTrue(np.all(np.diff(result.frequency_hz) >= 0))
        with tempfile.TemporaryDirectory() as directory:
            path = result.save(Path(directory) / "model.npz")
            loaded = RVMDResult.load(path)
            np.testing.assert_allclose(loaded.phi, result.phi)
            np.testing.assert_allclose(loaded.c, result.c)
            np.testing.assert_allclose(loaded.frequency_hz, result.frequency_hz)
            self.assertEqual(loaded.info["k"], 30)

    def test_restart_40_plus_60_matches_one_100_sweep_trajectory_exactly(self):
        field, _, settings = self._restart_problem()
        first = rvmd(field, maximum_steps=40, **settings)
        self.assertEqual(first.info["completed_steps"], 40)
        self.assertFalse(first.info["converged"])
        self.assertIsNotNone(first.restart_state)
        with tempfile.TemporaryDirectory() as directory:
            saved = first.save(Path(directory) / "checkpoint.npz")
            checkpoint = RVMDResult.load(saved)
            continued = rvmd(
                field,
                maximum_steps=100,
                restart=checkpoint,
                **settings,
            )
        uninterrupted = rvmd(field, maximum_steps=100, **settings)

        self.assertEqual(continued.info["restart_from_completed_steps"], 40)
        self.assertEqual(continued.info["maximum_steps"], 100)
        self.assertEqual(continued.info["completed_steps"], 100)
        self.assertEqual(
            continued.info["maximum_steps_semantics"],
            "total sweeps including saved restart sweeps",
        )
        np.testing.assert_array_equal(
            continued.omega_history[:, :41], first.omega_history
        )
        np.testing.assert_array_equal(
            continued.difference_history[:40], first.difference_history
        )
        for observed, expected in (
            (continued.omega_history, uninterrupted.omega_history),
            (continued.difference_history, uninterrupted.difference_history),
            (continued.phi, uninterrupted.phi),
            (continued.c, uninterrupted.c),
            (continued.coefficient_spectrum, uninterrupted.coefficient_spectrum),
            (
                continued.restart_state.phi_unsorted,
                uninterrupted.restart_state.phi_unsorted,
            ),
            (
                continued.restart_state.residual_spectrum,
                uninterrupted.restart_state.residual_spectrum,
            ),
        ):
            np.testing.assert_array_equal(observed, expected)

    def test_restart_rejects_every_immutable_problem_change_and_corruption(self):
        field, weights, settings = self._restart_problem()
        first = rvmd(field, maximum_steps=40, **settings)

        def resume(**overrides):
            request = dict(settings)
            request.update(overrides)
            return rvmd(field, maximum_steps=100, restart=first, **request)

        cases = (
            ("k", {"k": 5}),
            ("alpha", {"alpha": 301.0}),
            ("dt", {"dt": 0.021}),
            ("dtype", {"dtype": "float32"}),
            ("n_dc", {"n_dc": 0}),
            (
                "weights",
                {"spatial_weights": weights + np.linspace(0.0, 0.2, weights.size)},
            ),
            ("maximum_steps", {"maximum_steps": 39}),
        )
        for label, overrides in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                if "maximum_steps" in overrides:
                    request = dict(settings)
                    request.update(overrides)
                    rvmd(field, restart=first, **request)
                else:
                    resume(**overrides)
        changed_field = field.copy()
        changed_field[0, 0] += 1.0e-12
        with self.assertRaisesRegex(ValueError, "input data"):
            rvmd(changed_field, maximum_steps=100, restart=first, **settings)

        first.restart_state.residual_spectrum[0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "checksum"):
            resume()

    def test_legacy_npz_without_internal_state_remains_readable(self):
        field, _, settings = self._restart_problem()
        result = rvmd(field, maximum_steps=2, **settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(
                str(path),
                phi=result.phi,
                c=result.c,
                omega_cycles_per_sample=result.omega_cycles_per_sample,
                frequency_hz=result.frequency_hz,
                strouhal=result.strouhal,
                energy=result.energy,
                coefficient_spectrum=result.coefficient_spectrum,
                omega_history=result.omega_history,
                difference_history=result.difference_history,
                info_json=np.asarray(json.dumps(result.info)),
                audit_json=np.asarray(json.dumps(result.audit)),
            )
            loaded = RVMDResult.load(path)
            self.assertIsNone(loaded.restart_state)
            np.testing.assert_array_equal(loaded.phi, result.phi)
            with self.assertRaisesRegex(ValueError, "legacy NPZ"):
                rvmd(field, maximum_steps=4, restart=loaded, **settings)

    def test_unknown_compute_backend_is_rejected(self):
        field = np.ones((3, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "compute_backend"):
            rvmd(
                field,
                2,
                100.0,
                dt=0.0125,
                maximum_steps=1,
                compute_backend="unknown",
            )

    def test_torch_cuda_can_continue_numpy_checkpoint_with_close_trajectory(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch dependency is unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable to PyTorch")

        time = np.arange(96, dtype=np.float32) * np.float32(0.0125)
        field = np.vstack(
            [
                np.sin(2.0 * np.pi * 0.8 * time + index * 0.2)
                + 0.35 * np.cos(2.0 * np.pi * 2.1 * time - index * 0.1)
                for index in range(8)
            ]
        ).astype(np.float32)
        settings = {
            "k": 4,
            "alpha": 1000.0,
            "dt": 0.0125,
            "tolerance": 0.0,
            "init_frequency_max_hz": 3.0,
            "dtype": "float32",
        }
        checkpoint = rvmd(field, maximum_steps=4, **settings)
        cpu = rvmd(field, maximum_steps=8, restart=checkpoint, **settings)
        gpu = rvmd(
            field,
            maximum_steps=8,
            restart=checkpoint,
            compute_backend="torch-cuda",
            **settings,
        )

        np.testing.assert_array_equal(gpu.difference_history[:4], checkpoint.difference_history)
        np.testing.assert_allclose(
            gpu.difference_history[4:], cpu.difference_history[4:], rtol=2e-3, atol=2e-5
        )
        np.testing.assert_allclose(gpu.frequency_hz, cpu.frequency_hz, rtol=5e-4, atol=5e-4)
        relative_reconstruction_difference = np.linalg.norm(
            gpu.reconstruct() - cpu.reconstruct()
        ) / np.linalg.norm(cpu.reconstruct())
        self.assertLess(relative_reconstruction_difference, 1e-4)
        self.assertEqual(gpu.info["compute_backend_effective"], "torch-cuda")
        self.assertEqual(gpu.info["compute_device"], torch.cuda.get_device_name(0))
        self.assertFalse(gpu.info["allow_tf32"])


if __name__ == "__main__":
    unittest.main()
