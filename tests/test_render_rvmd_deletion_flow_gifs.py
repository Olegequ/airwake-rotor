# Copyright (c) 2026 Olegequ.
import tempfile
import unittest
from pathlib import Path

import imageio
import numpy as np

from airwake_rotor.render_rvmd_deletion_flow_gifs import (
    nearest_index,
    reconstruct_slice,
    selected_frame_indices,
    validate_gif,
)


class RVMDGifTests(unittest.TestCase):
    def test_reconstruction_retains_low_frequency_prefix(self):
        mean = np.ones((2, 3, 3))
        modes = np.zeros((2, 3, 3, 3))
        modes[..., 0] = 1.0
        modes[..., 1] = 2.0
        modes[..., 2] = 4.0
        coefficient = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(reconstruct_slice(mean, modes, coefficient, 0), 1.0)
        np.testing.assert_allclose(reconstruct_slice(mean, modes, coefficient, 2), 22.0)
        np.testing.assert_allclose(reconstruct_slice(mean, modes, coefficient, 3), 422.0)

    def test_exactly_200_unique_common_frames(self):
        indices = selected_frame_indices(2500, 200)
        self.assertEqual(indices.size, 200)
        self.assertEqual(np.unique(indices).size, 200)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 2499)
        with self.assertRaises(ValueError):
            selected_frame_indices(199, 200)

    def test_nearest_slice_must_remain_inside_axis(self):
        axis = np.array([0.0, 0.5, 1.0])
        self.assertEqual(nearest_index(axis, 0.49), 1)
        with self.assertRaises(ValueError):
            nearest_index(axis, -1e-12)

    def test_gif_validator_checks_exact_frame_count_and_dimensions(self):
        frames = [
            np.full((6, 8, 3), value, dtype=np.uint8) for value in (0, 127, 255)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "three.gif"
            imageio.mimsave(str(path), frames, format="GIF", duration=0.1)
            audit = validate_gif(path, 3)
            self.assertEqual(audit["frame_count"], 3)
            self.assertEqual((audit["pixel_width"], audit["pixel_height"]), (8, 6))
            with self.assertRaises(ValueError):
                validate_gif(path, 200)


if __name__ == "__main__":
    unittest.main()
