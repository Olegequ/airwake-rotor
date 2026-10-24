# Copyright (c) 2026 Olegequ.
import math
import sys
from pathlib import Path

import numpy as np
import pytest



from airwake_rotor.finalize_time_weighted_statistics import (  # noqa: E402
    StatisticsFinalizationError,
    boundary_augmented_window,
    time_weighted_statistics,
)


def test_boundary_interpolation_and_linear_time_mean():
    time = np.asarray([0.0, 0.2, 0.5, 0.9, 1.0])
    values = time.copy()
    window_time, window_values, start_interp, end_interp = boundary_augmented_window(
        time, values, 0.15, 0.85
    )
    assert window_time[0] == pytest.approx(0.15)
    assert window_time[-1] == pytest.approx(0.85)
    assert np.allclose(window_values, window_time)
    assert start_interp and end_interp
    stats, audit = time_weighted_statistics(time, values, 0.15, 0.85)
    assert stats["mean"] == pytest.approx(0.5)
    expected_rms = math.sqrt(np.trapz(window_values ** 2, window_time) / (0.85 - 0.15))
    assert stats["rms"] == pytest.approx(expected_rms)
    assert stats["peak_to_peak"] == pytest.approx(0.7)
    assert audit["integration_interval_count"] == len(window_time) - 1


def test_rejects_non_monotone_time():
    with pytest.raises(StatisticsFinalizationError, match="strictly increasing"):
        boundary_augmented_window([0.0, 0.2, 0.1], [1.0, 2.0, 3.0], 0.0, 0.1)
