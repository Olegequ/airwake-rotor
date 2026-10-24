# Copyright (c) 2026 Olegequ.
"""Synthetic end-to-end verification without the external CFD archive."""
import csv
import json

import numpy as np

from airwake_rotor.demo import run_demo


def test_frequency_recovery_and_shared_trim_demo(tmp_path):
    report = run_demo(tmp_path)
    np.testing.assert_allclose(report["recovered_frequency_hz"], report["known_frequency_hz"], atol=0.02, rtol=0)
    assert report["relative_reconstruction_error"] < 0.04
    assert report["rvmd_converged"]
    assert report["hover_trim_relative_thrust_residual"] < 1e-6
    assert json.loads((tmp_path / "summary.json").read_text())["synthetic_data"]
    rows = {}
    for case in report["coupled_cases"]:
        with (tmp_path / f"{case}_timeseries.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows[case] = list(csv.DictReader(stream))
        assert len(rows[case]) == 51
    # The zero-mean case must remain at the common initial hover equilibrium.
    assert max(abs(float(row["heave_m"])) for row in rows["mean"]) < 1e-6
    assert max(abs(float(row["heave_m"])) for row in rows["full"]) > 1e-5
    assert (tmp_path / "demo.png").stat().st_size > 1000
