#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Independently audit RVMD temporal-coefficient center frequencies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .rvmd_core import RVMDResult, coefficient_spectrum_moments


class FrequencyAuditError(ValueError):
    """Raised when a saved temporal model fails its frequency contract."""


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _windowed_fft_centroid_cycles_per_sample(coefficients: np.ndarray) -> np.ndarray:
    centered = np.asarray(coefficients, dtype=np.float64) - np.mean(
        coefficients, axis=0, keepdims=True, dtype=np.float64
    )
    spectrum = np.fft.rfft(centered, axis=0)
    frequency = np.fft.rfftfreq(centered.shape[0], d=1.0)
    power = np.abs(spectrum) ** 2
    denominator = np.sum(power, axis=0, dtype=np.float64)
    output = np.full(centered.shape[1], np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).tiny
    output[valid] = (
        np.sum(power[:, valid] * frequency[:, None], axis=0, dtype=np.float64)
        / denominator[valid]
    )
    return output


def audit_frequency_artifact(
    artifact_directory: Path,
    output_directory: Optional[Path] = None,
    *,
    absolute_tolerance_cycles_per_sample: float = 5.0e-8,
    relative_tolerance: float = 5.0e-6,
) -> Dict[str, Any]:
    artifact = Path(artifact_directory).resolve()
    output = artifact if output_directory is None else Path(output_directory).resolve()
    model_path = artifact / "temporal_model.npz"
    if not model_path.is_file():
        raise FrequencyAuditError("missing temporal model: {}".format(model_path))
    result = RVMDResult.load(model_path)
    n_time = int(result.c.shape[0])
    dt = float(result.info.get("dt_seconds", float("nan")))
    length = float(result.info.get("strouhal_length", float("nan")))
    velocity = float(result.info.get("strouhal_velocity", float("nan")))
    if not all(math.isfinite(value) and value > 0.0 for value in (dt, length, velocity)):
        raise FrequencyAuditError("temporal model has invalid dt or Strouhal references")
    center, bandwidth = coefficient_spectrum_moments(result.coefficient_spectrum, n_time)
    saved = np.asarray(result.omega_cycles_per_sample, dtype=np.float64)
    absolute_error = np.abs(center - saved)
    relative_error = absolute_error / np.maximum(np.abs(saved), np.finfo(float).tiny)
    frequency_hz = center / dt
    strouhal = frequency_hz * length / velocity
    windowed = _windowed_fft_centroid_cycles_per_sample(result.c)
    pass_mask = absolute_error <= (
        float(absolute_tolerance_cycles_per_sample)
        + float(relative_tolerance) * np.abs(saved)
    )
    formula_pass = bool(
        np.allclose(result.frequency_hz, frequency_hz, rtol=relative_tolerance, atol=1.0e-10)
        and np.allclose(result.strouhal, strouhal, rtol=relative_tolerance, atol=1.0e-10)
    )
    ordering_pass = bool(np.all(np.diff(frequency_hz) >= -1.0e-12))
    nyquist_hz = 0.5 / dt
    nyquist_pass = bool(np.all((frequency_hz >= -1.0e-12) & (frequency_hz <= nyquist_hz + 1.0e-10)))
    rows = []
    for index in range(saved.size):
        rows.append(
            {
                "mode": index + 1,
                "saved_center_cycles_per_sample": saved[index],
                "audited_center_cycles_per_sample": center[index],
                "absolute_error_cycles_per_sample": absolute_error[index],
                "relative_error": relative_error[index],
                "center_frequency_hz": frequency_hz[index],
                "strouhal_St": strouhal[index],
                "rms_bandwidth_cycles_per_sample": bandwidth[index],
                "rms_bandwidth_hz": bandwidth[index] / dt,
                "windowed_exported_coefficient_fft_centroid_hz_diagnostic": windowed[index] / dt,
                "center_frequency_audit_pass": bool(pass_mask[index]),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "frequency_audit.csv"
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)
    summary = {
        "status": "pass" if bool(np.all(pass_mask)) and formula_pass and ordering_pass and nyquist_pass else "fail",
        "frequency_definition": (
            "power centroid of each RVMD temporal coefficient on the one-sided spectrum "
            "of its 2N mirror extension; DC and Nyquist bins have half weight"
        ),
        "dt_seconds": dt,
        "sample_rate_hz": 1.0 / dt,
        "nyquist_hz": nyquist_hz,
        "frame_count": n_time,
        "relative_time_span_seconds": (n_time - 1) * dt,
        "mode_count": int(saved.size),
        "strouhal_definition": "St=f_Hz*{:.12g}/{:.12g}".format(length, velocity),
        "maximum_absolute_center_error_cycles_per_sample": float(np.max(absolute_error)),
        "maximum_relative_center_error": float(np.max(relative_error)),
        "all_modal_centers_pass": bool(np.all(pass_mask)),
        "frequency_and_strouhal_formula_pass": formula_pass,
        "ascending_frequency_order_pass": ordering_pass,
        "nyquist_gate_pass": nyquist_pass,
        "windowed_fft_centroid_role": (
            "diagnostic only; it is not the canonical RVMD center because truncating the "
            "mirror-extended coefficient changes spectral leakage"
        ),
        "outputs": {"per_mode_csv": str(csv_path)},
    }
    json_path = output / "frequency_audit.json"
    _write_json(json_path, summary)
    summary["outputs"]["summary_json"] = str(json_path)
    if summary["status"] != "pass":
        raise FrequencyAuditError("RVMD frequency audit failed; see {}".format(json_path))
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", help="RVMD artifact directory containing temporal_model.npz")
    parser.add_argument("--output-directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    summary = audit_frequency_artifact(
        Path(args.artifact_directory),
        None if args.output_directory is None else Path(args.output_directory),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["FrequencyAuditError", "audit_frequency_artifact"]
