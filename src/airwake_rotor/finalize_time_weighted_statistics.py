#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Rebuild response statistics as exact time-weighted window statistics.

The production output grid uses the user-specified interval but appends the
requested final time when it is not an integer multiple of that interval.  A
plain arithmetic sample mean would therefore overweight the short final
interval and would also omit a non-sampled analysis-window boundary.  This
postprocessor linearly interpolates both boundaries and uses trapezoidal time
integration for mean, standard deviation and RMS.  Peak quantities are taken
over the same boundary-augmented window.

The per-case dynamics CSV/JSON files are read-only inputs.  Only the aggregate
``05_statistics/statistics_long.csv`` is replaced; the original sample-based
table is preserved once under ``00_provenance``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CORE_METRICS = ("lift_N", "engine_power_W", "roll_deg", "pitch_deg", "heave_m")
STATISTICS = ("mean", "std", "rms", "peak_to_peak", "max_abs")
METHOD = "boundary_interpolated_trapezoidal_time_weighting_v1"


class StatisticsFinalizationError(RuntimeError):
    pass


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "rotor"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise StatisticsFinalizationError("JSON root must be an object: {}".format(path))
    return value


def boundary_augmented_window(
    time_s: Sequence[float],
    values: Sequence[float],
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """Return a strictly increasing window with exact, interpolated boundaries."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 2:
        raise StatisticsFinalizationError("time and signal must be equal one-dimensional arrays")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise StatisticsFinalizationError("time or signal contains non-finite values")
    if not np.all(np.diff(time) > 0.0):
        raise StatisticsFinalizationError("time must be strictly increasing")
    if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
        raise StatisticsFinalizationError("analysis window is invalid")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(start_s), abs(end_s))
    if time[0] > start_s + tolerance or time[-1] < end_s - tolerance:
        raise StatisticsFinalizationError("time series does not cover the analysis window")
    interior = (time > start_s + tolerance) & (time < end_s - tolerance)
    window_time = np.concatenate(([start_s], time[interior], [end_s]))
    window_values = np.interp(window_time, time, signal)
    start_interpolated = not bool(np.any(np.isclose(time, start_s, rtol=0.0, atol=tolerance)))
    end_interpolated = not bool(np.any(np.isclose(time, end_s, rtol=0.0, atol=tolerance)))
    if not np.all(np.diff(window_time) > 0.0):
        raise StatisticsFinalizationError("boundary-augmented time window is not strictly increasing")
    return window_time, window_values, start_interpolated, end_interpolated


def time_weighted_statistics(
    time_s: Sequence[float],
    values: Sequence[float],
    start_s: float,
    end_s: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Calculate time-weighted statistics over exactly ``[start_s, end_s]``."""

    window_time, window_values, start_interp, end_interp = boundary_augmented_window(
        time_s, values, start_s, end_s
    )
    duration = float(end_s - start_s)
    mean = float(np.trapz(window_values, window_time) / duration)
    mean_square = float(np.trapz(window_values ** 2, window_time) / duration)
    variance = float(np.trapz((window_values - mean) ** 2, window_time) / duration)
    statistics = {
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "rms": math.sqrt(max(mean_square, 0.0)),
        "peak_to_peak": float(np.ptp(window_values)),
        "max_abs": float(np.max(np.abs(window_values))),
    }
    audit = {
        "sample_count": int(window_time.size),
        "integration_interval_count": int(window_time.size - 1),
        "start_boundary_interpolated": bool(start_interp),
        "end_boundary_interpolated": bool(end_interp),
    }
    return statistics, audit


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise StatisticsFinalizationError("cannot write an empty statistics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def finalize_statistics(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    project_root = config_path.parent.parent
    output_dir = (project_root / str(config["study"]["output_dir"])).resolve()
    rotor_config = _read_json((project_root / str(config["rotor_model"]["base_rotor_config"])).resolve())
    rotors = rotor_config.get("rotors")
    if not isinstance(rotors, list) or not rotors:
        raise StatisticsFinalizationError("rotor configuration has no rotors")
    body = config["body_response"]
    mode_count = int(config["study"]["mode_count"])
    if mode_count <= 0 or int(config["rvmd"]["mode_count"]) != mode_count:
        raise StatisticsFinalizationError("study and RVMD mode counts must be equal and positive")
    index_width = max(2, len(str(mode_count)))
    start_s = float(body["analysis_window_start_s"])
    end_s = float(body["analysis_window_end_s"])
    time_series_dir = output_dir / "04_timeseries_csv"
    rows = []
    run_signatures = set()
    case_count = 0
    interpolated_start_cases = 0
    interpolated_end_cases = 0
    for rotor in rotors:
        rotor_name = str(rotor["name"])
        case_ids = [
            "stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
                stage=stage,
                retained=mode_count - stage,
                width=index_width,
            )
            for stage in range(mode_count + 1)
        ]
        case_ids.append("original_cfd")
        for case_id in case_ids:
            stem = "{}__{}".format(_slug(rotor_name), case_id)
            csv_path = time_series_dir / (stem + ".csv")
            metadata_path = time_series_dir / (stem + ".metadata.json")
            if not csv_path.is_file() or not metadata_path.is_file():
                raise StatisticsFinalizationError("missing production case pair: {}".format(stem))
            frame = pd.read_csv(str(csv_path), encoding="utf-8-sig")
            required = ["rotor_name", "rotor_diameter_m", "time_s"] + list(CORE_METRICS)
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise StatisticsFinalizationError("{} misses columns {}".format(csv_path, missing))
            if frame.empty or set(frame["rotor_name"].astype(str)) != {rotor_name}:
                raise StatisticsFinalizationError("rotor identity mismatch in {}".format(csv_path))
            numeric = frame[["time_s"] + list(CORE_METRICS)].apply(pd.to_numeric, errors="coerce")
            if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
                raise StatisticsFinalizationError("non-finite core time-series value in {}".format(csv_path))
            metadata = _read_json(metadata_path)
            definition = metadata.get("case_definition")
            if not isinstance(definition, dict) or str(definition.get("case_id")) != case_id:
                raise StatisticsFinalizationError("case metadata mismatch for {}".format(stem))
            signature = str(metadata.get("run_signature_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", signature):
                raise StatisticsFinalizationError("invalid run signature for {}".format(stem))
            run_signatures.add(signature)
            common_audit: Optional[Dict[str, Any]] = None
            for metric in CORE_METRICS:
                stats, audit = time_weighted_statistics(
                    numeric["time_s"].to_numpy(dtype=float),
                    numeric[metric].to_numpy(dtype=float),
                    start_s,
                    end_s,
                )
                if common_audit is None:
                    common_audit = audit
                elif audit != common_audit:
                    raise StatisticsFinalizationError("metric window audits disagree for {}".format(stem))
                for statistic in STATISTICS:
                    rows.append(
                        {
                            "rotor_name": rotor_name,
                            "rotor_diameter_m": float(frame["rotor_diameter_m"].iloc[0]),
                            "case_id": case_id,
                            "flow_kind": definition.get("flow_kind", ""),
                            "stage_index": "" if definition.get("stage_index") is None else definition["stage_index"],
                            "retained_mode_count": "" if definition.get("retained_mode_count") is None else definition["retained_mode_count"],
                            "cutoff_frequency_hz": "" if definition.get("cutoff_frequency_hz") is None else definition["cutoff_frequency_hz"],
                            "cutoff_St": "" if definition.get("cutoff_St") is None else definition["cutoff_St"],
                            "analysis_start_s": start_s,
                            "analysis_end_s": end_s,
                            "sample_count": audit["sample_count"],
                            "integration_interval_count": audit["integration_interval_count"],
                            "time_weighting_method": METHOD,
                            "start_boundary_interpolated": audit["start_boundary_interpolated"],
                            "end_boundary_interpolated": audit["end_boundary_interpolated"],
                            "metric": metric,
                            "statistic": statistic,
                            "value": stats[statistic],
                        }
                    )
            if common_audit is None:
                raise StatisticsFinalizationError("no statistics produced for {}".format(stem))
            interpolated_start_cases += int(common_audit["start_boundary_interpolated"])
            interpolated_end_cases += int(common_audit["end_boundary_interpolated"])
            case_count += 1
    expected_case_count = len(rotors) * (mode_count + 2)
    if (
        case_count != expected_case_count
        or len(rows) != expected_case_count * len(CORE_METRICS) * len(STATISTICS)
    ):
        raise StatisticsFinalizationError("unexpected aggregate table size")
    if len(run_signatures) != 1:
        raise StatisticsFinalizationError("production cases do not share one run signature")
    statistics_path = output_dir / "05_statistics" / "statistics_long.csv"
    backup_path = output_dir / "00_provenance" / "statistics_long_original_sample_mean.csv"
    if statistics_path.is_file() and not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(statistics_path), str(backup_path))
    _write_rows(statistics_path, rows)
    audit = {
        "status": "pass",
        "method": METHOD,
        "analysis_window_s": [start_s, end_s],
        "duration_s": end_s - start_s,
        "case_count": case_count,
        "statistics_row_count": len(rows),
        "rotor_count": len(rotors),
        "metrics": list(CORE_METRICS),
        "statistics": list(STATISTICS),
        "run_signature_sha256": next(iter(run_signatures)),
        "start_boundary_interpolated_case_count": interpolated_start_cases,
        "end_boundary_interpolated_case_count": interpolated_end_cases,
        "statistics_csv": str(statistics_path),
        "original_sample_mean_backup": str(backup_path) if backup_path.exists() else None,
    }
    _write_json(output_dir / "05_statistics" / "time_weighted_statistics_audit.json", audit)
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    args = parser.parse_args(argv)
    result = finalize_statistics(Path(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "METHOD",
    "StatisticsFinalizationError",
    "boundary_augmented_window",
    "finalize_statistics",
    "time_weighted_statistics",
]
