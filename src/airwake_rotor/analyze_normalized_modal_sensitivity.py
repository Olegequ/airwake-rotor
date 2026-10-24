#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Build normalized rotor-response sensitivity spectra for RVMD modes.

The analysis requires four flow cases for every rotor and RVMD mode:

``mean``
    The stored RVMD mean field only.
``full``
    The mean plus every retained RVMD mode.
``single``
    The mean plus one mode ``k``.
``leave_one_out``
    The mean plus every mode except ``k``.

For a response history ``y``, the isolated and conditional modal effects are

``delta_single = y(single_k) - y(mean)`` and
``delta_loo = y(full) - y(leave_one_out_k)``.

The primary response magnitude is the time-weighted RMS of the *centred*
difference in the common analysis window.  It is divided by a declared
response scale to obtain ``R``.  The unit-modal-amplitude gain is

``G = R / sqrt(E_k / E_0)``,

where ``E_k`` is the individual modal self-energy and ``E_0`` is the energy of
the full mean-removed CFD field.  RVMD spatial modes are generally
non-orthogonal: individual self-energies are therefore used only as per-mode
amplitude scales and are deliberately never summed to estimate captured flow
energy.

The leave-one-out spectrum is the primary spectrum because it measures the
marginal contribution in the complete nonlinear modal environment.  The
single-mode spectrum is retained as an isolated-response diagnostic.  Their
disagreement is reported as a bounded interaction index.

Inputs
------
``case_manifest.csv`` contains one row per time-history CSV with columns

``rotor_name, rotor_diameter_m, case_kind, mode_index_one_based, csv_path``.

``case_kind`` is one of ``full``, ``mean``, ``single`` or
``leave_one_out``.  ``mode_index_one_based`` is blank for ``full`` and
``mean`` and mandatory otherwise.  Relative CSV paths are resolved against
the manifest directory.

``analysis_config.json`` has the following compact schema::

    {
      "analysis_window_s": [1.0, 11.0],
      "reference_length_m": 13.716,
      "primary_estimator": "leave_one_out",
      "response_metrics": {
        "lift_N": {
          "unit": "N",
          "normalization_strategy": "mean_case_abs_mean"
        },
        "roll_deg": {
          "unit": "deg",
          "normalization_strategy": "constant",
          "normalization_value": 12.0
        }
      }
    }

For rotor-specific physical limits, a metric may instead declare
``normalization_strategy: rotor_specific_constant`` and
``normalization_values: {rotor_name: positive_value}``.

The RVMD directory must contain ``frequencies.csv`` and
``full_space_regression_audit.json``.  Outputs are
``mode_sensitivity_long.csv``, ``diameter_frequency_metrics.csv`` and a small
JSON audit/provenance record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CASE_KINDS = ("full", "mean", "single", "leave_one_out")
CASE_STATUSES = ("valid", "model_validity_failure")
ESTIMATORS = ("single", "leave_one_out")
PRIMARY_ESTIMATOR_DEFAULT = "leave_one_out"
SCHEMA_VERSION = "normalized-modal-sensitivity-v1"
NONORTHOGONAL_ENERGY_NOTE = (
    "individual_modal_self_energy_used_only_for_per_mode_amplitude_normalization;"
    "self_energies_not_summed_because_RVMD_spatial_modes_are_nonorthogonal"
)


class SensitivityAnalysisError(RuntimeError):
    """Raised when a sensitivity input violates an analysis invariant."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    strategy: str
    constant_value: Optional[float]
    rotor_values: Mapping[str, float]


@dataclass(frozen=True)
class AnalysisSettings:
    start_s: float
    end_s: float
    reference_length_m: float
    primary_estimator: str
    interaction_epsilon: float
    metrics: Tuple[MetricSpec, ...]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise SensitivityAnalysisError("cannot read JSON {}: {}".format(path, error)) from error
    if not isinstance(value, dict):
        raise SensitivityAnalysisError("JSON root must be an object: {}".format(path))
    return value


def _positive_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SensitivityAnalysisError("{} must be numeric".format(label)) from error
    if not math.isfinite(result) or result <= 0.0:
        raise SensitivityAnalysisError("{} must be positive and finite".format(label))
    return result


def load_analysis_settings(path: Path) -> AnalysisSettings:
    """Load and strictly validate the normalization/analysis contract."""

    raw = _read_json(path)
    window = raw.get("analysis_window_s")
    if not isinstance(window, list) or len(window) != 2:
        raise SensitivityAnalysisError("analysis_window_s must contain [start, end]")
    try:
        start_s, end_s = float(window[0]), float(window[1])
    except (TypeError, ValueError) as error:
        raise SensitivityAnalysisError("analysis_window_s must be numeric") from error
    if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
        raise SensitivityAnalysisError("analysis_window_s is invalid")

    reference_length_m = _positive_finite(
        raw.get("reference_length_m"), "reference_length_m"
    )
    primary = str(raw.get("primary_estimator", PRIMARY_ESTIMATOR_DEFAULT))
    if primary not in ESTIMATORS:
        raise SensitivityAnalysisError(
            "primary_estimator must be one of {}".format(ESTIMATORS)
        )
    epsilon = _positive_finite(
        raw.get("interaction_epsilon", 1.0e-15), "interaction_epsilon"
    )

    raw_metrics = raw.get("response_metrics")
    if not isinstance(raw_metrics, dict) or not raw_metrics:
        raise SensitivityAnalysisError("response_metrics must be a non-empty object")
    metrics: List[MetricSpec] = []
    for name, definition in raw_metrics.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(definition, dict):
            raise SensitivityAnalysisError("each response metric must have an object definition")
        strategy = str(definition.get("normalization_strategy", ""))
        if strategy not in (
            "mean_case_abs_mean",
            "constant",
            "rotor_specific_constant",
        ):
            raise SensitivityAnalysisError(
                "unsupported normalization_strategy for {}: {}".format(name, strategy)
            )
        constant_value: Optional[float] = None
        rotor_values: Dict[str, float] = {}
        if strategy == "constant":
            constant_value = _positive_finite(
                definition.get("normalization_value"),
                "{}.normalization_value".format(name),
            )
        if strategy == "rotor_specific_constant":
            values = definition.get("normalization_values")
            if not isinstance(values, dict) or not values:
                raise SensitivityAnalysisError(
                    "{}.normalization_values must be a non-empty object".format(name)
                )
            rotor_values = {
                str(rotor): _positive_finite(
                    value, "{}.normalization_values.{}".format(name, rotor)
                )
                for rotor, value in values.items()
            }
        metrics.append(
            MetricSpec(
                name=name,
                unit=str(definition.get("unit", "")),
                strategy=strategy,
                constant_value=constant_value,
                rotor_values=rotor_values,
            )
        )
    return AnalysisSettings(
        start_s=start_s,
        end_s=end_s,
        reference_length_m=reference_length_m,
        primary_estimator=primary,
        interaction_epsilon=epsilon,
        metrics=tuple(metrics),
    )


def _resolve_required_column(
    frame: pd.DataFrame, candidates: Iterable[str], label: str
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise SensitivityAnalysisError(
        "{} is missing; accepted columns are {}".format(label, tuple(candidates))
    )


def load_mode_metadata(rvmd_dir: Path) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Load modal frequencies and per-mode self energy without summing it."""

    frequency_path = rvmd_dir / "frequencies.csv"
    audit_path = rvmd_dir / "full_space_regression_audit.json"
    if not frequency_path.is_file():
        raise SensitivityAnalysisError("missing RVMD frequency table: {}".format(frequency_path))
    if not audit_path.is_file():
        raise SensitivityAnalysisError("missing RVMD full-space audit: {}".format(audit_path))
    try:
        frequency = pd.read_csv(str(frequency_path), encoding="utf-8-sig")
    except Exception as error:
        raise SensitivityAnalysisError(
            "cannot read RVMD frequency table {}: {}".format(frequency_path, error)
        ) from error
    if frequency.empty:
        raise SensitivityAnalysisError("RVMD frequency table is empty")
    mode_col = _resolve_required_column(
        frequency, ("mode", "mode_index_one_based"), "one-based mode index"
    )
    hz_col = _resolve_required_column(
        frequency, ("frequency_hz", "center_frequency_hz"), "centre frequency"
    )
    st_col = _resolve_required_column(
        frequency,
        ("strouhal_St_f13.716_over_20", "strouhal_St", "St", "st"),
        "Strouhal number",
    )
    energy_col = _resolve_required_column(
        frequency,
        (
            "full_normalized_modal_self_energy",
            "individual_modal_self_energy",
            "modal_self_energy",
        ),
        "individual modal self-energy",
    )
    selected = pd.DataFrame(
        {
            "mode_index_one_based": pd.to_numeric(frequency[mode_col], errors="coerce"),
            "frequency_hz": pd.to_numeric(frequency[hz_col], errors="coerce"),
            "St": pd.to_numeric(frequency[st_col], errors="coerce"),
            "mode_self_energy_Ek": pd.to_numeric(frequency[energy_col], errors="coerce"),
        }
    )
    if not np.all(np.isfinite(selected.to_numpy(dtype=float))):
        raise SensitivityAnalysisError("RVMD mode metadata contains non-finite values")
    rounded = np.rint(selected["mode_index_one_based"].to_numpy(dtype=float))
    if not np.allclose(
        selected["mode_index_one_based"].to_numpy(dtype=float), rounded, rtol=0.0, atol=1e-12
    ):
        raise SensitivityAnalysisError("RVMD mode indices must be integers")
    selected["mode_index_one_based"] = rounded.astype(int)
    if (selected["mode_index_one_based"] <= 0).any() or selected[
        "mode_index_one_based"
    ].duplicated().any():
        raise SensitivityAnalysisError("RVMD mode indices must be unique positive integers")
    if (selected["frequency_hz"] < 0.0).any() or (selected["St"] < 0.0).any():
        raise SensitivityAnalysisError("RVMD frequencies and Strouhal numbers cannot be negative")
    if (selected["mode_self_energy_Ek"] <= 0.0).any():
        raise SensitivityAnalysisError("every modal self-energy must be positive")

    audit = _read_json(audit_path)
    E0 = _positive_finite(
        audit.get("full_normalized_centered_data_energy"),
        "full_normalized_centered_data_energy",
    )
    selected = selected.sort_values(["St", "mode_index_one_based"]).reset_index(drop=True)
    return selected, E0, audit


def load_case_manifest(path: Path, expected_modes: Sequence[int]) -> pd.DataFrame:
    """Read the explicit mapping from physical cases to time-history files."""

    try:
        frame = pd.read_csv(str(path), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception as error:
        raise SensitivityAnalysisError("cannot read case manifest {}: {}".format(path, error)) from error
    required = (
        "rotor_name",
        "rotor_diameter_m",
        "case_kind",
        "mode_index_one_based",
        "csv_path",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SensitivityAnalysisError("case manifest misses columns {}".format(missing))
    if frame.empty:
        raise SensitivityAnalysisError("case manifest is empty")

    rows: List[Dict[str, Any]] = []
    manifest_parent = path.resolve().parent
    for index, raw in frame.iterrows():
        rotor = str(raw["rotor_name"]).strip()
        if not rotor:
            raise SensitivityAnalysisError("manifest row {} has an empty rotor_name".format(index + 2))
        diameter = _positive_finite(
            raw["rotor_diameter_m"], "manifest row {} rotor_diameter_m".format(index + 2)
        )
        kind = str(raw["case_kind"]).strip()
        if kind not in CASE_KINDS:
            raise SensitivityAnalysisError(
                "manifest row {} has unsupported case_kind {}".format(index + 2, kind)
            )
        mode_text = str(raw["mode_index_one_based"]).strip()
        if kind in ("full", "mean"):
            if mode_text:
                raise SensitivityAnalysisError(
                    "{} case must have a blank mode index".format(kind)
                )
            mode: Optional[int] = None
        else:
            try:
                mode_float = float(mode_text)
            except ValueError as error:
                raise SensitivityAnalysisError(
                    "{} case has an invalid mode index".format(kind)
                ) from error
            mode = int(round(mode_float))
            if abs(mode_float - mode) > 1e-12 or mode <= 0:
                raise SensitivityAnalysisError("case mode index must be a positive integer")
        status = str(raw.get("case_status", "valid")).strip() or "valid"
        if status not in CASE_STATUSES:
            raise SensitivityAnalysisError(
                "manifest row {} has unsupported case_status {}".format(index + 2, status)
            )
        if kind in ("full", "mean") and status != "valid":
            raise SensitivityAnalysisError("full and mean cases must be valid")
        path_text = str(raw["csv_path"]).strip()
        case_path: Optional[Path]
        if status == "valid":
            if not path_text:
                raise SensitivityAnalysisError(
                    "valid manifest row {} has an empty csv_path".format(index + 2)
                )
            case_path = Path(path_text)
            if not case_path.is_absolute():
                case_path = manifest_parent / case_path
            case_path = case_path.resolve()
            if not case_path.is_file():
                raise SensitivityAnalysisError(
                    "time-history CSV does not exist: {}".format(case_path)
                )
        else:
            if kind not in ("single", "leave_one_out"):
                raise SensitivityAnalysisError(
                    "only modal cases may be marked as model_validity_failure"
                )
            case_path = None
        rows.append(
            {
                "rotor_name": rotor,
                "rotor_diameter_m": diameter,
                "case_kind": kind,
                "mode_index_one_based": mode,
                "csv_path": case_path,
                "case_status": status,
                "failure_classification": str(
                    raw.get("failure_classification", "")
                ).strip(),
                "failure_reason": str(raw.get("failure_reason", "")).strip(),
            }
        )
    result = pd.DataFrame(rows)

    expected = set(int(value) for value in expected_modes)
    for rotor, group in result.groupby("rotor_name", sort=False):
        diameters = set(group["rotor_diameter_m"].astype(float))
        if len(diameters) != 1:
            raise SensitivityAnalysisError("rotor {} has inconsistent diameters".format(rotor))
        for kind in ("full", "mean"):
            subset = group[group["case_kind"] == kind]
            count = int(len(subset))
            if count != 1 or str(subset.iloc[0]["case_status"]) != "valid":
                raise SensitivityAnalysisError(
                    "rotor {} requires exactly one valid {} case, found {}".format(
                        rotor, kind, count
                    )
                )
        for kind in ("single", "leave_one_out"):
            subset = group[group["case_kind"] == kind]
            indices = subset["mode_index_one_based"].astype(int).tolist()
            if len(indices) != len(set(indices)):
                raise SensitivityAnalysisError(
                    "rotor {} has duplicate {} mode cases".format(rotor, kind)
                )
            actual = set(indices)
            if actual != expected:
                raise SensitivityAnalysisError(
                    "rotor {} {} modes do not match RVMD metadata; missing={}, extra={}".format(
                        rotor, kind, sorted(expected - actual), sorted(actual - expected)
                    )
                )
            invalid = subset[subset["case_status"] != "valid"]
            if not invalid.empty and (
                invalid["failure_classification"].astype(str).str.len() == 0
            ).any():
                raise SensitivityAnalysisError(
                    "rotor {} {} invalid cases require a failure classification".format(
                        rotor, kind
                    )
                )
    return result


def _load_history(
    path: Path,
    rotor_name: str,
    rotor_diameter_m: float,
    metric_names: Sequence[str],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(str(path), encoding="utf-8-sig")
    except Exception as error:
        raise SensitivityAnalysisError("cannot read time history {}: {}".format(path, error)) from error
    required = ["time_s"] + list(metric_names)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SensitivityAnalysisError("{} misses columns {}".format(path, missing))
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    if numeric.shape[0] < 2 or not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise SensitivityAnalysisError("{} has insufficient or non-finite numeric data".format(path))
    time = numeric["time_s"].to_numpy(dtype=float)
    if not np.all(np.diff(time) > 0.0):
        raise SensitivityAnalysisError("time_s must be strictly increasing in {}".format(path))
    if "rotor_name" in frame.columns:
        names = set(frame["rotor_name"].astype(str))
        if names != {rotor_name}:
            raise SensitivityAnalysisError("rotor identity mismatch in {}".format(path))
    if "rotor_diameter_m" in frame.columns:
        diameter = pd.to_numeric(frame["rotor_diameter_m"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(diameter)) or not np.allclose(
            diameter, rotor_diameter_m, rtol=0.0, atol=1.0e-12
        ):
            raise SensitivityAnalysisError("rotor diameter mismatch in {}".format(path))
    return numeric


def _common_time_grid(histories: Mapping[Tuple[str, Optional[int]], pd.DataFrame]) -> np.ndarray:
    reference: Optional[np.ndarray] = None
    for key, frame in histories.items():
        time = frame["time_s"].to_numpy(dtype=float)
        if reference is None:
            reference = time
            continue
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(reference))), float(np.max(np.abs(time)))
        )
        if time.shape != reference.shape or not np.allclose(
            time, reference, rtol=0.0, atol=tolerance
        ):
            raise SensitivityAnalysisError(
                "all full/mean/single/leave-one-out histories for one rotor must share one time grid; "
                "mismatch at {}".format(key)
            )
    if reference is None:
        raise SensitivityAnalysisError("no time histories were loaded")
    return reference


def boundary_augmented_signal(
    time_s: Sequence[float], values: Sequence[float], start_s: float, end_s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a signal on the exact requested window including interpolated boundaries."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 2:
        raise SensitivityAnalysisError("time and signal must be equal one-dimensional arrays")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(start_s), abs(end_s))
    if time[0] > start_s + tolerance or time[-1] < end_s - tolerance:
        raise SensitivityAnalysisError("time history does not cover the common analysis window")
    interior = (time > start_s + tolerance) & (time < end_s - tolerance)
    window_time = np.concatenate(([start_s], time[interior], [end_s]))
    window_signal = np.interp(window_time, time, signal)
    if not np.all(np.diff(window_time) > 0.0):
        raise SensitivityAnalysisError("boundary-augmented time grid is not strictly increasing")
    return window_time, window_signal


def time_weighted_mean_and_dynamic_rms(
    time_s: Sequence[float], values: Sequence[float], start_s: float, end_s: float
) -> Tuple[float, float]:
    """Return signed mean and centred RMS with trapezoidal time weighting."""

    time, signal = boundary_augmented_signal(time_s, values, start_s, end_s)
    duration = float(end_s - start_s)
    mean = float(np.trapz(signal, time) / duration)
    variance = float(np.trapz((signal - mean) ** 2, time) / duration)
    return mean, math.sqrt(max(variance, 0.0))


def _response_scale(
    spec: MetricSpec,
    rotor_name: str,
    time_s: np.ndarray,
    mean_values: np.ndarray,
    start_s: float,
    end_s: float,
) -> Tuple[float, str]:
    if spec.strategy == "constant":
        assert spec.constant_value is not None
        return spec.constant_value, "constant_physical_reference"
    if spec.strategy == "rotor_specific_constant":
        if rotor_name not in spec.rotor_values:
            raise SensitivityAnalysisError(
                "metric {} has no normalization value for rotor {}".format(spec.name, rotor_name)
            )
        return float(spec.rotor_values[rotor_name]), "rotor_specific_physical_reference"
    mean_value, _ = time_weighted_mean_and_dynamic_rms(
        time_s, mean_values, start_s, end_s
    )
    scale = abs(mean_value)
    if not math.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise SensitivityAnalysisError(
            "mean-case normalization for {} / {} is zero or invalid".format(rotor_name, spec.name)
        )
    return scale, "absolute_time_mean_of_mean_flow_case"


def _safe_interaction(a: float, b: float, epsilon: float) -> float:
    if a == 0.0 and b == 0.0:
        return 0.0
    return abs(a - b) / (abs(a) + abs(b) + epsilon)


def _normalise_gain(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    gains = np.asarray(values, dtype=float)
    if gains.ndim != 1 or np.any(~np.isfinite(gains)) or np.any(gains < 0.0):
        raise SensitivityAnalysisError("gain spectrum must be a finite non-negative vector")
    l1_norm = float(np.sum(gains))
    l2_norm = float(np.linalg.norm(gains))
    l1 = gains / l1_norm if l1_norm > 0.0 else np.zeros_like(gains)
    l2 = gains / l2_norm if l2_norm > 0.0 else np.zeros_like(gains)
    return l1, l2, l1_norm, l2_norm


def _normalise_available_gain(
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Normalise valid entries while preserving marked invalid entries as NaN."""

    gains = np.asarray(values, dtype=float)
    if gains.ndim != 1 or np.any(gains[np.isfinite(gains)] < 0.0):
        raise SensitivityAnalysisError(
            "available gain spectrum must be one-dimensional and non-negative"
        )
    available = np.isfinite(gains)
    l1 = np.full_like(gains, np.nan)
    l2 = np.full_like(gains, np.nan)
    if not np.any(available):
        return l1, l2, 0.0, 0.0, available
    valid_l1, valid_l2, total, l2_norm = _normalise_gain(gains[available])
    l1[available] = valid_l1
    l2[available] = valid_l2
    return l1, l2, total, l2_norm, available


def _step_quantile(support: np.ndarray, weights: np.ndarray, probability: float) -> float:
    if not (0.0 < probability <= 1.0):
        raise ValueError("probability must be in (0, 1]")
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(support[min(index, support.size - 1)])


def frequency_spectrum_metrics(
    support: Sequence[float], gains: Sequence[float]
) -> Dict[str, Any]:
    """Summarize a discrete non-negative gain spectrum on sorted support."""

    x = np.asarray(support, dtype=float)
    g = np.asarray(gains, dtype=float)
    if x.ndim != 1 or g.shape != x.shape or x.size == 0:
        raise SensitivityAnalysisError("frequency support and gains must be equal non-empty vectors")
    if np.any(~np.isfinite(x)) or np.any(x < 0.0) or np.any(~np.isfinite(g)) or np.any(g < 0.0):
        raise SensitivityAnalysisError("frequency spectrum contains invalid values")
    order = np.argsort(x, kind="stable")
    x = x[order]
    g = g[order]
    weights, _, total, l2_norm = _normalise_gain(g)
    if total <= 0.0:
        return {
            "spectrum_status": "zero_total_gain",
            "peak": math.nan,
            "geometric_centroid": math.nan,
            "q50": math.nan,
            "q90": math.nan,
            "log_bandwidth": math.nan,
            "total_gain": 0.0,
            "l2_gain_norm": 0.0,
            "effective_mode_count": 0.0,
            "nonpositive_support_weight": 0.0,
        }
    peak = float(x[int(np.argmax(g))])
    q50 = _step_quantile(x, weights, 0.5)
    q90 = _step_quantile(x, weights, 0.9)
    nonpositive_weight = float(np.sum(weights[x <= 0.0]))
    if nonpositive_weight > 0.0:
        geometric = math.nan
        bandwidth = math.nan
        status = "geometric_metrics_undefined_nonpositive_support"
    else:
        log_x = np.log(x)
        mean_log = float(np.sum(weights * log_x))
        geometric = math.exp(mean_log)
        bandwidth = math.sqrt(max(float(np.sum(weights * (log_x - mean_log) ** 2)), 0.0))
        status = "ok"
    return {
        "spectrum_status": status,
        "peak": peak,
        "geometric_centroid": geometric,
        "q50": q50,
        "q90": q90,
        "log_bandwidth": bandwidth,
        "total_gain": total,
        "l2_gain_norm": l2_norm,
        "effective_mode_count": float(1.0 / np.sum(weights ** 2)),
        "nonpositive_support_weight": nonpositive_weight,
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(str(temporary), index=False, encoding="utf-8-sig", float_format="%.15g")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def analyze_normalized_modal_sensitivity(
    case_manifest_path: Path,
    rvmd_dir: Path,
    analysis_config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the complete normalized modal-sensitivity postprocessing workflow."""

    case_manifest_path = case_manifest_path.resolve()
    rvmd_dir = rvmd_dir.resolve()
    analysis_config_path = analysis_config_path.resolve()
    output_dir = output_dir.resolve()
    settings = load_analysis_settings(analysis_config_path)
    modes, E0, full_space_audit = load_mode_metadata(rvmd_dir)
    mode_indices = modes["mode_index_one_based"].astype(int).tolist()
    manifest = load_case_manifest(case_manifest_path, mode_indices)
    metric_names = [spec.name for spec in settings.metrics]

    long_rows: List[Dict[str, Any]] = []
    diameter_rows: List[Dict[str, Any]] = []
    for rotor_name, rotor_cases in manifest.groupby("rotor_name", sort=False):
        rotor_name = str(rotor_name)
        diameter_m = float(rotor_cases["rotor_diameter_m"].iloc[0])
        histories: Dict[Tuple[str, Optional[int]], pd.DataFrame] = {}
        case_records: Dict[Tuple[str, Optional[int]], Mapping[str, Any]] = {}
        for _, case in rotor_cases.iterrows():
            mode_value = case["mode_index_one_based"]
            mode_index = None if pd.isna(mode_value) else int(mode_value)
            key = (str(case["case_kind"]), mode_index)
            case_records[key] = case.to_dict()
            if str(case["case_status"]) == "valid":
                histories[key] = _load_history(
                    Path(case["csv_path"]), rotor_name, diameter_m, metric_names
                )
        common_time = _common_time_grid(histories)
        if common_time[0] > settings.start_s or common_time[-1] < settings.end_s:
            raise SensitivityAnalysisError(
                "rotor {} histories do not cover the common analysis window".format(rotor_name)
            )

        for metric_spec in settings.metrics:
            mean_values = histories[("mean", None)][metric_spec.name].to_numpy(dtype=float)
            full_values = histories[("full", None)][metric_spec.name].to_numpy(dtype=float)
            scale, scale_basis = _response_scale(
                metric_spec,
                rotor_name,
                common_time,
                mean_values,
                settings.start_s,
                settings.end_s,
            )
            metric_rows: List[MutableMapping[str, Any]] = []
            for mode_record in modes.to_dict(orient="records"):
                mode = int(mode_record["mode_index_one_based"])
                single_record = case_records[("single", mode)]
                loo_record = case_records[("leave_one_out", mode)]
                single_valid = str(single_record["case_status"]) == "valid"
                loo_valid = str(loo_record["case_status"]) == "valid"
                if single_valid:
                    single_values = histories[("single", mode)][
                        metric_spec.name
                    ].to_numpy(dtype=float)
                    mean_single, rms_single = time_weighted_mean_and_dynamic_rms(
                        common_time,
                        single_values - mean_values,
                        settings.start_s,
                        settings.end_s,
                    )
                else:
                    mean_single = rms_single = math.nan
                if loo_valid:
                    loo_values = histories[("leave_one_out", mode)][
                        metric_spec.name
                    ].to_numpy(dtype=float)
                    mean_loo, rms_loo = time_weighted_mean_and_dynamic_rms(
                        common_time,
                        full_values - loo_values,
                        settings.start_s,
                        settings.end_s,
                    )
                else:
                    mean_loo = rms_loo = math.nan
                Ek = float(mode_record["mode_self_energy_Ek"])
                amplitude = math.sqrt(Ek / E0)
                R_single = rms_single / scale
                R_loo = rms_loo / scale
                G_single = R_single / amplitude
                G_loo = R_loo / amplitude
                St = float(mode_record["St"])
                frequency_hz = float(mode_record["frequency_hz"])
                St_rotor = St * diameter_m / settings.reference_length_m
                metric_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "rotor_name": rotor_name,
                        "rotor_diameter_m": diameter_m,
                        "reference_length_m": settings.reference_length_m,
                        "metric": metric_spec.name,
                        "physical_unit": metric_spec.unit,
                        "mode_index_one_based": mode,
                        "center_frequency_hz": frequency_hz,
                        "St_reference_length": St,
                        "St_rotor_diameter": St_rotor,
                        "analysis_start_s": settings.start_s,
                        "analysis_end_s": settings.end_s,
                        "response_scale_value": scale,
                        "response_scale_basis": scale_basis,
                        "full_centered_flow_energy_E0": E0,
                        "mode_self_energy_Ek": Ek,
                        "modal_amplitude_ratio_sqrt_Ek_over_E0": amplitude,
                        "modal_energy_use": NONORTHOGONAL_ENERGY_NOTE,
                        "single_case_status": str(single_record["case_status"]),
                        "single_failure_classification": str(
                            single_record.get("failure_classification", "")
                        ),
                        "single_failure_reason": str(
                            single_record.get("failure_reason", "")
                        ),
                        "leave_one_out_case_status": str(loo_record["case_status"]),
                        "leave_one_out_failure_classification": str(
                            loo_record.get("failure_classification", "")
                        ),
                        "leave_one_out_failure_reason": str(
                            loo_record.get("failure_reason", "")
                        ),
                        "single_mean_effect_physical": mean_single,
                        "leave_one_out_mean_effect_physical": mean_loo,
                        "single_mean_effect_normalized_signed": mean_single / scale,
                        "leave_one_out_mean_effect_normalized_signed": mean_loo / scale,
                        "single_dynamic_rms_physical": rms_single,
                        "leave_one_out_dynamic_rms_physical": rms_loo,
                        "R_single": R_single,
                        "R_leave_one_out": R_loo,
                        "G_single": G_single,
                        "G_leave_one_out": G_loo,
                        "interaction_index": _safe_interaction(
                            G_single, G_loo, settings.interaction_epsilon
                        ),
                        "primary_estimator": settings.primary_estimator,
                    }
                )

            for estimator in ESTIMATORS:
                gain_key = "G_{}".format(estimator)
                gains = np.asarray([float(row[gain_key]) for row in metric_rows])
                l1, l2, total, l2_norm, available = _normalise_available_gain(gains)
                for row, l1_value, l2_value in zip(metric_rows, l1, l2):
                    row["sensitivity_L1_{}".format(estimator)] = float(l1_value)
                    row["sensitivity_L2_{}".format(estimator)] = float(l2_value)
                    row["total_gain_G_{}".format(estimator)] = total
                    row["gain_L2_norm_{}".format(estimator)] = l2_norm
                    row["available_mode_count_{}".format(estimator)] = int(
                        np.sum(available)
                    )
                    row["invalid_mode_count_{}".format(estimator)] = int(
                        len(available) - np.sum(available)
                    )

                response_key = "R_{}".format(estimator)
                responses = np.asarray(
                    [float(row[response_key]) for row in metric_rows], dtype=float
                )
                response_l1, response_l2, response_total, response_l2_norm, _ = (
                    _normalise_available_gain(responses)
                )
                for row, l1_value, l2_value in zip(
                    metric_rows, response_l1, response_l2
                ):
                    row["response_sensitivity_L1_{}".format(estimator)] = float(
                        l1_value
                    )
                    row["response_sensitivity_L2_{}".format(estimator)] = float(
                        l2_value
                    )
                    row["total_response_R_{}".format(estimator)] = response_total
                    row["response_L2_norm_{}".format(estimator)] = response_l2_norm

            for row in metric_rows:
                primary = settings.primary_estimator
                row["R_primary"] = row["R_{}".format(primary)]
                row["G_primary"] = row["G_{}".format(primary)]
                row["sensitivity_L1_primary"] = row[
                    "sensitivity_L1_{}".format(primary)
                ]
                row["sensitivity_L2_primary"] = row[
                    "sensitivity_L2_{}".format(primary)
                ]
                row["response_sensitivity_L1_primary"] = row[
                    "response_sensitivity_L1_{}".format(primary)
                ]
                row["response_sensitivity_L2_primary"] = row[
                    "response_sensitivity_L2_{}".format(primary)
                ]
            long_rows.extend(dict(row) for row in metric_rows)

            St_support = np.asarray(
                [float(row["St_reference_length"]) for row in metric_rows], dtype=float
            )
            frequency_support = np.asarray(
                [float(row["center_frequency_hz"]) for row in metric_rows], dtype=float
            )
            interaction = np.asarray(
                [float(row["interaction_index"]) for row in metric_rows], dtype=float
            )
            for estimator in ESTIMATORS:
                gains = np.asarray(
                    [float(row["G_{}".format(estimator)]) for row in metric_rows], dtype=float
                )
                available = np.isfinite(gains)
                St_metrics = frequency_spectrum_metrics(
                    St_support[available], gains[available]
                )
                hz_metrics = frequency_spectrum_metrics(
                    frequency_support[available], gains[available]
                )
                weights = np.zeros_like(gains)
                if np.any(available):
                    weights[available], _, _, _ = _normalise_gain(gains[available])
                responses = np.asarray(
                    [float(row["R_{}".format(estimator)]) for row in metric_rows],
                    dtype=float,
                )
                response_available = np.isfinite(responses)
                response_St_metrics = frequency_spectrum_metrics(
                    St_support[response_available], responses[response_available]
                )
                response_hz_metrics = frequency_spectrum_metrics(
                    frequency_support[response_available], responses[response_available]
                )
                invalid_indices = [
                    int(metric_rows[index]["mode_index_one_based"])
                    for index in np.flatnonzero(~available)
                ]
                spectrum_completeness = (
                    "complete"
                    if not invalid_indices
                    else "valid_only_incomplete_model_domain"
                )
                interaction_available = np.isfinite(interaction) & (weights > 0.0)
                diameter_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "rotor_name": rotor_name,
                        "rotor_diameter_m": diameter_m,
                        "reference_length_m": settings.reference_length_m,
                        "metric": metric_spec.name,
                        "physical_unit": metric_spec.unit,
                        "estimator": estimator,
                        "is_primary_estimator": estimator == settings.primary_estimator,
                        "response_scale_value": scale,
                        "response_scale_basis": scale_basis,
                        "analysis_start_s": settings.start_s,
                        "analysis_end_s": settings.end_s,
                        "spectrum_completeness": spectrum_completeness,
                        "valid_mode_count": int(np.sum(available)),
                        "invalid_mode_count": int(np.sum(~available)),
                        "invalid_mode_indices_one_based": ";".join(
                            str(value) for value in invalid_indices
                        ),
                        "spectrum_status": St_metrics["spectrum_status"],
                        "total_gain_G": St_metrics["total_gain"],
                        "gain_L2_norm": St_metrics["l2_gain_norm"],
                        "effective_mode_count": St_metrics["effective_mode_count"],
                        "St_peak": St_metrics["peak"],
                        "St_g": St_metrics["geometric_centroid"],
                        "St50": St_metrics["q50"],
                        "St90": St_metrics["q90"],
                        "log_St_bandwidth": St_metrics["log_bandwidth"],
                        "nonpositive_St_spectral_weight": St_metrics[
                            "nonpositive_support_weight"
                        ],
                        "frequency_peak_hz": hz_metrics["peak"],
                        "frequency_g_hz": hz_metrics["geometric_centroid"],
                        "frequency50_hz": hz_metrics["q50"],
                        "frequency90_hz": hz_metrics["q90"],
                        "StD_peak": St_metrics["peak"]
                        * diameter_m
                        / settings.reference_length_m,
                        "StD_g": St_metrics["geometric_centroid"]
                        * diameter_m
                        / settings.reference_length_m,
                        "StD50": St_metrics["q50"]
                        * diameter_m
                        / settings.reference_length_m,
                        "StD90": St_metrics["q90"]
                        * diameter_m
                        / settings.reference_length_m,
                        "response_spectrum_status": response_St_metrics[
                            "spectrum_status"
                        ],
                        "response_total_R": response_St_metrics["total_gain"],
                        "response_effective_mode_count": response_St_metrics[
                            "effective_mode_count"
                        ],
                        "response_St_peak": response_St_metrics["peak"],
                        "response_St_g": response_St_metrics[
                            "geometric_centroid"
                        ],
                        "response_St50": response_St_metrics["q50"],
                        "response_St90": response_St_metrics["q90"],
                        "response_log_St_bandwidth": response_St_metrics[
                            "log_bandwidth"
                        ],
                        "response_frequency_peak_hz": response_hz_metrics["peak"],
                        "response_frequency_g_hz": response_hz_metrics[
                            "geometric_centroid"
                        ],
                        "response_frequency50_hz": response_hz_metrics["q50"],
                        "response_frequency90_hz": response_hz_metrics["q90"],
                        "response_StD_peak": response_St_metrics["peak"]
                        * diameter_m
                        / settings.reference_length_m,
                        "response_StD_g": response_St_metrics[
                            "geometric_centroid"
                        ]
                        * diameter_m
                        / settings.reference_length_m,
                        "response_StD50": response_St_metrics["q50"]
                        * diameter_m
                        / settings.reference_length_m,
                        "response_StD90": response_St_metrics["q90"]
                        * diameter_m
                        / settings.reference_length_m,
                        "gain_weighted_mean_interaction_index": float(
                            np.sum(
                                weights[interaction_available]
                                * interaction[interaction_available]
                            )
                        )
                        if np.any(interaction_available)
                        else math.nan,
                        "maximum_interaction_index": float(
                            np.max(interaction[np.isfinite(interaction)])
                        )
                        if np.any(np.isfinite(interaction))
                        else math.nan,
                    }
                )

    long_frame = pd.DataFrame(long_rows).sort_values(
        ["rotor_diameter_m", "rotor_name", "metric", "St_reference_length"],
        ascending=[False, True, True, True],
    )
    diameter_frame = pd.DataFrame(diameter_rows).sort_values(
        ["metric", "estimator", "rotor_diameter_m", "rotor_name"],
        ascending=[True, True, False, True],
    )
    mode_output = output_dir / "mode_sensitivity_long.csv"
    diameter_output = output_dir / "diameter_frequency_metrics.csv"
    _write_csv(mode_output, long_frame)
    _write_csv(diameter_output, diameter_frame)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "case_manifest": str(case_manifest_path),
        "analysis_config": str(analysis_config_path),
        "rvmd_directory": str(rvmd_dir),
        "mode_sensitivity_csv": str(mode_output),
        "diameter_frequency_metrics_csv": str(diameter_output),
        "rotor_count": int(manifest["rotor_name"].nunique()),
        "mode_count": int(len(modes)),
        "metric_count": int(len(settings.metrics)),
        "mode_sensitivity_row_count": int(len(long_frame)),
        "diameter_frequency_row_count": int(len(diameter_frame)),
        "analysis_window_s": [settings.start_s, settings.end_s],
        "primary_estimator": settings.primary_estimator,
        "full_centered_flow_energy_E0": E0,
        "captured_energy_fraction_one_minus_residual": full_space_audit.get(
            "full_normalized_captured_energy_fraction_one_minus_residual"
        ),
        "modal_energy_policy": NONORTHOGONAL_ENERGY_NOTE,
        "response_definition": (
            "R=time_weighted_centered_difference_RMS/declared_response_scale; "
            "G=R/sqrt(E_k/E_0)"
        ),
        "frequency_metric_definition": (
            "computed independently for actual response R and unit-modal-energy gain G: "
            "peak=max-weight support; St_g=L1-weighted geometric centroid; "
            "St50/St90=first discrete support where cumulative L1 weight reaches 50%/90%; "
            "log_St_bandwidth=L1-weighted standard deviation of ln(St)"
        ),
        "marked_model_validity_failure_count": int(
            (manifest["case_status"] == "model_validity_failure").sum()
        ),
        "marked_model_validity_failures": [
            {
                "rotor_name": str(row["rotor_name"]),
                "case_kind": str(row["case_kind"]),
                "mode_index_one_based": int(row["mode_index_one_based"]),
                "failure_classification": str(row["failure_classification"]),
                "failure_reason": str(row["failure_reason"]),
            }
            for _, row in manifest[
                manifest["case_status"] == "model_validity_failure"
            ].iterrows()
        ],
        "incomplete_spectrum_policy": (
            "marked invalid modes remain NaN in the long table; valid-only normalized "
            "spectra and frequency summaries are reported with explicit completeness flags"
        ),
    }
    _write_json(output_dir / "normalized_sensitivity_audit.json", audit)
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--rvmd-dir", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    audit = analyze_normalized_modal_sensitivity(
        args.case_manifest, args.rvmd_dir, args.analysis_config, args.output_dir
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "NONORTHOGONAL_ENERGY_NOTE",
    "SCHEMA_VERSION",
    "SensitivityAnalysisError",
    "analyze_normalized_modal_sensitivity",
    "boundary_augmented_signal",
    "frequency_spectrum_metrics",
    "load_analysis_settings",
    "load_case_manifest",
    "load_mode_metadata",
    "time_weighted_mean_and_dynamic_rms",
]
