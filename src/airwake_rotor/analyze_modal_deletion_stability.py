#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Audit finite-window closed-loop stability across all RVMD deletion stages.

The formal observed-instability flag is deliberately conservative and
reproducible: a case is unstable only when the solver reports a hard failure
or a configured rigid-body/dynamic-inflow safety limit is crossed.  Aerodynamic
model-domain validity and terminal-growth advisories are reported separately;
neither is promoted to asymptotic instability because every case is a finite,
non-autonomous, forced response.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .finalize_time_weighted_statistics import boundary_augmented_window
from .nato_gd_study import load_study_context


BODY_STATE_COLUMNS = ("heave_m", "heave_velocity_mps", "roll_deg", "pitch_deg")
INFLOW_STATE_COLUMNS = ("delta_lambda_0", "delta_lambda_c", "delta_lambda_s", "delta_mu_z")
RESPONSE_COLUMNS = BODY_STATE_COLUMNS + INFLOW_STATE_COLUMNS
FORBIDDEN_VALIDITY_COLUMNS = (
    "reverse_flow_count",
    "alpha_domain_violation_count",
    "mach_domain_violation_count",
    "reynolds_domain_violation_count",
)
GROWTH_BLOCK_COUNT = 4
GROWTH_BLOCK_DURATION_S = 1.0
GROWTH_RATIO_ADVISORY = 1.25
GROWTH_MINIMUM_FINAL_PEAK = 0.25
MINIMUM_SUSTAINED_UNSTABLE_STAGE_COUNT = 3


class StabilityAuditError(RuntimeError):
    pass


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "rotor"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise StabilityAuditError("JSON root must be an object: {}".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(str(temporary), index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _validated_stage_manifest(
    stage_manifest: pd.DataFrame, expected_mode_count: int
) -> Tuple[pd.DataFrame, int]:
    """Validate and index the cumulative high-frequency-deletion contract.

    The manifest is the authoritative source for case identifiers.  This is
    important once ``K`` reaches three digits because the production naming
    convention widens both numeric fields automatically.
    """

    required_columns = {
        "stage_index",
        "case_id",
        "retained_mode_count",
        "deleted_mode_this_stage",
        "deleted_frequency_hz_this_stage",
        "deleted_St_this_stage",
        "cutoff_frequency_hz",
        "cutoff_St",
    }
    if not required_columns.issubset(stage_manifest.columns):
        missing = sorted(required_columns - set(stage_manifest.columns))
        raise StabilityAuditError(
            "stage manifest is incomplete; missing {}".format(missing)
        )
    mode_count = int(expected_mode_count)
    if mode_count <= 0:
        raise StabilityAuditError("configured mode count must be positive")
    manifest = stage_manifest.copy()
    for column in ("stage_index", "retained_mode_count"):
        numeric = pd.to_numeric(manifest[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise StabilityAuditError(
                "stage manifest column {} must contain finite integers".format(column)
            )
        manifest[column] = numeric.astype(int)
    expected_stages = np.arange(mode_count + 1, dtype=int)
    actual_stages = manifest["stage_index"].to_numpy(dtype=int)
    if not np.array_equal(actual_stages, expected_stages):
        raise StabilityAuditError(
            "stage manifest must contain ordered stages 0..{}".format(mode_count)
        )
    expected_retained = mode_count - expected_stages
    if not np.array_equal(
        manifest["retained_mode_count"].to_numpy(dtype=int), expected_retained
    ):
        raise StabilityAuditError(
            "stage manifest retained_mode_count must descend from {} to 0".format(
                mode_count
            )
        )
    if manifest["case_id"].astype(str).duplicated().any():
        raise StabilityAuditError("stage manifest case_id values must be unique")
    return manifest.set_index("stage_index", verify_integrity=True), mode_count


def _windowed_states(
    frame: pd.DataFrame, start_s: float, end_s: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    required = ["time_s"] + list(RESPONSE_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise StabilityAuditError("time-series CSV is missing columns: {}".format(", ".join(missing)))
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise StabilityAuditError("time-series state values contain missing/non-finite data")
    source_time = numeric["time_s"].to_numpy(dtype=float)
    window_time: Optional[np.ndarray] = None
    states: Dict[str, np.ndarray] = {}
    for column in RESPONSE_COLUMNS:
        current_time, values, _, _ = boundary_augmented_window(
            source_time,
            numeric[column].to_numpy(dtype=float),
            start_s,
            end_s,
        )
        if window_time is None:
            window_time = current_time
        elif not np.array_equal(window_time, current_time):
            raise StabilityAuditError("state variables produced inconsistent analysis grids")
        states[column] = values
    assert window_time is not None
    return window_time, states


def _terminal_block_growth_details(
    time_s: np.ndarray,
    normalized_states: Mapping[str, np.ndarray],
    end_s: float,
) -> List[Dict[str, Any]]:
    first_start = end_s - GROWTH_BLOCK_COUNT * GROWTH_BLOCK_DURATION_S
    details: List[Dict[str, Any]] = []
    for quantity, values in normalized_states.items():
        peaks = []
        for index in range(GROWTH_BLOCK_COUNT):
            block_start = first_start + index * GROWTH_BLOCK_DURATION_S
            block_end = block_start + GROWTH_BLOCK_DURATION_S
            _, block_values, _, _ = boundary_augmented_window(
                time_s, values, block_start, block_end
            )
            peaks.append(float(np.max(np.abs(block_values))))
        monotonically_increasing = all(
            peaks[index + 1] > peaks[index] + 1.0e-12
            for index in range(len(peaks) - 1)
        )
        ratio = peaks[-1] / peaks[0] if peaks[0] > 1.0e-12 else float("nan")
        advisory = bool(
            monotonically_increasing
            and math.isfinite(ratio)
            and ratio >= GROWTH_RATIO_ADVISORY
            and peaks[-1] >= GROWTH_MINIMUM_FINAL_PEAK
        )
        details.append(
            {
                "quantity": quantity,
                "block_1_peak_limit_fraction": peaks[0],
                "block_2_peak_limit_fraction": peaks[1],
                "block_3_peak_limit_fraction": peaks[2],
                "block_4_peak_limit_fraction": peaks[3],
                "last_to_first_peak_ratio": ratio,
                "four_blocks_strictly_increasing": monotonically_increasing,
                "terminal_growth_advisory": advisory,
            }
        )
    return details


def analyze_case(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    limits: Mapping[str, float],
    start_s: float,
    end_s: float,
) -> Dict[str, Any]:
    """Return hard-limit and terminal-growth diagnostics for one case."""

    limit_lookup = {
        "heave_m": float(limits["maximum_abs_heave_m"]),
        "heave_velocity_mps": float(limits["maximum_abs_heave_speed_mps"]),
        "roll_deg": float(limits["maximum_abs_attitude_deg"]),
        "pitch_deg": float(limits["maximum_abs_attitude_deg"]),
    }
    if not all(math.isfinite(value) and value > 0.0 for value in limit_lookup.values()):
        raise StabilityAuditError("all rigid-body safety limits must be finite and positive")

    full_required = list(RESPONSE_COLUMNS) + [
        "lambda_total_uniform",
        "induced_velocity_min_mps",
        "cl_saturation_count",
    ] + list(FORBIDDEN_VALIDITY_COLUMNS)
    missing_full = [column for column in full_required if column not in frame.columns]
    if missing_full:
        raise StabilityAuditError(
            "time-series validity audit is missing columns: {}".format(", ".join(missing_full))
        )
    full_numeric = frame[full_required].apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(full_numeric.to_numpy(dtype=float))):
        raise StabilityAuditError("full time-series state values contain missing/non-finite data")
    full_maximum_by_quantity = {
        column: float(np.max(np.abs(full_numeric[column].to_numpy(dtype=float))))
        / limit_lookup[column]
        for column in BODY_STATE_COLUMNS
    }

    time_s, states = _windowed_states(frame, start_s, end_s)

    utilization_series = {
        column: np.abs(states[column]) / limit_lookup[column] for column in BODY_STATE_COLUMNS
    }
    analysis_maximum_by_quantity = {
        column: float(np.max(values)) for column, values in utilization_series.items()
    }
    diagnostics = metadata.get("dynamics_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise StabilityAuditError("dynamics_diagnostics must be an object")
    inflow_metadata = metadata.get("inflow_metadata", {})
    if not isinstance(inflow_metadata, Mapping):
        raise StabilityAuditError("inflow_metadata must be an object")
    reference_lambda = float(inflow_metadata.get("reference_lambda", float("nan")))
    if not math.isfinite(reference_lambda) or reference_lambda <= 0.0:
        raise StabilityAuditError("reference_lambda is missing or invalid")
    delta_lambda_limit_fraction = float(
        inflow_metadata.get("maximum_delta_lambda_fraction", 0.75)
    )
    delta_mu_limit_fraction = float(
        inflow_metadata.get("maximum_delta_mu_z_fraction", 1.0)
    )
    if not (
        math.isfinite(delta_lambda_limit_fraction)
        and delta_lambda_limit_fraction > 0.0
        and math.isfinite(delta_mu_limit_fraction)
        and delta_mu_limit_fraction > 0.0
    ):
        raise StabilityAuditError("dynamic-inflow hard-limit fractions are invalid")
    delta_lambda_limit = delta_lambda_limit_fraction * reference_lambda
    delta_mu_limit = delta_mu_limit_fraction * reference_lambda

    hard_failures = int(diagnostics.get("hard_failures", -1))
    internal_attitude = float(diagnostics.get("maximum_abs_attitude_deg", float("nan")))
    internal_delta_lambda = float(
        diagnostics.get("maximum_abs_delta_inflow_ratio", float("nan"))
    )
    internal_delta_mu = float(diagnostics.get("maximum_abs_delta_mu_z", float("nan")))
    if (
        hard_failures < 0
        or not math.isfinite(internal_attitude)
        or not math.isfinite(internal_delta_lambda)
        or not math.isfinite(internal_delta_mu)
    ):
        raise StabilityAuditError("dynamics diagnostics are incomplete")
    full_maximum_by_quantity["internal_attitude"] = internal_attitude / limit_lookup["roll_deg"]
    full_maximum_by_quantity["delta_lambda_internal"] = (
        internal_delta_lambda / delta_lambda_limit
    )
    full_maximum_by_quantity["delta_mu_z_internal"] = internal_delta_mu / delta_mu_limit
    limiting_quantity = max(full_maximum_by_quantity, key=full_maximum_by_quantity.get)
    maximum_utilization = float(full_maximum_by_quantity[limiting_quantity])

    normalized_states = dict(utilization_series)
    for column in INFLOW_STATE_COLUMNS[:3]:
        normalized_states[column] = np.abs(states[column]) / delta_lambda_limit
    normalized_states["delta_mu_z"] = np.abs(states["delta_mu_z"]) / delta_mu_limit
    growth_details = _terminal_block_growth_details(time_s, normalized_states, end_s)
    warning_quantities = [
        row["quantity"] for row in growth_details if row["terminal_growth_advisory"]
    ]
    growth_advisory = bool(warning_quantities)

    forbidden_count = int(
        sum(float(full_numeric[column].sum()) for column in FORBIDDEN_VALIDITY_COLUMNS)
    )
    minimum_uniform_lambda = float(full_numeric["lambda_total_uniform"].min())
    minimum_local_induced_velocity = float(full_numeric["induced_velocity_min_mps"].min())
    model_validity_failure = bool(
        forbidden_count > 0
        or minimum_uniform_lambda <= 0.0
        or minimum_local_induced_velocity <= 0.0
    )
    hard_limit_exceeded = bool(maximum_utilization > 1.0 + 1.0e-9)
    observed_instability = bool(hard_failures > 0 or hard_limit_exceeded)
    classification = (
        "model_domain_invalid_with_hard_instability"
        if model_validity_failure and observed_instability
        else "model_domain_invalid"
        if model_validity_failure
        else "observed_hard_instability"
        if observed_instability
        else "bounded_with_terminal_growth_advisory"
        if growth_advisory
        else "bounded_no_hard_instability"
    )

    result: Dict[str, Any] = {
        "maximum_hard_limit_utilization": maximum_utilization,
        "limiting_quantity": limiting_quantity,
        "maximum_abs_heave_m": float(
            np.max(np.abs(full_numeric["heave_m"].to_numpy(dtype=float)))
        ),
        "maximum_abs_heave_velocity_mps": float(
            np.max(np.abs(full_numeric["heave_velocity_mps"].to_numpy(dtype=float)))
        ),
        "maximum_abs_roll_deg": float(
            np.max(np.abs(full_numeric["roll_deg"].to_numpy(dtype=float)))
        ),
        "maximum_abs_pitch_deg": float(
            np.max(np.abs(full_numeric["pitch_deg"].to_numpy(dtype=float)))
        ),
        "analysis_window_maximum_abs_heave_m": float(np.max(np.abs(states["heave_m"]))),
        "analysis_window_maximum_abs_heave_velocity_mps": float(
            np.max(np.abs(states["heave_velocity_mps"]))
        ),
        "analysis_window_maximum_abs_roll_deg": float(np.max(np.abs(states["roll_deg"]))),
        "analysis_window_maximum_abs_pitch_deg": float(np.max(np.abs(states["pitch_deg"]))),
        "analysis_window_maximum_body_state_limit_utilization": float(
            max(analysis_maximum_by_quantity.values())
        ),
        "internal_maximum_abs_attitude_deg": internal_attitude,
        "reference_lambda": reference_lambda,
        "delta_lambda_limit_fraction_of_reference_lambda": delta_lambda_limit_fraction,
        "delta_mu_z_limit_fraction_of_reference_lambda": delta_mu_limit_fraction,
        "delta_lambda_hard_limit": delta_lambda_limit,
        "delta_mu_z_hard_limit": delta_mu_limit,
        "internal_maximum_abs_delta_lambda": internal_delta_lambda,
        "internal_maximum_abs_delta_mu_z": internal_delta_mu,
        "metadata_hard_failure_count": hard_failures,
        "hard_limit_exceeded": hard_limit_exceeded,
        "model_validity_failure": model_validity_failure,
        "hard_or_model_validity_failure": bool(
            observed_instability or model_validity_failure
        ),
        "forbidden_aerodynamic_validity_count": forbidden_count,
        "minimum_uniform_total_lambda": minimum_uniform_lambda,
        "minimum_local_induced_velocity_mps": minimum_local_induced_velocity,
        "cl_static_limit_saturation_count": int(full_numeric["cl_saturation_count"].sum()),
        "observed_hard_instability": observed_instability,
        "terminal_growth_warning_quantities": ";".join(warning_quantities),
        "terminal_growth_warning_quantity_count": len(warning_quantities),
        "terminal_growth_advisory": growth_advisory,
        "classification": classification,
        "_terminal_growth_details": growth_details,
    }
    return result


def summarize_rotor(stage_rows: pd.DataFrame) -> Dict[str, Any]:
    ordered = stage_rows.sort_values("stage_index")
    mode_count = int(
        pd.to_numeric(ordered["retained_mode_count"], errors="raise").max()
    )
    unstable = ordered[ordered["observed_hard_instability"].astype(bool)]
    growth = ordered[ordered["terminal_growth_advisory"].astype(bool)]
    worst = ordered.iloc[
        int(np.argmax(ordered["maximum_hard_limit_utilization"].to_numpy(dtype=float)))
    ]
    sustained_cutoff: Any = ""
    sustained_tail_length = 0
    for candidate in ordered["stage_index"].to_numpy(dtype=int):
        tail = ordered[ordered["stage_index"] >= candidate]
        if (
            len(tail) >= MINIMUM_SUSTAINED_UNSTABLE_STAGE_COUNT
            and bool(tail["observed_hard_instability"].astype(bool).all())
        ):
            sustained_cutoff = int(candidate)
            sustained_tail_length = int(len(tail))
            break
    if unstable.empty:
        first_stage: Any = ""
        first_deleted: Any = ""
        first_retained: Any = ""
        last_stable = int(ordered["stage_index"].max())
        width = max(2, len(str(mode_count)))
        status = "not_observed_through_stage_{:0{width}d}".format(
            mode_count, width=width
        )
    else:
        first = unstable.iloc[0]
        first_stage = int(first["stage_index"])
        first_deleted = first["deleted_mode_this_stage"]
        first_retained = int(first["retained_mode_count"])
        last_stable = first_stage - 1
        status = (
            "sustained_from_stage_{:02d}".format(int(sustained_cutoff))
            if sustained_cutoff != ""
            else "hard_event_observed_but_no_sustained_cutoff"
        )
    first_growth = "" if growth.empty else int(growth.iloc[0]["stage_index"])
    return {
        "rotor_name": str(ordered.iloc[0]["rotor_name"]),
        "rotor_diameter_m": float(ordered.iloc[0]["rotor_diameter_m"]),
        "first_observed_hard_instability_stage": first_stage,
        "first_unstable_deleted_mode": first_deleted,
        "first_unstable_retained_mode_count": first_retained,
        "last_stable_stage_before_hard_instability": last_stable,
        "hard_instability_cutoff_status": status,
        "sustained_hard_instability_cutoff_stage": sustained_cutoff,
        "sustained_hard_instability_tail_length": sustained_tail_length,
        "minimum_sustained_unstable_stage_count": MINIMUM_SUSTAINED_UNSTABLE_STAGE_COUNT,
        "worst_stage_index": int(worst["stage_index"]),
        "worst_stage_deleted_mode": worst["deleted_mode_this_stage"],
        "worst_stage_retained_mode_count": int(worst["retained_mode_count"]),
        "worst_maximum_hard_limit_utilization": float(
            worst["maximum_hard_limit_utilization"]
        ),
        "worst_limiting_quantity": str(worst["limiting_quantity"]),
        "first_terminal_growth_advisory_stage": first_growth,
        "terminal_growth_advisory_stage_count": int(len(growth)),
        "terminal_growth_advisory_quantity_count": int(
            pd.to_numeric(growth["terminal_growth_warning_quantity_count"], errors="coerce").sum()
        ),
        "controller_interpretation": "finite-window idealized closed-loop hover-hold response",
    }


def run_stability_audit(config_path: Path) -> Dict[str, Any]:
    context = load_study_context(config_path, allow_provisional=True)
    body = context.config["body_response"]
    start_s = float(body["analysis_window_start_s"])
    end_s = float(body["analysis_window_end_s"])
    limits = {
        "maximum_abs_attitude_deg": float(body["maximum_abs_attitude_deg"]),
        "maximum_abs_heave_m": float(body["maximum_abs_heave_m"]),
        "maximum_abs_heave_speed_mps": float(body["maximum_abs_heave_speed_mps"]),
    }
    rotor_config = _read_json(context.rotor_config_path)
    rotors = rotor_config.get("rotors")
    if not isinstance(rotors, list) or len(rotors) != 7:
        raise StabilityAuditError("exactly seven rotor configurations are required")
    coupling_run = _read_json(context.output_dir / "00_provenance" / "coupling_run.json")
    expected_signature = str(coupling_run.get("run_signature_sha256", ""))
    if not expected_signature:
        raise StabilityAuditError("production coupling run has no signature")
    stage_manifest_path = context.output_dir / "00_provenance" / "stage_manifest.csv"
    if not stage_manifest_path.is_file():
        raise StabilityAuditError("stage manifest is missing: {}".format(stage_manifest_path))
    stage_manifest = pd.read_csv(str(stage_manifest_path), encoding="utf-8-sig")
    stage_manifest, mode_count = _validated_stage_manifest(
        stage_manifest, context.mode_count
    )

    rows: List[Dict[str, Any]] = []
    growth_rows: List[Dict[str, Any]] = []
    signatures = set()
    for rotor in rotors:
        rotor_name = str(rotor["name"])
        diameter = float(rotor["diameter_m"])
        for stage, manifest_row in stage_manifest.iterrows():
            stage = int(stage)
            retained = int(manifest_row["retained_mode_count"])
            case_id = str(manifest_row["case_id"])
            stem = "{}__{}".format(_slug(rotor_name), case_id)
            csv_path = context.output_dir / "04_timeseries_csv" / (stem + ".csv")
            metadata_path = context.output_dir / "04_timeseries_csv" / (stem + ".metadata.json")
            if not csv_path.is_file() or not metadata_path.is_file():
                raise StabilityAuditError("missing production case pair: {}".format(stem))
            frame = pd.read_csv(str(csv_path), encoding="utf-8-sig")
            metadata = _read_json(metadata_path)
            signatures.add(str(metadata.get("run_signature_sha256", "")))
            case_definition = metadata.get("case_definition", {})
            if not isinstance(case_definition, Mapping):
                raise StabilityAuditError("case_definition must be an object: {}".format(stem))
            identity_columns = [
                "rotor_name",
                "rotor_diameter_m",
                "case_id",
                "stage_index",
                "retained_mode_count",
                "cutoff_frequency_hz",
                "cutoff_St",
            ]
            if not set(identity_columns).issubset(frame.columns):
                raise StabilityAuditError("case identity columns are incomplete: {}".format(stem))
            identity = frame[identity_columns].drop_duplicates()
            if len(identity) != 1:
                raise StabilityAuditError("case identity changes within time series: {}".format(stem))
            identity_row = identity.iloc[0]

            def optional_float_matches(left: Any, right: Any) -> bool:
                left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
                right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
                if pd.isna(left_value) and pd.isna(right_value):
                    return True
                return bool(
                    math.isclose(
                        float(left_value),
                        float(right_value),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )

            identity_valid = bool(
                str(identity_row["rotor_name"]) == rotor_name
                and math.isclose(
                    float(identity_row["rotor_diameter_m"]), diameter, rel_tol=0.0, abs_tol=1.0e-12
                )
                and str(identity_row["case_id"]) == case_id
                and int(identity_row["stage_index"]) == stage
                and int(identity_row["retained_mode_count"]) == retained
                and str(case_definition.get("case_id", "")) == case_id
                and int(case_definition.get("stage_index", -1)) == stage
                and int(case_definition.get("retained_mode_count", -1)) == retained
                and str(manifest_row["case_id"]) == case_id
                and int(manifest_row["retained_mode_count"]) == retained
                and optional_float_matches(
                    identity_row["cutoff_frequency_hz"], manifest_row["cutoff_frequency_hz"]
                )
                and optional_float_matches(identity_row["cutoff_St"], manifest_row["cutoff_St"])
            )
            hover_hold_enabled = bool(
                metadata.get("hover_hold_controller", {}).get("enabled", False)
            )
            if not identity_valid or not hover_hold_enabled:
                raise StabilityAuditError(
                    "case identity/controller contract failed: {}".format(stem)
                )
            result = analyze_case(frame, metadata, limits, start_s, end_s)
            growth_details = result.pop("_terminal_growth_details")
            result.update(
                {
                    "rotor_name": rotor_name,
                    "rotor_diameter_m": diameter,
                    "case_id": case_id,
                    "stage_index": stage,
                    "deleted_mode_this_stage": (
                        ""
                        if pd.isna(manifest_row["deleted_mode_this_stage"])
                        else int(manifest_row["deleted_mode_this_stage"])
                    ),
                    "deleted_frequency_hz_this_stage": manifest_row[
                        "deleted_frequency_hz_this_stage"
                    ],
                    "deleted_St_this_stage": manifest_row["deleted_St_this_stage"],
                    "retained_mode_count": retained,
                    "cutoff_frequency_hz": manifest_row["cutoff_frequency_hz"],
                    "cutoff_St": manifest_row["cutoff_St"],
                    "run_signature_sha256": str(metadata.get("run_signature_sha256", "")),
                    "hover_hold_enabled": hover_hold_enabled,
                    "case_identity_validated": identity_valid,
                }
            )
            rows.append(result)
            for detail in growth_details:
                growth_rows.append(
                    {
                        "rotor_name": rotor_name,
                        "rotor_diameter_m": diameter,
                        "case_id": case_id,
                        "stage_index": stage,
                        "deleted_mode_this_stage": (
                            ""
                            if pd.isna(manifest_row["deleted_mode_this_stage"])
                            else int(manifest_row["deleted_mode_this_stage"])
                        ),
                        "deleted_frequency_hz_this_stage": manifest_row[
                            "deleted_frequency_hz_this_stage"
                        ],
                        "deleted_St_this_stage": manifest_row["deleted_St_this_stage"],
                        "retained_mode_count": retained,
                        "cutoff_frequency_hz": manifest_row["cutoff_frequency_hz"],
                        "cutoff_St": manifest_row["cutoff_St"],
                        **detail,
                    }
                )
    if signatures != {expected_signature}:
        raise StabilityAuditError("stage outputs do not share the production run signature")

    stage_table = pd.DataFrame(rows).sort_values(["rotor_diameter_m", "stage_index"])
    summaries = [summarize_rotor(group) for _, group in stage_table.groupby("rotor_name", sort=False)]
    summary_table = pd.DataFrame(summaries).sort_values("rotor_diameter_m")
    qa_directory = context.output_dir / "09_qa"
    stage_path = qa_directory / "modal_deletion_stability_by_stage.csv"
    summary_path = qa_directory / "modal_deletion_stability_summary.csv"
    growth_path = qa_directory / "modal_deletion_terminal_growth_by_quantity.csv"
    _write_csv(stage_path, stage_table)
    _write_csv(summary_path, summary_table)
    _write_csv(growth_path, pd.DataFrame(growth_rows))

    hard_count = int(stage_table["observed_hard_instability"].astype(bool).sum())
    model_validity_failure_count = int(
        stage_table["model_validity_failure"].astype(bool).sum()
    )
    growth_count = int(stage_table["terminal_growth_advisory"].astype(bool).sum())
    growth_quantity_count = int(
        pd.to_numeric(
            stage_table["terminal_growth_warning_quantity_count"], errors="coerce"
        ).sum()
    )
    frequency = pd.read_csv(
        str(context.output_dir / "02_rvmd" / "frequencies.csv"), encoding="utf-8-sig"
    )
    if "mode" not in frequency.columns:
        raise StabilityAuditError("RVMD frequency table has no mode column")
    frequency_modes = pd.to_numeric(frequency["mode"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.array_equal(frequency_modes, np.arange(1, mode_count + 1, dtype=float)):
        raise StabilityAuditError(
            "RVMD frequency table must contain modes 1..{}".format(mode_count)
        )
    lowest_frequency_hz = float(pd.to_numeric(frequency["frequency_hz"]).min())
    lowest_period_s = 1.0 / lowest_frequency_hz
    delta_lambda_fractions = sorted(
        set(
            pd.to_numeric(
                stage_table["delta_lambda_limit_fraction_of_reference_lambda"],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
    )
    delta_mu_fractions = sorted(
        set(
            pd.to_numeric(
                stage_table["delta_mu_z_limit_fraction_of_reference_lambda"],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
    )
    if len(delta_lambda_fractions) != 1 or len(delta_mu_fractions) != 1:
        raise StabilityAuditError("dynamic-inflow limits differ across audited cases")
    terminal_blocks = [
        [
            end_s - GROWTH_BLOCK_COUNT * GROWTH_BLOCK_DURATION_S
            + index * GROWTH_BLOCK_DURATION_S,
            end_s - GROWTH_BLOCK_COUNT * GROWTH_BLOCK_DURATION_S
            + (index + 1) * GROWTH_BLOCK_DURATION_S,
        ]
        for index in range(GROWTH_BLOCK_COUNT)
    ]
    audit = {
        "schema_version": 1,
        "status": "pass",
        "status_meaning": "audit completed and all input/output contracts passed",
        "run_signature_sha256": expected_signature,
        "rotor_count": 7,
        "mode_count": mode_count,
        "final_stage_index": mode_count,
        "stage_count_per_rotor": mode_count + 1,
        "case_count": int(len(stage_table)),
        "analysis_window_s": [start_s, end_s],
        "controller_scope": "idealized generalized-force PD hover hold enabled in every audited case",
        "formal_observed_instability_definition": (
            "metadata hard failure or configured rigid-body/dynamic-inflow hard-limit crossing"
        ),
        "model_domain_validity_definition": (
            "aerodynamic validity counter or non-positive uniform/local induced flow; "
            "reported separately and not labelled as dynamic instability"
        ),
        "configured_limits": limits,
        "dynamic_inflow_limits": {
            "maximum_abs_delta_lambda_fraction_of_reference_lambda": delta_lambda_fractions[0],
            "maximum_abs_delta_mu_z_fraction_of_reference_lambda": delta_mu_fractions[0],
            "uniform_and_local_induced_flow_must_remain_positive": True,
        },
        "minimum_consecutive_tail_stages_for_sustained_cutoff": (
            MINIMUM_SUSTAINED_UNSTABLE_STAGE_COUNT
        ),
        "terminal_growth_advisory_definition": {
            "last_four_blocks_s": terminal_blocks,
            "all_four_peak_envelopes_must_increase": True,
            "last_to_first_peak_ratio_minimum": GROWTH_RATIO_ADVISORY,
            "final_peak_limit_fraction_minimum": GROWTH_MINIMUM_FINAL_PEAK,
            "interpretation": "advisory only; not an asymptotic or Lyapunov-stability claim",
        },
        "observed_hard_instability_case_count": hard_count,
        "model_domain_invalid_case_count": model_validity_failure_count,
        "terminal_growth_advisory_case_count": growth_count,
        "terminal_growth_advisory_quantity_count": growth_quantity_count,
        "all_rotors_bounded_through_final_stage": bool(
            hard_count == 0 and model_validity_failure_count == 0
        ),
        "finite_window_limitation": (
            "A bounded 1-11 s forced closed-loop response cannot prove asymptotic or open-loop stability."
        ),
        "lowest_retained_mode_frequency_hz": lowest_frequency_hz,
        "lowest_retained_mode_period_s": lowest_period_s,
        "analysis_duration_fraction_of_lowest_mode_period": (end_s - start_s) / lowest_period_s,
        "outputs": {
            "by_stage_csv": str(stage_path),
            "summary_csv": str(summary_path),
            "terminal_growth_by_quantity_csv": str(growth_path),
        },
    }
    if mode_count == 30:
        # Backward-compatible alias for the original K=30 result package.
        audit["all_rotors_bounded_through_stage_30"] = audit[
            "all_rotors_bounded_through_final_stage"
        ]
    _write_json(qa_directory / "modal_deletion_stability_audit.json", audit)
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    args = parser.parse_args(argv)
    result = run_stability_audit(Path(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "_validated_stage_manifest",
    "analyze_case",
    "run_stability_audit",
    "summarize_rotor",
]
