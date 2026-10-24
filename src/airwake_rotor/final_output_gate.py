#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Strict read-only production gate for the NATO-GD/RVMD result package.

This audit separates numerical/artifact validity from scientific claim scope.
A result package may pass all computational checks while remaining provisional
because an input assumption is unconfirmed, and an energy-selected calculation
may still fail the stronger claim that it closes the original-CFD response.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .nato_gd_study import load_study_context
from .rvmd_nato_coupling_pipeline import (
    _artifact_signature,
    _validate_full_space_energy_gate,
)


CORE_METRICS = ("lift_N", "engine_power_W", "roll_deg", "pitch_deg", "heave_m")
TIME_WEIGHTING_METHOD = "boundary_interpolated_trapezoidal_time_weighting_v1"
STRONG_CLOSURE_R_MIN = 0.90
STRONG_CLOSURE_NRMSE_MAX = 0.30
TREND_CLOSURE_R_MIN = 0.80
TREND_CLOSURE_NRMSE_MAX = 0.60


class FinalGateError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise FinalGateError("JSON root must be an object: {}".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "rotor"


def _expected_time_grid(end_s: float, dt_s: float) -> np.ndarray:
    count = int(math.floor(end_s / dt_s + 1.0e-12))
    values = np.arange(count + 1, dtype=float) * dt_s
    if not math.isclose(float(values[-1]), end_s, rel_tol=0.0, abs_tol=1.0e-12):
        values = np.concatenate((values, [end_s]))
    else:
        values[-1] = end_s
    return values


def _current_run_signature(context: Any) -> str:
    rvmd_directory = context.output_dir / "02_rvmd"
    pipeline = Path(__file__).with_name("rvmd_nato_coupling_pipeline.py")
    return _artifact_signature(
        [
            context.config_path,
            context.rotor_config_path,
            context.dynamic_overrides_path,
            pipeline,
            pipeline.with_name("nonlinear_blade_element.py"),
            pipeline.with_name("nonlinear_rotorcraft_dynamics.py"),
            pipeline.with_name("rvmd_3d_flow.py"),
            context.cache_dir / "metadata.json",
            context.cache_dir / "manifest.json",
            rvmd_directory / "metadata.json",
            rvmd_directory / "temporal_model.npz",
            rvmd_directory / "mean_field_physical.npy",
            rvmd_directory / "spatial_modes_physical.npy",
        ]
    )


def _validate_gif(path: Path, expected_frames: int) -> Dict[str, int]:
    with Image.open(str(path)) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        width, height = image.size
    if frame_count != expected_frames or width <= 0 or height <= 0:
        raise FinalGateError("invalid GIF dimensions/frame count: {}".format(path))
    return {"frames": frame_count, "pixel_width": int(width), "pixel_height": int(height)}


def run_final_gate(config_path: Path) -> Dict[str, Any]:
    context = load_study_context(config_path, allow_provisional=True)
    config = context.config
    output = context.output_dir
    rotor_config = _read_json(context.rotor_config_path)
    rotors = rotor_config.get("rotors")
    if not isinstance(rotors, list) or len(rotors) != 7:
        raise FinalGateError("expected seven configured rotorcraft")
    rotor_names = [str(item["name"]) for item in rotors]
    rotor_diameter = {str(item["name"]): float(item["diameter_m"]) for item in rotors}
    rotor_radius = {str(item["name"]): 0.5 * float(item["diameter_m"]) for item in rotors}
    mode_count = int(context.mode_count)
    index_width = max(2, len(str(mode_count)))
    body = config["body_response"]
    end_s = float(body["analysis_window_end_s"])
    dt_s = float(context.dt_s)
    maximum_step_deg = float(config["rotor_model"]["maximum_azimuth_step_deg"])
    expected_time = _expected_time_grid(end_s, dt_s)
    coupling_run = _read_json(output / "00_provenance" / "coupling_run.json")
    declared_signature = str(coupling_run.get("run_signature_sha256", ""))
    current_signature = _current_run_signature(context)
    checks: List[Dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    record(
        "current_input_signature_matches_production_run",
        bool(declared_signature and declared_signature == current_signature),
        {"declared": declared_signature, "current": current_signature},
    )
    rvmd_metadata = _read_json(output / "02_rvmd" / "metadata.json")
    try:
        energy_gate = _validate_full_space_energy_gate(
            rvmd_metadata, context.full_space_energy_target
        )
        energy_gate_passed = bool(energy_gate["passed"])
    except ValueError as exception:
        energy_gate = {"passed": False, "error": str(exception)}
        energy_gate_passed = False
    record(
        "mean_removed_full_space_fluctuation_energy_target_passed",
        energy_gate_passed,
        energy_gate,
    )

    time_series_dir = output / "04_timeseries_csv"
    sensitivity_requested = bool(coupling_run.get("modal_sensitivity_requested", False))
    sensitivity_modes = sorted(
        set(int(value) for value in coupling_run.get("sensitivity_mode_indices_one_based", []))
    )
    sensitivity_complete = sensitivity_requested and sensitivity_modes == list(
        range(1, mode_count + 1)
    )
    sensitivity_width = max(3, len(str(mode_count)))
    expected_stems = []
    for rotor in rotor_names:
        expected_stems.extend(
            "{}__stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
                _slug(rotor), stage=stage, retained=mode_count - stage, width=index_width
            )
            for stage in range(mode_count + 1)
        )
        expected_stems.append("{}__original_cfd".format(_slug(rotor)))
        if sensitivity_complete:
            for mode in sensitivity_modes:
                tag = "M{:0{width}d}".format(mode, width=sensitivity_width)
                expected_stems.append(
                    "{}__sensitivity_single_{}".format(_slug(rotor), tag)
                )
                expected_stems.append(
                    "{}__sensitivity_loo_{}".format(_slug(rotor), tag)
                )
    actual_csv = {path.stem for path in time_series_dir.glob("*.csv")}
    actual_metadata = {
        path.name[: -len(".metadata.json")]
        for path in time_series_dir.glob("*.metadata.json")
    }
    record(
        "all_dynamic_K_and_modal_sensitivity_atomic_case_pairs_present",
        sensitivity_complete
        and actual_csv == set(expected_stems)
        and actual_metadata == set(expected_stems),
        {
            "modal_sensitivity_requested": sensitivity_requested,
            "all_modes_requested": sensitivity_complete,
            "expected": len(expected_stems),
            "csv_count": len(actual_csv),
            "metadata_count": len(actual_metadata),
            "missing_csv": sorted(set(expected_stems) - actual_csv),
            "missing_metadata": sorted(set(expected_stems) - actual_metadata),
            "unexpected_csv": sorted(actual_csv - set(expected_stems)),
            "unexpected_metadata": sorted(actual_metadata - set(expected_stems)),
        },
    )

    signatures = set()
    controller_contracts: Dict[str, set] = {name: set() for name in rotor_names}
    inflow_initials: Dict[str, set] = {name: set() for name in rotor_names}
    hard_failure_sum = 0
    internal_max_attitude = 0.0
    internal_max_delta_lambda = 0.0
    internal_max_delta_mu_z = 0.0
    maximum_state = {"heave_m": 0.0, "heave_velocity_mps": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0}
    maximum_control = {
        "hover_hold_Fz_body_N": 0.0,
        "hover_hold_Mx_body_Nm": 0.0,
        "hover_hold_My_body_Nm": 0.0,
    }
    forbidden_domain_columns = [
        "reverse_flow_count",
        "alpha_domain_violation_count",
        "mach_domain_violation_count",
        "reynolds_domain_violation_count",
    ]
    forbidden_domain_sum = {column: 0 for column in forbidden_domain_columns}
    cl_saturation_sum = 0
    minimum_disk_domain_clearance_m = float("inf")
    common_time_passed = True
    finite_passed = True
    metadata_contract_passed = True
    for rotor in rotor_names:
        for case_index, case_id in enumerate(
            [
                "stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
                    stage=stage, retained=mode_count - stage, width=index_width
                )
                for stage in range(mode_count + 1)
            ]
            + ["original_cfd"]
        ):
            stem = "{}__{}".format(_slug(rotor), case_id)
            csv_path = time_series_dir / (stem + ".csv")
            metadata_path = time_series_dir / (stem + ".metadata.json")
            if not csv_path.is_file() or not metadata_path.is_file():
                common_time_passed = False
                finite_passed = False
                metadata_contract_passed = False
                continue
            metadata = _read_json(metadata_path)
            signatures.add(str(metadata.get("run_signature_sha256", "")))
            diagnostics = metadata.get("dynamics_diagnostics", {})
            hard_failure_sum += int(diagnostics.get("hard_failures", -1))
            internal_max_attitude = max(
                internal_max_attitude, float(diagnostics.get("maximum_abs_attitude_deg", float("inf")))
            )
            internal_max_delta_lambda = max(
                internal_max_delta_lambda,
                float(diagnostics.get("maximum_abs_delta_inflow_ratio", float("inf"))),
            )
            internal_max_delta_mu_z = max(
                internal_max_delta_mu_z,
                float(diagnostics.get("maximum_abs_delta_mu_z", float("inf"))),
            )
            controller_contracts[rotor].add(
                json.dumps(metadata.get("hover_hold_controller"), sort_keys=True, separators=(",", ":"))
            )
            inflow_initials[rotor].add(
                tuple(float(value) for value in metadata.get("initial_state_delta_inflow", []))
            )
            metadata_contract_passed &= (
                math.isclose(float(metadata.get("end_time_s", float("nan"))), end_s, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(float(metadata.get("output_interval_s", float("nan"))), dt_s, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(
                    float(metadata.get("maximum_azimuth_step_deg", float("nan"))),
                    maximum_step_deg,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and int(diagnostics.get("inflow_rhs_calls", 0)) > 0
                and int(diagnostics.get("rigid_body_rhs_calls", 0)) > 0
            )
            frame = pd.read_csv(str(csv_path), encoding="utf-8-sig")
            if len(frame) != len(expected_time):
                common_time_passed = False
            else:
                actual_time = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
                common_time_passed &= bool(
                    np.all(np.isfinite(actual_time))
                    and np.allclose(actual_time, expected_time, rtol=0.0, atol=2.0e-12)
                )
            first_numeric_column = frame.columns.get_loc("time_s")
            numeric = frame.iloc[:, first_numeric_column:].apply(pd.to_numeric, errors="coerce")
            finite_passed &= bool(np.all(np.isfinite(numeric.to_numpy(dtype=float))))
            for column in maximum_state:
                maximum_state[column] = max(
                    maximum_state[column], float(np.max(np.abs(numeric[column].to_numpy(dtype=float))))
                )
            for column in maximum_control:
                maximum_control[column] = max(
                    maximum_control[column], float(np.max(np.abs(numeric[column].to_numpy(dtype=float))))
                )
            for column in forbidden_domain_columns:
                forbidden_domain_sum[column] += int(np.sum(numeric[column].to_numpy(dtype=float)))
            cl_saturation_sum += int(np.sum(numeric["cl_saturation_count"].to_numpy(dtype=float)))
            radius = rotor_radius[rotor]
            hub = numeric[["hub_x_m", "hub_y_m", "hub_z_m"]].to_numpy(dtype=float)
            normal = numeric[["rotor_normal_x", "rotor_normal_y", "rotor_normal_z"]].to_numpy(dtype=float)
            bounds = np.asarray(coupling_run["grid_and_time_audit"]["bounds_xyz_m"], dtype=float)
            disk_extent = radius * np.sqrt(np.maximum(1.0 - normal ** 2, 0.0))
            lower_clearance = hub - disk_extent - bounds[:, 0]
            upper_clearance = bounds[:, 1] - hub - disk_extent
            minimum_disk_domain_clearance_m = min(
                minimum_disk_domain_clearance_m,
                float(np.min(lower_clearance)),
                float(np.min(upper_clearance)),
            )

    sensitivity_metadata_passed = sensitivity_complete
    if sensitivity_complete:
        for rotor in rotor_names:
            for mode in sensitivity_modes:
                tag = "M{:0{width}d}".format(mode, width=sensitivity_width)
                for kind in ("single", "loo"):
                    case_id = "sensitivity_{}_{}".format(kind, tag)
                    stem = "{}__{}".format(_slug(rotor), case_id)
                    metadata_path = time_series_dir / (stem + ".metadata.json")
                    if not metadata_path.is_file():
                        sensitivity_metadata_passed = False
                        continue
                    metadata = _read_json(metadata_path)
                    signatures.add(str(metadata.get("run_signature_sha256", "")))
                    diagnostics = metadata.get("dynamics_diagnostics", {})
                    hard_failure_sum += int(diagnostics.get("hard_failures", -1))
                    saved_case = metadata.get("case_definition", {})
                    expected_kind = "single" if kind == "single" else "leave_one_out"
                    sensitivity_metadata_passed &= bool(
                        isinstance(saved_case, dict)
                        and saved_case.get("case_id") == case_id
                        and saved_case.get("sensitivity_case_kind") == expected_kind
                        and int(saved_case.get("sensitivity_mode_index_one_based", -1))
                        == mode
                        and math.isclose(
                            float(metadata.get("end_time_s", float("nan"))),
                            end_s,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        and math.isclose(
                            float(metadata.get("output_interval_s", float("nan"))),
                            dt_s,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    )
                    controller_contracts[rotor].add(
                        json.dumps(
                            metadata.get("hover_hold_controller"),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    inflow_initials[rotor].add(
                        tuple(
                            float(value)
                            for value in metadata.get("initial_state_delta_inflow", [])
                        )
                    )
    record(
        "all_modal_sensitivity_metadata_matches_case_contract",
        sensitivity_metadata_passed,
        {
            "mode_count": mode_count,
            "expected_sensitivity_cases": len(rotor_names) * 2 * mode_count,
        },
    )
    record("all_cases_share_current_signature", signatures == {current_signature}, sorted(signatures))
    record(
        "common_0_to_11_second_output_grid",
        common_time_passed,
        {"row_count_each": len(expected_time), "first_s": 0.0, "last_s": end_s, "dt_s": dt_s},
    )
    record("all_time_series_numeric_values_finite", finite_passed, None)
    record(
        "per_case_metadata_contract",
        metadata_contract_passed
        and all(len(values) == 1 for values in controller_contracts.values())
        and all(len(values) == 1 for values in inflow_initials.values()),
        {
            "controller_contract_count_per_rotor": {key: len(value) for key, value in controller_contracts.items()},
            "initial_inflow_count_per_rotor": {key: len(value) for key, value in inflow_initials.items()},
        },
    )
    state_passed = (
        maximum_state["heave_m"] <= float(body["maximum_abs_heave_m"])
        and maximum_state["heave_velocity_mps"] <= float(body["maximum_abs_heave_speed_mps"])
        and maximum_state["roll_deg"] <= float(body["maximum_abs_attitude_deg"])
        and maximum_state["pitch_deg"] <= float(body["maximum_abs_attitude_deg"])
        and internal_max_attitude <= float(body["maximum_abs_attitude_deg"])
    )
    record(
        "dynamics_state_and_hard_failure_gates",
        state_passed and hard_failure_sum == 0,
        {
            "sampled_maximum_absolute_states": maximum_state,
            "internal_maximum_abs_attitude_deg": internal_max_attitude,
            "internal_maximum_abs_delta_lambda": internal_max_delta_lambda,
            "internal_maximum_abs_delta_mu_z": internal_max_delta_mu_z,
            "hard_failure_sum": hard_failure_sum,
        },
    )
    record(
        "disk_stays_inside_flow_domain",
        minimum_disk_domain_clearance_m >= 0.0,
        {"minimum_conservative_axis_aligned_disk_clearance_m": minimum_disk_domain_clearance_m},
    )
    record(
        "aerodynamic_domain_violations_zero",
        all(value == 0 for value in forbidden_domain_sum.values()),
        {
            "forbidden_counts": forbidden_domain_sum,
            "cl_static_limit_saturation_count_reported_separately": cl_saturation_sum,
        },
    )

    trim = pd.read_csv(str(output / "03_hover_trim" / "hover_trim_all_rotors.csv"), encoding="utf-8-sig")
    trim_passed = (
        len(trim) == 7
        and set(trim["rotor_name"].astype(str)) == set(rotor_names)
        and bool(trim["mean_flow_trim_maximum_abs_scaled_residual"].astype(float).max() <= 1.0e-8)
        and bool(trim["relative_thrust_residual"].astype(float).abs().max() <= 1.0e-8)
    )
    record("all_seven_mean_flow_hover_trims_close", trim_passed, {"row_count": len(trim)})

    statistics_path = output / "05_statistics" / "statistics_long.csv"
    statistics = pd.read_csv(str(statistics_path), encoding="utf-8-sig")
    expected_statistics_rows = (
        len(rotor_names) * (mode_count + 2) * len(CORE_METRICS) * 5
    )
    statistics_passed = (
        len(statistics) == expected_statistics_rows
        and set(statistics["time_weighting_method"].astype(str)) == {TIME_WEIGHTING_METHOD}
        and len(
            statistics[["rotor_name", "case_id", "metric", "statistic"]]
            .drop_duplicates()
        )
        == expected_statistics_rows
        and np.all(np.isfinite(pd.to_numeric(statistics["value"], errors="coerce")))
    )
    statistics_audit = _read_json(output / "05_statistics" / "time_weighted_statistics_audit.json")
    statistics_passed &= str(statistics_audit.get("run_signature_sha256", "")) == current_signature
    record(
        "strict_time_weighted_statistics_complete",
        statistics_passed,
        {
            "row_count": len(statistics),
            "expected_row_count": expected_statistics_rows,
            "method": TIME_WEIGHTING_METHOD,
        },
    )

    sensitivity_manifest_path = (
        output / "05_statistics" / "modal_sensitivity_case_manifest.csv"
    )
    sensitivity_output = output / "05_statistics" / "normalized_modal_sensitivity"
    sensitivity_long_path = sensitivity_output / "mode_sensitivity_long.csv"
    diameter_metrics_path = sensitivity_output / "diameter_frequency_metrics.csv"
    sensitivity_audit_path = sensitivity_output / "normalized_sensitivity_audit.json"
    sensitivity_figure_dir = output / "06_figures" / "normalized_modal_sensitivity"
    sensitivity_figure_audit_path = (
        sensitivity_figure_dir / "normalized_sensitivity_figure_audit.json"
    )
    sensitivity_fit_path = (
        sensitivity_figure_dir / "diameter_frequency_power_law_fits.csv"
    )
    normalized_sensitivity_passed = False
    normalized_relationship_summary: Dict[str, Any] = {}
    normalized_sensitivity_details: Dict[str, Any] = {
        "expected_manifest_rows": len(rotor_names) * (2 * mode_count + 2),
        "expected_spectrum_rows": len(rotor_names) * len(CORE_METRICS) * mode_count,
    }
    if all(
        path.is_file()
        for path in (
            sensitivity_manifest_path,
            sensitivity_long_path,
            diameter_metrics_path,
            sensitivity_audit_path,
            sensitivity_figure_audit_path,
            sensitivity_fit_path,
        )
    ):
        sensitivity_manifest = pd.read_csv(
            str(sensitivity_manifest_path), encoding="utf-8-sig"
        )
        sensitivity_long = pd.read_csv(str(sensitivity_long_path), encoding="utf-8-sig")
        diameter_metrics = pd.read_csv(str(diameter_metrics_path), encoding="utf-8-sig")
        sensitivity_audit = _read_json(sensitivity_audit_path)
        sensitivity_figure_audit = _read_json(sensitivity_figure_audit_path)
        sensitivity_fits = pd.read_csv(
            str(sensitivity_fit_path), encoding="utf-8-sig"
        )
        long_required = {
            "rotor_name",
            "metric",
            "mode_index_one_based",
            "St_reference_length",
            "R_primary",
            "G_primary",
            "sensitivity_L1_primary",
            "response_sensitivity_L1_primary",
            "interaction_index",
        }
        diameter_required = {
            "rotor_name",
            "metric",
            "estimator",
            "St90",
            "StD90",
            "response_St90",
            "response_StD90",
        }
        manifest_counts_ok = bool(
            len(sensitivity_manifest) == len(rotor_names) * (2 * mode_count + 2)
            and set(sensitivity_manifest["case_kind"].astype(str))
            == {"full", "mean", "single", "leave_one_out"}
        )
        long_shape_ok = bool(
            long_required.issubset(sensitivity_long.columns)
            and len(sensitivity_long)
            == len(rotor_names) * len(CORE_METRICS) * mode_count
            and len(
                sensitivity_long[
                    ["rotor_name", "metric", "mode_index_one_based"]
                ].drop_duplicates()
            )
            == len(sensitivity_long)
        )
        diameter_shape_ok = bool(
            diameter_required.issubset(diameter_metrics.columns)
            and len(diameter_metrics)
            == len(rotor_names) * len(CORE_METRICS) * 2
            and set(diameter_metrics["estimator"].astype(str))
            == {"single", "leave_one_out"}
        )
        expected_figure_stems = {
            "normalized_sensitivity_spectra",
            "actual_response_spectra",
            "diameter_vs_St90",
            "diameter_vs_St90_basis_comparison",
            "diameter_vs_StD90",
        }
        figure_files_ok = all(
            (sensitivity_figure_dir / (stem + suffix)).is_file()
            for stem in expected_figure_stems
            for suffix in (".png", ".svg", ".pdf")
        )
        fit_table_ok = bool(
            len(sensitivity_fits) == len(CORE_METRICS) * 2
            and set(sensitivity_fits["metric"].astype(str)) == set(CORE_METRICS)
            and set(sensitivity_fits["spectrum_basis"].astype(str))
            == {"actual_response_contribution_R", "unit_modal_energy_gain_G"}
        )
        if fit_table_ok:
            fit_beta = pd.to_numeric(sensitivity_fits["beta"], errors="coerce")
            normalized_relationship_summary = {
                "fit_row_count": len(sensitivity_fits),
                "positive_beta_metrics_by_basis": {},
                "dual_basis_positive_beta_metrics": [],
            }
            positive_sets: List[set] = []
            for basis, group in sensitivity_fits.assign(beta_numeric=fit_beta).groupby(
                "spectrum_basis"
            ):
                positive = set(
                    group.loc[
                        np.isfinite(group["beta_numeric"])
                        & (group["beta_numeric"] > 0.0),
                        "metric",
                    ].astype(str)
                )
                positive_sets.append(positive)
                normalized_relationship_summary[
                    "positive_beta_metrics_by_basis"
                ][str(basis)] = sorted(positive)
            if positive_sets:
                normalized_relationship_summary[
                    "dual_basis_positive_beta_metrics"
                ] = sorted(set.intersection(*positive_sets))
        numeric_ok = False
        spectra_normalized = False
        if long_shape_ok:
            numeric_columns = [
                "St_reference_length",
                "R_primary",
                "G_primary",
                "sensitivity_L1_primary",
                "response_sensitivity_L1_primary",
                "interaction_index",
            ]
            numeric = sensitivity_long[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            numeric_ok = bool(
                np.all(np.isfinite(numeric.to_numpy(dtype=float)))
                and np.all(numeric.to_numpy(dtype=float) >= 0.0)
            )
            if numeric_ok:
                sums = sensitivity_long.groupby(["rotor_name", "metric"])[
                    ["sensitivity_L1_primary", "response_sensitivity_L1_primary"]
                ].sum()
                sum_values = sums.to_numpy(dtype=float)
                spectra_normalized = bool(
                    np.all(
                        np.isclose(sum_values, 1.0, rtol=0.0, atol=1.0e-10)
                        | np.isclose(sum_values, 0.0, rtol=0.0, atol=1.0e-10)
                    )
                )
        normalized_sensitivity_passed = bool(
            sensitivity_complete
            and manifest_counts_ok
            and long_shape_ok
            and diameter_shape_ok
            and numeric_ok
            and spectra_normalized
            and sensitivity_audit.get("status") == "pass"
            and int(sensitivity_audit.get("mode_count", -1)) == mode_count
            and sensitivity_figure_audit.get("status") == "pass"
            and figure_files_ok
            and fit_table_ok
        )
        normalized_sensitivity_details.update(
            {
                "manifest_rows": len(sensitivity_manifest),
                "spectrum_rows": len(sensitivity_long),
                "diameter_metric_rows": len(diameter_metrics),
                "spectra_normalized": spectra_normalized,
                "normalized_figure_files_complete": figure_files_ok,
                "power_law_fit_rows": len(sensitivity_fits),
            }
        )
    record(
        "normalized_all_St_modal_sensitivity_complete",
        normalized_sensitivity_passed,
        normalized_sensitivity_details,
    )

    qa = _read_json(output / "09_qa" / "numerical_qa_summary.json")
    qa_passed = (
        bool(qa.get("production_time_step_gate_passed", False))
        and bool(qa.get("production_signature_bound", False))
        and str(qa.get("production_run_signature_sha256", "")) == current_signature
        and math.isclose(
            float(qa.get("recommended_production_maximum_azimuth_step_deg", float("nan"))),
            maximum_step_deg,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    record("numerical_and_parameter_QA_bound_to_production_signature", qa_passed, qa)

    stability_audit_path = output / "09_qa" / "modal_deletion_stability_audit.json"
    stability_stage_path = output / "09_qa" / "modal_deletion_stability_by_stage.csv"
    stability_summary_path = output / "09_qa" / "modal_deletion_stability_summary.csv"
    stability_growth_path = output / "09_qa" / "modal_deletion_terminal_growth_by_quantity.csv"
    stability_audit = _read_json(stability_audit_path)
    stability_stage = pd.read_csv(str(stability_stage_path), encoding="utf-8-sig")
    stability_summary = pd.read_csv(str(stability_summary_path), encoding="utf-8-sig")
    stability_required_stage_columns = {
        "rotor_name",
        "stage_index",
        "retained_mode_count",
        "maximum_hard_limit_utilization",
        "observed_hard_instability",
        "model_validity_failure",
        "run_signature_sha256",
        "hover_hold_enabled",
        "case_identity_validated",
    }
    stability_required_summary_columns = {
        "rotor_name",
        "hard_instability_cutoff_status",
        "sustained_hard_instability_cutoff_stage",
        "worst_stage_index",
        "worst_maximum_hard_limit_utilization",
    }
    stability_stage_columns_passed = stability_required_stage_columns.issubset(stability_stage.columns)
    stability_summary_columns_passed = stability_required_summary_columns.issubset(
        stability_summary.columns
    )
    stability_stages_passed = False
    stability_utilization_finite = False
    stability_signature_passed = False
    stability_hover_hold_passed = False
    stability_identity_passed = False
    observed_hard_instability_count = -1
    model_domain_invalid_count = -1
    if stability_stage_columns_passed:
        stability_stages_passed = bool(
            len(stability_stage) == len(rotor_names) * (mode_count + 1)
            and set(stability_stage["rotor_name"].astype(str)) == set(rotor_names)
            and all(
                np.array_equal(
                    np.sort(
                        pd.to_numeric(
                            stability_stage.loc[
                                stability_stage["rotor_name"].astype(str) == rotor,
                                "stage_index",
                            ],
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    ),
                    np.arange(mode_count + 1, dtype=float),
                )
                for rotor in rotor_names
            )
        )
        utilization = pd.to_numeric(
            stability_stage["maximum_hard_limit_utilization"], errors="coerce"
        ).to_numpy(dtype=float)
        stability_utilization_finite = bool(np.all(np.isfinite(utilization)))
        stability_signature_passed = set(
            stability_stage["run_signature_sha256"].astype(str)
        ) == {current_signature}
        stability_hover_hold_passed = bool(
            stability_stage["hover_hold_enabled"].astype(str).str.lower().eq("true").all()
        )
        stability_identity_passed = bool(
            stability_stage["case_identity_validated"]
            .astype(str)
            .str.lower()
            .eq("true")
            .all()
        )
        observed_hard_instability_count = int(
            stability_stage["observed_hard_instability"]
            .astype(str)
            .str.lower()
            .eq("true")
            .sum()
        )
        model_domain_invalid_count = int(
            stability_stage["model_validity_failure"]
            .astype(str)
            .str.lower()
            .eq("true")
            .sum()
        )
    stability_summary_passed = bool(
        stability_summary_columns_passed
        and len(stability_summary) == 7
        and set(stability_summary["rotor_name"].astype(str)) == set(rotor_names)
    )
    stability_passed = bool(
        stability_audit.get("status") == "pass"
        and str(stability_audit.get("run_signature_sha256", "")) == current_signature
        and int(stability_audit.get("case_count", -1))
        == len(rotor_names) * (mode_count + 1)
        and int(stability_audit.get("rotor_count", -1)) == 7
        and int(stability_audit.get("stage_count_per_rotor", -1)) == mode_count + 1
        and stability_stages_passed
        and stability_utilization_finite
        and stability_signature_passed
        and stability_hover_hold_passed
        and stability_identity_passed
        and stability_summary_passed
        and stability_growth_path.is_file()
        and int(stability_audit.get("observed_hard_instability_case_count", -1))
        == observed_hard_instability_count
        and int(stability_audit.get("model_domain_invalid_case_count", -1))
        == model_domain_invalid_count
    )
    record(
        "modal_deletion_stability_audit_complete",
        stability_passed,
        {
            "case_count": len(stability_stage),
            "summary_row_count": len(stability_summary),
            "observed_hard_instability_case_count": observed_hard_instability_count,
            "model_domain_invalid_case_count": model_domain_invalid_count,
            "terminal_growth_advisory_case_count": stability_audit.get(
                "terminal_growth_advisory_case_count"
            ),
            "terminal_growth_advisory_quantity_count": stability_audit.get(
                "terminal_growth_advisory_quantity_count"
            ),
            "analysis_duration_fraction_of_lowest_mode_period": stability_audit.get(
                "analysis_duration_fraction_of_lowest_mode_period"
            ),
        },
    )
    record(
        "all_modal_deletion_cases_inside_model_domain",
        model_domain_invalid_count == 0,
        {"model_domain_invalid_case_count": model_domain_invalid_count},
    )

    figure_manifest = _read_json(output / "06_figures" / "figure_manifest.json")
    figure_groups = figure_manifest.get("figures", {})
    expected_figure_groups = {
        "figure_01",
        "figure_02",
        "figure_03",
        "figure_04",
        "figure_05",
        "figure_06",
        "figure_07",
        "figure_08",
        "figure_S01",
        "figure_S02",
        "figure_S03",
        "figure_S04",
    }
    figure_files = [Path(item) for values in figure_groups.values() for item in values]
    figure_passed = bool(
        set(figure_groups) == expected_figure_groups
        and len(figure_files) == 48
        and all(
            len(figure_groups[key]) == 4
            and {Path(path).suffix.lower() for path in figure_groups[key]}
            == {".svg", ".pdf", ".tiff", ".png"}
            for key in expected_figure_groups
        )
        and all(path.is_file() and path.stat().st_size > 0 for path in figure_files)
    )
    record(
        "twelve_figure_sets_all_four_formats_present",
        figure_passed,
        {"file_count": len(figure_files), "figure_groups": sorted(figure_groups)},
    )

    dominant_path = output / "06_figures" / "source_data" / "dominant_modes.csv"
    dominant = pd.read_csv(str(dominant_path), encoding="utf-8-sig")
    dominant_required_columns = {
        "effect_kind",
        "rotor_name",
        "rotor_diameter_m",
        "metric",
        "dominant_deleted_mode",
        "dominant_frequency_hz",
        "dominant_St",
        "absolute_normalized_adjacent_effect",
        "runner_up_deleted_mode",
        "runner_up_frequency_hz",
        "runner_up_St",
        "runner_up_absolute_normalized_adjacent_effect",
        "dominance_ratio_top_over_runner_up",
        "near_tie_top_over_runner_up_below_1p10",
    }
    dominant_columns_passed = dominant_required_columns.issubset(dominant.columns)
    dominant_mapping_passed = False
    near_tie_count = -1
    if dominant_columns_passed:
        numeric_columns = [
            "rotor_diameter_m",
            "dominant_deleted_mode",
            "dominant_frequency_hz",
            "dominant_St",
            "absolute_normalized_adjacent_effect",
            "runner_up_deleted_mode",
            "runner_up_frequency_hz",
            "runner_up_St",
            "runner_up_absolute_normalized_adjacent_effect",
            "dominance_ratio_top_over_runner_up",
        ]
        dominant_numeric = dominant[numeric_columns].apply(pd.to_numeric, errors="coerce")
        frequency = pd.read_csv(
            str(output / "02_rvmd" / "frequencies.csv"), encoding="utf-8-sig"
        ).set_index("mode")
        frequency_hz_lookup = pd.to_numeric(frequency["frequency_hz"], errors="coerce")
        st_lookup = pd.to_numeric(
            frequency["strouhal_St_f13.716_over_20"], errors="coerce"
        )
        dominant_mode = dominant_numeric["dominant_deleted_mode"].to_numpy(dtype=int)
        runner_mode = dominant_numeric["runner_up_deleted_mode"].to_numpy(dtype=int)
        near_tie = (
            dominant["near_tie_top_over_runner_up_below_1p10"]
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy(dtype=bool)
        )
        ratio = dominant_numeric["dominance_ratio_top_over_runner_up"].to_numpy(dtype=float)
        expected_diameter = dominant["rotor_name"].astype(str).map(rotor_diameter)
        dominant_mapping_passed = bool(
            len(dominant) == 70
            and len(dominant[["effect_kind", "rotor_name", "metric"]].drop_duplicates()) == 70
            and set(dominant["effect_kind"].astype(str)) == {"time_mean", "peak_to_peak"}
            and set(dominant["metric"].astype(str)) == set(CORE_METRICS)
            and set(dominant["rotor_name"].astype(str)) == set(rotor_names)
            and np.all(np.isfinite(dominant_numeric.to_numpy(dtype=float)))
            and np.all((dominant_mode >= 1) & (dominant_mode <= mode_count))
            and np.all((runner_mode >= 1) & (runner_mode <= mode_count))
            and np.allclose(
                dominant_numeric["rotor_diameter_m"].to_numpy(dtype=float),
                expected_diameter.to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                dominant_numeric["dominant_frequency_hz"].to_numpy(dtype=float),
                frequency_hz_lookup.loc[dominant_mode].to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                dominant_numeric["runner_up_frequency_hz"].to_numpy(dtype=float),
                frequency_hz_lookup.loc[runner_mode].to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                dominant_numeric["dominant_St"].to_numpy(dtype=float),
                st_lookup.loc[dominant_mode].to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                dominant_numeric["runner_up_St"].to_numpy(dtype=float),
                st_lookup.loc[runner_mode].to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                dominant_numeric["dominant_St"].to_numpy(dtype=float),
                dominant_numeric["dominant_frequency_hz"].to_numpy(dtype=float)
                * 13.716
                / 20.0,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.all(
                dominant_numeric["absolute_normalized_adjacent_effect"].to_numpy(dtype=float)
                >= dominant_numeric[
                    "runner_up_absolute_normalized_adjacent_effect"
                ].to_numpy(dtype=float)
            )
            and np.array_equal(near_tie, ratio < 1.10)
        )
        near_tie_count = int(np.sum(near_tie))
    record(
        "rotor_size_dominant_St_source_complete",
        dominant_mapping_passed,
        {
            "row_count": len(dominant),
            "near_tie_count_top_over_runner_up_below_1p10": near_tie_count,
            "strouhal_definition": "St=f*13.716/20",
        },
    )

    closure_path = output / "06_figures" / "source_data" / "full_RVMD_vs_original_CFD_closure.csv"
    closure = pd.read_csv(str(closure_path), encoding="utf-8-sig")
    closure_numeric = closure[["pearson_r", "rmse_normalized_by_CFD_fluctuation_rms"]].apply(
        pd.to_numeric, errors="coerce"
    )
    closure_complete = len(closure) == 35 and bool(np.all(np.isfinite(closure_numeric.to_numpy(dtype=float))))
    strong = (
        (closure_numeric["pearson_r"] >= STRONG_CLOSURE_R_MIN)
        & (closure_numeric["rmse_normalized_by_CFD_fluctuation_rms"] <= STRONG_CLOSURE_NRMSE_MAX)
    )
    trend = (
        (closure_numeric["pearson_r"] >= TREND_CLOSURE_R_MIN)
        & (closure_numeric["rmse_normalized_by_CFD_fluctuation_rms"] <= TREND_CLOSURE_NRMSE_MAX)
    )
    full_cfd_claim_allowed = closure_complete and bool(strong.all())
    closure_summary = {
        "complete": closure_complete,
        "project_specific_non_universal_thresholds": {
            "strong": {"minimum_r": STRONG_CLOSURE_R_MIN, "maximum_nrmse": STRONG_CLOSURE_NRMSE_MAX},
            "trend_only": {"minimum_r": TREND_CLOSURE_R_MIN, "maximum_nrmse": TREND_CLOSURE_NRMSE_MAX},
        },
        "strong_count_of_35": int(strong.sum()),
        "trend_or_stronger_count_of_35": int(trend.sum()),
        "minimum_pearson_r": float(closure_numeric["pearson_r"].min()),
        "maximum_normalized_rmse": float(
            closure_numeric["rmse_normalized_by_CFD_fluctuation_rms"].max()
        ),
        "full_CFD_response_attribution_claim_allowed": full_cfd_claim_allowed,
    }
    record("response_closure_table_complete", closure_complete, closure_summary)

    gif_dir = output / "07_flow_gifs_200frames"
    gif_paths = sorted(gif_dir.glob("stage_*_delete_mode_*_retain_*.gif"))
    gif_details = []
    gif_passed = len(gif_paths) == mode_count
    if gif_passed:
        try:
            gif_details = [_validate_gif(path, 200) for path in gif_paths]
        except (OSError, FinalGateError):
            gif_passed = False
    gif_audit = _read_json(gif_dir / "gif_audit.json") if (gif_dir / "gif_audit.json").is_file() else {}
    gif_passed &= (
        int(gif_audit.get("gif_count", -1)) == mode_count
        and int(gif_audit.get("frames_per_gif", -1)) == 200
        and len(gif_audit.get("outputs", [])) == mode_count
    )
    record(
        "dynamic_K_deletion_GIFs_each_have_200_frames",
        gif_passed,
        {"gif_count": len(gif_paths), "validated_dimensions": gif_details[:1]},
    )

    computational_checks = [item for item in checks if item["name"] != "response_closure_table_complete"]
    numerical_artifact_gate_passed = all(bool(item["passed"]) for item in computational_checks)
    provisional_reasons = list(coupling_run.get("study_context", {}).get("provisional_reasons", []))
    release_status = "validated" if numerical_artifact_gate_passed and not provisional_reasons else "provisional"
    result = {
        "schema_version": 1,
        "numerical_artifact_gate_passed": numerical_artifact_gate_passed,
        "release_status": release_status,
        "provisional_reasons": provisional_reasons,
        "run_signature_sha256": current_signature,
        "case_count": len(expected_stems),
        "statistics_row_count": len(statistics),
        "maximum_sampled_state": maximum_state,
        "maximum_hover_hold_generalized_force": maximum_control,
        "minimum_disk_domain_clearance_m": minimum_disk_domain_clearance_m,
        "cl_static_limit_saturation_count": cl_saturation_sum,
        "modal_deletion_stability": {
            "audit_complete": stability_passed,
            "observed_hard_instability_case_count": observed_hard_instability_count,
            "model_domain_invalid_case_count": model_domain_invalid_count,
            "all_rotors_bounded_through_final_stage": stability_audit.get(
                "all_rotors_bounded_through_final_stage",
                stability_audit.get("all_rotors_bounded_through_stage_30"),
            ),
            "terminal_growth_advisory_case_count": stability_audit.get(
                "terminal_growth_advisory_case_count"
            ),
            "terminal_growth_advisory_quantity_count": stability_audit.get(
                "terminal_growth_advisory_quantity_count"
            ),
            "analysis_duration_fraction_of_lowest_mode_period": stability_audit.get(
                "analysis_duration_fraction_of_lowest_mode_period"
            ),
        },
        "dominant_mode_mapping": {
            "row_count": len(dominant),
            "near_tie_count_top_over_runner_up_below_1p10": near_tie_count,
            "strouhal_definition": "St=f*13.716/20",
        },
        "response_closure": closure_summary,
        "rotor_size_frequency_relationship": {
            "normalized_sensitivity_audit_complete": normalized_sensitivity_passed,
            "fit_summary": normalized_relationship_summary,
            "claim_scope": (
                "descriptive_configuration_association_only; diameter is confounded with mass, "
                "inertia, rpm, disk loading, solidity, geometry, and controller parameters; a "
                "similitude-controlled virtual rotor sweep is required for a pure size-causation claim"
            ),
        },
        "permitted_claim_scope": (
            "full_original_CFD_response_attribution"
            if full_cfd_claim_allowed
            else "conditional_marginal_sensitivity_within_the_energy_selected_RVMD_subspace_and_declared_cumulative_deletion_path"
        ),
        "checks": checks,
    }
    _write_json(output / "09_qa" / "final_output_gate.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    args = parser.parse_args(argv)
    result = run_final_gate(Path(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["numerical_artifact_gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["FinalGateError", "run_final_gate"]
