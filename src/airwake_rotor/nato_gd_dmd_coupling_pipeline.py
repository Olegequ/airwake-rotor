#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Couple grouped 3-D NATO-GD DMD modes to seven nonlinear rotor models.

This driver reuses the audited hover-trim, nonlinear blade-element, dynamic
inflow, rigid-body, statistics, resume-signature, and CSV routines from the
RVMD study.  The only changed scientific input is the modal flow provider:
physical DMD groups are reconstructed jointly from their one or two real
quadratures.

For K frequency-sorted DMD groups the run contains:

* K+1 cumulative high-frequency-deletion stages;
* mean, full, K single-mode, and K leave-one-out sensitivity cases;
* the original-CFD baseline when requested.

Known aerodynamic/dynamic/spatial domain failures in non-baseline cases are
recorded and the remaining independent cases continue.  Programming and
configuration errors still fail the run immediately.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import rvmd_nato_coupling_pipeline as coupling
from .dmd_3d_flow import DMD3DFlow, SpatialOutOfBoundsError
from .nato_gd_study import (
    StudyContext,
    build_stage_rows,
    load_study_context,
    validate_grid_and_time,
    write_stage_manifest,
)
from .nonlinear_blade_element import AerodynamicDomainError
from .nonlinear_rotorcraft_dynamics import DynamicsDomainError
from .rvmd_3d_flow import NatoGD3DBaselineFlow


CaseDefinition = coupling.CaseDefinition
CaseResult = coupling.CaseResult
RotorStudyModel = coupling.RotorStudyModel
CouplingConfigurationError = coupling.CouplingConfigurationError
CouplingRunError = coupling.CouplingRunError

_WORKER_DMD_FLOW: Optional[DMD3DFlow] = None
_WORKER_BASELINE_FLOW: Optional[NatoGD3DBaselineFlow] = None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CouplingConfigurationError("JSON root must be an object: {}".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
    return path


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CouplingConfigurationError("{} must be a non-empty path".format(label))
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_dmd_study_context(
    config_path: Path, *, allow_provisional: bool = False
) -> Tuple[StudyContext, Dict[str, Any]]:
    """Load a small DMD overlay on top of the established NATO-GD study."""

    overlay_path = Path(config_path).expanduser().resolve()
    overlay = _read_json(overlay_path)
    if overlay.get("schema_version") != "nato-gd-dmd-coupling-v1":
        raise CouplingConfigurationError("unsupported DMD coupling schema")
    root = overlay_path.parent.parent.resolve()
    base_path = _resolve(root, overlay.get("base_study_config"), "base_study_config")
    base = load_study_context(base_path, project_root=root, allow_provisional=allow_provisional)
    decomposition = overlay.get("decomposition")
    if not isinstance(decomposition, dict) or str(decomposition.get("method", "")).upper() != "DMD":
        raise CouplingConfigurationError("decomposition.method must be DMD")
    output_dir = _resolve(root, overlay.get("output_dir"), "output_dir")
    mode_count = int(decomposition.get("mode_count", -1))
    energy_target = float(decomposition.get("full_space_energy_target", float("nan")))
    if mode_count < 1 or not math.isfinite(energy_target) or not (0.0 < energy_target <= 1.0):
        raise CouplingConfigurationError("invalid DMD mode count or energy target")
    merged = copy.deepcopy(base.config)
    merged["study"]["output_dir"] = str(output_dir)
    merged["study"]["mode_count"] = mode_count
    merged["decomposition"] = copy.deepcopy(decomposition)
    context = replace(
        base,
        config_path=overlay_path,
        config=merged,
        output_dir=output_dir,
        mode_count=mode_count,
        full_space_energy_target=energy_target,
    )
    return context, overlay


def _case_definitions(flow: DMD3DFlow, stages: Iterable[int]) -> List[CaseDefinition]:
    rows = build_stage_rows(flow.frequency_hz, flow.strouhal)
    output = []
    for stage in sorted(set(int(value) for value in stages)):
        if stage < 0 or stage > flow.mode_count:
            raise CouplingConfigurationError(
                "stage index must lie in [0,{}]".format(flow.mode_count)
            )
        row = rows[stage]
        output.append(
            CaseDefinition(
                case_id=str(row["case_id"]),
                stage_index=stage,
                retained_mode_count=int(row["retained_mode_count"]),
                flow_kind="DMD",
                cutoff_frequency_hz=row["cutoff_frequency_hz"],
                cutoff_St=row["cutoff_St"],
            )
        )
    return output


def _sensitivity_definitions(
    flow: DMD3DFlow, mode_indices_one_based: Optional[Iterable[int]]
) -> List[CaseDefinition]:
    selected = (
        list(range(1, flow.mode_count + 1))
        if mode_indices_one_based is None
        else sorted(set(int(value) for value in mode_indices_one_based))
    )
    if not selected or selected[0] < 1 or selected[-1] > flow.mode_count:
        raise CouplingConfigurationError(
            "sensitivity modes must lie in [1,{}]".format(flow.mode_count)
        )
    width = max(3, len(str(flow.mode_count)))
    all_modes = tuple(range(flow.mode_count))
    definitions = [
        CaseDefinition(
            "sensitivity_full", None, flow.mode_count, "DMD",
            float(flow.frequency_hz[-1]), float(flow.strouhal[-1]),
            sensitivity_case_kind="full",
        ),
        CaseDefinition(
            "sensitivity_mean", None, 0, "DMD", None, None,
            sensitivity_case_kind="mean",
        ),
    ]
    for one_based in selected:
        zero_based = one_based - 1
        tag = "M{:0{width}d}".format(one_based, width=width)
        definitions.append(
            CaseDefinition(
                "sensitivity_single_{}".format(tag), None, 1, "DMD",
                float(flow.frequency_hz[zero_based]),
                float(flow.strouhal[zero_based]),
                mode_indices_zero_based=(zero_based,),
                sensitivity_case_kind="single",
                sensitivity_mode_index_one_based=one_based,
            )
        )
        definitions.append(
            CaseDefinition(
                "sensitivity_loo_{}".format(tag), None, flow.mode_count - 1, "DMD",
                float(flow.frequency_hz[-1]), float(flow.strouhal[-1]),
                mode_indices_zero_based=tuple(
                    value for value in all_modes if value != zero_based
                ),
                sensitivity_case_kind="leave_one_out",
                sensitivity_mode_index_one_based=one_based,
            )
        )
    return definitions


def _definition_dict(definition: CaseDefinition) -> Dict[str, Any]:
    return {
        "case_id": definition.case_id,
        "stage_index": definition.stage_index,
        "retained_mode_count": definition.retained_mode_count,
        "flow_kind": definition.flow_kind,
        "cutoff_frequency_hz": definition.cutoff_frequency_hz,
        "cutoff_St": definition.cutoff_St,
        "mode_indices_zero_based": (
            None if definition.mode_indices_zero_based is None
            else list(definition.mode_indices_zero_based)
        ),
        "sensitivity_case_kind": definition.sensitivity_case_kind,
        "sensitivity_mode_index_one_based": definition.sensitivity_mode_index_one_based,
    }


def _failure_matches(path: Path, definition: CaseDefinition, signature: str) -> bool:
    try:
        saved = _read_json(path)
        return (
            saved.get("case_definition") == _definition_dict(definition)
            and saved.get("run_signature_sha256") == signature
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _initialize_worker(dmd_directory: str, cache_directory: str, dt_s: float) -> None:
    global _WORKER_DMD_FLOW, _WORKER_BASELINE_FLOW
    _WORKER_DMD_FLOW = DMD3DFlow(Path(dmd_directory))
    _WORKER_BASELINE_FLOW = NatoGD3DBaselineFlow(
        Path(cache_directory), expected_dt_s=float(dt_s)
    )


def _execute_worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    if _WORKER_DMD_FLOW is None or _WORKER_BASELINE_FLOW is None:
        raise CouplingRunError("DMD coupling worker was not initialized")
    definition = task["definition"]
    model = task["model"]
    output_path = Path(str(task["output_path"]))
    metadata_path = output_path.with_suffix(".metadata.json")
    failure_path = output_path.with_suffix(".failure.json")
    signature = str(task["run_signature_sha256"])
    resume = bool(task["resume"])
    reusable = (
        resume and output_path.is_file() and metadata_path.is_file()
        and coupling._resume_output_matches_request(
            metadata_path, definition, model,
            end_time_s=float(task["end_time_s"]),
            output_interval_s=float(task["output_interval_s"]),
            maximum_azimuth_step_deg=float(task["maximum_azimuth_step_deg"]),
            run_signature_sha256=signature,
        )
    )
    if reusable:
        result = coupling.read_case_csv(output_path, definition)
        return {"status": "valid", "definition": definition, "result": result,
                "output_path": output_path}
    if resume and failure_path.is_file() and _failure_matches(
        failure_path, definition, signature
    ):
        return {"status": "model_validity_failure", "definition": definition,
                "output_path": output_path, "failure": _read_json(failure_path)}

    active_flow = (
        _WORKER_BASELINE_FLOW
        if definition.flow_kind == "original_CFD" else _WORKER_DMD_FLOW
    )
    print("simulate {} {}".format(model.rotor["name"], definition.case_id), flush=True)
    started = time.monotonic()
    try:
        result = coupling.simulate_case(
            model, active_flow, definition,
            end_time_s=float(task["end_time_s"]),
            output_interval_s=float(task["output_interval_s"]),
            maximum_azimuth_step_deg=float(task["maximum_azimuth_step_deg"]),
        )
    except (DynamicsDomainError, AerodynamicDomainError, SpatialOutOfBoundsError) as error:
        failure = {
            "status": "model_validity_failure",
            "failure_classification": type(error).__name__,
            "failure_reason": str(error),
            "rotor_name": str(model.rotor["name"]),
            "case_definition": _definition_dict(definition),
            "elapsed_wall_s": time.monotonic() - started,
            "run_signature_sha256": signature,
        }
        _write_json(failure_path, failure)
        return {"status": "model_validity_failure", "definition": definition,
                "output_path": output_path, "failure": failure}
    result.metadata["run_signature_sha256"] = signature
    coupling.write_case_csv(output_path, result)
    return {"status": "valid", "definition": definition, "result": result,
            "output_path": output_path}


def write_dmd_group_frequency_table(flow: DMD3DFlow, directory: Path) -> Path:
    """Persist group self-energies used only for unit-energy gain normalization."""

    scales = np.asarray(flow.metadata.get("component_scales_physical"), dtype=np.float64)
    if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise CouplingConfigurationError("DMD component scales are invalid")
    rows: List[Dict[str, Any]] = []
    for mode in range(flow.mode_count):
        start = int(flow.group_component_start[mode])
        stop = int(flow.group_component_stop[mode])
        spatial = np.asarray(flow.spatial_modes[:, :, start:stop], dtype=np.float64)
        spatial /= scales[None, :, None]
        spatial = spatial.reshape(-1, stop - start)
        temporal = np.asarray(flow.coefficients[:, start:stop], dtype=np.float64)
        spatial_gram = spatial.T @ spatial
        temporal_gram = temporal.T @ temporal
        self_energy = float(np.trace(spatial_gram @ temporal_gram))
        if not math.isfinite(self_energy) or self_energy <= 0.0:
            raise CouplingRunError("non-positive DMD group self-energy at mode {}".format(mode + 1))
        rows.append({
            "mode": mode + 1,
            "frequency_cycles_per_sample": float(flow.frequency_hz[mode] * flow.dt_s),
            "frequency_hz": float(flow.frequency_hz[mode]),
            "strouhal_St_f13.716_over_20": float(flow.strouhal[mode]),
            "temporal_coefficient_energy": float(np.sum(temporal ** 2)),
            "temporal_coefficient_energy_fraction": float("nan"),
            "full_normalized_modal_self_energy": self_energy,
            "quadrature_count": stop - start,
            "component_start_zero_based": start,
            "component_stop_exclusive": stop,
        })
    temporal_total = float(sum(row["temporal_coefficient_energy"] for row in rows))
    for row in rows:
        row["temporal_coefficient_energy_fraction"] = (
            row["temporal_coefficient_energy"] / temporal_total
        )
    path = directory / "frequencies.csv"
    coupling.write_rows_csv(path, rows)
    _write_json(directory / "dmd_group_energy_audit.json", {
        "status": "pass",
        "mode_count": flow.mode_count,
        "real_quadrature_count": flow.component_count,
        "group_self_energy_sum": float(sum(row["full_normalized_modal_self_energy"] for row in rows)),
        "full_normalized_captured_energy": float(
            flow.full_space_audit["full_normalized_captured_energy"]
        ),
        "interpretation": (
            "Each group self-energy is ||Phi_g C_g^T||_F^2 in the common normalized "
            "velocity metric. DMD groups are non-orthogonal, so their sum is not used "
            "as the 99.8% captured-energy gate."
        ),
    })
    return path


def _trim_row(model: RotorStudyModel, rotor: Mapping[str, Any]) -> Dict[str, Any]:
    row = model.trim.as_row()
    audit = model.mean_flow_trim_audit
    force = audit["mean_force_body_N"]
    moment = audit["mean_hub_aero_moment_body_Nm"]
    row.update({
        "trim_environment": "NATO-GD DMD time-mean 3-D field",
        "trim_method": audit["method"],
        "periodic_orbit_shooting": audit["periodic_orbit_shooting"],
        "mean_flow_trim_phase_samples": audit["phase_samples"],
        "still_air_seed_collective_deg": model.trim.collective_trim_deg,
        "still_air_seed_induced_velocity_mps": model.trim.induced_velocity_mps,
        "collective_trim_deg": audit["collective_deg"],
        "induced_velocity_trim_mps": audit["uniform_induced_velocity_mps"],
        "initial_delta_lambda_0": model.initial_inflow_state[0],
        "initial_delta_lambda_c": model.initial_inflow_state[1],
        "initial_delta_lambda_s": model.initial_inflow_state[2],
        "initial_uniform_total_lambda": audit["uniform_total_lambda"],
        "trim_thrust_N": audit["mean_thrust_N"],
        "trim_Fx_N": force[0], "trim_Fy_N": force[1], "trim_Fz_N": force[2],
        "trim_hub_Mx_Nm": moment[0], "trim_hub_My_Nm": moment[1],
        "trim_hub_Mz_Nm": moment[2],
        "trim_shaft_torque_Nm": audit["mean_shaft_torque_Nm"],
        "trim_shaft_power_W": audit["mean_shaft_power_W"],
        "trim_engine_power_W": audit["mean_engine_power_W"],
        "alpha_domain_violation_count": audit["maximum_alpha_domain_violation_count_per_phase"],
        "cl_saturation_count": audit["maximum_cl_saturation_count_per_phase"],
        "relative_thrust_residual": (
            float(audit["mean_thrust_N"]) - model.trim.weight_N
        ) / model.trim.weight_N,
        "mean_flow_trim_maximum_abs_scaled_residual": audit["maximum_abs_scaled_residual"],
        "parameter_status": rotor.get("parameter_status", "unspecified"),
        "balance_definition": model.balance_definition,
        "balance_Fx_body_N": model.balance_other_force_body_N[0],
        "balance_Fy_body_N": model.balance_other_force_body_N[1],
        "balance_Fz_body_N": model.balance_other_force_body_N[2],
        "balance_Mx_body_Nm": model.balance_other_moment_body_Nm[0],
        "balance_My_body_Nm": model.balance_other_moment_body_Nm[1],
        "balance_Mz_body_Nm": model.balance_other_moment_body_Nm[2],
        "hover_hold_enabled": model.hover_hold_metadata["enabled"],
    })
    return row


def _manifest_kind(definition: CaseDefinition, mode_count: int) -> Optional[str]:
    if definition.sensitivity_case_kind is not None:
        return definition.sensitivity_case_kind
    if definition.stage_index == 0:
        return "full"
    if definition.stage_index == mode_count:
        return "mean"
    return None


def run_study(
    context: StudyContext,
    overlay: Mapping[str, Any],
    *,
    rotor_names: Optional[Sequence[str]] = None,
    stages: Optional[Sequence[int]] = None,
    include_modal_sensitivity: bool = True,
    sensitivity_mode_indices_one_based: Optional[Sequence[int]] = None,
    include_original_cfd: bool = True,
    resume: bool = True,
    end_time_override_s: Optional[float] = None,
    maximum_azimuth_step_override_deg: Optional[float] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    started = time.monotonic()
    for relative in (
        "00_provenance", "03_hover_trim", "04_timeseries_csv",
        "05_statistics", "06_figures/manuscript", "06_figures/source_data", "09_qa",
    ):
        (context.output_dir / relative).mkdir(parents=True, exist_ok=True)
    _write_json(context.output_dir / "00_provenance" / "study_context.json", {
        **context.metadata(), "decomposition": dict(overlay["decomposition"])
    })
    simulation, rotors = coupling.load_rotor_mappings(context)
    if rotor_names:
        requested = set(rotor_names)
        available = {str(rotor["name"]) for rotor in rotors}
        missing = requested - available
        if missing:
            raise CouplingConfigurationError("unknown rotors: {}".format(sorted(missing)))
        rotors = [rotor for rotor in rotors if str(rotor["name"]) in requested]

    decomposition = overlay["decomposition"]
    dmd_directory = context.output_dir / str(
        decomposition.get("artifact_subdirectory", "02_dmd")
    )
    dmd_flow = DMD3DFlow(dmd_directory)
    if dmd_flow.mode_count != context.mode_count:
        raise CouplingConfigurationError("configured and stored DMD mode counts disagree")
    if not math.isclose(dmd_flow.dt_s, context.dt_s, rel_tol=0.0, abs_tol=1e-15):
        raise CouplingConfigurationError("configured and stored DMD time steps disagree")
    flow_audit = dmd_flow.validate()
    energy_gate = coupling._validate_full_space_energy_gate(
        dmd_flow.metadata, context.full_space_energy_target
    )
    write_dmd_group_frequency_table(dmd_flow, dmd_directory)
    baseline_flow = NatoGD3DBaselineFlow(context.cache_dir, expected_dt_s=context.dt_s)
    grid_audit = validate_grid_and_time(
        context, dmd_flow.bounds_xyz_m, dmd_flow.frame_count,
        [float(rotor["diameter_m"]) for rotor in rotors],
    )
    baseline_bounds = np.asarray(
        baseline_flow.metadata["bounds_xyz_m"], dtype=np.float32
    ).astype(float)
    if not np.array_equal(dmd_flow.bounds_xyz_m, baseline_bounds):
        raise CouplingConfigurationError("DMD and original-CFD spatial bounds disagree")

    signature_paths = [
        Path(__file__).resolve(),
        Path(coupling.__file__).resolve(),
        Path(sys.modules[DMD3DFlow.__module__].__file__).resolve(),
        context.config_path, context.rotor_config_path, context.dynamic_overrides_path,
        context.cache_dir / "manifest.json", dmd_directory / "metadata.json",
        dmd_directory / "full_space_regression_audit.json",
        dmd_directory / "temporal_model.npz",
        dmd_directory / "mean_field_physical.npy",
        dmd_directory / "spatial_modes_physical.npy",
    ]
    run_signature = coupling._artifact_signature(signature_paths)
    stage_indices = (
        list(range(dmd_flow.mode_count + 1))
        if stages is None else sorted(set(int(value) for value in stages))
    )
    definitions = _case_definitions(dmd_flow, stage_indices)
    if include_modal_sensitivity:
        extra = _sensitivity_definitions(dmd_flow, sensitivity_mode_indices_one_based)
        has_full = any(item.stage_index == 0 for item in definitions)
        has_mean = any(item.stage_index == dmd_flow.mode_count for item in definitions)
        definitions.extend(
            item for item in extra
            if not ((item.sensitivity_case_kind == "full" and has_full)
                    or (item.sensitivity_case_kind == "mean" and has_mean))
        )
    write_stage_manifest(
        context.output_dir / "00_provenance" / "stage_manifest.csv",
        build_stage_rows(dmd_flow.frequency_hz, dmd_flow.strouhal),
    )
    body = context.config["body_response"]
    end_time = float(body["analysis_window_end_s"] if end_time_override_s is None
                     else end_time_override_s)
    maximum_azimuth_step = float(
        context.config["rotor_model"]["maximum_azimuth_step_deg"]
        if maximum_azimuth_step_override_deg is None
        else maximum_azimuth_step_override_deg
    )
    worker_count = int(workers)
    if worker_count < 1 or worker_count > 8:
        raise CouplingConfigurationError("workers must lie in [1,8]")

    trim_rows: List[Dict[str, Any]] = []
    statistics: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    case_audits: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    completed_count = 0
    total_expected = len(rotors) * (len(definitions) + int(include_original_cfd))

    def update_progress(current_rotor: str) -> None:
        _write_json(context.output_dir / "00_provenance" / "coupling_progress.json", {
            "status": "running", "decomposition_method": "DMD",
            "current_rotor": current_rotor, "completed_case_count": completed_count,
            "expected_case_count": total_expected, "failure_count": len(failures),
            "rotors_completed": sorted(set(
                row["rotor_name"] for row in case_audits
                if row.get("rotor_finished") is True
            )),
            "run_signature_sha256": run_signature,
        })

    for rotor in rotors:
        model = coupling.prepare_rotor_model(rotor, simulation, context, dmd_flow)
        model.balance_definition = model.balance_definition.replace(
            "RVMD time-mean NATO-GD field",
            "DMD time-mean NATO-GD field",
        )
        trim_rows.append(_trim_row(model, rotor))
        rotor_definitions = list(definitions)
        if include_original_cfd:
            rotor_definitions.append(
                CaseDefinition("original_cfd", None, None, "original_CFD", None, None)
            )
        tasks = []
        for definition in rotor_definitions:
            output_name = "{}__{}.csv".format(
                coupling._slug(str(rotor["name"])), definition.case_id
            )
            tasks.append({
                "definition": definition, "model": model,
                "output_path": str(context.output_dir / "04_timeseries_csv" / output_name),
                "resume": resume, "end_time_s": end_time,
                "output_interval_s": context.dt_s,
                "maximum_azimuth_step_deg": maximum_azimuth_step,
                "run_signature_sha256": run_signature,
            })
        outcomes: List[Dict[str, Any]] = []
        if worker_count == 1:
            _initialize_worker(str(dmd_directory), str(context.cache_dir), context.dt_s)
            outcomes = [_execute_worker(task) for task in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(tasks)), initializer=_initialize_worker,
                initargs=(str(dmd_directory), str(context.cache_dir), context.dt_s),
            ) as executor:
                future_map = {executor.submit(_execute_worker, task): task for task in tasks}
                for future in as_completed(future_map):
                    outcomes.append(future.result())
                    completed_count += 1
                    if completed_count % 10 == 0:
                        update_progress(str(rotor["name"]))
        if worker_count == 1:
            completed_count += len(outcomes)
        outcomes.sort(key=lambda item: item["definition"].case_id)
        for outcome in outcomes:
            definition = outcome["definition"]
            kind = _manifest_kind(definition, dmd_flow.mode_count)
            if outcome["status"] == "valid":
                result = outcome["result"]
                output_path = outcome["output_path"]
                statistics.extend(coupling.statistics_rows(
                    result, float(body["analysis_window_start_s"]),
                    float(body["analysis_window_end_s"]), float(rotor["diameter_m"])
                ))
                diag = result.metadata.get("dynamics_diagnostics", {})
                if int(diag.get("hard_failures", 0)) != 0:
                    raise CouplingRunError("completed DMD case reports a hard failure")
                case_audits.append({
                    "rotor_name": rotor["name"], "case_id": definition.case_id,
                    "status": "valid", "csv": str(output_path),
                    "metadata": result.metadata,
                })
                if kind is not None:
                    manifest_rows.append({
                        "rotor_name": rotor["name"],
                        "rotor_diameter_m": rotor["diameter_m"],
                        "case_kind": kind,
                        "mode_index_one_based": (
                            "" if definition.sensitivity_mode_index_one_based is None
                            else definition.sensitivity_mode_index_one_based
                        ),
                        "case_status": "valid", "failure_classification": "",
                        "failure_reason": "", "csv_path": str(output_path.resolve()),
                        "metadata_path": str(output_path.with_suffix(".metadata.json").resolve()),
                        "run_signature_sha256": run_signature,
                    })
            else:
                failure = dict(outcome["failure"])
                failures.append(failure)
                case_audits.append({
                    "rotor_name": rotor["name"], "case_id": definition.case_id,
                    "status": "model_validity_failure", "failure": failure,
                })
                if kind in ("full", "mean"):
                    raise CouplingRunError(
                        "required {} baseline failed for {}: {}".format(
                            kind, rotor["name"], failure["failure_reason"]
                        )
                    )
                if kind is not None:
                    manifest_rows.append({
                        "rotor_name": rotor["name"],
                        "rotor_diameter_m": rotor["diameter_m"],
                        "case_kind": kind,
                        "mode_index_one_based": definition.sensitivity_mode_index_one_based,
                        "case_status": "model_validity_failure",
                        "failure_classification": failure["failure_classification"],
                        "failure_reason": failure["failure_reason"],
                        "csv_path": "", "metadata_path": "",
                        "run_signature_sha256": "",
                    })
        case_audits.append({"rotor_name": rotor["name"], "rotor_finished": True})
        update_progress(str(rotor["name"]))

    if trim_rows:
        coupling.write_rows_csv(
            context.output_dir / "03_hover_trim" / "hover_trim_all_rotors.csv", trim_rows
        )
    if statistics:
        coupling.write_rows_csv(
            context.output_dir / "05_statistics" / "statistics_long.csv", statistics
        )
    if manifest_rows:
        coupling.write_rows_csv(
            context.output_dir / "05_statistics" / "modal_sensitivity_case_manifest.csv",
            manifest_rows,
        )
    _write_json(context.output_dir / "09_qa" / "model_validity_failures.json", {
        "failure_count": len(failures), "failures": failures,
        "policy": "known model-domain failures are marked; independent cases continue",
    })
    requested_modes = (
        list(range(1, dmd_flow.mode_count + 1))
        if include_modal_sensitivity and sensitivity_mode_indices_one_based is None
        else list(sensitivity_mode_indices_one_based or [])
    )
    summary = {
        "status": "provisional" if not context.production_ready else "production_ready",
        "decomposition_method": "DMD",
        "dmd_flow_audit": flow_audit,
        "study_context": context.metadata(), "grid_and_time_audit": grid_audit,
        "rotors_completed": [rotor["name"] for rotor in rotors],
        "stage_indices_requested": stage_indices,
        "modal_sensitivity_requested": bool(include_modal_sensitivity),
        "sensitivity_mode_indices_one_based": requested_modes,
        "full_space_energy_gate": energy_gate,
        "original_cfd_requested": include_original_cfd,
        "parallel_case_workers": worker_count,
        "run_signature_sha256": run_signature,
        "completed_valid_case_count": sum(
            item.get("status") == "valid" for item in case_audits
        ),
        "model_validity_failure_count": len(failures),
        "elapsed_wall_s": time.monotonic() - started,
        "case_audits": case_audits,
    }
    _write_json(context.output_dir / "00_provenance" / "coupling_run.json", summary)
    _write_json(context.output_dir / "00_provenance" / "coupling_progress.json", {
        "status": "complete", "completed_case_count": completed_count,
        "expected_case_count": total_expected, "failure_count": len(failures),
        "rotors_completed": summary["rotors_completed"],
        "run_signature_sha256": run_signature,
    })
    dmd_flow.close()
    # NatoGD3DBaselineFlow delegates to a cache object without a close method.
    return summary


def _parse_int_ranges(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            values.extend(range(first, last + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_dmd_3d_coupling.json", type=Path)
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--rotor", action="append", dest="rotors")
    parser.add_argument("--stages", help="comma/range list; default all 0..K")
    parser.add_argument("--no-modal-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-modes", help="optional one-based comma/range subset")
    parser.add_argument("--skip-original-cfd", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--maximum-azimuth-step-deg", type=float)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    context, overlay = load_dmd_study_context(
        args.config, allow_provisional=args.allow_provisional
    )
    summary = run_study(
        context, overlay, rotor_names=args.rotors,
        stages=_parse_int_ranges(args.stages),
        include_modal_sensitivity=not args.no_modal_sensitivity,
        sensitivity_mode_indices_one_based=_parse_int_ranges(args.sensitivity_modes),
        include_original_cfd=not args.skip_original_cfd,
        resume=not args.no_resume, end_time_override_s=args.end_time,
        maximum_azimuth_step_override_deg=args.maximum_azimuth_step_deg,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build_argument_parser", "load_dmd_study_context", "run_study",
    "write_dmd_group_frequency_table",
]
