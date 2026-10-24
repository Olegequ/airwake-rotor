#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Numerical-step and parameter-sensitivity QA for NATO-GD/RVMD coupling.

The QA uses the full energy-selected RVMD case for the smallest, median and largest
rotor diameters.  It does not alter or overwrite production time histories.
All comparisons use the same physical hover trim, initial inflow state and
analysis window as the production calculation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .nato_gd_study import load_study_context
from .plot_rvmd_3d_results import (
    COLORS,
    METRICS,
    SHORT_LABELS,
    _panel_label,
)
from .rvmd_3d_flow import RVMD3DFlow
from .rvmd_nato_coupling_pipeline import (
    CaseDefinition,
    RotorStudyModel,
    load_rotor_mappings,
    prepare_rotor_model,
    simulate_case,
)


class NumericalQAError(RuntimeError):
    pass


_QA_WORKER_FLOW: Optional[RVMD3DFlow] = None


def _initialize_qa_worker(artifact_directory: str) -> None:
    global _QA_WORKER_FLOW
    _QA_WORKER_FLOW = RVMD3DFlow(Path(artifact_directory))


def _run_case_worker(task: Mapping[str, Any]) -> Tuple[str, float, Any]:
    if _QA_WORKER_FLOW is None:
        raise NumericalQAError("QA worker flow was not initialized")
    result = simulate_case(
        task["model"],
        _QA_WORKER_FLOW,
        task["definition"],
        end_time_s=float(task["end_time_s"]),
        output_interval_s=float(task["output_interval_s"]),
        maximum_azimuth_step_deg=float(task["step_deg"]),
    )
    return str(task["name"]), float(task["step_deg"]), result


FIGURE_WIDTH_MM = 183.0
QA_RASTER_DPI = 600
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def _save_qa_figure(
    figure: plt.Figure, base: Path, formats: Sequence[str], dpi: int
) -> List[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.8)
    outputs = []
    for raw_format in formats:
        fmt = raw_format.lower().lstrip(".")
        if fmt == "svg":
            path = base.with_suffix(".svg")
        elif fmt == "pdf":
            path = base.with_suffix(".pdf")
        elif fmt in ("tif", "tiff"):
            path = base.with_suffix(".tiff")
        elif fmt == "png":
            path = base.with_suffix(".png")
        else:
            raise NumericalQAError("unsupported figure format: {}".format(fmt))
        figure.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise NumericalQAError("JSON root must be an object: {}".format(path))
    return value


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise NumericalQAError("cannot write an empty QA table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _representative_rotors(rotors: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    ordered = sorted(rotors, key=lambda item: float(item["diameter_m"]))
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    output: List[Mapping[str, Any]] = []
    seen = set()
    for rotor in selected:
        name = str(rotor["name"])
        if name not in seen:
            output.append(rotor)
            seen.add(name)
    return output


def _full_case(flow: RVMD3DFlow) -> CaseDefinition:
    width = max(2, len(str(flow.mode_count)))
    return CaseDefinition(
        case_id="stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
            stage=0, retained=flow.mode_count, width=width
        ),
        stage_index=0,
        retained_mode_count=flow.mode_count,
        flow_kind="RVMD",
        cutoff_frequency_hz=float(flow.frequency_hz[-1]),
        cutoff_St=float(flow.strouhal[-1]),
    )


def _metric_array(result: Any, metric: str) -> np.ndarray:
    values = np.asarray([float(row[metric]) for row in result.rows], dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise NumericalQAError("non-finite metric history: {}".format(metric))
    return values


def _analysis_mask(result: Any, start_s: float, end_s: float) -> np.ndarray:
    time_values = np.asarray([float(row["time_s"]) for row in result.rows], dtype=float)
    return (time_values >= start_s - 1.0e-12) & (time_values <= end_s + 1.0e-12)


def _metric_reference(model: RotorStudyModel, metric: str, values: np.ndarray) -> Tuple[float, str]:
    if metric == "lift_N":
        return float(model.trim.weight_N), "weight_N"
    if metric == "engine_power_W":
        return max(
            abs(float(model.mean_flow_trim_audit["mean_engine_power_W"])),
            np.finfo(float).tiny,
        ), "mean_flow_trim_engine_power_W"
    rms = float(np.sqrt(np.mean(values ** 2)))
    if metric in ("roll_deg", "pitch_deg"):
        return max(rms, 1.0), "max(response_rms,1_deg)"
    if metric == "heave_m":
        return max(rms, 0.1), "max(response_rms,0.1_m)"
    raise NumericalQAError("unknown metric {}".format(metric))


def _compare_steps(
    rotor_name: str,
    model: RotorStudyModel,
    coarse_result: Any,
    fine_result: Any,
    coarse_step_deg: float,
    fine_step_deg: float,
    start_s: float,
    end_s: float,
    rmse_tolerance: float,
    maximum_tolerance: float,
) -> List[Dict[str, Any]]:
    coarse_time = np.asarray([float(row["time_s"]) for row in coarse_result.rows])
    fine_time = np.asarray([float(row["time_s"]) for row in fine_result.rows])
    if not np.array_equal(coarse_time, fine_time):
        raise NumericalQAError("time grids disagree for {}".format(rotor_name))
    mask = _analysis_mask(fine_result, start_s, end_s)
    if int(np.count_nonzero(mask)) < 2:
        raise NumericalQAError("analysis window is empty for {}".format(rotor_name))
    rows = []
    for metric in METRICS:
        coarse = _metric_array(coarse_result, metric)[mask]
        fine = _metric_array(fine_result, metric)[mask]
        error = coarse - fine
        reference, reference_definition = _metric_reference(model, metric, fine)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        maximum = float(np.max(np.abs(error)))
        rmse_normalized = rmse / reference
        maximum_normalized = maximum / reference
        rows.append(
            {
                "rotor_name": rotor_name,
                "metric": metric,
                "coarse_maximum_azimuth_step_deg": coarse_step_deg,
                "fine_maximum_azimuth_step_deg": fine_step_deg,
                "analysis_start_s": start_s,
                "analysis_end_s": end_s,
                "sample_count": int(np.count_nonzero(mask)),
                "rmse_physical": rmse,
                "maximum_abs_error_physical": maximum,
                "reference_scale": reference,
                "reference_definition": reference_definition,
                "rmse_normalized": rmse_normalized,
                "maximum_abs_error_normalized": maximum_normalized,
                "rmse_tolerance": rmse_tolerance,
                "maximum_abs_error_tolerance": maximum_tolerance,
                "passes": bool(
                    rmse_normalized <= rmse_tolerance
                    and maximum_normalized <= maximum_tolerance
                ),
            }
        )
    return rows


def _variant_model(
    model: RotorStudyModel, parameter_group: str, scale: float
) -> RotorStudyModel:
    candidate = copy.deepcopy(model)
    factor = float(scale)
    if parameter_group == "roll_pitch_inertia":
        inertia = np.asarray(candidate.body_parameters.inertia_body_kgm2, dtype=float).copy()
        inertia[:2, :2] *= factor
        candidate.body_parameters = replace(
            candidate.body_parameters, inertia_body_kgm2=inertia
        )
    elif parameter_group == "airframe_damping":
        candidate.body_parameters = replace(
            candidate.body_parameters,
            vertical_damping_N_per_mps=(
                candidate.body_parameters.vertical_damping_N_per_mps * factor
            ),
            angular_damping_body_Nms=(
                candidate.body_parameters.angular_damping_body_Nms * factor
            ),
        )
    elif parameter_group == "hover_hold_gains":
        candidate.hover_hold_vertical_stiffness_Npm *= factor
        candidate.hover_hold_vertical_rate_gain_Ns_pm *= factor
        candidate.hover_hold_angular_stiffness_Nm_per_rad *= factor
        candidate.hover_hold_angular_rate_gain_Nms *= factor
    else:
        raise NumericalQAError("unknown sensitivity group {}".format(parameter_group))
    return candidate


def _summarize_sensitivity(
    rotor_name: str,
    model: RotorStudyModel,
    baseline: Any,
    variant: Any,
    parameter_group: str,
    scale: float,
    start_s: float,
    end_s: float,
) -> List[Dict[str, Any]]:
    mask = _analysis_mask(baseline, start_s, end_s)
    if not np.array_equal(mask, _analysis_mask(variant, start_s, end_s)):
        raise NumericalQAError("sensitivity time grids disagree for {}".format(rotor_name))
    rows = []
    for metric in METRICS:
        baseline_values = _metric_array(baseline, metric)[mask]
        variant_values = _metric_array(variant, metric)[mask]
        reference, reference_definition = _metric_reference(model, metric, baseline_values)
        statistics = {
            "mean": (float(np.mean(baseline_values)), float(np.mean(variant_values))),
            "rms": (
                float(np.sqrt(np.mean(baseline_values ** 2))),
                float(np.sqrt(np.mean(variant_values ** 2))),
            ),
            "peak_to_peak": (
                float(np.ptp(baseline_values)), float(np.ptp(variant_values))
            ),
            "max_abs": (
                float(np.max(np.abs(baseline_values))),
                float(np.max(np.abs(variant_values))),
            ),
        }
        for statistic, (baseline_value, variant_value) in statistics.items():
            delta = variant_value - baseline_value
            relative = (
                delta / abs(baseline_value)
                if abs(baseline_value) > 1.0e-14 * max(reference, 1.0)
                else float("nan")
            )
            rows.append(
                {
                    "rotor_name": rotor_name,
                    "parameter_group": parameter_group,
                    "parameter_scale": scale,
                    "metric": metric,
                    "statistic": statistic,
                    "baseline_value": baseline_value,
                    "variant_value": variant_value,
                    "delta_physical": delta,
                    "reference_scale": reference,
                    "reference_definition": reference_definition,
                    "delta_normalized_by_reference": delta / reference,
                    "relative_change_when_defined": relative,
                    "analysis_start_s": start_s,
                    "analysis_end_s": end_s,
                }
            )
    return rows


def _plot_qa(
    convergence: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    rotors = list(dict.fromkeys(convergence["rotor_name"].tolist()))
    convergence_matrix = np.zeros((len(rotors), len(METRICS)), dtype=float)
    for row_index, rotor in enumerate(rotors):
        for column_index, metric in enumerate(METRICS):
            selected = convergence[
                (convergence["rotor_name"] == rotor)
                & (convergence["metric"] == metric)
            ]
            if selected.empty:
                raise NumericalQAError("convergence figure input is incomplete")
            convergence_matrix[row_index, column_index] = float(
                selected["maximum_abs_error_normalized"].max()
            )
    groups = ["roll_pitch_inertia", "airframe_damping", "hover_hold_gains"]
    sensitivity_matrix = np.zeros((len(groups), len(METRICS)), dtype=float)
    for row_index, group in enumerate(groups):
        for column_index, metric in enumerate(METRICS):
            selected = sensitivity[
                (sensitivity["parameter_group"] == group)
                & (sensitivity["metric"] == metric)
            ]
            sensitivity_matrix[row_index, column_index] = float(
                selected["delta_normalized_by_reference"].abs().max()
            )
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_MM / 25.4, 78.0 / 25.4))
    image_a = axes[0].imshow(convergence_matrix, aspect="auto", cmap="viridis", vmin=0.0)
    figure.colorbar(
        image_a, ax=axes[0], fraction=0.046, pad=0.03,
        label="Maximum step-pair error / reference",
    )
    axes[0].set_yticks(np.arange(len(rotors)))
    axes[0].set_yticklabels(rotors)
    axes[0].set_xticks(np.arange(len(METRICS)))
    axes[0].set_xticklabels(
        [SHORT_LABELS[item] for item in METRICS], rotation=30, ha="right",
        rotation_mode="anchor",
    )
    axes[0].set_title("Azimuth-step convergence", fontsize=7.0)
    _panel_label(axes[0], "a")
    image_b = axes[1].imshow(sensitivity_matrix, aspect="auto", cmap="magma", vmin=0.0)
    figure.colorbar(
        image_b, ax=axes[1], fraction=0.046, pad=0.03,
        label="±20% sensitivity envelope / reference",
    )
    axes[1].set_yticks(np.arange(len(groups)))
    axes[1].set_yticklabels(["Roll/pitch inertia", "Airframe damping", "Hover-hold gains"])
    axes[1].set_xticks(np.arange(len(METRICS)))
    axes[1].set_xticklabels(
        [SHORT_LABELS[item] for item in METRICS], rotation=30, ha="right",
        rotation_mode="anchor",
    )
    axes[1].set_title("Parameter sensitivity", fontsize=7.0)
    _panel_label(axes[1], "b")
    return _save_qa_figure(
        figure, output / "figure_QA01_numerical_and_parameter_sensitivity", formats, dpi
    )


def run_qa(
    config_path: Path,
    *,
    workers: int = 3,
    coarse_step_deg: float = 5.0,
    fine_step_deg: float = 2.5,
    finer_step_deg: float = 1.25,
    rmse_tolerance: float = 0.02,
    maximum_tolerance: float = 0.05,
    formats: Sequence[str] = ("svg", "pdf", "tiff", "png"),
    dpi: int = QA_RASTER_DPI,
) -> Dict[str, Any]:
    context = load_study_context(config_path, allow_provisional=True)
    simulation, all_rotors = load_rotor_mappings(context)
    selected_rotors = _representative_rotors(all_rotors)
    flow = RVMD3DFlow(context.output_dir / "02_rvmd")
    definition = _full_case(flow)
    body_config = context.config["body_response"]
    start_s = float(body_config["analysis_window_start_s"])
    end_s = float(body_config["analysis_window_end_s"])
    models = {
        str(rotor["name"]): prepare_rotor_model(rotor, simulation, context, flow)
        for rotor in selected_rotors
    }
    worker_count = max(1, min(int(workers), 8))

    def task_payload(
        name: str, model: RotorStudyModel, step_deg: float
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "model": copy.deepcopy(model),
            "definition": definition,
            "end_time_s": end_s,
            "output_interval_s": context.dt_s,
            "step_deg": float(step_deg),
        }

    step_results: Dict[Tuple[str, float], Any] = {}
    tasks = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_qa_worker,
        initargs=(str(flow.artifact_directory),),
    ) as executor:
        for name, model in models.items():
            tasks.append(
                executor.submit(_run_case_worker, task_payload(name, model, coarse_step_deg))
            )
            tasks.append(
                executor.submit(_run_case_worker, task_payload(name, model, fine_step_deg))
            )
        for future in as_completed(tasks):
            name, step, result = future.result()
            step_results[(name, step)] = result

    convergence_rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        convergence_rows.extend(
            _compare_steps(
                name,
                model,
                step_results[(name, coarse_step_deg)],
                step_results[(name, fine_step_deg)],
                coarse_step_deg,
                fine_step_deg,
                start_s,
                end_s,
                rmse_tolerance,
                maximum_tolerance,
            )
        )
    first_pair_passes = all(bool(row["passes"]) for row in convergence_rows)
    production_step = coarse_step_deg if first_pair_passes else fine_step_deg
    second_pair_evaluated = False
    if not first_pair_passes:
        failed_rotors = sorted(
            {str(row["rotor_name"]) for row in convergence_rows if not bool(row["passes"])}
        )
        second_pair_evaluated = True
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_qa_worker,
            initargs=(str(flow.artifact_directory),),
        ) as executor:
            futures = {
                executor.submit(
                    _run_case_worker, task_payload(name, models[name], finer_step_deg)
                ): name
                for name in failed_rotors
            }
            for future in as_completed(futures):
                name, step, result = future.result()
                step_results[(name, step)] = result
        second_rows: List[Dict[str, Any]] = []
        for name in failed_rotors:
            second_rows.extend(
                _compare_steps(
                    name,
                    models[name],
                    step_results[(name, fine_step_deg)],
                    step_results[(name, finer_step_deg)],
                    fine_step_deg,
                    finer_step_deg,
                    start_s,
                    end_s,
                    rmse_tolerance,
                    maximum_tolerance,
                )
            )
        convergence_rows.extend(second_rows)
        if not all(bool(row["passes"]) for row in second_rows):
            production_step = finer_step_deg

    baseline_results = {
        name: step_results[(name, production_step)]
        for name in models
        if (name, production_step) in step_results
    }
    # If only failed representatives were refined, retain the finest already
    # accepted result for every other rotor as the baseline at the global step.
    missing_baselines = [name for name in models if name not in baseline_results]
    if missing_baselines:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_qa_worker,
            initargs=(str(flow.artifact_directory),),
        ) as executor:
            futures = {
                executor.submit(
                    _run_case_worker, task_payload(name, models[name], production_step)
                ): name
                for name in missing_baselines
            }
            for future in as_completed(futures):
                name, _, result = future.result()
                baseline_results[name] = result

    sensitivity_specs = [
        (group, scale)
        for group in ("roll_pitch_inertia", "airframe_damping", "hover_hold_gains")
        for scale in (0.8, 1.2)
    ]
    sensitivity_results: Dict[Tuple[str, str, float], Any] = {}
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_qa_worker,
        initargs=(str(flow.artifact_directory),),
    ) as executor:
        futures = {}
        for name, model in models.items():
            for group, scale in sensitivity_specs:
                variant = _variant_model(model, group, scale)
                future = executor.submit(
                    _run_case_worker, task_payload(name, variant, production_step)
                )
                futures[future] = (name, group, scale)
        for future in as_completed(futures):
            name, group, scale = futures[future]
            _, _, result = future.result()
            sensitivity_results[(name, group, scale)] = result
    sensitivity_rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        for group, scale in sensitivity_specs:
            sensitivity_rows.extend(
                _summarize_sensitivity(
                    name,
                    model,
                    baseline_results[name],
                    sensitivity_results[(name, group, scale)],
                    group,
                    scale,
                    start_s,
                    end_s,
                )
            )

    qa_dir = context.output_dir / "09_qa"
    convergence_path = qa_dir / "time_step_convergence.csv"
    sensitivity_path = qa_dir / "parameter_sensitivity.csv"
    _write_rows(convergence_path, convergence_rows)
    _write_rows(sensitivity_path, sensitivity_rows)
    convergence_frame = pd.DataFrame(convergence_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    figure_files = _plot_qa(convergence_frame, sensitivity_frame, qa_dir, formats, dpi)
    if math.isclose(production_step, coarse_step_deg, rel_tol=0.0, abs_tol=1.0e-12):
        final_pair = convergence_frame[
            convergence_frame["coarse_maximum_azimuth_step_deg"] == coarse_step_deg
        ]
        unresolved = final_pair.empty or not bool(final_pair["passes"].all())
    elif math.isclose(production_step, fine_step_deg, rel_tol=0.0, abs_tol=1.0e-12):
        final_pair = convergence_frame[
            convergence_frame["coarse_maximum_azimuth_step_deg"] == fine_step_deg
        ]
        unresolved = final_pair.empty or not bool(final_pair["passes"].all())
    else:
        # The finest requested step has not itself been compared with a still
        # finer reference, so the production gate must remain unresolved.
        final_pair = convergence_frame.iloc[0:0]
        unresolved = True
    coupling_run_path = context.output_dir / "00_provenance" / "coupling_run.json"
    production_signature = None
    representative_case_signatures: Dict[str, str] = {}
    if coupling_run_path.is_file():
        coupling_run = _read_json(coupling_run_path)
        candidate = coupling_run.get("run_signature_sha256")
        if isinstance(candidate, str) and candidate:
            production_signature = candidate
        for name in models:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_") or "rotor"
            metadata_path = (
                context.output_dir
                / "04_timeseries_csv"
                / (safe + "__" + definition.case_id + ".metadata.json")
            )
            if metadata_path.is_file():
                representative_case_signatures[name] = str(
                    _read_json(metadata_path).get("run_signature_sha256", "")
                )
    production_signature_bound = bool(
        production_signature
        and len(representative_case_signatures) == len(models)
        and set(representative_case_signatures.values()) == {production_signature}
    )
    summary = {
        "status": "pass" if not unresolved else "unresolved",
        "representative_rotors": list(models.keys()),
        "full_rvmd_case": definition.case_id,
        "analysis_window_s": [start_s, end_s],
        "sample_interval_s": context.dt_s,
        "coarse_step_deg": coarse_step_deg,
        "fine_step_deg": fine_step_deg,
        "finer_step_deg": finer_step_deg,
        "second_pair_evaluated": second_pair_evaluated,
        "rmse_normalized_tolerance": rmse_tolerance,
        "maximum_abs_error_normalized_tolerance": maximum_tolerance,
        "recommended_production_maximum_azimuth_step_deg": production_step,
        "production_time_step_gate_passed": not unresolved,
        "production_run_signature_sha256": production_signature,
        "production_signature_bound": production_signature_bound,
        "representative_case_run_signatures": representative_case_signatures,
        "sensitivity_scales": [0.8, 1.2],
        "sensitivity_groups": [item[0] for item in sensitivity_specs[::2]],
        "statistical_interpretation": (
            "deterministic numerical and local parameter sensitivity; no confidence "
            "interval or population-level inference"
        ),
        "scope_limitations": [
            "step convergence covers the full energy-selected RVMD input, not the original-CFD residual",
            "three representative rotor diameters are tested rather than every production case",
            "two accepted step levels do not constitute an observed-order or GCI study",
            "parameter sensitivity is local to plus/minus 20 percent perturbations",
        ],
        "outputs": {
            "time_step_convergence_csv": str(convergence_path),
            "parameter_sensitivity_csv": str(sensitivity_path),
            "figure_files": figure_files,
        },
    }
    _write_json(qa_dir / "numerical_qa_summary.json", summary)
    flow.close()
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--coarse-step-deg", type=float, default=5.0)
    parser.add_argument("--fine-step-deg", type=float, default=2.5)
    parser.add_argument("--finer-step-deg", type=float, default=1.25)
    parser.add_argument("--rmse-tolerance", type=float, default=0.02)
    parser.add_argument("--maximum-tolerance", type=float, default=0.05)
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "tiff", "png"])
    parser.add_argument("--dpi", type=int, default=QA_RASTER_DPI)
    args = parser.parse_args(argv)
    result = run_qa(
        Path(args.config),
        workers=args.workers,
        coarse_step_deg=args.coarse_step_deg,
        fine_step_deg=args.fine_step_deg,
        finer_step_deg=args.finer_step_deg,
        rmse_tolerance=args.rmse_tolerance,
        maximum_tolerance=args.maximum_tolerance,
        formats=args.formats,
        dpi=args.dpi,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["production_time_step_gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["NumericalQAError", "run_qa"]
