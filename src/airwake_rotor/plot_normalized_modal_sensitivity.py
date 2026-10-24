#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Plot the normalized modal-sensitivity evidence for the rotor-size hypothesis.

The primary plot is St90 versus rotor diameter for each response variable.  A
negative power-law exponent supports the hypothesis that smaller rotors retain
sensitivity to higher-frequency wake content.  Full spectra and rotor-scaled
StD90 are retained so a single peak or a visually convenient cutoff cannot
drive the conclusion by itself.
"""

from __future__ import annotations

from .branding import save_figure

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Publication export contract: keep text editable in SVG/PDF and use a
# deterministic sans-serif fallback chain on Windows and headless runners.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
EXPORT_DPI = 600


METRIC_LABELS = {
    "lift_N": "Lift",
    "engine_power_W": "Engine power",
    "roll_deg": "Roll",
    "pitch_deg": "Pitch",
    "heave_m": "Heave",
}
COLORS = {
    "lift_N": "#0072B2",
    "engine_power_W": "#D55E00",
    "roll_deg": "#009E73",
    "pitch_deg": "#CC79A7",
    "heave_m": "#E69F00",
}


class SensitivityPlotError(ValueError):
    pass


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SensitivityPlotError("{} misses columns {}".format(label, missing))


def _power_law_fit(diameter: np.ndarray, value: np.ndarray) -> Dict[str, float]:
    mask = (
        np.isfinite(diameter)
        & np.isfinite(value)
        & (diameter > 0.0)
        & (value > 0.0)
    )
    x = np.log(diameter[mask])
    y = np.log(value[mask])
    if x.size < 3 or np.unique(x).size < 3:
        return {
            "sample_count": float(x.size),
            "coefficient_C": math.nan,
            "beta": math.nan,
            "r_squared_log_space": math.nan,
            "loocv_rmse_log_space": math.nan,
        }
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    residual = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else math.nan
    loo_errors: List[float] = []
    for held_out in range(x.size):
        keep = np.arange(x.size) != held_out
        loo_slope, loo_intercept = np.polyfit(x[keep], y[keep], 1)
        loo_errors.append(float(y[held_out] - (loo_intercept + loo_slope * x[held_out])))
    return {
        "sample_count": float(x.size),
        "coefficient_C": float(math.exp(intercept)),
        "beta": float(-slope),
        "r_squared_log_space": float(r_squared),
        "loocv_rmse_log_space": float(math.sqrt(np.mean(np.square(loo_errors)))),
    }


def _save_figure(figure: plt.Figure, stem: Path, dpi: int) -> Dict[str, str]:
    raster_dpi = max(int(dpi), EXPORT_DPI)
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    save_figure(figure, paths["svg"], bbox_inches="tight")
    save_figure(figure, paths["pdf"], bbox_inches="tight")
    save_figure(figure, paths["png"], dpi=raster_dpi, bbox_inches="tight")
    save_figure(figure, paths["tiff"], dpi=raster_dpi, bbox_inches="tight")
    outputs = {name: str(path) for name, path in paths.items()}
    plt.close(figure)
    return outputs


def plot_normalized_modal_sensitivity(
    mode_sensitivity_csv: Path,
    diameter_metrics_csv: Path,
    output_directory: Path,
    *,
    dpi: int = 300,
) -> Dict[str, Any]:
    mode_path = Path(mode_sensitivity_csv).resolve()
    diameter_path = Path(diameter_metrics_csv).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode = pd.read_csv(mode_path, encoding="utf-8-sig")
    diameter = pd.read_csv(diameter_path, encoding="utf-8-sig")
    _require_columns(
        mode,
        (
            "rotor_name",
            "rotor_diameter_m",
            "reference_length_m",
            "metric",
            "St_reference_length",
            "sensitivity_L1_primary",
            "response_sensitivity_L1_primary",
            "interaction_index",
        ),
        "mode sensitivity table",
    )
    _require_columns(
        diameter,
        (
            "rotor_name",
            "rotor_diameter_m",
            "reference_length_m",
            "metric",
            "estimator",
            "is_primary_estimator",
            "St90",
            "StD90",
            "St_g",
            "response_St90",
            "response_StD90",
            "response_St_g",
        ),
        "diameter metrics table",
    )
    if mode.empty or diameter.empty:
        raise SensitivityPlotError("sensitivity tables must not be empty")
    for column in (
        "rotor_diameter_m",
        "St_reference_length",
        "sensitivity_L1_primary",
        "response_sensitivity_L1_primary",
        "interaction_index",
    ):
        mode[column] = pd.to_numeric(mode[column], errors="raise")
    for column in (
        "rotor_diameter_m",
        "reference_length_m",
        "St90",
        "StD90",
        "St_g",
        "response_St90",
        "response_StD90",
        "response_St_g",
    ):
        diameter[column] = pd.to_numeric(diameter[column], errors="raise")
    primary_flag = diameter["is_primary_estimator"].astype(str).str.lower().isin(("true", "1"))
    primary = diameter.loc[primary_flag].copy()
    if primary.empty:
        raise SensitivityPlotError("diameter table has no primary-estimator rows")
    metrics = [name for name in METRIC_LABELS if name in set(mode["metric"])]
    metrics += sorted(set(mode["metric"]) - set(metrics))
    if not metrics:
        raise SensitivityPlotError("no response metrics are available")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )
    outputs: Dict[str, Mapping[str, str]] = {}

    figure, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 2.05 * len(metrics)), sharex=True)
    axes_array = np.atleast_1d(axes)
    all_diameters = sorted(mode["rotor_diameter_m"].unique())
    color_map = plt.get_cmap("viridis")
    diameter_min, diameter_max = min(all_diameters), max(all_diameters)
    span = max(diameter_max - diameter_min, np.finfo(float).eps)
    for axis, metric_name in zip(axes_array, metrics):
        subset = mode.loc[mode["metric"] == metric_name]
        for (rotor_name, rotor_diameter), group in subset.groupby(
            ["rotor_name", "rotor_diameter_m"], sort=False
        ):
            ordered = group.sort_values("St_reference_length")
            color = color_map((float(rotor_diameter) - diameter_min) / span)
            axis.plot(
                ordered["St_reference_length"],
                ordered["sensitivity_L1_primary"],
                color=color,
                linewidth=1.15,
                alpha=0.9,
                label="{} ({:.2f} m)".format(rotor_name, rotor_diameter),
            )
        axis.set_xscale("log")
        axis.set_ylabel("Normalized\nsensitivity")
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name), loc="left", fontweight="bold")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.35)
    axes_array[-1].set_xlabel(r"Ship-scale center-frequency $St_L=fL/U$")
    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, 0.975))
    figure.suptitle("Unit-modal-energy normalized sensitivity spectra", y=0.999)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    outputs["normalized_sensitivity_spectra"] = _save_figure(
        figure, output / "normalized_sensitivity_spectra", dpi
    )

    figure, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 2.05 * len(metrics)), sharex=True)
    axes_array = np.atleast_1d(axes)
    for axis, metric_name in zip(axes_array, metrics):
        subset = mode.loc[mode["metric"] == metric_name]
        for (rotor_name, rotor_diameter), group in subset.groupby(
            ["rotor_name", "rotor_diameter_m"], sort=False
        ):
            ordered = group.sort_values("St_reference_length")
            color = color_map((float(rotor_diameter) - diameter_min) / span)
            axis.plot(
                ordered["St_reference_length"],
                ordered["response_sensitivity_L1_primary"],
                color=color,
                linewidth=1.15,
                alpha=0.9,
                label="{} ({:.2f} m)".format(rotor_name, rotor_diameter),
            )
        axis.set_xscale("log")
        axis.set_ylabel("Normalized actual\nresponse share")
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name), loc="left", fontweight="bold")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.35)
    axes_array[-1].set_xlabel(r"Ship-scale center-frequency $St_L=fL/U$")
    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, 0.975))
    figure.suptitle("Actual NATO-GD modal-response contribution spectra", y=0.999)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    outputs["actual_response_spectra"] = _save_figure(
        figure, output / "actual_response_spectra", dpi
    )

    fit_rows: List[Dict[str, Any]] = []
    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    for metric_name in metrics:
        subset = primary.loc[primary["metric"] == metric_name].sort_values("rotor_diameter_m")
        if subset.empty:
            continue
        x = subset["rotor_diameter_m"].to_numpy(dtype=float)
        y = subset["St90"].to_numpy(dtype=float)
        reference_length = float(subset["reference_length_m"].iloc[0])
        fit = _power_law_fit(x / reference_length, y)
        fit_rows.append(
            {
                "metric": metric_name,
                "estimator": str(subset["estimator"].iloc[0]),
                "spectrum_basis": "unit_modal_energy_gain_G",
                "reference_length_m": reference_length,
                **fit,
            }
        )
        color = COLORS.get(metric_name)
        label = METRIC_LABELS.get(metric_name, metric_name)
        axis.plot(x, y, marker="o", markersize=4.5, linewidth=1.25, color=color, label=label)
        if math.isfinite(fit["beta"]):
            fit_x = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
            fit_y = fit["coefficient_C"] * (fit_x / reference_length) ** (-fit["beta"])
            axis.plot(fit_x, fit_y, linestyle="--", linewidth=0.9, color=color, alpha=0.8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Rotor diameter D (m)")
    axis.set_ylabel(r"Effective upper sensitive band $St_{90}$")
    axis.set_title("Rotor size versus effective disturbance-frequency range")
    axis.grid(True, which="both", linewidth=0.4, alpha=0.4)
    axis.legend(ncol=2)
    figure.tight_layout()
    outputs["diameter_vs_St90"] = _save_figure(figure, output / "diameter_vs_St90", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True, sharey=True)
    comparison_specs = (
        ("response_St90", "Actual response contribution R"),
        ("St90", "Unit-modal-energy gain G"),
    )
    for axis, (column, title) in zip(axes, comparison_specs):
        for metric_name in metrics:
            subset = primary.loc[primary["metric"] == metric_name].sort_values(
                "rotor_diameter_m"
            )
            if subset.empty:
                continue
            x = subset["rotor_diameter_m"].to_numpy(dtype=float)
            y = subset[column].to_numpy(dtype=float)
            reference_length = float(subset["reference_length_m"].iloc[0])
            fit = _power_law_fit(x / reference_length, y)
            if column == "response_St90":
                fit_rows.append(
                    {
                        "metric": metric_name,
                        "estimator": str(subset["estimator"].iloc[0]),
                        "spectrum_basis": "actual_response_contribution_R",
                        "reference_length_m": reference_length,
                        **fit,
                    }
                )
            color = COLORS.get(metric_name)
            axis.plot(
                x,
                y,
                marker="o",
                markersize=4.2,
                linewidth=1.15,
                color=color,
                label=METRIC_LABELS.get(metric_name, metric_name),
            )
            if math.isfinite(fit["beta"]):
                fit_x = np.geomspace(float(np.min(x)), float(np.max(x)), 100)
                fit_y = fit["coefficient_C"] * (fit_x / reference_length) ** (-fit["beta"])
                axis.plot(fit_x, fit_y, linestyle="--", linewidth=0.8, color=color, alpha=0.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Rotor diameter D (m)")
        axis.set_title(title)
        axis.grid(True, which="both", linewidth=0.4, alpha=0.4)
    axes[0].set_ylabel(r"Effective upper sensitive band $St_{90}$")
    axes[1].legend(ncol=1, fontsize=8)
    figure.suptitle("Robustness of the rotor-size/frequency relationship to spectrum definition")
    figure.tight_layout()
    outputs["diameter_vs_St90_basis_comparison"] = _save_figure(
        figure, output / "diameter_vs_St90_basis_comparison", dpi
    )

    fit_table = pd.DataFrame(fit_rows)
    fit_path = output / "diameter_frequency_power_law_fits.csv"
    fit_table.to_csv(fit_path, index=False, encoding="utf-8-sig", float_format="%.15g")

    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    for metric_name in metrics:
        subset = primary.loc[primary["metric"] == metric_name].sort_values("rotor_diameter_m")
        if subset.empty:
            continue
        axis.plot(
            subset["rotor_diameter_m"],
            subset["StD90"],
            marker="o",
            linewidth=1.25,
            color=COLORS.get(metric_name),
            label=METRIC_LABELS.get(metric_name, metric_name),
        )
    axis.set_xlabel("Rotor diameter D (m)")
    axis.set_ylabel(r"Rotor-scaled upper band $St_{D,90}=f_{90}D/U$")
    axis.set_title("Similarity check: constancy of rotor-scaled sensitive frequency")
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.legend(ncol=2)
    figure.tight_layout()
    outputs["diameter_vs_StD90"] = _save_figure(figure, output / "diameter_vs_StD90", dpi)

    audit = {
        "status": "pass",
        "mode_sensitivity_csv": str(mode_path),
        "diameter_metrics_csv": str(diameter_path),
        "output_directory": str(output),
        "primary_hypothesis_test": (
            "beta>0 in St90=C*(D/Lref)^(-beta), checked separately for actual "
            "response contribution R and unit-modal-energy gain G"
        ),
        "required_interpretation": (
            "R describes engineering contribution in this NATO-GD wake; G describes the "
            "rotor transfer tendency per unit modal energy; compare both before claiming a law"
        ),
        "causal_scope": (
            "seven aircraft configurations are descriptive and confounded by mass, inertia, rpm, "
            "disk loading, solidity, and controller parameters; a similitude-controlled diameter "
            "sweep is required for a pure size-effect claim"
        ),
        "figures": outputs,
        "power_law_fit_table": str(fit_path),
    }
    audit_path = output / "normalized_sensitivity_figure_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-sensitivity", required=True, type=Path)
    parser.add_argument("--diameter-metrics", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    audit = plot_normalized_modal_sensitivity(
        args.mode_sensitivity,
        args.diameter_metrics,
        args.output_directory,
        dpi=args.dpi,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SensitivityPlotError", "plot_normalized_modal_sensitivity"]
