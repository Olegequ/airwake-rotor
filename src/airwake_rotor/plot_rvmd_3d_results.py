#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
# -*- coding: utf-8 -*-
"""Create publication-grade figures and a Chinese figure manual for NATO-GD RVMD results."""

from __future__ import annotations

from .branding import save_figure

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

from .nato_gd_study import load_study_context
from .finalize_time_weighted_statistics import boundary_augmented_window


COLORS = ["#0F4D92", "#D55E00", "#009E73", "#7A5195", "#56B4E9", "#B64342", "#595959"]
DEFAULT_RASTER_DPI = 600
FIGURE_WIDTH_MM = 183.0
CLOSURE_STRONG_R = 0.90
CLOSURE_STRONG_NRMSE = 0.30
CLOSURE_TREND_R = 0.80
CLOSURE_TREND_NRMSE = 0.60
METRICS = ["lift_N", "engine_power_W", "roll_deg", "pitch_deg", "heave_m"]
METRIC_LABELS = {
    "lift_N": "Vertical lift (N)",
    "engine_power_W": "Engine power (W)",
    "roll_deg": "Roll (deg)",
    "pitch_deg": "Pitch (deg)",
    "heave_m": "Heave (m)",
}
SHORT_LABELS = {
    "lift_N": "Lift",
    "engine_power_W": "Power",
    "roll_deg": "Roll",
    "pitch_deg": "Pitch",
    "heave_m": "Heave",
}


class ResultPlotError(ValueError):
    pass


def _mode_count_from_frequency(frequency: pd.DataFrame) -> int:
    mode_count = int(len(frequency))
    if mode_count < 1:
        raise ResultPlotError("frequency table is empty")
    expected = np.arange(1, mode_count + 1)
    if not np.array_equal(frequency["mode"].to_numpy(dtype=int), expected):
        raise ResultPlotError(
            "frequency table must contain modes 1..{} exactly once".format(mode_count)
        )
    return mode_count


def _mode_count_from_statistics(statistics: pd.DataFrame) -> int:
    stage_numeric = pd.to_numeric(statistics["stage_index"], errors="coerce")
    selected = statistics[
        (statistics["flow_kind"] == "RVMD") & stage_numeric.notna()
    ]
    if selected.empty:
        raise ResultPlotError("RVMD statistics are empty")
    retained = pd.to_numeric(selected["retained_mode_count"], errors="coerce")
    if retained.isna().any():
        raise ResultPlotError("RVMD retained_mode_count contains missing/non-numeric values")
    mode_count = int(retained.max())
    if mode_count < 1 or not np.isclose(float(retained.max()), mode_count):
        raise ResultPlotError("invalid RVMD mode count in statistics")
    return mode_count


def _stage_case_id(stage: int, retained: int, mode_count: int) -> str:
    width = max(2, len(str(int(mode_count))))
    return "stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
        stage=int(stage), retained=int(retained), width=width
    )


def _mode_ticks(mode_count: int) -> np.ndarray:
    return np.unique(
        np.rint(np.linspace(0, int(mode_count), min(7, int(mode_count) + 1))).astype(int)
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ResultPlotError("JSON root must be an object: {}".format(path))
    return value


def _required_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ResultPlotError("{} is missing columns: {}".format(label, ", ".join(missing)))


def _numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.all(np.isfinite(frame[column].to_numpy(dtype=float))):
            raise ResultPlotError("{} column {} contains missing/non-finite values".format(label, column))


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.11, 1.03, label, transform=axis.transAxes, fontsize=8.0, fontweight="bold", va="bottom")


def _save_figure(
    figure: plt.Figure,
    base: Path,
    formats: Sequence[str],
    dpi: int,
    *,
    apply_tight_layout: bool = True,
) -> List[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    if apply_tight_layout:
        figure.tight_layout(pad=0.8)
    outputs = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt == "svg":
            path = base.with_suffix(".svg")
        elif fmt == "pdf":
            path = base.with_suffix(".pdf")
        elif fmt in ("tif", "tiff"):
            path = base.with_suffix(".tiff")
        elif fmt == "png":
            path = base.with_suffix(".png")
        else:
            raise ResultPlotError("unsupported figure format: {}".format(fmt))
        save_figure(figure, str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def _load_inputs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    frequency_path = root / "02_rvmd" / "frequencies.csv"
    statistics_path = root / "05_statistics" / "statistics_long.csv"
    trim_path = root / "03_hover_trim" / "hover_trim_all_rotors.csv"
    metadata_path = root / "02_rvmd" / "metadata.json"
    context_path = root / "00_provenance" / "study_context.json"
    for path in (frequency_path, statistics_path, trim_path, metadata_path, context_path):
        if not path.is_file():
            raise ResultPlotError("required result is missing: {}".format(path))
    frequency = pd.read_csv(str(frequency_path), encoding="utf-8-sig")
    statistics = pd.read_csv(str(statistics_path), encoding="utf-8-sig")
    trim = pd.read_csv(str(trim_path), encoding="utf-8-sig")
    metadata = _read_json(metadata_path)
    context = _read_json(context_path)
    _required_columns(
        frequency,
        ["mode", "frequency_hz", "strouhal_St_f13.716_over_20", "temporal_coefficient_energy"],
        "frequencies.csv",
    )
    _required_columns(
        statistics,
        [
            "rotor_name", "rotor_diameter_m", "case_id", "flow_kind", "stage_index",
            "retained_mode_count", "metric", "statistic", "value",
        ],
        "statistics_long.csv",
    )
    _required_columns(
        trim,
        ["rotor_name", "weight_N", "trim_engine_power_W", "collective_trim_deg", "induced_velocity_trim_mps"],
        "hover_trim_all_rotors.csv",
    )
    _numeric(frequency, ["mode", "frequency_hz", "strouhal_St_f13.716_over_20", "temporal_coefficient_energy"], "frequencies.csv")
    _mode_count_from_frequency(frequency)
    if np.any(np.diff(frequency["frequency_hz"].to_numpy(dtype=float)) < 0.0):
        raise ResultPlotError("frequency table is not sorted low to high")
    _numeric(statistics, ["rotor_diameter_m", "value"], "statistics_long.csv")
    _numeric(trim, ["weight_N", "trim_engine_power_W", "collective_trim_deg", "induced_velocity_trim_mps"], "hover trim")
    return frequency, statistics, trim, metadata, context


def _modal_bandwidths(rvmd_dir: Path) -> np.ndarray:
    with np.load(str(rvmd_dir / "temporal_model.npz"), allow_pickle=False) as data:
        spectrum = np.asarray(data["coefficient_spectrum"], dtype=np.complex128)
        frequency_hz = np.asarray(data["frequency_hz"], dtype=float)
        info = json.loads(str(np.asarray(data["info_json"]).item()))
    n_time = int(info["n_time"])
    dt = float(info["dt_seconds"])
    bins = np.arange(n_time + 1, dtype=float) / (2.0 * n_time * dt)
    weights = np.ones(n_time + 1, dtype=float)
    weights[[0, -1]] = 0.5
    power = weights[:, None] * np.abs(spectrum) ** 2
    denominator = np.sum(power, axis=0)
    if np.any(denominator <= 0.0):
        raise ResultPlotError("a temporal coefficient has zero spectral energy")
    return np.sqrt(np.sum(power * (bins[:, None] - frequency_hz[None, :]) ** 2, axis=0) / denominator)


def plot_rvmd_summary(
    root: Path,
    frequency: pd.DataFrame,
    metadata: Mapping[str, Any],
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    mode = frequency["mode"].to_numpy(dtype=int)
    hz = frequency["frequency_hz"].to_numpy(dtype=float)
    st = frequency["strouhal_St_f13.716_over_20"].to_numpy(dtype=float)
    energy_column = (
        "full_normalized_modal_self_energy"
        if "full_normalized_modal_self_energy" in frequency.columns
        else "temporal_coefficient_energy"
    )
    self_energy = pd.to_numeric(frequency[energy_column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(self_energy)) or np.any(self_energy < 0.0):
        raise ResultPlotError("modal self-energy column is invalid")
    share = 100.0 * self_energy / max(float(np.sum(self_energy)), np.finfo(float).tiny)
    bandwidth = _modal_bandwidths(root / "02_rvmd")
    audit = metadata.get("full_space_regression_audit", {})
    pod = metadata.get("pod_projection", {})
    identification = metadata.get("rvmd_identification_space_audit", {})
    closure_labels = ["POD\nretained", "RVMD sketch\ncaptured", "Full field\ncaptured"]
    closure_values = [
        float(pod.get("retained_energy_fraction", np.nan)),
        float(identification.get("captured_energy_fraction_one_minus_residual", np.nan)),
        float(audit.get("full_normalized_captured_energy_fraction_one_minus_residual", np.nan)),
    ]
    if not np.all(np.isfinite(closure_values)):
        raise ResultPlotError("RVMD reconstruction audit is incomplete")

    figure, axes = plt.subplots(2, 2, figsize=(183.0 / 25.4, 118.0 / 25.4))
    axis = axes[0, 0]
    axis.plot(mode, hz, color=COLORS[0], marker="o", ms=2.8, lw=1.0)
    axis.set_xlabel("Mode (ascending centre frequency)")
    axis.set_ylabel("Centre frequency (Hz)")
    secondary = axis.secondary_yaxis(
        "right", functions=(lambda value: value * 13.716 / 20.0, lambda value: value * 20.0 / 13.716)
    )
    secondary.set_ylabel("Strouhal number, St")
    mode_count = _mode_count_from_frequency(frequency)
    axis.set_xlim(0.5, mode_count + 0.5)
    _panel_label(axis, "a")

    axis = axes[0, 1]
    axis.bar(mode, share, color=COLORS[1], width=0.78, edgecolor="none")
    axis.set_xlabel("Mode")
    axis.set_ylabel("Modal self-energy share (%)")
    axis.set_xlim(0.5, mode_count + 0.5)
    axis.text(
        0.02, 0.96, "Self-energy is non-additive when modes are non-orthogonal",
        transform=axis.transAxes, ha="left", va="top", fontsize=5.5,
    )
    _panel_label(axis, "b")

    axis = axes[1, 0]
    sizes = 10.0 + 90.0 * share / max(float(np.max(share)), np.finfo(float).tiny)
    axis.scatter(hz, bandwidth, s=sizes, color=COLORS[2], alpha=0.82, edgecolors="white", linewidths=0.3)
    for index in np.argsort(share)[-3:]:
        axis.annotate(str(mode[index]), (hz[index], bandwidth[index]), xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    axis.set_xlabel("Centre frequency (Hz)")
    axis.set_ylabel("Spectral bandwidth (Hz)")
    _panel_label(axis, "c")

    axis = axes[1, 1]
    percentages = 100.0 * np.asarray(closure_values)
    axis.bar(np.arange(3), percentages, color=[COLORS[4], COLORS[2], COLORS[0]], width=0.66)
    axis.set_xticks(np.arange(3))
    axis.set_xticklabels(closure_labels)
    axis.set_ylabel("Energy fraction (%)")
    axis.set_ylim(min(0.0, float(np.min(percentages)) - 5.0), max(100.0, float(np.max(percentages)) + 5.0))
    for x_value, value in enumerate(percentages):
        axis.text(x_value, value, "{:.2f}".format(value), ha="center", va="bottom", fontsize=5.5)
    _panel_label(axis, "d")
    return _save_figure(figure, output / "figure_01_rvmd_frequency_energy_audit", formats, dpi)


def _rotor_order(statistics: pd.DataFrame) -> List[str]:
    table = statistics[["rotor_name", "rotor_diameter_m"]].drop_duplicates()
    if table.groupby("rotor_name")["rotor_diameter_m"].nunique().max() != 1:
        raise ResultPlotError("one rotor name maps to multiple diameters")
    return table.sort_values("rotor_diameter_m", ascending=False)["rotor_name"].tolist()


def _rvmd_stat(statistics: pd.DataFrame, statistic: str) -> pd.DataFrame:
    stage_numeric = pd.to_numeric(statistics["stage_index"], errors="coerce")
    selected = statistics[
        (statistics["flow_kind"] == "RVMD")
        & (statistics["statistic"] == statistic)
        & stage_numeric.notna()
    ].copy()
    _numeric(selected, ["stage_index", "retained_mode_count", "value"], "RVMD statistics")
    if selected.empty:
        raise ResultPlotError("RVMD {} table is empty".format(statistic))
    mode_count = _mode_count_from_statistics(statistics)
    expected = len(selected["rotor_name"].unique()) * (mode_count + 1) * len(METRICS)
    if len(selected) != expected:
        raise ResultPlotError("RVMD {} table has {} rows; expected {}".format(statistic, len(selected), expected))
    key = ["rotor_name", "metric", "stage_index"]
    if selected.duplicated(key).any():
        raise ResultPlotError("RVMD {} table contains duplicate rotor/metric/stage rows".format(statistic))
    return selected


def plot_response_means(
    statistics: pd.DataFrame,
    trim: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    selected = _rvmd_stat(statistics, "mean")
    mode_count = _mode_count_from_statistics(statistics)
    rotors = _rotor_order(statistics)
    trim_index = trim.set_index("rotor_name")
    source_rows = []
    figure, axes = plt.subplots(2, 3, figsize=(183.0 / 25.4, 116.0 / 25.4), sharex=True)
    axes_flat = list(axes.flat)
    for metric_index, metric in enumerate(METRICS):
        axis = axes_flat[metric_index]
        for rotor_index, rotor in enumerate(rotors):
            values = selected[(selected["rotor_name"] == rotor) & (selected["metric"] == metric)].sort_values("retained_mode_count")
            x = values["retained_mode_count"].to_numpy(dtype=float)
            y = values["value"].to_numpy(dtype=float)
            if metric == "lift_N":
                reference = float(trim_index.loc[rotor, "weight_N"])
                display = 100.0 * (y - y[-1]) / reference
                ylabel = "Change from {}-mode lift (% weight)".format(mode_count)
            elif metric == "engine_power_W":
                reference = abs(float(trim_index.loc[rotor, "trim_engine_power_W"]))
                display = 100.0 * (y - y[-1]) / max(reference, np.finfo(float).tiny)
                ylabel = "Change from {}-mode power (% trim)".format(mode_count)
            else:
                display = y - y[-1]
                ylabel = "Change from {}-mode {} ({})".format(
                    mode_count, SHORT_LABELS[metric].lower(), "deg" if metric.endswith("deg") else "m"
                )
            axis.plot(x, display, color=COLORS[rotor_index], lw=1.0, marker="o", ms=2.0, label=rotor)
            for retained, raw, shown in zip(x, y, display):
                source_rows.append(
                    {
                        "rotor_name": rotor,
                        "metric": metric,
                        "retained_mode_count": int(retained),
                        "raw_time_mean": raw,
                        "display_change_from_full_mode_set": shown,
                        "display_normalization": ylabel,
                    }
                )
        axis.axhline(0.0, color="#767676", lw=0.7, ls="--")
        axis.set_ylabel(ylabel)
        axis.set_title(SHORT_LABELS[metric], fontsize=7.0)
        axis.set_xlim(0, mode_count)
        axis.set_xticks(_mode_ticks(mode_count))
        _panel_label(axis, string.ascii_lowercase[metric_index])
    axes_flat[3].set_xlabel("Retained low-frequency modes")
    axes_flat[4].set_xlabel("Retained low-frequency modes")
    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, loc="center", ncol=1, fontsize=6.2)
    files = _save_figure(figure, output / "figure_02_response_means_vs_retained_modes", formats, dpi)
    return files, pd.DataFrame(source_rows)


def adjacent_effect_table(
    statistics: pd.DataFrame, trim: pd.DataFrame, statistic: str
) -> pd.DataFrame:
    selected = _rvmd_stat(statistics, statistic)
    trim_index = trim.set_index("rotor_name")
    mode_count = _mode_count_from_statistics(statistics)
    rows = []
    for rotor in _rotor_order(statistics):
        for metric in METRICS:
            values = selected[(selected["rotor_name"] == rotor) & (selected["metric"] == metric)].sort_values("stage_index")
            if not np.array_equal(values["stage_index"].to_numpy(dtype=int), np.arange(mode_count + 1)):
                raise ResultPlotError("missing or duplicate stages for {} {} {}".format(rotor, metric, statistic))
            raw = values["value"].to_numpy(dtype=float)
            delta = raw[1:] - raw[:-1]
            if metric == "lift_N":
                scale = float(trim_index.loc[rotor, "weight_N"])
                normalized = 100.0 * delta / scale
                unit = "% weight"
            elif metric == "engine_power_W":
                scale = max(abs(float(trim_index.loc[rotor, "trim_engine_power_W"])), np.finfo(float).tiny)
                normalized = 100.0 * delta / scale
                unit = "% trim power"
            else:
                scale = 1.0
                normalized = delta
                unit = "deg" if metric.endswith("deg") else "m"
            # Stage 1 deletes mode K, ... stage K deletes mode 1.
            for stage, raw_delta, display in zip(range(1, mode_count + 1), delta, normalized):
                rows.append(
                    {
                        "rotor_name": rotor,
                        "rotor_diameter_m": float(values["rotor_diameter_m"].iloc[0]),
                        "metric": metric,
                        "statistic": statistic,
                        "stage_index": stage,
                        "deleted_mode": mode_count + 1 - stage,
                        "raw_adjacent_effect": raw_delta,
                        "normalized_adjacent_effect": display,
                        "display_unit": unit,
                    }
                )
    return pd.DataFrame(rows)


def plot_adjacent_effect_heatmaps(
    effects: pd.DataFrame,
    frequency: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
    stem: str,
    title_suffix: str,
) -> List[str]:
    rotors = effects[["rotor_name", "rotor_diameter_m"]].drop_duplicates().sort_values("rotor_diameter_m", ascending=False)["rotor_name"].tolist()
    figure, axes = plt.subplots(
        len(METRICS),
        1,
        figsize=(183.0 / 25.4, 190.0 / 25.4),
        sharex=True,
    )
    st = frequency["strouhal_St_f13.716_over_20"].to_numpy(dtype=float)
    mode_count = _mode_count_from_frequency(frequency)
    tick_positions = np.unique(
        np.rint(np.linspace(0, mode_count - 1, min(7, mode_count))).astype(int)
    ).tolist()
    tick_labels = ["{}\n{:.3f}".format(index + 1, st[index]) for index in tick_positions]
    for metric_index, metric in enumerate(METRICS):
        axis = axes[metric_index]
        subset = effects[effects["metric"] == metric]
        matrix = np.empty((len(rotors), mode_count), dtype=float)
        for row_index, rotor in enumerate(rotors):
            row = subset[subset["rotor_name"] == rotor].sort_values("deleted_mode")
            if not np.array_equal(row["deleted_mode"].to_numpy(dtype=int), np.arange(1, mode_count + 1)):
                raise ResultPlotError("adjacent-effect modes are incomplete")
            matrix[row_index] = row["normalized_adjacent_effect"].to_numpy(dtype=float)
        limit = float(np.max(np.abs(matrix)))
        limit = max(limit, np.finfo(float).tiny)
        image = axis.imshow(matrix, aspect="auto", origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_yticks(np.arange(len(rotors)))
        axis.set_yticklabels(rotors)
        axis.set_xticks(tick_positions)
        unit = str(subset["display_unit"].iloc[0])
        axis.set_title("{} ({})".format(SHORT_LABELS[metric], unit), fontsize=7.0, loc="left")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.022, pad=0.02, aspect=20)
        colorbar.ax.tick_params(labelsize=5.5)
        _panel_label(axis, string.ascii_lowercase[metric_index])
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_xlabel(
        "Deleted mode (ascending centre frequency)\nTick labels: mode number (top), St (bottom)"
    )
    figure.suptitle(title_suffix, fontsize=8.0, y=1.01)
    return _save_figure(figure, output / stem, formats, dpi)


def plot_peak_to_peak(
    statistics: pd.DataFrame,
    trim: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    selected = _rvmd_stat(statistics, "peak_to_peak")
    mode_count = _mode_count_from_statistics(statistics)
    rotors = _rotor_order(statistics)
    trim_index = trim.set_index("rotor_name")
    rows = []
    figure, axes = plt.subplots(2, 3, figsize=(183.0 / 25.4, 116.0 / 25.4), sharex=True)
    for metric_index, metric in enumerate(METRICS):
        axis = axes.flat[metric_index]
        for rotor_index, rotor in enumerate(rotors):
            values = selected[(selected["rotor_name"] == rotor) & (selected["metric"] == metric)].sort_values("retained_mode_count")
            x = values["retained_mode_count"].to_numpy(dtype=float)
            raw = values["value"].to_numpy(dtype=float)
            if metric == "lift_N":
                shown = 100.0 * raw / float(trim_index.loc[rotor, "weight_N"])
                ylabel = "Peak-to-peak (% weight)"
            elif metric == "engine_power_W":
                shown = 100.0 * raw / max(abs(float(trim_index.loc[rotor, "trim_engine_power_W"])), np.finfo(float).tiny)
                ylabel = "Peak-to-peak (% trim power)"
            else:
                shown = raw
                ylabel = "Peak-to-peak ({})".format("deg" if metric.endswith("deg") else "m")
            axis.plot(x, shown, color=COLORS[rotor_index], lw=1.0, marker="o", ms=2.0, label=rotor)
            for retained, value, display in zip(x, raw, shown):
                rows.append({"rotor_name": rotor, "metric": metric, "retained_mode_count": int(retained), "raw_peak_to_peak": value, "display_peak_to_peak": display, "display_unit": ylabel})
        axis.set_title(SHORT_LABELS[metric], fontsize=7.0)
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, mode_count)
        axis.set_xticks(_mode_ticks(mode_count))
        _panel_label(axis, string.ascii_lowercase[metric_index])
    axes.flat[3].set_xlabel("Retained low-frequency modes")
    axes.flat[4].set_xlabel("Retained low-frequency modes")
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", fontsize=6.2)
    files = _save_figure(figure, output / "figure_04_response_peak_to_peak", formats, dpi)
    return files, pd.DataFrame(rows)


def dominant_mode_table(
    mean_effects: pd.DataFrame,
    variability_effects: pd.DataFrame,
    frequency: pd.DataFrame,
) -> pd.DataFrame:
    lookup = frequency.set_index("mode")
    rows = []
    for effect_kind, table in (("time_mean", mean_effects), ("peak_to_peak", variability_effects)):
        for (rotor, metric), group in table.groupby(["rotor_name", "metric"], sort=False):
            magnitudes = np.abs(group["normalized_adjacent_effect"].to_numpy(dtype=float))
            order = np.argsort(-magnitudes, kind="mergesort")
            if order.size < 2:
                raise ResultPlotError("dominant-mode ranking requires at least two candidates")
            chosen = group.iloc[int(order[0])]
            runner_up = group.iloc[int(order[1])]
            mode = int(chosen["deleted_mode"])
            runner_up_mode = int(runner_up["deleted_mode"])
            top_magnitude = float(magnitudes[int(order[0])])
            runner_up_magnitude = float(magnitudes[int(order[1])])
            dominance_ratio = (
                top_magnitude / runner_up_magnitude
                if runner_up_magnitude > np.finfo(float).eps
                else float("inf")
            )
            rows.append(
                {
                    "effect_kind": effect_kind,
                    "rotor_name": rotor,
                    "rotor_diameter_m": float(chosen["rotor_diameter_m"]),
                    "metric": metric,
                    "dominant_deleted_mode": mode,
                    "dominant_frequency_hz": float(lookup.loc[mode, "frequency_hz"]),
                    "dominant_St": float(lookup.loc[mode, "strouhal_St_f13.716_over_20"]),
                    "signed_normalized_adjacent_effect": float(chosen["normalized_adjacent_effect"]),
                    "absolute_normalized_adjacent_effect": abs(float(chosen["normalized_adjacent_effect"])),
                    "runner_up_deleted_mode": runner_up_mode,
                    "runner_up_frequency_hz": float(lookup.loc[runner_up_mode, "frequency_hz"]),
                    "runner_up_St": float(
                        lookup.loc[runner_up_mode, "strouhal_St_f13.716_over_20"]
                    ),
                    "runner_up_absolute_normalized_adjacent_effect": runner_up_magnitude,
                    "dominance_ratio_top_over_runner_up": dominance_ratio,
                    "dominance_relative_margin": (
                        (top_magnitude - runner_up_magnitude) / top_magnitude
                        if top_magnitude > np.finfo(float).eps
                        else 0.0
                    ),
                    "near_tie_top_over_runner_up_below_1p10": bool(dominance_ratio < 1.10),
                    "normalization_unit": chosen["display_unit"],
                    "selection_rule": (
                        "largest absolute adjacent cumulative-deletion change; near tie if "
                        "top/runner-up effect ratio < 1.10"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_dominant_modes(
    dominant: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    required = [
        "effect_kind",
        "rotor_name",
        "rotor_diameter_m",
        "metric",
        "dominant_St",
        "near_tie_top_over_runner_up_below_1p10",
    ]
    _required_columns(dominant, required, "dominant_modes.csv")
    rotors = (
        dominant[["rotor_name", "rotor_diameter_m"]]
        .drop_duplicates()
        .sort_values("rotor_diameter_m")
    )
    expected = len(rotors) * len(METRICS) * 2
    if len(rotors) < 1 or len(dominant) != expected:
        raise ResultPlotError(
            "dominant-mode table must contain one row per rotor x metric x effect"
        )
    rotor_markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    marker_lookup = {
        str(row.rotor_name): rotor_markers[index]
        for index, row in enumerate(rotors.itertuples(index=False))
    }
    figure, axes = plt.subplots(
        len(METRICS),
        2,
        figsize=(183.0 / 25.4, 175.0 / 25.4),
        sharex=True,
        sharey=True,
    )
    effect_columns = (("time_mean", "Time-weighted mean shift"), ("peak_to_peak", "Peak-to-peak change"))
    for metric_index, metric in enumerate(METRICS):
        for column_index, (effect_kind, column_title) in enumerate(effect_columns):
            axis = axes[metric_index, column_index]
            values = dominant[
                (dominant["effect_kind"] == effect_kind) & (dominant["metric"] == metric)
            ].sort_values("rotor_diameter_m")
            if len(values) != len(rotors):
                raise ResultPlotError("dominant-mode curve is incomplete")
            x = values["rotor_diameter_m"].to_numpy(dtype=float)
            y = values["dominant_St"].to_numpy(dtype=float)
            axis.plot(x, y, color=COLORS[metric_index], lw=0.9, zorder=1)
            for row in values.itertuples(index=False):
                near_tie = bool(row.near_tie_top_over_runner_up_below_1p10)
                axis.scatter(
                    float(row.rotor_diameter_m),
                    float(row.dominant_St),
                    marker=marker_lookup[str(row.rotor_name)],
                    s=19.0,
                    facecolor="white" if near_tie else COLORS[metric_index],
                    edgecolor=COLORS[metric_index],
                    linewidth=0.8,
                    zorder=2,
                )
            if metric_index == 0:
                axis.set_title(column_title, fontsize=7.0)
            axis.set_xlim(2.5, 19.5)
            axis.set_ylim(0.0, 0.35)
            axis.set_yticks([0.0, 0.1, 0.2, 0.3])
            axis.grid(color="#E4E4E4", lw=0.45)
            axis.tick_params(labelsize=5.5)
            _panel_label(axis, string.ascii_lowercase[2 * metric_index + column_index])
        axes[metric_index, 0].set_ylabel("{}\nDominant St".format(SHORT_LABELS[metric]))
    for axis in axes[-1, :]:
        axis.set_xlabel("Rotor diameter (m)")
        axis.set_xticks([4, 8, 12, 16, 20])
    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_lookup[str(row.rotor_name)],
            markersize=4.2,
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            label="{} ({:.2f} m)".format(row.rotor_name, float(row.rotor_diameter_m)),
        )
        for row in rotors.itertuples(index=False)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=4,
        fontsize=5.5,
    )
    figure.suptitle(
        "Rotor size versus path-conditional dominant ship-scale disturbance St",
        fontsize=8.0,
        y=0.995,
    )
    figure.text(
        0.5,
        0.002,
        "Lines guide the eye; open markers denote a near tie (top/runner-up effect ratio < 1.10).",
        ha="center",
        va="bottom",
        fontsize=5.2,
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.875), pad=0.8)
    return _save_figure(
        figure,
        output / "figure_05_dominant_St_vs_rotor_diameter",
        formats,
        dpi,
        apply_tight_layout=False,
    )


def plot_modal_deletion_stability(
    stage_table: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    _required_columns(
        stage_table,
        [
            "rotor_name",
            "rotor_diameter_m",
            "stage_index",
            "retained_mode_count",
            "maximum_hard_limit_utilization",
            "observed_hard_instability",
        ],
        "modal_deletion_stability_by_stage.csv",
    )
    _required_columns(
        summary,
        [
            "rotor_name",
            "rotor_diameter_m",
            "worst_stage_index",
            "worst_stage_deleted_mode",
            "worst_stage_retained_mode_count",
            "worst_maximum_hard_limit_utilization",
            "hard_instability_cutoff_status",
        ],
        "modal_deletion_stability_summary.csv",
    )
    rotor_count = int(summary["rotor_name"].nunique())
    if rotor_count < 1:
        raise ResultPlotError("modal-deletion stability summary is empty")
    retained = pd.to_numeric(stage_table["retained_mode_count"], errors="coerce")
    if retained.isna().any():
        raise ResultPlotError("stability table has invalid retained_mode_count")
    mode_count = int(retained.max())
    if len(stage_table) != rotor_count * (mode_count + 1) or len(summary) != rotor_count:
        raise ResultPlotError("modal-deletion stability tables are incomplete")
    rotor_order = (
        summary.sort_values("rotor_diameter_m")["rotor_name"].astype(str).tolist()
    )
    colour_lookup = {rotor: COLORS[index] for index, rotor in enumerate(rotor_order)}
    figure, axes = plt.subplots(1, 2, figsize=(183.0 / 25.4, 94.0 / 25.4))
    for rotor in rotor_order:
        values = stage_table[stage_table["rotor_name"] == rotor].sort_values("stage_index")
        if not np.array_equal(values["stage_index"].to_numpy(dtype=int), np.arange(mode_count + 1)):
            raise ResultPlotError("stability stages are incomplete for {}".format(rotor))
        axes[0].plot(
            values["stage_index"],
            values["maximum_hard_limit_utilization"],
            color=colour_lookup[rotor],
            marker="o",
            ms=2.0,
            lw=0.9,
            label=rotor,
        )
    axes[0].axhline(1.0, color="#B64342", lw=1.0, ls="--")
    axes[0].text(float(mode_count), 0.98, "Hard limit", color="#B64342", ha="right", va="top", fontsize=5.5)
    axes[0].set_xlim(0, mode_count)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Cumulative deletion stage")
    axes[0].set_ylabel("Maximum hard-limit utilization")
    axes[0].set_title("All {} RVMD stages".format(mode_count + 1), fontsize=7.0)
    axes[0].grid(color="#E4E4E4", lw=0.45)
    secondary = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda stage: float(mode_count) - stage,
            lambda retained: float(mode_count) - retained,
        ),
    )
    secondary.set_xlabel("Retained low-frequency modes", fontsize=6.2)
    secondary.tick_params(labelsize=5.5)

    ordered = summary.sort_values("rotor_diameter_m").reset_index(drop=True)
    positions = np.arange(len(ordered))
    for index, row in ordered.iterrows():
        value = float(row["worst_maximum_hard_limit_utilization"])
        rotor = str(row["rotor_name"])
        axes[1].scatter(value, index, s=27.0, color=colour_lookup[rotor], zorder=2)
        deleted = row["worst_stage_deleted_mode"]
        deleted_text = (
            "full {}".format(mode_count)
            if pd.isna(deleted) or str(deleted).strip() == ""
            else "delete M{}".format(int(float(deleted)))
        )
        label_to_left = value > 0.70
        axes[1].text(
            value - 0.025 if label_to_left else value + 0.025,
            index,
            "stage {:02d}, {}".format(int(row["worst_stage_index"]), deleted_text),
            va="center",
            ha="right" if label_to_left else "left",
            fontsize=5.2,
        )
    axes[1].axvline(1.0, color="#B64342", lw=1.0, ls="--")
    axes[1].set_xlim(0.0, 1.05)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(ordered["rotor_name"].astype(str))
    axes[1].set_xlabel("Worst utilization across deletion stages")
    axes[1].set_title("No crossings: closest approach to any hard limit", fontsize=7.0)
    axes[1].grid(axis="x", color="#E4E4E4", lw=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        fontsize=5.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    figure.suptitle(
        "Finite-window closed-loop stability audit during cumulative modal deletion",
        fontsize=8.0,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.94), pad=0.8)
    return _save_figure(
        figure,
        output / "figure_08_modal_deletion_stability",
        formats,
        dpi,
        apply_tight_layout=False,
    )


def _read_timeseries(root: Path, rotor: str, case_id: str) -> pd.DataFrame:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", rotor.strip()).strip("_") or "rotor"
    path = root / "04_timeseries_csv" / "{}__{}.csv".format(safe, case_id)
    if not path.is_file():
        raise ResultPlotError("time-series CSV is missing: {}".format(path))
    frame = pd.read_csv(str(path), encoding="utf-8-sig")
    _required_columns(frame, ["time_s"] + METRICS, path.name)
    _numeric(frame, ["time_s"] + METRICS, path.name)
    return frame


def reconstruction_response_closure(
    root: Path, statistics: pd.DataFrame, trim: pd.DataFrame
) -> pd.DataFrame:
    trim_index = trim.set_index("rotor_name")
    mode_count = _mode_count_from_statistics(statistics)
    full_case_id = _stage_case_id(0, mode_count, mode_count)
    _required_columns(statistics, ["analysis_start_s", "analysis_end_s"], "statistics_long.csv")
    window = statistics[["analysis_start_s", "analysis_end_s"]].drop_duplicates()
    if len(window) != 1:
        raise ResultPlotError("all statistics must use exactly one common analysis window")
    analysis_start_s = float(window.iloc[0]["analysis_start_s"])
    analysis_end_s = float(window.iloc[0]["analysis_end_s"])
    if not (np.isfinite(analysis_start_s) and np.isfinite(analysis_end_s) and analysis_end_s > analysis_start_s):
        raise ResultPlotError("analysis window is invalid")
    rows = []
    for rotor in _rotor_order(statistics):
        rvmd = _read_timeseries(root, rotor, full_case_id)
        raw = _read_timeseries(root, rotor, "original_cfd")
        merged = rvmd[["time_s"] + METRICS].merge(
            raw[["time_s"] + METRICS], on="time_s", suffixes=("_rvmd", "_cfd"), validate="one_to_one"
        )
        if len(merged) != len(rvmd) or len(merged) != len(raw):
            raise ResultPlotError("RVMD/original time grids do not match for {}".format(rotor))
        time = merged["time_s"].to_numpy(dtype=float)
        for metric in METRICS:
            window_time, first, start_interpolated, end_interpolated = boundary_augmented_window(
                time,
                merged[metric + "_rvmd"].to_numpy(dtype=float),
                analysis_start_s,
                analysis_end_s,
            )
            second_time, second, raw_start_interpolated, raw_end_interpolated = boundary_augmented_window(
                time,
                merged[metric + "_cfd"].to_numpy(dtype=float),
                analysis_start_s,
                analysis_end_s,
            )
            if not np.array_equal(window_time, second_time):
                raise ResultPlotError("closure boundary grids disagree for {}".format(rotor))
            duration = analysis_end_s - analysis_start_s
            error = first - second
            rmse = math.sqrt(max(float(np.trapz(error ** 2, window_time) / duration), 0.0))
            first_mean = float(np.trapz(first, window_time) / duration)
            second_mean = float(np.trapz(second, window_time) / duration)
            first_variance = max(
                float(np.trapz((first - first_mean) ** 2, window_time) / duration), 0.0
            )
            second_variance = max(
                float(np.trapz((second - second_mean) ** 2, window_time) / duration), 0.0
            )
            first_rms = math.sqrt(first_variance)
            signal_rms = math.sqrt(second_variance)
            nrmse = rmse / signal_rms if signal_rms > np.finfo(float).eps else np.nan
            covariance = float(
                np.trapz((first - first_mean) * (second - second_mean), window_time) / duration
            )
            correlation = (
                covariance / (first_rms * signal_rms)
                if first_rms > np.finfo(float).eps and signal_rms > np.finfo(float).eps
                else np.nan
            )
            if metric == "lift_N":
                reference_error = rmse / float(trim_index.loc[rotor, "weight_N"])
                reference_unit = "fraction of weight"
            elif metric == "engine_power_W":
                reference_error = rmse / max(abs(float(trim_index.loc[rotor, "trim_engine_power_W"])), np.finfo(float).tiny)
                reference_unit = "fraction of trim power"
            else:
                reference_error = rmse
                reference_unit = "deg" if metric.endswith("deg") else "m"
            rows.append(
                {
                    "rotor_name": rotor,
                    "metric": metric,
                    "rmse": rmse,
                    "rmse_normalized_by_CFD_fluctuation_rms": nrmse,
                    "pearson_r": correlation,
                    "mean_bias_RVMD_minus_CFD": first_mean - second_mean,
                    "RVMD_fluctuation_rms": first_rms,
                    "CFD_fluctuation_rms": signal_rms,
                    "fluctuation_rms_ratio_RVMD_over_CFD": (
                        first_rms / signal_rms if signal_rms > np.finfo(float).eps else np.nan
                    ),
                    "reference_scaled_rmse": reference_error,
                    "reference_scaled_rmse_unit": reference_unit,
                    "sample_count": len(window_time),
                    "time_weighting_method": "boundary_interpolated_trapezoidal_time_weighting_v1",
                    "start_boundary_interpolated": bool(start_interpolated or raw_start_interpolated),
                    "end_boundary_interpolated": bool(end_interpolated or raw_end_interpolated),
                    "analysis_start_s": analysis_start_s,
                    "analysis_end_s": analysis_end_s,
                }
            )
    return pd.DataFrame(rows)


def plot_reconstruction_closure(
    closure: pd.DataFrame,
    rotors: Sequence[str],
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    matrices = []
    for column in ("pearson_r", "rmse_normalized_by_CFD_fluctuation_rms"):
        matrix = np.full((len(METRICS), len(rotors)), np.nan)
        for i, metric in enumerate(METRICS):
            for j, rotor in enumerate(rotors):
                value = closure[(closure["metric"] == metric) & (closure["rotor_name"] == rotor)][column]
                if len(value) != 1:
                    raise ResultPlotError("closure table is incomplete")
                matrix[i, j] = float(value.iloc[0])
        matrices.append(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(183.0 / 25.4, 84.0 / 25.4))
    image = axes[0].imshow(matrices[0], aspect="auto", cmap="cividis", vmin=-1.0, vmax=1.0)
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03, label="Pearson r")
    finite_error = matrices[1][np.isfinite(matrices[1])]
    vmax = float(np.max(finite_error)) if finite_error.size else 1.0
    vmax = max(vmax, np.finfo(float).tiny)
    image2 = axes[1].imshow(matrices[1], aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    figure.colorbar(image2, ax=axes[1], fraction=0.046, pad=0.03, label="RMSE / CFD fluctuation RMS")
    for axis, matrix, image_artist in zip(axes, matrices, (image, image2)):
        normalizer = image_artist.norm
        color_map = image_artist.cmap
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                label = "NA" if not np.isfinite(value) else "{:.2f}".format(value)
                rgba = color_map(normalizer(value)) if np.isfinite(value) else (1.0, 1.0, 1.0, 1.0)
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="black" if luminance > 0.57 else "white",
                )
    for axis, title, label in zip(axes, ("Temporal correlation", "Normalized reconstruction error"), ("a", "b")):
        axis.set_xticks(np.arange(len(rotors)))
        axis.set_xticklabels(rotors, rotation=35, ha="right", rotation_mode="anchor")
        axis.set_yticks(np.arange(len(METRICS)))
        axis.set_yticklabels([SHORT_LABELS[metric] for metric in METRICS])
        axis.set_title(title, fontsize=7.0)
        _panel_label(axis, label)
    return _save_figure(figure, output / "figure_06_full_RVMD_vs_original_CFD_closure", formats, dpi)


def plot_representative_histories(
    root: Path,
    statistics: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    mode_count = _mode_count_from_statistics(statistics)
    diameter = statistics[["rotor_name", "rotor_diameter_m"]].drop_duplicates().sort_values("rotor_diameter_m")
    representatives = [diameter.iloc[0]["rotor_name"], diameter.iloc[len(diameter) // 2]["rotor_name"], diameter.iloc[-1]["rotor_name"]]
    representatives = list(dict.fromkeys(representatives))
    cases = [
        ("original_cfd", "Original CFD", "#272727", "-"),
        (_stage_case_id(0, mode_count, mode_count), "{}-mode RVMD".format(mode_count), COLORS[0], "--"),
        (_stage_case_id(mode_count, 0, mode_count), "Mean only", COLORS[1], ":"),
    ]
    figure, axes = plt.subplots(len(METRICS), len(representatives), figsize=(183.0 / 25.4, 190.0 / 25.4), sharex="col")
    source = []
    for column, rotor in enumerate(representatives):
        frames = [(case_id, label, color, style, _read_timeseries(root, rotor, case_id)) for case_id, label, color, style in cases]
        for row_index, metric in enumerate(METRICS):
            axis = axes[row_index, column] if len(representatives) > 1 else axes[row_index]
            for case_id, label, color, style, frame in frames:
                axis.plot(frame["time_s"], frame[metric], color=color, ls=style, lw=0.8, label=label)
                selected = frame[["time_s", metric]].copy()
                selected["rotor_name"] = rotor
                selected["case_id"] = case_id
                selected["metric"] = metric
                selected.rename(columns={metric: "value"}, inplace=True)
                source.append(selected[["rotor_name", "case_id", "metric", "time_s", "value"]])
            if column == 0:
                axis.set_ylabel(METRIC_LABELS[metric])
            if row_index == 0:
                rotor_diameter = float(diameter[diameter["rotor_name"] == rotor]["rotor_diameter_m"].iloc[0])
                axis.set_title("{} ({:.2f} m)".format(rotor, rotor_diameter), fontsize=7.0)
            if row_index == len(METRICS) - 1:
                axis.set_xlabel("Relative time (s)")
            if column == 0:
                _panel_label(axis, string.ascii_lowercase[row_index])
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=3,
        fontsize=6.2,
        borderaxespad=0.0,
    )
    files = _save_figure(figure, output / "figure_07_representative_time_histories", formats, dpi)
    return files, pd.concat(source, ignore_index=True)


def plot_hover_trim(
    trim: pd.DataFrame,
    statistics: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Plot physical hover-trim quantities as an auditable scale comparison."""

    diameter = statistics[["rotor_name", "rotor_diameter_m"]].drop_duplicates()
    table = trim.merge(diameter, on="rotor_name", validate="one_to_one")
    table = table.sort_values("rotor_diameter_m")
    table["disk_loading_Npm2"] = table["weight_N"] / (
        math.pi * (0.5 * table["rotor_diameter_m"]) ** 2
    )
    table["specific_trim_power_WpN"] = table["trim_engine_power_W"] / table["weight_N"]
    if not np.all(np.isfinite(table[
        ["disk_loading_Npm2", "specific_trim_power_WpN"]
    ].to_numpy(dtype=float))):
        raise ResultPlotError("hover-trim derived quantities contain non-finite values")

    panels = [
        ("induced_velocity_trim_mps", "Trim induced velocity (m/s)"),
        ("collective_trim_deg", "Trim collective (deg)"),
        ("disk_loading_Npm2", "Disk loading (N/m^2)"),
        ("specific_trim_power_WpN", "Trim power / weight (W/N)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(183.0 / 25.4, 106.0 / 25.4), sharex=True)
    x = table["rotor_diameter_m"].to_numpy(dtype=float)
    for index, (column, ylabel) in enumerate(panels):
        axis = axes.flat[index]
        axis.plot(x, table[column], marker="o", ms=4.0, lw=1.0, color=COLORS[index])
        for _, row in table.iterrows():
            axis.annotate(
                str(row["rotor_name"]),
                (float(row["rotor_diameter_m"]), float(row[column])),
                xytext=(2, 3), textcoords="offset points", fontsize=5.2, rotation=18,
                rotation_mode="anchor",
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E0E0E0", lw=0.5)
        if index >= 2:
            axis.set_xlabel("Rotor diameter (m)")
        _panel_label(axis, string.ascii_lowercase[index])
    files = _save_figure(figure, output / "figure_S01_physical_hover_trim", formats, dpi)
    return files, table


def plot_dynamic_inflow_histories(
    root: Path,
    statistics: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Expose the three Pitt–Peters disturbance states for representative scales."""

    mode_count = _mode_count_from_statistics(statistics)
    full_case_id = _stage_case_id(0, mode_count, mode_count)
    diameter = statistics[["rotor_name", "rotor_diameter_m"]].drop_duplicates().sort_values("rotor_diameter_m")
    representatives = [
        diameter.iloc[0]["rotor_name"],
        diameter.iloc[len(diameter) // 2]["rotor_name"],
        diameter.iloc[-1]["rotor_name"],
    ]
    representatives = list(dict.fromkeys(representatives))
    state_columns = ["delta_lambda_0", "delta_lambda_c", "delta_lambda_s", "delta_mu_z"]
    state_labels = ["δλ0", "δλc", "δλs", "δμz"]
    figure, axes = plt.subplots(
        len(state_columns), len(representatives),
        figsize=(183.0 / 25.4, 145.0 / 25.4), sharex="col",
    )
    source_frames = []
    for column_index, rotor in enumerate(representatives):
        frame = _read_timeseries(root, rotor, full_case_id)
        _required_columns(frame, state_columns, "{} full-RVMD history".format(rotor))
        _numeric(frame, state_columns, "{} full-RVMD history".format(rotor))
        for row_index, (state_column, state_label) in enumerate(zip(state_columns, state_labels)):
            axis = axes[row_index, column_index] if len(representatives) > 1 else axes[row_index]
            axis.plot(frame["time_s"], frame[state_column], color=COLORS[row_index], lw=0.8)
            axis.axhline(0.0, color="#767676", lw=0.5, ls="--")
            if column_index == 0:
                axis.set_ylabel(state_label)
            if row_index == 0:
                rotor_diameter = float(diameter[diameter["rotor_name"] == rotor]["rotor_diameter_m"].iloc[0])
                axis.set_title("{} ({:.2f} m)".format(rotor, rotor_diameter), fontsize=7.0)
            if row_index == len(state_columns) - 1:
                axis.set_xlabel("Relative time (s)")
            if column_index == 0:
                _panel_label(axis, string.ascii_lowercase[row_index])
            selected = frame[["time_s", state_column]].copy()
            selected["rotor_name"] = rotor
            selected["case_id"] = full_case_id
            selected["state"] = state_column
            selected.rename(columns={state_column: "value"}, inplace=True)
            source_frames.append(selected[["rotor_name", "case_id", "state", "time_s", "value"]])
    files = _save_figure(figure, output / "figure_S02_dynamic_inflow_states", formats, dpi)
    return files, pd.concat(source_frames, ignore_index=True)


def plot_hover_hold_demands(
    root: Path,
    statistics: pd.DataFrame,
    trim: pd.DataFrame,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Audit idealized hover-hold effort without hiding it in response plots."""

    mode_count = _mode_count_from_statistics(statistics)
    full_case_id = _stage_case_id(0, mode_count, mode_count)
    window = statistics[["analysis_start_s", "analysis_end_s"]].drop_duplicates()
    if len(window) != 1:
        raise ResultPlotError("hover-hold audit requires one common analysis window")
    start_s = float(window.iloc[0]["analysis_start_s"])
    end_s = float(window.iloc[0]["analysis_end_s"])
    diameter = statistics[["rotor_name", "rotor_diameter_m"]].drop_duplicates()
    trim_index = trim.set_index("rotor_name")
    rows = []
    for _, rotor_row in diameter.iterrows():
        rotor = str(rotor_row["rotor_name"])
        rotor_diameter = float(rotor_row["rotor_diameter_m"])
        weight = float(trim_index.loc[rotor, "weight_N"])
        moment_reference = weight * 0.5 * rotor_diameter
        for case_id, flow_kind in (
            (full_case_id, "{}-mode RVMD".format(mode_count)),
            ("original_cfd", "Original CFD"),
        ):
            frame = _read_timeseries(root, rotor, case_id)
            columns = [
                "hover_hold_Fz_body_N",
                "hover_hold_Mx_body_Nm",
                "hover_hold_My_body_Nm",
            ]
            _required_columns(frame, columns, "{} hover-hold history".format(rotor))
            _numeric(frame, columns, "{} hover-hold history".format(rotor))
            selected = frame[
                (frame["time_s"] >= start_s - 1.0e-12)
                & (frame["time_s"] <= end_s + 1.0e-12)
            ]
            if len(selected) < 2:
                raise ResultPlotError("hover-hold audit window is empty for {}".format(rotor))
            normalizers = {
                "hover_hold_Fz_body_N": weight,
                "hover_hold_Mx_body_Nm": moment_reference,
                "hover_hold_My_body_Nm": moment_reference,
            }
            for metric in columns:
                values = selected[metric].to_numpy(dtype=float) / normalizers[metric]
                rows.append(
                    {
                        "rotor_name": rotor,
                        "rotor_diameter_m": rotor_diameter,
                        "case_id": case_id,
                        "flow_kind": flow_kind,
                        "metric": metric,
                        "normalizer": normalizers[metric],
                        "maximum_absolute_normalized_demand": float(np.max(np.abs(values))),
                        "rms_normalized_demand": float(np.sqrt(np.mean(values ** 2))),
                        "analysis_start_s": start_s,
                        "analysis_end_s": end_s,
                    }
                )
    source = pd.DataFrame(rows)
    panels = [
        ("hover_hold_Fz_body_N", "max |Fz,hold| / weight"),
        ("hover_hold_Mx_body_Nm", "max |Mx,hold| / (weight R)"),
        ("hover_hold_My_body_Nm", "max |My,hold| / (weight R)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(183.0 / 25.4, 67.0 / 25.4), sharex=True)
    styles = {
        "{}-mode RVMD".format(mode_count): (COLORS[0], "o", "-"),
        "Original CFD": ("#272727", "s", "--"),
    }
    for panel_index, (metric, ylabel) in enumerate(panels):
        axis = axes[panel_index]
        for flow_kind, (color, marker, linestyle) in styles.items():
            selected = source[
                (source["metric"] == metric) & (source["flow_kind"] == flow_kind)
            ].sort_values("rotor_diameter_m")
            axis.plot(
                selected["rotor_diameter_m"],
                selected["maximum_absolute_normalized_demand"],
                color=color,
                marker=marker,
                ls=linestyle,
                ms=3.5,
                lw=0.9,
                label=flow_kind,
            )
        axis.set_xlabel("Rotor diameter (m)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E0E0E0", lw=0.5)
        _panel_label(axis, string.ascii_lowercase[panel_index])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.0)
    files = _save_figure(figure, output / "figure_S04_hover_hold_demand_audit", formats, dpi)
    return files, source


def plot_rvmd_convergence(
    root: Path,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Plot the stopping metric and unsorted centre-frequency trajectories."""

    with np.load(str(root / "02_rvmd" / "temporal_model.npz"), allow_pickle=False) as data:
        difference = np.asarray(data["difference_history"], dtype=float)
        omega = np.asarray(data["omega_history"], dtype=float)
        info = json.loads(str(np.asarray(data["info_json"]).item()))
    mode_count = int(omega.shape[0]) if omega.ndim == 2 else -1
    if difference.ndim != 1 or mode_count < 1 or omega.shape[1] != difference.size + 1:
        raise ResultPlotError("RVMD convergence histories have inconsistent dimensions")
    dt = float(info["dt_seconds"])
    tolerance = float(info["tolerance"])
    if not np.all(np.isfinite(difference)) or np.any(difference < 0.0):
        raise ResultPlotError("RVMD difference history is invalid")
    steps = np.arange(1, difference.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(183.0 / 25.4, 76.0 / 25.4))
    axes[0].semilogy(steps, np.maximum(difference, np.finfo(float).tiny), color=COLORS[0], lw=1.0)
    # Matplotlib renders log-decade exponents at roughly 70% of the parent
    # tick size.  Raising only this axis keeps those PDF glyph runs above the
    # 5 pt publication floor without inflating the rest of the figure.
    axes[0].tick_params(axis="y", labelsize=7.2)
    axes[0].axhline(tolerance, color=COLORS[1], lw=0.8, ls="--", label="Tolerance")
    axes[0].set_xlabel("Completed RVMD sweep")
    axes[0].set_ylabel("Sum of relative modal changes")
    axes[0].legend(fontsize=5.8)
    _panel_label(axes[0], "a")
    frequency_history = omega / dt
    visible_frequency_history = frequency_history[:, 1:]
    visible_steps = np.arange(1, frequency_history.shape[1])
    for mode_index in range(frequency_history.shape[0]):
        axes[1].plot(
            visible_steps,
            visible_frequency_history[mode_index],
            lw=0.55,
            alpha=0.75,
        )
    initial_min = float(np.min(frequency_history[:, 0]))
    initial_max = float(np.max(frequency_history[:, 0]))
    axes[1].text(
        0.02,
        0.98,
        "Sweep 0 initialization: {:.2f}–{:.2f} Hz (outside view)".format(
            initial_min, initial_max
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
    )
    axes[1].set_ylim(
        0.0,
        1.08 * float(np.max(visible_frequency_history)),
    )
    axes[1].set_xlabel("Completed RVMD sweep")
    axes[1].set_ylabel("Internal centre frequency (Hz)")
    _panel_label(axes[1], "b")
    rows = []
    for step_index in range(frequency_history.shape[1]):
        row = {
            "completed_sweep": step_index,
            "difference_after_sweep": "" if step_index == 0 else difference[step_index - 1],
        }
        for mode_index in range(mode_count):
            row["internal_mode_{:02d}_frequency_hz".format(mode_index + 1)] = frequency_history[mode_index, step_index]
        rows.append(row)
    files = _save_figure(figure, output / "figure_S03_rvmd_convergence", formats, dpi)
    return files, pd.DataFrame(rows)


def write_figure_manual(
    root: Path,
    outputs: Mapping[str, Sequence[str]],
    dominant: pd.DataFrame,
    closure: pd.DataFrame,
    stability_summary: pd.DataFrame,
    stability_audit: Mapping[str, Any],
    metadata: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Path:
    _required_columns(
        closure,
        ["pearson_r", "rmse_normalized_by_CFD_fluctuation_rms"],
        "full_RVMD_vs_original_CFD_closure.csv",
    )
    closure_values = closure[["pearson_r", "rmse_normalized_by_CFD_fluctuation_rms"]].to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(closure_values)):
        raise ResultPlotError("closure table contains missing/non-finite values")
    strong_count = int(
        np.sum(
            (closure["pearson_r"].to_numpy(dtype=float) >= CLOSURE_STRONG_R)
            & (
                closure["rmse_normalized_by_CFD_fluctuation_rms"].to_numpy(dtype=float)
                <= CLOSURE_STRONG_NRMSE
            )
        )
    )
    trend_count = int(
        np.sum(
            (closure["pearson_r"].to_numpy(dtype=float) >= CLOSURE_TREND_R)
            & (
                closure["rmse_normalized_by_CFD_fluctuation_rms"].to_numpy(dtype=float)
                <= CLOSURE_TREND_NRMSE
            )
        )
    )
    closure_count = int(len(closure))
    rvmd_info = metadata.get("rvmd_info", {})
    mode_count = int(metadata.get("k", rvmd_info.get("k", context.get("mode_count", -1))))
    if mode_count < 1:
        raise ResultPlotError("figure manual cannot determine the selected RVMD mode count")
    dt_seconds = float(metadata.get("dt_seconds", rvmd_info.get("dt_seconds", context.get("dt_s", float("nan")))))
    completed_steps = int(rvmd_info.get("completed_steps", -1))
    initial_frequency = np.asarray(rvmd_info.get("initial_frequencies_hz", []), dtype=float)
    initial_frequency_text = (
        "{:.2f}–{:.2f} Hz".format(float(np.min(initial_frequency)), float(np.max(initial_frequency)))
        if initial_frequency.size and np.all(np.isfinite(initial_frequency))
        else "见源数据"
    )
    _required_columns(
        stability_summary,
        [
            "rotor_name",
            "worst_stage_index",
            "worst_maximum_hard_limit_utilization",
            "hard_instability_cutoff_status",
        ],
        "modal_deletion_stability_summary.csv",
    )
    rotor_count = int(stability_summary["rotor_name"].nunique())
    stage_width = max(2, len(str(mode_count)))
    no_cutoff_status = "not_observed_through_stage_{:0{width}d}".format(
        mode_count, width=stage_width
    )
    no_cutoff_count = int(
        np.sum(
            stability_summary["hard_instability_cutoff_status"].astype(str)
            == no_cutoff_status
        )
    )
    worst_stability = stability_summary.iloc[
        int(
            np.argmax(
                pd.to_numeric(
                    stability_summary["worst_maximum_hard_limit_utilization"],
                    errors="coerce",
                ).to_numpy(dtype=float)
            )
        )
    ]
    growth_case_count = int(stability_audit.get("terminal_growth_advisory_case_count", -1))
    growth_quantity_count = int(
        stability_audit.get("terminal_growth_advisory_quantity_count", -1)
    )
    low_mode_fraction = float(
        stability_audit.get("analysis_duration_fraction_of_lowest_mode_period", float("nan"))
    )
    path = root / "08_figure_manual" / "结果图说明手册.md"
    lines = [
        "# NATO-GD 三维 RVMD—旋翼动力学结果图说明手册",
        "",
        "## 阅读前必须明确的结论边界",
        "",
        f"本图组用于识别“按中心频率从高到低累计删除 RVMD 模态”对不同旋翼构型响应的影响。它能支持经 99.8% 全空间能量门槛选定的 K={mode_count} 模态重构子空间内的频带敏感性和构型差异判断，但不能把某一个累计删模差值直接解释为线性、孤立的单模态因果效应，也不能在未通过 Figure 06 闭合审计时宣称已完整解释原始 CFD 响应。",
        "",
        f"- RVMD K={mode_count} 是由全空间、去均值重构残差能量门槛选出的截断表示；任何未重构部分均在残差审计中显式报告，不宣称“无能量损失”。",
        "- “模态自能”对非正交模态不具可加性；判定动力学重要性应优先看相邻删模响应，不只看流场自能。",
        "- Pitt–Peters 部分是近悬停、冻结系数的三状态扰动入流模型；整体非线性来自叶素载荷、姿态投影、移动倾斜桨盘取样及状态反馈。",
        "- 未使用动态失速、自由尾迹或双向 CFD–旋翼耦合；水平平移与偏航被约束。",
        "- 方位平均配平先用静止空气动量理论/BEM 解作初值，再在 NATO-GD 三维时均场中联立求解总距与三个入流状态；这是固定机体的方位平均平衡，不是周期轨道打靶。",
        "- 为在统一 11 s 窗口内研究扰动响应，所有构型使用按质量/惯量归一的理想 PD 悬停保持器；它以广义力/力矩实现，未建模执行机饱和，因此必须同时给出增益敏感性。",
        "- 整机惯量、阻尼和悬停保持增益中含工程估计，定量结论应配合参数敏感性。",
        "",
        "## 共同坐标与删模规则",
        "",
        f"- 模态 1→{mode_count} 按时间演化系数的中心频率升序。`stage 0` 保留全部 {mode_count} 阶；`stage j` 删除当前最高频模态并保留 `{mode_count}-j` 阶；`stage {mode_count}` 仅保留时间均值场。",
        "- Strouhal 数一律按 `St = f×13.716/20` 计算，其中 f 的单位为 Hz。",
        "- 对于同一旋翼构型，所有删模工况采用同一分析窗、同一 NATO-GD 时均场配平初值、同一操纵量、同一常值固定机体反力和同一悬停保持增益；不针对单个删模工况重新减均值或配平。",
        "- 1–11 s 内的均值、标准差和 RMS 均先在窗口边界做线性插值，再用梯形法做物理时间加权；峰—峰值与最大绝对值使用同一边界增广窗口。",
        "",
        "## 图件逐项说明",
        "",
        "### Figure 01：RVMD 频率、自能、带宽与重构审计",
        "",
        "a 给出升序中心频率及对应 St；b 是模态自能占比，仅用于强度参考；c 用频谱二阶矩表示频带宽度，点大小对应自能；d 分开报告 POD、压缩识别空间和全空间重构的能量审计，这三个百分比属于不同阶段，不应混为一个指标。",
        "",
        "### Figure 02：时间均值对保留模态数的变化",
        "",
        f"横轴从 0 到 {mode_count} 表示保留的低频模态数。升力用重量归一化，功率用悬停配平功率归一化，姿态和升沉保留物理单位。纵轴是相对全 {mode_count} 模态工况的改变，因而越靠近零越接近完整选定模态集的响应。",
        "",
        "### Figure 03：相邻累计删模步之间的条件边际均值差",
        "",
        "每一列是“再删除当前最高频模态”前后的差值；横轴标号指向被删除的原始升频模态。红/蓝表示带符号的变化，色彩强度表示绝对影响大小。这是定位敏感模态的主图，但属于累计上下文中的边际差值。",
        "",
        "### Figure 04：峰—峰值对模态保留数的变化",
        "",
        "该图补充时间均值，用于识别“均值改变不大但非定常振幅显著变化”的模态频带。升力和功率分别用重量和配平功率归一化。",
        "",
        "### Figure 05：主导 St 与旋翼尺度",
        "",
        "该图采用 5 行×2 列小多图：行依次为升力、功率、横滚、俯仰和升沉，列分别为时间加权均值与峰—峰值。每格横轴为实际旋翼直径，纵轴为绝对相邻删模改变最大的主导模态舰船尺度 St；连线仅作视觉引导，不做平滑或回归。空心点表示第一/第二候选效应比小于 1.10，提示主导频带存在近并列。当前七构型没有给出清晰的直径—主导 St 单调律；除直径外，质量、盘载、转速、桨叶几何、惯量与控制参数均是混杂因素，因此只能作描述性频带选择比较。",
        "",
        f"### Figure 06：{mode_count} 模态 RVMD 与原始 CFD 的动力学闭合检查",
        "",
        f"a 是统一分析窗内的时间加权 Pearson 相关系数；b 是时间加权 RVMD–CFD 时历 RMSE 除以原始 CFD 的脉动 RMS，格内数字是未截顶的定量值。高相关与低误差需要同时满足；高相关但归一化 RMSE 较大表示相位趋势相似但幅值未闭合。若闭合不足，Figure 03–05 只能解读为“{mode_count} 模态 RVMD 子空间、给定累计删减路径下的条件边际敏感模态”。",
        "",
        "本项目预设的非通用解释门槛为：强闭合 `r≥{:.2f}` 且 `nRMSE≤{:.2f}`，当前通过 {}/{} 个“构型×响应量”组合；趋势闭合 `r≥{:.2f}` 且 `nRMSE≤{:.2f}`，当前通过 {}/{} 个组合。该门槛仅用于本项目分级，不能替代学科通用验证标准。".format(
            CLOSURE_STRONG_R,
            CLOSURE_STRONG_NRMSE,
            strong_count,
            closure_count,
            CLOSURE_TREND_R,
            CLOSURE_TREND_NRMSE,
            trend_count,
            closure_count,
        ),
        "",
        "### Figure 07：大、中、小代表构型的时历",
        "",
        f"每列是一个尺度代表，每行是一个动力学量，并置原始 CFD、{mode_count} 模态 RVMD 与仅均值场。该图用于检查相位、尖峰、短时偏差以及统计图可能隐藏的非平稳过程。",
        "",
        "### Figure 08：累计模态删除的有限时间硬限审计",
        "",
        "a 给出每种构型在 stage 0–{} 的最大硬限利用率，1.0 为正式越界线；上轴同步标出保留的低频模态数。b 汇总每种构型全过程中最接近任一硬限的工况及其删模阶段。当前 {}/{} 种构型均未观测到硬失稳截止阶；最接近硬限的是 {} 的 stage {}，最大利用率为 {:.4f}，仍低于 1.0。气动/入流模型适用域失效单独标记为结果不可判读，不会被重新命名为动力学失稳；当前此类工况为 0。末段增长筛查在 {} 个工况、{} 个状态量上给出提示，但这些提示未越界且不连续，不能当作失稳截止阶。分析窗只有最低保留模态周期的 {:.3f} 倍，因此结论只是给定理想 PD 悬停保持闭环在统一有限时间窗内有界，不等于渐近稳定、开环稳定或真实执行机构稳定。".format(
            mode_count,
            no_cutoff_count,
            rotor_count,
            worst_stability["rotor_name"],
            int(worst_stability["worst_stage_index"]),
            float(worst_stability["worst_maximum_hard_limit_utilization"]),
            growth_case_count,
            growth_quantity_count,
            low_mode_fraction,
        ),
        "",
        "### Figure S01：物理悬停配平与旋翼尺度",
        "",
        "对比 NATO-GD 时均场配平的诱导速度、总距、盘载与单位重量配平功率。静止空气动量理论结果仅是数值初值；图中的最终值来自总距与三入流状态的联立方位平均平衡。",
        "",
        "### Figure S02：动态入流状态时历",
        "",
        "对大、中、小代表构型直接给出 Pitt–Peters 三个扰动入流状态及外界轴向速度比扰动。三个 δλ 是相对静止空气参考入流的扰动，因此 NATO-GD 时均场配平初值一般不为零。该图是动态入流确已参与计算的可视化证据，不代表自由尾迹模型。",
        "",
        "### Figure S03：RVMD 收敛历史",
        "",
        f"a 给出每次 Gauss–Seidel 扫描的相对模态变化量之和及停止阈值；b 给出 sweep 1–{completed_steps} 的未排序内部模态中心频率轨迹。sweep 0 的初始范围 {initial_frequency_text} 在图内文字注明但不纳入主轴缩放，完整值仍保留在源数据 CSV。只有最终变化量不高于阈值时，分解才标记为已收敛。",
        "",
        "### Figure S04：悬停保持器需求审计",
        "",
        f"对 {mode_count} 模态 RVMD 与原始 CFD 分别给出分析窗内理想悬停保持广义力的最大绝对值。升沉通道用重量归一，滚转/俯仰通道用重量×旋翼半径归一。该图用于判断响应是否依赖过大的理想控制作用；它不等同于真实执行机载荷，因为当前未建模饱和与带宽。",
        "",
        "### Figure QA01：数值收敛与参数敏感性",
        "",
        "位于 `09_qa/`。a 比较大、中、小代表构型在两级最大桨叶方位步长下的最大归一化差异；b 报告滚转/俯仰惯量、机体阻尼和悬停保持增益各 ±20% 时的最大归一化统计量变化。这是确定性数值/局部参数审计，不是置信区间或总体推断。",
        "",
        "## 主导模态检索表的使用",
        "",
        "`06_figures/source_data/dominant_modes.csv` 给出每个构型、每个响应量在均值与峰—峰两种准则下的主导与次主导模态、f、St、效应比和近并列标记。建议在论文中同时报告带符号效应、绝对效应及主/次候选差距，并结合 Figure 03/04 判断其稳健性。",
        "",
        "## 核心尺寸—频率假设专用图组",
        "",
        "该图组位于 `06_figures/normalized_modal_sensitivity/`，使用全部 St 模态的单模态与留一法动力学算例，不以累计删模相邻差分代替单模态贡献。",
        "",
        "- `normalized_sensitivity_spectra`：单位模态能量增益 G 的 L1 归一化频谱，用于识别旋翼自身的频率传递倾向。",
        "- `actual_response_spectra`：实际响应 R 的 L1 贡献频谱，保留当前 NATO-GD 尾流各模态真实能量差异。",
        "- `diameter_vs_St90`：G 谱的旋翼直径—有效上频带 St90 点线图及功率律拟合。",
        "- `diameter_vs_St90_basis_comparison`：并列比较 R 谱与 G 谱的 St90(D)。两者均满足 β>0 时，对“小旋翼对应更高敏感频带”的支持最稳健；仅 G 满足时只能说明传递能力，不能声称当前尾流已产生显著高频响应。",
        "- `diameter_vs_StD90`：检查旋翼尺度 StD90=f90D/U 是否近似塌缩。",
        "- `diameter_frequency_power_law_fits.csv`：按 `spectrum_basis` 分别保存 R、G 的 C、β、对数空间 R² 与留一构型交叉验证误差。",
        "",
        "## 当前数据与几何门槛",
        "",
        f"- 采样间隔采用用户最终指定的 {dt_seconds:.7g} s；中心频率和 St 均以此时间轴重新计算。",
        "- 轮毂位置暂取输出体几何中心 (150.9, 0, 11.8) m，当前标记为待用户确认的几何推断。",
        "",
        "## 可追溯性",
        "",
        "每张定量图的整理数据位于 `06_figures/source_data/`；原始时历位于 `04_timeseries_csv/`；悬停配平位于 `03_hover_trim/`；RVMD 频率和完整重构审计位于 `02_rvmd/`。",
        "",
        "图件导出格式：SVG（可编辑文字）、PDF（矢量）、TIFF/PNG（600 dpi）。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_all_figures(
    root: Path,
    *,
    formats: Sequence[str] = ("svg", "pdf", "tiff", "png"),
    dpi: int = DEFAULT_RASTER_DPI,
) -> Dict[str, Any]:
    frequency, statistics, trim, metadata, context = _load_inputs(root)
    output = root / "06_figures" / "manuscript"
    supplementary = root / "06_figures" / "supplementary"
    source_dir = root / "06_figures" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Sequence[str]] = {}
    outputs["figure_01"] = plot_rvmd_summary(root, frequency, metadata, output, formats, dpi)
    outputs["figure_02"], mean_source = plot_response_means(statistics, trim, output, formats, dpi)
    mean_source.to_csv(str(source_dir / "response_mean_curves.csv"), index=False, encoding="utf-8-sig")
    mean_effects = adjacent_effect_table(statistics, trim, "mean")
    mean_effects.to_csv(str(source_dir / "adjacent_mean_effects.csv"), index=False, encoding="utf-8-sig")
    outputs["figure_03"] = plot_adjacent_effect_heatmaps(
        mean_effects, frequency, output, formats, dpi,
        "figure_03_adjacent_mode_mean_effects",
        "Conditional adjacent response-mean difference along cumulative deletion path",
    )
    outputs["figure_04"], variability_source = plot_peak_to_peak(statistics, trim, output, formats, dpi)
    variability_source.to_csv(str(source_dir / "response_peak_to_peak_curves.csv"), index=False, encoding="utf-8-sig")
    variability_effects = adjacent_effect_table(statistics, trim, "peak_to_peak")
    variability_effects.to_csv(str(source_dir / "adjacent_peak_to_peak_effects.csv"), index=False, encoding="utf-8-sig")
    dominant = dominant_mode_table(mean_effects, variability_effects, frequency)
    dominant.to_csv(str(source_dir / "dominant_modes.csv"), index=False, encoding="utf-8-sig")
    outputs["figure_05"] = plot_dominant_modes(dominant, output, formats, dpi)
    closure = reconstruction_response_closure(root, statistics, trim)
    closure.to_csv(str(source_dir / "full_RVMD_vs_original_CFD_closure.csv"), index=False, encoding="utf-8-sig")
    outputs["figure_06"] = plot_reconstruction_closure(closure, _rotor_order(statistics), output, formats, dpi)
    outputs["figure_07"], history_source = plot_representative_histories(root, statistics, output, formats, dpi)
    history_source.to_csv(str(source_dir / "representative_time_histories.csv"), index=False, encoding="utf-8-sig")
    stability_stage_path = root / "09_qa" / "modal_deletion_stability_by_stage.csv"
    stability_summary_path = root / "09_qa" / "modal_deletion_stability_summary.csv"
    stability_audit_path = root / "09_qa" / "modal_deletion_stability_audit.json"
    for required_path in (stability_stage_path, stability_summary_path, stability_audit_path):
        if not required_path.is_file():
            raise ResultPlotError(
                "required stability audit is missing; run analyze_modal_deletion_stability.py first: "
                + str(required_path)
            )
    stability_stage = pd.read_csv(str(stability_stage_path), encoding="utf-8-sig")
    stability_summary = pd.read_csv(str(stability_summary_path), encoding="utf-8-sig")
    stability_audit = _read_json(stability_audit_path)
    stability_stage.to_csv(
        str(source_dir / "modal_deletion_stability_by_stage.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    stability_summary.to_csv(
        str(source_dir / "modal_deletion_stability_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    outputs["figure_08"] = plot_modal_deletion_stability(
        stability_stage, stability_summary, output, formats, dpi
    )
    outputs["figure_S01"], trim_source = plot_hover_trim(trim, statistics, supplementary, formats, dpi)
    trim_source.to_csv(str(source_dir / "physical_hover_trim.csv"), index=False, encoding="utf-8-sig")
    outputs["figure_S02"], inflow_source = plot_dynamic_inflow_histories(
        root, statistics, supplementary, formats, dpi
    )
    inflow_source.to_csv(str(source_dir / "dynamic_inflow_histories.csv"), index=False, encoding="utf-8-sig")
    outputs["figure_S03"], convergence_source = plot_rvmd_convergence(
        root, supplementary, formats, dpi
    )
    convergence_source.to_csv(str(source_dir / "rvmd_convergence_history.csv"), index=False, encoding="utf-8-sig")
    outputs["figure_S04"], controller_source = plot_hover_hold_demands(
        root, statistics, trim, supplementary, formats, dpi
    )
    controller_source.to_csv(
        str(source_dir / "hover_hold_demand_audit.csv"), index=False, encoding="utf-8-sig"
    )
    manual = write_figure_manual(
        root,
        outputs,
        dominant,
        closure,
        stability_summary,
        stability_audit,
        metadata,
        context,
    )
    manifest = {
        "figures": outputs,
        "source_data_directory": str(source_dir),
        "manual": str(manual),
        "formats": list(formats),
        "raster_dpi": int(dpi),
        "backend": "Python/matplotlib only",
        "figure_width_mm": 183,
        "minimum_source_font_pt": 5.2,
        "provisional_reasons": context.get("provisional_reasons", []),
    }
    manifest_path = root / "06_figures" / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "tiff", "png"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_RASTER_DPI)
    args = parser.parse_args(argv)
    context = load_study_context(args.config, allow_provisional=args.allow_provisional)
    manifest = create_all_figures(context.output_dir, formats=args.formats, dpi=args.dpi)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ResultPlotError",
    "adjacent_effect_table",
    "create_all_figures",
    "dominant_mode_table",
    "plot_dominant_modes",
    "plot_dynamic_inflow_histories",
    "plot_hover_trim",
    "plot_hover_hold_demands",
    "plot_modal_deletion_stability",
    "plot_rvmd_convergence",
    "reconstruction_response_closure",
]
