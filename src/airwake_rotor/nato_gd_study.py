#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Configuration, provenance and stage semantics for the NATO-GD study.

This module keeps scientific assumptions out of the numerical kernels.  In
particular, it records whether the CFD time interval and rotor-hub position
were measured, supplied by the user, or inferred.  A provisional run is
allowed only when the caller opts into it explicitly; the resulting metadata
continues to carry the unresolved gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class StudyConfigurationError(ValueError):
    """Raised when a study configuration is incomplete or contradictory."""


OUTPUT_SUBDIRECTORIES = (
    "00_provenance",
    "01_cache",
    "02_rvmd",
    "03_hover_trim",
    "04_timeseries_csv",
    "05_statistics",
    "06_figures/manuscript",
    "06_figures/supplementary",
    "06_figures/source_data",
    "07_flow_gifs_200frames",
    "08_figure_manual",
    "09_qa",
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise StudyConfigurationError("cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise StudyConfigurationError("JSON root must be an object: {}".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
    return path


def _finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise StudyConfigurationError("{} must be finite and positive".format(name))
    return number


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise StudyConfigurationError("configuration section {!r} is required".format(key))
    return value


def _resolve_path(project_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StudyConfigurationError("{} must be a nonempty path string".format(name))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class StudyContext:
    config_path: Path
    project_root: Path
    config: Dict[str, Any]
    source_data_dir: Path
    output_dir: Path
    cache_dir: Path
    rotor_config_path: Path
    dynamic_overrides_path: Path
    dt_s: float
    mode_count: int
    full_space_energy_target: Optional[float]
    hub_world_m: np.ndarray
    provisional_reasons: Tuple[str, ...]
    warnings: Tuple[str, ...]

    @property
    def production_ready(self) -> bool:
        return not self.provisional_reasons

    def metadata(self) -> Dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
            "source_data_dir": str(self.source_data_dir),
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "rotor_config_path": str(self.rotor_config_path),
            "dynamic_overrides_path": str(self.dynamic_overrides_path),
            "dt_s": self.dt_s,
            "mode_count": self.mode_count,
            "full_space_energy_target": self.full_space_energy_target,
            "hub_world_m": self.hub_world_m.tolist(),
            "production_ready": self.production_ready,
            "provisional_reasons": list(self.provisional_reasons),
            "warnings": list(self.warnings),
        }


def load_study_context(
    config_path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    allow_provisional: bool = False,
) -> StudyContext:
    """Load and validate one study file.

    ``allow_provisional`` does not erase any warning.  It only allows a caller
    to continue while the metadata remains explicitly non-production.
    """

    config_file = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else config_file.parent.parent.resolve()
    )
    config = _read_json(config_file)
    if config.get("schema_version") != "nato-gd-rvmd-3d-v1":
        raise StudyConfigurationError("unsupported or missing schema_version")
    study = _require_mapping(config, "study")
    coordinates = _require_mapping(config, "coordinates")
    rvmd = _require_mapping(config, "rvmd")
    rotor_model = _require_mapping(config, "rotor_model")
    body = _require_mapping(config, "body_response")
    gate = _require_mapping(config, "production_gate")

    dt_s = _finite_positive(study.get("sample_interval_s"), "study.sample_interval_s")
    mode_count = int(study.get("mode_count", -1))
    if mode_count <= 0 or int(rvmd.get("mode_count", -1)) != mode_count:
        raise StudyConfigurationError(
            "study.mode_count and rvmd.mode_count must be the same positive integer"
        )
    raw_energy_target = rvmd.get("full_space_energy_target")
    full_space_energy_target: Optional[float]
    if raw_energy_target is None:
        full_space_energy_target = None
    else:
        full_space_energy_target = float(raw_energy_target)
        if (
            not math.isfinite(full_space_energy_target)
            or full_space_energy_target <= 0.0
            or full_space_energy_target > 1.0
        ):
            raise StudyConfigurationError(
                "rvmd.full_space_energy_target must lie in (0, 1]"
            )
    reference_length = _finite_positive(
        study.get("strouhal_reference_length_m"), "Strouhal reference length"
    )
    reference_velocity = _finite_positive(
        study.get("strouhal_reference_velocity_mps"), "Strouhal reference velocity"
    )
    if not math.isclose(reference_length, 13.716, rel_tol=0.0, abs_tol=1.0e-12):
        raise StudyConfigurationError("Strouhal reference length must be 13.716 m")
    if not math.isclose(reference_velocity, 20.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise StudyConfigurationError("Strouhal reference velocity must be 20 m/s")

    hub = np.asarray(coordinates.get("initial_hub_world_m"), dtype=float)
    if hub.shape != (3,) or not np.all(np.isfinite(hub)):
        raise StudyConfigurationError("coordinates.initial_hub_world_m must be three finite values")
    if coordinates.get("out_of_bounds_policy") != "hard_fail_no_clip_no_extrapolation":
        raise StudyConfigurationError("flow sampling must hard-fail outside the CFD domain")
    start = float(body.get("analysis_window_start_s", float("nan")))
    end = float(body.get("analysis_window_end_s", float("nan")))
    pre_roll = float(body.get("pre_roll_s", float("nan")))
    if not all(math.isfinite(value) for value in (start, end, pre_roll)):
        raise StudyConfigurationError("body time-window settings must be finite")
    if pre_roll < 0.0 or start < 0.0 or end <= start:
        raise StudyConfigurationError("invalid pre-roll or common analysis window")
    if pre_roll > start + 1e-12:
        raise StudyConfigurationError(
            "pre_roll_s must not extend beyond analysis_window_start_s"
        )
    if body.get("common_window_required") is not True:
        raise StudyConfigurationError("common_window_required must be true")
    if rotor_model.get("flapping_moment_transfer") != "prohibited":
        raise StudyConfigurationError("legacy flapping_moment_transfer must be prohibited")
    if rotor_model.get("hub_moment_policy") != "force_line_only":
        raise StudyConfigurationError("primary hub moment policy must be force_line_only")

    source_dir = _resolve_path(root, study.get("source_data_dir"), "source_data_dir")
    output_dir = _resolve_path(root, study.get("output_dir"), "output_dir")
    cache_dir = _resolve_path(root, study.get("cache_dir"), "cache_dir")
    rotor_config = _resolve_path(root, rotor_model.get("base_rotor_config"), "base_rotor_config")
    overrides = _resolve_path(root, rotor_model.get("dynamic_overrides"), "dynamic_overrides")
    if not source_dir.is_dir():
        raise StudyConfigurationError("source data directory is missing: {}".format(source_dir))
    for required_file in (rotor_config, overrides):
        if not required_file.is_file():
            raise StudyConfigurationError("required configuration is missing: {}".format(required_file))

    provisional: List[str] = []
    warnings: List[str] = []
    hub_status = str(coordinates.get("hub_position_status", ""))
    if gate.get("require_confirmed_hub_world_position", True) and (
        "pending" in hub_status.lower() or "inferred" in hub_status.lower()
    ):
        provisional.append("rotor hub position is inferred and awaits user confirmation")
    dt_status = str(study.get("sample_interval_status", ""))
    if gate.get("require_confirmed_sample_interval", True) and not dt_status:
        provisional.append("sampling interval has no provenance status")

    label = str(source_dir.parent.name)
    match = re.search(r"(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s", label, re.IGNORECASE)
    first_id = int(study.get("first_snapshot_id", 0))
    last_id = int(study.get("last_snapshot_id", -1))
    if last_id < first_id:
        raise StudyConfigurationError("last_snapshot_id must not precede first_snapshot_id")
    expected_frames = last_id - first_id + 1
    if match:
        labelled_span = float(match.group(2)) - float(match.group(1))
        # N snapshots contain N-1 time intervals between the first and last
        # stored frame.  Using N*dt here would introduce an off-by-one warning
        # even though the actual relative time axis is built correctly.
        implied_span = max(expected_frames - 1, 0) * dt_s
        if not math.isclose(labelled_span, implied_span, rel_tol=0.0, abs_tol=max(dt_s, 1e-6)):
            warnings.append(
                "directory label implies {:.9g} s between first and last frame, whereas "
                "{} frames at configured dt imply {:.9g} s over {} intervals; frequencies "
                "follow explicit dt={:.9g} s".format(
                    labelled_span, expected_frames, implied_span, max(expected_frames - 1, 0), dt_s
                )
            )

    context = StudyContext(
        config_path=config_file,
        project_root=root,
        config=config,
        source_data_dir=source_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        rotor_config_path=rotor_config,
        dynamic_overrides_path=overrides,
        dt_s=dt_s,
        mode_count=mode_count,
        full_space_energy_target=full_space_energy_target,
        hub_world_m=hub,
        provisional_reasons=tuple(provisional),
        warnings=tuple(warnings),
    )
    if provisional and not allow_provisional:
        raise StudyConfigurationError(
            "production gates are unresolved: {}. Re-run with an explicit provisional override "
            "only for labelled exploratory output.".format("; ".join(provisional))
        )
    return context


def initialise_output_tree(context: StudyContext) -> List[Path]:
    created = []
    for relative in OUTPUT_SUBDIRECTORIES:
        path = context.output_dir / Path(relative)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def validate_grid_and_time(
    context: StudyContext,
    bounds_xyz_m: Sequence[Sequence[float]],
    frame_count: int,
    rotor_diameters_m: Sequence[float],
) -> Dict[str, Any]:
    """Check the common time window and every initial horizontal rotor disk."""

    bounds = np.asarray(bounds_xyz_m, dtype=float)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise StudyConfigurationError("bounds_xyz_m must have shape (3,2) with increasing bounds")
    if int(frame_count) != frame_count or int(frame_count) < 2:
        raise StudyConfigurationError("frame_count must be an integer of at least two")
    hub = context.hub_world_m
    if np.any(hub < bounds[:, 0]) or np.any(hub > bounds[:, 1]):
        raise StudyConfigurationError("initial hub lies outside the CFD volume")
    clearances = np.array(
        [hub[0] - bounds[0, 0], bounds[0, 1] - hub[0], hub[1] - bounds[1, 0], bounds[1, 1] - hub[1]],
        dtype=float,
    )
    radii = 0.5 * np.asarray(rotor_diameters_m, dtype=float)
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise StudyConfigurationError("rotor diameters must be finite and positive")
    if np.max(radii) > np.min(clearances) + 1e-12:
        raise StudyConfigurationError("at least one initial horizontal rotor disk leaves the CFD x-y bounds")
    available_end = (int(frame_count) - 1) * context.dt_s
    body = context.config["body_response"]
    required_end = float(body["analysis_window_end_s"])
    if required_end > available_end + 1e-12:
        raise StudyConfigurationError(
            "common response window ends at {:.9g} s but CFD ends at {:.9g} s".format(
                required_end, available_end
            )
        )
    return {
        "passed": True,
        "bounds_xyz_m": bounds.tolist(),
        "hub_world_m": hub.tolist(),
        "initial_xy_clearances_m": clearances.tolist(),
        "maximum_rotor_radius_m": float(np.max(radii)),
        "frame_count": int(frame_count),
        "relative_time_end_s": available_end,
        "required_simulation_end_s": required_end,
    }


def build_stage_rows(
    frequencies_hz: Sequence[float], strouhal: Sequence[float]
) -> List[Dict[str, Any]]:
    """Return the cumulative high-frequency-deletion stage contract."""

    frequency = np.asarray(frequencies_hz, dtype=float)
    st = np.asarray(strouhal, dtype=float)
    if frequency.ndim != 1 or frequency.size < 1 or st.shape != frequency.shape:
        raise StudyConfigurationError(
            "one or more paired frequencies and Strouhal values are required"
        )
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(st)):
        raise StudyConfigurationError("frequency table contains NaN or infinity")
    if np.any(np.diff(frequency) < 0.0) or np.any(np.diff(st) < -1e-12):
        raise StudyConfigurationError("modes must be sorted from low to high frequency")
    mode_count = int(frequency.size)
    index_width = max(2, len(str(mode_count)))
    rows: List[Dict[str, Any]] = []
    for stage in range(mode_count + 1):
        retained = mode_count - stage
        deleted_mode = None if stage == 0 else mode_count + 1 - stage
        cutoff_frequency = None if retained == 0 else float(frequency[retained - 1])
        cutoff_st = None if retained == 0 else float(st[retained - 1])
        rows.append(
            {
                "stage_index": stage,
                "case_id": "stage_{stage:0{width}d}_retain_{retained:0{width}d}".format(
                    stage=stage, retained=retained, width=index_width
                ),
                "retained_mode_count": retained,
                "deleted_mode_this_stage": deleted_mode,
                "deleted_frequency_hz_this_stage": (
                    None if deleted_mode is None else float(frequency[deleted_mode - 1])
                ),
                "deleted_St_this_stage": None if deleted_mode is None else float(st[deleted_mode - 1]),
                "cutoff_frequency_hz": cutoff_frequency,
                "cutoff_St": cutoff_st,
                "retained_mode_indices_one_based": list(range(1, retained + 1)),
            }
        )
    return rows


def write_stage_manifest(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage_index",
        "case_id",
        "retained_mode_count",
        "deleted_mode_this_stage",
        "deleted_frequency_hz_this_stage",
        "deleted_St_this_stage",
        "cutoff_frequency_hz",
        "cutoff_St",
        "retained_mode_indices_one_based",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            row = dict(item)
            row["retained_mode_indices_one_based"] = ";".join(
                str(value) for value in row["retained_mode_indices_one_based"]
            )
            writer.writerow(row)
    return output


def write_context_provenance(context: StudyContext) -> Path:
    initialise_output_tree(context)
    return _write_json(context.output_dir / "00_provenance" / "study_context.json", context.metadata())


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args(argv)
    context = load_study_context(args.config, allow_provisional=args.allow_provisional)
    write_context_provenance(context)
    print(json.dumps(context.metadata(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())


__all__ = [
    "OUTPUT_SUBDIRECTORIES",
    "StudyConfigurationError",
    "StudyContext",
    "build_stage_rows",
    "initialise_output_tree",
    "load_study_context",
    "validate_grid_and_time",
    "write_context_provenance",
    "write_stage_manifest",
]
