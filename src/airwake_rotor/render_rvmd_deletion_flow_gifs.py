#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Render one cumulative high-frequency-deletion RVMD GIF per selected mode.

Each GIF contains exactly 200 common physical-time frames and three mutually
orthogonal slices through the configured rotor hub.  All GIFs share one speed
colour scale so apparent changes can be compared across deletion stages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.linewidth"] = 0.7

from .nato_gd_study import build_stage_rows, load_study_context
from .rvmd_3d_flow import RVMD3DFlow


@dataclass(frozen=True)
class OrthogonalSlices:
    xy_mean: np.ndarray
    xy_modes: np.ndarray
    xz_mean: np.ndarray
    xz_modes: np.ndarray
    yz_mean: np.ndarray
    yz_modes: np.ndarray
    ix: int
    iy: int
    iz: int


def nearest_index(axis: np.ndarray, value: float) -> int:
    if value < float(axis[0]) or value > float(axis[-1]):
        raise ValueError("slice coordinate lies outside axis bounds")
    return int(np.argmin(np.abs(np.asarray(axis, dtype=float) - float(value))))


def extract_orthogonal_slices(flow: RVMD3DFlow, hub_xyz_m: Sequence[float]) -> OrthogonalSlices:
    hub = np.asarray(hub_xyz_m, dtype=float)
    if hub.shape != (3,) or not np.all(np.isfinite(hub)):
        raise ValueError("hub_xyz_m must be three finite coordinates")
    ix = nearest_index(flow.x, hub[0])
    iy = nearest_index(flow.y, hub[1])
    iz = nearest_index(flow.z, hub[2])

    def fields(rows: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        flat = rows.reshape(-1)
        mean = np.asarray(flow.mean_field[flat], dtype=np.float64).reshape(shape + (3,))
        modes = np.asarray(flow.spatial_modes[flat], dtype=np.float64).reshape(
            shape + (3, flow.mode_count)
        )
        return mean, modes

    xy_mean, xy_modes = fields(flow.row_index_zyx[iz, :, :], (flow.y.size, flow.x.size))
    xz_mean, xz_modes = fields(flow.row_index_zyx[:, iy, :], (flow.z.size, flow.x.size))
    yz_mean, yz_modes = fields(flow.row_index_zyx[:, :, ix], (flow.z.size, flow.y.size))
    return OrthogonalSlices(
        xy_mean, xy_modes, xz_mean, xz_modes, yz_mean, yz_modes, ix, iy, iz
    )


def reconstruct_slice(
    mean: np.ndarray,
    modes: np.ndarray,
    centered_coefficients: np.ndarray,
    retained_mode_count: int,
) -> np.ndarray:
    if mean.ndim != 3 or mean.shape[-1] != 3:
        raise ValueError("mean slice must have shape (a,b,3)")
    if modes.shape[:3] != mean.shape or modes.ndim != 4:
        raise ValueError("mode slice must have shape (a,b,3,k)")
    coefficient = np.asarray(centered_coefficients, dtype=float)
    if coefficient.shape != (modes.shape[-1],):
        raise ValueError("coefficient vector length disagrees with modal slice")
    retained = int(retained_mode_count)
    if retained < 0 or retained > modes.shape[-1]:
        raise ValueError("retained_mode_count is outside [0,k]")
    if retained == 0:
        return np.asarray(mean, dtype=np.float64).copy()
    return mean + np.tensordot(
        modes[..., :retained], coefficient[:retained], axes=([-1], [0])
    )


def selected_frame_indices(frame_count: int, requested_frames: int = 200) -> np.ndarray:
    if frame_count < requested_frames:
        raise ValueError(
            "{} source frames cannot provide {} unique GIF frames".format(
                frame_count, requested_frames
            )
        )
    indices = np.rint(np.linspace(0, frame_count - 1, requested_frames)).astype(np.int64)
    if np.unique(indices).size != requested_frames:
        raise RuntimeError("frame selection did not produce unique indices")
    return indices


def _reconstruct_all(slices: OrthogonalSlices, coefficient: np.ndarray, retained: int) -> Tuple[np.ndarray, ...]:
    return (
        reconstruct_slice(slices.xy_mean, slices.xy_modes, coefficient, retained),
        reconstruct_slice(slices.xz_mean, slices.xz_modes, coefficient, retained),
        reconstruct_slice(slices.yz_mean, slices.yz_modes, coefficient, retained),
    )


def estimate_common_speed_limits(
    flow: RVMD3DFlow,
    slices: OrthogonalSlices,
    frame_indices: np.ndarray,
) -> Tuple[float, float, Dict[str, Any]]:
    retained_candidates = sorted(
        set(np.rint(np.linspace(0, flow.mode_count, 7)).astype(int).tolist())
    )
    samples: List[np.ndarray] = []
    frame_stride = max(1, int(math.ceil(frame_indices.size / 40.0)))
    for frame_index in frame_indices[::frame_stride]:
        coefficient = flow.coefficients[frame_index] - flow.coefficient_mean
        for retained in retained_candidates:
            velocity_fields = _reconstruct_all(slices, coefficient, retained)
            for velocity in velocity_fields:
                speed = np.linalg.norm(velocity, axis=-1)
                samples.append(speed[::3, ::3].reshape(-1))
    values = np.concatenate(samples)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite reconstructed speed encountered during colour audit")
    lower = max(0.0, float(np.percentile(values, 0.5)))
    upper = float(np.percentile(values, 99.5))
    if not upper > lower:
        upper = lower + 1.0
    return lower, upper, {
        "policy": "shared robust percentile limits",
        "percentiles": [0.5, 99.5],
        "sample_count": int(values.size),
        "retained_mode_counts_sampled": retained_candidates,
        "vmin_mps": lower,
        "vmax_mps": upper,
        "clipping_is_visual_only": True,
        "underlying_reconstruction_not_clipped": True,
    }


def _quiver_steps(shape: Tuple[int, int], target: int = 14) -> Tuple[int, int]:
    return max(1, int(math.ceil(shape[0] / target))), max(1, int(math.ceil(shape[1] / target)))


def render_frame(
    flow: RVMD3DFlow,
    slices: OrthogonalSlices,
    velocity_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    retained: int,
    stage: int,
    time_s: float,
    cutoff_st: Optional[float],
    vmin: float,
    vmax: float,
    hub_xyz_m: Sequence[float],
) -> np.ndarray:
    xy, xz, yz = velocity_fields
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("cividis")
    figure = plt.figure(figsize=(183.0 / 25.4, 105.0 / 25.4), dpi=120)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], wspace=0.30, hspace=0.42)
    axes = [figure.add_subplot(grid[:, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 1])]
    panels = (
        (axes[0], xy, flow.x, flow.y, (0, 1), "x (m)", "y (m)", "a", "Horizontal, z={:.1f} m".format(flow.z[slices.iz])),
        (axes[1], xz, flow.x, flow.z, (0, 2), "x (m)", "z (m)", "b", "Longitudinal, y={:.1f} m".format(flow.y[slices.iy])),
        (axes[2], yz, flow.y, flow.z, (1, 2), "y (m)", "z (m)", "c", "Transverse, x={:.1f} m".format(flow.x[slices.ix])),
    )
    for axis, velocity, horizontal, vertical, components, xlabel, ylabel, label, title in panels:
        speed = np.linalg.norm(velocity, axis=-1)
        axis.imshow(
            speed,
            origin="lower",
            extent=[horizontal[0], horizontal[-1], vertical[0], vertical[-1]],
            aspect="auto",
            cmap=cmap,
            norm=normalizer,
            interpolation="nearest",
        )
        sy, sx = _quiver_steps(speed.shape)
        x_pick = np.asarray(horizontal)[::sx]
        y_pick = np.asarray(vertical)[::sy]
        xx, yy = np.meshgrid(x_pick, y_pick)
        axis.quiver(
            xx,
            yy,
            velocity[::sy, ::sx, components[0]],
            velocity[::sy, ::sx, components[1]],
            color="white",
            alpha=0.72,
            scale=None,
            scale_units="xy",
            angles="xy",
            width=0.003,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=7.0)
        axis.text(-0.10, 1.02, label, transform=axis.transAxes, fontweight="bold", fontsize=8.0)
        axis.tick_params(direction="out", length=2.5, width=0.7)
    hub = np.asarray(hub_xyz_m, dtype=float)
    axes[0].plot(hub[0], hub[1], marker="+", color="#E64B35", ms=7, mew=1.2)
    axes[1].plot(hub[0], hub[2], marker="+", color="#E64B35", ms=7, mew=1.2)
    axes[2].plot(hub[1], hub[2], marker="+", color="#E64B35", ms=7, mew=1.2)
    scalar = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axes, fraction=0.028, pad=0.035)
    colorbar.set_label("Speed (m/s)", fontsize=7.0)
    st_text = "mean only" if cutoff_st is None else "cutoff St={:.4g}".format(cutoff_st)
    figure.suptitle(
        "NATO-GD RVMD | deletion stage {} | retained {}/{} | {} | t={:.4f} s".format(
            stage, retained, flow.mode_count, st_text, time_s
        ),
        fontsize=8.0,
        y=0.99,
    )
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(figure)
    return image


def _render_stage_worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Render one deletion stage in an isolated process (Matplotlib is not thread-safe)."""

    flow = RVMD3DFlow(Path(str(task["artifact_directory"])))
    path = Path(str(task["path"]))
    temporary = path.with_name(path.stem + ".tmp.gif")
    indices = np.asarray(task["frame_indices"], dtype=np.int64)
    hub_xyz_m = np.asarray(task["hub_xyz_m"], dtype=float)
    retained = int(task["retained_mode_count"])
    stage = int(task["stage_index"])
    deleted = int(task["deleted_mode"])
    started = time.monotonic()
    try:
        slices = extract_orthogonal_slices(flow, hub_xyz_m)
        with imageio.get_writer(
            str(temporary), mode="I", format="GIF",
            duration=1.0 / float(task["fps"]), loop=0,
        ) as writer:
            for number, frame_index in enumerate(indices):
                coefficient = flow.coefficients[frame_index] - flow.coefficient_mean
                fields = _reconstruct_all(slices, coefficient, retained)
                frame_image = render_frame(
                    flow,
                    slices,
                    fields,
                    retained=retained,
                    stage=stage,
                    time_s=float(frame_index * flow.dt_s),
                    cutoff_st=task["cutoff_St"],
                    vmin=float(task["vmin_mps"]),
                    vmax=float(task["vmax_mps"]),
                    hub_xyz_m=hub_xyz_m,
                )
                writer.append_data(frame_image)
                if (number + 1) % 25 == 0:
                    print(
                        "GIF stage {:02d}: {}/{} frames".format(
                            stage, number + 1, indices.size
                        ),
                        flush=True,
                    )
        temporary.replace(path)
    finally:
        flow.close()
        if temporary.exists():
            temporary.unlink()
    validation = validate_gif(path, int(indices.size))
    return {
        "stage_index": stage,
        "deleted_mode": deleted,
        "retained_mode_count": retained,
        "path": str(path),
        "frames": int(indices.size),
        "pixel_width": validation["pixel_width"],
        "pixel_height": validation["pixel_height"],
        "elapsed_wall_s": time.monotonic() - started,
        "status": "rendered",
    }


def validate_gif(path: Path, expected_frames: int = 200) -> Dict[str, int]:
    """Verify that an animation is readable and contains the exact frame contract."""

    try:
        with Image.open(str(path)) as image:
            count = int(getattr(image, "n_frames", 1))
            width, height = (int(value) for value in image.size)
            image.seek(count - 1)
            image.convert("RGB").getpixel((width - 1, height - 1))
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError("unreadable GIF {}: {}".format(path, exc)) from exc
    if count != int(expected_frames):
        raise ValueError(
            "GIF {} contains {} frames, expected {}".format(path, count, expected_frames)
        )
    if width <= 0 or height <= 0:
        raise ValueError("GIF has invalid pixel dimensions: {}".format(path))
    return {"frame_count": count, "pixel_width": width, "pixel_height": height}


def render_all_deletion_gifs(
    flow: RVMD3DFlow,
    output_directory: Path,
    hub_xyz_m: Sequence[float],
    *,
    frames: int = 200,
    fps: float = 10.0,
    overwrite: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    if frames != 200:
        raise ValueError("the requested study contract requires exactly 200 frames per GIF")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    worker_count = int(workers)
    if worker_count < 1 or worker_count > 8:
        raise ValueError("workers must lie in [1,8]")
    output_directory.mkdir(parents=True, exist_ok=True)
    indices = selected_frame_indices(flow.frame_count, frames)
    slices = extract_orthogonal_slices(flow, hub_xyz_m)
    vmin, vmax, scale_audit = estimate_common_speed_limits(flow, slices, indices)
    stages = build_stage_rows(flow.frequency_hz, flow.strouhal)
    outputs: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    started_all = time.monotonic()
    index_width = max(2, len(str(flow.mode_count)))
    for stage in range(1, flow.mode_count + 1):
        row = stages[stage]
        retained = int(row["retained_mode_count"])
        deleted = int(row["deleted_mode_this_stage"])
        path = output_directory / (
            "stage_{stage:0{width}d}_delete_mode_{deleted:0{width}d}_"
            "retain_{retained:0{width}d}.gif"
        ).format(
            stage=stage,
            deleted=deleted,
            retained=retained,
            width=index_width,
        )
        if path.exists() and not overwrite:
            try:
                existing = validate_gif(path, frames)
            except ValueError:
                print("replace invalid existing GIF {}".format(path.name), flush=True)
            else:
                outputs.append(
                    {
                        "stage_index": stage,
                        "deleted_mode": deleted,
                        "retained_mode_count": retained,
                        "path": str(path),
                        "frames": frames,
                        "pixel_width": existing["pixel_width"],
                        "pixel_height": existing["pixel_height"],
                        "status": "existing_validated",
                    }
                )
                continue
        tasks.append(
            {
                "artifact_directory": str(flow.artifact_directory),
                "stage_index": stage,
                "deleted_mode": deleted,
                "retained_mode_count": retained,
                "path": str(path),
                "frame_indices": indices.tolist(),
                "hub_xyz_m": np.asarray(hub_xyz_m, dtype=float).tolist(),
                "fps": float(fps),
                "cutoff_St": row["cutoff_St"],
                "vmin_mps": vmin,
                "vmax_mps": vmax,
            }
        )
    if worker_count == 1:
        outputs.extend(_render_stage_worker(task) for task in tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=min(worker_count, max(1, len(tasks)))
        ) as executor:
            futures = [executor.submit(_render_stage_worker, task) for task in tasks]
            for future in as_completed(futures):
                outputs.append(future.result())
    outputs.sort(key=lambda item: int(item["stage_index"]))
    audit = {
        "schema_version": 1,
        "gif_count": flow.mode_count,
        "frames_per_gif": frames,
        "common_frame_indices": indices.tolist(),
        "common_relative_times_s": (indices.astype(float) * flow.dt_s).tolist(),
        "views": ["xy through hub z", "xz through hub y", "yz through hub x"],
        "scalar": "velocity magnitude in m/s",
        "vectors": "in-plane velocity components",
        "hub_marker": "red plus",
        "colour_scale": scale_audit,
        "parallel_process_workers": worker_count,
        "outputs": outputs,
        "elapsed_wall_s": time.monotonic() - started_all,
    }
    audit_path = output_directory / "gif_audit.json"
    with audit_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    context = load_study_context(args.config, allow_provisional=args.allow_provisional)
    flow = RVMD3DFlow(context.output_dir / "02_rvmd")
    audit = render_all_deletion_gifs(
        flow,
        context.output_dir / "07_flow_gifs_200frames",
        context.hub_world_m,
        frames=args.frames,
        fps=args.fps,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    flow.close()
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "OrthogonalSlices",
    "estimate_common_speed_limits",
    "extract_orthogonal_slices",
    "nearest_index",
    "reconstruct_slice",
    "render_all_deletion_gifs",
    "selected_frame_indices",
    "validate_gif",
]
