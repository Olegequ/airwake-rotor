# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Strict, resumable I/O for the NATO-GD three-dimensional CFD export.

The source dataset is a sequence of CSV files.  Every frame is validated before
it is admitted to the cache: filename numbering, exact header, row count,
finite values, and the raw coordinate order must all agree.  The cache stores
three time-major ``float32`` NPY memmaps with shape ``(nt, nz, ny, nx)``.

Coordinates passed to :class:`NatoGD3DCache` are always ordered ``(x, y, z)``.
Interpolation never clips or extrapolates: one out-of-domain point fails the
whole request with :class:`CoordinateOutOfBoundsError`.

Examples
--------
Scan every source frame without creating a cache::

    python nato_gd_3d_io.py scan --source "data/cfd" --dt 0.01

Build (or safely resume) the cache::

    python nato_gd_3d_io.py build-cache --source "data/cfd" \
        --cache "runs/research/01_cache" --dt 0.01

Run read-only cache checks::

    python nato_gd_3d_io.py smoke --cache "runs/research/01_cache" --dt 0.01

The ``--dt`` option is intentionally mandatory for all three commands.  A
wrong time step must not be inferred from filenames and silently propagated
into modal frequencies.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CACHE_FORMAT = "nato-gd-3d-cache-v1"
DEFAULT_PREFIX = "流场数据_table_"
DEFAULT_START_ID = 2001
DEFAULT_END_ID = 4500
EXPECTED_HEADER = (
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Pressure (Pa)",
    "X (m)",
    "Y (m)",
    "Z (m)",
)
VELOCITY_COLUMNS = EXPECTED_HEADER[:3]
COORDINATE_COLUMNS = EXPECTED_HEADER[4:]


class DatasetValidationError(RuntimeError):
    """Raised when a source frame or cache violates a required invariant."""


class CoordinateOutOfBoundsError(ValueError):
    """Raised instead of clipping or extrapolating an interpolation request."""


ProgressCallback = Optional[Callable[[int, int, Path], None]]


def _require_positive_finite_dt(dt_s: float) -> float:
    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and strictly positive")
    return dt_s


def _require_coordinate_atol(coordinate_atol: float) -> float:
    coordinate_atol = float(coordinate_atol)
    if not math.isfinite(coordinate_atol) or coordinate_atol < 0.0:
        raise ValueError("coordinate_atol must be finite and non-negative")
    return coordinate_atol


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON by atomic replacement, keeping interrupted state readable."""

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def _json_read(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DatasetValidationError("JSON root must be an object: {}".format(path))
    return value


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(entries: Sequence[Dict[str, Any]]) -> str:
    encoded = json.dumps(
        list(entries), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_frames(
    source_dir: Path,
    prefix: str = DEFAULT_PREFIX,
    expected_start_id: int = DEFAULT_START_ID,
    expected_end_id: int = DEFAULT_END_ID,
) -> Tuple[List[Tuple[int, Path]], List[Dict[str, Any]], str]:
    """Discover an exact, gap-free CSV sequence and return a stable manifest."""

    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise DatasetValidationError("source directory does not exist: {}".format(source_dir))
    expected_start_id = int(expected_start_id)
    expected_end_id = int(expected_end_id)
    if expected_end_id < expected_start_id:
        raise ValueError("expected_end_id must be >= expected_start_id")
    if not prefix:
        raise ValueError("filename prefix must not be empty")

    matcher = re.compile(r"^{}([0-9]+)\.csv$".format(re.escape(prefix)))
    csv_paths = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    unexpected = [path.name for path in csv_paths if matcher.match(path.name) is None]
    if unexpected:
        preview = unexpected[:5]
        raise DatasetValidationError(
            "unexpected CSV filename(s) in source directory: {}{}".format(
                preview, " ..." if len(unexpected) > len(preview) else ""
            )
        )

    by_id: Dict[int, Path] = {}
    for path in csv_paths:
        match = matcher.match(path.name)
        if match is None:  # Defensive; handled above.
            continue
        frame_id = int(match.group(1))
        if frame_id in by_id:
            raise DatasetValidationError(
                "duplicate numeric frame id {}: {!s}, {!s}".format(frame_id, by_id[frame_id], path)
            )
        by_id[frame_id] = path.resolve()

    expected_ids = list(range(expected_start_id, expected_end_id + 1))
    missing = [frame_id for frame_id in expected_ids if frame_id not in by_id]
    outside = sorted(frame_id for frame_id in by_id if frame_id < expected_start_id or frame_id > expected_end_id)
    if missing or outside or len(by_id) != len(expected_ids):
        raise DatasetValidationError(
            "frame numbering is not the exact closed interval [{}, {}]: "
            "missing={}{}; outside={}{}; discovered_count={}".format(
                expected_start_id,
                expected_end_id,
                missing[:10],
                " ..." if len(missing) > 10 else "",
                outside[:10],
                " ..." if len(outside) > 10 else "",
                len(by_id),
            )
        )

    frames = [(frame_id, by_id[frame_id]) for frame_id in expected_ids]
    manifest: List[Dict[str, Any]] = []
    for frame_id, path in frames:
        stat = path.stat()
        manifest.append(
            {
                "frame_id": frame_id,
                "name": path.name,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return frames, manifest, _manifest_digest(manifest)


def _read_header(path: Path) -> Tuple[str, ...]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.reader(handle))
    except StopIteration:
        raise DatasetValidationError("empty CSV file: {}".format(path))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError("cannot read CSV header {}: {}".format(path, exc))
    return tuple(row)


def _read_frame(path: Path, expected_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Read one frame; return float32 velocity and float64 raw coordinates."""

    header = _read_header(path)
    if header != EXPECTED_HEADER:
        raise DatasetValidationError(
            "header mismatch in {}: expected {!r}, got {!r}".format(path, EXPECTED_HEADER, header)
        )
    try:
        frame = pd.read_csv(
            str(path),
            header=0,
            engine="c",
            dtype=np.float64,
            usecols=list(EXPECTED_HEADER),
        )
    except Exception as exc:
        raise DatasetValidationError("failed parsing {}: {}".format(path, exc))
    if tuple(frame.columns) != EXPECTED_HEADER:
        raise DatasetValidationError(
            "parsed column order mismatch in {}: {!r}".format(path, tuple(frame.columns))
        )

    values = frame.to_numpy(dtype=np.float64, copy=False)
    n_rows = int(values.shape[0])
    if expected_rows is not None and n_rows != int(expected_rows):
        raise DatasetValidationError(
            "row-count mismatch in {}: expected {}, got {}".format(path, expected_rows, n_rows)
        )
    if values.ndim != 2 or values.shape[1] != len(EXPECTED_HEADER):
        raise DatasetValidationError("unexpected parsed shape in {}: {}".format(path, values.shape))
    finite_mask = np.isfinite(values)
    if not bool(finite_mask.all()):
        bad_row, bad_col = np.argwhere(~finite_mask)[0]
        raise DatasetValidationError(
            "non-finite value in {} at data row {}, column {!r}: {!r}".format(
                path, int(bad_row) + 2, EXPECTED_HEADER[int(bad_col)], values[bad_row, bad_col]
            )
        )

    velocity = np.asarray(values[:, :3], dtype=np.float32)
    if not bool(np.isfinite(velocity).all()):
        bad_row, bad_col = np.argwhere(~np.isfinite(velocity))[0]
        raise DatasetValidationError(
            "velocity overflows float32 in {} at data row {}, column {!r}".format(
                path, int(bad_row) + 2, VELOCITY_COLUMNS[int(bad_col)]
            )
        )
    coordinates = np.asarray(values[:, 4:7], dtype=np.float64)
    return velocity, coordinates


def _regular_axis(values: np.ndarray, name: str) -> Tuple[np.ndarray, float]:
    axis = np.unique(np.asarray(values, dtype=np.float64))
    if axis.size < 2:
        raise DatasetValidationError("{} axis needs at least two distinct points".format(name))
    differences = np.diff(axis)
    if not bool(np.all(differences > 0.0)):
        raise DatasetValidationError("{} axis is not strictly increasing".format(name))
    spacing = float(np.median(differences))
    tolerance = max(1.0e-11, abs(spacing) * 1.0e-10)
    if not bool(np.allclose(differences, spacing, rtol=1.0e-10, atol=tolerance)):
        index = int(np.argmax(np.abs(differences - spacing)))
        raise DatasetValidationError(
            "{} axis is not uniform near index {}: spacing={!r}, reference={!r}".format(
                name, index, float(differences[index]), spacing
            )
        )
    return axis, spacing


def establish_grid(coordinates: np.ndarray) -> Dict[str, Any]:
    """Build axes and raw-row -> C-order ``(z,y,x)`` flattened indices."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise DatasetValidationError("coordinates must have shape (n_points, 3)")
    if not bool(np.isfinite(coordinates).all()):
        raise DatasetValidationError("coordinates contain non-finite values")

    x, dx = _regular_axis(coordinates[:, 0], "x")
    y, dy = _regular_axis(coordinates[:, 1], "y")
    z, dz = _regular_axis(coordinates[:, 2], "z")
    nx, ny, nz = int(x.size), int(y.size), int(z.size)
    expected_points = nx * ny * nz
    if coordinates.shape[0] != expected_points:
        raise DatasetValidationError(
            "coordinates do not form a complete Cartesian grid: rows={}, nx*ny*nz={}*{}*{}={}".format(
                coordinates.shape[0], nx, ny, nz, expected_points
            )
        )

    ix = np.searchsorted(x, coordinates[:, 0]).astype(np.int64)
    iy = np.searchsorted(y, coordinates[:, 1]).astype(np.int64)
    iz = np.searchsorted(z, coordinates[:, 2]).astype(np.int64)
    if (
        bool(np.any(ix >= nx))
        or bool(np.any(iy >= ny))
        or bool(np.any(iz >= nz))
        or not bool(np.array_equal(x[ix], coordinates[:, 0]))
        or not bool(np.array_equal(y[iy], coordinates[:, 1]))
        or not bool(np.array_equal(z[iz], coordinates[:, 2]))
    ):
        raise DatasetValidationError("at least one coordinate does not map exactly to a grid axis")

    raw_to_grid = iz * (ny * nx) + iy * nx + ix
    counts = np.bincount(raw_to_grid, minlength=expected_points)
    if counts.size != expected_points or not bool(np.all(counts == 1)):
        duplicate_count = int(np.count_nonzero(counts > 1))
        missing_count = int(np.count_nonzero(counts == 0))
        raise DatasetValidationError(
            "Cartesian grid contains duplicate or missing coordinates: duplicates={}, missing={}".format(
                duplicate_count, missing_count
            )
        )
    return {
        "x": x,
        "y": y,
        "z": z,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "shape_zyx": (nz, ny, nx),
        "raw_to_grid": raw_to_grid.astype(np.int64, copy=False),
    }


def _assert_same_coordinate_order(
    coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    path: Path,
    coordinate_atol: float = 0.0,
) -> None:
    if coordinates.shape != reference_coordinates.shape:
        raise DatasetValidationError(
            "coordinate shape mismatch in {}: expected {}, got {}".format(
                path, reference_coordinates.shape, coordinates.shape
            )
        )
    coordinate_atol = _require_coordinate_atol(coordinate_atol)
    if coordinate_atol == 0.0:
        same = np.equal(coordinates, reference_coordinates)
    else:
        same = np.isclose(coordinates, reference_coordinates, rtol=0.0, atol=coordinate_atol)
    if not bool(same.all()):
        row, column = np.argwhere(~same)[0]
        raise DatasetValidationError(
            "raw coordinate order/value mismatch in {} at data row {}, axis {}: "
            "expected {!r}, got {!r}".format(
                path,
                int(row) + 2,
                ("X", "Y", "Z")[int(column)],
                float(reference_coordinates[row, column]),
                float(coordinates[row, column]),
            )
        )


def _scan_summary_template(
    source_dir: Path,
    dt_s: float,
    frames: Sequence[Tuple[int, Path]],
    manifest_digest: str,
) -> Dict[str, Any]:
    return {
        "format": "nato-gd-3d-scan-v1",
        "source_dir": str(Path(source_dir).resolve()),
        "dt_s": float(dt_s),
        "frame_count": len(frames),
        "first_frame_id": int(frames[0][0]),
        "last_frame_id": int(frames[-1][0]),
        "time_axis": {
            "relative_time_from_first_frame_seconds": {
                "first": 0.0,
                "last": float((len(frames) - 1) * dt_s),
                "sample_interval": float(dt_s),
            },
            "absolute_source_time_origin_seconds": None,
            "absolute_source_time_status": "unknown_not_inferred_from_frame_id_or_filename",
            "frame_id_times_dt_used_as_absolute_time": False,
        },
        "manifest_sha256": manifest_digest,
    }


def scan_dataset(
    source_dir: Path,
    dt_s: float,
    prefix: str = DEFAULT_PREFIX,
    expected_start_id: int = DEFAULT_START_ID,
    expected_end_id: int = DEFAULT_END_ID,
    coordinate_atol: float = 0.0,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Strictly read and validate every CSV without writing a cache."""

    dt_s = _require_positive_finite_dt(dt_s)
    coordinate_atol = _require_coordinate_atol(coordinate_atol)
    frames, _manifest, manifest_digest = discover_frames(
        source_dir, prefix, expected_start_id, expected_end_id
    )
    first_velocity, reference_coordinates = _read_frame(frames[0][1])
    grid = establish_grid(reference_coordinates)
    rows = int(reference_coordinates.shape[0])
    velocity_min = np.min(first_velocity, axis=0).astype(np.float64)
    velocity_max = np.max(first_velocity, axis=0).astype(np.float64)

    if progress is not None:
        progress(1, len(frames), frames[0][1])
    for frame_index, (_frame_id, path) in enumerate(frames[1:], start=1):
        velocity, coordinates = _read_frame(path, expected_rows=rows)
        _assert_same_coordinate_order(coordinates, reference_coordinates, path, coordinate_atol)
        velocity_min = np.minimum(velocity_min, np.min(velocity, axis=0))
        velocity_max = np.maximum(velocity_max, np.max(velocity, axis=0))
        if progress is not None:
            progress(frame_index + 1, len(frames), path)

    report = _scan_summary_template(source_dir, dt_s, frames, manifest_digest)
    report.update(
        {
            "validated": True,
            "row_count_per_frame": rows,
            "grid_shape_zyx": list(grid["shape_zyx"]),
            "axis_lengths_xyz": [int(grid["x"].size), int(grid["y"].size), int(grid["z"].size)],
            "bounds_xyz_m": [
                [float(grid["x"][0]), float(grid["x"][-1])],
                [float(grid["y"][0]), float(grid["y"][-1])],
                [float(grid["z"][0]), float(grid["z"][-1])],
            ],
            "spacing_xyz_m": [float(grid["dx"]), float(grid["dy"]), float(grid["dz"])],
            "velocity_min_uvw_m_s": velocity_min.tolist(),
            "velocity_max_uvw_m_s": velocity_max.tolist(),
            "coordinate_order_validation": "exact" if coordinate_atol == 0.0 else "absolute_tolerance",
            "coordinate_atol_m": float(coordinate_atol),
        }
    )
    return report


def _known_cache_paths(cache_dir: Path) -> List[Path]:
    names = (
        "metadata.json",
        "manifest.json",
        "build_state.json",
        "x.npy",
        "y.npy",
        "z.npy",
        "raw_coordinates.npy",
        "raw_to_grid.npy",
        "u.npy",
        "v.npy",
        "w.npy",
    )
    return [cache_dir / name for name in names]


def _flush_arrays(arrays: Sequence[np.memmap]) -> None:
    for array in arrays:
        array.flush()


def build_cache(
    source_dir: Path,
    cache_dir: Path,
    dt_s: float,
    prefix: str = DEFAULT_PREFIX,
    expected_start_id: int = DEFAULT_START_ID,
    expected_end_id: int = DEFAULT_END_ID,
    coordinate_atol: float = 0.0,
    flush_every: int = 10,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Build or resume a strict time-major velocity cache.

    Existing completed caches are treated idempotently after their identifying
    metadata are checked.  An interrupted build resumes only if the exact
    filename/size/mtime manifest is unchanged.
    """

    dt_s = _require_positive_finite_dt(dt_s)
    coordinate_atol = _require_coordinate_atol(coordinate_atol)
    flush_every = int(flush_every)
    if flush_every < 1:
        raise ValueError("flush_every must be >= 1")
    source_dir = Path(source_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    frames, manifest, manifest_digest = discover_frames(
        source_dir, prefix, expected_start_id, expected_end_id
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    state_path = cache_dir / "build_state.json"
    manifest_path = cache_dir / "manifest.json"

    if metadata_path.exists():
        metadata = _json_read(metadata_path)
        _validate_completed_identity(metadata, source_dir, dt_s, manifest_digest, len(frames))
        NatoGD3DCache(cache_dir, expected_dt_s=dt_s).validate_storage()
        return metadata

    state: Dict[str, Any]
    arrays: List[np.memmap] = []
    first_frame_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if state_path.exists():
        state = _json_read(state_path)
        _validate_resume_state(
            state, source_dir, dt_s, manifest_digest, len(frames), coordinate_atol
        )
        reference_coordinates = np.load(str(cache_dir / "raw_coordinates.npy"), mmap_mode="r")
        raw_to_grid = np.load(str(cache_dir / "raw_to_grid.npy"), mmap_mode="r")
        axes = {
            "x": np.load(str(cache_dir / "x.npy"), mmap_mode="r"),
            "y": np.load(str(cache_dir / "y.npy"), mmap_mode="r"),
            "z": np.load(str(cache_dir / "z.npy"), mmap_mode="r"),
        }
        shape_zyx = tuple(int(value) for value in state["grid_shape_zyx"])
        start_index = int(state["completed_frames"])
        arrays = [
            np.load(str(cache_dir / "{}.npy".format(name)), mmap_mode="r+")
            for name in ("u", "v", "w")
        ]
        reconstructed = establish_grid(reference_coordinates)
        for name in ("x", "y", "z"):
            if not bool(np.array_equal(np.asarray(axes[name]), reconstructed[name])):
                raise DatasetValidationError(
                    "cannot resume: cached {} axis disagrees with raw coordinates".format(name)
                )
        if tuple(reconstructed["shape_zyx"]) != shape_zyx:
            raise DatasetValidationError("cannot resume: cached grid shape is inconsistent")
        if not bool(
            np.array_equal(np.asarray(raw_to_grid), reconstructed["raw_to_grid"])
        ):
            raise DatasetValidationError(
                "cannot resume: raw_to_grid disagrees with raw coordinates"
            )
        expected_cache_shape = (len(frames),) + shape_zyx
        for name, array in zip(("u", "v", "w"), arrays):
            if array.shape != expected_cache_shape or array.dtype != np.dtype(np.float32):
                raise DatasetValidationError(
                    "cannot resume: {} cache has shape/dtype {}/{}, expected {}/float32".format(
                        name, array.shape, array.dtype, expected_cache_shape
                    )
                )
    else:
        occupied = [path for path in _known_cache_paths(cache_dir) if path.exists()]
        if occupied:
            raise DatasetValidationError(
                "cache directory contains cache artifacts but has neither a complete "
                "metadata file nor a resumable build state: {}".format([path.name for path in occupied])
            )
        first_frame_data = _read_frame(frames[0][1])
        first_velocity, reference_coordinates = first_frame_data
        grid = establish_grid(reference_coordinates)
        raw_to_grid = grid["raw_to_grid"]
        axes = {"x": grid["x"], "y": grid["y"], "z": grid["z"]}
        shape_zyx = tuple(int(value) for value in grid["shape_zyx"])
        cache_shape = (len(frames),) + shape_zyx
        expected_velocity_bytes = int(np.prod(cache_shape, dtype=np.int64)) * 3 * 4
        expected_static_bytes = int(reference_coordinates.nbytes + raw_to_grid.nbytes)
        required_bytes = expected_velocity_bytes + expected_static_bytes + 64 * 1024 * 1024
        free_bytes = int(shutil.disk_usage(str(cache_dir)).free)
        if free_bytes < required_bytes:
            raise DatasetValidationError(
                "insufficient free space for cache: need at least {:.3f} GiB, have {:.3f} GiB".format(
                    required_bytes / float(1024 ** 3), free_bytes / float(1024 ** 3)
                )
            )
        np.save(str(cache_dir / "x.npy"), axes["x"], allow_pickle=False)
        np.save(str(cache_dir / "y.npy"), axes["y"], allow_pickle=False)
        np.save(str(cache_dir / "z.npy"), axes["z"], allow_pickle=False)
        np.save(str(cache_dir / "raw_coordinates.npy"), reference_coordinates, allow_pickle=False)
        np.save(str(cache_dir / "raw_to_grid.npy"), raw_to_grid, allow_pickle=False)
        _atomic_json_write(
            manifest_path,
            {"manifest_sha256": manifest_digest, "frames": manifest},
        )
        arrays = [
            np.lib.format.open_memmap(
                str(cache_dir / "{}.npy".format(name)), mode="w+", dtype=np.float32, shape=cache_shape
            )
            for name in ("u", "v", "w")
        ]
        state = {
            "format": CACHE_FORMAT,
            "complete": False,
            "source_dir": str(source_dir),
            "dt_s": dt_s,
            "manifest_sha256": manifest_digest,
            "frame_count": len(frames),
            "completed_frames": 0,
            "first_frame_id": int(frames[0][0]),
            "last_frame_id": int(frames[-1][0]),
            "row_count_per_frame": int(reference_coordinates.shape[0]),
            "grid_shape_zyx": list(shape_zyx),
            "coordinate_atol_m": float(coordinate_atol),
            "expected_velocity_cache_bytes": expected_velocity_bytes,
            "started_unix_s": time.time(),
            "last_error": None,
        }
        _atomic_json_write(state_path, state)
        start_index = 0

    expected_rows = int(reference_coordinates.shape[0])
    raw_to_grid_array = np.asarray(raw_to_grid, dtype=np.int64)
    grid_buffer = np.empty((3, expected_rows), dtype=np.float32)
    try:
        for frame_index in range(start_index, len(frames)):
            _frame_id, path = frames[frame_index]
            if frame_index == 0 and first_frame_data is not None:
                velocity, coordinates = first_frame_data
            else:
                velocity, coordinates = _read_frame(path, expected_rows=expected_rows)
            _assert_same_coordinate_order(coordinates, reference_coordinates, path, coordinate_atol)
            for component in range(3):
                grid_buffer[component, raw_to_grid_array] = velocity[:, component]
                arrays[component][frame_index, ...] = grid_buffer[component].reshape(shape_zyx)

            completed = frame_index + 1
            if completed % flush_every == 0 or completed == len(frames):
                _flush_arrays(arrays)
                state["completed_frames"] = completed
                state["last_update_unix_s"] = time.time()
                state["last_error"] = None
                _atomic_json_write(state_path, state)
            if progress is not None:
                progress(completed, len(frames), path)
    except Exception as exc:
        _flush_arrays(arrays)
        state["last_update_unix_s"] = time.time()
        state["last_error"] = "{}: {}".format(type(exc).__name__, exc)
        _atomic_json_write(state_path, state)
        raise

    x_array = np.asarray(axes["x"], dtype=np.float64)
    y_array = np.asarray(axes["y"], dtype=np.float64)
    z_array = np.asarray(axes["z"], dtype=np.float64)
    metadata: Dict[str, Any] = {
        "format": CACHE_FORMAT,
        "complete": True,
        "source_dir": str(source_dir),
        "dt_s": dt_s,
        "frame_count": len(frames),
        "first_frame_id": int(frames[0][0]),
        "last_frame_id": int(frames[-1][0]),
        "frame_ids_are_contiguous": True,
        "time_axis": {
            "relative_time_from_first_frame_seconds": {
                "first": 0.0,
                "last": float((len(frames) - 1) * dt_s),
                "sample_interval": float(dt_s),
            },
            "absolute_source_time_origin_seconds": None,
            "absolute_source_time_status": "unknown_not_inferred_from_frame_id_or_filename",
            "frame_id_times_dt_used_as_absolute_time": False,
        },
        "manifest_sha256": manifest_digest,
        "row_count_per_frame": expected_rows,
        "grid_shape_zyx": list(shape_zyx),
        "array_shape_tzyx": [len(frames)] + list(shape_zyx),
        "array_dtype": "float32",
        "array_order": "C",
        "coordinate_input_order": ["x", "y", "z"],
        "velocity_output_order": ["u", "v", "w"],
        "bounds_xyz_m": [
            [float(x_array[0]), float(x_array[-1])],
            [float(y_array[0]), float(y_array[-1])],
            [float(z_array[0]), float(z_array[-1])],
        ],
        "spacing_xyz_m": [
            float(x_array[1] - x_array[0]),
            float(y_array[1] - y_array[0]),
            float(z_array[1] - z_array[0]),
        ],
        "coordinate_order_validation": "exact" if coordinate_atol == 0.0 else "absolute_tolerance",
        "coordinate_atol_m": float(coordinate_atol),
        "velocity_cache_bytes": int(np.prod((len(frames),) + shape_zyx, dtype=np.int64)) * 3 * 4,
        "files": {
            "u": "u.npy",
            "v": "v.npy",
            "w": "w.npy",
            "x": "x.npy",
            "y": "y.npy",
            "z": "z.npy",
            "raw_coordinates": "raw_coordinates.npy",
            "raw_to_grid": "raw_to_grid.npy",
            "manifest": "manifest.json",
        },
        "sha256": {
            "x": _sha256_file(cache_dir / "x.npy"),
            "y": _sha256_file(cache_dir / "y.npy"),
            "z": _sha256_file(cache_dir / "z.npy"),
            "raw_coordinates": _sha256_file(cache_dir / "raw_coordinates.npy"),
            "raw_to_grid": _sha256_file(cache_dir / "raw_to_grid.npy"),
        },
        "completed_unix_s": time.time(),
    }
    _atomic_json_write(metadata_path, metadata)
    state["complete"] = True
    state["completed_frames"] = len(frames)
    state["last_update_unix_s"] = time.time()
    state["last_error"] = None
    _atomic_json_write(state_path, state)
    return metadata


def _validate_completed_identity(
    metadata: Dict[str, Any],
    source_dir: Path,
    dt_s: float,
    manifest_digest: str,
    frame_count: int,
) -> None:
    checks = {
        "format": CACHE_FORMAT,
        "complete": True,
        "source_dir": str(source_dir),
        "manifest_sha256": manifest_digest,
        "frame_count": int(frame_count),
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise DatasetValidationError(
                "completed cache identity mismatch for {!r}: expected {!r}, got {!r}".format(
                    key, expected, metadata.get(key)
                )
            )
    if not math.isclose(float(metadata.get("dt_s", float("nan"))), dt_s, rel_tol=0.0, abs_tol=1.0e-15):
        raise DatasetValidationError("completed cache dt does not match requested dt")


def normalize_cache_time_metadata(
    cache_dir: Path, absolute_source_time_origin_s: Optional[float] = None
) -> Dict[str, Any]:
    """Atomically migrate legacy cache time metadata without touching arrays.

    Numeric frame identifiers establish order only.  They are never multiplied
    by ``dt`` to invent an absolute CFD time.  An absolute origin is recorded
    solely when the caller supplies it explicitly.
    """

    directory = Path(cache_dir).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    metadata = _json_read(metadata_path)
    if metadata.get("format") != CACHE_FORMAT or metadata.get("complete") is not True:
        raise DatasetValidationError("cache is not a completed {} cache".format(CACHE_FORMAT))
    dt_s = _require_positive_finite_dt(float(metadata["dt_s"]))
    frame_count = int(metadata["frame_count"])
    if frame_count < 2:
        raise DatasetValidationError("completed cache has fewer than two frames")
    if absolute_source_time_origin_s is not None:
        origin = float(absolute_source_time_origin_s)
        if not math.isfinite(origin):
            raise DatasetValidationError("absolute source-time origin must be finite")
        status = "explicit_user_supplied"
    else:
        origin = None
        status = "unknown_not_inferred_from_frame_id_or_filename"
    for obsolete in (
        "relative_time_start_s",
        "relative_time_end_s",
        "source_time_definition",
        "source_time_start_s",
        "source_time_end_s",
    ):
        metadata.pop(obsolete, None)
    metadata["time_axis"] = {
        "relative_time_from_first_frame_seconds": {
            "first": 0.0,
            "last": float((frame_count - 1) * dt_s),
            "sample_interval": dt_s,
        },
        "absolute_source_time_origin_seconds": origin,
        "absolute_source_time_status": status,
        "frame_id_times_dt_used_as_absolute_time": False,
    }
    _atomic_json_write(metadata_path, metadata)
    return metadata


def _validate_resume_state(
    state: Dict[str, Any],
    source_dir: Path,
    dt_s: float,
    manifest_digest: str,
    frame_count: int,
    coordinate_atol: float,
) -> None:
    checks = {
        "format": CACHE_FORMAT,
        "source_dir": str(source_dir),
        "manifest_sha256": manifest_digest,
        "frame_count": int(frame_count),
    }
    for key, expected in checks.items():
        if state.get(key) != expected:
            raise DatasetValidationError(
                "cannot resume: state mismatch for {!r}: expected {!r}, got {!r}".format(
                    key, expected, state.get(key)
                )
            )
    if not math.isclose(float(state.get("dt_s", float("nan"))), dt_s, rel_tol=0.0, abs_tol=1.0e-15):
        raise DatasetValidationError("cannot resume: dt mismatch")
    if not math.isclose(
        float(state.get("coordinate_atol_m", float("nan"))),
        coordinate_atol,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise DatasetValidationError("cannot resume: coordinate_atol mismatch")
    completed = int(state.get("completed_frames", -1))
    if completed < 0 or completed > frame_count:
        raise DatasetValidationError("cannot resume: invalid completed_frames={}".format(completed))


class NatoGD3DCache(object):
    """Read-only, bounds-safe access to a completed NATO-GD cache."""

    def __init__(self, cache_dir: Path, expected_dt_s: Optional[float] = None):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.metadata = _json_read(self.cache_dir / "metadata.json")
        if self.metadata.get("format") != CACHE_FORMAT or self.metadata.get("complete") is not True:
            raise DatasetValidationError("cache is not a completed {} cache".format(CACHE_FORMAT))
        self.dt_s = _require_positive_finite_dt(float(self.metadata["dt_s"]))
        if expected_dt_s is not None:
            expected_dt_s = _require_positive_finite_dt(expected_dt_s)
            if not math.isclose(self.dt_s, expected_dt_s, rel_tol=0.0, abs_tol=1.0e-15):
                raise DatasetValidationError(
                    "cache dt_s={!r} differs from required dt_s={!r}".format(self.dt_s, expected_dt_s)
                )
        self.x = np.load(str(self.cache_dir / "x.npy"), mmap_mode="r")
        self.y = np.load(str(self.cache_dir / "y.npy"), mmap_mode="r")
        self.z = np.load(str(self.cache_dir / "z.npy"), mmap_mode="r")
        self.u = np.load(str(self.cache_dir / "u.npy"), mmap_mode="r")
        self.v = np.load(str(self.cache_dir / "v.npy"), mmap_mode="r")
        self.w = np.load(str(self.cache_dir / "w.npy"), mmap_mode="r")
        self.frame_count = int(self.metadata["frame_count"])
        self.shape_zyx = tuple(int(value) for value in self.metadata["grid_shape_zyx"])

    @property
    def relative_time_end_s(self) -> float:
        return float((self.frame_count - 1) * self.dt_s)

    @property
    def source_time_start_s(self) -> float:
        time_axis = self.metadata.get("time_axis", {})
        value = time_axis.get("absolute_source_time_origin_seconds")
        if value is None:
            raise DatasetValidationError(
                "absolute CFD source-time origin is unknown; use relative time"
            )
        return float(value)

    def validate_storage(self, verify_static_hashes: bool = True) -> Dict[str, Any]:
        expected_shape = (self.frame_count,) + self.shape_zyx
        for name, array in (("u", self.u), ("v", self.v), ("w", self.w)):
            if array.shape != expected_shape:
                raise DatasetValidationError(
                    "{} cache shape mismatch: expected {}, got {}".format(name, expected_shape, array.shape)
                )
            if array.dtype != np.dtype(np.float32):
                raise DatasetValidationError(
                    "{} cache dtype mismatch: expected float32, got {}".format(name, array.dtype)
                )
        for name, axis, expected_length in (
            ("x", self.x, self.shape_zyx[2]),
            ("y", self.y, self.shape_zyx[1]),
            ("z", self.z, self.shape_zyx[0]),
        ):
            if axis.ndim != 1 or axis.size != expected_length:
                raise DatasetValidationError("{} axis shape is inconsistent with grid".format(name))
            if not bool(np.isfinite(axis).all()) or not bool(np.all(np.diff(axis) > 0.0)):
                raise DatasetValidationError("{} axis is non-finite or non-monotonic".format(name))
        if verify_static_hashes:
            expected_hashes = self.metadata.get("sha256", {})
            for name in ("x", "y", "z", "raw_coordinates", "raw_to_grid"):
                expected = expected_hashes.get(name)
                if not isinstance(expected, str):
                    raise DatasetValidationError("missing static SHA-256 for {}".format(name))
                actual = _sha256_file(self.cache_dir / self.metadata["files"][name])
                if actual != expected:
                    raise DatasetValidationError(
                        "static cache file checksum mismatch for {}".format(name)
                    )
        return {
            "valid": True,
            "shape_tzyx": list(expected_shape),
            "dtype": "float32",
            "dt_s": self.dt_s,
        }

    @staticmethod
    def _normalise_points(points_xyz: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], bool]:
        points = np.asarray(points_xyz, dtype=np.float64)
        single = points.ndim == 1
        if single:
            if points.shape != (3,):
                raise ValueError("one point must have shape (3,)")
            batch_shape: Tuple[int, ...] = ()
            flat = points.reshape(1, 3)
        else:
            if points.ndim < 2 or points.shape[-1] != 3:
                raise ValueError("batch points must have shape (..., 3)")
            batch_shape = tuple(int(value) for value in points.shape[:-1])
            flat = points.reshape(-1, 3)
        if flat.shape[0] == 0:
            raise ValueError("point batch must not be empty")
        if not bool(np.isfinite(flat).all()):
            row, column = np.argwhere(~np.isfinite(flat))[0]
            raise CoordinateOutOfBoundsError(
                "point {} has non-finite {} coordinate".format(int(row), ("x", "y", "z")[int(column)])
            )
        return flat, batch_shape, single

    def _check_bounds(self, flat_points: np.ndarray) -> None:
        lower = np.array([self.x[0], self.y[0], self.z[0]], dtype=np.float64)
        upper = np.array([self.x[-1], self.y[-1], self.z[-1]], dtype=np.float64)
        invalid = (flat_points < lower) | (flat_points > upper)
        if bool(invalid.any()):
            row, column = np.argwhere(invalid)[0]
            axis = ("x", "y", "z")[int(column)]
            raise CoordinateOutOfBoundsError(
                "point batch index {} is outside the {} domain: value={!r}, bounds=[{!r}, {!r}]; "
                "clipping and extrapolation are disabled".format(
                    int(row),
                    axis,
                    float(flat_points[row, column]),
                    float(lower[column]),
                    float(upper[column]),
                )
            )

    @staticmethod
    def _brackets(axis: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        upper = np.searchsorted(axis, query, side="right").astype(np.int64)
        at_upper_bound = upper == axis.size
        upper[at_upper_bound] = axis.size - 1
        lower = upper - 1
        lower[lower < 0] = 0
        zero_width = upper == lower
        if bool(zero_width.any()):
            # This occurs only at the lower bound; use its first cell with weight zero.
            upper[zero_width] = lower[zero_width] + 1
        denominator = axis[upper] - axis[lower]
        fraction = (query - axis[lower]) / denominator
        return lower, upper, fraction

    def _interpolate_one_array(
        self, field: np.ndarray, frame_index: int, flat_points: np.ndarray
    ) -> np.ndarray:
        ix0, ix1, fx = self._brackets(self.x, flat_points[:, 0])
        iy0, iy1, fy = self._brackets(self.y, flat_points[:, 1])
        iz0, iz1, fz = self._brackets(self.z, flat_points[:, 2])

        c000 = field[frame_index, iz0, iy0, ix0].astype(np.float64)
        c001 = field[frame_index, iz0, iy0, ix1].astype(np.float64)
        c010 = field[frame_index, iz0, iy1, ix0].astype(np.float64)
        c011 = field[frame_index, iz0, iy1, ix1].astype(np.float64)
        c100 = field[frame_index, iz1, iy0, ix0].astype(np.float64)
        c101 = field[frame_index, iz1, iy0, ix1].astype(np.float64)
        c110 = field[frame_index, iz1, iy1, ix0].astype(np.float64)
        c111 = field[frame_index, iz1, iy1, ix1].astype(np.float64)

        x00 = c000 * (1.0 - fx) + c001 * fx
        x01 = c010 * (1.0 - fx) + c011 * fx
        x10 = c100 * (1.0 - fx) + c101 * fx
        x11 = c110 * (1.0 - fx) + c111 * fx
        y0 = x00 * (1.0 - fy) + x01 * fy
        y1 = x10 * (1.0 - fy) + x11 * fy
        return y0 * (1.0 - fz) + y1 * fz

    def interpolate_frame(self, frame_index: int, points_xyz: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate a batch of points in one stored frame."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
            raise TypeError("frame_index must be an integer")
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(
                "frame_index {} outside [0, {}]".format(frame_index, self.frame_count - 1)
            )
        flat, batch_shape, single = self._normalise_points(points_xyz)
        self._check_bounds(flat)
        output = np.column_stack(
            [
                self._interpolate_one_array(field, frame_index, flat)
                for field in (self.u, self.v, self.w)
            ]
        )
        if single:
            return output[0]
        return output.reshape(batch_shape + (3,))

    def interpolate_time(
        self, relative_time_s: float, points_xyz: np.ndarray
    ) -> np.ndarray:
        """Linearly interpolate in time around spatially trilinear samples.

        ``relative_time_s=0`` corresponds to the first cached frame.  Time
        extrapolation is prohibited just like spatial extrapolation.
        """

        relative_time_s = float(relative_time_s)
        if not math.isfinite(relative_time_s):
            raise CoordinateOutOfBoundsError("relative_time_s must be finite")
        last_time = self.relative_time_end_s
        if relative_time_s < 0.0 or relative_time_s > last_time:
            raise CoordinateOutOfBoundsError(
                "relative_time_s={!r} outside [0.0, {!r}]; extrapolation is disabled".format(
                    relative_time_s, last_time
                )
            )
        position = relative_time_s / self.dt_s
        lower = int(math.floor(position))
        if lower >= self.frame_count - 1:
            return self.interpolate_frame(self.frame_count - 1, points_xyz)
        fraction = position - lower
        first = self.interpolate_frame(lower, points_xyz)
        if fraction == 0.0:
            return first
        second = self.interpolate_frame(lower + 1, points_xyz)
        return first * (1.0 - fraction) + second * fraction


def smoke_cache(cache_dir: Path, dt_s: float) -> Dict[str, Any]:
    """Run deterministic, read-only structural and interpolation checks."""

    cache = NatoGD3DCache(cache_dir, expected_dt_s=dt_s)
    storage = cache.validate_storage(verify_static_hashes=True)
    centre = np.array(
        [
            0.5 * (float(cache.x[0]) + float(cache.x[-1])),
            0.5 * (float(cache.y[0]) + float(cache.y[-1])),
            0.5 * (float(cache.z[0]) + float(cache.z[-1])),
        ],
        dtype=np.float64,
    )
    corners_and_centre = np.array(
        [
            [cache.x[0], cache.y[0], cache.z[0]],
            [cache.x[-1], cache.y[-1], cache.z[-1]],
            centre,
        ],
        dtype=np.float64,
    )
    sampled_frames = sorted(set([0, cache.frame_count // 2, cache.frame_count - 1]))
    extrema = []
    for frame_index in sampled_frames:
        values = cache.interpolate_frame(frame_index, corners_and_centre)
        if not bool(np.isfinite(values).all()):
            raise DatasetValidationError(
                "non-finite interpolated velocity in smoke frame {}".format(frame_index)
            )
        extrema.append(
            {
                "frame_index": frame_index,
                "min_uvw_m_s": np.min(values, axis=0).tolist(),
                "max_uvw_m_s": np.max(values, axis=0).tolist(),
            }
        )
    return {
        "smoke_passed": True,
        "storage": storage,
        "sampled_frames": extrema,
        "bounds_policy": "hard-fail (no clip, no extrapolation)",
    }


def _stderr_progress(done: int, total: int, path: Path) -> None:
    if done == 1 or done == total or done % 25 == 0:
        print(
            "validated/written {}/{}: {}".format(done, total, path.name),
            file=sys.stderr,
            flush=True,
        )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, type=Path, help="directory containing source CSV files")
    parser.add_argument("--dt", required=True, type=float, help="physical frame interval in seconds")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="exact filename prefix before the numeric id")
    parser.add_argument("--start-id", type=int, default=DEFAULT_START_ID)
    parser.add_argument("--end-id", type=int, default=DEFAULT_END_ID)
    parser.add_argument(
        "--coordinate-atol",
        type=float,
        default=0.0,
        help="absolute tolerance for cross-frame coordinate equality; default 0 is exact",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    scan_parser = subparsers.add_parser("scan", help="strictly validate all source frames")
    _add_source_arguments(scan_parser)
    scan_parser.add_argument("--report", type=Path, help="optional JSON report path")

    build_parser = subparsers.add_parser("build-cache", help="build or resume float32 velocity memmaps")
    _add_source_arguments(build_parser)
    build_parser.add_argument("--cache", required=True, type=Path, help="cache output directory")
    build_parser.add_argument("--flush-every", type=int, default=10, help="flush/state interval in frames")

    smoke_parser = subparsers.add_parser("smoke", help="validate and sample a completed cache")
    smoke_parser.add_argument("--cache", required=True, type=Path, help="completed cache directory")
    smoke_parser.add_argument("--dt", required=True, type=float, help="required physical frame interval in seconds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "scan":
        report = scan_dataset(
            args.source,
            args.dt,
            prefix=args.prefix,
            expected_start_id=args.start_id,
            expected_end_id=args.end_id,
            coordinate_atol=args.coordinate_atol,
            progress=_stderr_progress,
        )
        if args.report is not None:
            report_path = args.report.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json_write(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "build-cache":
        metadata = build_cache(
            args.source,
            args.cache,
            args.dt,
            prefix=args.prefix,
            expected_start_id=args.start_id,
            expected_end_id=args.end_id,
            coordinate_atol=args.coordinate_atol,
            flush_every=args.flush_every,
            progress=_stderr_progress,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        report = smoke_cache(args.cache, args.dt)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DatasetValidationError, CoordinateOutOfBoundsError, ValueError, OSError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(2)
