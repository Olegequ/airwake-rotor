# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Bounded-memory NATO-GD 3-D CFD adapter for RVMD.

The 62 GB CSV series is too large for an in-memory full-state RVMD solve.
This adapter therefore implements a documented computational approximation:

1. identify the temporal coefficients from either a deterministic CountSketch
   of *all* selected spatial degrees of freedom (default), or from a recorded
   stratified point sample;
2. optionally project that identification matrix onto a randomized POD basis;
3. run the reference-aligned RVMD core; and
4. recover physical full-space modes in a second streaming pass using blocked
   least-squares sufficient statistics.

The mean field is stored separately and does not consume one of the K RVMD
modes.  Variable-wise fluctuation RMS scaling prevents unlike physical units
from being mixed without an explicit metric.  The output metadata records
every compression dimension, seed, hash rule, and retained-energy audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .rvmd_core import RVMDResult, energy_audit, rvmd


COLUMN_NAMES: Mapping[str, str] = {
    "u": "Velocity[i] (m/s)",
    "v": "Velocity[j] (m/s)",
    "w": "Velocity[k] (m/s)",
    "p": "Pressure (Pa)",
    "x": "X (m)",
    "y": "Y (m)",
    "z": "Z (m)",
}

HASH_DESCRIPTION = (
    "SplitMix64 over zero-based point-major state indices; one +1/-1 entry "
    "per input column; independent local bucket spaces for each variable"
)


def resolve_compute_backend(requested: str) -> Tuple[str, str]:
    """Resolve the adapter's GPU-preferred backend policy.

    ``auto`` selects CUDA only when the current Python environment contains a
    CUDA-enabled PyTorch build.  An explicit ``torch-cuda`` request remains
    strict and is validated by :func:`rvmd`; it never silently falls back.
    """

    policy = str(requested).strip().lower()
    if policy not in ("auto", "numpy", "torch-cuda"):
        raise ValueError("compute_backend must be auto, numpy, or torch-cuda")
    if policy != "auto":
        return policy, "explicit backend request"
    try:
        import torch
    except (ImportError, OSError) as exception:
        return "numpy", "auto fallback: PyTorch unavailable ({})".format(
            type(exception).__name__
        )
    if bool(torch.cuda.is_available()):
        return "torch-cuda", "auto selected CUDA-enabled PyTorch"
    return "numpy", "auto fallback: torch.cuda.is_available() is false"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_sha256(value: np.ndarray, block_rows: int = 4096) -> str:
    """Hash an array's dtype, shape, and values in bounded row blocks."""

    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(b"nato-gd-rvmd-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        for start in range(0, int(array.shape[0]), block_rows):
            digest.update(np.ascontiguousarray(array[start : start + block_rows]).tobytes())
    return digest.hexdigest()


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(dict(payload)), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
    return output


def _natural_key(path: Path) -> Tuple[Any, ...]:
    pieces = re.split(r"(\d+)", path.stem)
    return tuple(int(piece) if piece.isdigit() else piece.lower() for piece in pieces)


def _frame_id(path: Path) -> Optional[int]:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


@dataclass
class DatasetInventory:
    input_directory: Path
    paths: List[Path]
    frame_ids: List[Optional[int]]
    header: List[str]
    n_points: int
    coordinates: np.ndarray
    variables: Tuple[str, ...]
    variable_columns: Tuple[str, ...]
    coordinate_columns: Tuple[str, str, str]

    @property
    def n_time(self) -> int:
        return len(self.paths)


class NatoGDCsvSeries:
    """Validated time-ordered view of the NATO-GD CSV snapshots."""

    def __init__(
        self,
        input_directory: Union[str, Path],
        variables: Sequence[str] = ("u", "v", "w"),
        pattern: str = "*.csv",
        frame_limit: Optional[int] = None,
        strict_contiguous_ids: bool = True,
        reader_engine: str = "auto",
    ) -> None:
        directory = Path(input_directory)
        if not directory.is_dir():
            raise FileNotFoundError("input directory does not exist: {}".format(directory))
        paths = sorted(directory.glob(pattern), key=_natural_key)
        if frame_limit is not None:
            if frame_limit < 2:
                raise ValueError("frame_limit must be at least two")
            paths = paths[:frame_limit]
        if len(paths) < 2:
            raise ValueError("at least two CSV frames are required")
        selected = tuple(str(item).lower() for item in variables)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("variables must be a nonempty list without duplicates")
        if any(item not in ("u", "v", "w", "p") for item in selected):
            raise ValueError("variables may only contain u, v, w, and p")
        ids = [_frame_id(path) for path in paths]
        if strict_contiguous_ids and all(item is not None for item in ids):
            expected = list(range(int(ids[0]), int(ids[0]) + len(ids)))
            if ids != expected:
                raise ValueError("CSV frame identifiers are not contiguous")
        with paths[0].open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
        required = [COLUMN_NAMES[item] for item in selected] + [
            COLUMN_NAMES["x"], COLUMN_NAMES["y"], COLUMN_NAMES["z"]
        ]
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError("missing CSV columns: {}".format(", ".join(missing)))
        self.input_directory = directory.resolve()
        self.paths = [path.resolve() for path in paths]
        self.frame_ids = ids
        self.header = header
        self.variables = selected
        self.variable_columns = tuple(COLUMN_NAMES[item] for item in selected)
        self.coordinate_columns = (COLUMN_NAMES["x"], COLUMN_NAMES["y"], COLUMN_NAMES["z"])
        if reader_engine not in ("auto", "pandas", "numpy"):
            raise ValueError("reader_engine must be auto, pandas, or numpy")
        self.reader_engine = reader_engine
        first_values, first_coordinates = self.read_frame(0, include_coordinates=True)
        self._first_values = first_values
        self.coordinates = first_coordinates
        self.n_points = int(first_values.shape[0])
        self.sentinel_indices = deterministic_sentinel_indices(self.n_points, 96)

    def inventory(self) -> DatasetInventory:
        return DatasetInventory(
            input_directory=self.input_directory,
            paths=self.paths,
            frame_ids=self.frame_ids,
            header=self.header,
            n_points=self.n_points,
            coordinates=self.coordinates,
            variables=self.variables,
            variable_columns=self.variable_columns,
            coordinate_columns=self.coordinate_columns,
        )

    def _read_pandas(self, path: Path, names: Sequence[str]) -> np.ndarray:
        try:
            import pandas as pd
        except ImportError as exception:  # pragma: no cover - depends on runtime
            raise RuntimeError("pandas reader requested but pandas is not installed") from exception
        frame = pd.read_csv(
            str(path),
            usecols=list(names),
            dtype={name: np.float32 for name in names},
            engine="c",
        )
        return np.column_stack([frame[name].to_numpy(copy=False) for name in names]).astype(
            np.float32, copy=False
        )

    def _read_numpy(self, path: Path, names: Sequence[str]) -> np.ndarray:
        indices = [self.header.index(name) for name in names]
        values = np.loadtxt(
            str(path), delimiter=",", skiprows=1, usecols=indices, dtype=np.float32, ndmin=2
        )
        return values

    def _read_columns(self, path: Path, names: Sequence[str]) -> np.ndarray:
        engine = self.reader_engine
        if engine == "auto":
            try:
                return self._read_pandas(path, names)
            except (ImportError, RuntimeError):
                return self._read_numpy(path, names)
        if engine == "pandas":
            return self._read_pandas(path, names)
        return self._read_numpy(path, names)

    def read_frame(
        self, index: int, include_coordinates: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        path = self.paths[index]
        names: Tuple[str, ...]
        if include_coordinates:
            names = self.variable_columns + self.coordinate_columns
        else:
            names = self.variable_columns
        data = self._read_columns(path, names)
        if data.ndim != 2 or data.shape[1] != len(names):
            raise ValueError("unexpected CSV matrix shape in {}".format(path))
        if not np.all(np.isfinite(data)):
            raise ValueError("NaN or infinite data in {}".format(path))
        values = data[:, : len(self.variables)]
        coordinates = data[:, len(self.variables) :] if include_coordinates else None
        if values.shape[0] != getattr(self, "n_points", values.shape[0]):
            raise ValueError("point count changed in {}".format(path))
        return values, coordinates

    def read_validated_frame(
        self, index: int, coordinate_check_interval: int = 1
    ) -> np.ndarray:
        check = index == 0 or index == len(self.paths) - 1
        if coordinate_check_interval > 0 and index % coordinate_check_interval == 0:
            check = True
        if index == 0 and hasattr(self, "_first_values"):
            values = self._first_values
            coordinates = self.coordinates
        else:
            values, coordinates = self.read_frame(index, include_coordinates=check)
        if check:
            assert coordinates is not None
            observed = coordinates[self.sentinel_indices]
            expected = self.coordinates[self.sentinel_indices]
            if not np.array_equal(observed, expected):
                max_error = float(np.max(np.abs(observed.astype(np.float64) - expected)))
                raise ValueError(
                    "coordinate row order changed in {} (sentinel max error {})".format(
                        self.paths[index], max_error
                    )
                )
        return values


class NatoGDCacheSeries:
    """Read the validated time-major 3-D cache in original CSV row order."""

    def __init__(
        self,
        cache_directory: Union[str, Path],
        variables: Sequence[str] = ("u", "v", "w"),
        frame_limit: Optional[int] = None,
        expected_dt: Optional[float] = None,
    ) -> None:
        directory = Path(cache_directory).expanduser().resolve()
        metadata_path = directory / "metadata.json"
        manifest_path = directory / "manifest.json"
        if not metadata_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("validated cache metadata or manifest is missing")
        metadata = _load_metadata(metadata_path)
        if metadata.get("format") != "nato-gd-3d-cache-v1" or metadata.get("complete") is not True:
            raise ValueError("cache is incomplete or has an unsupported format")
        if expected_dt is not None and not math.isclose(
            float(metadata.get("dt_s", float("nan"))),
            float(expected_dt),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("cache dt disagrees with the requested RVMD dt")
        selected = tuple(str(item).lower() for item in variables)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("variables must be a nonempty list without duplicates")
        if any(item not in ("u", "v", "w") for item in selected):
            raise ValueError("the 3-D cache contains only u, v, and w")
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        frames = manifest.get("frames") if isinstance(manifest, dict) else None
        if not isinstance(frames, list) or len(frames) != int(metadata.get("frame_count", -1)):
            raise ValueError("cache manifest frame count is inconsistent")
        if frame_limit is not None:
            if frame_limit < 2:
                raise ValueError("frame_limit must be at least two")
            frames = frames[: int(frame_limit)]
        source = Path(str(metadata["source_dir"]))
        self.input_directory = source
        self.paths = [(source / str(item["name"])).resolve() for item in frames]
        self.frame_ids = [int(item["frame_id"]) for item in frames]
        self.header = [COLUMN_NAMES[key] for key in ("u", "v", "w", "p", "x", "y", "z")]
        self.variables = selected
        self.variable_columns = tuple(COLUMN_NAMES[item] for item in selected)
        self.coordinate_columns = (COLUMN_NAMES["x"], COLUMN_NAMES["y"], COLUMN_NAMES["z"])
        self.coordinates = np.load(str(directory / "raw_coordinates.npy"), mmap_mode="r")
        self.raw_to_grid = np.load(str(directory / "raw_to_grid.npy"), mmap_mode="r")
        self._components = {
            name: np.load(str(directory / (name + ".npy")), mmap_mode="r")
            for name in ("u", "v", "w")
        }
        self.n_points = int(self.coordinates.shape[0])
        declared_rows = int(metadata.get("row_count_per_frame", -1))
        if self.coordinates.shape != (declared_rows, 3) or self.n_points <= 0:
            raise ValueError("cache coordinate shape is inconsistent")
        if self.raw_to_grid.shape != (self.n_points,):
            raise ValueError("cache raw_to_grid shape is inconsistent")
        expected_shape = tuple(int(item) for item in metadata["array_shape_tzyx"])
        for name, array in self._components.items():
            if tuple(array.shape) != expected_shape:
                raise ValueError("cache {} array shape is inconsistent".format(name))
        if len(self.paths) > expected_shape[0]:
            raise ValueError("cache frame selection exceeds stored arrays")
        if np.unique(np.asarray(self.raw_to_grid)).size != self.n_points:
            raise ValueError("cache raw_to_grid is not a permutation")
        self.sentinel_indices = deterministic_sentinel_indices(self.n_points, 96)
        self.cache_directory = directory
        self.cache_metadata = metadata

    def inventory(self) -> DatasetInventory:
        return DatasetInventory(
            input_directory=self.input_directory,
            paths=self.paths,
            frame_ids=self.frame_ids,
            header=self.header,
            n_points=self.n_points,
            coordinates=self.coordinates,
            variables=self.variables,
            variable_columns=self.variable_columns,
            coordinate_columns=self.coordinate_columns,
        )

    def read_frame(
        self, index: int, include_coordinates: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mapping = np.asarray(self.raw_to_grid, dtype=np.int64)
        columns = [
            np.asarray(self._components[name][index]).reshape(-1)[mapping]
            for name in self.variables
        ]
        values = np.column_stack(columns).astype(np.float32, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError("NaN or infinite data in cached frame {}".format(index))
        return values, self.coordinates if include_coordinates else None

    def read_validated_frame(
        self, index: int, coordinate_check_interval: int = 1
    ) -> np.ndarray:
        del coordinate_check_interval
        values, _ = self.read_frame(index, include_coordinates=False)
        return values

    def close(self) -> None:
        for array in (self.coordinates, self.raw_to_grid) + tuple(self._components.values()):
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __enter__(self) -> "NatoGDCacheSeries":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.close()

def deterministic_sentinel_indices(n_points: int, count: int) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    count = min(max(int(count), 2), n_points)
    return np.unique(np.rint(np.linspace(0, n_points - 1, count)).astype(np.int64))


def infer_cartesian_grid(coordinates: np.ndarray) -> Dict[str, Any]:
    """Audit whether coordinates form a complete Cartesian grid."""

    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must be a finite n_points-by-3 array")
    axes = [np.unique(coords[:, axis]) for axis in range(3)]
    shape = tuple(int(axis.size) for axis in axes)
    complete_size = int(np.prod(shape, dtype=np.int64))
    spacings: List[Optional[float]] = []
    uniform = True
    for axis in axes:
        if axis.size <= 1:
            spacings.append(None)
            continue
        differences = np.diff(axis)
        spacing = float(np.median(differences))
        spacings.append(spacing)
        # CSV values are intentionally read as float32 for bounded memory.  A
        # nominal 0.5 m spacing around x=190 m can consequently differ by
        # roughly 1.5e-5 m after subtraction, so the grid audit must use a
        # tolerance consistent with the stored precision.
        uniform = uniform and bool(np.allclose(differences, spacing, rtol=1e-4, atol=2e-5))
    complete = complete_size == coords.shape[0]
    if complete:
        ix = np.searchsorted(axes[0], coords[:, 0])
        iy = np.searchsorted(axes[1], coords[:, 1])
        iz = np.searchsorted(axes[2], coords[:, 2])
        flat = (ix * shape[1] + iy) * shape[2] + iz
        complete = np.unique(flat).size == coords.shape[0]
    return {
        "shape_nx_ny_nz": list(shape),
        "n_cartesian_cells": complete_size,
        "is_complete_cartesian_product": bool(complete),
        "is_uniform_each_axis": bool(uniform),
        "axis_min": [float(axis[0]) for axis in axes],
        "axis_max": [float(axis[-1]) for axis in axes],
        "axis_spacing": spacings,
    }


def stratified_point_indices(coordinates: np.ndarray, target_points: int) -> np.ndarray:
    """Choose deterministic full-domain points and return original row indices."""

    coords = np.asarray(coordinates, dtype=np.float64)
    n_points = coords.shape[0]
    if target_points <= 0 or target_points > n_points:
        raise ValueError("target_points must be between one and the number of points")
    if target_points == n_points:
        return np.arange(n_points, dtype=np.int64)
    axes = [np.unique(coords[:, axis]) for axis in range(3)]
    shape = tuple(axis.size for axis in axes)
    complete = int(np.prod(shape, dtype=np.int64)) == n_points
    if complete:
        ix = np.searchsorted(axes[0], coords[:, 0])
        iy = np.searchsorted(axes[1], coords[:, 1])
        iz = np.searchsorted(axes[2], coords[:, 2])
        flat = (ix * shape[1] + iy) * shape[2] + iz
        row_of_cell = np.full(n_points, -1, dtype=np.int64)
        row_of_cell[flat] = np.arange(n_points, dtype=np.int64)
        complete = bool(np.all(row_of_cell >= 0))
    if not complete:
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        positions = np.rint(np.linspace(0, n_points - 1, target_points)).astype(np.int64)
        return np.sort(order[positions])

    nz_take = min(shape[2], max(1, int(round(target_points ** (1.0 / 3.0)))))
    remaining = max(target_points / float(nz_take), 1.0)
    ratio = max(shape[0] / float(shape[1]), np.finfo(float).tiny)
    nx_take = min(shape[0], max(1, int(round(math.sqrt(remaining * ratio)))))
    ny_take = min(shape[1], max(1, int(target_points // max(nx_take * nz_take, 1))))
    x_pick = np.unique(np.rint(np.linspace(0, shape[0] - 1, nx_take)).astype(np.int64))
    y_pick = np.unique(np.rint(np.linspace(0, shape[1] - 1, ny_take)).astype(np.int64))
    z_pick = np.unique(np.rint(np.linspace(0, shape[2] - 1, nz_take)).astype(np.int64))
    selected_flat = np.asarray(
        [(x * shape[1] + y) * shape[2] + z for x in x_pick for y in y_pick for z in z_pick],
        dtype=np.int64,
    )
    selected = list(row_of_cell[selected_flat])
    if len(selected) < target_points:
        chosen = np.zeros(n_points, dtype=bool)
        chosen[np.asarray(selected, dtype=np.int64)] = True
        remaining_rows = row_of_cell[~chosen[row_of_cell]]
        needed = target_points - len(selected)
        positions = np.rint(np.linspace(0, remaining_rows.size - 1, needed)).astype(np.int64)
        selected.extend(remaining_rows[positions].tolist())
    return np.sort(np.asarray(selected[:target_points], dtype=np.int64))


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=False)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def countsketch_map(
    n_points: int, n_variables: int, rows_per_variable: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return stable per-variable CountSketch buckets and signs."""

    if n_points <= 0 or n_variables <= 0 or rows_per_variable <= 0:
        raise ValueError("CountSketch dimensions must be positive")
    point = np.arange(n_points, dtype=np.uint64)
    buckets = np.empty((n_variables, n_points), dtype=np.int64)
    signs = np.empty((n_variables, n_points), dtype=np.float64)
    occupancy: List[Dict[str, int]] = []
    for variable_index in range(n_variables):
        state_index = point * np.uint64(n_variables) + np.uint64(variable_index)
        bucket_hash = _splitmix64(state_index ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF))
        sign_hash = _splitmix64(
            state_index ^ np.uint64((seed + 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF)
        )
        buckets[variable_index] = (bucket_hash % np.uint64(rows_per_variable)).astype(np.int64)
        signs[variable_index] = np.where((sign_hash & np.uint64(1)) == 0, -1.0, 1.0)
        counts = np.bincount(buckets[variable_index], minlength=rows_per_variable)
        occupancy.append(
            {
                "empty_buckets": int(np.count_nonzero(counts == 0)),
                "minimum_occupancy": int(np.min(counts)),
                "maximum_occupancy": int(np.max(counts)),
            }
        )
    metadata = {
        "method": "deterministic_countsketch",
        "hash": HASH_DESCRIPTION,
        "seed": int(seed),
        "rows_per_variable": int(rows_per_variable),
        "input_spatial_rows": int(n_points * n_variables),
        "output_sketch_rows": int(rows_per_variable * n_variables),
        "variable_bucket_occupancy": occupancy,
        "all_input_degrees_of_freedom_participate": True,
        "computational_approximation": True,
    }
    return buckets, signs, metadata


def apply_countsketch_frame(
    values: np.ndarray, buckets: np.ndarray, signs: np.ndarray, rows_per_variable: int
) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or buckets.shape != signs.shape or buckets.shape != values.T.shape:
        raise ValueError("values, buckets, and signs have inconsistent shapes")
    output = np.empty(values.shape[1] * rows_per_variable, dtype=np.float64)
    for variable_index in range(values.shape[1]):
        start = variable_index * rows_per_variable
        output[start : start + rows_per_variable] = np.bincount(
            buckets[variable_index],
            weights=signs[variable_index] * values[:, variable_index],
            minlength=rows_per_variable,
        )
    return output


def randomized_pod_projection(
    q: np.ndarray,
    rank: int,
    seed: int = 0,
    oversampling: int = 12,
    power_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return an orthonormal spatial basis and POD score matrix."""

    matrix = np.asarray(q)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("q must be a finite two-dimensional matrix")
    maximum_rank = min(matrix.shape)
    if rank <= 0 or rank > maximum_rank:
        raise ValueError("rank must be between one and min(q.shape)")
    sample_rank = min(maximum_rank, rank + max(int(oversampling), 0))
    rng = np.random.RandomState(seed)
    random_matrix = rng.standard_normal((matrix.shape[1], sample_rank)).astype(matrix.dtype)
    y = matrix @ random_matrix
    for _ in range(max(int(power_iterations), 0)):
        q_basis, _ = np.linalg.qr(y, mode="reduced")
        y = matrix @ (matrix.T @ q_basis)
    q_basis, _ = np.linalg.qr(y, mode="reduced")
    small = q_basis.T @ matrix
    left, singular_values, right_t = np.linalg.svd(small, full_matrices=False)
    basis = (q_basis @ left[:, :rank]).astype(matrix.dtype, copy=False)
    scores = (singular_values[:rank, None] * right_t[:rank, :]).astype(matrix.dtype, copy=False)
    total_energy = float(np.sum(np.asarray(matrix, dtype=np.float64) ** 2))
    retained_energy = float(np.sum(np.asarray(scores, dtype=np.float64) ** 2))
    metadata = {
        "method": "randomized_spatial_pod_projection",
        "input_rows": int(matrix.shape[0]),
        "output_rows": int(rank),
        "rank": int(rank),
        "oversampling": int(oversampling),
        "power_iterations": int(power_iterations),
        "seed": int(seed),
        "input_energy": total_energy,
        "retained_energy": retained_energy,
        "retained_energy_fraction": retained_energy / max(total_energy, np.finfo(float).tiny),
        "computational_approximation": rank < maximum_rank,
    }
    return basis, scores, metadata


def exact_pod_projection_by_energy(
    q: np.ndarray,
    retained_energy_target: float,
    minimum_rank: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return the smallest exact POD basis satisfying an energy target.

    ``minimum_rank`` lets an RVMD request require at least ``K`` coordinates
    even when the energy target alone would select fewer.  This target applies
    only to the compressed identification matrix; the final scientific gate
    remains the independently audited full-space reconstruction energy.
    """

    matrix = np.asarray(q)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("q must be a finite two-dimensional matrix")
    target = float(retained_energy_target)
    if not math.isfinite(target) or target <= 0.0 or target > 1.0:
        raise ValueError("retained_energy_target must lie in (0, 1]")
    maximum_rank = min(matrix.shape)
    if not isinstance(minimum_rank, (int, np.integer)) or minimum_rank < 1:
        raise ValueError("minimum_rank must be a positive integer")
    if int(minimum_rank) > maximum_rank:
        raise ValueError("minimum_rank exceeds min(q.shape)")
    left, singular_values, right_t = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64), full_matrices=False
    )
    energy = singular_values * singular_values
    total_energy = float(np.sum(energy, dtype=np.float64))
    if not math.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("q must have positive finite centered energy")
    cumulative = np.cumsum(energy, dtype=np.float64) / total_energy
    target_rank = int(np.searchsorted(cumulative, target, side="left") + 1)
    rank = max(target_rank, int(minimum_rank))
    basis64 = left[:, :rank]
    scores64 = singular_values[:rank, None] * right_t[:rank, :]
    retained_energy = float(np.sum(scores64 * scores64, dtype=np.float64))
    basis = basis64.astype(matrix.dtype, copy=False)
    scores = scores64.astype(matrix.dtype, copy=False)
    return basis, scores, {
        "method": "exact_spatial_pod_energy_target",
        "input_rows": int(matrix.shape[0]),
        "output_rows": int(rank),
        "rank": int(rank),
        "minimum_rank": int(minimum_rank),
        "energy_target_rank": int(target_rank),
        "retained_energy_target": target,
        "input_energy": total_energy,
        "retained_energy": retained_energy,
        "retained_energy_fraction": retained_energy / total_energy,
        "computational_approximation": rank < maximum_rank,
    }


def _component_rms_from_sums(
    sums: np.ndarray, sums_of_squares: np.ndarray, n_time: int
) -> Tuple[np.ndarray, np.ndarray]:
    centered_ss = np.asarray(sums_of_squares, dtype=np.float64) - (
        np.asarray(sums, dtype=np.float64) ** 2 / float(n_time)
    )
    tolerance = np.finfo(np.float64).eps * np.maximum(np.abs(sums_of_squares), 1.0) * 32.0
    if np.any(centered_ss < -tolerance):
        raise FloatingPointError("negative centered sum of squares exceeds roundoff tolerance")
    centered_ss = np.maximum(centered_ss, 0.0)
    component_rms = np.sqrt(np.sum(centered_ss, axis=0) / float(sums.shape[0] * n_time))
    if not np.all(np.isfinite(component_rms)) or np.any(component_rms <= np.finfo(float).tiny):
        raise ValueError("one or more variables have zero or invalid fluctuation RMS")
    return component_rms, centered_ss


def choose_variable_scales(
    component_rms: Sequence[float],
    component_centered_energy: Sequence[float],
    variables: Sequence[str],
    n_points: int,
    n_time: int,
    policy: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Choose the identification metric without silently reweighting velocity."""

    rms = np.asarray(component_rms, dtype=np.float64)
    energy = np.asarray(component_centered_energy, dtype=np.float64)
    if rms.shape != (len(variables),) or energy.shape != rms.shape:
        raise ValueError("component diagnostics have inconsistent dimensions")
    policy = policy.lower()
    if policy == "velocity_global_rms":
        if any(variable not in ("u", "v", "w") for variable in variables):
            raise ValueError(
                "velocity_global_rms is only valid for velocity variables; choose none or "
                "explicit per_variable_rms when pressure is included"
            )
        # One common scalar leaves the relative u/v/w kinetic-energy metric intact.
        velocity_vector_rms = math.sqrt(float(np.sum(energy)) / float(n_points * n_time))
        if not np.isfinite(velocity_vector_rms) or velocity_vector_rms <= np.finfo(float).tiny:
            raise ValueError("global velocity fluctuation RMS is zero or invalid")
        scales = np.full(len(variables), velocity_vector_rms, dtype=np.float64)
        metric = (
            "u/v/w share one velocity-vector fluctuation RMS; Euclidean norm therefore "
            "preserves their relative kinetic-energy weighting up to a common scalar"
        )
        physical_kinetic_weighting = True
    elif policy == "none":
        scales = np.ones(len(variables), dtype=np.float64)
        metric = "no variable scaling; Euclidean norm is evaluated in native units"
        physical_kinetic_weighting = all(variable in ("u", "v", "w") for variable in variables)
    elif policy == "per_variable_rms":
        scales = rms.copy()
        metric = (
            "each variable divided by its own RMS; this deliberately equalizes components "
            "and is not the native velocity kinetic-energy metric"
        )
        physical_kinetic_weighting = False
    else:
        raise ValueError("scaling policy must be velocity_global_rms, none, or per_variable_rms")
    return scales, {
        "scaling_policy": policy,
        "normalization_metric": metric,
        "preserves_relative_velocity_kinetic_energy_weighting": physical_kinetic_weighting,
        "component_fluctuation_rms_diagnostic_physical": rms.tolist(),
        "applied_divisors": scales.tolist(),
    }


def _print_progress(label: str, completed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = completed / max(elapsed, np.finfo(float).tiny)
    remaining = (total - completed) / max(rate, np.finfo(float).tiny)
    print(
        "{}: {}/{} frames, elapsed {:.1f}s, ETA {:.1f}s".format(
            label, completed, total, elapsed, remaining
        ),
        flush=True,
    )


def build_countsketch_identification_matrix(
    series: NatoGDCsvSeries,
    rows_per_variable: int,
    seed: int,
    coordinate_check_interval: int = 1,
    progress_interval: int = 25,
    scaling_policy: str = "velocity_global_rms",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stream all frames once and build a centered, scaled CountSketch."""

    n_points = series.n_points
    n_variables = len(series.variables)
    n_time = len(series.paths)
    buckets, signs, metadata = countsketch_map(n_points, n_variables, rows_per_variable, seed)
    sketch = np.empty((rows_per_variable * n_variables, n_time), dtype=np.float64)
    sums = np.zeros((n_points, n_variables), dtype=np.float64)
    sums_of_squares = np.zeros((n_points, n_variables), dtype=np.float64)
    started = time.monotonic()
    for time_index in range(n_time):
        values = series.read_validated_frame(time_index, coordinate_check_interval)
        values64 = np.asarray(values, dtype=np.float64)
        sums += values64
        sums_of_squares += values64 * values64
        sketch[:, time_index] = apply_countsketch_frame(
            values64, buckets, signs, rows_per_variable
        )
        if progress_interval > 0 and (
            (time_index + 1) % progress_interval == 0 or time_index + 1 == n_time
        ):
            _print_progress("CountSketch identification", time_index + 1, n_time, started)
    component_rms, centered_ss = _component_rms_from_sums(sums, sums_of_squares, n_time)
    scales, scaling_metadata = choose_variable_scales(
        component_rms,
        np.sum(centered_ss, axis=0),
        series.variables,
        n_points,
        n_time,
        scaling_policy,
    )
    sketch -= np.mean(sketch, axis=1, keepdims=True)
    for variable_index, scale in enumerate(scales):
        start = variable_index * rows_per_variable
        sketch[start : start + rows_per_variable] /= scale
    metadata.update(
        {
            "component_fluctuation_rms_physical": component_rms.tolist(),
            "component_centered_energy_physical": np.sum(centered_ss, axis=0).tolist(),
            "centering": "per full spatial DOF in time; commuted through linear sketch",
            "variable_scaling": scaling_metadata,
            "normalization_metric": scaling_metadata["normalization_metric"],
            "identification_matrix_shape": list(sketch.shape),
        }
    )
    return sketch.astype(np.float32), scales, metadata


def build_stratified_identification_matrix(
    series: NatoGDCsvSeries,
    sample_points: int,
    coordinate_check_interval: int = 1,
    progress_interval: int = 25,
    scaling_policy: str = "velocity_global_rms",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build the recorded stratified-sampling fallback identification matrix."""

    indices = stratified_point_indices(series.coordinates, sample_points)
    n_variables = len(series.variables)
    n_time = len(series.paths)
    raw = np.empty((sample_points * n_variables, n_time), dtype=np.float32)
    started = time.monotonic()
    for time_index in range(n_time):
        values = series.read_validated_frame(time_index, coordinate_check_interval)
        raw[:, time_index] = values[indices, :].reshape(-1)
        if progress_interval > 0 and (
            (time_index + 1) % progress_interval == 0 or time_index + 1 == n_time
        ):
            _print_progress("Stratified identification", time_index + 1, n_time, started)
    row_mean = np.mean(raw, axis=1, dtype=np.float64).astype(np.float32)
    raw -= row_mean[:, None]
    component_rms = np.empty(n_variables, dtype=np.float64)
    component_energy = np.empty(n_variables, dtype=np.float64)
    for variable_index in range(n_variables):
        component = raw[variable_index::n_variables]
        component_energy[variable_index] = float(np.sum(component.astype(np.float64) ** 2))
        component_rms[variable_index] = math.sqrt(
            component_energy[variable_index] / float(sample_points * n_time)
        )
    scales, scaling_metadata = choose_variable_scales(
        component_rms,
        component_energy,
        series.variables,
        sample_points,
        n_time,
        scaling_policy,
    )
    for variable_index in range(n_variables):
        component = raw[variable_index::n_variables]
        component /= scales[variable_index]
    metadata = {
        "method": "deterministic_stratified_cartesian_point_sample",
        "input_spatial_rows": int(series.n_points * n_variables),
        "sampled_points": int(sample_points),
        "output_sample_rows": int(sample_points * n_variables),
        "sample_fraction_of_points": float(sample_points / series.n_points),
        "sample_indices_rule": "Cartesian axis coverage plus deterministic lexicographic fill",
        "all_input_degrees_of_freedom_participate": sample_points == series.n_points,
        "computational_approximation": sample_points < series.n_points,
        "component_fluctuation_rms_physical": component_rms.tolist(),
        "centering": "per sampled spatial DOF in time",
        "variable_scaling": scaling_metadata,
        "normalization_metric": scaling_metadata["normalization_metric"],
        "identification_matrix_shape": list(raw.shape),
    }
    return raw, scales, indices, metadata


def write_frequency_table(
    path: Union[str, Path], result: RVMDResult, extra_columns: Optional[Mapping[str, Sequence[float]]] = None
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    extras = dict(extra_columns or {})
    for name, values in extras.items():
        if len(values) != result.frequency_hz.size:
            raise ValueError("extra frequency-table column {} has wrong length".format(name))
    total_reference_energy = max(float(np.sum(result.energy)), np.finfo(float).tiny)
    fieldnames = [
        "mode",
        "frequency_cycles_per_sample",
        "frequency_hz",
        "strouhal_St_f13.716_over_20",
        "temporal_coefficient_energy",
        "temporal_coefficient_energy_fraction",
    ] + list(extras.keys())
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(result.frequency_hz.size):
            row: Dict[str, Any] = {
                "mode": index + 1,
                "frequency_cycles_per_sample": "{:.12g}".format(
                    result.omega_cycles_per_sample[index]
                ),
                "frequency_hz": "{:.12g}".format(result.frequency_hz[index]),
                "strouhal_St_f13.716_over_20": "{:.12g}".format(result.strouhal[index]),
                "temporal_coefficient_energy": "{:.12g}".format(result.energy[index]),
                "temporal_coefficient_energy_fraction": "{:.12g}".format(
                    result.energy[index] / total_reference_energy
                ),
            }
            for name, values in extras.items():
                row[name] = "{:.12g}".format(float(values[index]))
            writer.writerow(row)
    return output


def _require_restart_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            "restart {} mismatch: saved {!r}, requested {!r}".format(
                label, observed, expected
            )
        )


def _load_and_validate_restart_assets(
    restart_directory: Union[str, Path],
    series: NatoGDCsvSeries,
    *,
    dt: float,
    k: int,
    alpha: float,
    method: str,
    countsketch_rows_per_variable: int,
    sample_points: int,
    sketch_seed: int,
    pod_rank: int,
    pod_energy_target: Optional[float],
    pod_seed: int,
    pod_oversampling: int,
    pod_power_iterations: int,
    maximum_steps: int,
    n_dc: int,
    dtype: str,
    scaling_policy: str,
    source_time_origin_seconds: Optional[float],
) -> Tuple[RVMDResult, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Load a restart directory and reject every immutable adapter mismatch."""

    restart_path = Path(restart_directory).expanduser().resolve()
    if not restart_path.is_dir():
        raise FileNotFoundError("restart directory does not exist: {}".format(restart_path))
    required_files = (
        "metadata.json",
        "temporal_model.npz",
        "coordinates.npy",
        "identification_projection_basis.npy",
        "component_scales.npy",
        "rvmd_input.npy",
    )
    missing = [name for name in required_files if not (restart_path / name).is_file()]
    if missing:
        raise ValueError(
            "restart directory is missing required artifacts: {}".format(", ".join(missing))
        )
    metadata = _load_metadata(restart_path / "metadata.json")
    inventory = series.inventory()
    _require_restart_equal(
        "input directory",
        str(Path(metadata.get("input_directory", "")).expanduser().resolve()),
        str(inventory.input_directory),
    )
    for label, observed, expected in (
        ("frame files", metadata.get("frame_files"), [path.name for path in inventory.paths]),
        ("frame ids", metadata.get("frame_ids"), inventory.frame_ids),
        ("number of frames", metadata.get("n_frames"), inventory.n_time),
        ("number of points", metadata.get("n_points"), inventory.n_points),
        ("variables", metadata.get("variables"), list(inventory.variables)),
        ("variable columns", metadata.get("variable_columns"), list(inventory.variable_columns)),
        ("coordinate columns", metadata.get("coordinate_columns"), list(inventory.coordinate_columns)),
        ("k", metadata.get("k"), int(k)),
        ("alpha", metadata.get("alpha_normalized_frequency"), float(alpha)),
        ("dt", metadata.get("dt_seconds"), float(dt)),
    ):
        _require_restart_equal(label, observed, expected)
    saved_origin = metadata.get("time_axis", {}).get("absolute_source_time_origin_seconds")
    requested_origin = (
        None if source_time_origin_seconds is None else float(source_time_origin_seconds)
    )
    _require_restart_equal("absolute source time origin", saved_origin, requested_origin)

    rvmd_info = metadata.get("rvmd_info")
    if not isinstance(rvmd_info, dict):
        raise ValueError("restart metadata has no valid rvmd_info object")
    _require_restart_equal("dtype", rvmd_info.get("dtype"), np.dtype(dtype).name)
    _require_restart_equal("n_dc", rvmd_info.get("n_dc"), int(n_dc))

    compression = metadata.get("identification_compression")
    if not isinstance(compression, dict):
        raise ValueError("restart metadata has no identification_compression object")
    expected_method_label = {
        "countsketch": "deterministic_countsketch",
        "stratified": "deterministic_stratified_cartesian_point_sample",
    }[method]
    _require_restart_equal("compression method", compression.get("method"), expected_method_label)
    if method == "countsketch":
        _require_restart_equal(
            "CountSketch rows per variable",
            compression.get("rows_per_variable"),
            int(countsketch_rows_per_variable),
        )
        _require_restart_equal("CountSketch seed", compression.get("seed"), int(sketch_seed))
    else:
        _require_restart_equal(
            "stratified sample point count", compression.get("sampled_points"), int(sample_points)
        )
    variable_scaling = compression.get("variable_scaling")
    if not isinstance(variable_scaling, dict):
        raise ValueError("restart compression metadata has no variable_scaling object")
    _require_restart_equal(
        "scaling policy", variable_scaling.get("scaling_policy"), scaling_policy.lower()
    )

    pod = metadata.get("pod_projection")
    if not isinstance(pod, dict):
        raise ValueError("restart metadata has no pod_projection object")
    if pod_energy_target is not None:
        for label, key, expected in (
            ("POD method", "method", "exact_spatial_pod_energy_target"),
            ("POD retained-energy target", "retained_energy_target", float(pod_energy_target)),
            ("POD minimum rank", "minimum_rank", int(k)),
        ):
            _require_restart_equal(label, pod.get(key), expected)
        if float(pod.get("retained_energy_fraction", -1.0)) + 1.0e-12 < float(
            pod_energy_target
        ):
            raise ValueError("restart POD projection does not satisfy its energy target")
    elif pod_rank > 0:
        identification_shape = compression.get("identification_matrix_shape")
        if (
            not isinstance(identification_shape, list)
            or len(identification_shape) != 2
        ):
            raise ValueError("restart compression has no valid identification matrix shape")
        expected_effective_rank = min(
            int(pod_rank), int(identification_shape[0]), int(identification_shape[1])
        )
        for label, key, expected in (
            ("POD method", "method", "randomized_spatial_pod_projection"),
            ("POD rank", "rank", expected_effective_rank),
            ("POD seed", "seed", int(pod_seed)),
            ("POD oversampling", "oversampling", int(pod_oversampling)),
            ("POD power iterations", "power_iterations", int(pod_power_iterations)),
        ):
            _require_restart_equal(label, pod.get(key), expected)
    else:
        _require_restart_equal("POD method", pod.get("method"), "none")

    restart_artifacts = metadata.get("restart_artifacts")
    if not isinstance(restart_artifacts, dict) or restart_artifacts.get("schema_version") != 1:
        raise ValueError(
            "restart metadata lacks the version-1 projected-input audit; rerun identification "
            "with the restart-capable adapter"
        )
    basis = np.load(
        str(restart_path / "identification_projection_basis.npy"), allow_pickle=False
    )
    scales = np.load(str(restart_path / "component_scales.npy"), allow_pickle=False)
    rvmd_input = np.load(str(restart_path / "rvmd_input.npy"), allow_pickle=False)
    saved_coordinates = np.load(str(restart_path / "coordinates.npy"), allow_pickle=False)
    if not np.array_equal(saved_coordinates, series.coordinates.astype(np.float32, copy=False)):
        raise ValueError("restart coordinates or their row order differ from the current series")
    for label, array, key in (
        ("projection basis", basis, "projection_basis_sha256"),
        ("component scales", scales, "component_scales_sha256"),
        ("RVMD projected input", rvmd_input, "rvmd_input_sha256"),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError("restart {} contains NaN or infinity".format(label))
        if restart_artifacts.get(key) != _array_sha256(array):
            raise ValueError("restart {} checksum mismatch".format(label))
    result = RVMDResult.load(restart_path / "temporal_model.npz")
    if result.restart_state is None:
        raise ValueError("restart temporal model is readable but has no continuation state")
    completed_steps = int(result.info.get("completed_steps", -1))
    if maximum_steps < completed_steps:
        raise ValueError(
            "maximum_steps is a total limit and cannot be less than the {} saved sweeps".format(
                completed_steps
            )
        )
    return result, metadata, basis, scales, rvmd_input


def identify_temporal_coefficients(
    series: NatoGDCsvSeries,
    output_directory: Union[str, Path],
    *,
    dt: float,
    k: int,
    alpha: float,
    method: str = "countsketch",
    countsketch_rows_per_variable: int = 256,
    sample_points: int = 512,
    sketch_seed: int = 20260911,
    pod_rank: int = 128,
    pod_energy_target: Optional[float] = None,
    pod_seed: int = 20260912,
    pod_oversampling: int = 12,
    pod_power_iterations: int = 1,
    tolerance: float = 5e-3,
    maximum_steps: int = 500,
    init_frequency_max_hz: Optional[float] = None,
    initial_frequencies_hz: Optional[Sequence[float]] = None,
    n_dc: int = 0,
    coordinate_check_interval: int = 1,
    progress_interval: int = 25,
    dtype: str = "float32",
    scaling_policy: str = "velocity_global_rms",
    source_time_origin_seconds: Optional[float] = None,
    restart_directory: Optional[Union[str, Path]] = None,
    verbose_rvmd: bool = False,
    compute_backend: str = "auto",
) -> Tuple[RVMDResult, Dict[str, Any]]:
    """Identify RVMD time coefficients and write a self-describing stage result."""

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt is required and must be positive")
    if source_time_origin_seconds is not None and not np.isfinite(source_time_origin_seconds):
        raise ValueError("source_time_origin_seconds must be finite when supplied")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    method = method.lower()
    if method not in ("countsketch", "stratified"):
        raise ValueError("method must be countsketch or stratified")
    restart_result: Optional[RVMDResult] = None
    restart_metadata: Optional[Dict[str, Any]] = None
    restart_basis: Optional[np.ndarray] = None
    restart_scales: Optional[np.ndarray] = None
    restart_rvmd_input: Optional[np.ndarray] = None
    restart_path: Optional[Path] = None
    if restart_directory is not None:
        restart_path = Path(restart_directory).expanduser().resolve()
        (
            restart_result,
            restart_metadata,
            restart_basis,
            restart_scales,
            restart_rvmd_input,
        ) = _load_and_validate_restart_assets(
            restart_path,
            series,
            dt=dt,
            k=k,
            alpha=alpha,
            method=method,
            countsketch_rows_per_variable=countsketch_rows_per_variable,
            sample_points=sample_points,
            sketch_seed=sketch_seed,
            pod_rank=pod_rank,
            pod_energy_target=pod_energy_target,
            pod_seed=pod_seed,
            pod_oversampling=pod_oversampling,
            pod_power_iterations=pod_power_iterations,
            maximum_steps=maximum_steps,
            n_dc=n_dc,
            dtype=dtype,
            scaling_policy=scaling_policy,
            source_time_origin_seconds=source_time_origin_seconds,
        )
    if method == "countsketch":
        identification, scales, method_metadata = build_countsketch_identification_matrix(
            series,
            countsketch_rows_per_variable,
            sketch_seed,
            coordinate_check_interval,
            progress_interval,
            scaling_policy,
        )
        sample_indices = None
    elif method == "stratified":
        identification, scales, sample_indices, method_metadata = build_stratified_identification_matrix(
            series,
            sample_points,
            coordinate_check_interval,
            progress_interval,
            scaling_policy,
        )
        np.save(str(output / "sample_indices.npy"), sample_indices)
        np.save(str(output / "sample_coordinates.npy"), series.coordinates[sample_indices])
    else:
        raise ValueError("method must be countsketch or stratified")

    identification_sha256 = _array_sha256(identification)
    if pod_energy_target is None and pod_rank > 0 and pod_rank < k:
        raise ValueError("pod_rank must be at least k, or zero to disable POD")
    if restart_result is not None:
        assert restart_metadata is not None
        assert restart_basis is not None
        assert restart_scales is not None
        assert restart_rvmd_input is not None
        assert restart_path is not None
        restart_artifacts = restart_metadata["restart_artifacts"]
        if restart_artifacts.get("identification_matrix_sha256") != identification_sha256:
            raise ValueError(
                "current compressed identification matrix differs from the restart run"
            )
        if not np.array_equal(scales, restart_scales):
            raise ValueError("current component scales differ from the restart run")
        if method == "stratified":
            saved_sample_path = restart_path / "sample_indices.npy"
            if not saved_sample_path.is_file():
                raise ValueError("stratified restart is missing sample_indices.npy")
            saved_sample_indices = np.load(str(saved_sample_path), allow_pickle=False)
            if sample_indices is None or not np.array_equal(sample_indices, saved_sample_indices):
                raise ValueError("stratified sample indices differ from the restart run")
        basis = np.asarray(restart_basis)
        scales = np.asarray(restart_scales)
        rvmd_input = np.asarray(restart_rvmd_input)
        effective_pod_rank = int(basis.shape[1])
        expected_basis_shape = (identification.shape[0], effective_pod_rank)
        if basis.shape != expected_basis_shape:
            raise ValueError(
                "restart projection basis shape mismatch: expected {}, got {}".format(
                    expected_basis_shape, basis.shape
                )
            )
        expected_input_shape = (basis.shape[1], identification.shape[1])
        if rvmd_input.shape != expected_input_shape:
            raise ValueError(
                "restart RVMD input shape mismatch: expected {}, got {}".format(
                    expected_input_shape, rvmd_input.shape
                )
            )
        gram = np.asarray(basis, dtype=np.float64).T @ np.asarray(basis, dtype=np.float64)
        if not np.allclose(gram, np.eye(gram.shape[0]), rtol=2e-5, atol=2e-5):
            raise ValueError("restart identification projection basis is not orthonormal")
        projected_now = (
            identification
            if pod_rank == 0
            else (basis.T @ identification).astype(rvmd_input.dtype, copy=False)
        )
        if not np.array_equal(projected_now, rvmd_input):
            raise ValueError(
                "saved RVMD projected input is not the exact projection of current data "
                "through the saved basis"
            )
        pod_metadata = dict(restart_metadata["pod_projection"])
    elif pod_energy_target is not None:
        basis, _, pod_metadata = exact_pod_projection_by_energy(
            identification,
            float(pod_energy_target),
            minimum_rank=int(k),
        )
        rvmd_input = (basis.T @ identification).astype(identification.dtype, copy=False)
    elif pod_rank > 0:
        basis, _, pod_metadata = randomized_pod_projection(
            identification,
            min(pod_rank, min(identification.shape)),
            seed=pod_seed,
            oversampling=pod_oversampling,
            power_iterations=pod_power_iterations,
        )
        # Persist the exact matrix sent to RVMD.  Reusing this value on restart
        # avoids changing the trajectory through an algebraically equivalent
        # but roundoff-different projection formula.
        rvmd_input = (basis.T @ identification).astype(identification.dtype, copy=False)
    else:
        basis = np.eye(identification.shape[0], dtype=identification.dtype)
        rvmd_input = identification
        pod_metadata = {
            "method": "none",
            "input_rows": int(identification.shape[0]),
            "output_rows": int(identification.shape[0]),
            "retained_energy_fraction": 1.0,
            "computational_approximation": False,
        }

    requested_compute_backend = str(compute_backend).strip().lower()
    effective_compute_backend, backend_selection_reason = resolve_compute_backend(
        requested_compute_backend
    )
    result = rvmd(
        rvmd_input,
        k,
        alpha,
        dt=dt,
        tolerance=tolerance,
        maximum_steps=maximum_steps,
        init_frequency_max_hz=init_frequency_max_hz,
        initial_frequencies_hz=initial_frequencies_hz,
        n_dc=n_dc,
        random_seed=pod_seed,
        dtype=dtype,
        restart=restart_result,
        verbose=bool(verbose_rvmd),
        compute_backend=effective_compute_backend,
    )
    result.info["compute_backend_requested"] = requested_compute_backend
    result.info["compute_backend_selection_reason"] = backend_selection_reason
    result.info["compute_backend_policy"] = "prefer_cuda_then_numpy_fallback"
    identification_phi = basis @ result.phi
    identification_audit = energy_audit(
        identification, identification_phi, result.c, weights=None
    )

    np.save(str(output / "coordinates.npy"), series.coordinates.astype(np.float32, copy=False))
    np.save(str(output / "identification_spatial_modes.npy"), identification_phi.astype(np.float32))
    np.save(str(output / "identification_projection_basis.npy"), basis.astype(np.float32))
    np.save(str(output / "component_scales.npy"), scales.astype(np.float64))
    np.save(str(output / "identification_matrix.npy"), identification, allow_pickle=False)
    np.save(str(output / "rvmd_input.npy"), rvmd_input, allow_pickle=False)
    result.save(output / "temporal_model.npz", compressed=False)
    write_frequency_table(output / "frequencies.csv", result)

    inventory = series.inventory()
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "stage": "temporal_identification_complete",
        "method_label": "spatially-compressed RVMD plus full-space regression",
        "computational_approximation": True,
        "approximation_statement": (
            "Temporal coefficients are identified after deterministic spatial compression; "
            "this is not numerically identical to applying RVMD to the full state matrix."
        ),
        "rvmd_reference": {
            "citation": "Liao et al., Journal of Fluid Mechanics 966, A7 (2023)",
            "doi": "10.1017/jfm.2023.435",
        },
        "input_directory": str(inventory.input_directory),
        "frame_files": [path.name for path in inventory.paths],
        "frame_ids": inventory.frame_ids,
        "n_frames": inventory.n_time,
        "n_points": inventory.n_points,
        "variables": list(inventory.variables),
        "variable_columns": list(inventory.variable_columns),
        "coordinate_columns": list(inventory.coordinate_columns),
        "coordinate_row_order_check_interval_frames": int(coordinate_check_interval),
        "grid_audit": infer_cartesian_grid(inventory.coordinates),
        "dt_seconds": float(dt),
        "sample_rate_hz": float(1.0 / dt),
        "time_axis": {
            "relative_time_from_first_frame_seconds": {
                "first": 0.0,
                "last": float((inventory.n_time - 1) * dt),
            },
            "absolute_source_time_origin_seconds": (
                None if source_time_origin_seconds is None else float(source_time_origin_seconds)
            ),
            "absolute_source_time_status": (
                "unknown_not_inferred_from_frame_id_or_filename"
                if source_time_origin_seconds is None
                else "explicit_user_supplied"
            ),
            "frame_id_times_dt_used_as_absolute_time": False,
        },
        "k": int(k),
        "alpha_normalized_frequency": float(alpha),
        "mean_field_handling": "removed before RVMD and stored separately during full regression",
        "identification_compression": method_metadata,
        "pod_projection": pod_metadata,
        "rvmd_info": result.info,
        "rvmd_reduced_input_audit": result.audit,
        "rvmd_identification_space_audit": identification_audit,
        "component_scales_physical": scales.tolist(),
        "restart_artifacts": {
            "schema_version": 1,
            "identification_matrix_sha256": identification_sha256,
            "projection_basis_sha256": _array_sha256(basis.astype(np.float32, copy=False)),
            "component_scales_sha256": _array_sha256(scales.astype(np.float64, copy=False)),
            "rvmd_input_sha256": _array_sha256(rvmd_input),
            "rvmd_input_definition": (
                "the exact persisted reduced matrix supplied to rvmd(); reused byte-for-byte "
                "when continuation is requested"
            ),
        },
        "restart_provenance": (
            None
            if restart_result is None
            else {
                "source_directory": str(restart_path),
                "source_completed_steps": int(restart_result.info["completed_steps"]),
                "requested_total_maximum_steps": int(maximum_steps),
                "projection_basis_reused": True,
                "component_scales_reused": True,
                "rvmd_projected_input_reused": True,
            }
        ),
        "output_arrays": {
            "temporal_model.npz/c": list(result.c.shape),
            "temporal_model.npz/phi": list(result.phi.shape),
            "identification_spatial_modes.npy": list(identification_phi.shape),
            "coordinates.npy": list(series.coordinates.shape),
            "identification_matrix.npy": list(identification.shape),
            "rvmd_input.npy": list(rvmd_input.shape),
        },
    }
    write_json(output / "metadata.json", metadata)
    write_json(
        output / "audit.json",
        {
            "pod_projection": pod_metadata,
            "rvmd_reduced_input": result.audit,
            "rvmd_identification_space": identification_audit,
        },
    )
    return result, metadata


def solve_spatial_modes_from_statistics(
    centered_cross_qc: np.ndarray,
    coefficient_gram: np.ndarray,
    ridge_relative: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve ``Phi = (Q C) (C.T C)^-1`` robustly for one spatial block."""

    cross = np.asarray(centered_cross_qc, dtype=np.float64)
    gram = np.asarray(coefficient_gram, dtype=np.float64)
    if cross.ndim != 2 or gram.shape != (cross.shape[1], cross.shape[1]):
        raise ValueError("cross and coefficient_gram have inconsistent dimensions")
    if not np.isfinite(ridge_relative) or ridge_relative < 0:
        raise ValueError("ridge_relative must be finite and non-negative")
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    ridge_absolute = ridge_relative * max(scale, np.finfo(float).tiny)
    regularized = gram + ridge_absolute * np.eye(gram.shape[0])
    inverse = np.linalg.pinv(regularized, rcond=1e-12)
    modes = cross @ inverse
    singular_values = np.linalg.svd(gram, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).tiny]
    condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
    return modes, {
        "ridge_relative": float(ridge_relative),
        "ridge_absolute": float(ridge_absolute),
        "coefficient_gram_condition_number_nonzero_spectrum": condition,
        "coefficient_gram_rank": int(np.linalg.matrix_rank(gram)),
    }


def regress_full_space_modes(
    series: NatoGDCsvSeries,
    result: RVMDResult,
    component_scales: Sequence[float],
    output_directory: Union[str, Path],
    *,
    time_batch_size: int = 8,
    spatial_block_rows: int = 65536,
    ridge_relative: float = 1e-10,
    coordinate_check_interval: int = 1,
    progress_interval: int = 25,
    metric_description: str = "Euclidean norm after applying the recorded variable divisors",
) -> Dict[str, Any]:
    """Recover full physical modes with two-pass-compatible sufficient statistics."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    coefficients = np.asarray(result.c, dtype=np.float64)
    if coefficients.shape[0] != len(series.paths):
        raise ValueError("temporal coefficient length does not match the CSV frame count")
    scales = np.asarray(component_scales, dtype=np.float64)
    n_points = series.n_points
    n_variables = len(series.variables)
    n_time, k = coefficients.shape
    if scales.shape != (n_variables,) or np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("component_scales must contain one finite positive value per variable")
    if time_batch_size <= 0 or spatial_block_rows <= 0:
        raise ValueError("batch and block sizes must be positive")

    n_state = n_points * n_variables
    sums = np.zeros(n_state, dtype=np.float64)
    sums_of_squares = np.zeros(n_state, dtype=np.float64)
    cross_qc = np.zeros((n_state, k), dtype=np.float64)
    started = time.monotonic()
    completed = 0
    for batch_start in range(0, n_time, time_batch_size):
        batch_stop = min(batch_start + time_batch_size, n_time)
        batch_length = batch_stop - batch_start
        batch = np.empty((n_state, batch_length), dtype=np.float32)
        for local, time_index in enumerate(range(batch_start, batch_stop)):
            values = series.read_validated_frame(time_index, coordinate_check_interval)
            normalized = np.asarray(values, dtype=np.float64) / scales[None, :]
            batch[:, local] = normalized.reshape(-1).astype(np.float32)
        c_batch = coefficients[batch_start:batch_stop]
        for row_start in range(0, n_state, spatial_block_rows):
            row_stop = min(row_start + spatial_block_rows, n_state)
            block = np.asarray(batch[row_start:row_stop], dtype=np.float64)
            sums[row_start:row_stop] += np.sum(block, axis=1)
            sums_of_squares[row_start:row_stop] += np.sum(block * block, axis=1)
            cross_qc[row_start:row_stop] += block @ c_batch
        completed = batch_stop
        if progress_interval > 0 and (
            completed % progress_interval < batch_length or completed == n_time
        ):
            _print_progress("Full-space regression statistics", completed, n_time, started)

    mean_normalized = sums / float(n_time)
    centered_cross = cross_qc - mean_normalized[:, None] * np.sum(coefficients, axis=0)[None, :]
    coefficient_mean = np.mean(coefficients, axis=0)
    centered_coefficients = coefficients - coefficient_mean[None, :]
    gram = centered_coefficients.T @ centered_coefficients
    mode_file = output / "spatial_modes_physical.npy"
    mean_file = output / "mean_field_physical.npy"
    physical_modes = np.lib.format.open_memmap(
        str(mode_file), mode="w+", dtype=np.float32, shape=(n_points, n_variables, k)
    )
    physical_mean = np.lib.format.open_memmap(
        str(mean_file), mode="w+", dtype=np.float32, shape=(n_points, n_variables)
    )
    centered_data_energy_by_row = sums_of_squares - sums * sums / float(n_time)
    roundoff_tolerance = (
        np.finfo(np.float64).eps * np.maximum(np.abs(sums_of_squares), 1.0) * 64.0
    )
    if np.any(centered_data_energy_by_row < -roundoff_tolerance):
        raise FloatingPointError("negative full-space centered energy exceeds roundoff tolerance")
    centered_data_energy_by_row = np.maximum(centered_data_energy_by_row, 0.0)
    data_energy = float(np.sum(centered_data_energy_by_row))
    reconstruction_energy = 0.0
    data_reconstruction_inner = 0.0
    individual_mode_self_energy = np.zeros(k, dtype=np.float64)
    regression_info: Optional[Dict[str, Any]] = None
    scale_by_state = np.tile(scales, n_points)
    for row_start in range(0, n_state, spatial_block_rows):
        row_stop = min(row_start + spatial_block_rows, n_state)
        modes_normalized, block_info = solve_spatial_modes_from_statistics(
            centered_cross[row_start:row_stop], gram, ridge_relative
        )
        if regression_info is None:
            regression_info = block_info
        reconstruction_energy += float(
            np.einsum("ik,kl,il->", modes_normalized, gram, modes_normalized, optimize=True)
        )
        data_reconstruction_inner += float(
            np.sum(modes_normalized * centered_cross[row_start:row_stop])
        )
        individual_mode_self_energy += np.sum(modes_normalized * modes_normalized, axis=0) * np.diag(gram)
        physical_flat = modes_normalized * scale_by_state[row_start:row_stop, None]
        physical_modes.reshape(n_state, k)[row_start:row_stop] = physical_flat.astype(np.float32)
        physical_mean.reshape(n_state)[row_start:row_stop] = (
            mean_normalized[row_start:row_stop] * scale_by_state[row_start:row_stop]
        ).astype(np.float32)
    physical_modes.flush()
    physical_mean.flush()
    del physical_modes, physical_mean
    np.save(str(output / "coefficient_mean_for_full_regression.npy"), coefficient_mean)

    residual_energy = data_energy + reconstruction_energy - 2.0 * data_reconstruction_inner
    residual_energy = max(residual_energy, 0.0)
    audit = {
        "metric": metric_description,
        "full_normalized_centered_data_energy": data_energy,
        "full_normalized_reconstruction_energy": reconstruction_energy,
        "full_normalized_data_reconstruction_inner_product": data_reconstruction_inner,
        "full_normalized_residual_energy": residual_energy,
        "full_normalized_residual_energy_fraction": residual_energy
        / max(data_energy, np.finfo(float).tiny),
        "full_normalized_captured_energy_fraction_one_minus_residual": 1.0
        - residual_energy / max(data_energy, np.finfo(float).tiny),
        "full_normalized_relative_frobenius_error": math.sqrt(
            residual_energy / max(data_energy, np.finfo(float).tiny)
        ),
        "full_normalized_individual_modal_self_energy": individual_mode_self_energy.tolist(),
        "full_normalized_modal_cross_energy": reconstruction_energy
        - float(np.sum(individual_mode_self_energy)),
        "least_squares": regression_info or {},
        "n_full_state_rows": int(n_state),
        "n_time": int(n_time),
        "k": int(k),
        "time_batch_size": int(time_batch_size),
        "spatial_block_rows": int(spatial_block_rows),
        "reconstruction_physical_units": (
            "field[n,v,t] = mean_field_physical[n,v] + "
            "sum_k spatial_modes_physical[n,v,k] * "
            "(c[t,k] - coefficient_mean_for_full_regression[k])"
        ),
        "output_arrays": {
            "mean_field_physical.npy": [int(n_points), int(n_variables)],
            "spatial_modes_physical.npy": [int(n_points), int(n_variables), int(k)],
            "coefficient_mean_for_full_regression.npy": [int(k)],
        },
    }
    write_json(output / "full_space_regression_audit.json", audit)
    write_frequency_table(
        output / "frequencies.csv",
        result,
        {"full_normalized_modal_self_energy": individual_mode_self_energy},
    )
    return audit


def _load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("metadata JSON root must be an object")
    return value


def _parse_frequency_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None or not text.strip():
        return None
    return [float(item.strip()) for item in text.split(",")]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spatially compressed RVMD and full-space regression for NATO-GD 3-D CSV data"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="directory containing time-ordered CSV frames")
    source.add_argument(
        "--cache-dir",
        help="validated nato-gd-3d-cache-v1 directory (faster than reparsing CSV)",
    )
    parser.add_argument("--output-dir", required=True, help="directory for the staged RVMD result")
    parser.add_argument(
        "--restart-directory",
        help=(
            "prior identification output containing a restart-capable temporal_model.npz; "
            "MaximumSteps remains the total sweep limit and output-dir may be different"
        ),
    )
    parser.add_argument("--dt", type=float, required=True, help="sampling interval in seconds (mandatory)")
    parser.add_argument(
        "--source-time-origin-seconds",
        type=float,
        help="optional absolute CFD time of the first frame; never inferred from its filename",
    )
    parser.add_argument("--alpha", type=float, required=True, help="RVMD penalty in cycles/sample convention")
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help=(
            "candidate RVMD mode count; no implicit default is allowed because formal K "
            "must be selected by the converged full-space mean-removed residual-energy gate"
        ),
    )
    parser.add_argument("--phase", choices=("all", "identify", "regress"), default="all")
    parser.add_argument("--variables", nargs="+", choices=("u", "v", "w", "p"), default=["u", "v", "w"])
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--allow-frame-gaps", action="store_true")
    parser.add_argument("--reader-engine", choices=("auto", "pandas", "numpy"), default="auto")
    parser.add_argument("--method", choices=("countsketch", "stratified"), default="countsketch")
    parser.add_argument("--countsketch-rows-per-variable", type=int, default=256)
    parser.add_argument("--sample-points", type=int, default=512)
    parser.add_argument("--sketch-seed", type=int, default=20260911)
    parser.add_argument("--pod-rank", type=int, default=128, help="zero disables the second projection")
    parser.add_argument(
        "--pod-energy-target",
        type=float,
        help=(
            "use exact POD and automatically retain the smallest compressed-space rank "
            "meeting this energy fraction; the selected rank is also forced to be at least K"
        ),
    )
    parser.add_argument("--pod-seed", type=int, default=20260912)
    parser.add_argument("--pod-oversampling", type=int, default=12)
    parser.add_argument("--pod-power-iterations", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--maximum-steps", type=int, default=500)
    parser.add_argument("--init-frequency-max-hz", type=float)
    parser.add_argument("--initial-frequencies-hz", help="comma-separated list containing exactly K values")
    parser.add_argument("--n-dc", type=int, default=0)
    parser.add_argument("--coordinate-check-interval", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--verbose-rvmd",
        action="store_true",
        help="print convergence diagnostics every 20 RVMD sweeps",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "numpy", "torch-cuda"),
        default="auto",
        help=(
            "RVMD sweep backend; auto prefers CUDA-enabled PyTorch and otherwise falls "
            "back to NumPy, while explicit torch-cuda remains strict"
        ),
    )
    parser.add_argument(
        "--scaling-policy",
        choices=("velocity_global_rms", "none", "per_variable_rms"),
        default="velocity_global_rms",
        help="default preserves relative u/v/w kinetic-energy weighting",
    )
    parser.add_argument("--time-batch-size", type=int, default=8)
    parser.add_argument("--spatial-block-rows", type=int, default=65536)
    parser.add_argument("--ridge-relative", type=float, default=1e-10)
    parser.add_argument(
        "--minimum-full-space-captured-energy",
        type=float,
        help=(
            "fail after full regression unless the mean-removed full 3-D reconstruction "
            "reaches this energy fraction"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not np.isfinite(args.dt) or args.dt <= 0:
        raise SystemExit("--dt must be a finite positive number")
    if args.restart_directory is not None and args.phase == "regress":
        raise SystemExit("--restart-directory is only valid for phase identify or all")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        if args.allow_frame_gaps:
            raise SystemExit("--allow-frame-gaps is not applicable to a validated cache")
        series = NatoGDCacheSeries(
            args.cache_dir,
            variables=args.variables,
            frame_limit=args.frame_limit,
            expected_dt=args.dt,
        )
    else:
        series = NatoGDCsvSeries(
            args.input_dir,
            variables=args.variables,
            pattern=args.pattern,
            frame_limit=args.frame_limit,
            strict_contiguous_ids=not args.allow_frame_gaps,
            reader_engine=args.reader_engine,
        )
    result: RVMDResult
    metadata: Dict[str, Any]
    if args.phase in ("all", "identify"):
        result, metadata = identify_temporal_coefficients(
            series,
            output,
            dt=args.dt,
            k=args.k,
            alpha=args.alpha,
            method=args.method,
            countsketch_rows_per_variable=args.countsketch_rows_per_variable,
            sample_points=args.sample_points,
            sketch_seed=args.sketch_seed,
            pod_rank=args.pod_rank,
            pod_energy_target=args.pod_energy_target,
            pod_seed=args.pod_seed,
            pod_oversampling=args.pod_oversampling,
            pod_power_iterations=args.pod_power_iterations,
            tolerance=args.tolerance,
            maximum_steps=args.maximum_steps,
            init_frequency_max_hz=args.init_frequency_max_hz,
            initial_frequencies_hz=_parse_frequency_list(args.initial_frequencies_hz),
            n_dc=args.n_dc,
            coordinate_check_interval=args.coordinate_check_interval,
            progress_interval=args.progress_interval,
            dtype=args.dtype,
            scaling_policy=args.scaling_policy,
            source_time_origin_seconds=args.source_time_origin_seconds,
            restart_directory=args.restart_directory,
            verbose_rvmd=args.verbose_rvmd,
            compute_backend=args.compute_backend,
        )
    else:
        result = RVMDResult.load(output / "temporal_model.npz")
        metadata = _load_metadata(output / "metadata.json")
        if abs(float(metadata["dt_seconds"]) - args.dt) > 1e-12:
            raise ValueError("--dt disagrees with the saved temporal model")
        if int(metadata["k"]) != args.k:
            raise ValueError("--k disagrees with the saved temporal model")

    if args.phase in ("all", "regress"):
        scales = np.load(str(output / "component_scales.npy"), allow_pickle=False)
        full_audit = regress_full_space_modes(
            series,
            result,
            scales,
            output,
            time_batch_size=args.time_batch_size,
            spatial_block_rows=args.spatial_block_rows,
            ridge_relative=args.ridge_relative,
            coordinate_check_interval=args.coordinate_check_interval,
            progress_interval=args.progress_interval,
            metric_description=metadata["identification_compression"].get(
                "normalization_metric", "recorded normalized Euclidean metric"
            ),
        )
        metadata["stage"] = "full_space_regression_complete"
        metadata["full_space_regression_audit"] = full_audit
        metadata["output_arrays"].update(full_audit["output_arrays"])
        target = args.minimum_full_space_captured_energy
        captured = float(
            full_audit["full_normalized_captured_energy_fraction_one_minus_residual"]
        )
        if target is not None:
            target = float(target)
            if not math.isfinite(target) or target <= 0.0 or target > 1.0:
                raise ValueError("--minimum-full-space-captured-energy must lie in (0, 1]")
            metadata["full_space_energy_gate"] = {
                "definition": "mean-removed full 3-D u/v/w reconstruction energy",
                "target_fraction": target,
                "captured_fraction": captured,
                "passed": bool(captured + 1.0e-12 >= target),
                "modal_self_energy_sum_prohibited_for_gate": True,
            }
        write_json(output / "metadata.json", metadata)
        if target is not None and captured + 1.0e-12 < target:
            raise ValueError(
                "full-space captured energy {:.9f} is below required {:.9f}".format(
                    captured, target
                )
            )
    print("RVMD stage complete: {}".format(output.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DatasetInventory",
    "NatoGDCsvSeries",
    "NatoGDCacheSeries",
    "apply_countsketch_frame",
    "build_countsketch_identification_matrix",
    "build_stratified_identification_matrix",
    "countsketch_map",
    "choose_variable_scales",
    "identify_temporal_coefficients",
    "infer_cartesian_grid",
    "exact_pod_projection_by_energy",
    "randomized_pod_projection",
    "regress_full_space_modes",
    "solve_spatial_modes_from_statistics",
    "stratified_point_indices",
    "write_frequency_table",
]
