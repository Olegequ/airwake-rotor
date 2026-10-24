#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Three-dimensional exact-DMD identification and full-space regression.

The script deliberately uses the same NATO-GD snapshots, variable scaling,
CountSketch matrix, and 99.98%-energy POD coordinates used by the audited
RVMD run.  Complex-conjugate eigenvectors are grouped into real physical DMD
modes.  Each pair is represented by two real temporal quadratures; the lone
real eigenvector has one quadrature.  The final number of physical modes is
the first energy-ordered prefix whose *full 3-D mean-removed residual* captures
at least 99.8% of the normalized u/v/w fluctuation energy.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def log(message: str) -> None:
    print("[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def exact_dmd(x: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Exact DMD in the supplied reduced coordinates."""
    centered = np.asarray(x, dtype=np.float64)
    centered = centered - np.mean(centered, axis=1, keepdims=True)
    x1, x2 = centered[:, :-1], centered[:, 1:]
    u, singular_values, vh = np.linalg.svd(x1, full_matrices=False)
    valid = singular_values / singular_values[0] > 1e-10
    u, singular_values, vh = u[:, valid], singular_values[valid], vh[valid]
    v = vh.T
    shifted = x2 @ (v / singular_values[None, :])
    a_tilde = u.T @ shifted
    eigenvalues, eigenvectors = np.linalg.eig(a_tilde)
    phi = shifted @ eigenvectors
    coefficients = (np.linalg.pinv(phi, rcond=1e-10) @ centered).T
    omega = np.log(eigenvalues.astype(np.complex128)) / float(dt)
    return {
        "centered": centered,
        "phi": phi,
        "coefficients": coefficients,
        "eigenvalues": eigenvalues,
        "omega": omega,
        "singular_values": singular_values,
    }


def conjugate_groups(eigenvalues: np.ndarray) -> List[List[int]]:
    unvisited = set(range(eigenvalues.size))
    groups: List[List[int]] = []
    while unvisited:
        index = min(unvisited)
        unvisited.remove(index)
        members = [index]
        if abs(eigenvalues[index].imag) > 1e-9 and unvisited:
            candidates = np.asarray(sorted(unvisited), dtype=int)
            distance = np.abs(eigenvalues[candidates] - np.conj(eigenvalues[index]))
            partner = int(candidates[int(np.argmin(distance))])
            relative = float(np.min(distance) / max(abs(eigenvalues[index]), 1e-12))
            if relative < 1e-5:
                members.append(partner)
                unvisited.remove(partner)
        groups.append(members)
    return groups


def orient_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def physical_group_basis(phi: np.ndarray, coefficients: np.ndarray,
                         members: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return a stable real time basis and singular strengths for one group.

    The real contribution is factored through at most four small real columns,
    then compressed to rank two for a conjugate pair or rank one for a real
    eigenmode.  This is a within-group change of basis, not a change to the
    DMD subspace.
    """
    spatial_columns: List[np.ndarray] = []
    temporal_columns: List[np.ndarray] = []
    for member in members:
        spatial_columns.extend([np.real(phi[:, member]), -np.imag(phi[:, member])])
        temporal_columns.extend([
            np.real(coefficients[:, member]), np.imag(coefficients[:, member])])
    a = np.column_stack(spatial_columns)
    b = np.column_stack(temporal_columns)
    qa, ra = np.linalg.qr(a, mode="reduced")
    qb, rb = np.linalg.qr(b, mode="reduced")
    _, singular_values, vh = np.linalg.svd(ra @ rb.T, full_matrices=False)
    requested_rank = 2 if len(members) == 2 else 1
    rank = min(requested_rank, int(np.count_nonzero(
        singular_values > singular_values[0] * 1e-10)))
    if rank < 1:
        raise FloatingPointError("zero-rank physical DMD group")
    basis = qb @ vh.T[:, :rank]
    basis -= np.mean(basis, axis=0, keepdims=True)
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return orient_columns(basis[:, :rank]), singular_values[:rank]


def psd_centroid_hz(coefficients: np.ndarray, dt: float) -> float:
    values = np.asarray(coefficients, dtype=np.float64)
    values = values - np.mean(values, axis=0, keepdims=True)
    spectrum = np.fft.rfft(values, axis=0)
    frequency = np.fft.rfftfreq(values.shape[0], d=dt)
    weights = np.ones(frequency.size, dtype=np.float64)
    weights[0] = 0.5
    if values.shape[0] % 2 == 0:
        weights[-1] = 0.5
    power = weights[:, None] * np.abs(spectrum) ** 2
    denominator = float(np.sum(power))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.sum(power * frequency[:, None]) / denominator)


def build_physical_groups(model: Mapping[str, np.ndarray], dt: float
                          ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    groups: List[Dict[str, Any]] = []
    bases: List[np.ndarray] = []
    for candidate, members in enumerate(conjugate_groups(model["eigenvalues"]), start=1):
        basis, strengths = physical_group_basis(
            model["phi"], model["coefficients"], members)
        offset = sum(item.shape[1] for item in bases)
        frequency = psd_centroid_hz(basis, dt)
        member_array = np.asarray(members, dtype=int)
        eigenfrequency = float(np.mean(np.abs(
            model["omega"][member_array].imag) / (2.0 * np.pi)))
        groups.append({
            "candidate_group": candidate,
            "members": [int(value) for value in members],
            "component_start": offset,
            "component_stop": offset + basis.shape[1],
            "quadrature_count": int(basis.shape[1]),
            "frequency_hz_coefficient_centroid": frequency,
            "strouhal": frequency * 13.716 / 20.0,
            "frequency_hz_dmd_eigenvalue": eigenfrequency,
            "growth_rate_1ps": float(np.mean(model["omega"][member_array].real)),
            "eigenvalue_abs_mean": float(np.mean(np.abs(
                model["eigenvalues"][member_array]))),
            "reduced_group_energy_proxy": float(np.sum(strengths ** 2)),
        })
        bases.append(basis)
    coefficients = np.column_stack(bases)
    if coefficients.shape[1] != model["phi"].shape[1]:
        raise RuntimeError("physical quadrature count does not close the DMD rank")
    order = np.argsort([-item["reduced_group_energy_proxy"] for item in groups])
    ordered_groups: List[Dict[str, Any]] = []
    ordered_columns: List[int] = []
    for selection_order, group_index in enumerate(order, start=1):
        group = dict(groups[int(group_index)])
        old = list(range(group["component_start"], group["component_stop"]))
        group["selection_order_by_reduced_energy"] = selection_order
        group["component_start"] = len(ordered_columns)
        ordered_columns.extend(old)
        group["component_stop"] = len(ordered_columns)
        ordered_groups.append(group)
    return ordered_groups, coefficients[:, ordered_columns]


class CublasRowMajor:
    """Small ctypes SGEMM wrapper; B remains resident on the CUDA device."""

    def __init__(self, b: np.ndarray, cuda_bin: Path):
        if os.name != "nt":
            raise RuntimeError("the bundled cuBLAS path is Windows-only")
        if hasattr(os, "add_dll_directory"):
            self._dll_cookie = os.add_dll_directory(str(cuda_bin))
        else:  # pragma: no cover
            self._dll_cookie = None
        self.cudart = ctypes.CDLL(str(cuda_bin / "cudart64_110.dll"))
        self.cublas = ctypes.CDLL(str(cuda_bin / "cublas64_11.dll"))
        self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.c_int]
        self.cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        self.cudart.cudaDeviceSynchronize.argtypes = []
        self.cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self.cublas.cublasSgemm_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int]
        self._check_cuda(self.cudart.cudaSetDevice(0), "cudaSetDevice")
        self.handle = ctypes.c_void_p()
        self._check_cublas(self.cublas.cublasCreate_v2(ctypes.byref(self.handle)),
                           "cublasCreate")
        self.b = np.ascontiguousarray(b, dtype=np.float32)
        self.k, self.n = (int(value) for value in self.b.shape)
        self.d_b = ctypes.c_void_p()
        self._check_cuda(self.cudart.cudaMalloc(ctypes.byref(self.d_b), self.b.nbytes),
                         "cudaMalloc(B)")
        self._copy_h2d(self.d_b, self.b)

    @staticmethod
    def _check_cuda(status: int, operation: str) -> None:
        if int(status) != 0:
            raise RuntimeError("{} failed with CUDA status {}".format(operation, status))

    @staticmethod
    def _check_cublas(status: int, operation: str) -> None:
        if int(status) != 0:
            raise RuntimeError("{} failed with cuBLAS status {}".format(operation, status))

    def _copy_h2d(self, destination: ctypes.c_void_p, source: np.ndarray) -> None:
        pointer = ctypes.c_void_p(int(source.ctypes.data))
        self._check_cuda(self.cudart.cudaMemcpy(
            destination, pointer, source.nbytes, 1), "cudaMemcpy H2D")

    def matmul(self, a: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != self.k:
            raise ValueError("A has incompatible shape")
        m = int(a.shape[0])
        result = np.empty((m, self.n), dtype=np.float32)
        d_a, d_c = ctypes.c_void_p(), ctypes.c_void_p()
        self._check_cuda(self.cudart.cudaMalloc(ctypes.byref(d_a), a.nbytes),
                         "cudaMalloc(A)")
        self._check_cuda(self.cudart.cudaMalloc(ctypes.byref(d_c), result.nbytes),
                         "cudaMalloc(C)")
        try:
            self._copy_h2d(d_a, a)
            alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
            # Row-major C=A@B is column-major C^T=B^T@A^T.
            self._check_cublas(self.cublas.cublasSgemm_v2(
                self.handle, 0, 0, self.n, m, self.k,
                ctypes.byref(alpha), self.d_b, self.n,
                d_a, self.k, ctypes.byref(beta), d_c, self.n), "cublasSgemm")
            self._check_cuda(self.cudart.cudaMemcpy(
                ctypes.c_void_p(int(result.ctypes.data)), d_c, result.nbytes, 2),
                "cudaMemcpy D2H")
        finally:
            self.cudart.cudaFree(d_a)
            self.cudart.cudaFree(d_c)
        return result

    def close(self) -> None:
        if getattr(self, "d_b", None):
            self.cudart.cudaFree(self.d_b)
            self.d_b = None
        if getattr(self, "handle", None):
            self.cublas.cublasDestroy_v2(self.handle)
            self.handle = None
        if getattr(self, "_dll_cookie", None) is not None:
            self._dll_cookie.close()
            self._dll_cookie = None

    def __enter__(self) -> "CublasRowMajor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class CublasStreamingCross:
    """Accumulate values.T @ coefficients on-device over time batches.

    The source cache is time-major. Keeping the output matrix resident on the
    GPU avoids pathological point-major strided reads and thousands of disk
    seeks per spatial block.
    """

    def __init__(self, n_points: int, n_components: int, maximum_batch: int,
                 cuda_bin: Path):
        if hasattr(os, "add_dll_directory"):
            self._dll_cookie = os.add_dll_directory(str(cuda_bin))
        else:  # pragma: no cover
            self._dll_cookie = None
        self.cudart = ctypes.CDLL(str(cuda_bin / "cudart64_110.dll"))
        self.cublas = ctypes.CDLL(str(cuda_bin / "cublas64_11.dll"))
        self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.c_int]
        self.cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        self.cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self.cublas.cublasSgemm_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int]
        CublasRowMajor._check_cuda(self.cudart.cudaSetDevice(0), "cudaSetDevice")
        self.handle = ctypes.c_void_p()
        CublasRowMajor._check_cublas(
            self.cublas.cublasCreate_v2(ctypes.byref(self.handle)), "cublasCreate")
        self.n_points = int(n_points)
        self.n_components = int(n_components)
        self.maximum_batch = int(maximum_batch)
        self.d_values, self.d_coefficients, self.d_result = (
            ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p())
        value_bytes = self.maximum_batch * self.n_points * 4
        coefficient_bytes = self.maximum_batch * self.n_components * 4
        result_bytes = self.n_points * self.n_components * 4
        for pointer, size, name in (
                (self.d_values, value_bytes, "values"),
                (self.d_coefficients, coefficient_bytes, "coefficients"),
                (self.d_result, result_bytes, "result")):
            CublasRowMajor._check_cuda(
                self.cudart.cudaMalloc(ctypes.byref(pointer), size),
                "cudaMalloc({})".format(name))
        CublasRowMajor._check_cuda(
            self.cudart.cudaMemset(self.d_result, 0, result_bytes),
            "cudaMemset(result)")
        self.result_bytes = result_bytes

    def accumulate(self, values_time_by_point: np.ndarray,
                   coefficients_time_by_component: np.ndarray) -> None:
        values = np.ascontiguousarray(values_time_by_point, dtype=np.float32)
        coefficients = np.ascontiguousarray(
            coefficients_time_by_component, dtype=np.float32)
        batch = int(values.shape[0])
        if values.shape != (batch, self.n_points) or coefficients.shape != (
                batch, self.n_components) or batch > self.maximum_batch:
            raise ValueError("streaming cuBLAS batch has incompatible shape")
        CublasRowMajor._check_cuda(self.cudart.cudaMemcpy(
            self.d_values, ctypes.c_void_p(int(values.ctypes.data)), values.nbytes, 1),
            "cudaMemcpy values H2D")
        CublasRowMajor._check_cuda(self.cudart.cudaMemcpy(
            self.d_coefficients, ctypes.c_void_p(int(coefficients.ctypes.data)),
            coefficients.nbytes, 1), "cudaMemcpy coefficients H2D")
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(1.0)
        # values is row-major (batch, points), hence column-major
        # (points, batch). Transposing that interpretation supplies the
        # desired (batch, points) right factor in C^T=B^T@values.
        CublasRowMajor._check_cublas(self.cublas.cublasSgemm_v2(
            self.handle, 0, 1, self.n_components, self.n_points, batch,
            ctypes.byref(alpha), self.d_coefficients, self.n_components,
            self.d_values, self.n_points, ctypes.byref(beta),
            self.d_result, self.n_components), "cublasSgemm streaming")

    def result(self) -> np.ndarray:
        output = np.empty((self.n_points, self.n_components), dtype=np.float32)
        CublasRowMajor._check_cuda(self.cudart.cudaMemcpy(
            ctypes.c_void_p(int(output.ctypes.data)), self.d_result,
            self.result_bytes, 2), "cudaMemcpy result D2H")
        return output

    def close(self) -> None:
        for name in ("d_values", "d_coefficients", "d_result"):
            pointer = getattr(self, name, None)
            if pointer:
                self.cudart.cudaFree(pointer)
                setattr(self, name, None)
        if getattr(self, "handle", None):
            self.cublas.cublasDestroy_v2(self.handle)
            self.handle = None
        if getattr(self, "_dll_cookie", None) is not None:
            self._dll_cookie.close()
            self._dll_cookie = None


def choose_backend(coefficients: np.ndarray, requested: str,
                   cuda_bin: Path) -> Tuple[Any, Dict[str, Any]]:
    requested = requested.lower()
    if requested not in ("auto", "cuda", "numpy"):
        raise ValueError("backend must be auto, cuda, or numpy")
    if requested in ("auto", "cuda"):
        try:
            engine = CublasRowMajor(coefficients, cuda_bin)
            test_a = np.arange(35, dtype=np.float32).reshape(5, 7)
            test_b = np.asarray(coefficients[:7, :min(3, coefficients.shape[1])],
                                dtype=np.float32)
            with CublasRowMajor(test_b, cuda_bin) as tester:
                observed = tester.matmul(test_a)
            expected = test_a @ test_b
            error = float(np.max(np.abs(observed - expected)))
            if not np.allclose(observed, expected, rtol=2e-5, atol=2e-5):
                engine.close()
                raise RuntimeError("cuBLAS validation max error {}".format(error))
            return engine, {
                "requested": requested, "effective": "cuda-cublas",
                "library": str(cuda_bin / "cublas64_11.dll"),
                "validation_max_abs_error": error,
            }
        except Exception as exc:
            if requested == "cuda":
                raise
            log("CUDA unavailable; falling back to NumPy: {}".format(exc))
    return None, {"requested": requested, "effective": "numpy",
                  "library": np.__version__}


def matmul(engine: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if engine is None:
        return np.asarray(a @ b, dtype=np.float32)
    return engine.matmul(a)


def prefix_capture(a_cross: np.ndarray, gram: np.ndarray, data_energy: float,
                   groups: Sequence[Mapping[str, Any]]) -> List[float]:
    result: List[float] = []
    stop = 0
    for group in groups:
        stop = int(group["component_stop"])
        inverse = np.linalg.pinv(gram[:stop, :stop], rcond=1e-10)
        captured_energy = float(np.trace(inverse @ a_cross[:stop, :stop]))
        result.append(captured_energy / max(data_energy, np.finfo(float).tiny))
    return result


def full_space_statistics(cache: Path, output: Path, coefficients: np.ndarray,
                          scales: np.ndarray, block_points: int, backend: str,
                          cuda_bin: Path, time_batch_size: int,
                          exact_data_energy: float) -> Tuple[np.memmap, np.memmap,
                                                             np.ndarray, float,
                                                             Dict[str, Any]]:
    arrays = [np.load(str(cache / "{}.npy".format(name)), mmap_mode="r")
              for name in ("u", "v", "w")]
    n_time = int(arrays[0].shape[0])
    n_points = int(np.prod(arrays[0].shape[1:]))
    n_components = int(coefficients.shape[1])
    if coefficients.shape[0] != n_time:
        raise ValueError("coefficient and snapshot counts differ")
    cross_path = output / "_full_cross_qc_normalized.npy"
    mean_path = output / "mean_field_physical.npy"
    cross = np.lib.format.open_memmap(str(cross_path), mode="w+", dtype=np.float32,
                                      shape=(n_points, 3, n_components))
    mean_field = np.lib.format.open_memmap(str(mean_path), mode="w+", dtype=np.float32,
                                           shape=(n_points, 3))
    a_cross = np.zeros((n_components, n_components), dtype=np.float64)
    data_energy = float(exact_data_energy)
    backend_info: Dict[str, Any] = {"requested": backend}
    use_cuda = backend in ("auto", "cuda")
    if use_cuda:
        try:
            test_a = np.arange(35, dtype=np.float32).reshape(5, 7)
            test_b = coefficients[:7, :min(3, n_components)]
            with CublasRowMajor(test_b, cuda_bin) as tester:
                observed = tester.matmul(test_a)
            expected = test_a @ test_b
            error = float(np.max(np.abs(observed - expected)))
            if not np.allclose(observed, expected, rtol=2e-5, atol=2e-5):
                raise RuntimeError("cuBLAS validation max error {}".format(error))
            backend_info.update(
                effective="cuda-cublas-streaming",
                validation_max_abs_error=error,
                library=str(cuda_bin / "cublas64_11.dll"))
        except Exception as exc:
            if backend == "cuda":
                raise
            use_cuda = False
            backend_info.update(effective="numpy", fallback_reason=str(exc))
    else:
        backend_info.update(effective="numpy")
    started = time.monotonic()
    for variable, source in enumerate(arrays):
        flat = source.reshape(n_time, n_points)
        sums = np.zeros(n_points, dtype=np.float64)
        for start in range(0, n_time, time_batch_size):
            stop = min(start + time_batch_size, n_time)
            values = np.asarray(flat[start:stop], dtype=np.float32)
            sums += np.sum(values, axis=0, dtype=np.float64)
        mean = (sums / n_time).astype(np.float32)
        mean_field[:, variable] = mean
        if use_cuda:
            accumulator = CublasStreamingCross(
                n_points, n_components, time_batch_size, cuda_bin)
            q_cross = None
        else:
            accumulator = None
            q_cross = np.zeros((n_points, n_components), dtype=np.float32)
        try:
            for start in range(0, n_time, time_batch_size):
                stop = min(start + time_batch_size, n_time)
                values = np.array(flat[start:stop], dtype=np.float32,
                                  order="C", copy=True)
                values -= mean[None, :]
                values /= np.float32(scales[variable])
                coefficient_batch = coefficients[start:stop]
                if accumulator is not None:
                    accumulator.accumulate(values, coefficient_batch)
                else:
                    q_cross += values.T @ coefficient_batch
                if stop % (time_batch_size * 5) == 0 or stop == n_time:
                    log("cross statistics variable {}/3 frames {}/{} ({:.1f} min)".format(
                        variable + 1, stop, n_time,
                        (time.monotonic() - started) / 60.0))
            if accumulator is not None:
                q_cross = accumulator.result()
        finally:
            if accumulator is not None:
                accumulator.close()
        assert q_cross is not None
        cross[:, variable, :] = q_cross
        for start in range(0, n_points, block_points):
            stop = min(start + block_points, n_points)
            q64 = np.asarray(q_cross[start:stop], dtype=np.float64)
            a_cross += q64.T @ q64
        del q_cross
        log("full statistics variable {}/3 complete ({:.1f} min)".format(
            variable + 1, (time.monotonic() - started) / 60.0))
    cross.flush()
    mean_field.flush()
    backend_info.update({
        "block_points": int(block_points), "n_time": n_time,
        "time_batch_size": int(time_batch_size),
        "n_points": n_points, "n_components": n_components,
        "elapsed_seconds": time.monotonic() - started,
    })
    return cross, mean_field, a_cross, data_energy, backend_info


def save_modes(output: Path, cross: np.ndarray, scales: np.ndarray,
               coefficients: np.ndarray, selected_columns: Sequence[int],
               data_energy: float, a_cross: np.ndarray,
               block_points: int) -> Dict[str, Any]:
    columns = np.asarray(selected_columns, dtype=int)
    temporal = np.asarray(coefficients[:, columns], dtype=np.float64)
    gram = temporal.T @ temporal
    inverse = np.linalg.pinv(gram, rcond=1e-10)
    n_points = int(cross.shape[0])
    modes_path = output / "spatial_modes_physical.npy"
    modes = np.lib.format.open_memmap(str(modes_path), mode="w+", dtype=np.float32,
                                      shape=(n_points, 3, columns.size))
    for start in range(0, n_points, block_points):
        stop = min(start + block_points, n_points)
        block = np.asarray(cross[start:stop, :, columns], dtype=np.float64)
        normalized_modes = block.reshape(-1, columns.size) @ inverse
        physical_modes = normalized_modes.reshape(stop - start, 3, columns.size)
        physical_modes *= scales[None, :, None]
        modes[start:stop] = physical_modes.astype(np.float32)
    modes.flush()
    selected_a = a_cross[np.ix_(columns, columns)]
    captured_energy = float(np.trace(inverse @ selected_a))
    residual = max(float(data_energy - captured_energy), 0.0)
    return {
        "metric": "Euclidean norm after common velocity-vector RMS scaling",
        "full_normalized_centered_data_energy": float(data_energy),
        "full_normalized_captured_energy": captured_energy,
        "full_normalized_residual_energy": residual,
        "full_normalized_residual_energy_fraction": residual / data_energy,
        "full_normalized_captured_energy_fraction_one_minus_residual": 1.0 - residual / data_energy,
        "full_normalized_relative_frobenius_error": math.sqrt(residual / data_energy),
        "coefficient_gram_rank": int(np.linalg.matrix_rank(gram)),
        "coefficient_gram_condition_number": float(np.linalg.cond(gram)),
        "n_full_state_rows": int(n_points * 3),
        "n_time": int(temporal.shape[0]),
        "physical_mode_groups": None,
        "real_quadrature_components": int(columns.size),
        "reconstruction_physical_units": (
            "field[n,v,t] = mean_field_physical[n,v] + "
            "sum_q spatial_modes_physical[n,v,q] * temporal_coefficients[t,q]"),
    }


def write_tables(output: Path, groups: Sequence[Mapping[str, Any]],
                 prefix_full: Sequence[float], selected_count: int) -> None:
    candidate_path = output / "candidate_mode_summary.csv"
    fields = [
        "selection_order_by_reduced_energy", "candidate_group",
        "eigenvector_members_1based", "quadrature_count",
        "frequency_hz_coefficient_centroid", "strouhal",
        "frequency_hz_dmd_eigenvalue", "growth_rate_1ps",
        "eigenvalue_abs_mean", "reduced_group_energy_proxy",
        "full_space_captured_fraction_prefix", "selected"]
    with candidate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, group in enumerate(groups):
            writer.writerow({
                **{key: group[key] for key in fields if key in group},
                "eigenvector_members_1based": ";".join(
                    str(value + 1) for value in group["members"]),
                "full_space_captured_fraction_prefix": prefix_full[index],
                "selected": index < selected_count,
            })


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rvmd = Path(args.rvmd_artifact)
    cache = Path(args.cache)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_rvmd = json.loads((rvmd / "metadata.json").read_text(encoding="utf-8"))
    dt = float(metadata_rvmd["dt_seconds"])
    if abs(dt - 0.0125) > 1e-12:
        raise ValueError("DMD comparison requires dt=0.0125 s")
    x_reduced = np.load(str(rvmd / "rvmd_input.npy"), allow_pickle=False)
    scales = np.load(str(rvmd / "component_scales.npy"), allow_pickle=False)
    log("exact DMD on reduced matrix {}".format(x_reduced.shape))
    model = exact_dmd(x_reduced, dt)
    groups, coefficients = build_physical_groups(model, dt)
    log("recovered {} physical groups and {} real quadratures".format(
        len(groups), coefficients.shape[1]))
    component_energy = np.asarray(metadata_rvmd["identification_compression"][
        "component_centered_energy_physical"], dtype=np.float64)
    exact_data_energy = float(np.sum(component_energy / (scales ** 2)))
    cross, mean_field, a_cross, data_energy, backend_info = full_space_statistics(
        cache, output, coefficients.astype(np.float32), scales,
        args.block_points, args.backend, Path(args.cuda_bin),
        args.time_batch_size, exact_data_energy)
    gram_all = np.asarray(coefficients, dtype=np.float64).T @ np.asarray(
        coefficients, dtype=np.float64)
    captures = prefix_capture(a_cross, gram_all, data_energy, groups)
    selected_count = next((index + 1 for index, value in enumerate(captures)
                           if value >= args.energy_target), len(groups))
    passed = captures[selected_count - 1] >= args.energy_target
    selected_energy_groups = list(groups[:selected_count])
    selected_frequency_groups = sorted(selected_energy_groups, key=lambda item: (
        item["frequency_hz_coefficient_centroid"],
        -item["reduced_group_energy_proxy"]))
    final_columns: List[int] = []
    final_groups: List[Dict[str, Any]] = []
    for order, group in enumerate(selected_frequency_groups, start=1):
        copied = dict(group)
        source = list(range(group["component_start"], group["component_stop"]))
        copied["physical_mode_order_low_to_high_frequency"] = order
        copied["final_component_start"] = len(final_columns)
        final_columns.extend(source)
        copied["final_component_stop"] = len(final_columns)
        final_groups.append(copied)
    audit = save_modes(output, cross, scales, coefficients, final_columns,
                       data_energy, a_cross, args.block_points)
    audit["physical_mode_groups"] = int(selected_count)
    audit["target"] = float(args.energy_target)
    audit["passed"] = bool(audit[
        "full_normalized_captured_energy_fraction_one_minus_residual"] >= args.energy_target)
    write_json(output / "full_space_regression_audit.json", audit)
    write_tables(output, groups, captures, selected_count)
    x_axis = np.load(str(cache / "x.npy"), allow_pickle=False)
    y_axis = np.load(str(cache / "y.npy"), allow_pickle=False)
    z_axis = np.load(str(cache / "z.npy"), allow_pickle=False)
    coordinates = np.column_stack([
        np.broadcast_to(x_axis[None, None, :],
                        (z_axis.size, y_axis.size, x_axis.size)).ravel(),
        np.broadcast_to(y_axis[None, :, None],
                        (z_axis.size, y_axis.size, x_axis.size)).ravel(),
        np.broadcast_to(z_axis[:, None, None],
                        (z_axis.size, y_axis.size, x_axis.size)).ravel(),
    ]).astype(np.float32)
    if coordinates.shape[0] != mean_field.shape[0]:
        raise RuntimeError("Cartesian coordinates do not match cached field rows")
    np.save(str(output / "coordinates.npy"), coordinates, allow_pickle=False)
    for axis in ("x", "y", "z"):
        shutil.copy2(str(cache / "{}.npy".format(axis)), str(output / "{}.npy".format(axis)))
    temporal_final = np.asarray(coefficients[:, final_columns], dtype=np.float32)
    np.savez(str(output / "temporal_model.npz"),
             temporal_coefficients=temporal_final,
             time_s=np.arange(temporal_final.shape[0], dtype=np.float64) * dt,
             group_component_start=np.asarray(
                 [item["final_component_start"] for item in final_groups], dtype=np.int32),
             group_component_stop=np.asarray(
                 [item["final_component_stop"] for item in final_groups], dtype=np.int32),
             frequency_hz=np.asarray(
                 [item["frequency_hz_coefficient_centroid"] for item in final_groups]),
             strouhal=np.asarray([item["strouhal"] for item in final_groups]),
             eigenfrequency_hz=np.asarray(
                 [item["frequency_hz_dmd_eigenvalue"] for item in final_groups]),
             growth_rate_1ps=np.asarray(
                 [item["growth_rate_1ps"] for item in final_groups]),
             eigenvalues=model["eigenvalues"], omega=model["omega"],
             reduced_phi=model["phi"].astype(np.complex64),
             projected_complex_coefficients=model["coefficients"].astype(np.complex64),
             singular_values=model["singular_values"])
    selected_table = output / "selected_mode_summary.csv"
    with selected_table.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["physical_mode_order_low_to_high_frequency", "candidate_group",
                  "selection_order_by_reduced_energy", "eigenvector_members_1based",
                  "quadrature_count", "frequency_hz_coefficient_centroid", "strouhal",
                  "frequency_hz_dmd_eigenvalue", "growth_rate_1ps",
                  "eigenvalue_abs_mean", "reduced_group_energy_proxy",
                  "final_component_start", "final_component_stop"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in final_groups:
            row = {key: item[key] for key in fields if key in item}
            row["eigenvector_members_1based"] = ";".join(
                str(value + 1) for value in item["members"])
            writer.writerow(row)
    metadata = {
        "schema_version": 1,
        "method_label": "exact DMD in controlled compressed coordinates plus full-space regression",
        "dataset": "NATO-GD 3-D u/v/w",
        "dt_seconds": dt,
        "n_frames": int(coefficients.shape[0]),
        "n_points": int(coordinates.shape[0]),
        "variables": ["u", "v", "w"],
        "grid_audit": metadata_rvmd["grid_audit"],
        "component_scales_physical": scales.tolist(),
        "identification_compression": metadata_rvmd["identification_compression"],
        "pod_projection": metadata_rvmd["pod_projection"],
        "dmd_rank": int(model["phi"].shape[1]),
        "candidate_physical_groups": len(groups),
        "selected_physical_groups": selected_count,
        "selected_real_quadrature_components": len(final_columns),
        "mode_definition": "one real mode or one complex-conjugate pair; pair rendered through two real quadratures",
        "frequency_definition": "one-sided PSD centroid summed over the real temporal quadratures",
        "eigenfrequency_status": "saved as a diagnostic; not used for St ordering",
        "strouhal_definition": "St = f_center_Hz * 13.716 / 20",
        "selection_definition": "energy-proxy ordering followed by exact mean-removed full-space residual gate",
        "full_space_energy_gate": {
            "target_fraction": float(args.energy_target),
            "captured_fraction": audit["full_normalized_captured_energy_fraction_one_minus_residual"],
            "passed": bool(passed and audit["passed"]),
            "modal_self_energy_sum_prohibited_for_gate": True,
        },
        "full_space_regression_audit": audit,
        "compute_backend": backend_info,
        "source_artifact": str(rvmd.resolve()),
        "source_cache": str(cache.resolve()),
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(output / "metadata.json", metadata)
    del cross, mean_field
    temporary = output / "_full_cross_qc_normalized.npy"
    if temporary.exists():
        temporary.unlink()
    log("DMD full-space result: K={}, capture={:.9f}, passed={}".format(
        selected_count, audit["full_normalized_captured_energy_fraction_one_minus_residual"],
        audit["passed"]))
    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvmd-artifact", default=(
        "runs/research/02_rvmd"))
    parser.add_argument("--cache", default=(
        "runs/research/01_cache"))
    parser.add_argument("--output", default=(
        "runs/dmd/02_dmd"))
    parser.add_argument("--energy-target", type=float, default=0.998)
    parser.add_argument("--block-points", type=int, default=4096)
    parser.add_argument("--time-batch-size", type=int, default=128)
    parser.add_argument("--backend", choices=("auto", "cuda", "numpy"), default="auto")
    parser.add_argument("--cuda-bin", default=".")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if (not 0.0 < args.energy_target <= 1.0 or args.block_points < 1
            or args.time_batch_size < 1):
        raise ValueError("invalid energy target or block size")
    result = run(args)
    print(json.dumps(result["full_space_energy_gate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
