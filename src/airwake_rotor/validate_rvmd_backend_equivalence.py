# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Audit CPU/GPU RVMD continuation from one persisted checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .rvmd_core import RVMDResult


def _maximum_relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    denominator = np.maximum.reduce(
        [np.abs(first64), np.abs(second64), np.full(first64.shape, 1.0e-30)]
    )
    return float(np.max(np.abs(first64 - second64) / denominator))


def audit_backend_equivalence(
    cpu: RVMDResult,
    gpu: RVMDResult,
    *,
    maximum_stopping_metric_relative_difference: float = 1.0e-3,
    maximum_frequency_difference_hz: float = 1.0e-5,
    maximum_reconstruction_relative_difference: float = 1.0e-5,
    maximum_captured_energy_difference: float = 1.0e-8,
) -> Dict[str, Any]:
    if cpu.phi.shape != gpu.phi.shape or cpu.c.shape != gpu.c.shape:
        raise ValueError("CPU and GPU result shapes differ")
    for key in ("k", "n_state", "n_time", "dt_seconds", "completed_steps"):
        if cpu.info.get(key) != gpu.info.get(key):
            raise ValueError("CPU and GPU metadata disagree for {}".format(key))
    cpu_restart = int(cpu.info.get("restart_from_completed_steps", 0))
    gpu_restart = int(gpu.info.get("restart_from_completed_steps", 0))
    if cpu_restart != gpu_restart:
        raise ValueError("CPU and GPU restart steps differ")
    if cpu_restart <= 0:
        raise ValueError("backend equivalence requires continuation from a saved checkpoint")

    checkpoint_difference_prefix_exact = bool(
        np.array_equal(
            cpu.difference_history[:cpu_restart], gpu.difference_history[:gpu_restart]
        )
    )
    checkpoint_frequency_prefix_exact = bool(
        np.array_equal(
            cpu.omega_history[:, : cpu_restart + 1],
            gpu.omega_history[:, : gpu_restart + 1],
        )
    )
    continuation_slice = slice(cpu_restart, None)
    stopping_metric_maximum_absolute_difference = float(
        np.max(
            np.abs(
                np.asarray(cpu.difference_history[continuation_slice], dtype=np.float64)
                - np.asarray(gpu.difference_history[continuation_slice], dtype=np.float64)
            )
        )
    )
    stopping_metric_maximum_relative_difference = _maximum_relative_difference(
        cpu.difference_history[continuation_slice],
        gpu.difference_history[continuation_slice],
    )
    frequency_difference = np.asarray(cpu.frequency_hz) - np.asarray(gpu.frequency_hz)
    frequency_maximum_absolute_difference_hz = float(np.max(np.abs(frequency_difference)))
    frequency_rms_difference_hz = float(
        np.sqrt(np.mean(np.asarray(frequency_difference, dtype=np.float64) ** 2))
    )
    cpu_reconstruction = cpu.reconstruct()
    gpu_reconstruction = gpu.reconstruct()
    reconstruction_relative_difference = float(
        np.linalg.norm(cpu_reconstruction - gpu_reconstruction)
        / max(np.linalg.norm(cpu_reconstruction), np.finfo(np.float64).tiny)
    )
    cpu_capture = float(cpu.audit["captured_energy_fraction_one_minus_residual"])
    gpu_capture = float(gpu.audit["captured_energy_fraction_one_minus_residual"])
    captured_energy_absolute_difference = float(abs(cpu_capture - gpu_capture))
    cpu_seconds = float(cpu.info.get("iteration_elapsed_seconds_this_call", float("nan")))
    gpu_seconds = float(gpu.info.get("iteration_elapsed_seconds_this_call", float("nan")))
    speedup = (
        float(cpu_seconds / gpu_seconds)
        if np.isfinite(cpu_seconds) and np.isfinite(gpu_seconds) and gpu_seconds > 0
        else None
    )

    checks = {
        "checkpoint_difference_prefix_exact": checkpoint_difference_prefix_exact,
        "checkpoint_frequency_prefix_exact": checkpoint_frequency_prefix_exact,
        "stopping_metric_relative_difference_pass": bool(
            stopping_metric_maximum_relative_difference
            <= maximum_stopping_metric_relative_difference
        ),
        "frequency_difference_pass": bool(
            frequency_maximum_absolute_difference_hz <= maximum_frequency_difference_hz
        ),
        "reconstruction_difference_pass": bool(
            reconstruction_relative_difference
            <= maximum_reconstruction_relative_difference
        ),
        "captured_energy_difference_pass": bool(
            captured_energy_absolute_difference <= maximum_captured_energy_difference
        ),
    }
    return {
        "schema": "rvmd-cpu-gpu-equivalence-audit-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "thresholds": {
            "maximum_stopping_metric_relative_difference": float(
                maximum_stopping_metric_relative_difference
            ),
            "maximum_frequency_difference_hz": float(maximum_frequency_difference_hz),
            "maximum_reconstruction_relative_difference": float(
                maximum_reconstruction_relative_difference
            ),
            "maximum_captured_energy_difference": float(
                maximum_captured_energy_difference
            ),
        },
        "observed": {
            "restart_step": cpu_restart,
            "completed_steps": int(cpu.info["completed_steps"]),
            "stopping_metric_maximum_absolute_difference": (
                stopping_metric_maximum_absolute_difference
            ),
            "stopping_metric_maximum_relative_difference": (
                stopping_metric_maximum_relative_difference
            ),
            "frequency_maximum_absolute_difference_hz": (
                frequency_maximum_absolute_difference_hz
            ),
            "frequency_rms_difference_hz": frequency_rms_difference_hz,
            "reconstruction_relative_difference": reconstruction_relative_difference,
            "cpu_captured_energy_fraction": cpu_capture,
            "gpu_captured_energy_fraction": gpu_capture,
            "captured_energy_absolute_difference": captured_energy_absolute_difference,
            "cpu_iteration_seconds": cpu_seconds,
            "gpu_iteration_seconds": gpu_seconds,
            "gpu_speedup_over_cpu": speedup,
            "gpu_peak_memory_bytes": int(gpu.info.get("peak_gpu_memory_bytes", 0)),
        },
        "cpu_backend": {
            key: cpu.info.get(key)
            for key in (
                "compute_backend_effective",
                "compute_library",
                "compute_library_version",
                "compute_device",
            )
        },
        "gpu_backend": {
            key: gpu.info.get(key)
            for key in (
                "compute_backend_effective",
                "compute_library",
                "compute_library_version",
                "cuda_runtime_version",
                "compute_device",
                "compute_capability",
                "deterministic_algorithms",
                "allow_tf32",
            )
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-model", required=True)
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    cpu_path = Path(args.cpu_model).expanduser().resolve()
    gpu_path = Path(args.gpu_model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    audit = audit_backend_equivalence(
        RVMDResult.load(cpu_path), RVMDResult.load(gpu_path)
    )
    audit["cpu_model"] = str(cpu_path)
    audit["gpu_model"] = str(gpu_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(output_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
