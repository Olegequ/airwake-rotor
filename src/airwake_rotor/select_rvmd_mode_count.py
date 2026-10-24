#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Select and audit an RVMD mode count using full-space residual energy only.

The accepted energy definition is the reconstruction capture of the mean-removed,
full three-dimensional flow field::

    eta_K = 1 - ||Q' - Q'_K||^2 / ||Q'||^2

In particular, the sum of individual RVMD modal self-energies is never used for
selection because RVMD spatial modes need not be mutually orthogonal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_TARGET = 0.998
NUMERICAL_TOLERANCE = 1.0e-12


class ModeCountSelectionError(ValueError):
    """Raised when a candidate cannot satisfy the mode-count audit contract."""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ModeCountSelectionError("cannot read JSON {}: {}".format(path, error)) from error
    if not isinstance(payload, dict):
        raise ModeCountSelectionError("JSON root must be an object: {}".format(path))
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _finite_fraction(value: Any, label: str, source: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ModeCountSelectionError(
            "{} is not numeric in {}".format(label, source)
        ) from error
    if not math.isfinite(number) or number < -NUMERICAL_TOLERANCE or number > 1.0 + NUMERICAL_TOLERANCE:
        raise ModeCountSelectionError(
            "{} must be a finite fraction in [0, 1] in {}".format(label, source)
        )
    return min(1.0, max(0.0, number))


def _positive_integer(value: Any, label: str, source: Path) -> int:
    if isinstance(value, bool):
        raise ModeCountSelectionError("{} must be a positive integer in {}".format(label, source))
    try:
        integer = int(value)
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ModeCountSelectionError(
            "{} must be a positive integer in {}".format(label, source)
        ) from error
    if integer <= 0 or not math.isfinite(number) or number != float(integer):
        raise ModeCountSelectionError("{} must be a positive integer in {}".format(label, source))
    return integer


def _mode_count(payload: Mapping[str, Any], audit: Mapping[str, Any], source: Path) -> int:
    values: List[Tuple[str, int]] = []
    for label, container, key in (
        ("metadata.k", payload, "k"),
        ("metadata.mode_count", payload, "mode_count"),
        ("full_space_regression_audit.k", audit, "k"),
    ):
        if key in container and container[key] is not None:
            values.append((label, _positive_integer(container[key], label, source)))
    if not values:
        raise ModeCountSelectionError("candidate has no mode count k: {}".format(source))
    unique = {value for _, value in values}
    if len(unique) != 1:
        raise ModeCountSelectionError(
            "inconsistent mode counts in {}: {}".format(source, values)
        )
    return values[0][1]


def _captured_from_full_space_audit(
    audit: Mapping[str, Any], source: Path
) -> Tuple[float, float, str]:
    """Return capture, residual fraction and the residual evidence used.

    Raw residual/data energy has first priority.  A declared full-space residual
    fraction is the supported fallback.  A captured fraction without residual
    evidence is intentionally insufficient.
    """

    data_key = "full_normalized_centered_data_energy"
    residual_energy_key = "full_normalized_residual_energy"
    residual_fraction_key = "full_normalized_residual_energy_fraction"
    captured_key = "full_normalized_captured_energy_fraction_one_minus_residual"

    residual_candidates: List[Tuple[str, float]] = []
    if data_key in audit and residual_energy_key in audit:
        try:
            data_energy = float(audit[data_key])
            residual_energy = float(audit[residual_energy_key])
        except (TypeError, ValueError) as error:
            raise ModeCountSelectionError(
                "full-space data/residual energy is not numeric in {}".format(source)
            ) from error
        if not math.isfinite(data_energy) or data_energy <= 0.0:
            raise ModeCountSelectionError(
                "full-space centered data energy must be positive in {}".format(source)
            )
        if not math.isfinite(residual_energy) or residual_energy < -NUMERICAL_TOLERANCE * data_energy:
            raise ModeCountSelectionError(
                "full-space residual energy must be non-negative in {}".format(source)
            )
        residual_candidates.append(
            (
                "raw_full_space_residual_over_centered_data_energy",
                _finite_fraction(max(residual_energy, 0.0) / data_energy, "residual fraction", source),
            )
        )
    if residual_fraction_key in audit:
        residual_candidates.append(
            (
                "declared_full_space_residual_fraction",
                _finite_fraction(audit[residual_fraction_key], residual_fraction_key, source),
            )
        )
    if not residual_candidates:
        if captured_key in audit:
            raise ModeCountSelectionError(
                "captured fraction without full-space residual evidence is not accepted: {}".format(
                    source
                )
            )
        raise ModeCountSelectionError(
            "missing full-space residual energy audit in {}".format(source)
        )

    residual_fraction = residual_candidates[0][1]
    for label, other in residual_candidates[1:]:
        if not math.isclose(other, residual_fraction, rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise ModeCountSelectionError(
                "inconsistent full-space residual fractions in {}: {} versus {}".format(
                    source, residual_candidates[0], (label, other)
                )
            )
    captured = 1.0 - residual_fraction
    if captured_key in audit:
        declared = _finite_fraction(audit[captured_key], captured_key, source)
        if not math.isclose(declared, captured, rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise ModeCountSelectionError(
                "captured fraction is not one minus the full-space residual in {}".format(source)
            )
    return captured, residual_fraction, residual_candidates[0][0]


def _captured_from_gate(gate: Mapping[str, Any], source: Path) -> Tuple[float, float, str]:
    definition = str(gate.get("definition", "")).lower()
    if "mean-removed" not in definition or "full" not in definition:
        raise ModeCountSelectionError(
            "full_space_energy_gate must explicitly describe a mean-removed full-space metric: {}".format(
                source
            )
        )
    if gate.get("modal_self_energy_sum_prohibited_for_gate") is not True:
        raise ModeCountSelectionError(
            "full_space_energy_gate must explicitly prohibit modal self-energy summation: {}".format(
                source
            )
        )
    captured = _finite_fraction(gate.get("captured_fraction"), "captured_fraction", source)
    return captured, 1.0 - captured, "declared_mean_removed_full_space_energy_gate"


def _resolve_candidate_payload(candidate: Path) -> Tuple[Path, Dict[str, Any], Dict[str, Any], str]:
    candidate = Path(candidate).resolve()
    if candidate.is_file():
        payload = _load_json(candidate)
        source = candidate
    elif candidate.is_dir():
        metadata_path = candidate / "metadata.json"
        audit_path = candidate / "full_space_regression_audit.json"
        energy_audit_path = candidate / "energy_audit.json"
        if metadata_path.is_file():
            source = metadata_path
            payload = _load_json(metadata_path)
            # Prefer independently written full-space evidence when present, but
            # validate it against the embedded copy below.
            standalone_path = audit_path if audit_path.is_file() else energy_audit_path
            if standalone_path.is_file():
                standalone = _load_json(standalone_path)
                embedded = payload.get("full_space_regression_audit")
                if isinstance(embedded, dict):
                    standalone_capture = _captured_from_full_space_audit(standalone, standalone_path)[0]
                    embedded_capture = _captured_from_full_space_audit(embedded, metadata_path)[0]
                    if not math.isclose(
                        standalone_capture, embedded_capture, rel_tol=1.0e-10, abs_tol=1.0e-12
                    ):
                        raise ModeCountSelectionError(
                            "standalone and embedded full-space audits disagree in {}".format(candidate)
                        )
                payload = dict(payload)
                payload["full_space_regression_audit"] = standalone
                source = standalone_path
        elif audit_path.is_file() or energy_audit_path.is_file():
            source = audit_path if audit_path.is_file() else energy_audit_path
            payload = _load_json(source)
        else:
            raise ModeCountSelectionError(
                "candidate directory has no metadata.json or full-space energy audit: {}".format(
                    candidate
                )
            )
    else:
        raise ModeCountSelectionError("candidate does not exist: {}".format(candidate))

    nested = payload.get("full_space_regression_audit")
    if isinstance(nested, dict):
        return candidate, payload, nested, "full_space_regression_audit"
    # A standalone full-space audit has the residual fields at its root.
    if (
        "full_normalized_residual_energy_fraction" in payload
        or "full_normalized_residual_energy" in payload
    ):
        return candidate, payload, payload, "standalone_full_space_energy_audit"
    gate = payload.get("full_space_energy_gate")
    if isinstance(gate, dict):
        return candidate, payload, gate, "full_space_energy_gate"
    raise ModeCountSelectionError(
        "candidate lacks full_space_energy_gate or full-space residual audit: {}".format(source)
    )


def audit_candidate(candidate: Path, *, target: float = DEFAULT_TARGET) -> Dict[str, Any]:
    target = _finite_fraction(target, "target", Path(candidate))
    if target <= 0.0:
        raise ModeCountSelectionError("target must lie in (0, 1]")
    resolved, payload, evidence, evidence_kind = _resolve_candidate_payload(Path(candidate))
    source = resolved if resolved.is_file() else resolved / "metadata.json"
    if evidence_kind == "full_space_energy_gate":
        captured, residual, residual_source = _captured_from_gate(evidence, source)
        k = _mode_count(payload, {}, source)
        declared_target = evidence.get("target_fraction")
        if declared_target is not None:
            declared_target = _finite_fraction(declared_target, "target_fraction", source)
    else:
        captured, residual, residual_source = _captured_from_full_space_audit(evidence, source)
        k = _mode_count(payload, evidence, source)
        declared_target = None

    rvmd_info = payload.get("rvmd_info")
    converged = bool(isinstance(rvmd_info, dict) and rvmd_info.get("converged") is True)
    energy_passed = bool(captured + NUMERICAL_TOLERANCE >= target)
    declared_target_consistent = bool(
        declared_target is None
        or math.isclose(declared_target, target, rel_tol=0.0, abs_tol=NUMERICAL_TOLERANCE)
    )
    declared_pass_consistent = bool(
        evidence_kind != "full_space_energy_gate"
        or evidence.get("passed") is energy_passed
    )
    eligible = bool(
        energy_passed
        and converged
        and declared_target_consistent
        and declared_pass_consistent
    )

    return {
        "k": k,
        "candidate_path": str(resolved),
        "evidence_kind": evidence_kind,
        "residual_evidence": residual_source,
        "full_space_residual_energy_fraction": residual,
        "full_space_captured_energy_fraction_one_minus_residual": captured,
        "target_fraction": target,
        "passes_energy_target": energy_passed,
        "rvmd_converged": converged,
        "declared_gate_target_consistent": declared_target_consistent,
        "declared_gate_pass_consistent": declared_pass_consistent,
        "eligible_for_selection": eligible,
        "passes_target": eligible,
        "declared_gate_target_fraction": declared_target,
        "selection_metric": "mean_removed_full_space_one_minus_residual_fraction",
        "modal_self_energy_sum_used": False,
    }


def _selection_summary(rows: Sequence[Mapping[str, Any]], target: float) -> Dict[str, Any]:
    passing = [row for row in rows if bool(row["passes_target"])]
    if not passing:
        return {
            "status": "no_evaluated_candidate_meets_target",
            "minimum_evaluated_passing_k": None,
            "strict_minimum_k": None,
            "strict_minimum_demonstrated": False,
            "strict_minimum_reason": "no evaluated K passes the {:.6f} target".format(target),
            "adjacent_predecessor_k": None,
            "adjacent_predecessor_evaluated": False,
            "adjacent_predecessor_failed": False,
        }

    selected = min(passing, key=lambda row: int(row["k"]))
    k = int(selected["k"])
    by_k = {int(row["k"]): row for row in rows}
    predecessor = by_k.get(k - 1)
    predecessor_failed = predecessor is not None and not bool(predecessor["passes_target"])
    strict = bool(predecessor_failed)
    if strict:
        status = "strict_adjacent_boundary_demonstrated"
        reason = "K-1 was evaluated below target and K was evaluated at or above target"
    else:
        status = "minimum_evaluated_passing_k_only"
        if predecessor is None:
            reason = (
                "K={} passes, but K-1={} was not evaluated; this is only the minimum passing "
                "K in the supplied candidate set"
            ).format(k, k - 1)
        else:  # Defensive: normally impossible because k is the smallest passing row.
            reason = "K-1 does not provide a failing adjacent lower boundary"
    return {
        "status": status,
        "minimum_evaluated_passing_k": k,
        "strict_minimum_k": k if strict else None,
        "strict_minimum_demonstrated": strict,
        "strict_minimum_reason": reason,
        "adjacent_predecessor_k": k - 1,
        "adjacent_predecessor_evaluated": predecessor is not None,
        "adjacent_predecessor_failed": predecessor_failed,
    }


def select_mode_count(
    candidate_directories: Iterable[Path],
    output_directory: Path,
    *,
    target: float = DEFAULT_TARGET,
) -> Dict[str, Any]:
    """Audit candidates, write CSV/JSON, and return the selection report."""

    target = _finite_fraction(target, "target", Path(output_directory))
    if target <= 0.0:
        raise ModeCountSelectionError("target must lie in (0, 1]")
    candidates = [Path(path) for path in candidate_directories]
    if not candidates:
        raise ModeCountSelectionError("at least one candidate directory is required")
    rows = sorted((audit_candidate(path, target=target) for path in candidates), key=lambda row: row["k"])
    counts = [int(row["k"]) for row in rows]
    if len(counts) != len(set(counts)):
        raise ModeCountSelectionError(
            "each supplied candidate must have a unique K; received {}".format(counts)
        )

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "mode_count_selection.csv"
    json_path = output / "mode_count_selection.json"
    csv_fields = [
        "k",
        "candidate_path",
        "evidence_kind",
        "residual_evidence",
        "full_space_residual_energy_fraction",
        "full_space_captured_energy_fraction_one_minus_residual",
        "target_fraction",
        "passes_target",
        "passes_energy_target",
        "rvmd_converged",
        "declared_gate_target_consistent",
        "declared_gate_pass_consistent",
        "eligible_for_selection",
        "declared_gate_target_fraction",
        "selection_metric",
        "modal_self_energy_sum_used",
    ]
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    selection = _selection_summary(rows, target)
    pass_pattern = [bool(row["passes_target"]) for row in rows]
    nonmonotonic = any(
        pass_pattern[index] and not pass_pattern[index + 1]
        for index in range(len(pass_pattern) - 1)
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "criterion": {
            "name": "mean_removed_full_space_residual_energy_capture",
            "formula": "eta_K = 1 - ||Q_prime - Q_prime_K||_F^2 / ||Q_prime||_F^2",
            "target_fraction": target,
            "comparison_tolerance": NUMERICAL_TOLERANCE,
            "mean_field_excluded": True,
            "full_spatial_domain_used": True,
            "modal_self_energy_sum_is_not_a_valid_total": True,
            "modal_self_energy_sum_used_for_selection": False,
            "rvmd_convergence_required": True,
        },
        "candidate_count": len(rows),
        "evaluated_mode_counts": counts,
        "candidate_pass_pattern_nonmonotonic": nonmonotonic,
        "selection": selection,
        "strict_minimum_rule": (
            "A strict adjacent-integer boundary is reported only when K passes and the supplied "
            "K-1 candidate fails. Otherwise minimum_evaluated_passing_k is a candidate-grid result, "
            "not a strict minimum claim."
        ),
        "nonlinear_rvmd_caveat": (
            "Independent RVMD optimizations are not guaranteed to be nested or monotone in K; "
            "candidate_pass_pattern_nonmonotonic must therefore be inspected."
        ),
        "candidates": rows,
        "outputs": {
            "candidate_csv": str(csv_path),
            "selection_json": str(json_path),
        },
    }
    _write_json(json_path, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate_directories",
        nargs="+",
        help="RVMD result directories or full-space audit/metadata JSON files",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = select_mode_count(
        [Path(path) for path in args.candidate_directories],
        Path(args.output_directory),
        target=args.target,
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TARGET",
    "ModeCountSelectionError",
    "audit_candidate",
    "select_mode_count",
]
