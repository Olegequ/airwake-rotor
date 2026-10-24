# Copyright (c) 2026 Olegequ.
"""Portable entry points for the research modules."""
import argparse
import importlib
import sys

from . import __author__, __version__

COMMANDS = {
    "demo": "demo",
    "cache": "nato_gd_3d_io",
    "rvmd": "nato_gd_rvmd_adapter",
    "select-k": "select_rvmd_mode_count",
    "couple": "rvmd_nato_coupling_pipeline",
    "dmd": "nato_gd_dmd_3d",
    "couple-dmd": "nato_gd_dmd_coupling_pipeline",
    "study": "nato_gd_study",
    "statistics": "finalize_time_weighted_statistics",
    "sensitivity": "analyze_normalized_modal_sensitivity",
    "plot-sensitivity": "plot_normalized_modal_sensitivity",
    "plot-results": "plot_rvmd_3d_results",
    "stability": "analyze_modal_deletion_stability",
    "numerical-qa": "run_numerical_sensitivity_qa",
    "render-gifs": "render_rvmd_deletion_flow_gifs",
    "audit-gifs": "audit_rvmd_gifs",
    "audit-frequency": "audit_rvmd_frequency",
    "hover-audit": "hover_stability_audit",
    "backend-check": "validate_rvmd_backend_equivalence",
    "final-gate": "final_output_gate",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in COMMANDS:
        command, rest = args[0], args[1:]
        previous = sys.argv
        try:
            sys.argv = ["airwake-rotor " + command] + rest
            module = importlib.import_module("airwake_rotor." + COMMANDS[command])
            entrypoint = module._main if command == "study" else module.main
            return entrypoint(rest)
        finally:
            sys.argv = previous
    parser = argparse.ArgumentParser(
        prog="airwake-rotor",
        description="Airwake modal analysis and nonlinear rotorcraft dynamics | Olegequ",
        epilog="Run airwake-rotor COMMAND --help for command options. Author: Olegequ.",
    )
    parser.add_argument("--version", action="version", version=f"airwake-rotor {__version__} | {__author__}")
    parser.add_argument("command", choices=sorted(COMMANDS), nargs="?")
    parser.parse_args(args)
    parser.print_help()
    return 0
