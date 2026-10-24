# Copyright (c) 2026 Olegequ.
"""Run after installing airwake-rotor; no original research data are needed."""
from pathlib import Path

from airwake_rotor.demo import run_demo

if __name__ == "__main__":
    print(run_demo(Path("runs/demo")))
