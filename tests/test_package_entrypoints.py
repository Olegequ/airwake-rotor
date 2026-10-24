# Copyright (c) 2026 Olegequ.
"""Exercise the installed command interface from outside the source tree."""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_installed_version_from_unrelated_directory():
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-m", "airwake_rotor", "--version"], cwd=directory,
            capture_output=True, text=True,
        )
    assert result.returncode == 0, result.stderr
    assert "0.1.0" in result.stdout and "Olegequ" in result.stdout


def test_invalid_cli_command_fails_without_creating_outputs():
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-m", "airwake_rotor", "not-a-command"], cwd=directory,
            capture_output=True, text=True,
        )
        assert list(Path(directory).iterdir()) == []
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
