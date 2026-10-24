# Copyright (c) 2026 Olegequ.
"""Scan a prospective source release; print finding locations, never secrets.

This targeted heuristic check complements a human review. It does not prove
absence of all credentials or establish ownership/licensing of source code.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
FORBIDDEN = {"runs", "results", "outputs", "modal_coupling_results", "manuscript", ".codex", ".agents"}
ALLOWED_ROOT = {".github", "src", "tests", "config", "docs", "examples", "scripts", "LICENSES", "data"}
PATTERNS = {
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "OpenAI credential": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "absolute Windows path": re.compile(r"\b[A-Za-z]:[\\/](?:Users|Anaconda|cuda|1\.)", re.I),
}


def source_paths():
    # An index is used once staged, keeping generated/ignored files out of the
    # publication audit. Before staging, inspect the prospective source tree.
    if (ROOT / ".git").exists():
        command = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
        if command.returncode:
            raise RuntimeError("Cannot read the Git publication index")
        names = command.stdout.decode("utf-8").split("\0")
        if any(names):
            return [ROOT / name for name in names if name]
    paths = []
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if path.is_file() and not any(part in SKIP or part.endswith(".egg-info") for part in parts):
            paths.append(path)
    return paths


def main():
    findings = []
    paths = source_paths()
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            findings.append((relative, "symlink requires review"))
        if relative.parts[0] in FORBIDDEN or (relative.parts[0] == "data" and relative.as_posix() != "data/README.md"):
            findings.append((relative, "excluded research/runtime material"))
        if len(relative.parts) > 1 and relative.parts[0] not in ALLOWED_ROOT:
            findings.append((relative, "unexpected release directory"))
        if path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".pfx", ".docx", ".xlsx", ".pdf", ".npz", ".npy"}:
            findings.append((relative, "credential, private document, or generated data file"))
        if path.stat().st_size > 10_000_000:
            findings.append((relative, "file exceeds the source-release size budget"))
        if path.suffix.lower() in {".png", ".jpg", ".gif"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append((relative, "unexpected binary file"))
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                findings.append((relative, f"{label} at line {line}"))
    for relative, message in findings:
        print(f"REVIEW {relative.as_posix()}: {message}")
    print(f"Scanned {len(paths)} release files; {len(findings)} findings.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
