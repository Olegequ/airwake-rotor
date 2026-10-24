# airwake-rotor

**Airwake modal decomposition and nonlinear rotorcraft response · Olegequ**

[中文说明](README.zh-CN.md) · [Research workflow](docs/workflow.md) · [Model assumptions](docs/model_assumptions.md) · [Attribution](THIRD_PARTY_NOTICES.md)

`airwake-rotor` packages a Python research workflow for decomposing unsteady
three-dimensional airwakes and studying their one-way influence on rotor loads,
dynamic inflow, roll, pitch, and heave. It includes real-valued reduced-order
variational mode decomposition (RVMD), grouped DMD reconstruction, nonlinear
blade-element loads, a near-hover three-state Pitt–Peters model, and modal
deletion / leave-one-out response analysis.

The software is a research prototype. The included configurations retain
study-specific assumptions and representative aircraft parameters. The
synthetic demo verifies installation and exercises the numerical workflow; it
does not reproduce a published aircraft or CFD result.

## Quick start

Use Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv
```

Activate the environment in your shell:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install and run the example:

```bash
python -m pip install -e ".[dev]"
airwake-rotor --version
airwake-rotor demo --output runs/demo
python -m pytest
```

`python -m airwake_rotor` is equivalent to the `airwake-rotor` command.
The demo creates two known temporal frequencies, decomposes them, trims a
synthetic rotor, and runs full, highest-frequency-deleted, and mean-only cases
through the nonlinear blade-element / inflow / rigid-body solver. It writes
CSV histories, an NPZ decomposition, JSON metadata, and a figure bearing the
`Olegequ` watermark. It needs no external data, GPU, or account.

![Synthetic RVMD and coupled rotor response; Olegequ](docs/assets/demo.png)

## What is included

| Capability | Interface |
| --- | --- |
| Validated, resumable CFD velocity cache | `airwake-rotor cache --help` |
| CountSketch / POD → RVMD → full-space regression | `airwake-rotor rvmd --help` |
| Converged residual-energy selection of K | `airwake-rotor select-k --help` |
| Hover trim and nonlinear one-way coupling | `airwake-rotor couple --help` |
| Grouped DMD and paired-quadrature reconstruction | `airwake-rotor dmd --help`, `couple-dmd --help` |
| Boundary-interpolated, time-weighted statistics | `airwake-rotor statistics --help` |
| Single-mode / leave-one-out sensitivity and St90 | `airwake-rotor sensitivity --help` |
| Plots, GIFs, stability and provenance audits | `airwake-rotor --help` |

For programmatic use:

```python
import numpy as np
from airwake_rotor.rvmd_core import rvmd

dt = 0.02
t = np.arange(256) * dt
q = np.vstack([np.sin(2*np.pi*1.5625*t), np.cos(2*np.pi*3.125*t)])
result = rvmd(q, k=2, alpha=1000, dt=dt, dtype="float64")
reconstruction = result.reconstruct()  # state × time
print(result.frequency_hz)
result.save("runs/example.npz")
```

## Bring your own CFD data

The original CFD dataset, computed results, papers, and local automation are
not distributed. Generate the demo locally or supply data you are authorized
to use. The expected CSV schema, path rules, required configuration edits,
energy gates, and complete execution sequence are in
[docs/workflow.md](docs/workflow.md).

The full study configuration uses a reference length of **13.716 m**, reference
speed of **20 m/s**, and an example sampling interval of **0.0125 s**. Its loader
currently enforces the two reference scales. Do not reuse these values for a
different experiment without adapting and validating the study layer.
Set your hub position, verified timing, source-frame range and selected K
before running the study. The template deliberately retains an unresolved
geometry status. `--allow-provisional` keeps results labelled provisional.

CUDA acceleration of RVMD is optional. Install a PyTorch build appropriate for
your hardware, then use `--compute-backend torch-cuda`; explicit CUDA requests
fail if unavailable. `--compute-backend auto` falls back to NumPy. The separate
DMD cuBLAS backend is experimental and Windows-specific; `--backend numpy`
provides the portable DMD path. GPU execution is not part of the default CI.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/check_release.py
python -m build
```

CI runs the regression suite and synthetic demo on Linux and Windows with
Python 3.10 and 3.12, then checks the built wheel from outside the checkout.
See [CONTRIBUTING.md](CONTRIBUTING.md) and the [local validation record](docs/validation.md). The source-to-package mapping is recorded
in [docs/source_manifest.json](docs/source_manifest.json).

## Credit and license

Project author and maintainer: **Olegequ**. Code files, package metadata, the
command-line version, and exported primary figures identify the author. The
watermark is added at figure export and does not alter numerical arrays or CSVs.

The project uses the [MIT license](LICENSE). It preserves the RVMD
reference implementation's copyright and permission notice in
[LICENSES/rvmd-MIT.txt](LICENSES/rvmd-MIT.txt). The method is described by
[Liao et al., JFM 966, A7 (2023)](https://doi.org/10.1017/jfm.2023.435), with
reference code at [ZimoLiao/rvmd](https://github.com/ZimoLiao/rvmd).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for scope and attribution.
