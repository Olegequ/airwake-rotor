# Complete research workflow · Olegequ

The numerical kernels are reusable; this study layer retains the original
NATO-GD reference scales and output contracts. All examples below run from the
repository root after installation. Commands are written on single lines so
they also work in PowerShell.

## Input contract

Provide a contiguous sequence of CSV frames with an identical header and
identical point order in every frame. Required columns, in order:

```text
Velocity[i] (m/s),Velocity[j] (m/s),Velocity[k] (m/s),Pressure (Pa),X (m),Y (m),Z (m)
```

Use finite SI values. The strict cache reader verifies numbering, row count,
coordinates, regular Cartesian-grid structure, and finite values. Supply the
true physical `--dt`; filenames do not establish time spacing. Pressure is
present in this input schema but excluded from the velocity-only RVMD and
density model.

The default filename prefix is `流场数据_table_`; change `--prefix` to match
your export. The following commands assume `flow_2001.csv` through
`flow_4500.csv`. Replace the IDs and interval to match your own data:

```bash
airwake-rotor cache scan --source data/cfd --prefix flow_ --start-id 2001 --end-id 4500 --dt 0.0125 --report runs/input_scan.json
airwake-rotor cache build-cache --source data/cfd --cache runs/research/01_cache --prefix flow_ --start-id 2001 --end-id 4500 --dt 0.0125
airwake-rotor cache smoke --cache runs/research/01_cache --dt 0.0125
```

Cached velocities are float32, time-major memory-mapped NPY arrays. Keep cache
metadata and source data available when resuming a build. Queries outside the
stored space/time domain fail; the production reconstruction does not clip or
extrapolate.

## RVMD identification and model order

Run candidate decompositions. The example values below reproduce the original
study's *settings*, without claiming to reproduce its dataset or results:

```bash
airwake-rotor rvmd --cache-dir runs/research/01_cache --output-dir runs/research/02_rvmd_candidates/K152 --dt 0.0125 --alpha 1000 --k 152 --variables u v w --method countsketch --countsketch-rows-per-variable 256 --pod-energy-target 0.9998 --maximum-steps 30000 --tolerance 0.005 --init-frequency-max-hz 6 --compute-backend numpy --minimum-full-space-captured-energy 0.998
airwake-rotor select-k runs/research/02_rvmd_candidates/K144 runs/research/02_rvmd_candidates/K152 --output-directory runs/research/09_qa --target 0.998
```

The second command assumes **both** candidate runs exist. Run the first
command again with `--k 144` and the matching output directory before using it.
Selection evaluates convergence and full-space mean-removed residual energy.
It does not simply sum modal self-energies: RVMD modes need not be orthogonal.
Do not assume K=152 is valid for another dataset. A passing K is strictly
minimal only if the required lower candidate has also been evaluated and fails.

The coupling workflow expects the chosen artifact under
`runs/research/02_rvmd`. To create that directory by a portable, audited run,
repeat identification with the selected K and
`--output-dir runs/research/02_rvmd` (or copy the complete selected artifact,
including metadata and audits, to that location).

For exact continuation, provide `--restart-directory` pointing to the previous
identification output. `--maximum-steps` is the total sweep limit. Retain the
unsorted restart state; sorted public modes cannot replace it.

## Configure coupling

Edit `config/nato_gd_rvmd_3d.json` before coupling:

| Field | Required action |
| --- | --- |
| `study.source_data_dir` | Point to the authorized raw-frame directory |
| `study.snapshot_prefix`, frame IDs | Match the input filenames |
| `study.sample_interval_s`, status | Use verified timing and describe its provenance |
| `study.mode_count`, `rvmd.mode_count` | Set both to the selected artifact's K |
| `coordinates.initial_hub_world_m`, status | Replace the placeholder with verified CFD coordinates |
| `rotor_model.base_rotor_config` | Supply or review rotor parameters in `rotor_config.json` |
| `rotor_model.dynamic_overrides` | Review inertia, damping, and status declarations |
| `body_response` | Set the common analysis window and review control assumptions |

Relative paths in a study file resolve against the parent of its `config/`
directory. Keep the delivered `config/` layout, or call `load_study_context`
with an explicit `project_root` from Python. The model checks that all rotor
points remain inside the CFD volume and the time window is available.

```bash
airwake-rotor study --config config/nato_gd_rvmd_3d.json
airwake-rotor couple --config config/nato_gd_rvmd_3d.json --modal-sensitivity --workers 1
airwake-rotor statistics --config config/nato_gd_rvmd_3d.json
```

Use `--allow-provisional` only for exploratory configurations with an inferred
hub position; it retains provisional metadata. It does not resolve the missing
geometry confirmation or relax space/time boundaries. For a smoke run, inspect
`couple --help` for rotor, stage and time-limit options. Full research runs can
be expensive.

## Sensitivity and plots

```bash
airwake-rotor sensitivity --case-manifest runs/research/05_statistics/modal_sensitivity_case_manifest.csv --rvmd-dir runs/research/02_rvmd --analysis-config config/normalized_modal_sensitivity.json --output-dir runs/research/05_statistics/normalized_modal_sensitivity
airwake-rotor plot-sensitivity --mode-sensitivity runs/research/05_statistics/normalized_modal_sensitivity/mode_sensitivity_long.csv --diameter-metrics runs/research/05_statistics/normalized_modal_sensitivity/diameter_frequency_metrics.csv --output-directory runs/research/06_figures/normalized_modal_sensitivity --dpi 600
airwake-rotor stability --config config/nato_gd_rvmd_3d.json
airwake-rotor numerical-qa --config config/nato_gd_rvmd_3d.json --workers 1
airwake-rotor plot-results --config config/nato_gd_rvmd_3d.json
airwake-rotor render-gifs --config config/nato_gd_rvmd_3d.json --frames 200 --fps 10 --workers 1
airwake-rotor audit-gifs --config config/nato_gd_rvmd_3d.json
airwake-rotor final-gate --config config/nato_gd_rvmd_3d.json
```

Main plots from `plot-results` and `plot-sensitivity` carry a small `Olegequ`
watermark. Source CSVs and arrays are unchanged. GIF generation retains the
original visualization implementation.

Time-weighted statistics use interpolation at exact analysis-window boundaries.
The final output gate expects the *complete* study output tree, including the
required sensitivity and figure artifacts. A short demo is not a passing
production gate.

## DMD comparison

The DMD path consumes the RVMD precompression artifacts and the same validated
velocity cache. It retains complex-conjugate quadratures together as physical
groups. The portable execution path is:

```bash
airwake-rotor dmd --rvmd-artifact runs/research/02_rvmd --cache runs/research/01_cache --output runs/dmd/02_dmd --backend numpy --energy-target 0.998
airwake-rotor couple-dmd --config config/nato_gd_dmd_3d_coupling.json --workers 1
```

Before coupling DMD, replace the illustrative `decomposition.mode_count` and
`real_quadrature_component_count` in the overlay with your artifact's audited
values. The GPU cuBLAS implementation currently loads Windows CUDA libraries;
explicit CUDA use needs the appropriate `--cuda-bin` directory.

## Output layout

`runs/research/` contains provenance, velocity cache, RVMD artifacts, trim,
time-history CSVs, statistics, figures, GIFs, the figure manual and QA reports.
Outputs are ignored by Git. Metadata signatures depend on code and paths;
moving the package or changing source/configuration can invalidate an old
resume signature. Regenerate affected results after such changes.
