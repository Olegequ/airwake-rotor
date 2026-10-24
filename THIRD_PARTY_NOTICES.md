# Third-party attribution

The project and integration are maintained under the author ID **Olegequ**.
That attribution does not replace credit for the underlying methods.

## RVMD reference algorithm

`src/airwake_rotor/rvmd_core.py` describes itself as a NumPy/SciPy port aligned
with the real-valued block-coordinate algorithm and public `rvmd.m` reference
implementation by Zimo Liao and collaborators. Its optional PyTorch backend
implements the same update convention. The upstream MIT copyright and
permission notice are retained in [LICENSES/rvmd-MIT.txt](LICENSES/rvmd-MIT.txt).

- Reference repository: <https://github.com/ZimoLiao/rvmd>
- License: <https://github.com/ZimoLiao/rvmd/blob/main/LICENSE>
- Paper: Liao, Z. et al. (2023), “Reduced-order variational mode decomposition
  to reveal transient and non-stationary dynamics in fluid flows”, Journal of
  Fluid Mechanics 966, A7. <https://doi.org/10.1017/jfm.2023.435>
- Upstream license checked: 2026-10-24. No upstream datasets or MATLAB files
  are bundled in this distribution.

## Scientific methods and dependencies

The dynamic inflow and blade-element modules document their own physical
conventions and references. Method citations do not imply endorsement by the
original authors or aircraft manufacturers. Aircraft names identify illustrative
configurations; the project makes no affiliation claim.

NumPy, SciPy, pandas, Matplotlib, Pillow, imageio, and optional PyTorch are
installed as dependencies. They are not vendored into this repository and
retain their respective licenses.

## Data

No rights to an external CFD dataset, publication, aircraft parameter source,
or computed research archive are granted by the software license. The demo's
inputs are generated analytically at runtime. Research data must be acquired
separately by each user.
