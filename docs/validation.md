# Local validation · Olegequ

Validated on **2026-10-24**, Windows, Python **3.12.14**, using an isolated
virtual environment created for the release. The existing research environment
and original numerical sources were retained.

| Check | Result |
| --- | --- |
| Regression and package tests | 102 passed, 1 skipped; 9 subtests passed |
| Skipped check | Optional PyTorch test; PyTorch absent from the CPU environment |
| Dependency consistency | `pip check` passed |
| Build | Source distribution and platform-independent wheel built successfully |
| Wheel isolation | Imported the installed wheel from outside the source checkout; synthetic coupled demo passed |
| CLI | All 20 command help entry points passed from the installed wheel |
| Source audit | No findings in the targeted release scan; no original CFD, result arrays, manuscripts, or private documents in the release list |
| Original code | SHA-256 values of all 24 extracted source modules still match the extraction manifest |

The two-frequency demo recovered **1.56234445 Hz** and **3.12468463 Hz** from
known **1.5625 Hz** and **3.125 Hz** inputs. Relative reconstruction error was
**0.0231831** and RVMD converged. The synthetic hover trim's relative thrust
residual was **1.61 × 10⁻⁹**. Full, highest-frequency-deleted and mean-only
cases produced finite coupled histories; the mean-only case retained the
shared hover equilibrium.

Direct runtime dependencies used locally:

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| pandas | 2.3.3 |
| Matplotlib | 3.11.2 |
| Pillow | 12.3.0 |
| imageio | 2.38.1 |
| pytest | 9.1.1 |
| setuptools | 84.2.1 |
| build | 1.6.1 |

The GitHub Actions matrix is provided but has not run on GitHub before the
initial upload. Linux, Python 3.10, and CUDA execution have not been verified
locally. Full original-data research reproduction was not rerun during
packaging; it needs the external CFD dataset and confirmed study geometry.
The release scanner uses targeted heuristics and does not establish legal
ownership or prove the absence of every possible secret.
