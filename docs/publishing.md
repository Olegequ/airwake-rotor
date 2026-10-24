# Publishing this project · Olegequ

The working release is prepared as a standalone repository. Review the package,
author attribution, third-party notice, license, and local validation record
before making the repository public. Original research data and private
materials are outside this release.

## Local verification

```bash
python -m pip install -e ".[dev]"
python -m pytest
airwake-rotor demo --output runs/demo
python scripts/check_release.py
python -m build
```

The release uses MIT, with copyright attributed to **Olegequ** and
the reference RVMD notice preserved separately. Confirm that the release owner
is authorized to distribute the selected code under this license. Private
research/patent documents are excluded from the source tree; any decision about
publishing the underlying method belongs to its owner.

## GitHub repository

Ensure `main` has an initial commit before creating the remote. A source ZIP
does not contain Git history, so initialize it with `git init -b main` first.
Configure your actual Git author identity if Git asks for it; do not substitute
an unverified GitHub email address. Then run:

```bash
git add .
git commit -m "Prepare airwake-rotor 0.1.0"
```

After selecting the GitHub owner and confirming the public release, create an
empty repository named `airwake-rotor`. If using the GitHub CLI after connecting
your account, from this directory:

```bash
gh auth status
gh repo create Olegequ/airwake-rotor --public --source . --remote origin --push --description "Airwake modal decomposition and nonlinear rotorcraft response | Olegequ"
```

The selected owner is `Olegequ`. The command above is
an explicit public publication step. The author ID alone does not establish
which GitHub account owns the repository.

If the empty remote repository already exists:

```bash
git remote add origin https://github.com/Olegequ/airwake-rotor.git
git push -u origin main
```

Once hosted, enable private vulnerability reporting if desired. Inspect
GitHub Actions results before tagging the first version. Hosting the source
on GitHub does not automatically publish the Python package to PyPI; a separate
package-index release is optional.
