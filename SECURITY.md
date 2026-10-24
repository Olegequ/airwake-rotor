# Security

Maintainer: **Olegequ**.

For an ordinary bug, provide a minimal synthetic example in the repository
issue tracker. Do not include credentials or restricted datasets. For a
vulnerability, use GitHub's private vulnerability reporting if the maintainer
has enabled it; otherwise contact the maintainer through the account hosting
the repository before publishing sensitive details.

Load artifacts and configuration files only from sources you trust. Portable
NumPy artifacts in the main workflow are read without pickle. The software
processes local data and does not include credential handling or telemetry.
