# External input data

Raw CFD data are not included. The `airwake-rotor demo` command generates its
own analytic inputs and writes outputs under `runs/demo`.

For the complete study, place authorized CSV frames under `data/cfd/` or edit
`study.source_data_dir` in `config/nato_gd_rvmd_3d.json`. Data files in this
directory are ignored by Git. See [the input contract](../docs/workflow.md).
