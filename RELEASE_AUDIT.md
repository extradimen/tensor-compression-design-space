# Public-release audit (2026-10-18)

This repository is a deliberately limited code-and-derived-data package for project `bac14391-c670-45aa-891f-2a0aab49abc7`.

- Included: project-authored reference code, synthetic unit tests, legal configuration dictionaries, derived aggregate result tables, frozen analysis scripts, and a 30-record literature map.
- Excluded: base/compressed model weights, raw source texts, token sequences, per-source-cluster scores, campus paths/credentials, detailed run logs and third-party author implementations.
- Evidence scope: 36 initial development endpoints; 36 Step5 cells with 108 nominal labels (12 cells have identical three-label quality vectors); 18 Step5 paired conditions/54 bindings; 14 predefined confirmation rows. Public aggregates do not permit rebuilding bootstrap intervals from sealed per-cluster inputs.
- Validation: 30 synthetic unit tests and `analysis/verify_public_tables.py` passed locally. SHA-256 values for included files are in `SHA256SUMS.txt`.
- License: no repository-wide reuse license has been selected. Consult each upstream model/dataset license for independent acquisition. Public access to these files is not a grant to redistribute omitted materials.
