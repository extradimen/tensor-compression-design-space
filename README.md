# Objective-dependent tensor-compression design space

This is the **public, lightweight evidence package** for a fixed-model study of tensor compression of Qwen3-0.6B-Base. It contains project-authored reference modules, synthetic unit tests, legal design coordinates, 36 initial development endpoints, the 36-cell/three-nominal-label Tucker development summary, paired layout and confirmation tables, and a 30-source literature index. It is not a download of the original model or a stand-alone rerun of the entire GPU study.

## Reproduce the published aggregate checks

From this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m unittest discover -s tests -q
python3 analysis/verify_public_tables.py
```

The first command runs synthetic code tests (not evidence-producing experiments). The second checks public table denominators and descriptive contrast directions without a GPU. `analysis/analyze_protocol86.py` and `analysis/analyze_protocol87.py` are the frozen statistical analysis source as executed in the private evidence workspace; they require the sealed source-cluster inputs and per-model returns, which are **not redistributed here**. Their aggregate outputs are provided in `data/`.

## Scientific scope

The initial matrix/MPO grid contains 36 single-label development endpoints. The Tucker grid contains 36 legal cells and 108 nominal seed-label bindings. Twelve of the 36 Tucker cells have identical text and math NLL vectors across all three labels, so these labels are not automatically independent replicates. The two selected interactions were later evaluated on separate held-out project train-source clusters (32 text and 128 math for each confirmation protocol) using existing models. The results are limited to this model and these fixed conditions. No cross-model generalization, pretraining decontamination, causal speedup or energy saving is claimed.

`data/STEP5_P85_UNIFIED_QUALITY_COST_TABLE_20261018.json` retains `null` for unavailable cost measurements. The 54 newly run final model payloads were audited by independent SHA on the same-project campus node; only a subset also had Mac-local whole-payload SHA proof. Historical and new execution costs cross batches and are observational. The project retained the original Step5-04 source SHA mismatch incident alongside the later isolated successful return.

## Files

- `src/qai_tensor_robust/`: project-authored reference operators, objectives, repair, accounting, splits, and fixed-model measurement components.
- `tests/`: synthetic tests for those components.
- `configs/`: original legal coordinate dictionary and initial grid.
- `data/`: derived numerical tables and Protocol86/87 aggregate results, without raw text, token sequences, model weights or full run logs.
- `analysis/`: public-table verifier and frozen analysis source (the latter requires non-redistributed evidence inputs).
- `docs/`: literature review and checked bibliography.

## Data and code limits

The public numerical tables permit auditing the stated denominators and descriptive directions. They do **not** permit independent recomputation of cluster bootstrap confidence bounds from raw per-source-cluster scores. The sealed text/math inputs are not republished because the source dataset terms and derivation obligations require separate clearance. The 0.6B base model and compressed checkpoints are not republished; model identifiers and SHA attestations remain in the project's controlled evidence store. Public code is project-authored reference code, not third-party author implementations, and is not represented as a complete one-command experimental runner.

The TMLR submission is double-blind. An author-identifying public repository URL should be handled in the manuscript according to current TMLR anonymity rules. This package carries no explicit reuse license pending the research owner's license decision; public visibility alone does not grant unrestricted reuse.
