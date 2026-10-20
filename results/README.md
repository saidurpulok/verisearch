# Tracked measured research artifacts

[pilot/](pilot/) contains the real 128/32/48 GSM8K run on the RTX5070 Laptop GPU.
See [RESULTS.md](../RESULTS.md) for measured tables, uncertainty and interpretation.
The learned verifier did not consistently outperform self-consistency, and the
answer-presence shortcut is documented prominently.

- [results.json](pilot/results.json): configurations, metrics, policy summaries,
  paired accuracy-difference intervals, training-size and shortcut diagnostics.
- [decisions.jsonl.gz](pilot/decisions.jsonl.gz): all 6,288 per-pool policy outcomes;
  read with Python gzip or `gzip -dc`, then parse each JSON line.
- [artifact_manifest.json](pilot/artifact_manifest.json): exported artifact SHA256s.
- [tuning.json](pilot/tuning.json): validation-only threshold selection (0.5).
- [data/manifest.json](pilot/data/manifest.json): pinned official data, split IDs/hashes.
- [verifiers/](pilot/verifiers/): training metadata/checksums for 25/50/100% problem subsets.
- [gpu_tests.xml](pilot/gpu_tests.xml): real GPU generation/scoring/live-adaptive evidence.
- [errors.md](pilot/errors.md): deterministic favorable/unfavorable failure categories,
  with distinct-problem counts and per-pool seed/budget context.
- [plots/](pilot/plots/): required budget/count/calibration/score/adaptive figures,
  plus training-size analysis, each in PNG and PDF.

Full trace pools, prepared datasets, downloaded models and trained weights are
ignored under runs/ and .cache/. Their manifests/hashes are retained. Reproduction
commands are in README; scripts/export_results.py verifies source integrity before
exporting to a new directory. scripts/write_report.py renders measured tables and
requires actual passing GPU JUnit evidence.
The `git_commit` fields in these files were recorded in the development repository
before publication, so they do not resolve here; the published code is the v0.1.0 release. The final human interpretation in
RESULTS.md is specific to this pilot and should be reviewed for any new run.
