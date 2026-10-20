# v0.1 validation and critical review

All implementation, measured experiment and local acceptance checks passed on
2026-10-03. This is a small pilot with a negative/qualified finding, not evidence
of algorithmic novelty or mathematical verification.

## Final commands and outcomes

| Command | Actual outcome |
|---|---|
| `.venv/bin/pytest -m 'not gpu and not pretrained and not slow'` |46 passed,1 GPU deselected;3.49s|
| `HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 VERISEARCH_GPU_TESTS=1 .venv/bin/pytest -m gpu --junitxml=runs/pilot/gpu_tests.xml` |1 real GPU test passed,46 CPU tests deselected;28.27s|
| `.venv/bin/ruff check .` |Pass|
| `.venv/bin/ruff format --check .` |46 files already formatted|
| `.venv/bin/mypy src` |Pass,22 source files at configured strictness|
| `UV_CACHE_DIR=/tmp/verisearch-uv uv pip check --python .venv/bin/python` |83 packages compatible|
| `UV_CACHE_DIR=/tmp/verisearch-uv uv build --offline` |Source archive and wheel built successfully|

The cache override was specific to the development machine; normal reproduction
uses `uv sync --frozen`.
[Engineering outputs and tested-source hashes](../results/pilot/engineering_checks.json)
and [GPU JUnit evidence](../results/pilot/gpu_tests.xml) preserve actual evidence.
GPU tests use the real pinned Qwen model, real trained encoder, deterministic
repeat checks, best-of-N, and live adaptive early/full sampling with both models
on CUDA. CPU controlled traces are mechanism tests, not GPU evidence.

## Research validity review

- Audited 944 immutable pools and 6288 policy decisions: query/split/gold consistency,
  pinned generator revision, actual token caps, selected labels and oracle bounds.
- Verified train/validation/test query disjointness and trained-checkpoint hashes.
  Policies receive Query/Candidate with no gold; labels attach only for evaluation.
- Verified validation threshold 0.5 was frozen before test generation. No test-set
  tuning, model changes, budget changes or favorable-result cherry-picking.
- Independently recomputed binary metrics, policy summaries and paired problem-
  cluster bootstrap intervals from saved probabilities/decisions.
- Temperature comparisons use the same seed 17. Data-size subsets are nested at
  problem level; fixed epochs also change update count, which is documented.
- Identified the missing-answer shortcut from validation data and added an
  exploratory diagnostic without changing selection. Test conditional AUROC 0.4695
  and parity with answer-presence classification limit mathematical-checking claims.
- Expanded descriptive failure coverage to both seeds/all primary caps to expose
  actual policy disagreements. Counts separate distinct problems from pool cases;
  examples are the first three distinct IDs in deterministic order, with all traces.
- Oracle is explicitly analysis-only. Tokens exclude verifier/prefill compute.
  Adaptive cost/latency outcomes are replay, not end-to-end measured speedups.
- Inspected all six plots, fixed overlapping threshold labels, checked artifact
  checksums and documentation links. No weights, environments, caches or raw large
  trace datasets are tracked. Small intentionally exported evidence is under 1 MiB.

## Hosted CI

The CPU checks (ruff, mypy and the CPU test suite) run in GitHub Actions on every
push; see the badge in the README for the current status.

## Not verified or established

No full 1319-problem GSM8K evaluation, secondary dataset, multiple
verifier training seeds, bitwise-deterministic retraining, equal total FLOPs,
end-to-end adaptive speedup, statistical significance, or algorithmic novelty is
claimed. CUDA attention backward emitted a nondeterminism warning during training.

## Release provenance

The annotated `v0.1.0` tag marks the published release: the implementation,
measured evidence and local acceptance records. The `git_commit` values recorded
in `results/pilot/` were captured in the development repository before publication
and do not resolve in this one.
