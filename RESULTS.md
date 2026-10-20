# VeriSearch v0.1 measured pilot results

**OBSERVED RESULT.** These measurements come from real local model generation and full encoder fine-tuning. This is a small, predeclared GSM8K pilot, not a full benchmark evaluation. All percentages below concern exact extracted final answers, not independently checked reasoning steps.

Run timestamp (UTC): `2026-10-03T16:15:55.239462+00:00`; evaluation source commit (development repository): `b502d2067ff026f3bd0409dd82f0bec17dcb692e`. Frozen configuration: [pilot.yaml](configs/pilot.yaml). Machine-readable evidence and checksums: [results/pilot](results/pilot/).

**LIMITATION: answer-format shortcut.** Fit data includes 4 extracted incorrect training answers and 335 missing training answers. Validation includes 0 extracted incorrect answers and 75 missing answers. The diagnostic below checks this confound without changing the frozen policies. High candidate-level scores alone cannot establish mathematical verification.

## Environment and checkpoints

Python 3.13.13; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35. GPU: NVIDIA GeForce RTX 5070 Laptop GPU; Torch 2.10.0+cu130, CUDA runtime 13.0. Generator peak allocated memory: 2974.0 MiB. Peak allocation excludes driver reservations and other processes.

| Component | Checkpoint | Immutable revision |
| --- | --- | --- |
| GSM8K | openai/gsm8k | 740312add88f781978c0658806c59bc2815b9866 |
| Generator | Qwen/Qwen2.5-Math-1.5B-Instruct | aafeb0fc6f22cbf0eaeed126eff8be45b0360a35 |
| Verifier base | distilbert/distilroberta-base | fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b |

Problem-level partitions: 128 train / 32 validation / 48 official-test problems; split seed 42. Generated 416 fit pools (1110 traces) and 528 test pools (1149 traces). Pools at different caps/seeds can contain repeated prefixes; these counts are not independent examples. Verifier fitting uses only one declared pool per problem.

Generation: at most 4 traces/problem, at most 256 new tokens/trace, temperature 0.8, top-p 0.95, top-k 50, sampling seeds [17, 29], total token caps [256, 512, 1024]. Greedy uses one trace. Every recorded pool respects its total generated-token cap.

## Verifier learning and candidate-level evaluation

| Partition | Problems | Traces | Correct | Incorrect | Positive fraction |
| --- | --- | --- | --- | --- | --- |
| Train | 128 | 512 | 173 | 339 | 33.8% |
| Validation | 32 | 128 | 53 | 75 | 41.4% |

Full fine-tuning uses AdamW, learning rate 2e-05, batch size 8, 3 epochs, seed 101, maximum input length 384. Minimum validation cross-entropy chooses the checkpoint. Input truncation: train 6.4%, validation 6.2%.

Test verifier metrics use 192 traces from one maximum-budget pool per test problem at seed 17; correctness prevalence 37.0%.

| Metric | Observed value |
| --- | --- |
| Accuracy at 0.5 | 0.9531 |
| AUROC | 0.9605 |
| AUPRC (average precision) | 0.9075 |
| Brier score | 0.0468 |
| ECE (10 bins) | 0.0469 |

[Reliability plot](results/pilot/plots/reliability.png) and [score distributions](results/pilot/plots/score_distribution.png). **INTERPRETATION.** Ranking, probability quality and downstream selection measure different properties. Classification accuracy alone is insufficient under imbalance.

### Exploratory answer-presence diagnostic

| Test trace category | Count |
| --- | --- |
| Extracted and correct | 71 |
| Extracted and incorrect | 9 |
| Missing extracted answer | 112 |

A trivial classifier assigning probability 1 to extracted answers and 0 to missing answers has accuracy 95.3%, AUROC 0.9628. This exploratory diagnostic was motivated by validation composition and was not used for policy selection, threshold tuning or hyperparameters. Among extracted answers only, learned-verifier AUROC is 0.4695, Brier is 0.1124.

## Fixed generated-token budgets

**OBSERVED RESULT.** Stochastic rows average both declared generation seeds on the same test problems. Intervals are 95% percentile bootstrap intervals from 2000 problem resamples, retaining all seed outcomes within each sampled problem. They are individual accuracy intervals, not significance tests of policy differences.

| Cap | Policy | Accuracy [95% CI] | Mean / median tokens | Mean candidates | Generation s |
| --- | --- | --- | --- | --- | --- |
| 256 | greedy | 35.4% [22.9%, 50.0%] | 237.9 / 256.0 | 1.00 | 7.39 |
| 256 | single | 37.5% [25.0%, 50.0%] | 238.1 / 256.0 | 1.00 | 7.55 |
| 256 | self_consistency | 37.5% [25.0%, 50.0%] | 256.0 / 256.0 | 1.42 | 8.11 |
| 256 | verifier | 37.5% [25.0%, 50.0%] | 256.0 / 256.0 | 1.42 | 8.11 |
| 256 | adaptive_tuned | 37.5% [25.0%, 50.0%] | 238.1 / 256.0 | 1.00 | 7.55 |
| 256 | oracle | 37.5% [25.0%, 50.0%] | 256.0 / 256.0 | 1.42 | 8.11 |
| 512 | greedy | 35.4% [22.9%, 50.0%] | 237.9 / 256.0 | 1.00 | 7.38 |
| 512 | single | 37.5% [25.0%, 50.0%] | 238.1 / 256.0 | 1.00 | 7.51 |
| 512 | self_consistency | 45.8% [33.3%, 58.3%] | 512.0 / 512.0 | 2.54 | 16.19 |
| 512 | verifier | 46.9% [34.4%, 59.4%] | 512.0 / 512.0 | 2.54 | 16.19 |
| 512 | adaptive_tuned | 45.8% [33.3%, 58.3%] | 383.4 / 509.5 | 1.57 | 12.16 |
| 512 | oracle | 47.9% [35.4%, 60.4%] | 512.0 / 512.0 | 2.54 | 16.19 |
| 1024 | greedy | 35.4% [22.9%, 50.0%] | 237.9 / 256.0 | 1.00 | 7.58 |
| 1024 | single | 37.5% [25.0%, 50.0%] | 238.1 / 256.0 | 1.00 | 7.62 |
| 1024 | self_consistency | 55.2% [41.7%, 67.7%] | 953.5 / 1011.0 | 4.00 | 30.36 |
| 1024 | verifier | 54.2% [41.7%, 66.7%] | 953.5 / 1011.0 | 4.00 | 30.36 |
| 1024 | adaptive_tuned | 53.1% [40.6%, 65.6%] | 617.0 / 509.5 | 2.49 | 19.78 |
| 1024 | oracle | 55.2% [41.7%, 67.7%] | 953.5 / 1011.0 | 4.00 | 30.36 |

Paired policy accuracy differences (first minus second), in percentage points, with problem-cluster bootstrap intervals. These are predeclared comparisons; the report makes no statistical significance claim.

| Cap | First policy | Second policy | Difference [95% CI] (pp) |
| --- | --- | --- | --- |
| 256 | verifier | self_consistency | +0.0 [+0.0, +0.0] |
| 256 | adaptive_tuned | verifier | +0.0 [+0.0, +0.0] |
| 512 | verifier | self_consistency | +1.0 [+0.0, +3.1] |
| 512 | adaptive_tuned | verifier | -1.0 [-3.1, +0.0] |
| 1024 | verifier | self_consistency | -1.0 [-3.1, +0.0] |
| 1024 | adaptive_tuned | verifier | -1.0 [-3.1, +0.0] |

**LIMITATION.** Oracle is an analysis-only any-correct upper bound and cannot be deployed. Adaptive token/latency figures are offline prefix replay. Generation latencies are synchronized per-call measurements; verifier time, prompt prefill accounting and model loading are not included in the token budget. No end-to-end speedup or equal-FLOPs claim follows from these figures.

[Accuracy vs token cap](results/pilot/plots/accuracy_vs_budget.png); [accuracy vs candidate count](results/pilot/plots/accuracy_vs_candidates.png).

## Predeclared ablations

### Training data size

| Problem fraction | Train problems | Train traces | AUROC | Brier | Best-of-N accuracy (seed 17) |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 32 | 128 | 0.9639 | 0.0508 | 50.0% |
| 0.5 | 64 | 256 | 0.9671 | 0.0495 | 50.0% |
| 1.0 | 128 | 512 | 0.9605 | 0.0468 | 54.2% |

**LIMITATION.** Nested problem subsets use the same initialization seed and three epochs. Larger subsets also receive more optimizer updates; this does not isolate a causal data-size effect. One training seed cannot establish monotonic scaling. [Training-size plot](results/pilot/plots/training_size.png).

### Sampling temperature

The following table uses the middle budget and **only seed 17 for every temperature**, including the primary temperature, so the comparison is matched.

| Temperature | Policy | Accuracy | Mean tokens |
| --- | --- | --- | --- |
| 0.6 | oracle | 45.8% | 512.0 |
| 0.6 | self_consistency | 43.8% | 512.0 |
| 0.6 | verifier | 45.8% | 512.0 |
| 0.8 | oracle | 47.9% | 512.0 |
| 0.8 | self_consistency | 47.9% | 512.0 |
| 0.8 | verifier | 47.9% | 512.0 |
| 1.0 | oracle | 43.8% | 512.0 |
| 1.0 | self_consistency | 43.8% | 512.0 |
| 1.0 | verifier | 43.8% | 512.0 |

### Adaptive threshold

Validation froze threshold **0.5** before any test generation, using accuracy minus 0.02 times mean consumed budget fraction. The full validation grid and selection are in [tuning.json](results/pilot/tuning.json). The test grid below is descriptive; no test-optimal threshold is selected.

| Threshold | Accuracy | Mean tokens | Mean candidates |
| --- | --- | --- | --- |
| disabled | 54.2% | 953.5 | 4.00 |
| 0.5 | 53.1% | 617.0 | 2.49 |
| 0.7 | 53.1% | 617.0 | 2.49 |
| 0.9 | 53.1% | 617.0 | 2.49 |
| 1.0 | 54.2% | 953.5 | 4.00 |

[Adaptive accuracy vs consumed tokens](results/pilot/plots/adaptive_accuracy_vs_tokens.png). Candidate-count results for N=1,2,4 and all budgets are included in results.json.

## Failure analysis and interpretation

[Structured failure report](results/pilot/errors.md) selects the first examples in stable problem/budget/seed order for each category across all primary-temperature budgets and both seeds, including unfavorable cases. Categories may overlap; the report includes every candidate from each selected pool.

**OBSERVED RESULT.** At the 1024-token cap, learned reranking scored 54.2% versus voting at 55.2%; its paired difference was -1.04 percentage points (95% interval -3.125 to 0). At 512 tokens, the direction reversed by one paired trial (+1.04 points). Adaptive stopping used 617.0 versus 953.5 mean tokens (35.3% fewer) and scored 53.1% versus full reranking's 54.2%. These are generated-token savings under replay, not end-to-end speedups.

**INTERPRETATION.** This pilot does not demonstrate a consistent verifier advantage over self-consistency. Overall candidate classification accuracy matches a presence-only classifier, while conditional AUROC on extracted answers is 0.4695 (only nine incorrect extracted test answers). The evidence is consistent with a strong format shortcut and does not establish mathematical verification. Brier improves with training data here, but AUROC does not improve monotonically. Larger or differently constructed data would be needed to answer the broader research question; no such extension is included in v0.1.

**OBSERVED RESULT.** Across all 288 primary-temperature stochastic pools (both seeds and all three caps), failure analysis finds one distinct problem where voting succeeds and the verifier fails, and one in the opposite direction. Six distinct problems produce high-confidence false-positive selections across 16 pool cases. These counts concern pool conditions and can overlap; they do not imply that a problem fails under every seed or budget.

**INTERPRETATION.** Policy comparisons describe this fixed pilot, generator, prompt and verifier. They do not establish algorithmic novelty, a universal benefit of verification, or a statistically significant improvement. A verifier can prefer incorrect traces even when a correct candidate is available; oracle headroom alone does not show that a learnable verifier can recover it.

**LIMITATION.** Only 48 held-out problems, short traces, one domain, 2 generation seeds and one training seed limit precision and generalization. Exact-answer labels can reward invalid reasoning, and the verifier can learn stylistic shortcuts. Local problem-level isolation does not rule out contamination in pretrained models. See [LIMITATIONS.md](docs/LIMITATIONS.md).

## Reproduction and verification

See [README.md](README.md) for stage commands and [METHODOLOGY.md](docs/METHODOLOGY.md) for data boundaries, metrics and uncertainty. Weights and full trace pools remain outside Git; hashes and configuration are tracked. Compressed per-problem policy outcomes permit reanalysis without rerunning inference. [GPU test evidence](results/pilot/gpu_tests.xml) records actual pretrained generation, scoring, seeded repeatability and live adaptive stopping. Final engineering checks and critical review are recorded in [VALIDATION.md](docs/VALIDATION.md) and [engineering_checks.json](results/pilot/engineering_checks.json). Local acceptance checks passed, and the CPU checks run in GitHub Actions on every push. The release is tagged v0.1.0.
