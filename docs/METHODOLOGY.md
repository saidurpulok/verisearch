# Methodology

The primary question is whether an outcome verifier improves small-model math
answer selection under generated-token caps. The pilot is specified by
[configs/pilot.yaml](../configs/pilot.yaml) before examining held-out outcomes.
No algorithmic novelty is asserted.

## Data boundaries
Official GSM8K training questions are deterministically hash-ranked using seed
42, then divided into 128 training and 32 validation questions. The official
test split independently supplies 48 questions. These are small subsets, not
the full benchmark. Whitespace-normalized question hashes prevent duplicates;
inconsistent duplicate answers and official split overlap cause errors.
Every trace from one question stays in its partition.

Train pools use four stochastic candidates, temperature 0.8, seed 17 and a
1024-token cap with at most 256 tokens/trace. Verifier validation uses the
1024-token pools at seed 17. Correctness labels compare exact numeric values;
reasoning validity is not independently labeled.

The training-size ablation uses nested 25%, 50%, 100% **problem** subsets, each
initialized from the same encoder checkpoint/seed and trained for three epochs.
Minimum validation cross-entropy selects each saved epoch. These are one-seed
training runs, not a test of a universal scaling law. Fixed epochs imply more
optimizer updates for larger subsets, so this ablation combines data quantity
with training compute; it does not isolate a causal data-size effect.

## Policies and fair comparisons
Greedy uses one deterministic completion. Single stochastic uses candidate zero.
Self-consistency votes among valid extracted answers, ignoring missing answers;
ties choose the earliest tied answer. If all answers are missing, return the
first trace. Verifier best-of-N chooses the highest probability, ties earliest.
Oracle succeeds if any consumed candidate is correct and is exclusively a
nondeployable analytical upper bound.

For each question/budget/temperature/seed, reranking, voting, single and adaptive
reuse the same sequentially generated pool. Greedy has its own deterministic
completion. Candidate-count caps are 1, 2 and 4. Generated-token caps are
256, 512 and 1024, including EOS, with remaining-budget truncation enforced at
generation time. Different budgets require their own pools.
These are ceilings; policies need not consume all available tokens. Confidence,
candidate caps and EOS can leave unused budget.

Two seeds (17,29) evaluate the main temperature 0.8 at each budget.
Temperature 0.6/1.0 ablations use seed 17 at the middle budget (512), and
must be compared to the same-seed 0.8 reference, not a two-seed average.

## Tuning and isolation
After training, threshold selection evaluates only validation pools over all
three budgets and both generation seeds. The predefined grid is disabled,
0.5, 0.7, 0.9, 1.0. Maximize validation accuracy minus 0.02 times mean consumed
budget fraction. Ties prefer lower compute, then grid order. This is one global
threshold, not separate test-tuned thresholds.
The tuning file is frozen before test generation; tuning is refused once test
pools exist. Test labels are unavailable to the generator and selectors.
Test threshold sweeps are descriptive ablations, not used to replace the frozen
deployed threshold. No test-informed hyperparameter changes are allowed.

## Metrics and uncertainty
Candidate accuracy uses threshold 0.5. AUROC measures score ranking;
AUPRC is average precision (step-integral, not trapezoidal PR area).
Report prevalence, Brier mean squared probability error and equal-width,
10-bin ECE. Scores equal to one enter the final bin. Empty bins contribute zero.
AUROC/AUPRC are null if only one label class exists.
Candidate-level metrics use one primary 1024-token seed-17 pool per test query
to avoid duplication across budget/seed conditions.

Reasoning accuracy treats missing answers as incorrect. Report mean/median
generated tokens, mean consumed candidates and summed synchronized generation
latency. Bootstrap 2000 samples of **problems**, retaining paired seed outcomes
within a query. Intervals are percentile 95% intervals for this pilot population.
Seeds are not treated as independent queries. No statistical significance claim
is made; small test size creates large uncertainty.

Failure categories are collected automatically for primary pools; examples
are the first three stable IDs per category, including unfavorable cases.
All candidate traces for those examples are shown.

## Reproducibility and limits
Exact public checkpoint revisions, official dataset fingerprints, question IDs,
config, code commit, timestamp, seeds and hardware are persisted. Per-query seeds
depend on problem ID and candidate index, never processing order. Training uses
deterministic algorithms where PyTorch supports them (warn-only otherwise).
Cross-driver/architecture floating-point reproducibility is not guaranteed.
Adaptive experiment costs are prefix replay, excluding verifier overhead.
See [LIMITATIONS](LIMITATIONS.md) before interpreting results.

### Paired policy uncertainty
The predeclared verifier-minus-self-consistency and validation-tuned-adaptive-
minus-verifier comparisons pair identical pool IDs at each primary token cap.
We compute per-problem mean differences across generation seeds, then bootstrap
whole problems (2000 draws, seed2026). The resulting intervals quantify paired
accuracy differences; they do not select methods or thresholds on the test set.
No multiple-comparison significance claim is made.

### Exploratory shortcut diagnostic
Validation-only inspection after training found zero wrong numeric answers among
extracted validation answers; negatives were all missing final answers. Train
contained 335 missing answers and four extracted incorrect answers. Before
inspecting test metrics we added an answer-presence diagnostic: compare learned
classification metrics against probabilities 1 for present answers / 0 for absent
answers, and report learned metrics conditional on answer presence. It does not
change the frozen experiment, training, thresholds, policies or test decisions.
It is exploratory evidence about a confound, not a new tuned baseline claim.

Training uses seeded shuffling and initialization and deterministic-algorithm
warnings. The actual CUDA attention backward emitted a nondeterminism warning;
we do not claim bitwise reproducible retraining. Actual GPU greedy/stochastic
repeatability and live adaptive sampling were validated by the integration test.

### Final failure-analysis coverage
The initial report used only the single primary verifier-metric pool (seed17,
maximum cap), which missed downstream policy disagreements from the other seed.
Final artifact review expanded failure coverage to all primary-temperature
stochastic pools, both seeds and all declared budgets. Counts distinguish unique
problems from pool cases; each category shows the first three distinct problem
IDs ordered by problem, budget, seed and pool ID, with all their traces. This
changes only descriptive error coverage, not model training or selection/tuning.
