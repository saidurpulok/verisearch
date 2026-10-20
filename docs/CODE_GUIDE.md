# Code guide

This guide maps the research mechanisms to their implementation. For experimental
protocol and measured outcomes, see [Methodology](METHODOLOGY.md) and
[RESULTS.md](../RESULTS.md).

## Component map

| Mechanism | Source | Implementation detail |
|---|---|---|
| Problem boundaries | [schema.py](../src/verisearch/data/schema.py), [split.py](../src/verisearch/data/split.py) | Gold-bearing records are separate from policy inputs; normalized question identities define disjoint partitions. |
| Autoregressive generation | [generator.py](../src/verisearch/generation/generator.py) | Transformers emits reasoning tokens; accounting counts newly generated IDs, including EOS. |
| Stochastic decoding | [config.py](../src/verisearch/config.py), [generator.py](../src/verisearch/generation/generator.py) | Temperature, top-p and top-k control sampling; per-problem/candidate seeds are independent of traversal order. |
| Answer extraction | [answer_extraction.py](../src/verisearch/generation/answer_extraction.py) | Explicit final-answer markers are parsed conservatively and normalized for exact numeric comparison. |
| Self-consistency | [selection.py](../src/verisearch/policies/selection.py) | Valid extracted answers vote; ties use the earliest candidate with a tied answer. |
| Verifier input | [dataset.py](../src/verisearch/verifier/dataset.py) | Question, candidate answer and reasoning are serialized without gold; the answer precedes reasoning to survive truncation. |
| Verifier training | [train.py](../src/verisearch/verifier/train.py) | Full encoder fine-tuning uses two-class cross-entropy; validation loss selects the checkpoint. |
| Probability scoring | [model.py](../src/verisearch/verifier/model.py) | Softmax converts classifier logits to an estimated probability of final-answer correctness. |
| Best-of-N and adaptive stopping | [selection.py](../src/verisearch/policies/selection.py) | Highest-score selection shares a lazy candidate interface with threshold-based early stopping. |
| Fixed budgets | [generator.py](../src/verisearch/generation/generator.py), [experiment.py](../src/verisearch/experiment.py) | Each generation call is capped by remaining tokens; comparable policies reuse the same pool. |
| Calibration and ranking | [metrics.py](../src/verisearch/evaluation/metrics.py) | Brier score, ECE, AUROC and average precision distinguish probability quality from ranking quality. |
| Uncertainty | [metrics.py](../src/verisearch/evaluation/metrics.py) | Bootstrap utilities resample problems while preserving paired seed outcomes. |
| Failure analysis | [error_analysis.py](../src/verisearch/evaluation/error_analysis.py) | Deterministic categories expose both favorable and unfavorable selection cases. |
| Experiment orchestration | [experiment.py](../src/verisearch/experiment.py), [cli.py](../src/verisearch/cli.py) | Saved configuration, immutable manifests and frozen validation tuning gate final evaluation. |

## One problem through the pipeline

1. A `Query` supplies the question and problem identity to the generator. Its
   gold answer is stored separately in a `Problem`.
2. Generation produces `Candidate` records with reasoning, extracted answers,
   seeds and token counts. No gold answer is available to generation.
3. For verifier training, a `LabeledCandidate` adds an exact-answer correctness
   label. Training problems and validation problems remain disjoint.
4. The verifier receives the question, candidate answer and reasoning. At
   inference it returns a score without access to the gold answer or label.
5. Voting, reranking and adaptive stopping select from gold-free candidates.
   Adaptive stopping uses the threshold frozen on validation data.
6. Final evaluation attaches correctness labels to selected test candidates.
   The oracle uses labels solely to compute a nondeployable upper bound.

Using a test gold answer in a prompt, scorer input, vote, stopping decision or
threshold search would violate these boundaries. Representative recorded traces
and scores are available in [the failure report](../results/pilot/errors.md).

## Interpreting the verifier

An outcome label establishes final-answer agreement, not reasoning validity.
The pilot's missing-answer negatives allow a formatting shortcut: overall
classification metrics are strong, while ranking among extracted answers is
weak. See the [shortcut diagnostic](../RESULTS.md#exploratory-answer-presence-diagnostic)
and [limitations](LIMITATIONS.md) before interpreting scores as mathematical
verification. v0.1 does not implement process rewards, reinforcement learning or
tree search.
