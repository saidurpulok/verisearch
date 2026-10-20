# Limitations

- This is a small GSM8K pilot, not an evaluation of the full benchmark, and not
  evidence of algorithmic novelty or state-of-the-art performance.
- A 1.5B generator and 256-token trace limit restrict reasoning. Truncated or
  unmarked outputs count as failures even if intermediate arithmetic is correct.
- Exact numeric-answer labels do not verify individual reasoning steps. A lucky
  answer can be labeled correct despite an invalid derivation.
- Public pretrained generator/encoder training data are outside our control;
  the project enforces local split isolation, not absence of pretraining contamination.
- The verifier is trained on this generator's traces; transfer to different models,
  prompts, temperatures, tasks or long traces can degrade.
- A tiny verifier dataset and imbalance can encourage majority predictions and
  stylistic shortcuts. The observed fit data has 335 missing-answer negatives and
  only four numerically wrong extracted answers; all 75 validation negatives are
  missing answers. This makes answer presence a near-perfect shortcut. A high
  probability is not evidence of mathematical verification or a correctness guarantee.
- Training enabled deterministic algorithms in warning-only mode; CUDA memory-
  efficient attention backward reported nondeterminism. Seeds and configurations
  are reproducible, but bitwise-identical verifier retraining is not established.
- Encoder truncation may discard essential reasoning. Rates are recorded.
- Generated tokens are an imperfect compute proxy; prompt prefill, encoder
  scoring, model loading and unequal per-token cost are excluded. This is not
  fixed total FLOPs or equal wall-clock cost.
- Latencies are synchronized single-device batch-one generation measurements on
  a laptop, affected by temperature, background load and warm-up. Adaptive costs
  are prefix replay, not a measured end-to-end online speedup.
- Two sampling seeds and one verifier training seed give limited uncertainty
  coverage. Bootstrap intervals reflect problem sampling, not all model randomness.
- Small data-size and temperature ablations are descriptive, not causal evidence
  of monotonic scaling or optimal decoding settings. The fixed-epoch data-size
  ablation also changes the number of optimizer updates.
- The system uses conservative parsing; ambiguous unmarked final numbers are
  intentionally rejected. It is not a symbolic mathematics evaluator.
- Target hardware limits data volume and experiment breadth. The pilot
  evaluates only GSM8K.
- v0.1 implements no RL, tree search, process reward model, tool execution or
  proof checking.
