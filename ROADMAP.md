# NVIDIA-Oriented Extension Roadmap

## Project Direction
Build a **systems-aware tabular ML benchmark pipeline** that predicts fraud risk while explicitly tracking performance and reproducibility trade-offs in a way aligned with NVIDIA-style engineering expectations.

## Milestones

### Milestone 1 — Benchmarking Foundation (Week 1)
- Add repeatable model benchmarking CLI (`scripts/benchmark_models.py`).
- Add benchmark summarization utilities (`src/benchmarking/metrics.py`).
- Evaluate RF against stronger baselines (logistic regression and histogram gradient boosting).
- Produce machine-readable (`json`) and human-readable (`md`) reports.

### Milestone 2 — Experiment Governance (Week 2)
- Add experiment registry format under `runs/benchmark/`.
- Add deterministic seed management and report metadata.
- Integrate benchmark command in CI smoke checks.

### Milestone 3 — Systems Signal Enrichment (Week 3)
- Introduce latency- and throughput-oriented reporting in benchmark artifacts.
- Add profiling hooks for fit/predict stages.
- Add error-bucket analysis by transaction type/amount quantiles.

### Milestone 4 — C++/Performance Track (Week 4+)
- Add C++ feature engineering prototype for high-throughput preprocessing.
- Build Python bindings and compare throughput against pandas pipeline.
- Publish performance report and integration guidance.

## Definition of Success
- A benchmark run produces comparable macro-F1 and runtime metrics for at least 3 models.
- Results are reproducible with fixed seeds.
- Artifacts are shareable in interviews as engineering evidence (metrics + code + tests).
