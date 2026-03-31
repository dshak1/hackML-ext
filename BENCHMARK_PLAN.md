# Benchmark Plan

## Objective
Establish a rigorous, reproducible benchmark suite for imbalanced fraud classification with model-quality and system-performance metrics.

## Candidate Models
1. Logistic Regression (strong linear baseline)
2. Random Forest (current project family)
3. HistGradientBoostingClassifier (strong nonlinear baseline)

## Evaluation Protocol
- Repeated stratified CV: 5 folds × 2 repeats (default).
- Primary metric: Macro-F1.
- Secondary metrics: mean fit time, mean predict time.
- Class imbalance handling: `class_weight='balanced'` (where supported).

## Data Pipeline
1. Run schema inference and validation feature generation.
2. Apply deterministic cleaning and feature engineering.
3. Build training matrix from numeric, flag, and categorical features.
4. Optional stratified sample for faster iterations.

## Artifacts
- `benchmark_summary.json`: raw and aggregated fold metrics.
- `benchmark_summary.md`: ranked leaderboard for quick review.

## Suggested Commands
```bash
python scripts/benchmark_models.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs/benchmark \
  --n_splits 5 \
  --n_repeats 2 \
  --sample_size 250000
```

## Risk Controls
- If runtime is high, reduce `sample_size` and keep CV constant.
- If instability appears, increase repeats or run multiple seeds.
- If overfitting appears, add regularization sweeps and calibration checks.
