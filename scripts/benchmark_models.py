from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking.metrics import BenchmarkResult, summarize_results
from src.data_validation.clean import clean_data
from src.data_validation.schema import build_schema


NUMERIC_COLS = [
    "step",
    "amount",
    "log1p_amount",
    "orig_delta",
    "dest_delta",
    "balance_error_orig",
    "balance_error_dest",
    "nameOrig_len",
    "nameDest_len",
]

FLAG_COLS = [
    "balance_error_orig_high",
    "balance_error_dest_high",
    "is_merchant_dest",
    "amount_is_round_number",
    "amount_spike_flag",
    "amount_outlier_flag",
    "amount_negative_flag",
    "orig_balance_zero_before",
    "dest_balance_zero_before",
    "orig_balance_zero_after",
    "dest_balance_zero_after",
]

CAT_COLS = ["type", "nameOrig_prefix", "nameDest_prefix"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fraud models with repeated CV.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv for schema/feature parity")
    parser.add_argument("--out_dir", default="runs/benchmark", help="Output directory")
    parser.add_argument("--n_splits", type=int, default=5, help="CV folds")
    parser.add_argument("--n_repeats", type=int, default=2, help="CV repeats")
    parser.add_argument("--sample_size", type=int, default=250000, help="Training sample size (0 for full)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _assert_input_files(train_path: str, test_path: str) -> None:
    missing = [path for path in [train_path, test_path] if not Path(path).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            "Missing dataset file(s): "
            f"{missing_str}. "
            "Generate demo data with `python scripts/generate_demo_data.py` "
            "or provide valid --train/--test paths."
        )


def _add_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    copy_df = df.copy()
    for col in ["nameOrig", "nameDest"]:
        copy_df[f"{col}_prefix"] = copy_df[col].astype("string").str.slice(0, 1)
        copy_df[f"{col}_len"] = copy_df[col].astype("string").str.len()
    return copy_df


def _build_feature_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    schema = build_schema(train_df, test_df, mode="warn")
    clean_train, clean_test, validation_features = clean_data(train_df, test_df, schema)

    train_features = validation_features[validation_features["split"] == "train"].drop(columns=["split"])

    clean_train = _add_prefix_features(clean_train)
    clean_test = _add_prefix_features(clean_test)

    train_full = clean_train.merge(train_features, on=schema.id_col, how="left")
    _ = clean_test  # ensure cleaning parity has executed

    target = train_full[schema.target_col]
    feature_df = train_full.drop(columns=[schema.target_col])

    for col in NUMERIC_COLS + FLAG_COLS + CAT_COLS:
        if col not in feature_df.columns:
            feature_df[col] = np.nan

    feature_df = feature_df[NUMERIC_COLS + FLAG_COLS + CAT_COLS]
    return feature_df, target


def _sample_if_needed(X: pd.DataFrame, y: pd.Series, sample_size: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size <= 0 or sample_size >= len(X):
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=random_state,
        stratify=y,
    )
    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def _model_registry(seed: int) -> Dict[str, Pipeline]:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLS + FLAG_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )

    return {
        "logreg": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=600,
                        class_weight="balanced",
                        solver="saga",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=14,
                        min_samples_leaf=50,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=250,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Pipeline],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[BenchmarkResult]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    output: List[BenchmarkResult] = []

    for name, model in models.items():
        scores: List[float] = []
        fit_times: List[float] = []
        pred_times: List[float] = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            candidate = clone(model)
            fit_start = time.perf_counter()
            candidate.fit(X_train, y_train)
            fit_times.append(time.perf_counter() - fit_start)

            pred_start = time.perf_counter()
            y_pred = candidate.predict(X_val)
            pred_times.append(time.perf_counter() - pred_start)

            scores.append(f1_score(y_val, y_pred, average="macro"))

        output.append(
            BenchmarkResult(
                model_name=name,
                macro_f1_scores=scores,
                fit_times_sec=fit_times,
                predict_times_sec=pred_times,
            )
        )

    return output


def _write_outputs(results: List[BenchmarkResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)

    with (out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "raw": [r.to_dict() for r in results]}, handle, indent=2)

    lines = [
        "# Benchmark Results",
        "",
        "| Rank | Model | Macro-F1 (mean) | Macro-F1 (std) | Fit Time (s) | Predict Time (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary, start=1):
        lines.append(
            f"| {rank} | {row['model_name']} | {row['macro_f1_mean']:.5f} | {row['macro_f1_std']:.5f} | {row['fit_time_mean_sec']:.3f} | {row['predict_time_mean_sec']:.4f} |"
        )

    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    _assert_input_files(args.train, args.test)

    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)
    X, y = _build_feature_frame(train_df, test_df)
    X, y = _sample_if_needed(X, y, args.sample_size, args.random_state)

    logging.info("Benchmark dataset shape: %s", X.shape)
    results = _run_cv(
        X=X,
        y=y,
        models=_model_registry(args.random_state),
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.random_state,
    )

    _write_outputs(results, Path(args.out_dir))
    logging.info("Wrote benchmark artifacts to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
