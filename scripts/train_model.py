from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
 
from src.data_validation.clean import clean_data
from src.data_validation.schema import build_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline fraud model.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--out_dir", default="runs", help="Output directory")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500000,
        help="Stratified sample size (0 for full dataset)",
    )
    parser.add_argument(
        "--min_per_class",
        type=int,
        default=1000,
        help="Minimum samples per class for stratified sampling",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf"],
        default="logreg",
        help="Model type",
    )
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


def _stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    sample_size: int,
    min_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df

    counts = df[target_col].value_counts()
    proportions = counts / counts.sum()
    per_class = (proportions * sample_size).astype(int)
    per_class = per_class.clip(lower=min_per_class)
    per_class = per_class.clip(upper=counts)

    total = int(per_class.sum())
    if total > sample_size:
        overflow = total - sample_size
        if 0 in per_class.index:
            per_class.loc[0] = max(min_per_class, int(per_class.loc[0] - overflow))
        else:
            per_class.iloc[0] = max(min_per_class, int(per_class.iloc[0] - overflow))

    sampled = []
    for label, n in per_class.items():
        sampled.append(
            df[df[target_col] == label].sample(n=int(n), random_state=random_state)
        )
    return pd.concat(sampled, ignore_index=True)


def _add_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["nameOrig", "nameDest"]:
        prefix_col = f"{col}_prefix"
        df[prefix_col] = df[col].astype("string").str.slice(0, 1)
        df[f"{col}_len"] = df[col].astype("string").str.len()
    return df


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    _assert_input_files(args.train, args.test)

    logging.info("Loading data")
    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)

    schema = build_schema(train_df, test_df, mode="warn")
    clean_train, clean_test, validation_features = clean_data(train_df, test_df, schema)

    train_features = validation_features[validation_features["split"] == "train"].drop(
        columns=["split"]
    )
    test_features = validation_features[validation_features["split"] == "test"].drop(
        columns=["split"]
    )

    clean_train = _add_prefix_features(clean_train)
    clean_test = _add_prefix_features(clean_test)

    train_full = clean_train.merge(train_features, on=schema.id_col, how="left")
    test_full = clean_test.merge(test_features, on=schema.id_col, how="left")

    if args.sample_size > 0:
        train_full = _stratified_sample(
            train_full,
            schema.target_col,
            args.sample_size,
            args.min_per_class,
            args.random_state,
        )
        logging.info("Using stratified sample: %d rows", len(train_full))

    target = train_full[schema.target_col]
    feature_df = train_full.drop(columns=[schema.target_col])

    numeric_cols = [
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
    flag_cols = [
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
    categorical_cols = ["type", "nameOrig_prefix", "nameDest_prefix"]

    for col in numeric_cols + flag_cols + categorical_cols:
        if col not in feature_df.columns:
            feature_df[col] = np.nan
        if col not in test_full.columns:
            test_full[col] = np.nan

    X_train, X_val, y_train, y_val = train_test_split(
        feature_df[numeric_cols + flag_cols + categorical_cols],
        target,
        test_size=0.2,
        stratify=target,
        random_state=args.random_state,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    if args.model == "logreg":
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols + flag_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )
        clf = LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            class_weight="balanced",
            solver="saga",
        )
        model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        logging.info("Training model (logreg)")
        model.fit(X_train, y_train)
    else:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols + flag_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=50,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=args.random_state,
        )
        model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        logging.info("Training model (rf)")
        model.fit(X_train, y_train)

    logging.info("Evaluating")
    val_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, val_pred, average="macro")
    logging.info("Validation macro F1: %.6f", macro_f1)

    logging.info("Predicting test set")
    test_pred = model.predict(test_full[numeric_cols + flag_cols + categorical_cols])
    submission = pd.DataFrame(
        {schema.id_col: test_full[schema.id_col], schema.target_col: test_pred}
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    dump(model, out_dir / f"{args.model}_model.joblib")

    logging.info("Wrote submission to %s", submission_path)
    logging.info("Model used is %s", args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
