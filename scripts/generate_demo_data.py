from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo fraud train/test CSV files.")
    parser.add_argument("--out_dir", default="fraud", help="Output directory for CSV files")
    parser.add_argument("--n_train", type=int, default=50000, help="Number of train rows")
    parser.add_argument("--n_test", type=int, default=20000, help="Number of test rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _sample_names(prefix: str, n: int, rng: np.random.Generator) -> pd.Series:
    numbers = rng.integers(10_000_000, 99_999_999, size=n)
    return pd.Series([f"{prefix}{x}" for x in numbers], dtype="string")


def _make_frame(n_rows: int, start_id: int, rng: np.random.Generator) -> pd.DataFrame:
    step = rng.integers(1, 745, size=n_rows)
    amount = np.round(rng.lognormal(mean=8.0, sigma=1.0, size=n_rows), 2)
    tx_type = rng.choice(TX_TYPES, size=n_rows, p=[0.18, 0.22, 0.07, 0.28, 0.25])

    oldbalance_org = np.round(rng.lognormal(mean=9.0, sigma=1.1, size=n_rows), 2)
    debit_like = np.isin(tx_type, ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    credit_like = tx_type == "CASH_IN"

    newbalance_org = oldbalance_org.copy()
    newbalance_org[debit_like] = np.maximum(0.0, oldbalance_org[debit_like] - amount[debit_like])
    newbalance_org[credit_like] = oldbalance_org[credit_like] + amount[credit_like]

    oldbalance_dest = np.round(rng.lognormal(mean=9.5, sigma=1.2, size=n_rows), 2)
    newbalance_dest = oldbalance_dest + amount

    is_merchant = rng.random(n_rows) < 0.2
    name_orig = _sample_names("C", n_rows, rng)
    name_dest = _sample_names("C", n_rows, rng)
    name_dest[is_merchant] = _sample_names("M", int(is_merchant.sum()), rng).values

    df = pd.DataFrame(
        {
            "id": np.arange(start_id, start_id + n_rows, dtype=int),
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": np.round(newbalance_org, 2),
            "nameDest": name_dest,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": np.round(newbalance_dest, 2),
        }
    )
    return df


def _assign_urgency(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    risk_score = (
        (df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int) * 2.0)
        + (df["amount"] > df["amount"].quantile(0.95)).astype(int) * 2.0
        + (df["nameDest"].astype("string").str.startswith("M").astype(int) * 1.0)
        + rng.normal(0.0, 0.6, len(df))
    )

    bins = [-10, 1.0, 2.5, 4.0, 10]
    urgency = pd.cut(risk_score, bins=bins, labels=[0, 1, 2, 3]).astype(int)

    # Force stronger imbalance similar to real fraud settings.
    demote_mask = (urgency > 0) & (rng.random(len(df)) < 0.55)
    urgency.loc[demote_mask] = urgency.loc[demote_mask] - 1
    return urgency.clip(lower=0, upper=3)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train = _make_frame(args.n_train, start_id=1, rng=rng)
    test = _make_frame(args.n_test, start_id=args.n_train + 1, rng=rng)

    train["urgency_level"] = _assign_urgency(train, rng)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Wrote {train_path} ({len(train)} rows)")
    print(f"Wrote {test_path} ({len(test)} rows)")
    print("Class distribution:")
    print(train["urgency_level"].value_counts(normalize=True).sort_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
