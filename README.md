# HackML: Fraud Detection with Data Validation

A machine learning project for detecting fraudulent transactions using advanced data validation and model training pipelines.

## Project Overview

This project implements a comprehensive fraud detection system that includes:

- **Data Validation Layer**: Automated validation, cleaning, and quality checks for financial transaction data
- **Model Training**: Baseline machine learning models for fraud classification
- **Reporting**: Detailed validation reports and model performance metrics
- **Testing**: Comprehensive test suite for data validation components

## Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dshak1/hackML.git
cd hackML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Place your fraud detection datasets in the `fraud/` directory:
- `fraud/train.csv` - Training data with target column
- `fraud/test.csv` - Test data without target column

If you do not have data yet, generate a realistic demo dataset:

```bash
python scripts/generate_demo_data.py --out_dir fraud --n_train 50000 --n_test 20000
```

Then run training/benchmarking with:

```bash
python scripts/train_model.py --train fraud/train.csv --test fraud/test.csv --out_dir runs
```

> Tip (zsh/bash): line continuations require the backslash to be the final character on each line (no trailing spaces).

### 2. Validate Data

Run data validation to check data quality and generate reports:

```bash
python scripts/validate_data.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs \
  --mode warn
```

### 3. Train Model

Train a baseline fraud detection model:

```bash
python scripts/train_model.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs
```

### 4. Benchmark Multiple Models (New)

Run repeated cross-validated benchmarking across baseline and advanced models:

```bash
python scripts/benchmark_models.py \
  --train fraud/train.csv \
  --test fraud/test.csv \
  --out_dir runs/benchmark \
  --n_splits 5 \
  --n_repeats 2
```

This writes `benchmark_summary.json` and `benchmark_summary.md` with Macro-F1 and runtime comparisons.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the full test suite
6. Submit a pull request
