# Retail Intelligence

Predict whether a retail browsing session will end in a purchase using only **early, pre-terminal user behavior**.

This project builds a machine learning pipeline that uses the first few events in a user session to estimate the likelihood of conversion before the session ends. The goal is to support early intervention strategies such as personalization, promotions, retargeting, or sales optimization.

## Project Objective

Retail platforms often want to know whether a user is likely to purchase **before** the session is over. This repository focuses on:

- modeling user conversion from partial session behavior
- avoiding target leakage by using only early-session events
- comparing multiple classification models
- evaluating ranking quality, calibration, threshold trade-offs, and lift

## Problem Statement

Given user session events such as page views, clicks, wishlist actions, and add-to-cart behavior, predict whether the session will ultimately convert.

Instead of using the full session, this project limits features to the **early prefix** of each session so the prediction is useful in a real-world setting.

## Current Approach

The pipeline in `prediction.py`:

1. loads event-level retail session data
2. sorts events by session and event order
3. constructs leakage-safe prefix features from the first few session events
4. derives session-level behavioral, categorical, price, and timing features
5. performs a chronological train/validation/test split
6. trains multiple models:
   - Logistic Regression
   - Extra Trees Classifier
   - Random Forest Classifier
7. tunes a classification threshold on validation data
8. evaluates final performance on the test set
9. generates diagnostics such as:
   - PR-AUC
   - ROC-AUC
   - F1
   - precision / recall
   - calibration table
   - lift table
   - confusion matrix
   - feature effects / importance

## Repository Structure

```text
.
├── README.md
└── prediction.py
```

> Note: the repository is currently in an early stage and is centered around a single modeling script.

## Dataset

The script expects a CSV dataset named:

```text
retail_user_behavior_100k.csv
```

Expected local path:

```text
dataset/retail_user_behavior_100k.csv
```

The script also contains fallback paths for Kaggle-style environments.

### Expected data characteristics

Based on the code, the dataset should include fields like:

- `session_id`
- `timestamp_utc`
- `event_index`
- `session_length`
- `user_action`
- `category`
- `brand`
- `channel`
- `device_type`
- `region`
- `traffic_source`
- `product_id`
- `time_spent_sec`
- `price`
- `is_conversion`

## Installation

Create and activate a virtual environment, then install the required packages.

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn ipython
```

### Option 2: Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn ipython
```

## How to Run

1. Place the dataset at:

```text
dataset/retail_user_behavior_100k.csv
```

2. Run the script:

```bash
python prediction.py
```

## What the Script Does

The script:
- loads and validates the dataset path
- builds early-session features
- creates time-based train/validation/test splits
- trains multiple classifiers
- evaluates model performance
- selects the best model based on validation performance
- generates diagnostic plots and feature effect summaries

## Feature Engineering Highlights

Examples of engineered features include:

- first and last observed action
- first and last category / brand
- number of unique products, brands, and categories
- total / mean / max time spent
- median / mean / min / max / std of price
- elapsed session time
- session start hour and day of week
- weekend indicator
- counts and rates of:
  - views
  - clicks
  - wishlist actions
  - add-to-cart actions
- binary indicators such as:
  - saw click
  - saw wishlist
  - saw add_to_cart

## Modeling Notes

This project tries to stay realistic by:
- using only the first few session events
- excluding terminal outcome leakage
- splitting data chronologically instead of randomly
- tuning thresholds on validation data before testing

## Evaluation Outputs

The workflow includes:
- model leaderboard comparison
- precision-recall visualization
- calibration analysis
- lift analysis by score decile
- threshold trade-off analysis
- confusion matrix visualization
- feature effect / importance plots

## Potential Use Cases

- early conversion scoring
- personalized recommendations
- promotion triggering
- abandoned-session intervention
- traffic quality analysis
- merchandising optimization

## Limitations

Current limitations of the repository include:
- single-script structure
- no dependency file yet
- no automated tests
- no CI workflow
- minimal packaging / modularization
- dataset is not bundled in the repo

## Example Research Questions

This project can help answer questions like:
- How predictive are the first 3 events in a session?
- Which early behaviors are most associated with purchase?
- How much lift can we get by targeting top-scored sessions?
- Which model provides the best balance of recall and precision?

## Author

Created by [@ajibadeboluwatife](https://github.com/ajibadeboluwatife)
