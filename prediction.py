rom __future__ import annotations

from pathlib import Path
from typing import Mapping
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

sns.set_theme(style="white", context="notebook")
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
})

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "dataset" / "retail_user_behavior_100k.csv").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent


def resolve_data_path() -> Path:
    """Resolve dataset path for both local and Kaggle notebook runtimes."""
    candidate_paths = [
        PROJECT_ROOT / "dataset" / "retail_user_behavior_100k.csv",
        Path("/kaggle/input/datasets/noopurbhatt/retail-intelligence-100k-user-behavior-dataset/retail_user_behavior_100k.csv"),
        Path("/kaggle/input/retail-intelligence-100k-user-behavior-dataset/retail_user_behavior_100k.csv"),
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return candidate_paths[0]


DATA_PATH = resolve_data_path()
RANDOM_STATE = 42
MAX_PREFIX_EVENTS = 3

PRE_TERMINAL_ACTIONS = ["view", "click", "wishlist", "add_to_cart"]
SURFACE = "#F7F3EA"
INK = "#17212B"
MUTED = "#6B7280"
GRID = "#D8DEE9"
GREEN = "#1B9E77"
RED = "#D1495B"
GOLD = "#F4A261"
BLUE = "#2A6F97"
def pct(value: float, digits: int = 1) -> str:
    """Format a proportion as a readable percentage."""
    return f"{100 * value:.{digits}f}%"


def load_events(path: Path) -> pd.DataFrame:
    """Load raw event rows and preserve session order."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    events = pd.read_csv(path)
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], utc=True)
    return events.sort_values(["session_id", "event_index"]).reset_index(drop=True)


def build_prefix_features(events: pd.DataFrame, max_prefix_events: int = 3) -> pd.DataFrame:
    """Create leakage-safe early-session features and a final conversion target."""
    ordered = events.sort_values(["session_id", "event_index"]).copy()
    target = (
        ordered.groupby("session_id", sort=False)
        .agg(
            converted=("is_conversion", "max"),
            terminal_action=("user_action", "last"),
            final_session_length=("event_index", "max"),
        )
        .reset_index()
    )

    ordered["max_observed_event"] = np.minimum(max_prefix_events, ordered["session_length"] - 1)
    prefix = ordered.loc[ordered["event_index"].le(ordered["max_observed_event"])].copy()

    session_features = (
        prefix.groupby("session_id", sort=False)
        .agg(
            started_at=("timestamp_utc", "min"),
            snapshot_at=("timestamp_utc", "max"),
            observed_events=("event_index", "max"),
            first_action=("user_action", "first"),
            last_action=("user_action", "last"),
            first_category=("category", "first"),
            last_category=("category", "last"),
            first_brand=("brand", "first"),
            last_brand=("brand", "last"),
            channel=("channel", "first"),
            device_type=("device_type", "first"),
            region=("region", "first"),
            traffic_source=("traffic_source", "first"),
            unique_products=("product_id", "nunique"),
            unique_categories=("category", "nunique"),
            unique_brands=("brand", "nunique"),
            total_time_spent_sec=("time_spent_sec", "sum"),
            mean_time_spent_sec=("time_spent_sec", "mean"),
            max_time_spent_sec=("time_spent_sec", "max"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            price_std=("price", "std"),
        )
        .reset_index()
    )

    session_features["elapsed_sec"] = (
        session_features["snapshot_at"] - session_features["started_at"]
    ).dt.total_seconds()
    session_features["start_hour"] = session_features["started_at"].dt.hour
    session_features["start_dow"] = session_features["started_at"].dt.dayofweek
    session_features["is_weekend"] = session_features["start_dow"].isin([5, 6]).astype(int)
    session_features["price_range"] = session_features["max_price"] - session_features["min_price"]
    session_features["time_per_observed_event"] = (
        session_features["total_time_spent_sec"] / session_features["observed_events"].clip(lower=1)
    )

    action_counts = pd.crosstab(prefix["session_id"], prefix["user_action"])
    action_counts = action_counts.reindex(columns=PRE_TERMINAL_ACTIONS, fill_value=0)
    action_counts.columns = [f"action_count_{action}" for action in action_counts.columns]

    action_rates = action_counts.div(action_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    action_rates.columns = [column.replace("action_count_", "action_rate_") for column in action_counts.columns]

    frame = (
        session_features
        .merge(action_counts.reset_index(), on="session_id", how="left")
        .merge(action_rates.reset_index(), on="session_id", how="left")
        .merge(target, on="session_id", how="left")
    )

    for action in ["click", "wishlist", "add_to_cart"]:
        frame[f"saw_{action}"] = frame[f"action_count_{action}"].gt(0).astype(int)

    frame["converted"] = frame["converted"].astype(int)
    return frame


events = load_events(DATA_PATH)
model_frame = build_prefix_features(events, max_prefix_events=MAX_PREFIX_EVENTS)

model_summary = pd.DataFrame(
    {
        "metric": [
            "Event rows",
            "Modeling sessions",
            "Prefix event cap",
            "Positive class rate",
            "Mean observed events",
            "Excluded terminal labels/actions",
        ],
        "value": [
            f"{len(events):,}",
            f"{len(model_frame):,}",
            MAX_PREFIX_EVENTS,
            pct(model_frame["converted"].mean(), 2),
            f"{model_frame['observed_events'].mean():.2f}",
            "purchase, drop, is_conversion, drop_off_flag",
        ],
    }
)

display(model_summary)
display(model_frame.head())
EXCLUDE_COLUMNS = [
    "session_id",
    "started_at",
    "snapshot_at",
    "converted",
    "terminal_action",
    "final_session_length",
]
FEATURE_COLUMNS = [column for column in model_frame.columns if column not in EXCLUDE_COLUMNS]


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train, validation, and test splits."""
    ordered = frame.sort_values("started_at").reset_index(drop=True)
    train_end = int(len(ordered) * 0.70)
    valid_end = int(len(ordered) * 0.85)
    return ordered.iloc[:train_end], ordered.iloc[train_end:valid_end], ordered.iloc[valid_end:]


def describe_split(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Summarize split sizes, time windows, and target rates."""
    rows = []
    for name, split in [("train", train), ("validation", valid), ("test", test)]:
        rows.append(
            {
                "split": name,
                "sessions": len(split),
                "start": split["started_at"].min().strftime("%Y-%m-%d"),
                "end": split["started_at"].max().strftime("%Y-%m-%d"),
                "conversion_rate": split["converted"].mean(),
            }
        )
    return pd.DataFrame(rows)


train_df, valid_df, test_df = time_split(model_frame)
X_train, y_train = train_df[FEATURE_COLUMNS], train_df["converted"]
X_valid, y_valid = valid_df[FEATURE_COLUMNS], valid_df["converted"]
X_test, y_test = test_df[FEATURE_COLUMNS], test_df["converted"]

display(describe_split(train_df, valid_df, test_df))
print(f"Features used: {len(FEATURE_COLUMNS)}")
print(FEATURE_COLUMNS)
def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature lists from the training frame."""
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return numeric, categorical


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing pipeline."""
    numeric_features, categorical_features = infer_feature_types(X)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def make_model_pipelines(X: pd.DataFrame) -> Mapping[str, Pipeline]:
    """Create comparable model pipelines sharing the same preprocessing contract."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X)),
                (
                    "model",
                    LogisticRegression(max_iter=2_000, class_weight="balanced", solver="lbfgs"),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=250,
                        min_samples_leaf=12,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=15,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


pipelines = make_model_pipelines(X_train)
list(pipelines)
def best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    """Select the probability threshold that maximizes validation F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_values = 2 * precision * recall / (precision + recall + 1e-9)
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index]), float(f1_values[best_index])


def evaluate_predictions(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate probability and threshold metrics."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "log_loss": log_loss(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "predicted_positive_rate": y_pred.mean(),
    }


def train_and_score_models(
    pipelines: Mapping[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    """Fit models, tune validation thresholds, and evaluate final test performance."""
    fitted: dict[str, Pipeline] = {}
    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    rows = []

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        valid_score = pipeline.predict_proba(X_valid)[:, 1]
        threshold, valid_best_f1 = best_f1_threshold(y_valid, valid_score)
        test_score = pipeline.predict_proba(X_test)[:, 1]

        valid_metrics = evaluate_predictions(y_valid, valid_score, threshold)
        test_metrics = evaluate_predictions(y_test, test_score, threshold)
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "valid_best_f1": valid_best_f1,
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
        fitted[model_name] = pipeline
        scores[model_name] = test_score
        thresholds[model_name] = threshold

    leaderboard = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False).reset_index(drop=True)
    return fitted, leaderboard, scores, thresholds


fitted_models, leaderboard, test_scores, thresholds = train_and_score_models(
    pipelines, X_train, y_train, X_valid, y_valid, X_test, y_test
)

best_model_name = leaderboard.iloc[0]["model"]
best_model = fitted_models[best_model_name]
best_test_score = test_scores[best_model_name]
best_threshold = thresholds[best_model_name]

display(
    leaderboard[[
        "model",
        "threshold",
        "valid_pr_auc",
        "valid_roc_auc",
        "valid_f1",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_brier",
    ]]
)
print(f"Selected model: {best_model_name} | validation threshold: {best_threshold:.3f}")
def plot_model_leaderboard(leaderboard: pd.DataFrame) -> plt.Figure:
    """Show validation/test ranking across PR-AUC and ROC-AUC."""
    plot_data = leaderboard.sort_values("valid_pr_auc").reset_index(drop=True)
    y = np.arange(len(plot_data))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    metrics = [
        ("pr_auc", "PR-AUC", GREEN),
        ("roc_auc", "ROC-AUC", BLUE),
    ]

    for ax, (metric, label, color) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        ax.hlines(y, plot_data[f"valid_{metric}"], plot_data[f"test_{metric}"], color="#B6C2CF", linewidth=3, alpha=0.8)
        ax.scatter(plot_data[f"valid_{metric}"], y, s=150, color=color, edgecolor="white", linewidth=1.2, label="Validation")
        ax.scatter(plot_data[f"test_{metric}"], y, s=150, color=GOLD, edgecolor="white", linewidth=1.2, label="Test")
        for row_id, row in plot_data.iterrows():
            ax.text(row[f"test_{metric}"] + 0.004, row_id, f"{row[f'test_{metric}']:.3f}", va="center", fontsize=9.5)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel(label)
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_data["model"])
        ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Model leaderboard: validation selection versus test confirmation", x=0.02, y=1.04, ha="left", fontsize=17, weight="bold")
    plt.tight_layout()
    return fig

plot_model_leaderboard(leaderboard)
plt.show()
def calibration_table(y_true: pd.Series, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin probabilities and compare predicted versus observed conversion."""
    table = pd.DataFrame({"y": y_true.to_numpy(), "score": y_score})
    table["bin"] = pd.qcut(table["score"], q=n_bins, duplicates="drop")
    return (
        table.groupby("bin", observed=True)
        .agg(mean_score=("score", "mean"), observed_rate=("y", "mean"), sessions=("y", "size"))
        .reset_index(drop=True)
    )


def lift_table(y_true: pd.Series, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Create decile lift and cumulative conversion capture."""
    table = pd.DataFrame({"y": y_true.to_numpy(), "score": y_score}).sort_values("score", ascending=False)
    table["decile"] = pd.qcut(np.arange(len(table)), q=n_bins, labels=np.arange(1, n_bins + 1))
    grouped = (
        table.groupby("decile", observed=True)
        .agg(sessions=("y", "size"), conversions=("y", "sum"), mean_score=("score", "mean"))
        .reset_index()
    )
    grouped["conversion_rate"] = grouped["conversions"] / grouped["sessions"]
    grouped["lift"] = grouped["conversion_rate"] / table["y"].mean()
    grouped["cumulative_conversion_capture"] = grouped["conversions"].cumsum() / grouped["conversions"].sum()
    return grouped


def threshold_table(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    """Calculate precision, recall, F1, and predicted volume across thresholds."""
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "predicted_positive_rate": y_pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_model_diagnostics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    model_name: str,
) -> plt.Figure:
    """Render PR, calibration, lift, and threshold diagnostics."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    cal = calibration_table(y_true, y_score)
    lift = lift_table(y_true, y_score)
    thresh = threshold_table(y_true, y_score)
    baseline = y_true.mean()

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.75, alpha=0.65)

    axes[0, 0].plot(recall, precision, color=GREEN, linewidth=2.4)
    axes[0, 0].axhline(baseline, color=RED, linestyle="--", linewidth=1.2, label=f"Baseline {pct(baseline, 1)}")
    axes[0, 0].set_title("Precision-recall curve")
    axes[0, 0].set_xlabel("Recall")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot([0, 1], [0, 1], color="#8A94A6", linestyle="--", linewidth=1.1)
    axes[0, 1].scatter(cal["mean_score"], cal["observed_rate"], s=cal["sessions"] / 5, color=BLUE, alpha=0.8, edgecolor="white", linewidth=1.1)
    axes[0, 1].plot(cal["mean_score"], cal["observed_rate"], color=BLUE, linewidth=1.7, alpha=0.8)
    axes[0, 1].set_title("Calibration by score decile")
    axes[0, 1].set_xlabel("Mean predicted probability")
    axes[0, 1].set_ylabel("Observed conversion rate")

    axes[1, 0].bar(lift["decile"].astype(str), lift["lift"], color=GREEN, alpha=0.8, edgecolor="white")
    axes[1, 0].axhline(1, color=RED, linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("Lift by score decile")
    axes[1, 0].set_xlabel("Score decile, highest first")
    axes[1, 0].set_ylabel("Lift versus baseline")

    axes[1, 1].plot(thresh["threshold"], thresh["precision"], color=GREEN, linewidth=2, label="Precision")
    axes[1, 1].plot(thresh["threshold"], thresh["recall"], color=RED, linewidth=2, label="Recall")
    axes[1, 1].plot(thresh["threshold"], thresh["f1"], color=BLUE, linewidth=2, label="F1")
    axes[1, 1].axvline(threshold, color=INK, linestyle="--", linewidth=1.2, label=f"Chosen {threshold:.2f}")
    axes[1, 1].set_title("Threshold trade-off")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Metric value")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(f"Diagnostics for selected model: {model_name}", x=0.02, y=1.02, ha="left", fontsize=18, weight="bold")
    plt.tight_layout()
    return fig

plot_model_diagnostics(y_test, best_test_score, best_threshold, best_model_name)
plt.show()
def plot_confusion_matrix(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> plt.Figure:
    """Plot an annotated confusion matrix for a selected threshold."""
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = np.array([
        ["True drop-risk", "False converter flag"],
        ["Missed converter", "Captured converter"],
    ])
    annotations = np.empty_like(labels, dtype=object)
    for row in range(2):
        for col in range(2):
            annotations[row, col] = f"{labels[row, col]}\n{matrix[row, col]:,}"

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("cm", ["#FFF7EC", "#F4A261", "#264653"])
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap=cmap,
        linewidths=1.4,
        linecolor=SURFACE,
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": 11, "weight": "bold"},
    )
    ax.set_xticklabels(["Predicted non-convert", "Predicted convert"], rotation=0)
    ax.set_yticklabels(["Actual non-convert", "Actual convert"], rotation=0)
    ax.set_title(f"Confusion matrix at threshold {threshold:.3f}", fontsize=15, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig

plot_confusion_matrix(y_test, best_test_score, best_threshold)
plt.show()
def clean_feature_name(name: str) -> str:
    """Make transformed feature names easier to read."""
    cleaned = name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("action_count_", "count: ").replace("action_rate_", "rate: ")
    return cleaned.replace("_", " ")


def extract_feature_effects(pipeline: Pipeline) -> pd.DataFrame:
    """Extract coefficients or impurity importances from the fitted pipeline."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values = model.coef_[0]
        effect_type = "coefficient"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        effect_type = "importance"
    else:
        raise TypeError("Selected model does not expose coefficients or feature importances.")

    effects = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in names],
            "value": values,
            "abs_value": np.abs(values),
            "effect_type": effect_type,
        }
    )
    return effects.sort_values("abs_value", ascending=False).reset_index(drop=True)


def plot_feature_effects(effects: pd.DataFrame, top_n: int = 14) -> plt.Figure:
    """Plot top positive and negative model effects when available."""
    effect_type = effects["effect_type"].iloc[0]
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if effect_type == "coefficient":
        selected = pd.concat([
            effects.nlargest(top_n, "value"),
            effects.nsmallest(top_n, "value"),
        ]).sort_values("value")
        colors = np.where(selected["value"].ge(0), GREEN, RED)
        ax.hlines(selected["feature"], 0, selected["value"], color=colors, linewidth=4, alpha=0.65)
        ax.scatter(selected["value"], selected["feature"], s=95, color=colors, edgecolor="white", linewidth=1.1)
        ax.axvline(0, color=INK, linewidth=1.1)
        ax.set_xlabel("Standardized coefficient")
        ax.set_title("Top positive and negative conversion drivers")
    else:
        selected = effects.head(top_n).sort_values("value")
        ax.hlines(selected["feature"], 0, selected["value"], color=BLUE, linewidth=4, alpha=0.65)
        ax.scatter(selected["value"], selected["feature"], s=95, color=BLUE, edgecolor="white", linewidth=1.1)
        ax.set_xlabel("Feature importance")
        ax.set_title("Top feature importances")

    ax.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.75)
    plt.tight_layout()
    return fig

feature_effects = extract_feature_effects(best_model)
display(feature_effects.head(20))
plot_feature_effects(feature_effects)
plt.show()
