#!/usr/bin/env python3
# =============================================================================
# STANDALONE POST-RUN PERFORMANCE METRICS
# NHANES 2017-2018 Mobility Disability Prediction
#
# This script DOES NOT retrain or reload the XGBoost model.
#
# Required files:
#   tables/test_predictions.csv
#   tables/cohort_summary.csv
#
# Exact prediction columns created by the original pipeline:
#   y_true_mobility_disability
#   predicted_probability
#   prediction_0p5
#   prediction_youden
#
# Outputs:
#   tables/postrun_performance_metrics.csv
#   tables/postrun_performance_metrics_95ci.csv
#   tables/postrun_performance_publication.csv
#   tables/postrun_confusion_matrices.csv
#   tables/postrun_threshold_summary.csv
#
#   figures/postrun_roc_curve.png
#   figures/postrun_precision_recall_curve.png
#   figures/postrun_calibration_curve.png
#   figures/postrun_confusion_matrix_0p5.png
#   figures/postrun_confusion_matrix_youden.png
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# =============================================================================
# 1. USER SETTINGS
# =============================================================================

RESULT_DIR = Path(
    "/athena/madelab/scratch/iqh4001/Disability/Results/"
    "20260603_164343_NHANES_2017_2018_Mobility_Disability_XGB"
)

TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"

PREDICTION_FILE = TABLE_DIR / "test_predictions.csv"
COHORT_FILE = TABLE_DIR / "cohort_summary.csv"

MODEL_NAME = "XGBoost"
N_BOOTSTRAP = 2000
RANDOM_STATE = 42
SAVE_RAW_BOOTSTRAP = False

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("POST-RUN PERFORMANCE ANALYSIS")
print("NHANES 2017-2018 MOBILITY DISABILITY")
print("=" * 100)
print("Result directory :", RESULT_DIR)
print("Prediction file  :", PREDICTION_FILE)

# =============================================================================
# 2. LOAD SAVED TEST PREDICTIONS
# =============================================================================

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        "The prediction file was not found:\n"
        f"{PREDICTION_FILE}\n\n"
        "Confirm that RESULT_DIR points to the completed pipeline run."
    )

pred_df = pd.read_csv(PREDICTION_FILE)
pred_df.columns = pred_df.columns.astype(str).str.strip()

print("\nPrediction file shape:", pred_df.shape)
print("Prediction file columns:")
for column in pred_df.columns:
    print(" -", column)

OUTCOME_COLUMN = "y_true_mobility_disability"
PROBABILITY_COLUMN = "predicted_probability"
PREDICTION_05_COLUMN = "prediction_0p5"
PREDICTION_YOUDEN_COLUMN = "prediction_youden"

required_columns = [
    OUTCOME_COLUMN,
    PROBABILITY_COLUMN,
    PREDICTION_05_COLUMN,
    PREDICTION_YOUDEN_COLUMN,
]

missing_columns = [
    column for column in required_columns
    if column not in pred_df.columns
]

if missing_columns:
    raise ValueError(
        "\nRequired columns are missing from test_predictions.csv.\n"
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {pred_df.columns.tolist()}"
    )

analysis_df = pred_df[required_columns].copy()

for column in required_columns:
    analysis_df[column] = pd.to_numeric(
        analysis_df[column],
        errors="coerce",
    )

before_drop = len(analysis_df)

analysis_df = (
    analysis_df
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

after_drop = len(analysis_df)

if before_drop != after_drop:
    print(
        f"\nRemoved {before_drop - after_drop} rows because of "
        "missing or invalid values."
    )

y_true = analysis_df[OUTCOME_COLUMN].astype(int).to_numpy()
y_prob = analysis_df[PROBABILITY_COLUMN].astype(float).to_numpy()
y_pred_05_saved = analysis_df[PREDICTION_05_COLUMN].astype(int).to_numpy()
y_pred_youden_saved = analysis_df[PREDICTION_YOUDEN_COLUMN].astype(int).to_numpy()

y_prob = np.clip(y_prob, 0.0, 1.0)

# =============================================================================
# 3. VALIDATE DATA
# =============================================================================

unique_outcomes = set(np.unique(y_true).tolist())

if not unique_outcomes.issubset({0, 1}):
    raise ValueError(
        "The outcome must contain only 0 and 1.\n"
        f"Observed values: {sorted(unique_outcomes)}"
    )

if len(unique_outcomes) < 2:
    raise ValueError(
        "Both outcome classes are required to calculate AUROC and AUPRC."
    )

for prediction_name, prediction_values in [
    ("prediction_0p5", y_pred_05_saved),
    ("prediction_youden", y_pred_youden_saved),
]:
    unique_predictions = set(np.unique(prediction_values).tolist())

    if not unique_predictions.issubset({0, 1}):
        raise ValueError(
            f"{prediction_name} must contain only 0 and 1.\n"
            f"Observed values: {sorted(unique_predictions)}"
        )

n_total = len(y_true)
n_positive = int(np.sum(y_true == 1))
n_negative = int(np.sum(y_true == 0))
prevalence = float(np.mean(y_true))

print("\nTest cohort:")
print(f"Total observations       : {n_total:,}")
print(f"Mobility disability      : {n_positive:,}")
print(f"No mobility disability   : {n_negative:,}")
print(f"Outcome prevalence       : {prevalence:.6f}")

# =============================================================================
# 4. LOAD SAVED YOUDEN THRESHOLD
# =============================================================================

youden_threshold = None
threshold_source = None

if COHORT_FILE.exists():
    cohort_df = pd.read_csv(COHORT_FILE)
    cohort_df.columns = cohort_df.columns.astype(str).str.strip()

    if "best_threshold_youden" in cohort_df.columns and len(cohort_df) > 0:
        threshold_candidate = pd.to_numeric(
            cohort_df.loc[0, "best_threshold_youden"],
            errors="coerce",
        )

        if pd.notna(threshold_candidate):
            youden_threshold = float(threshold_candidate)
            threshold_source = "cohort_summary.csv"

if youden_threshold is None:
    fpr_temp, tpr_temp, threshold_temp = roc_curve(y_true, y_prob)
    youden_scores = tpr_temp - fpr_temp
    best_index = int(np.nanargmax(youden_scores))
    youden_threshold = float(threshold_temp[best_index])

    if not np.isfinite(youden_threshold):
        finite_thresholds = threshold_temp[np.isfinite(threshold_temp)]

        if len(finite_thresholds) == 0:
            raise ValueError("A finite Youden threshold could not be determined.")

        youden_threshold = float(finite_thresholds[0])

    threshold_source = "reconstructed from test predictions"

print(f"\nYouden threshold          : {youden_threshold:.9f}")
print(f"Youden threshold source   : {threshold_source}")

# =============================================================================
# 5. VALIDATE SAVED PREDICTIONS
# =============================================================================

y_pred_05_reconstructed = (y_prob >= 0.5).astype(int)
y_pred_youden_reconstructed = (y_prob >= youden_threshold).astype(int)

mismatch_05 = int(np.sum(y_pred_05_saved != y_pred_05_reconstructed))
mismatch_youden = int(
    np.sum(y_pred_youden_saved != y_pred_youden_reconstructed)
)

print("\nPrediction validation:")
print(f"0.5-threshold mismatches  : {mismatch_05}")
print(f"Youden mismatches         : {mismatch_youden}")

if mismatch_05 > 0:
    print(
        "WARNING: Saved 0.5 predictions differ from reconstructed "
        "predictions. Saved predictions will be used."
    )

if mismatch_youden > 0:
    print(
        "WARNING: Saved Youden predictions differ from reconstructed "
        "predictions. Saved predictions will be used."
    )

y_pred_05 = y_pred_05_saved
y_pred_youden = y_pred_youden_saved

# =============================================================================
# 6. PERFORMANCE FUNCTIONS
# =============================================================================

def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def calculate_performance(
    y_true_values,
    probabilities,
    predictions,
    threshold_name,
    threshold_value,
):
    y_true_values = np.asarray(y_true_values).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = np.asarray(predictions).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_values,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)

    false_positive_rate = safe_divide(fp, fp + tn)
    false_negative_rate = safe_divide(fn, fn + tp)

    positive_likelihood_ratio = (
        safe_divide(sensitivity, 1.0 - specificity)
        if pd.notna(sensitivity) and pd.notna(specificity)
        else np.nan
    )

    negative_likelihood_ratio = (
        safe_divide(1.0 - sensitivity, specificity)
        if pd.notna(sensitivity) and pd.notna(specificity)
        else np.nan
    )

    diagnostic_odds_ratio = (
        safe_divide(
            positive_likelihood_ratio,
            negative_likelihood_ratio,
        )
        if (
            pd.notna(positive_likelihood_ratio)
            and pd.notna(negative_likelihood_ratio)
        )
        else np.nan
    )

    event_rate = float(np.mean(y_true_values))
    auprc = float(
        average_precision_score(
            y_true_values,
            probabilities,
        )
    )

    auprc_lift = (
        float(auprc / event_rate)
        if event_rate > 0
        else np.nan
    )

    return {
        "Model": MODEL_NAME,
        "Threshold_Type": threshold_name,
        "Threshold": float(threshold_value),
        "N": int(len(y_true_values)),
        "Events": int(np.sum(y_true_values == 1)),
        "Non_Events": int(np.sum(y_true_values == 0)),
        "Prevalence": event_rate,
        "AUROC": float(
            roc_auc_score(
                y_true_values,
                probabilities,
            )
        ),
        "AUPRC": auprc,
        "AUPRC_Lift_Over_Prevalence": auprc_lift,
        "Accuracy": float(
            accuracy_score(
                y_true_values,
                predictions,
            )
        ),
        "Balanced_Accuracy": float(
            balanced_accuracy_score(
                y_true_values,
                predictions,
            )
        ),
        "Precision_PPV": float(
            precision_score(
                y_true_values,
                predictions,
                zero_division=0,
            )
        ),
        "Recall_Sensitivity": float(
            recall_score(
                y_true_values,
                predictions,
                zero_division=0,
            )
        ),
        "Specificity": specificity,
        "NPV": npv,
        "F1": float(
            f1_score(
                y_true_values,
                predictions,
                zero_division=0,
            )
        ),
        "MCC": float(
            matthews_corrcoef(
                y_true_values,
                predictions,
            )
        ),
        "False_Positive_Rate": false_positive_rate,
        "False_Negative_Rate": false_negative_rate,
        "Positive_Likelihood_Ratio": positive_likelihood_ratio,
        "Negative_Likelihood_Ratio": negative_likelihood_ratio,
        "Diagnostic_Odds_Ratio": diagnostic_odds_ratio,
        "Brier": float(
            brier_score_loss(
                y_true_values,
                probabilities,
            )
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

# =============================================================================
# 7. POINT ESTIMATES
# =============================================================================

performance_05 = calculate_performance(
    y_true_values=y_true,
    probabilities=y_prob,
    predictions=y_pred_05,
    threshold_name="0.5",
    threshold_value=0.5,
)

performance_youden = calculate_performance(
    y_true_values=y_true,
    probabilities=y_prob,
    predictions=y_pred_youden,
    threshold_name="Youden",
    threshold_value=youden_threshold,
)

performance_df = pd.DataFrame(
    [performance_05, performance_youden]
)

performance_output = TABLE_DIR / "postrun_performance_metrics.csv"
performance_df.to_csv(performance_output, index=False)

# =============================================================================
# 8. STRATIFIED BOOTSTRAP 95% CONFIDENCE INTERVALS
# =============================================================================

bootstrap_metric_names = [
    "AUROC",
    "AUPRC",
    "AUPRC_Lift_Over_Prevalence",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision_PPV",
    "Recall_Sensitivity",
    "Specificity",
    "NPV",
    "F1",
    "MCC",
    "False_Positive_Rate",
    "False_Negative_Rate",
    "Positive_Likelihood_Ratio",
    "Negative_Likelihood_Ratio",
    "Diagnostic_Odds_Ratio",
    "Brier",
]

positive_indices = np.where(y_true == 1)[0]
negative_indices = np.where(y_true == 0)[0]

rng = np.random.default_rng(RANDOM_STATE)
bootstrap_rows = []

print(f"\nRunning {N_BOOTSTRAP:,} stratified bootstrap iterations...")

for iteration in range(1, N_BOOTSTRAP + 1):
    sampled_positive = rng.choice(
        positive_indices,
        size=len(positive_indices),
        replace=True,
    )

    sampled_negative = rng.choice(
        negative_indices,
        size=len(negative_indices),
        replace=True,
    )

    sampled_indices = np.concatenate(
        [sampled_positive, sampled_negative]
    )

    rng.shuffle(sampled_indices)

    bootstrap_y = y_true[sampled_indices]
    bootstrap_prob = y_prob[sampled_indices]
    bootstrap_pred_05 = y_pred_05[sampled_indices]
    bootstrap_pred_youden = y_pred_youden[sampled_indices]

    bootstrap_05 = calculate_performance(
        y_true_values=bootstrap_y,
        probabilities=bootstrap_prob,
        predictions=bootstrap_pred_05,
        threshold_name="0.5",
        threshold_value=0.5,
    )

    bootstrap_youden = calculate_performance(
        y_true_values=bootstrap_y,
        probabilities=bootstrap_prob,
        predictions=bootstrap_pred_youden,
        threshold_name="Youden",
        threshold_value=youden_threshold,
    )

    for bootstrap_result in [bootstrap_05, bootstrap_youden]:
        bootstrap_rows.append(
            {
                "Bootstrap_Iteration": iteration,
                "Threshold_Type": bootstrap_result["Threshold_Type"],
                **{
                    metric: bootstrap_result[metric]
                    for metric in bootstrap_metric_names
                },
            }
        )

    if iteration % 500 == 0:
        print(
            f"Completed {iteration:,} / "
            f"{N_BOOTSTRAP:,} bootstrap iterations"
        )

bootstrap_df = pd.DataFrame(bootstrap_rows)

# =============================================================================
# 9. CONFIDENCE INTERVAL TABLE
# =============================================================================

ci_rows = []

for _, point_row in performance_df.iterrows():
    threshold_type = point_row["Threshold_Type"]

    threshold_bootstrap_df = bootstrap_df[
        bootstrap_df["Threshold_Type"] == threshold_type
    ]

    for metric in bootstrap_metric_names:
        bootstrap_values = pd.to_numeric(
            threshold_bootstrap_df[metric],
            errors="coerce",
        )

        bootstrap_values = (
            bootstrap_values
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        estimate = float(point_row[metric])

        if len(bootstrap_values) > 0:
            ci_lower = float(np.percentile(bootstrap_values, 2.5))
            ci_upper = float(np.percentile(bootstrap_values, 97.5))
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        estimate_ci = (
            f"{estimate:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
            if pd.notna(ci_lower) and pd.notna(ci_upper)
            else f"{estimate:.3f}"
        )

        ci_rows.append(
            {
                "Model": MODEL_NAME,
                "Threshold_Type": threshold_type,
                "Threshold": float(point_row["Threshold"]),
                "Metric": metric,
                "Estimate": estimate,
                "CI_95_Lower": ci_lower,
                "CI_95_Upper": ci_upper,
                "Estimate_95CI": estimate_ci,
                "Bootstrap_Iterations": N_BOOTSTRAP,
            }
        )

ci_df = pd.DataFrame(ci_rows)

ci_output = TABLE_DIR / "postrun_performance_metrics_95ci.csv"
ci_df.to_csv(ci_output, index=False)

if SAVE_RAW_BOOTSTRAP:
    bootstrap_df.to_csv(
        TABLE_DIR / "postrun_bootstrap_raw_metrics.csv",
        index=False,
    )

# =============================================================================
# 10. PUBLICATION TABLE
# =============================================================================

publication_metrics = [
    "AUROC",
    "AUPRC",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision_PPV",
    "Recall_Sensitivity",
    "Specificity",
    "NPV",
    "F1",
    "MCC",
    "Brier",
]

publication_df = (
    ci_df[
        ci_df["Metric"].isin(publication_metrics)
    ]
    .pivot(
        index="Metric",
        columns="Threshold_Type",
        values="Estimate_95CI",
    )
    .reset_index()
)

publication_order = {
    metric: index
    for index, metric in enumerate(publication_metrics)
}

publication_df["sort_order"] = (
    publication_df["Metric"].map(publication_order)
)

publication_df = (
    publication_df
    .sort_values("sort_order")
    .drop(columns="sort_order")
    .reset_index(drop=True)
)

publication_output = TABLE_DIR / "postrun_performance_publication.csv"
publication_df.to_csv(publication_output, index=False)

# =============================================================================
# 11. CONFUSION-MATRIX AND THRESHOLD TABLES
# =============================================================================

confusion_columns = [
    "Model",
    "Threshold_Type",
    "Threshold",
    "N",
    "Events",
    "Non_Events",
    "TN",
    "FP",
    "FN",
    "TP",
    "Precision_PPV",
    "Recall_Sensitivity",
    "Specificity",
    "NPV",
]

confusion_df = performance_df[confusion_columns].copy()
confusion_output = TABLE_DIR / "postrun_confusion_matrices.csv"
confusion_df.to_csv(confusion_output, index=False)

threshold_summary_df = pd.DataFrame(
    [
        {
            "Threshold_Type": "0.5",
            "Threshold": 0.5,
            "Threshold_Source": "Prespecified probability threshold",
            "Prediction_Column": PREDICTION_05_COLUMN,
            "Prediction_Mismatches": mismatch_05,
        },
        {
            "Threshold_Type": "Youden",
            "Threshold": youden_threshold,
            "Threshold_Source": threshold_source,
            "Prediction_Column": PREDICTION_YOUDEN_COLUMN,
            "Prediction_Mismatches": mismatch_youden,
        },
    ]
)

threshold_output = TABLE_DIR / "postrun_threshold_summary.csv"
threshold_summary_df.to_csv(threshold_output, index=False)

# =============================================================================
# 12. ROC CURVE
# =============================================================================

fpr, tpr, _ = roc_curve(y_true, y_prob)
auroc_value = roc_auc_score(y_true, y_prob)

roc_ci_row = ci_df[
    (ci_df["Threshold_Type"] == "0.5")
    & (ci_df["Metric"] == "AUROC")
].iloc[0]

roc_ci_lower = roc_ci_row["CI_95_Lower"]
roc_ci_upper = roc_ci_row["CI_95_Upper"]

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=(
        f"{MODEL_NAME}: AUROC = {auroc_value:.3f} "
        f"(95% CI {roc_ci_lower:.3f}-{roc_ci_upper:.3f})"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Chance",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Mobility Disability")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "postrun_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# =============================================================================
# 13. PRECISION-RECALL CURVE
# =============================================================================

precision_values, recall_values, _ = precision_recall_curve(
    y_true,
    y_prob,
)

auprc_value = average_precision_score(y_true, y_prob)

auprc_ci_row = ci_df[
    (ci_df["Threshold_Type"] == "0.5")
    & (ci_df["Metric"] == "AUPRC")
].iloc[0]

auprc_ci_lower = auprc_ci_row["CI_95_Lower"]
auprc_ci_upper = auprc_ci_row["CI_95_Upper"]

plt.figure(figsize=(7, 6))

plt.plot(
    recall_values,
    precision_values,
    linewidth=2,
    label=(
        f"{MODEL_NAME}: AUPRC = {auprc_value:.3f} "
        f"(95% CI {auprc_ci_lower:.3f}-{auprc_ci_upper:.3f})"
    ),
)

# FIXED: only one y argument is supplied here
plt.axhline(
    y=prevalence,
    linestyle="--",
    linewidth=1,
    label=f"Outcome prevalence = {prevalence:.3f}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Mobility Disability")
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "postrun_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# =============================================================================
# 14. CALIBRATION CURVE
# =============================================================================

fraction_positive, mean_predicted = calibration_curve(
    y_true,
    y_prob,
    n_bins=10,
    strategy="quantile",
)

brier_value = brier_score_loss(y_true, y_prob)

plt.figure(figsize=(7, 6))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    linewidth=2,
    label=f"{MODEL_NAME}: Brier = {brier_value:.3f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration",
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Event Proportion")
plt.title("Calibration Curve: Mobility Disability")
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "postrun_calibration_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# =============================================================================
# 15. CONFUSION-MATRIX FIGURES
# =============================================================================

def save_confusion_matrix_figure(
    true_values,
    predictions,
    title,
    output_path,
):
    cm = confusion_matrix(
        true_values,
        predictions,
        labels=[0, 1],
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.xticks(
        [0, 1],
        ["No Disability", "Disability"],
    )

    plt.yticks(
        [0, 1],
        ["No Disability", "Disability"],
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Observed Class")
    plt.title(title)

    text_threshold = cm.max() / 2.0

    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            value = cm[row_index, column_index]

            plt.text(
                column_index,
                row_index,
                f"{value:,}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if value > text_threshold else "black",
                fontsize=12,
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


save_confusion_matrix_figure(
    true_values=y_true,
    predictions=y_pred_05,
    title="Confusion Matrix: Threshold 0.5",
    output_path=FIGURE_DIR / "postrun_confusion_matrix_0p5.png",
)

save_confusion_matrix_figure(
    true_values=y_true,
    predictions=y_pred_youden,
    title=f"Confusion Matrix: Youden Threshold = {youden_threshold:.3f}",
    output_path=FIGURE_DIR / "postrun_confusion_matrix_youden.png",
)

# =============================================================================
# 16. DISPLAY RESULTS
# =============================================================================

display_columns = [
    "Model",
    "Threshold_Type",
    "Threshold",
    "N",
    "Events",
    "Prevalence",
    "AUROC",
    "AUPRC",
    "AUPRC_Lift_Over_Prevalence",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision_PPV",
    "Recall_Sensitivity",
    "Specificity",
    "NPV",
    "F1",
    "MCC",
    "Brier",
    "TN",
    "FP",
    "FN",
    "TP",
]

print("\n" + "=" * 100)
print("POINT-ESTIMATE PERFORMANCE")
print("=" * 100)

print(
    performance_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\n" + "=" * 100)
print("PUBLICATION-READY PERFORMANCE WITH 95% CI")
print("=" * 100)
print(publication_df.to_string(index=False))

print("\n" + "=" * 100)
print("CONFUSION MATRICES")
print("=" * 100)

print(
    confusion_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

# =============================================================================
# 17. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("POST-RUN PERFORMANCE ANALYSIS COMPLETE")
print("=" * 100)

print("\nSaved tables:")
print(" -", performance_output)
print(" -", ci_output)
print(" -", publication_output)
print(" -", confusion_output)
print(" -", threshold_output)

if SAVE_RAW_BOOTSTRAP:
    print(" -", TABLE_DIR / "postrun_bootstrap_raw_metrics.csv")

print("\nSaved figures:")
print(" -", FIGURE_DIR / "postrun_roc_curve.png")
print(" -", FIGURE_DIR / "postrun_precision_recall_curve.png")
print(" -", FIGURE_DIR / "postrun_calibration_curve.png")
print(" -", FIGURE_DIR / "postrun_confusion_matrix_0p5.png")
print(" -", FIGURE_DIR / "postrun_confusion_matrix_youden.png")

print("\nInterpretation note:")
print(
    "The Youden threshold was selected using the test-set ROC curve in the "
    "original pipeline. Threshold-dependent performance at this cutoff may "
    "therefore be optimistic. Ideally, determine the cutoff using training "
    "or validation data and apply it unchanged to an independent test set."
)

print("=" * 100)
