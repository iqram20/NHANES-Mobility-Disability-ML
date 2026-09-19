#!/usr/bin/env python3

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE = Path("/athena/madelab/scratch/iqh4001/Disability")

RUN = BASE / "Results" / (
    "20260727_004939_"
    "NHANES_2017_2018_Adult_Mobility_Disability_Classification_Revised"
)

SRC = RUN / "reviewer_revision_full_test_shap"
OUT = RUN / "reviewer_revision_native_treeshap"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# REQUIRED FILES
# ============================================================

required = [
    RUN / "models" / "xgboost_final_pipeline.joblib",
    SRC / "shap_feature_values_full_internal_validation.npy",
    SRC / "feature_names_shap.npy",
    SRC / "full_internal_validation_rows.csv",
    SRC / "grouped_shap_membership.csv",
]

for f in required:
    if not f.exists():
        raise FileNotFoundError(f"Required file missing: {f}")


# ============================================================
# LOAD FINAL MODEL + EXACT FULL TEST MATRIX
# ============================================================

pipeline = joblib.load(
    RUN / "models" / "xgboost_final_pipeline.joblib"
)

model = pipeline.named_steps["model"]
booster = model.get_booster()

X = np.load(
    SRC / "shap_feature_values_full_internal_validation.npy"
)

feature_names = np.load(
    SRC / "feature_names_shap.npy",
    allow_pickle=True
).astype(str)

rows = pd.read_csv(
    SRC / "full_internal_validation_rows.csv"
)

membership = pd.read_csv(
    SRC / "grouped_shap_membership.csv"
)

print("=" * 80)
print("NATIVE XGBOOST TREESHAP — FULL INTERNAL VALIDATION")
print("=" * 80)

print("Participants:", X.shape[0])
print("Transformed predictors:", X.shape[1])
print("Events:", int(rows["y_true"].sum()))

if X.shape[0] != 1171:
    raise RuntimeError(
        f"Expected 1171 participants; found {X.shape[0]}"
    )

if X.shape[1] != len(feature_names):
    raise RuntimeError(
        "Transformed feature-name count mismatch."
    )


# ============================================================
# NATIVE XGBOOST TREESHAP
# ============================================================

dmatrix = xgb.DMatrix(X)

contrib = booster.predict(
    dmatrix,
    pred_contribs=True
)

# Final column is bias/base value.
shap_values = contrib[:, :-1]
bias = contrib[:, -1]

margin = booster.predict(
    dmatrix,
    output_margin=True
)

reconstructed_margin = (
    shap_values.sum(axis=1) + bias
)

additivity_error = np.abs(
    reconstructed_margin - margin
)

print("\nNative contribution shape:", contrib.shape)
print("Native SHAP shape:", shap_values.shape)

print("\nBias")
print("Mean:", float(np.mean(bias)))
print("SD:  ", float(np.std(bias)))
print("Min: ", float(np.min(bias)))
print("Max: ", float(np.max(bias)))

print("\nAdditivity")
print("Max error: ", float(np.max(additivity_error)))
print("Mean error:", float(np.mean(additivity_error)))

additivity_pass = np.allclose(
    reconstructed_margin,
    margin,
    rtol=1e-6,
    atol=1e-5
)

print("PASS:", bool(additivity_pass))

if not additivity_pass:
    raise RuntimeError(
        "Native TreeSHAP additivity check failed."
    )


# ============================================================
# SAVE RAW NATIVE OUTPUTS
# ============================================================

np.save(
    OUT / "native_treeshap_values_FULL1171.npy",
    shap_values
)

np.save(
    OUT / "native_treeshap_bias_FULL1171.npy",
    bias
)

np.save(
    OUT / "transformed_feature_names.npy",
    feature_names
)

rows.to_csv(
    OUT / "internal_validation_rows_FULL1171.csv",
    index=False
)


# ============================================================
# VALIDATE TRANSFORMED FEATURE MEMBERSHIP
# ============================================================

membership = (
    membership
    .sort_values("transformed_index")
    .reset_index(drop=True)
)

expected_indices = np.arange(
    X.shape[1]
)

if not np.array_equal(
    membership["transformed_index"].astype(int).to_numpy(),
    expected_indices
):
    raise RuntimeError(
        "Transformed feature indices do not match."
    )

if not np.array_equal(
    membership["transformed_feature"].astype(str).to_numpy(),
    feature_names
):
    raise RuntimeError(
        "Transformed feature names do not match."
    )

print("\nFeature membership validation: PASS")


# ============================================================
# GROUP TRANSFORMED FEATURES BACK TO 326 ORIGINAL VARIABLES
# ============================================================

base_features = (
    membership
    .drop_duplicates(
        "base_feature",
        keep="first"
    )
    ["base_feature"]
    .tolist()
)

grouped_matrix = np.zeros(
    (
        X.shape[0],
        len(base_features)
    ),
    dtype=float
)

group_rows = []

for j, base in enumerate(base_features):

    sub = membership.loc[
        membership["base_feature"].eq(base)
    ]

    indices = (
        sub["transformed_index"]
        .astype(int)
        .to_numpy()
    )

    grouped_matrix[:, j] = (
        shap_values[:, indices]
        .sum(axis=1)
    )

    group_rows.append({
        "base_feature": base,
        "clinical_feature":
            sub["clinical_feature"].iloc[0],
        "domain":
            sub["domain"].iloc[0],
        "n_transformed_columns":
            len(indices),
    })

group_meta = pd.DataFrame(
    group_rows
)

if len(group_meta) != 326:
    raise RuntimeError(
        f"Expected 326 grouped features; found {len(group_meta)}"
    )

np.save(
    OUT / "native_grouped_SHAP_values_FULL1171.npy",
    grouped_matrix
)

group_meta.to_csv(
    OUT / "native_grouped_feature_membership.csv",
    index=False
)


# ============================================================
# GROUPED FEATURE IMPORTANCE
# ============================================================

grouped_importance = group_meta.copy()

grouped_importance[
    "mean_abs_grouped_shap"
] = np.mean(
    np.abs(grouped_matrix),
    axis=0
)

grouped_importance[
    "mean_signed_grouped_shap"
] = np.mean(
    grouped_matrix,
    axis=0
)

grouped_importance = (
    grouped_importance
    .sort_values(
        "mean_abs_grouped_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

grouped_importance.insert(
    0,
    "rank",
    np.arange(
        1,
        len(grouped_importance) + 1
    )
)

total = grouped_importance[
    "mean_abs_grouped_shap"
].sum()

grouped_importance[
    "relative_importance"
] = (
    grouped_importance[
        "mean_abs_grouped_shap"
    ] / total
)

grouped_importance[
    "relative_importance_pct"
] = (
    grouped_importance[
        "relative_importance"
    ] * 100
)

grouped_importance.to_csv(
    OUT / "native_grouped_feature_importance_FULL1171.csv",
    index=False
)


# ============================================================
# PATIENT-LEVEL DOMAIN ATTRIBUTION
# ============================================================

grouped_index = {
    feature: i
    for i, feature in enumerate(base_features)
}

domain_order = [
    "Demographics",
    "Lifestyle",
    "Mental_Health",
    "Medical_Conditions",
    "Examination",
    "Laboratory",
    "Diet",
]

domain_rows = []
domain_vectors = []

for domain in domain_order:

    domain_features = (
        group_meta.loc[
            group_meta["domain"].eq(domain),
            "base_feature"
        ]
        .tolist()
    )

    if not domain_features:
        continue

    indices = [
        grouped_index[f]
        for f in domain_features
    ]

    # Same primary definition used in manuscript:
    # participant-level sum of signed SHAP within domain.
    values = grouped_matrix[
        :, indices
    ].sum(axis=1)

    domain_vectors.append(values)

    domain_rows.append({
        "domain": domain,
        "n_original_features":
            len(domain_features),
        "mean_abs_domain_shap":
            float(
                np.mean(
                    np.abs(values)
                )
            ),
        "mean_signed_domain_shap":
            float(
                np.mean(values)
            ),
    })

domain_matrix = np.column_stack(
    domain_vectors
)

domain_importance = pd.DataFrame(
    domain_rows
)

domain_importance = (
    domain_importance
    .sort_values(
        "mean_abs_domain_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

domain_total = domain_importance[
    "mean_abs_domain_shap"
].sum()

domain_importance[
    "relative_importance"
] = (
    domain_importance[
        "mean_abs_domain_shap"
    ] / domain_total
)

domain_importance[
    "relative_importance_pct"
] = (
    domain_importance[
        "relative_importance"
    ] * 100
)

np.save(
    OUT / "native_domain_SHAP_values_FULL1171.npy",
    domain_matrix
)

domain_importance.to_csv(
    OUT / "native_domain_SHAP_importance_FULL1171.csv",
    index=False
)


# ============================================================
# DOMAIN-SIZE-NORMALIZED SENSITIVITY
# ============================================================

size_normalized = (
    grouped_importance
    .groupby(
        "domain",
        as_index=False
    )
    .agg(
        n_original_features=(
            "base_feature",
            "nunique"
        ),
        total_mean_abs_grouped_SHAP=(
            "mean_abs_grouped_shap",
            "sum"
        ),
        mean_abs_SHAP_per_original_variable=(
            "mean_abs_grouped_shap",
            "mean"
        ),
    )
)

normalizer = size_normalized[
    "mean_abs_SHAP_per_original_variable"
].sum()

size_normalized[
    "normalized_relative_share_pct"
] = (
    100
    * size_normalized[
        "mean_abs_SHAP_per_original_variable"
    ]
    / normalizer
)

size_normalized = (
    size_normalized
    .sort_values(
        "mean_abs_SHAP_per_original_variable",
        ascending=False
    )
    .reset_index(drop=True)
)

size_normalized.to_csv(
    OUT / "native_domain_size_normalized_SHAP_FULL1171.csv",
    index=False
)


# ============================================================
# AUDIT FILE
# ============================================================

audit = {
    "analysis":
        "Native XGBoost TreeSHAP on complete internal-validation cohort",
    "N_internal_validation":
        int(X.shape[0]),
    "N_events":
        int(rows["y_true"].sum()),
    "N_non_events":
        int((rows["y_true"] == 0).sum()),
    "N_transformed_predictors":
        int(X.shape[1]),
    "N_original_predictors":
        int(grouped_matrix.shape[1]),
    "model_refit":
        False,
    "new_split":
        False,
    "test_partition_reused":
        True,
    "algorithm":
        "XGBoost Booster.predict(pred_contribs=True)",
    "xgboost_version":
        xgb.__version__,
    "bias_mean":
        float(np.mean(bias)),
    "bias_sd":
        float(np.std(bias)),
    "max_additivity_error":
        float(np.max(additivity_error)),
    "mean_additivity_error":
        float(np.mean(additivity_error)),
    "additivity_pass_atol_1e-5":
        bool(additivity_pass),
}

with open(
    OUT / "native_TreeSHAP_audit.json",
    "w"
) as f:
    json.dump(
        audit,
        f,
        indent=2
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 ORIGINAL FEATURES")
print("=" * 80)

print(
    grouped_importance[
        [
            "rank",
            "clinical_feature",
            "base_feature",
            "domain",
            "mean_abs_grouped_shap",
            "relative_importance_pct",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("DOMAIN SHAP")
print("=" * 80)

print(
    domain_importance[
        [
            "domain",
            "n_original_features",
            "mean_abs_domain_shap",
            "relative_importance_pct",
        ]
    ]
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("DOMAIN-SIZE-NORMALIZED SHAP")
print("=" * 80)

print(
    size_normalized.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print("Output:", OUT)

