#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NHANES 2017-2018 ADULT MOBILITY-DISABILITY CLASSIFICATION
PUBLICATION-REVISED STANDALONE FULL PIPELINE
=========================================================

This is a concurrent classification/identification study, not prospective
prediction. The outcome and predictors are measured in the same NHANES cycle.

Main revisions relative to the previous pipeline
-------------------------------------------------
- Primary cohort: adults aged >=18 years.
- All DLQ variables excluded from predictors.
- Survey weights, strata, PSU, interviewer IDs, and administration metadata
  excluded from predictors.
- No global replacement of every 7/9/77/99 value.
- XPORT tiny-number missing artifacts are removed.
- Questionnaire special codes are cleaned conservatively and audited.
- Unresolved sentinel-like questionnaire variables are excluded and reported.
- Consolidated mean blood pressure replaces individual BP readings.
- One locked 20% test set is created and reused throughout.
- Five-fold development OOF probabilities are used for Platt calibration and
  model-specific Youden thresholds; the test outcomes are not used.
- XGBoost is compared with logistic regression and random forest.
- Paired stratified-bootstrap confidence intervals are saved.
- Calibration intercept, slope, O/E ratio, and binned calibration error are
  reported.
- SHAP values are grouped at the original clinical-variable level.
- Domain SHAP uses patient-level summed domain contributions.
- Domain ablation uses the same development/test split for every experiment.
- Subgroup performance is evaluated on the locked test set.

Run in Jupyter
--------------
%matplotlib inline
%run NHANES_publication_revised_mobility_pipeline.py
"""

from __future__ import annotations

import datetime
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as exc:
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = repr(exc)

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(
    "/athena/madelab/scratch/iqh4001/Disability/Data/NHANES_2017_2018"
)
RESULTS_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/Disability/Results"
)

RUN_ID = (
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_NHANES_2017_2018_Adult_Mobility_Disability_Classification_Revised"
)
OUT_DIR = RESULTS_ROOT / RUN_ID
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
AUDIT_DIR = OUT_DIR / "audit"
SHAP_DIR = OUT_DIR / "shap"
ABLATION_DIR = OUT_DIR / "domain_ablation"
SUBGROUP_DIR = OUT_DIR / "subgroups"

for directory in [
    OUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR,
    AUDIT_DIR, SHAP_DIR, ABLATION_DIR, SUBGROUP_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

OUTCOME_SOURCE = "DLQ050"
OUTCOME = "mobility_disability"
MINIMUM_AGE = 18
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = 8
N_BOOTSTRAP = 2000
N_ABLATION_BOOTSTRAP = 1000
N_SUBGROUP_BOOTSTRAP = 500
SHAP_SAMPLE_N = 1000
MINIMUM_NONMISSING_N = 50
MAXIMUM_MISSING_FRACTION = 0.95
MAXIMUM_CATEGORY_LEVELS = 30
CMAP_NAME = "plasma"
OUTPUT_DPI = 600

MODEL_NAMES = [
    "XGBoost",
    "Logistic Regression",
    "Random Forest",
]

SURVEY_WEIGHT_CANDIDATES = [
    "WTMEC2YR",
    "WTINT2YR",
]


# =============================================================================
# EXCLUSIONS AND DOMAINS
# =============================================================================

EXCLUDED_EXACT = {
    "SEQN", OUTCOME_SOURCE, OUTCOME,
    "any_disability", "disability_domain_score",

    # Survey design and weights.
    "WTINT2YR", "WTMEC2YR", "WTDRD1", "WTDR2D",
    "SDMVPSU", "SDMVSTRA", "SDDSRVYR",

    # Redundant/low-interpretability variables.
    "RIDEXAGM", "OSQ230",

    # Dietary administration metadata.
    "DR1EXMER", "DR2EXMER", "DR1MRESP", "DR2MRESP",
    "DR1LANG", "DR2LANG", "DR1HELP", "DR2HELP",
    "DR1DRSTZ", "DR2DRSTZ",

    # Interview administration metadata.
    "RIDSTATR", "RIDEXMON",
    "SIALANG", "SIAPROXY", "SIAINTRP",
    "MIALANG", "MIAPROXY", "MIAINTRP",
    "FIALANG", "FIAPROXY", "FIAINTRP",
}

# All concurrent disability questionnaire variables are excluded.
EXCLUDED_PREFIXES = ("DLQ",)

RAW_BP_COLUMNS = {
    "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
    "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
}

DOMAIN_PREFIXES: Dict[str, Sequence[str]] = {
    "Demographics": ("RID", "DMD", "IND", "INQ"),
    "Lifestyle": ("ALQ", "SMQ", "SMD", "PAQ", "PAD", "SLQ", "SLD"),
    "Mental_Health": ("DPQ", "PHQ9"),
    "Medical_Conditions": ("MCQ", "DIQ", "BPQ", "AGQ", "OSQ", "HSQ"),
    "Examination": ("BMX", "BPX", "SBP", "DBP"),
    "Laboratory": ("LBX", "LBD", "URX"),
    "Diet": ("DR1", "DR2"),
}


# =============================================================================
# SPECIAL-MISSING HANDLING
# =============================================================================

QUESTIONNAIRE_PREFIXES = (
    "ALQ", "DPQ", "HSQ", "MCQ", "PAQ", "PAD", "SLQ",
    "SMQ", "SMD", "DIQ", "BPQ", "AGQ", "OSQ", "INQ",
)

# Explicit rules for selected variables. These rules are applied only to the
# named variables and never globally.
EXPLICIT_SPECIAL_CODES: Dict[str, Sequence[float]] = {
    "RIAGENDR": (7, 9),
    "RIDRETH1": (7, 9),
    "RIDRETH3": (7, 9),
    "DMDEDUC2": (7, 9),
    "DMDMARTL": (77, 99),
    "INDHHIN2": (77, 99),
    "ALQ130": (777, 999),
    "PAD680": (7777, 9999),
    "SLD012": (77, 99),
    "SLD013": (77, 99),
}

# Values such as 7 or 9 can be legitimate for these count/continuous fields.
SPECIAL_CODE_EXCEPTIONS = {
    "RIDAGEYR", "RIDEXAGM", "DMDHHSIZ", "DMDFMSIZ",
    "DMDHHSZA", "DMDHHSZB", "DMDHHSZE", "INDFMPIR",
}

CONTINUOUS_PREFIXES = (
    "BMX", "BPX", "LBX", "LBD", "URX",
    "DR1T", "DR2T", "DR1K", "DR2K", "SBP", "DBP",
)

KNOWN_CONTINUOUS = {
    "RIDAGEYR", "INDFMPIR", "PHQ9_TOTAL", "SBP_MEAN", "DBP_MEAN",
    "SLD012", "SLD013", "PAD680",
}

SENTINEL_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999)


# =============================================================================
# LABELS
# =============================================================================

FEATURE_LABELS = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "RIDRETH3": "Race/Ethnicity",
    "DMDEDUC2": "Education",
    "DMDMARTL": "Marital Status",
    "INDHHIN2": "Household Income",
    "INDFMPIR": "Income-to-Poverty Ratio",
    "DMDHHSIZ": "Household Size",
    "DMDHHSZE": "Adults Aged >=60 in Household",
    "BMXBMI": "Body Mass Index",
    "BMXWT": "Weight",
    "BMXHT": "Height",
    "BMXWAIST": "Waist Circumference",
    "BMXHIP": "Hip Circumference",
    "BMXARMC": "Arm Circumference",
    "BMXARML": "Upper Arm Length",
    "SBP_MEAN": "Mean Systolic Blood Pressure",
    "DBP_MEAN": "Mean Diastolic Blood Pressure",
    "BPXPLS": "Pulse Rate",
    "PHQ9_TOTAL": "PHQ-9 Depression Score",
    "DPQ100": "Depression-Related Functional Difficulty",
    "MCQ160A": "Arthritis",
    "MCQ160B": "Congestive Heart Failure",
    "MCQ160C": "Coronary Heart Disease",
    "MCQ160D": "Angina",
    "MCQ160E": "Heart Attack",
    "MCQ160F": "Stroke",
    "MCQ160L": "Liver Condition",
    "MCQ160M": "Thyroid Condition",
    "MCQ220": "Cancer",
    "MCQ366D": "Advised to Reduce Dietary Fat or Calories",
    "MCQ092": "History of Blood Transfusion",
    "PAQ665": "Moderate Recreational Activity",
    "PAQ650": "Vigorous Recreational Activity",
    "PAD680": "Sedentary Time",
    "SLD012": "Weekday Sleep Duration",
    "SLD013": "Weekend Sleep Duration",
    "LBXGH": "HbA1c",
    "LBXGLU": "Fasting Glucose",
    "LBDHDD": "HDL Cholesterol",
    "LBXTC": "Total Cholesterol",
    "LBXHGB": "Hemoglobin",
    "LBXHCT": "Hematocrit",
    "LBXRDW": "Red Cell Distribution Width",
    "LBXPLTSI": "Platelet Count",
    "DR1TNUMF": "Day 1 Number of Foods/Beverages",
    "DR2TNUMF": "Day 2 Number of Foods/Beverages",
    "DR1TVC": "Day 1 Vitamin C Intake",
    "DR2TVC": "Day 2 Vitamin C Intake",
    "DR1TVARA": "Day 1 Vitamin A Intake",
    "DR2TVARA": "Day 2 Vitamin A Intake",
    "DR1TFF": "Day 1 Food Folate Intake",
    "DR2TFF": "Day 2 Food Folate Intake",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value))


def save_json(data: Mapping[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure_all_formats(fig: plt.Figure, png_path: Path) -> None:
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight")
    try:
        fig.savefig(
            png_path.with_suffix(".tiff"),
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception as exc:
        print("TIFF export skipped:", exc)


# =============================================================================
# LOAD AND MERGE
# =============================================================================

def safe_read_xpt(path: Path) -> pd.DataFrame:
    frame = pd.read_sas(path, format="xport")
    frame.columns = [str(column).upper() for column in frame.columns]
    return frame


def load_modules(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(list(data_dir.glob("*.xpt")) + list(data_dir.glob("*.XPT")))
    if not files:
        raise FileNotFoundError(f"No XPT files found in {data_dir}")

    modules: Dict[str, pd.DataFrame] = {}
    print("\nLoading XPT files...")
    for path in files:
        name = path.stem.upper()
        frame = safe_read_xpt(path)
        modules[name] = frame
        print(f"Loaded {name:12s}: {frame.shape}")

    if "DEMO_J" not in modules or "DLQ_J" not in modules:
        raise KeyError("DEMO_J and DLQ_J are required.")
    return modules


def merge_modules(modules: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    merged = modules["DEMO_J"].copy()
    for name, frame in modules.items():
        if name == "DEMO_J" or "SEQN" not in frame.columns:
            continue
        if frame["SEQN"].duplicated().any():
            raise RuntimeError(f"Duplicate SEQN values in {name}")
        overlap = [c for c in frame.columns if c in merged.columns and c != "SEQN"]
        merged = merged.merge(
            frame.drop(columns=overlap, errors="ignore"),
            on="SEQN",
            how="left",
            validate="one_to_one",
        )
        print(f"Merged {name:12s} -> {merged.shape}")
    return merged


# =============================================================================
# COHORT AND DERIVED VARIABLES
# =============================================================================

def replace_xport_tiny_artifacts(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    rows = []
    for column in output.columns:
        if not pd.api.types.is_numeric_dtype(output[column]):
            continue
        numeric = pd.to_numeric(output[column], errors="coerce")
        mask = numeric.notna() & numeric.ne(0) & numeric.abs().lt(1e-50)
        count = int(mask.sum())
        if count:
            output.loc[mask, column] = np.nan
            rows.append({"feature": column, "replaced_count": count})
    return output, pd.DataFrame(rows)


def derive_variables(merged: pd.DataFrame) -> pd.DataFrame:
    output = merged.copy()
    output[OUTCOME] = np.where(
        output[OUTCOME_SOURCE].eq(1), 1,
        np.where(output[OUTCOME_SOURCE].eq(2), 0, np.nan),
    )

    dpq = [f"DPQ0{i}0" for i in range(1, 10)]
    dpq = [column for column in dpq if column in output.columns]
    if dpq:
        output["PHQ9_TOTAL"] = (
            output[dpq].replace({7: np.nan, 9: np.nan}).sum(axis=1, min_count=1)
        )

    systolic = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in output]
    diastolic = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in output]
    if systolic:
        output["SBP_MEAN"] = output[systolic].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if diastolic:
        output["DBP_MEAN"] = output[diastolic].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return output


def construct_adult_cohort(merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{
        "Step": 1,
        "Description": "Merged NHANES participants",
        "N": len(merged),
        "Events": int(pd.to_numeric(merged[OUTCOME], errors="coerce").fillna(0).sum()),
    }]

    cohort = merged.dropna(subset=[OUTCOME]).copy()
    cohort[OUTCOME] = cohort[OUTCOME].astype(int)
    rows.append({
        "Step": 2,
        "Description": "Binary DLQ050 outcome available",
        "N": len(cohort),
        "Events": int(cohort[OUTCOME].sum()),
    })

    age = pd.to_numeric(cohort["RIDAGEYR"], errors="coerce")
    cohort = cohort.loc[age.ge(MINIMUM_AGE)].copy()
    rows.append({
        "Step": 3,
        "Description": f"Adults aged >={MINIMUM_AGE} years",
        "N": len(cohort),
        "Events": int(cohort[OUTCOME].sum()),
    })

    if cohort["SEQN"].duplicated().any():
        raise RuntimeError("Duplicate SEQN in adult cohort")
    return cohort.reset_index(drop=True), pd.DataFrame(rows)


# =============================================================================
# FEATURE SELECTION AND CLEANING
# =============================================================================

def assign_domain(feature: str) -> Optional[str]:
    for domain, prefixes in DOMAIN_PREFIXES.items():
        if any(feature.startswith(prefix) for prefix in prefixes):
            return domain
    return None


def exclusion_reason(feature: str) -> str:
    if feature in EXCLUDED_EXACT:
        return "explicitly excluded"
    if feature in RAW_BP_COLUMNS:
        return "replaced by mean blood pressure"
    if any(feature.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return "concurrent disability questionnaire variable"
    if assign_domain(feature) is None:
        return "outside predefined domains"
    return ""


def build_candidate_features(cohort: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    selected = []
    for feature in cohort.columns:
        reason = exclusion_reason(feature)
        domain = assign_domain(feature)
        included = not bool(reason)
        rows.append({
            "feature": feature,
            "domain": domain,
            "included_initially": included,
            "exclusion_reason": reason,
        })
        if included:
            selected.append(feature)
    return sorted(set(selected)), pd.DataFrame(rows)


def decode_objects(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype == object:
            output[column] = output[column].apply(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
    return output


def clean_special_codes(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Conservative variable-aware cleaning; no global 7/9 replacement."""
    output = frame.copy()
    rows = []
    unresolved = []

    for feature in output.columns:
        series = output[feature]
        if not pd.api.types.is_numeric_dtype(series):
            rows.append({
                "feature": feature,
                "replacement_count": 0,
                "rule": "not numeric",
                "unresolved_sentinel": False,
            })
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        mask = pd.Series(False, index=numeric.index)
        rules = []

        for code in EXPLICIT_SPECIAL_CODES.get(feature, ()):
            code_mask = numeric.eq(code)
            if code_mask.any():
                mask |= code_mask
                rules.append(f"explicit:{code:g}")

        is_questionnaire = feature.startswith(QUESTIONNAIRE_PREFIXES)
        is_continuous = feature in KNOWN_CONTINUOUS or feature.startswith(CONTINUOUS_PREFIXES)

        if feature not in SPECIAL_CODE_EXCEPTIONS and is_questionnaire and not is_continuous:
            observed = np.sort(numeric.dropna().unique().astype(float))
            low_cardinality = len(observed) <= MAXIMUM_CATEGORY_LEVELS

            for code in SENTINEL_CODES:
                if code not in observed:
                    continue
                ordinary = observed[~np.isin(observed, SENTINEL_CODES)]
                if len(ordinary) == 0:
                    continue
                ordinary_max = float(np.max(ordinary))
                if code in (7, 9):
                    supported = low_cardinality and ordinary_max <= 6
                else:
                    supported = (
                        low_cardinality
                        and ordinary_max < code
                        and code >= max(10.0, ordinary_max * 2.0)
                    )
                if supported:
                    mask |= numeric.eq(code)
                    rules.append(f"heuristic:{code:g}")

            remaining = set(numeric.mask(mask).dropna().unique().astype(float))
            remaining_sentinels = sorted(float(code) for code in SENTINEL_CODES if float(code) in remaining)
            if remaining_sentinels and not low_cardinality:
                unresolved.append(feature)
                rules.append("unresolved:" + ",".join(f"{v:g}" for v in remaining_sentinels))

        if mask.any():
            output.loc[mask, feature] = np.nan

        rows.append({
            "feature": feature,
            "replacement_count": int(mask.sum()),
            "rule": "; ".join(rules) if rules else "none",
            "unresolved_sentinel": feature in unresolved,
        })

    return output, pd.DataFrame(rows), sorted(set(unresolved))


def quality_filter(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    unresolved: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:
    unresolved_set = set(unresolved)
    rows = []
    retained = []

    for feature in candidates:
        series = frame[feature]
        nonmissing = int(series.notna().sum())
        missing_fraction = float(series.isna().mean())
        unique = int(series.nunique(dropna=True))
        reason = ""
        if feature in unresolved_set:
            reason = "unresolved sentinel-like values"
        elif nonmissing < MINIMUM_NONMISSING_N:
            reason = f"nonmissing n < {MINIMUM_NONMISSING_N}"
        elif missing_fraction > MAXIMUM_MISSING_FRACTION:
            reason = f"missing fraction > {MAXIMUM_MISSING_FRACTION}"
        elif unique < 2:
            reason = "constant/noninformative"

        included = not bool(reason)
        if included:
            retained.append(feature)
        rows.append({
            "feature": feature,
            "included_final": included,
            "nonmissing_n": nonmissing,
            "missing_fraction": missing_fraction,
            "unique_nonmissing": unique,
            "exclusion_reason": reason,
        })
    return retained, pd.DataFrame(rows)


def identify_feature_types(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for feature in frame.columns:
        series = frame[feature]
        if feature in KNOWN_CONTINUOUS or feature.startswith(CONTINUOUS_PREFIXES):
            numeric.append(feature)
        elif series.dtype == object or series.nunique(dropna=True) <= 15:
            categorical.append(feature)
        else:
            numeric.append(feature)
    return sorted(numeric), sorted(categorical)


def cast_matrix(
    frame: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """
    Cast the model matrix to stable scikit-learn-compatible dtypes.

    Numerical features become numeric with invalid values coerced to NaN.
    Every observed categorical value becomes a Python string, while missing
    values remain np.nan for SimpleImputer. This prevents OneHotEncoder from
    receiving mixed floats and strings.
    """

    output = frame.copy()

    for feature in numeric_features:
        output[
            feature
        ] = pd.to_numeric(
            output[
                feature
            ],
            errors="coerce",
        )

    for feature in categorical_features:
        series = output[
            feature
        ]

        output[
            feature
        ] = series.map(
            lambda value: (
                np.nan
                if pd.isna(
                    value
                )
                else (
                    value.decode(
                        "utf-8"
                    )
                    if isinstance(
                        value,
                        bytes,
                    )
                    else str(
                        value
                    )
                )
            )
        ).astype(
            object
        )

    return output


# =============================================================================
# PREPROCESSING AND MODELS
# =============================================================================

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            list(numeric_features),
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", make_ohe()),
            ]),
            list(categorical_features),
        ))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(
    model_name: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    y_train: pd.Series,
    random_state: int,
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    positive = int(y_train.eq(1).sum())
    negative = int(y_train.eq(0).sum())
    scale_pos_weight = negative / max(positive, 1)

    if model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=N_JOBS,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=2.0,
            reg_alpha=0.2,
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=N_JOBS,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# =============================================================================
# CALIBRATION, THRESHOLD, AND METRICS
# =============================================================================

def clip_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)


def logit_probability(probability: np.ndarray) -> np.ndarray:
    p = clip_probability(probability)
    return np.log(p / (1 - p))


def fit_platt(y_true: np.ndarray, raw_probability: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(logit_probability(raw_probability).reshape(-1, 1), np.asarray(y_true, dtype=int))
    return model


def apply_platt(model: LogisticRegression, raw_probability: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit_probability(raw_probability).reshape(-1, 1))[:, 1]


def choose_youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    scores = tpr - fpr
    scores = np.where(np.isfinite(thresholds), scores, -np.inf)
    return float(thresholds[int(np.argmax(scores))])


def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate calibration intercept and slope.

    For a constant-probability model, the calibration slope is not
    identifiable because the model has no probability variation. In that
    case, report:
      - intercept = logit(observed prevalence) - logit(predicted prevalence)
      - slope = NaN
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    valid = (
        np.isfinite(
            y_true
        )
        & np.isfinite(
            probability
        )
    )

    y_valid = y_true[
        valid
    ]

    probability_valid = probability[
        valid
    ]

    if (
        len(
            y_valid
        )
        == 0
        or len(
            np.unique(
                y_valid
            )
        )
        < 2
    ):
        return (
            np.nan,
            np.nan,
        )

    if len(
        np.unique(
            probability_valid
        )
    ) < 2:
        observed_prevalence = float(
            np.mean(
                y_valid
            )
        )

        predicted_prevalence = float(
            np.mean(
                probability_valid
            )
        )

        intercept = float(
            logit_probability(
                np.asarray(
                    [
                        observed_prevalence
                    ]
                )
            )[
                0
            ]
            - logit_probability(
                np.asarray(
                    [
                        predicted_prevalence
                    ]
                )
            )[
                0
            ]
        )

        return (
            intercept,
            np.nan,
        )

    try:
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
        )

        model.fit(
            logit_probability(
                probability_valid
            ).reshape(
                -1,
                1,
            ),
            y_valid,
        )

        return (
            float(
                model.intercept_[
                    0
                ]
            ),
            float(
                model.coef_[
                    0,
                    0,
                ]
            ),
        )

    except Exception:
        return (
            np.nan,
            np.nan,
        )


def binned_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Weighted absolute calibration error across quantile bins.

    Constant predictions, such as a prevalence-only baseline, cannot be
    divided into quantile bins. For those models, this correctly reduces to
    the absolute difference between observed and predicted prevalence.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    valid = (
        np.isfinite(
            y_true
        )
        & np.isfinite(
            probability
        )
    )

    y_valid = y_true[
        valid
    ]

    probability_valid = probability[
        valid
    ]

    if len(
        y_valid
    ) == 0:
        return np.nan

    overall_error = float(
        abs(
            np.mean(
                y_valid
            )
            - np.mean(
                probability_valid
            )
        )
    )

    unique_probability_count = int(
        len(
            np.unique(
                probability_valid
            )
        )
    )

    # A constant-probability model has one valid calibration bin.
    if unique_probability_count < 2:
        return overall_error

    requested_bins = int(
        min(
            max(
                n_bins,
                1,
            ),
            unique_probability_count,
            len(
                probability_valid
            ),
        )
    )

    if requested_bins < 2:
        return overall_error

    frame = pd.DataFrame(
        {
            "y": y_valid,
            "p": probability_valid,
        }
    )

    try:
        frame[
            "bin"
        ] = pd.qcut(
            frame[
                "p"
            ],
            q=requested_bins,
            duplicates="drop",
        )

    except Exception:
        return overall_error

    frame = frame.dropna(
        subset=[
            "bin",
        ]
    )

    if frame.empty:
        return overall_error

    grouped = (
        frame.groupby(
            "bin",
            observed=True,
        )
        .agg(
            observed=(
                "y",
                "mean",
            ),
            predicted=(
                "p",
                "mean",
            ),
            n=(
                "y",
                "size",
            ),
        )
        .reset_index(
            drop=True
        )
    )

    if grouped.empty:
        return overall_error

    errors = np.abs(
        grouped[
            "observed"
        ].to_numpy(
            dtype=float
        )
        - grouped[
            "predicted"
        ].to_numpy(
            dtype=float
        )
    )

    weights = grouped[
        "n"
    ].to_numpy(
        dtype=float
    )

    usable = (
        np.isfinite(
            errors
        )
        & np.isfinite(
            weights
        )
        & (
            weights
            > 0
        )
    )

    if (
        not usable.any()
        or float(
            weights[
                usable
            ].sum()
        )
        <= 0
    ):
        return overall_error

    return float(
        np.average(
            errors[
                usable
            ],
            weights=weights[
                usable
            ],
        )
    )


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    intercept, slope = calibration_intercept_slope(y_true, probability)
    return {
        "AUROC": float(roc_auc_score(y_true, probability)),
        "AUPRC": float(average_precision_score(y_true, probability)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "Observed_to_Expected_Ratio": float(y_true.sum() / max(probability.sum(), 1e-12)),
        "Quantile_Calibration_Error": binned_calibration_error(y_true, probability),
    }


def threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "Precision_PPV": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "NPV": float(tn / max(tn + fn, 1)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, prediction)),
        "Threshold": float(threshold),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def all_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    return {
        **probability_metrics(y_true, probability),
        **threshold_metrics(y_true, probability, threshold),
    }


def weighted_probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    valid = np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2 or len(np.unique(np.asarray(y_true)[valid])) < 2:
        return {}
    y = np.asarray(y_true)[valid]
    p = np.asarray(probability)[valid]
    w = np.asarray(weights)[valid]
    return {
        "Survey_Weighted_AUROC": float(roc_auc_score(y, p, sample_weight=w)),
        "Survey_Weighted_AUPRC": float(average_precision_score(y, p, sample_weight=w)),
        "Survey_Weighted_Brier": float(brier_score_loss(y, p, sample_weight=w)),
    }


# =============================================================================
# BOOTSTRAP
# =============================================================================

def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    output = np.concatenate([
        rng.choice(negative, len(negative), replace=True),
        rng.choice(positive, len(positive), replace=True),
    ])
    rng.shuffle(output)
    return output


BOOTSTRAP_METRICS = [
    "AUROC", "AUPRC", "Brier", "Accuracy", "Balanced_Accuracy",
    "Precision_PPV", "Recall_Sensitivity", "Specificity", "NPV", "F1", "MCC",
]


def bootstrap_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(n_bootstrap):
        index = stratified_bootstrap_indices(y_true, rng)
        metrics = all_metrics(np.asarray(y_true)[index], np.asarray(probability)[index], threshold)
        rows.append({
            "Bootstrap_Iteration": iteration + 1,
            **{metric: metrics[metric] for metric in BOOTSTRAP_METRICS},
        })
    return pd.DataFrame(rows)


def summarize_bootstrap(point: Mapping[str, float], boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in BOOTSTRAP_METRICS:
        values = boot[metric].dropna().to_numpy(dtype=float)
        rows.append({
            "Metric": metric,
            "Estimate": float(point[metric]),
            "CI_95_Lower": float(np.percentile(values, 2.5)),
            "CI_95_Upper": float(np.percentile(values, 97.5)),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(n_bootstrap):
        index = stratified_bootstrap_indices(y_true, rng)
        y = np.asarray(y_true)[index]
        a = np.asarray(probability_a)[index]
        b = np.asarray(probability_b)[index]
        rows.append({
            "Bootstrap_Iteration": iteration + 1,
            "AUROC_Difference": roc_auc_score(y, a) - roc_auc_score(y, b),
            "AUPRC_Difference": average_precision_score(y, a) - average_precision_score(y, b),
            "Brier_Difference": brier_score_loss(y, a) - brier_score_loss(y, b),
        })
    return pd.DataFrame(rows)


# =============================================================================
# OOF TRAINING AND CALIBRATION
# =============================================================================

def train_model(
    model_name: str,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_test: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Dict[str, Any]:
    header(f"TRAINING {model_name.upper()}")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_raw = np.full(len(X_development), np.nan)
    fold_assignment = np.full(len(X_development), -1, dtype=int)
    fold_rows = []

    for fold, (train_index, validation_index) in enumerate(cv.split(X_development, y_development), 1):
        pipeline = build_pipeline(
            model_name,
            numeric_features,
            categorical_features,
            y_development.iloc[train_index],
            RANDOM_STATE + fold,
        )
        pipeline.fit(X_development.iloc[train_index], y_development.iloc[train_index])
        probability = pipeline.predict_proba(X_development.iloc[validation_index])[:, 1]
        oof_raw[validation_index] = probability
        fold_assignment[validation_index] = fold
        metrics = probability_metrics(y_development.iloc[validation_index].to_numpy(), probability)
        fold_rows.append({
            "Model": model_name,
            "Fold": fold,
            "Training_N": len(train_index),
            "Validation_N": len(validation_index),
            "Validation_Events": int(y_development.iloc[validation_index].sum()),
            **metrics,
        })
        print(f"Fold {fold}: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}")

    if np.isnan(oof_raw).any():
        raise RuntimeError(f"Missing OOF probabilities for {model_name}")

    calibrator = fit_platt(y_development.to_numpy(), oof_raw)
    oof_calibrated = apply_platt(calibrator, oof_raw)
    threshold = choose_youden_threshold(y_development.to_numpy(), oof_calibrated)

    final_pipeline = build_pipeline(
        model_name,
        numeric_features,
        categorical_features,
        y_development,
        RANDOM_STATE,
    )
    final_pipeline.fit(X_development, y_development)
    test_raw = final_pipeline.predict_proba(X_test)[:, 1]
    test_calibrated = apply_platt(calibrator, test_raw)

    return {
        "pipeline": final_pipeline,
        "calibrator": calibrator,
        "locked_threshold": threshold,
        "oof_raw": oof_raw,
        "oof_calibrated": oof_calibrated,
        "fold_assignment": fold_assignment,
        "fold_metrics": pd.DataFrame(fold_rows),
        "test_raw": test_raw,
        "test_calibrated": test_calibrated,
    }


# =============================================================================
# PERFORMANCE FIGURE
# =============================================================================

def performance_figure(
    y_test: np.ndarray,
    model_results: Mapping[str, Mapping[str, Any]],
    ci_table: pd.DataFrame,
) -> Path:
    configure_style()
    cmap = mpl.colormaps[CMAP_NAME]
    colors = {
        "XGBoost": cmap(0.12),
        "Logistic Regression": cmap(0.45),
        "Random Forest": cmap(0.72),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # A: AUROC/AUPRC with CIs.
    metrics = ["AUROC", "AUPRC"]
    x = np.arange(len(metrics))
    width = 0.22
    for model_index, model_name in enumerate(MODEL_NAMES):
        rows = ci_table.loc[ci_table["Model"].eq(model_name)].set_index("Metric")
        estimate = np.array([rows.loc[m, "Estimate"] for m in metrics], dtype=float)
        lower = np.array([rows.loc[m, "CI_95_Lower"] for m in metrics], dtype=float)
        upper = np.array([rows.loc[m, "CI_95_Upper"] for m in metrics], dtype=float)
        bars = ax_a.bar(
            x + (model_index - 1) * width,
            estimate,
            width=width,
            yerr=np.vstack([estimate - lower, upper - estimate]),
            capsize=4,
            color=colors[model_name],
            edgecolor="black",
            linewidth=0.5,
            label=model_name,
        )
        for bar, value, upper_ci in zip(bars, estimate, upper):
            ax_a.text(bar.get_x() + bar.get_width() / 2, upper_ci + 0.025, f"{value:.2f}",
                      ha="center", va="bottom", fontsize=8.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metrics)
    ax_a.set_ylim(0, 1.12)
    ax_a.set_ylabel("Performance")
    ax_a.set_title("A. Model discrimination", loc="left", fontweight="bold")
    ax_a.grid(axis="y", linestyle="--", alpha=0.20)
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=8.5)

    # B: ROC.
    for model_name in MODEL_NAMES:
        p = model_results[model_name]["test_calibrated"]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax_b.plot(fpr, tpr, color=colors[model_name], linewidth=2.2,
                  label=f"{model_name}: {roc_auc_score(y_test, p):.3f}")
    ax_b.plot([0, 1], [0, 1], "--", color="dimgray", label="Chance")
    ax_b.set_xlabel("False positive rate")
    ax_b.set_ylabel("True positive rate")
    ax_b.set_title("B. Receiver operating characteristic curves", loc="left", fontweight="bold")
    ax_b.legend(loc="lower right", fontsize=8.2)

    # C: PR.
    prevalence = float(np.mean(y_test))
    for model_name in MODEL_NAMES:
        p = model_results[model_name]["test_calibrated"]
        precision, recall, _ = precision_recall_curve(y_test, p)
        ax_c.plot(recall, precision, color=colors[model_name], linewidth=2.2,
                  label=f"{model_name}: {average_precision_score(y_test, p):.3f}")
    ax_c.axhline(prevalence, linestyle="--", color="dimgray",
                 label=f"Outcome prevalence: {prevalence:.3f}")
    ax_c.set_xlabel("Recall")
    ax_c.set_ylabel("Precision")
    ax_c.set_title("C. Precision-recall curves", loc="left", fontweight="bold")
    ax_c.legend(loc="upper right", fontsize=8.2)

    # D: calibration.
    for model_name in MODEL_NAMES:
        p = model_results[model_name]["test_calibrated"]
        observed, predicted = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax_d.plot(predicted, observed, marker="o", markersize=4.5,
                  color=colors[model_name], linewidth=2.0,
                  label=f"{model_name}: Brier={brier_score_loss(y_test, p):.3f}")
    ax_d.plot([0, 1], [0, 1], "--", color="dimgray", label="Perfect calibration")
    ax_d.set_xlabel("Mean predicted probability")
    ax_d.set_ylabel("Observed event proportion")
    ax_d.set_title("D. Calibration curves", loc="left", fontweight="bold")
    ax_d.legend(loc="upper left", fontsize=8.2)

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    output = FIG_DIR / "Figure_Model_Performance_2x2.png"
    save_figure_all_formats(fig, output)
    plt.show()
    plt.close(fig)
    return output


# =============================================================================
# SHAP AND GROUPING
# =============================================================================

def ensure_2d_shap(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            array = array[:, :, 1]
        elif array.shape[0] == 2:
            array = array[1]
        else:
            raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"SHAP must be 2D: {array.shape}")
    return array.astype(float)


def recover_base_feature(transformed_name: str, original_features: Sequence[str]) -> str:
    cleaned = re.sub(r"^(num|cat|remainder)__", "", str(transformed_name))
    if cleaned in original_features:
        return cleaned
    for feature in sorted(original_features, key=len, reverse=True):
        if cleaned == feature or cleaned.startswith(feature + "_") or cleaned.startswith(feature + "="):
            return feature
    return cleaned.split("_")[0]


def clinical_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def run_shap(
    xgb_pipeline: Pipeline,
    X_test: pd.DataFrame,
    test_seqn: np.ndarray,
    original_features: Sequence[str],
    feature_to_domain: Mapping[str, str],
) -> Dict[str, Any]:
    if not SHAP_AVAILABLE:
        print("SHAP skipped:", SHAP_IMPORT_ERROR)
        return {"available": False, "error": SHAP_IMPORT_ERROR}

    header("RUNNING GROUPED SHAP")
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    model = xgb_pipeline.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(X_test), dtype=float)
    transformed_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object).astype(str)

    sample_n = min(SHAP_SAMPLE_N, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_index = rng.choice(len(X_test), sample_n, replace=False)
    transformed_sample = transformed[sample_index]

    explainer = shap.TreeExplainer(model)
    shap_values = ensure_2d_shap(explainer.shap_values(transformed_sample))
    if shap_values.shape != transformed_sample.shape:
        raise RuntimeError("SHAP and transformed feature matrices differ")

    np.save(SHAP_DIR / "shap_values_full.npy", shap_values)
    np.save(SHAP_DIR / "shap_feature_values.npy", transformed_sample)
    np.save(SHAP_DIR / "feature_names_shap.npy", transformed_names)
    pd.DataFrame({
        "test_row_position": sample_index,
        "SEQN": test_seqn[sample_index],
    }).to_csv(SHAP_DIR / "shap_sample_rows.csv", index=False)
    X_test.iloc[sample_index].reset_index(drop=True).to_csv(
        SHAP_DIR / "shap_sample_original_feature_values.csv", index=False
    )

    base_to_indices: Dict[str, List[int]] = defaultdict(list)
    membership_rows = []
    for index, transformed_name in enumerate(transformed_names):
        base = recover_base_feature(transformed_name, original_features)
        base_to_indices[base].append(index)
        membership_rows.append({
            "base_feature": base,
            "clinical_feature": clinical_label(base),
            "domain": feature_to_domain.get(base, "Unknown"),
            "transformed_index": index,
            "transformed_feature": transformed_name,
        })
    pd.DataFrame(membership_rows).to_csv(
        SHAP_DIR / "grouped_shap_membership.csv", index=False
    )

    grouped_features = list(base_to_indices)
    grouped_matrix = np.column_stack([
        shap_values[:, base_to_indices[feature]].sum(axis=1)
        for feature in grouped_features
    ])
    np.save(SHAP_DIR / "grouped_shap_values.npy", grouped_matrix)
    np.save(SHAP_DIR / "grouped_feature_names.npy", np.asarray(grouped_features, dtype=object))

    grouped_importance = pd.DataFrame({
        "base_feature": grouped_features,
        "clinical_feature": [clinical_label(feature) for feature in grouped_features],
        "domain": [feature_to_domain.get(feature, "Unknown") for feature in grouped_features],
        "mean_abs_grouped_shap": np.mean(np.abs(grouped_matrix), axis=0),
        "mean_signed_grouped_shap": np.mean(grouped_matrix, axis=0),
    }).sort_values("mean_abs_grouped_shap", ascending=False).reset_index(drop=True)
    grouped_importance.insert(0, "rank", np.arange(1, len(grouped_importance) + 1))
    grouped_importance["relative_importance"] = (
        grouped_importance["mean_abs_grouped_shap"]
        / grouped_importance["mean_abs_grouped_shap"].sum()
    )
    grouped_importance.to_csv(
        SHAP_DIR / "shap_grouped_clinical_feature_importance.csv", index=False
    )

    # Patient-level summed domain contributions.
    grouped_index = {feature: i for i, feature in enumerate(grouped_features)}
    domain_rows = []
    domain_matrices = []
    domain_names = []
    for domain in DOMAIN_PREFIXES:
        features = grouped_importance.loc[grouped_importance["domain"].eq(domain), "base_feature"].tolist()
        if not features:
            continue
        indices = [grouped_index[feature] for feature in features]
        domain_values = grouped_matrix[:, indices].sum(axis=1)
        domain_names.append(domain)
        domain_matrices.append(domain_values)
        domain_rows.append({
            "domain": domain,
            "n_original_features": len(features),
            "mean_abs_domain_shap": float(np.mean(np.abs(domain_values))),
            "mean_signed_domain_shap": float(np.mean(domain_values)),
        })

    domain_matrix = np.column_stack(domain_matrices)
    np.save(SHAP_DIR / "domain_shap_values.npy", domain_matrix)
    np.save(SHAP_DIR / "domain_names.npy", np.asarray(domain_names, dtype=object))
    domain_importance = pd.DataFrame(domain_rows).sort_values(
        "mean_abs_domain_shap", ascending=False
    ).reset_index(drop=True)
    domain_importance["relative_importance"] = (
        domain_importance["mean_abs_domain_shap"]
        / domain_importance["mean_abs_domain_shap"].sum()
    )
    domain_importance["relative_importance_pct"] = 100 * domain_importance["relative_importance"]
    domain_importance.to_csv(SHAP_DIR / "domain_shap_importance.csv", index=False)

    # Compact grouped importance figure.
    configure_style()
    top = grouped_importance.head(20).sort_values("mean_abs_grouped_shap")
    cmap = mpl.colormaps[CMAP_NAME]
    colors = [cmap(value) for value in np.linspace(0.15, 0.85, len(top))]
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.barh(top["clinical_feature"], top["mean_abs_grouped_shap"], color=colors)
    ax.set_xlabel("Mean absolute grouped SHAP value")
    ax.set_title("Top Grouped Clinical Features", loc="left", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.20)
    fig.tight_layout()
    output = FIG_DIR / "Figure_Grouped_SHAP_Top20.png"
    save_figure_all_formats(fig, output)
    plt.show()
    plt.close(fig)

    return {
        "available": True,
        "sample_n": sample_n,
        "transformed_features": len(transformed_names),
        "grouped_features": len(grouped_features),
        "grouped_table": SHAP_DIR / "shap_grouped_clinical_feature_importance.csv",
        "domain_table": SHAP_DIR / "domain_shap_importance.csv",
        "figure": output,
    }


# =============================================================================
# DOMAIN ABLATION
# =============================================================================

def run_domain_ablation(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_seqn: np.ndarray,
    full_probability: np.ndarray,
    feature_domains: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    header("RUNNING FIXED-SPLIT DOMAIN ABLATION")
    full_auroc = roc_auc_score(y_test, full_probability)
    full_auprc = average_precision_score(y_test, full_probability)
    rows = [{
        "Experiment": "All_Domains",
        "Removed_Domain": "",
        "N_Features": X_development.shape[1],
        "AUROC": full_auroc,
        "AUPRC": full_auprc,
        "AUROC_Drop": 0.0,
        "AUPRC_Drop": 0.0,
        "AUROC_Drop_CI_Lower": 0.0,
        "AUROC_Drop_CI_Upper": 0.0,
        "AUPRC_Drop_CI_Lower": 0.0,
        "AUPRC_Drop_CI_Upper": 0.0,
    }]

    for domain, domain_features in feature_domains.items():
        removable = set(feature for feature in domain_features if feature in X_development.columns)
        if not removable:
            continue
        retained = [
            feature
            for feature in X_development.columns
            if feature not in removable
        ]

        # Re-identify types for the reduced feature set, then immediately recast
        # both matrices. This is essential because a variable that was numeric
        # in the full cohort can become low-cardinality in the development
        # subset and therefore be reclassified as categorical. Without recasting,
        # SimpleImputer adds the string "Missing" to float category values and
        # OneHotEncoder receives mixed float/string input.
        numeric, categorical = identify_feature_types(
            X_development[
                retained
            ]
        )

        X_development_reduced = cast_matrix(
            X_development[
                retained
            ],
            numeric,
            categorical,
        )

        X_test_reduced = cast_matrix(
            X_test[
                retained
            ],
            numeric,
            categorical,
        )

        # Defensive audit: every categorical column must contain only strings
        # and missing values before entering the preprocessing pipeline.
        mixed_type_rows = []

        for feature in categorical:
            observed_types = sorted(
                {
                    type(value).__name__
                    for value
                    in X_development_reduced[
                        feature
                    ].dropna()
                }
            )

            if observed_types not in (
                [],
                [
                    "str"
                ],
            ):
                mixed_type_rows.append(
                    {
                        "Removed_Domain": domain,
                        "Feature": feature,
                        "Observed_Types": ",".join(
                            observed_types
                        ),
                    }
                )

        if mixed_type_rows:
            mixed_type_table = pd.DataFrame(
                mixed_type_rows
            )

            mixed_type_table.to_csv(
                ABLATION_DIR
                / (
                    "domain_ablation_"
                    "categorical_type_failure.csv"
                ),
                index=False,
            )

            raise TypeError(
                "Mixed categorical data types remain after recasting. "
                "See domain_ablation_categorical_type_failure.csv."
            )

        model = build_pipeline(
            "XGBoost",
            numeric,
            categorical,
            y_development,
            RANDOM_STATE
            + 100,
        )

        model.fit(
            X_development_reduced,
            y_development,
        )

        probability = model.predict_proba(
            X_test_reduced
        )[
            :,
            1,
        ]
        auroc = roc_auc_score(y_test, probability)
        auprc = average_precision_score(y_test, probability)

        rng = np.random.default_rng(RANDOM_STATE)
        auroc_drop_boot, auprc_drop_boot = [], []
        for _ in range(N_ABLATION_BOOTSTRAP):
            index = stratified_bootstrap_indices(y_test.to_numpy(), rng)
            y = y_test.to_numpy()[index]
            full = full_probability[index]
            reduced = probability[index]
            auroc_drop_boot.append(roc_auc_score(y, full) - roc_auc_score(y, reduced))
            auprc_drop_boot.append(
                average_precision_score(y, full) - average_precision_score(y, reduced)
            )

        row = {
            "Experiment": f"Remove_{domain}",
            "Removed_Domain": domain,
            "N_Features": len(retained),
            "AUROC": auroc,
            "AUPRC": auprc,
            "AUROC_Drop": full_auroc - auroc,
            "AUPRC_Drop": full_auprc - auprc,
            "AUROC_Drop_CI_Lower": float(np.percentile(auroc_drop_boot, 2.5)),
            "AUROC_Drop_CI_Upper": float(np.percentile(auroc_drop_boot, 97.5)),
            "AUPRC_Drop_CI_Lower": float(np.percentile(auprc_drop_boot, 2.5)),
            "AUPRC_Drop_CI_Upper": float(np.percentile(auprc_drop_boot, 97.5)),
        }
        rows.append(row)
        print(
            f"{domain}: AUROC drop={row['AUROC_Drop']:.4f}; "
            f"AUPRC drop={row['AUPRC_Drop']:.4f}"
        )

        pd.DataFrame({
            "SEQN": np.asarray(test_seqn),
            "y_true": y_test.to_numpy(),
            "full_probability": full_probability,
            "ablated_probability": probability,
        }).to_csv(
            ABLATION_DIR / f"predictions_remove_{re.sub(r'[^A-Za-z0-9]+', '_', domain)}.csv",
            index=False,
        )
        if JOBLIB_AVAILABLE:
            joblib.dump(
                model,
                ABLATION_DIR / f"xgboost_remove_{re.sub(r'[^A-Za-z0-9]+', '_', domain)}.joblib",
            )

    table = pd.DataFrame(rows)
    table.to_csv(ABLATION_DIR / "domain_ablation_results.csv", index=False)
    return table


# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

def sex_label(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map({1.0: "Male", 2.0: "Female"}).fillna("Missing/Other")


def race_label(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map({
        1.0: "Mexican American",
        2.0: "Other Hispanic",
        3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black",
        6.0: "Non-Hispanic Asian",
        7.0: "Other/Multiracial",
    }).fillna("Missing/Other")


def age_group(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    output = pd.Series("Missing", index=series.index, dtype=object)
    output.loc[age.ge(18) & age.lt(40)] = "18-39"
    output.loc[age.ge(40) & age.lt(60)] = "40-59"
    output.loc[age.ge(60) & age.lt(70)] = "60-69"
    output.loc[age.ge(70)] = "70+"
    return output


def subgroup_ci(y_true: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for _ in range(N_SUBGROUP_BOOTSTRAP):
        index = stratified_bootstrap_indices(y_true, rng)
        y = y_true[index]
        p = probability[index]
        rows.append([
            roc_auc_score(y, p),
            average_precision_score(y, p),
            brier_score_loss(y, p),
        ])
    array = np.asarray(rows)
    return {
        "AUROC_CI_Lower": float(np.percentile(array[:, 0], 2.5)),
        "AUROC_CI_Upper": float(np.percentile(array[:, 0], 97.5)),
        "AUPRC_CI_Lower": float(np.percentile(array[:, 1], 2.5)),
        "AUPRC_CI_Upper": float(np.percentile(array[:, 1], 97.5)),
        "Brier_CI_Lower": float(np.percentile(array[:, 2], 2.5)),
        "Brier_CI_Upper": float(np.percentile(array[:, 2], 97.5)),
    }


def run_subgroups(
    metadata: pd.DataFrame,
    y_test: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    header("RUNNING LOCKED-TEST SUBGROUP ANALYSIS")
    frame = metadata.copy()
    frame["y_true"] = y_test.to_numpy()
    frame["probability"] = probability
    frame["Age_Group"] = age_group(frame["RIDAGEYR"])
    frame["Sex"] = sex_label(frame["RIAGENDR"])
    frame["Race_Ethnicity"] = race_label(frame["RIDRETH3"])

    rows = []
    for subgroup_type in ["Age_Group", "Sex", "Race_Ethnicity"]:
        for subgroup, group in frame.groupby(subgroup_type, dropna=False):
            y = group["y_true"].to_numpy(dtype=int)
            p = group["probability"].to_numpy(dtype=float)
            n = len(group)
            events = int(y.sum())
            if n < 50 or events < 10 or (n - events) < 10:
                rows.append({
                    "Subgroup_Type": subgroup_type,
                    "Subgroup": str(subgroup),
                    "N": n,
                    "Events": events,
                    "Status": "Insufficient events/non-events",
                })
                continue
            metrics = all_metrics(y, p, threshold)
            rows.append({
                "Subgroup_Type": subgroup_type,
                "Subgroup": str(subgroup),
                "N": n,
                "Events": events,
                "Prevalence": float(y.mean()),
                "Status": "Estimated",
                **metrics,
                **subgroup_ci(y, p),
            })

    table = pd.DataFrame(rows)
    table.to_csv(SUBGROUP_DIR / "subgroup_performance.csv", index=False)
    return table


# =============================================================================
# COHORT FLOW FIGURE
# =============================================================================

def cohort_flow_figure(
    flow: pd.DataFrame,
    development_n: int,
    development_events: int,
    test_n: int,
    test_events: int,
) -> Path:
    configure_style()
    cmap = mpl.colormaps[CMAP_NAME]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    boxes = [(row.Description, int(row.N), int(row.Events)) for row in flow.itertuples()]
    boxes += [
        ("Development cohort", development_n, development_events),
        ("Locked held-out test cohort", test_n, test_events),
    ]
    y_positions = [0.86, 0.66, 0.46, 0.20, 0.20]
    x_positions = [0.50, 0.50, 0.50, 0.30, 0.70]

    for index, ((description, n, events), x, y) in enumerate(zip(boxes, x_positions, y_positions)):
        ax.text(
            x, y,
            f"{description}\nN={n:,}; mobility disability={events:,}",
            ha="center", va="center", fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": cmap(0.15 + 0.65 * index / max(len(boxes) - 1, 1)),
                "edgecolor": "black",
                "alpha": 0.90,
            },
        )

    # Vertical arrows then split.
    for y_start, y_end in [(0.80, 0.72), (0.60, 0.52)]:
        ax.annotate("", xy=(0.50, y_end), xytext=(0.50, y_start),
                    arrowprops={"arrowstyle": "->", "linewidth": 1.4})
    ax.plot([0.50, 0.50], [0.40, 0.34], color="black")
    ax.plot([0.30, 0.70], [0.34, 0.34], color="black")
    ax.plot([0.30, 0.30], [0.34, 0.27], color="black")
    ax.plot([0.70, 0.70], [0.34, 0.27], color="black")

    ax.set_title("Cohort Construction and Analytical Design", fontsize=16, fontweight="bold", pad=16)
    output = FIG_DIR / "Figure1_Cohort_Flow.png"
    save_figure_all_formats(fig, output)
    plt.show()
    plt.close(fig)
    return output


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    configure_style()
    header("NHANES 2017-2018 ADULT MOBILITY-DISABILITY CLASSIFICATION")
    print("Data:", DATA_DIR)
    print("Output:", OUT_DIR)

    # Load and cohort.
    modules = load_modules(DATA_DIR)
    merged = merge_modules(modules)
    print("\nMerged shape:", merged.shape)

    merged, xport_audit = replace_xport_tiny_artifacts(merged)
    xport_audit.to_csv(AUDIT_DIR / "xport_tiny_missing_artifacts.csv", index=False)
    merged = derive_variables(merged)
    cohort, flow = construct_adult_cohort(merged)
    flow.to_csv(TABLE_DIR / "cohort_flow.csv", index=False)
    y = cohort[OUTCOME].astype(int)

    header("PRIMARY ADULT COHORT")
    print(f"N={len(cohort):,}")
    print(f"Mobility disability={int(y.sum()):,}")
    print(f"Prevalence={100 * y.mean():.2f}%")

    # Features.
    candidates, initial_audit = build_candidate_features(cohort)
    initial_audit.to_csv(AUDIT_DIR / "initial_feature_selection.csv", index=False)
    X_candidate = decode_objects(cohort[candidates].copy())
    X_candidate, special_audit, unresolved = clean_special_codes(X_candidate)
    special_audit.to_csv(AUDIT_DIR / "special_code_cleaning.csv", index=False)
    pd.DataFrame({"feature": unresolved}).to_csv(
        AUDIT_DIR / "unresolved_special_code_features_excluded.csv", index=False
    )

    final_features, quality_audit = quality_filter(X_candidate, candidates, unresolved)
    quality_audit.to_csv(AUDIT_DIR / "feature_quality_filter.csv", index=False)
    if not final_features:
        raise RuntimeError("No features remain")

    X = X_candidate[final_features].copy()
    numeric_features, categorical_features = identify_feature_types(X)
    X = cast_matrix(X, numeric_features, categorical_features)

    feature_to_domain = {feature: assign_domain(feature) for feature in final_features}
    feature_domains = {
        domain: [feature for feature in final_features if feature_to_domain[feature] == domain]
        for domain in DOMAIN_PREFIXES
    }
    save_json(feature_domains, OUT_DIR / "feature_domains.json")
    pd.DataFrame({
        "feature": final_features,
        "domain": [feature_to_domain[feature] for feature in final_features],
        "feature_type": [
            "numeric" if feature in numeric_features else "categorical"
            for feature in final_features
        ],
        "missing_fraction": [float(X[feature].isna().mean()) for feature in final_features],
    }).to_csv(TABLE_DIR / "final_feature_list.csv", index=False)

    header("FINAL FEATURE SET")
    for domain, features in feature_domains.items():
        print(f"{domain:22s}: {len(features)}")
    print("Total:", len(final_features))
    print("Numeric:", len(numeric_features), "Categorical:", len(categorical_features))

    # Fixed split.
    positions = np.arange(len(cohort))
    development_positions, test_positions = train_test_split(
        positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    development_positions = np.asarray(development_positions, dtype=int)
    test_positions = np.asarray(test_positions, dtype=int)
    X_development = X.iloc[development_positions].copy()
    X_test = X.iloc[test_positions].copy()
    y_development = y.iloc[development_positions].copy()
    y_test = y.iloc[test_positions].copy()
    development_metadata = cohort.iloc[development_positions].copy()
    test_metadata = cohort.iloc[test_positions].copy()

    pd.DataFrame({
        "SEQN": development_metadata["SEQN"].to_numpy(),
        "cohort_row_position": development_positions,
        "y_true": y_development.to_numpy(),
        "partition": "development",
    }).to_csv(TABLE_DIR / "development_partition.csv", index=False)
    pd.DataFrame({
        "SEQN": test_metadata["SEQN"].to_numpy(),
        "cohort_row_position": test_positions,
        "y_true": y_test.to_numpy(),
        "partition": "locked_test",
    }).to_csv(TABLE_DIR / "test_partition.csv", index=False)

    header("LOCKED SPLIT")
    print(f"Development: {len(X_development):,} (events={int(y_development.sum()):,})")
    print(f"Test:        {len(X_test):,} (events={int(y_test.sum()):,})")

    # Model development.
    model_results: Dict[str, Dict[str, Any]] = {}
    fold_tables = []
    oof_table = pd.DataFrame({
        "SEQN": development_metadata["SEQN"].to_numpy(),
        "y_true": y_development.to_numpy(),
    })
    test_predictions = pd.DataFrame({
        "SEQN": test_metadata["SEQN"].to_numpy(),
        "y_true_mobility_disability": y_test.to_numpy(),
    })

    for model_name in MODEL_NAMES:
        result = train_model(
            model_name,
            X_development,
            y_development,
            X_test,
            numeric_features,
            categorical_features,
        )
        model_results[model_name] = result
        fold_tables.append(result["fold_metrics"])
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
        oof_table[f"{safe_name}_fold"] = result["fold_assignment"]
        oof_table[f"{safe_name}_raw_probability"] = result["oof_raw"]
        oof_table[f"{safe_name}_calibrated_probability"] = result["oof_calibrated"]
        test_predictions[f"{safe_name}_raw_probability"] = result["test_raw"]
        test_predictions[f"{safe_name}_calibrated_probability"] = result["test_calibrated"]
        test_predictions[f"{safe_name}_locked_prediction"] = (
            result["test_calibrated"] >= result["locked_threshold"]
        ).astype(int)
        if JOBLIB_AVAILABLE:
            joblib.dump(result["pipeline"], MODEL_DIR / f"{safe_name}_final_pipeline.joblib")
            joblib.dump(result["calibrator"], MODEL_DIR / f"{safe_name}_platt_calibrator.joblib")

    pd.concat(fold_tables, ignore_index=True).to_csv(
        TABLE_DIR / "development_cv_fold_metrics.csv", index=False
    )
    oof_table.to_csv(TABLE_DIR / "development_oof_predictions.csv", index=False)
    test_predictions.to_csv(TABLE_DIR / "test_predictions_all_models.csv", index=False)

    # Locked-test performance.
    weight_column = next((c for c in SURVEY_WEIGHT_CANDIDATES if c in test_metadata), None)
    weights = (
        pd.to_numeric(test_metadata[weight_column], errors="coerce").to_numpy(dtype=float)
        if weight_column else None
    )

    performance_rows = []
    ci_tables = []
    bootstrap_tables = []
    for model_name in MODEL_NAMES:
        result = model_results[model_name]
        p = result["test_calibrated"]
        metrics = all_metrics(y_test.to_numpy(), p, result["locked_threshold"])
        weighted = weighted_probability_metrics(y_test.to_numpy(), p, weights) if weights is not None else {}
        performance_rows.append({
            "Model": model_name,
            "Probability_Type": "Platt calibrated",
            "Threshold_Source": "Development OOF Youden",
            **metrics,
            **weighted,
        })
        boot = bootstrap_metrics(
            y_test.to_numpy(), p, result["locked_threshold"], N_BOOTSTRAP, RANDOM_STATE
        )
        boot["Model"] = model_name
        bootstrap_tables.append(boot)
        ci = summarize_bootstrap(metrics, boot)
        ci["Model"] = model_name
        ci_tables.append(ci)

    # Constant prevalence-only reference model. Because all participants
    # receive the same probability, its calibration slope is not identifiable
    # and is reported as NaN. Its binned calibration error reduces to the
    # absolute observed-versus-predicted prevalence difference.
    prevalence_probability = np.full(
        len(
            y_test
        ),
        y_development.mean(),
        dtype=float,
    )

    performance_rows.append(
        {
            "Model": "Prevalence Baseline",
            "Probability_Type": "Development prevalence",
            "Threshold_Source": "0.5",
            **all_metrics(
                y_test.to_numpy(),
                prevalence_probability,
                0.5,
            ),
        }
    )

    performance_table = pd.DataFrame(performance_rows)
    performance_ci = pd.concat(ci_tables, ignore_index=True)
    performance_table.to_csv(TABLE_DIR / "test_model_performance.csv", index=False)
    performance_ci.to_csv(TABLE_DIR / "test_model_performance_95ci.csv", index=False)
    pd.concat(bootstrap_tables, ignore_index=True).to_csv(
        TABLE_DIR / "test_model_bootstrap_distributions.csv", index=False
    )
    pd.DataFrame([
        {
            "Model": model_name,
            "Locked_Threshold": model_results[model_name]["locked_threshold"],
            "Threshold_Source": "Development 5-fold OOF calibrated probabilities",
        }
        for model_name in MODEL_NAMES
    ]).to_csv(TABLE_DIR / "locked_thresholds.csv", index=False)

    # Paired differences: XGBoost minus comparator.
    paired_summary = []
    paired_raw = []
    xgb_probability = model_results["XGBoost"]["test_calibrated"]
    for comparator in ["Logistic Regression", "Random Forest"]:
        comparator_probability = model_results[comparator]["test_calibrated"]
        boot = paired_bootstrap(
            y_test.to_numpy(), xgb_probability, comparator_probability, N_BOOTSTRAP, RANDOM_STATE
        )
        comparison = f"XGBoost minus {comparator}"
        boot["Comparison"] = comparison
        paired_raw.append(boot)
        point = {
            "AUROC": roc_auc_score(y_test, xgb_probability) - roc_auc_score(y_test, comparator_probability),
            "AUPRC": average_precision_score(y_test, xgb_probability) - average_precision_score(y_test, comparator_probability),
            "Brier": brier_score_loss(y_test, xgb_probability) - brier_score_loss(y_test, comparator_probability),
        }
        for metric, column in [
            ("AUROC", "AUROC_Difference"),
            ("AUPRC", "AUPRC_Difference"),
            ("Brier", "Brier_Difference"),
        ]:
            values = boot[column].to_numpy(dtype=float)
            paired_summary.append({
                "Comparison": comparison,
                "Metric": metric,
                "Difference": point[metric],
                "CI_95_Lower": float(np.percentile(values, 2.5)),
                "CI_95_Upper": float(np.percentile(values, 97.5)),
                "Interpretation": (
                    "Positive favors XGBoost" if metric in {"AUROC", "AUPRC"}
                    else "Negative favors XGBoost"
                ),
            })
    pd.DataFrame(paired_summary).to_csv(TABLE_DIR / "paired_model_differences.csv", index=False)
    pd.concat(paired_raw, ignore_index=True).to_csv(
        TABLE_DIR / "paired_model_difference_bootstrap.csv", index=False
    )

    # Figures and interpretation.
    model_figure = performance_figure(y_test.to_numpy(), model_results, performance_ci)
    shap_summary = run_shap(
        model_results["XGBoost"]["pipeline"],
        X_test,
        test_metadata["SEQN"].to_numpy(),
        final_features,
        feature_to_domain,
    )
    ablation_table = run_domain_ablation(
        X_development,
        y_development,
        X_test,
        y_test,
        test_metadata["SEQN"].to_numpy(),
        model_results["XGBoost"]["test_raw"],
        feature_domains,
    )
    subgroup_table = run_subgroups(
        test_metadata,
        y_test,
        model_results["XGBoost"]["test_calibrated"],
        model_results["XGBoost"]["locked_threshold"],
    )
    flow_figure = cohort_flow_figure(
        flow,
        len(X_development),
        int(y_development.sum()),
        len(X_test),
        int(y_test.sum()),
    )

    # Missingness by outcome.
    pd.DataFrame({
        "feature": final_features,
        "domain": [feature_to_domain[feature] for feature in final_features],
        "missing_fraction_overall": [float(X[feature].isna().mean()) for feature in final_features],
        "missing_fraction_no_disability": [
            float(X.loc[y.eq(0), feature].isna().mean()) for feature in final_features
        ],
        "missing_fraction_mobility_disability": [
            float(X.loc[y.eq(1), feature].isna().mean()) for feature in final_features
        ],
    }).sort_values("missing_fraction_overall", ascending=False).to_csv(
        TABLE_DIR / "missingness_by_outcome.csv", index=False
    )

    # Configuration and summary.
    summary = {
        "cycle": "NHANES 2017-2018",
        "analysis_type": "Concurrent adult mobility-disability classification",
        "minimum_age": MINIMUM_AGE,
        "merged_rows": len(merged),
        "adult_cohort_rows": len(cohort),
        "mobility_disability_cases": int(y.sum()),
        "outcome_prevalence": float(y.mean()),
        "development_rows": len(X_development),
        "development_events": int(y_development.sum()),
        "test_rows": len(X_test),
        "test_events": int(y_test.sum()),
        "original_feature_count": len(final_features),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "all_DLQ_predictors_excluded": True,
        "survey_design_predictors_excluded": True,
        "administrative_predictors_excluded": True,
        "test_used_for_threshold_selection": False,
        "test_used_for_calibration": False,
        "threshold_source": "Development 5-fold OOF calibrated probabilities",
        "calibration_method": "Platt scaling from development OOF probabilities",
        "secondary_survey_weight_column": weight_column,
        "data_dir": DATA_DIR,
        "out_dir": OUT_DIR,
        "cohort_flow_figure": flow_figure,
        "model_performance_figure": model_figure,
        "shap_available": shap_summary.get("available", False),
    }
    save_json(summary, OUT_DIR / "cohort_summary.json")
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "cohort_summary.csv", index=False)
    save_json({
        "MINIMUM_AGE": MINIMUM_AGE,
        "TEST_SIZE": TEST_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "N_FOLDS": N_FOLDS,
        "N_BOOTSTRAP": N_BOOTSTRAP,
        "N_ABLATION_BOOTSTRAP": N_ABLATION_BOOTSTRAP,
        "N_SUBGROUP_BOOTSTRAP": N_SUBGROUP_BOOTSTRAP,
        "SHAP_SAMPLE_N": SHAP_SAMPLE_N,
        "MODEL_NAMES": MODEL_NAMES,
    }, OUT_DIR / "config.json")

    header("PUBLICATION-REVISED PIPELINE COMPLETE")
    print("Output:", OUT_DIR)
    print("\nLocked-test performance:")
    print(performance_table[[
        "Model", "AUROC", "AUPRC", "Brier",
        "Calibration_Intercept", "Calibration_Slope",
        "Threshold", "Recall_Sensitivity", "Specificity",
    ]].round(4).to_string(index=False))
    print("\nDomain ablation:")
    print(ablation_table[[
        "Removed_Domain", "AUROC_Drop", "AUROC_Drop_CI_Lower", "AUROC_Drop_CI_Upper",
        "AUPRC_Drop", "AUPRC_Drop_CI_Lower", "AUPRC_Drop_CI_Upper",
    ]].round(4).to_string(index=False))
    print("\nSubgroup rows:", len(subgroup_table))
    print("\nInterpretation: concurrent classification/identification, not prospective prediction.")


if __name__ == "__main__":
    main()
