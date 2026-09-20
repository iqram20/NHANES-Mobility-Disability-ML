#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NHANES 2017–2018 ADULT MOBILITY-DISABILITY CLASSIFICATION
NATIVE TREESHAP GROUPED CLINICAL SUMMARY — FULL INTERNAL VALIDATION
==========================================================

Current publication-revised run
-------------------------------
/athena/madelab/scratch/iqh4001/Disability/Results/
20260727_004939_NHANES_2017_2018_Adult_
Mobility_Disability_Classification_Revised

This is a standalone post-run plotting script. It does not retrain the model
and does not recompute SHAP.

Current-run inputs
------------------
shap/grouped_shap_values.npy
shap/grouped_feature_names.npy
shap/shap_sample_original_feature_values.csv
tables/final_feature_list.csv

The revised pipeline already grouped transformed one-hot columns into their
original NHANES clinical variables by summing participant-level SHAP values.
This script therefore plots those saved grouped contributions directly.

Figure interpretation
---------------------
- Each dot represents one internal-validation participant.
- Horizontal position is the grouped SHAP contribution.
- Positive SHAP values increase the XGBoost model output for mobility
  disability; negative values decrease it.
- Plasma colors represent low-to-high original feature values.
- Gray dots indicate missing original feature values.
- Binary Yes/No variables are shown as No=0 and Yes=1.
- Multicategory variables are converted to stable numeric category codes only
  for within-feature coloring; colors should not be interpreted as a universal
  ordering across different features.

Outputs
-------
Figures
- figures_revised_manuscript/shap_summary/
  Figure_Grouped_Clinical_SHAP_Summary_Current_Run.png
- PDF, SVG, and TIFF versions

Tables
- shap/Figure_Grouped_SHAP_Clinical_Importance_Current_Run.csv
- shap/Figure_Grouped_SHAP_Top20_Current_Run.csv
- shap/Figure_Grouped_SHAP_Color_Value_Audit_Current_Run.csv

Jupyter
-------
%matplotlib inline
%run Figure_current_run_grouped_SHAP_summary.py

Use %run rather than !python when inline display is required.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CURRENT-RUN PATHS
# =============================================================================

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/Disability/Results/"
    "20260920_115949_NHANES_2017_2018_Adult_"
    "Mobility_Disability_Classification_Revised_XPORTZeroFixed"
)

TABLE_DIR = RUN_DIR / "tables"
SHAP_DIR = RUN_DIR / "shap"

FIG_DIR = (
    RUN_DIR
    / "figures_reviewer_revision_native_treeshap"
    / "shap_summary"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GROUPED_SHAP_FILE = (
    SHAP_DIR
    / "grouped_shap_values.npy"
)

GROUPED_FEATURE_NAMES_FILE = (
    SHAP_DIR
    / "grouped_feature_names.npy"
)

ORIGINAL_SAMPLE_VALUES_FILE = (
    SHAP_DIR
    / "shap_sample_original_feature_values.csv"
)

FINAL_FEATURE_FILE = (
    TABLE_DIR
    / "final_feature_list.csv"
)

PIPELINE_GROUPED_IMPORTANCE_FILE = (
    SHAP_DIR
    / "shap_grouped_clinical_feature_importance.csv"
)

OUTPUT_BASENAME = (
    "Figure3_Grouped_Clinical_SHAP_NativeTreeSHAP_FULL1171"
)

OUTPUT_PNG = (
    FIG_DIR
    / f"{OUTPUT_BASENAME}.png"
)

OUTPUT_PDF = (
    FIG_DIR
    / f"{OUTPUT_BASENAME}.pdf"
)

OUTPUT_SVG = (
    FIG_DIR
    / f"{OUTPUT_BASENAME}.svg"
)

OUTPUT_TIFF = (
    FIG_DIR
    / f"{OUTPUT_BASENAME}.tiff"
)

IMPORTANCE_OUTPUT = (
    SHAP_DIR
    / "Figure_Grouped_SHAP_Clinical_Importance_Current_Run.csv"
)

TOP20_OUTPUT = (
    SHAP_DIR
    / "Figure_Grouped_SHAP_Top20_Current_Run.csv"
)

COLOR_AUDIT_OUTPUT = (
    SHAP_DIR
    / "Figure_Grouped_SHAP_Color_Value_Audit_Current_Run.csv"
)

SUMMARY_OUTPUT = (
    FIG_DIR
    / "Figure_Grouped_Clinical_SHAP_Summary_Current_Run.json"
)


# =============================================================================
# 2. FIGURE SETTINGS
# =============================================================================

TOP_K = 20
CMAP_NAME = "plasma"
RANDOM_STATE = 42

DISPLAY_CLIP_PERCENTILE = 99.5
BEESWARM_JITTER = 0.50
BEESWARM_DOT_SIZE = 10
BEESWARM_ALPHA = 0.72

FIGURE_WIDTH = 15.0
FIGURE_HEIGHT = 11.0
OUTPUT_DPI = 600

MISSING_POINT_COLOR = "lightgray"


# =============================================================================
# 3. CLINICAL FEATURE LABELS
# =============================================================================

FEATURE_LABEL_MAP: Dict[str, str] = {
    "PAQ670": "Days of Moderate Recreational Activity",
    "DPQ080": "Psychomotor Symptoms",
    "DPQ030": "Sleep-Related Depressive Symptom",
    "DR1TATOA": "Day 1 Added Vitamin E",

    # Demographics
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "RIDRETH1": "Race/Ethnicity",
    "RIDRETH3": "Race/Ethnicity",
    "DMDEDUC2": "Education",
    "DMDMARTL": "Marital Status",
    "INDHHIN2": "Household Income",
    "INDFMPIR": "Income-to-Poverty Ratio",
    "DMDHHSIZ": "Household Size",
    "DMDFMSIZ": "Family Size",
    "DMDHHSZA": "Children Aged ≤5 in Household",
    "DMDHHSZB": "Children Aged 6–17 in Household",
    "DMDHHSZE": "Number of Adults Aged ≥60 in Household",

    # Body measures
    "BMXBMI": "Body Mass Index",
    "BMXWT": "Weight",
    "BMXHT": "Height",
    "BMXWAIST": "Waist Circumference",
    "BMXHIP": "Hip Circumference",
    "BMXARMC": "Arm Circumference",
    "BMXARML": "Upper Arm Length",
    "BMXLEG": "Upper Leg Length",

    # Blood pressure and pulse
    "SBP_MEAN": "Mean Systolic Blood Pressure",
    "DBP_MEAN": "Mean Diastolic Blood Pressure",
    "BPXPLS": "Pulse Rate",

    # Mental health
    "PHQ9_TOTAL": "PHQ-9 Depression Score",
    "DPQ010": "Little Interest or Pleasure",
    "DPQ020": "Feeling Depressed or Hopeless",
    "DPQ030": "Sleep Difficulty",
    "DPQ040": "Low Energy",
    "DPQ050": "Poor Appetite or Overeating",
    "DPQ060": "Feeling Bad About Self",
    "DPQ070": "Concentration Difficulty",
    "DPQ080": "Psychomotor Change",
    "DPQ090": "Thoughts of Self-Harm",
    "DPQ100": "Depression-Related Functional Difficulty",

    # Medical conditions
    "MCQ010": "Asthma",
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

    # Smoking and alcohol
    "SMQ020": "Ever Smoked 100 Cigarettes",
    "SMQ040": "Current Smoking",
    "ALQ101": "Alcohol Use",
    "ALQ130": "Average Drinks per Drinking Day",
    "ALQ142": "Binge Drinking Frequency",

    # Physical activity
    "PAQ605": "Vigorous Work Activity",
    "PAQ620": "Moderate Work Activity",
    "PAQ635": "Walking or Bicycling",
    "PAQ650": "Vigorous Recreational Activity",
    "PAQ665": "Moderate Recreational Activity",
    "PAD675": "Moderate Recreational Activity (min/day)",
    "PAD680": "Sedentary Time",

    # Sleep
    "SLD012": "Weekday Sleep Duration",
    "SLD013": "Weekend Sleep Duration",
    "SLQ030": "Snoring Frequency",
    "SLQ040": "Snorting, Gasping, or Breathing Pauses",
    "SLQ050": "Told Doctor About Sleep Trouble",
    "SLQ120": "Daytime Sleepiness",

    # Laboratory
    "LBDNENO": "Segmented Neutrophil Count (10³ cells/µL)",
    "LBXGH": "HbA1c",
    "LBXGLU": "Fasting Glucose",
    "LBDHDD": "HDL Cholesterol",
    "LBXTC": "Total Cholesterol",
    "LBXWBCSI": "White Blood Cell Count",
    "LBXRBCSI": "Red Blood Cell Count",
    "LBXHGB": "Hemoglobin",
    "LBXHCT": "Hematocrit",
    "LBXMCVSI": "Mean Corpuscular Volume",
    "LBXMCHSI": "Mean Corpuscular Hemoglobin",
    "LBXMC": "Mean Corpuscular Hemoglobin Concentration",
    "LBXRDW": "Red Cell Distribution Width",
    "LBXPLTSI": "Platelet Count",
    "LBXMPSI": "Mean Platelet Volume",
    "LBXNEPCT": "Neutrophil Percentage",
    "LBXLYPCT": "Lymphocyte Percentage",
    "LBXMOPCT": "Monocyte Percentage",
    "LBXEOPCT": "Eosinophil Percentage",
    "LBXBAPCT": "Basophil Percentage",

    # Dietary totals and nutrients
    "DR1TATOA": "Day 1 Added Vitamin E Intake (mg)",
    "DR1TNUMF": "Day 1 Number of Foods/Beverages",
    "DR2TATOC": "Day 2 Vitamin E Intake (mg)",
    "DR2TNUMF": "Day 2 Number of Foods/Beverages",
    "DR1TKCAL": "Day 1 Energy Intake",
    "DR2TKCAL": "Day 2 Energy Intake",
    "DR1TPROT": "Day 1 Protein Intake",
    "DR2TPROT": "Day 2 Protein Intake",
    "DR1TCARB": "Day 1 Carbohydrate Intake",
    "DR2TCARB": "Day 2 Carbohydrate Intake",
    "DR1TSUGR": "Day 1 Sugar Intake",
    "DR2TSUGR": "Day 2 Sugar Intake",
    "DR1TFIBE": "Day 1 Fiber Intake",
    "DR2TFIBE": "Day 2 Fiber Intake",
    "DR1TTFAT": "Day 1 Total Fat Intake",
    "DR2TTFAT": "Day 2 Total Fat Intake",
    "DR1TSFAT": "Day 1 Saturated Fat Intake",
    "DR2TSFAT": "Day 2 Saturated Fat Intake",
    "DR1TCHOL": "Day 1 Cholesterol Intake",
    "DR2TCHOL": "Day 2 Cholesterol Intake",
    "DR1TSODI": "Day 1 Sodium Intake",
    "DR2TSODI": "Day 2 Sodium Intake",
    "DR1TPOTA": "Day 1 Potassium Intake",
    "DR2TPOTA": "Day 2 Potassium Intake",
    "DR1TCALC": "Day 1 Calcium Intake",
    "DR2TCALC": "Day 2 Calcium Intake",
    "DR1TIRON": "Day 1 Iron Intake",
    "DR2TIRON": "Day 2 Iron Intake",
    "DR1TVC": "Day 1 Vitamin C Intake",
    "DR2TVC": "Day 2 Vitamin C Intake",
    "DR1TVARA": "Day 1 Vitamin A Intake",
    "DR2TVARA": "Day 2 Vitamin A Intake",
    "DR1TFF": "Day 1 Food Folate Intake",
    "DR2TFF": "Day 2 Food Folate Intake",
    "DR1TCAFF": "Day 1 Caffeine Intake",
    "DR2TCAFF": "Day 2 Caffeine Intake",
    "DR1TALCO": "Day 1 Alcohol Intake",
    "DR2TALCO": "Day 2 Alcohol Intake",
    "DR1TWATR": "Day 1 Water Intake",
    "DR2TWATR": "Day 2 Water Intake",
}


YES_NO_FEATURES = {
    "MCQ010",
    "MCQ160A",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
    "MCQ160L",
    "MCQ160M",
    "MCQ220",
    "MCQ366D",
    "MCQ092",
    "SMQ020",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
}


# =============================================================================
# 4. BASIC HELPERS
# =============================================================================

def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_file(
    path: Path,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired current-run SHAP artifact was not found:\n{path}\n"
            "Run the revised full pipeline with SHAP enabled before using "
            "this post-run plotting script."
        )

    return path


def save_figure_all_formats(
    figure: plt.Figure,
) -> None:
    figure.savefig(
        OUTPUT_PNG,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_SVG,
        bbox_inches="tight",
    )

    try:
        figure.savefig(
            OUTPUT_TIFF,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            pil_kwargs={
                "compression": "tiff_lzw",
            },
        )

    except Exception as error:
        print(
            "TIFF export skipped:",
            error,
        )


def fallback_feature_label(
    feature: str,
) -> str:
    code = str(
        feature
    )

    if code.startswith("DR1"):
        return f"Day 1 Dietary Measure ({code})"

    if code.startswith("DR2"):
        return f"Day 2 Dietary Measure ({code})"

    if code.startswith(
        (
            "LBX",
            "LBD",
        )
    ):
        return f"Laboratory Measure ({code})"

    if code.startswith("BMX"):
        return f"Body Measure ({code})"

    if code.startswith("BPX"):
        return f"Blood Pressure Measure ({code})"

    if code.startswith("MCQ"):
        return f"Medical Condition ({code})"

    if code.startswith(
        (
            "PAQ",
            "PAD",
        )
    ):
        return f"Physical Activity ({code})"

    if code.startswith(
        (
            "SMQ",
            "SMD",
        )
    ):
        return f"Smoking Measure ({code})"

    if code.startswith("ALQ"):
        return f"Alcohol Measure ({code})"

    if code.startswith(
        (
            "SLQ",
            "SLD",
        )
    ):
        return f"Sleep Measure ({code})"

    if code.startswith("DPQ"):
        return f"Depression Measure ({code})"

    if code.startswith(
        (
            "RID",
            "DMD",
            "IND",
        )
    ):
        return f"Demographic Measure ({code})"

    return code.replace(
        "_",
        " ",
    )


def clinical_feature_label(
    feature: str,
) -> str:
    return FEATURE_LABEL_MAP.get(
        feature,
        fallback_feature_label(
            feature
        ),
    )


def make_unique_labels(
    labels: Sequence[str],
    feature_codes: Sequence[str],
) -> List[str]:
    counts = Counter(
        labels
    )

    return [
        (
            label
            if counts[
                label
            ]
            == 1
            else f"{label} ({feature_code})"
        )
        for label, feature_code
        in zip(
            labels,
            feature_codes,
        )
    ]


# =============================================================================
# 5. LOAD CURRENT-RUN SHAP OUTPUTS
# =============================================================================

def load_grouped_shap_outputs() -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    grouped_shap = np.asarray(
        np.load(
            require_file(
                GROUPED_SHAP_FILE
            ),
            allow_pickle=False,
        ),
        dtype=float,
    )

    grouped_feature_names = np.asarray(
        np.load(
            require_file(
                GROUPED_FEATURE_NAMES_FILE
            ),
            allow_pickle=True,
        ),
        dtype=object,
    ).astype(str)

    sample_values = pd.read_csv(
        require_file(
            ORIGINAL_SAMPLE_VALUES_FILE
        )
    )

    sample_values.columns = (
        sample_values.columns
        .astype(str)
        .str.strip()
    )

    feature_metadata = pd.read_csv(
        require_file(
            FINAL_FEATURE_FILE
        )
    )

    feature_metadata.columns = (
        feature_metadata.columns
        .astype(str)
        .str.strip()
    )

    if grouped_shap.ndim != 2:
        raise ValueError(
            "Grouped SHAP values must be a two-dimensional matrix. "
            f"Observed shape: {grouped_shap.shape}"
        )

    if grouped_shap.shape[
        1
    ] != len(
        grouped_feature_names
    ):
        raise ValueError(
            "Grouped SHAP width does not match grouped feature-name count."
        )

    if grouped_shap.shape[
        0
    ] != len(
        sample_values
    ):
        raise ValueError(
            "Grouped SHAP row count does not match the original SHAP sample "
            "feature-value table.\n"
            f"Grouped SHAP rows: {grouped_shap.shape[0]}\n"
            f"Sample-value rows: {len(sample_values)}"
        )

    required_metadata_columns = {
        "feature",
        "feature_type",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(
            feature_metadata.columns
        )
    )

    if missing_metadata_columns:
        raise KeyError(
            "The following columns are missing from final_feature_list.csv: "
            f"{sorted(missing_metadata_columns)}"
        )

    return (
        grouped_shap,
        grouped_feature_names,
        sample_values,
        feature_metadata,
    )


# =============================================================================
# 6. ORIGINAL FEATURE VALUES FOR COLORING
# =============================================================================

def normalized_string(
    value,
) -> Optional[str]:
    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
        "missing",
    }:
        return None

    try:
        numeric = float(
            text
        )

        if np.isfinite(
            numeric
        ):
            if numeric.is_integer():
                return str(
                    int(
                        numeric
                    )
                )

            return f"{numeric:g}"

    except Exception:
        pass

    return text


def binary_yes_no_values(
    series: pd.Series,
) -> np.ndarray:
    output = np.full(
        len(
            series
        ),
        np.nan,
        dtype=float,
    )

    normalized = series.map(
        normalized_string
    )

    output[
        normalized.eq(
            "1"
        ).to_numpy()
    ] = 1.0

    output[
        normalized.eq(
            "2"
        ).to_numpy()
    ] = 0.0

    yes_text = normalized.str.lower().isin(
        [
            "yes",
            "true",
        ]
    )

    no_text = normalized.str.lower().isin(
        [
            "no",
            "false",
        ]
    )

    output[
        yes_text.fillna(
            False
        ).to_numpy()
    ] = 1.0

    output[
        no_text.fillna(
            False
        ).to_numpy()
    ] = 0.0

    return output


def stable_category_values(
    series: pd.Series,
) -> Tuple[
    np.ndarray,
    str,
]:
    normalized = series.map(
        normalized_string
    )

    numeric = pd.to_numeric(
        normalized,
        errors="coerce",
    )

    observed_n = int(
        normalized.notna()
        .sum()
    )

    numeric_n = int(
        numeric.notna()
        .sum()
    )

    if (
        observed_n
        > 0
        and numeric_n
        == observed_n
    ):
        return (
            numeric.to_numpy(
                dtype=float
            ),
            "numeric category codes",
        )

    categories = sorted(
        normalized.dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    category_map = {
        category: float(
            index
        )
        for index, category
        in enumerate(
            categories
        )
    }

    encoded = normalized.map(
        category_map
    )

    return (
        encoded.to_numpy(
            dtype=float
        ),
        (
            "stable alphabetical category codes: "
            + "; ".join(
                f"{category}={int(code)}"
                for category, code
                in category_map.items()
            )
        ),
    )


def build_grouped_feature_value_matrix(
    grouped_feature_names: np.ndarray,
    sample_values: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    pd.DataFrame,
]:
    feature_type_map = (
        feature_metadata
        .drop_duplicates(
            subset=[
                "feature",
            ]
        )
        .set_index(
            "feature"
        )[
            "feature_type"
        ]
        .astype(str)
        .str.lower()
        .to_dict()
    )

    output_columns: List[
        np.ndarray
    ] = []

    audit_rows: List[
        Dict[str, object]
    ] = []

    for feature in grouped_feature_names:
        if feature not in sample_values.columns:
            values = np.full(
                len(
                    sample_values
                ),
                np.nan,
                dtype=float,
            )

            feature_type = (
                feature_type_map.get(
                    feature,
                    "unknown",
                )
            )

            color_method = (
                "feature absent from saved original-value sample"
            )

        else:
            series = sample_values[
                feature
            ]

            feature_type = (
                feature_type_map.get(
                    feature,
                    "unknown",
                )
            )

            if feature in YES_NO_FEATURES:
                values = binary_yes_no_values(
                    series
                )

                color_method = (
                    "binary mapping: No=0, Yes=1"
                )

            elif feature_type in {
                "numeric",
                "numerical",
                "continuous",
            }:
                values = pd.to_numeric(
                    series,
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                color_method = (
                    "original numerical value"
                )

            else:
                values, color_method = (
                    stable_category_values(
                        series
                    )
                )

        output_columns.append(
            values
        )

        finite_values = values[
            np.isfinite(
                values
            )
        ]

        audit_rows.append(
            {
                "base_feature": feature,
                "clinical_feature": (
                    clinical_feature_label(
                        feature
                    )
                ),
                "feature_type": (
                    feature_type
                ),
                "color_method": (
                    color_method
                ),
                "n_nonmissing_color_values": int(
                    len(
                        finite_values
                    )
                ),
                "n_missing_color_values": int(
                    len(
                        values
                    )
                    - len(
                        finite_values
                    )
                ),
                "minimum_color_value": (
                    float(
                        np.min(
                            finite_values
                        )
                    )
                    if len(
                        finite_values
                    )
                    else np.nan
                ),
                "maximum_color_value": (
                    float(
                        np.max(
                            finite_values
                        )
                    )
                    if len(
                        finite_values
                    )
                    else np.nan
                ),
            }
        )

    value_matrix = np.column_stack(
        output_columns
    ).astype(float)

    return (
        value_matrix,
        pd.DataFrame(
            audit_rows
        ),
    )


# =============================================================================
# 7. GROUPED IMPORTANCE
# =============================================================================

def build_grouped_importance_table(
    grouped_shap: np.ndarray,
    grouped_feature_names: np.ndarray,
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    domain_map = {}

    if "domain" in feature_metadata.columns:
        domain_map = (
            feature_metadata
            .drop_duplicates(
                subset=[
                    "feature",
                ]
            )
            .set_index(
                "feature"
            )[
                "domain"
            ]
            .astype(str)
            .to_dict()
        )

    feature_type_map = (
        feature_metadata
        .drop_duplicates(
            subset=[
                "feature",
            ]
        )
        .set_index(
            "feature"
        )[
            "feature_type"
        ]
        .astype(str)
        .to_dict()
    )

    importance = pd.DataFrame(
        {
            "base_feature": (
                grouped_feature_names
            ),
            "clinical_feature": [
                clinical_feature_label(
                    feature
                )
                for feature
                in grouped_feature_names
            ],
            "domain": [
                domain_map.get(
                    feature,
                    "Unknown",
                )
                for feature
                in grouped_feature_names
            ],
            "feature_type": [
                feature_type_map.get(
                    feature,
                    "Unknown",
                )
                for feature
                in grouped_feature_names
            ],
            "mean_abs_grouped_shap": (
                np.mean(
                    np.abs(
                        grouped_shap
                    ),
                    axis=0,
                )
            ),
            "mean_signed_grouped_shap": (
                np.mean(
                    grouped_shap,
                    axis=0,
                )
            ),
            "median_signed_grouped_shap": (
                np.median(
                    grouped_shap,
                    axis=0,
                )
            ),
            "proportion_positive_grouped_shap": (
                np.mean(
                    grouped_shap
                    > 0,
                    axis=0,
                )
            ),
            "proportion_negative_grouped_shap": (
                np.mean(
                    grouped_shap
                    < 0,
                    axis=0,
                )
            ),
        }
    ).sort_values(
        "mean_abs_grouped_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    importance.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                importance
            )
            + 1,
        ),
    )

    total_importance = float(
        importance[
            "mean_abs_grouped_shap"
        ].sum()
    )

    importance[
        "relative_grouped_importance"
    ] = (
        importance[
            "mean_abs_grouped_shap"
        ]
        / max(
            total_importance,
            1e-15,
        )
    )

    importance[
        "relative_grouped_importance_pct"
    ] = (
        100.0
        * importance[
            "relative_grouped_importance"
        ]
    )

    return importance


# =============================================================================
# 8. BEESWARM PREPARATION
# =============================================================================

def scale_feature_values(
    values: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    scaled = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    missing_mask = ~np.isfinite(
        values
    )

    for column_index in range(
        values.shape[
            1
        ]
    ):
        column = values[
            :,
            column_index,
        ]

        finite = np.isfinite(
            column
        )

        if not finite.any():
            continue

        minimum = float(
            np.min(
                column[
                    finite
                ]
            )
        )

        maximum = float(
            np.max(
                column[
                    finite
                ]
            )
        )

        if maximum - minimum <= 0:
            scaled[
                finite,
                column_index,
            ] = 0.5

        else:
            scaled[
                finite,
                column_index,
            ] = (
                column[
                    finite
                ]
                - minimum
            ) / (
                maximum
                - minimum
            )

    return (
        np.clip(
            scaled,
            0.0,
            1.0,
        ),
        missing_mask,
    )


def prepare_top_features(
    grouped_shap: np.ndarray,
    grouped_values: np.ndarray,
    grouped_feature_names: np.ndarray,
    importance: pd.DataFrame,
    top_k: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    float,
    pd.DataFrame,
]:
    top_table = (
        importance
        .head(
            top_k
        )
        .copy()
    )

    feature_index = {
        feature: index
        for index, feature
        in enumerate(
            grouped_feature_names
        )
    }

    selected_codes_descending = (
        top_table[
            "base_feature"
        ]
        .astype(str)
        .tolist()
    )

    selected_indices_descending = np.asarray(
        [
            feature_index[
                feature
            ]
            for feature
            in selected_codes_descending
        ],
        dtype=int,
    )

    # Reverse for Matplotlib y ordering, placing rank 1 at the top.
    selected_indices = (
        selected_indices_descending[
            ::-1
        ]
    )

    selected_codes = grouped_feature_names[
        selected_indices
    ]

    selected_shap = grouped_shap[
        :,
        selected_indices,
    ]

    selected_values = grouped_values[
        :,
        selected_indices,
    ]

    labels = [
        clinical_feature_label(
            feature
        )
        for feature
        in selected_codes
    ]

    labels = make_unique_labels(
        labels,
        selected_codes,
    )

    scaled_values, missing_mask = (
        scale_feature_values(
            selected_values
        )
    )

    clip_value = float(
        np.nanpercentile(
            np.abs(
                selected_shap
            ),
            DISPLAY_CLIP_PERCENTILE,
        )
    )

    clip_value = max(
        clip_value,
        1e-12,
    )

    displayed_shap = np.clip(
        selected_shap,
        -clip_value,
        clip_value,
    )

    return (
        displayed_shap,
        scaled_values,
        missing_mask,
        labels,
        selected_codes,
        clip_value,
        top_table,
    )


# =============================================================================
# 9. CUSTOM GROUPED BEESWARM
# =============================================================================

def plot_grouped_beeswarm(
    grouped_shap: np.ndarray,
    grouped_values: np.ndarray,
    grouped_feature_names: np.ndarray,
    importance: pd.DataFrame,
) -> float:
    (
        displayed_shap,
        scaled_values,
        missing_mask,
        labels,
        selected_codes,
        clip_value,
        top_table,
    ) = prepare_top_features(
        grouped_shap,
        grouped_values,
        grouped_feature_names,
        importance,
        TOP_K,
    )

    cmap = mpl.colormaps[
        CMAP_NAME
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    figure = plt.figure(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT,
        )
    )

    grid = figure.add_gridspec(
        1,
        24,
    )

    axis = figure.add_subplot(
        grid[
            0,
            :23,
        ]
    )

    colorbar_axis = figure.add_subplot(
        grid[
            0,
            23,
        ]
    )

    axis.axvline(
        0,
        color="dimgray",
        linewidth=1.1,
        alpha=0.82,
        zorder=0,
    )

    for position in range(
        len(
            labels
        )
    ):
        x_values = displayed_shap[
            :,
            position,
        ]

        y_values = (
            np.full(
                len(
                    x_values
                ),
                position,
                dtype=float,
            )
            + (
                rng.random(
                    len(
                        x_values
                    )
                )
                - 0.5
            )
            * BEESWARM_JITTER
        )

        feature_missing = missing_mask[
            :,
            position,
        ]

        feature_observed = (
            ~feature_missing
        )

        if feature_observed.any():
            axis.scatter(
                x_values[
                    feature_observed
                ],
                y_values[
                    feature_observed
                ],
                s=BEESWARM_DOT_SIZE,
                c=cmap(
                    scaled_values[
                        feature_observed,
                        position,
                    ]
                ),
                edgecolor="none",
                alpha=BEESWARM_ALPHA,
                rasterized=True,
            )

        if feature_missing.any():
            axis.scatter(
                x_values[
                    feature_missing
                ],
                y_values[
                    feature_missing
                ],
                s=BEESWARM_DOT_SIZE,
                color=MISSING_POINT_COLOR,
                edgecolor="none",
                alpha=0.58,
                rasterized=True,
            )

        median_value = float(
            np.median(
                x_values
            )
        )

        axis.plot(
            [
                median_value,
                median_value,
            ],
            [
                position
                - 0.34,
                position
                + 0.34,
            ],
            color="black",
            linewidth=1.6,
            alpha=0.32,
        )

    axis.set_yticks(
        np.arange(
            len(
                labels
            )
        )
    )

    axis.set_yticklabels(
        labels,
        fontsize=10,
    )

    axis.set_ylim(
        -0.8,
        len(
            labels
        )
        - 0.2,
    )

    axis.set_xlabel(
        "SHAP value (impact on XGBoost model output)"
    )

    axis.set_title(
        "Grouped Clinical Feature Contributions to Mobility-Disability Classification",
        loc="left",
    )

    axis.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.22,
    )

    colorbar = mpl.colorbar.ColorbarBase(
        colorbar_axis,
        cmap=cmap,
        norm=mpl.colors.Normalize(
            vmin=0,
            vmax=1,
        ),
    )

    colorbar.set_ticks(
        [
            0,
            1,
        ]
    )

    colorbar.set_ticklabels(
        [
            "Low",
            "High",
        ]
    )

    colorbar.set_label(
        "Feature value",
        rotation=270,
        labelpad=17,
        fontweight="bold",
    )

    axis.text(
        0.995,
        0.006,
        (
            f"Display truncated at the "
            f"{DISPLAY_CLIP_PERCENTILE:g}th percentile of |SHAP value|; "
            "gray indicates a missing original feature value"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="dimgray",
    )

    figure.tight_layout()

    save_figure_all_formats(
        figure
    )

    plt.show()
    plt.close(
        figure
    )

    return clip_value


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:
    configure_style()

    print("=" * 100)
    print(
        "NHANES 2017–2018 ADULT MOBILITY-DISABILITY CLASSIFICATION"
    )
    print(
        "NATIVE TREESHAP GROUPED CLINICAL SUMMARY — FULL INTERNAL VALIDATION"
    )
    print("=" * 100)
    print("Run directory :", RUN_DIR)
    print("SHAP directory:", SHAP_DIR)
    print("Output folder :", FIG_DIR)

    (
        grouped_shap,
        grouped_feature_names,
        sample_values,
        feature_metadata,
    ) = load_grouped_shap_outputs()

    print("\nLoaded current-run artifacts:")
    print(
        "Grouped SHAP matrix :",
        grouped_shap.shape,
    )
    print(
        "Grouped features    :",
        len(
            grouped_feature_names
        ),
    )
    print(
        "Sample-value rows   :",
        len(
            sample_values
        ),
    )

    (
        grouped_values,
        color_audit,
    ) = build_grouped_feature_value_matrix(
        grouped_feature_names,
        sample_values,
        feature_metadata,
    )

    importance = build_grouped_importance_table(
        grouped_shap,
        grouped_feature_names,
        feature_metadata,
    )

    if len(
        importance
    ) < TOP_K:
        raise RuntimeError(
            f"Only {len(importance)} grouped features are available, "
            f"but TOP_K={TOP_K}."
        )

    importance.to_csv(
        IMPORTANCE_OUTPUT,
        index=False,
    )

    top20 = (
        importance
        .head(
            TOP_K
        )
        .copy()
    )

    top20.to_csv(
        TOP20_OUTPUT,
        index=False,
    )

    color_audit.to_csv(
        COLOR_AUDIT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print(
        "TOP GROUPED CLINICAL SHAP FEATURES"
    )
    print("=" * 100)

    print(
        top20[
            [
                "rank",
                "clinical_feature",
                "domain",
                "feature_type",
                "mean_abs_grouped_shap",
                "relative_grouped_importance_pct",
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    clip_value = plot_grouped_beeswarm(
        grouped_shap,
        grouped_values,
        grouped_feature_names,
        importance,
    )

    summary = {
        "analysis": (
            "NHANES 2017-2018 adult mobility-disability classification "
            "grouped XGBoost SHAP summary"
        ),
        "run_directory": str(
            RUN_DIR
        ),
        "grouped_shap_file": str(
            GROUPED_SHAP_FILE
        ),
        "grouped_feature_names_file": str(
            GROUPED_FEATURE_NAMES_FILE
        ),
        "original_sample_values_file": str(
            ORIGINAL_SAMPLE_VALUES_FILE
        ),
        "shap_sample_n": int(
            grouped_shap.shape[
                0
            ]
        ),
        "grouped_feature_count": int(
            grouped_shap.shape[
                1
            ]
        ),
        "displayed_feature_count": int(
            TOP_K
        ),
        "grouping_method": (
            "Participant-level SHAP values were summed across transformed "
            "columns belonging to each original NHANES feature by the "
            "publication-revised pipeline."
        ),
        "color_method": (
            "Original sampled feature values; binary Yes/No variables use "
            "No=0 and Yes=1; gray represents missing values."
        ),
        "display_clip_percentile": float(
            DISPLAY_CLIP_PERCENTILE
        ),
        "display_clip_value": float(
            clip_value
        ),
        "figure_png": str(
            OUTPUT_PNG
        ),
        "figure_pdf": str(
            OUTPUT_PDF
        ),
        "figure_svg": str(
            OUTPUT_SVG
        ),
        "figure_tiff": str(
            OUTPUT_TIFF
        ),
        "importance_table": str(
            IMPORTANCE_OUTPUT
        ),
        "top20_table": str(
            TOP20_OUTPUT
        ),
        "color_value_audit": str(
            COLOR_AUDIT_OUTPUT
        ),
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("\n" + "=" * 100)
    print(
        "CURRENT-RUN GROUPED SHAP FIGURE COMPLETE"
    )
    print("=" * 100)

    print("\nSaved figures:")
    print(" -", OUTPUT_PNG)
    print(" -", OUTPUT_PDF)
    print(" -", OUTPUT_SVG)
    print(" -", OUTPUT_TIFF)

    print("\nSaved tables:")
    print(" -", IMPORTANCE_OUTPUT)
    print(" -", TOP20_OUTPUT)
    print(" -", COLOR_AUDIT_OUTPUT)

    print("\nSaved summary:")
    print(" -", SUMMARY_OUTPUT)
    print("=" * 100)


if __name__ == "__main__":
    main()
