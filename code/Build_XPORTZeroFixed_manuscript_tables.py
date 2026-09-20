#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NHANES 2017–2018 ADULT MOBILITY-DISABILITY CLASSIFICATION
CURRENT-RUN MANUSCRIPT TABLE BUILDER
=========================================================

Current run
-----------
/athena/madelab/scratch/iqh4001/Disability/Results/
20260727_004939_NHANES_2017_2018_Adult_
Mobility_Disability_Classification_Revised

Creates
-------
Main manuscript tables
1. Table 1 — Adult cohort characteristics by mobility-disability status
2. Table 2 — Internal-validation model performance
3. Table 3A — Paired model comparisons
4. Table 3B — Domain ablation

Supplementary tables
5. Table S1 — Top 20 grouped SHAP features
6. Table S2 — Internal-validation subgroup performance
7. Table S3 — Final predictor/domain list
8. Table S4 — Missingness by outcome

Outputs
-------
All tables are saved under:

    <CURRENT RUN>/manuscript_tables/

Both raw numerical and formatted manuscript-facing CSV files are saved.
A combined Markdown document is also created.

Jupyter
-------
%run Build_current_run_manuscript_tables.py

The code displays tables inline when IPython is available.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CURRENT-RUN PATHS
# =============================================================================

DATA_DIR = Path(
    "/athena/madelab/scratch/iqh4001/Disability/Data/"
    "NHANES_2017_2018"
)

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/Disability/Results/"
    "20260920_115949_NHANES_2017_2018_Adult_"
    "Mobility_Disability_Classification_Revised_XPORTZeroFixed"
)

TABLE_DIR = RUN_DIR / "tables"
SHAP_DIR = RUN_DIR / "shap"
ABLATION_DIR = RUN_DIR / "domain_ablation"
SUBGROUP_DIR = RUN_DIR / "subgroups"

OUT_DIR = RUN_DIR / "manuscript_tables"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PERFORMANCE_FILE = (
    TABLE_DIR
    / "test_model_performance.csv"
)

PERFORMANCE_CI_FILE = (
    TABLE_DIR
    / "test_model_performance_95ci.csv"
)

LOCKED_THRESHOLD_FILE = (
    TABLE_DIR
    / "locked_thresholds.csv"
)

PAIRED_DIFFERENCE_FILE = (
    TABLE_DIR
    / "paired_model_differences.csv"
)

ABLATION_FILE = (
    ABLATION_DIR
    / "domain_ablation_results.csv"
)

TOP_SHAP_FILE = (
    SHAP_DIR
    / "Figure_Grouped_SHAP_Top20_Current_Run.csv"
)

FALLBACK_TOP_SHAP_FILE = (
    SHAP_DIR
    / "shap_grouped_clinical_feature_importance.csv"
)

SUBGROUP_FILE = (
    SUBGROUP_DIR
    / "subgroup_performance.csv"
)

FINAL_FEATURE_FILE = (
    TABLE_DIR
    / "final_feature_list.csv"
)

MISSINGNESS_FILE = (
    TABLE_DIR
    / "missingness_by_outcome.csv"
)

MODEL_ORDER = [
    "XGBoost",
    "Logistic Regression",
    "Random Forest",
]

MINIMUM_AGE_YEARS = 18

OUTCOME_SOURCE = "DLQ050"
OUTCOME_COLUMN = "mobility_disability"


# =============================================================================
# 2. OUTPUT FILES
# =============================================================================

TABLE1_RAW = (
    OUT_DIR
    / "Table1_Cohort_Characteristics_Raw.csv"
)

TABLE1_FORMATTED = (
    OUT_DIR
    / "Table1_Cohort_Characteristics_Formatted.csv"
)

TABLE2_RAW = (
    OUT_DIR
    / "Table2_Model_Performance_Raw.csv"
)

TABLE2_FORMATTED = (
    OUT_DIR
    / "Table2_Model_Performance_Formatted.csv"
)

TABLE3A_FORMATTED = (
    OUT_DIR
    / "Table3A_Paired_Model_Comparisons.csv"
)

TABLE3B_FORMATTED = (
    OUT_DIR
    / "Table3B_Domain_Ablation.csv"
)

TABLES1_SHAP = (
    OUT_DIR
    / "TableS1_Top20_Grouped_SHAP_Features.csv"
)

TABLES2_SUBGROUP = (
    OUT_DIR
    / "TableS2_Subgroup_Performance.csv"
)

TABLES3_FEATURES = (
    OUT_DIR
    / "TableS3_Final_Predictor_Domain_List.csv"
)

TABLES4_MISSINGNESS = (
    OUT_DIR
    / "TableS4_Missingness_by_Outcome.csv"
)

MARKDOWN_FILE = (
    OUT_DIR
    / "Current_Run_Manuscript_Tables.md"
)

NOTES_FILE = (
    OUT_DIR
    / "Current_Run_Table_Footnotes.txt"
)


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def print_header(
    title: str,
) -> None:
    print(
        "\n"
        + "=" * 100
    )
    print(title)
    print(
        "=" * 100
    )


def require_file(
    path: Path,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file was not found:\n{path}"
        )

    return path


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output.columns = (
        output.columns
        .astype(str)
        .str.strip()
    )

    return output


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> Optional[str]:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    if required:
        raise KeyError(
            "None of the expected columns were found:\n"
            f"{list(candidates)}\n"
            f"Available columns:\n"
            f"{dataframe.columns.tolist()}"
        )

    return None


def finite_float(
    value: Any,
) -> float:
    try:
        value = float(
            value
        )

        if np.isfinite(
            value
        ):
            return value

    except Exception:
        pass

    return np.nan


def format_number(
    value: Any,
    digits: int = 3,
) -> str:
    value = finite_float(
        value
    )

    if not np.isfinite(
        value
    ):
        return "NA"

    return f"{value:.{digits}f}"


def format_percent(
    value: Any,
    digits: int = 1,
) -> str:
    value = finite_float(
        value
    )

    if not np.isfinite(
        value
    ):
        return "NA"

    return f"{100.0 * value:.{digits}f}%"


def format_estimate_ci(
    estimate: Any,
    lower: Any,
    upper: Any,
    *,
    digits: int = 3,
    percent: bool = False,
) -> str:
    estimate = finite_float(
        estimate
    )

    lower = finite_float(
        lower
    )

    upper = finite_float(
        upper
    )

    if not np.isfinite(
        estimate
    ):
        return "NA"

    if (
        not np.isfinite(
            lower
        )
        or not np.isfinite(
            upper
        )
    ):
        return (
            format_percent(
                estimate,
                digits=1,
            )
            if percent
            else format_number(
                estimate,
                digits=digits,
            )
        )

    if percent:
        return (
            f"{100.0 * estimate:.1f}% "
            f"({100.0 * lower:.1f}–"
            f"{100.0 * upper:.1f})"
        )

    return (
        f"{estimate:.{digits}f} "
        f"({lower:.{digits}f}–"
        f"{upper:.{digits}f})"
    )


def markdown_table(
    dataframe: pd.DataFrame,
) -> str:
    """
    Create Markdown without requiring the optional tabulate package.
    """

    frame = dataframe.copy().fillna(
        ""
    )

    columns = [
        str(
            column
        )
        for column
        in frame.columns
    ]

    header = (
        "| "
        + " | ".join(
            columns
        )
        + " |"
    )

    separator = (
        "| "
        + " | ".join(
            [
                "---"
                for _ in columns
            ]
        )
        + " |"
    )

    rows = []

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        cells = [
            str(
                value
            )
            .replace(
                "\n",
                " "
            )
            .replace(
                "|",
                "\\|",
            )
            for value
            in row
        ]

        rows.append(
            "| "
            + " | ".join(
                cells
            )
            + " |"
        )

    return "\n".join(
        [
            header,
            separator,
            *rows,
        ]
    )


def display_table(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print_header(
        title
    )

    try:
        from IPython.display import display

        display(
            dataframe
        )

    except Exception:
        print(
            dataframe.to_string(
                index=False
            )
        )


# =============================================================================
# 4. TABLE 1 — LOAD AND RECONSTRUCT ADULT COHORT
# =============================================================================

REQUIRED_XPT_MODULES = [
    "DEMO_J",
    "DLQ_J",
    "BMX_J",
    "BPX_J",
    "DPQ_J",
    "MCQ_J",
    "PAQ_J",
    "SLQ_J",
    "CBC_J",
]


def find_module_file(
    module_name: str,
) -> Path:
    candidates = [
        DATA_DIR
        / f"{module_name}.XPT",
        DATA_DIR
        / f"{module_name}.xpt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {module_name}.XPT or {module_name}.xpt "
        f"in:\n{DATA_DIR}"
    )


def read_xpt_module(
    module_name: str,
) -> pd.DataFrame:
    path = find_module_file(
        module_name
    )

    dataframe = pd.read_sas(
        path,
        format="xport",
    )

    dataframe.columns = [
        str(
            column
        ).upper()
        for column
        in dataframe.columns
    ]

    return dataframe


SAS_XPORT_ZERO = 5.397605346934028e-79


def restore_xport_zero_artifacts(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restore the known SAS XPORT floating-point representation
    of true zero to 0.0 rather than treating it as missing.
    """

    output = dataframe.copy()

    for column in output.columns:

        if not pd.api.types.is_numeric_dtype(
            output[column]
        ):
            continue

        numeric = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        tiny_mask = (
            numeric.notna()
            & numeric.ne(0)
            & numeric.abs().lt(1e-50)
        )

        if not tiny_mask.any():
            continue

        zero_mask = (
            tiny_mask
            & np.isclose(
                np.abs(
                    numeric.to_numpy(dtype=float)
                ),
                SAS_XPORT_ZERO,
                rtol=1e-12,
                atol=0.0,
            )
        )

        unexpected_mask = (
            tiny_mask
            & ~zero_mask
        )

        if unexpected_mask.any():

            bad = (
                numeric.loc[
                    unexpected_mask
                ]
                .value_counts()
                .head(20)
            )

            raise RuntimeError(
                "Unexpected tiny nonzero XPORT values "
                f"in {column}:\n{bad}"
            )

        output.loc[
            zero_mask,
            column,
        ] = 0.0

    return output


def load_table1_cohort() -> pd.DataFrame:
    print_header(
        "LOADING TABLE 1 COHORT"
    )

    modules: Dict[
        str,
        pd.DataFrame,
    ] = {}

    for module_name in REQUIRED_XPT_MODULES:
        module = read_xpt_module(
            module_name
        )

        modules[
            module_name
        ] = module

        print(
            f"Loaded {module_name:8s}: "
            f"{module.shape}"
        )

    cohort = modules[
        "DEMO_J"
    ].copy()

    for module_name in REQUIRED_XPT_MODULES:
        if module_name == "DEMO_J":
            continue

        module = modules[
            module_name
        ]

        overlapping = [
            column
            for column
            in module.columns
            if (
                column
                in cohort.columns
                and column
                != "SEQN"
            )
        ]

        cohort = cohort.merge(
            module.drop(
                columns=overlapping,
                errors="ignore",
            ),
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

    cohort = restore_xport_zero_artifacts(
        cohort
    )

    cohort[
        OUTCOME_COLUMN
    ] = np.where(
        cohort[
            OUTCOME_SOURCE
        ].eq(
            1
        ),
        1,
        np.where(
            cohort[
                OUTCOME_SOURCE
            ].eq(
                2
            ),
            0,
            np.nan,
        ),
    )

    age = pd.to_numeric(
        cohort[
            "RIDAGEYR"
        ],
        errors="coerce",
    )

    cohort = cohort.loc[
        cohort[
            OUTCOME_COLUMN
        ].notna()
        & age.ge(
            MINIMUM_AGE_YEARS
        )
    ].copy()

    cohort[
        OUTCOME_COLUMN
    ] = (
        cohort[
            OUTCOME_COLUMN
        ]
        .astype(
            int
        )
    )

    # Derived PHQ-9 score, matching the current revised pipeline.
    phq_items = [
        "DPQ010",
        "DPQ020",
        "DPQ030",
        "DPQ040",
        "DPQ050",
        "DPQ060",
        "DPQ070",
        "DPQ080",
        "DPQ090",
    ]

    phq_frame = (
        cohort[
            phq_items
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            {
                7: np.nan,
                9: np.nan,
            }
        )
    )

    cohort[
        "PHQ9_TOTAL"
    ] = phq_frame.sum(
        axis=1,
        min_count=1,
    )

    systolic_columns = [
        column
        for column
        in [
            "BPXSY1",
            "BPXSY2",
            "BPXSY3",
            "BPXSY4",
        ]
        if column
        in cohort.columns
    ]

    cohort[
        "SBP_MEAN"
    ] = (
        cohort[
            systolic_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .mean(
            axis=1
        )
    )

    return cohort.reset_index(
        drop=True
    )


# =============================================================================
# 5. TABLE 1 — CHARACTERISTIC DEFINITIONS
# =============================================================================

def clean_continuous(
    series: pd.Series,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> pd.Series:
    output = pd.to_numeric(
        series,
        errors="coerce",
    )

    if minimum is not None:
        output = output.where(
            output.ge(
                minimum
            )
        )

    if maximum is not None:
        output = output.where(
            output.le(
                maximum
            )
        )

    return output


def map_categories(
    series: pd.Series,
    mapping: Mapping[float, str],
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.map(
        mapping
    )


def build_table1_variables(
    cohort: pd.DataFrame,
) -> Tuple[
    Dict[str, pd.Series],
    Dict[str, List[str]],
]:
    continuous = {
        "Age, years": clean_continuous(
            cohort[
                "RIDAGEYR"
            ],
            minimum=18,
            maximum=80,
        ),
        "Income-to-poverty ratio": clean_continuous(
            cohort[
                "INDFMPIR"
            ],
            minimum=0,
            maximum=5,
        ),
        "Body mass index, kg/m²": clean_continuous(
            cohort[
                "BMXBMI"
            ],
            minimum=10,
            maximum=80,
        ),
        "Waist circumference, cm": clean_continuous(
            cohort[
                "BMXWAIST"
            ],
            minimum=30,
            maximum=200,
        ),
        "PHQ-9 depression score": clean_continuous(
            cohort[
                "PHQ9_TOTAL"
            ],
            minimum=0,
            maximum=27,
        ),
        "Weekday sleep duration, h": clean_continuous(
            cohort[
                "SLD012"
            ],
            minimum=0,
            maximum=24,
        ),
        "Mean systolic blood pressure, mmHg": clean_continuous(
            cohort[
                "SBP_MEAN"
            ],
            minimum=60,
            maximum=260,
        ),
        "Hemoglobin, g/dL": clean_continuous(
            cohort[
                "LBXHGB"
            ],
            minimum=3,
            maximum=25,
        ),
    }

    categorical = {
        "Sex": map_categories(
            cohort[
                "RIAGENDR"
            ],
            {
                1.0: "Male",
                2.0: "Female",
            },
        ),
        "Race/ethnicity": map_categories(
            cohort[
                "RIDRETH3"
            ],
            {
                1.0: "Mexican American",
                2.0: "Other Hispanic",
                3.0: "Non-Hispanic White",
                4.0: "Non-Hispanic Black",
                6.0: "Non-Hispanic Asian",
                7.0: "Other/Multiracial",
            },
        ),
        "Education": map_categories(
            cohort[
                "DMDEDUC2"
            ],
            {
                1.0: "Less than 9th grade",
                2.0: "9th–11th grade",
                3.0: "High school/GED",
                4.0: "Some college/AA",
                5.0: "College graduate or above",
            },
        ),
        "Arthritis": map_categories(
            cohort[
                "MCQ160A"
            ],
            {
                1.0: "Yes",
                2.0: "No",
            },
        ),
        "Moderate recreational activity": map_categories(
            cohort[
                "PAQ665"
            ],
            {
                1.0: "Yes",
                2.0: "No",
            },
        ),
    }

    category_orders = {
        "Sex": [
            "Female",
            "Male",
        ],
        "Race/ethnicity": [
            "Mexican American",
            "Other Hispanic",
            "Non-Hispanic White",
            "Non-Hispanic Black",
            "Non-Hispanic Asian",
            "Other/Multiracial",
        ],
        "Education": [
            "Less than 9th grade",
            "9th–11th grade",
            "High school/GED",
            "Some college/AA",
            "College graduate or above",
        ],
        "Arthritis": [
            "Yes",
            "No",
        ],
        "Moderate recreational activity": [
            "Yes",
            "No",
        ],
    }

    return (
        {
            **continuous,
            **categorical,
        },
        category_orders,
    )


# =============================================================================
# 6. TABLE 1 — SUMMARY AND SMD
# =============================================================================

CONTINUOUS_CHARACTERISTICS = {
    "Age, years",
    "Income-to-poverty ratio",
    "Body mass index, kg/m²",
    "Waist circumference, cm",
    "PHQ-9 depression score",
    "Weekday sleep duration, h",
    "Mean systolic blood pressure, mmHg",
    "Hemoglobin, g/dL",
}


def summarize_continuous(
    series: pd.Series,
) -> Dict[str, float]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }

    return {
        "n": int(
            len(
                values
            )
        ),
        "mean": float(
            values.mean()
        ),
        "sd": float(
            values.std(
                ddof=1
            )
        ),
        "median": float(
            values.median()
        ),
        "q1": float(
            values.quantile(
                0.25
            )
        ),
        "q3": float(
            values.quantile(
                0.75
            )
        ),
    }


def continuous_smd(
    series: pd.Series,
    outcome: pd.Series,
) -> float:
    group0 = pd.to_numeric(
        series.loc[
            outcome.eq(
                0
            )
        ],
        errors="coerce",
    ).dropna()

    group1 = pd.to_numeric(
        series.loc[
            outcome.eq(
                1
            )
        ],
        errors="coerce",
    ).dropna()

    if (
        len(
            group0
        )
        < 2
        or len(
            group1
        )
        < 2
    ):
        return np.nan

    pooled_variance = (
        group0.var(
            ddof=1
        )
        + group1.var(
            ddof=1
        )
    ) / 2.0

    if (
        not np.isfinite(
            pooled_variance
        )
        or pooled_variance
        <= 0
    ):
        return np.nan

    return float(
        abs(
            group1.mean()
            - group0.mean()
        )
        / math.sqrt(
            pooled_variance
        )
    )


def binary_level_smd(
    series: pd.Series,
    outcome: pd.Series,
    level: str,
) -> float:
    valid = series.notna()

    series_valid = series.loc[
        valid
    ]

    outcome_valid = outcome.loc[
        valid
    ]

    group0 = (
        series_valid.loc[
            outcome_valid.eq(
                0
            )
        ]
        .eq(
            level
        )
        .astype(
            float
        )
    )

    group1 = (
        series_valid.loc[
            outcome_valid.eq(
                1
            )
        ]
        .eq(
            level
        )
        .astype(
            float
        )
    )

    if (
        len(
            group0
        )
        == 0
        or len(
            group1
        )
        == 0
    ):
        return np.nan

    p0 = float(
        group0.mean()
    )

    p1 = float(
        group1.mean()
    )

    pooled_p = (
        p0
        + p1
    ) / 2.0

    denominator = math.sqrt(
        max(
            pooled_p
            * (
                1.0
                - pooled_p
            ),
            0.0,
        )
    )

    if denominator <= 0:
        return 0.0

    return abs(
        p1
        - p0
    ) / denominator


def format_continuous_summary(
    summary: Mapping[str, float],
) -> str:
    if int(
        summary[
            "n"
        ]
    ) == 0:
        return "NA"

    return (
        f"{summary['median']:.1f} "
        f"[{summary['q1']:.1f}, "
        f"{summary['q3']:.1f}]"
    )


def format_category_count(
    series: pd.Series,
    level: str,
) -> str:
    nonmissing_n = int(
        series.notna()
        .sum()
    )

    if nonmissing_n == 0:
        return "NA"

    count = int(
        series.eq(
            level
        )
        .sum()
    )

    percent = (
        100.0
        * count
        / nonmissing_n
    )

    return (
        f"{count:,} "
        f"({percent:.1f}%)"
    )


def build_table1(
    cohort: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    outcome = cohort[
        OUTCOME_COLUMN
    ].astype(
        int
    )

    variables, category_orders = (
        build_table1_variables(
            cohort
        )
    )

    raw_rows: List[
        Dict[str, Any]
    ] = []

    formatted_rows: List[
        Dict[str, Any]
    ] = []

    for characteristic, series in (
        variables.items()
    ):
        if characteristic in CONTINUOUS_CHARACTERISTICS:
            overall = summarize_continuous(
                series
            )

            no_disability = summarize_continuous(
                series.loc[
                    outcome.eq(
                        0
                    )
                ]
            )

            disability = summarize_continuous(
                series.loc[
                    outcome.eq(
                        1
                    )
                ]
            )

            smd = continuous_smd(
                series,
                outcome,
            )

            raw_rows.append(
                {
                    "Characteristic": characteristic,
                    "Level": "",
                    "Type": "Continuous",
                    "Overall_N_nonmissing": overall[
                        "n"
                    ],
                    "Overall_Median": overall[
                        "median"
                    ],
                    "Overall_Q1": overall[
                        "q1"
                    ],
                    "Overall_Q3": overall[
                        "q3"
                    ],
                    "NoDisability_N_nonmissing": no_disability[
                        "n"
                    ],
                    "NoDisability_Median": no_disability[
                        "median"
                    ],
                    "NoDisability_Q1": no_disability[
                        "q1"
                    ],
                    "NoDisability_Q3": no_disability[
                        "q3"
                    ],
                    "Disability_N_nonmissing": disability[
                        "n"
                    ],
                    "Disability_Median": disability[
                        "median"
                    ],
                    "Disability_Q1": disability[
                        "q1"
                    ],
                    "Disability_Q3": disability[
                        "q3"
                    ],
                    "Absolute_SMD": smd,
                }
            )

            formatted_rows.append(
                {
                    "Characteristic": characteristic,
                    "Level": "",
                    "Overall": format_continuous_summary(
                        overall
                    ),
                    "No mobility disability": (
                        format_continuous_summary(
                            no_disability
                        )
                    ),
                    "Mobility disability": (
                        format_continuous_summary(
                            disability
                        )
                    ),
                    "Absolute SMD": (
                        format_number(
                            smd,
                            digits=2,
                        )
                    ),
                    "Nonmissing N": (
                        f"{overall['n']:,}"
                    ),
                }
            )

        else:
            ordered_levels = category_orders[
                characteristic
            ]

            overall_nonmissing = int(
                series.notna()
                .sum()
            )

            no_series = series.loc[
                outcome.eq(
                    0
                )
            ]

            disability_series = series.loc[
                outcome.eq(
                    1
                )
            ]

            for level in ordered_levels:
                overall_count = int(
                    series.eq(
                        level
                    )
                    .sum()
                )

                no_count = int(
                    no_series.eq(
                        level
                    )
                    .sum()
                )

                disability_count = int(
                    disability_series.eq(
                        level
                    )
                    .sum()
                )

                raw_rows.append(
                    {
                        "Characteristic": characteristic,
                        "Level": level,
                        "Type": "Categorical",
                        "Overall_N_nonmissing": overall_nonmissing,
                        "Overall_Count": overall_count,
                        "Overall_Percent": (
                            overall_count
                            / max(
                                overall_nonmissing,
                                1,
                            )
                        ),
                        "NoDisability_N_nonmissing": int(
                            no_series.notna()
                            .sum()
                        ),
                        "NoDisability_Count": no_count,
                        "NoDisability_Percent": (
                            no_count
                            / max(
                                int(
                                    no_series.notna()
                                    .sum()
                                ),
                                1,
                            )
                        ),
                        "Disability_N_nonmissing": int(
                            disability_series.notna()
                            .sum()
                        ),
                        "Disability_Count": disability_count,
                        "Disability_Percent": (
                            disability_count
                            / max(
                                int(
                                    disability_series.notna()
                                    .sum()
                                ),
                                1,
                            )
                        ),
                        "Absolute_SMD": binary_level_smd(
                            series,
                            outcome,
                            level,
                        ),
                    }
                )

                formatted_rows.append(
                    {
                        "Characteristic": (
                            characteristic
                            if level
                            == ordered_levels[
                                0
                            ]
                            else ""
                        ),
                        "Level": level,
                        "Overall": format_category_count(
                            series,
                            level,
                        ),
                        "No mobility disability": (
                            format_category_count(
                                no_series,
                                level,
                            )
                        ),
                        "Mobility disability": (
                            format_category_count(
                                disability_series,
                                level,
                            )
                        ),
                        "Absolute SMD": (
                            format_number(
                                binary_level_smd(
                                    series,
                                    outcome,
                                    level,
                                ),
                                digits=2,
                            )
                        ),
                        "Nonmissing N": (
                            f"{overall_nonmissing:,}"
                        ),
                    }
                )

            missing_count = int(
                series.isna()
                .sum()
            )

            if missing_count > 0:
                no_missing = int(
                    no_series.isna()
                    .sum()
                )

                disability_missing = int(
                    disability_series.isna()
                    .sum()
                )

                formatted_rows.append(
                    {
                        "Characteristic": "",
                        "Level": "Missing",
                        "Overall": (
                            f"{missing_count:,} "
                            f"({100.0 * missing_count / len(series):.1f}%)"
                        ),
                        "No mobility disability": (
                            f"{no_missing:,} "
                            f"({100.0 * no_missing / len(no_series):.1f}%)"
                        ),
                        "Mobility disability": (
                            f"{disability_missing:,} "
                            f"({100.0 * disability_missing / len(disability_series):.1f}%)"
                        ),
                        "Absolute SMD": "",
                        "Nonmissing N": (
                            f"{overall_nonmissing:,}"
                        ),
                    }
                )

    raw_table = pd.DataFrame(
        raw_rows
    )

    formatted_table = pd.DataFrame(
        formatted_rows
    )

    return (
        raw_table,
        formatted_table,
    )


# =============================================================================
# 7. TABLE 2 — MODEL PERFORMANCE
# =============================================================================

CI_METRIC_ALIASES = {
    "AUROC": [
        "AUROC",
    ],
    "AUPRC": [
        "AUPRC",
    ],
    "Brier": [
        "Brier",
    ],
    "Accuracy": [
        "Accuracy",
    ],
    "Balanced_Accuracy": [
        "Balanced_Accuracy",
    ],
    "Precision_PPV": [
        "Precision_PPV",
        "PPV",
        "Precision",
    ],
    "Recall_Sensitivity": [
        "Recall_Sensitivity",
        "Sensitivity",
        "Recall",
    ],
    "Specificity": [
        "Specificity",
    ],
    "NPV": [
        "NPV",
    ],
    "F1": [
        "F1",
        "F1_Score",
    ],
    "MCC": [
        "MCC",
    ],
}


def select_model_row(
    performance: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    matches = performance.loc[
        performance[
            "Model"
        ]
        .astype(
            str
        )
        .str.strip()
        .eq(
            model_name
        )
    ]

    if matches.empty:
        raise ValueError(
            f"Model '{model_name}' was not found in "
            f"{PERFORMANCE_FILE.name}."
        )

    return matches.iloc[
        0
    ]


def find_ci_row(
    ci_table: pd.DataFrame,
    model_name: str,
    metric_candidates: Sequence[str],
) -> Optional[pd.Series]:
    model_rows = ci_table.loc[
        ci_table[
            "Model"
        ]
        .astype(
            str
        )
        .str.strip()
        .eq(
            model_name
        )
    ]

    for metric_name in metric_candidates:
        matches = model_rows.loc[
            model_rows[
                "Metric"
            ]
            .astype(
                str
            )
            .str.strip()
            .eq(
                metric_name
            )
        ]

        if not matches.empty:
            return matches.iloc[
                0
            ]

    return None


def performance_value(
    row: pd.Series,
    candidates: Sequence[str],
) -> float:
    for candidate in candidates:
        if candidate in row.index:
            value = finite_float(
                row[
                    candidate
                ]
            )

            if np.isfinite(
                value
            ):
                return value

    return np.nan


def build_table2() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    performance = normalize_columns(
        pd.read_csv(
            require_file(
                PERFORMANCE_FILE
            )
        )
    )

    ci_table = normalize_columns(
        pd.read_csv(
            require_file(
                PERFORMANCE_CI_FILE
            )
        )
    )

    thresholds = (
        normalize_columns(
            pd.read_csv(
                LOCKED_THRESHOLD_FILE
            )
        )
        if LOCKED_THRESHOLD_FILE.exists()
        else pd.DataFrame()
    )

    raw_rows: List[
        Dict[str, Any]
    ] = []

    model_rows = {
        model_name: select_model_row(
            performance,
            model_name,
        )
        for model_name
        in MODEL_ORDER
    }

    for model_name, row in (
        model_rows.items()
    ):
        raw_row = {
            "Model": model_name,
        }

        for metric_name, candidates in (
            CI_METRIC_ALIASES.items()
        ):
            raw_row[
                metric_name
            ] = performance_value(
                row,
                candidates,
            )

            ci_row = find_ci_row(
                ci_table,
                model_name,
                candidates,
            )

            raw_row[
                f"{metric_name}_CI_Lower"
            ] = (
                finite_float(
                    ci_row[
                        "CI_95_Lower"
                    ]
                )
                if ci_row
                is not None
                else np.nan
            )

            raw_row[
                f"{metric_name}_CI_Upper"
            ] = (
                finite_float(
                    ci_row[
                        "CI_95_Upper"
                    ]
                )
                if ci_row
                is not None
                else np.nan
            )

        raw_row[
            "Calibration_Intercept"
        ] = performance_value(
            row,
            [
                "Calibration_Intercept",
            ],
        )

        raw_row[
            "Calibration_Slope"
        ] = performance_value(
            row,
            [
                "Calibration_Slope",
            ],
        )

        raw_row[
            "Observed_to_Expected_Ratio"
        ] = performance_value(
            row,
            [
                "Observed_to_Expected_Ratio",
            ],
        )

        raw_row[
            "Quantile_Calibration_Error"
        ] = performance_value(
            row,
            [
                "Quantile_Calibration_Error",
                "Binned_Calibration_Error",
            ],
        )

        threshold = performance_value(
            row,
            [
                "Threshold",
            ],
        )

        if (
            not np.isfinite(
                threshold
            )
            and not thresholds.empty
        ):
            threshold_column = first_existing_column(
                thresholds,
                [
                    "Locked_Threshold",
                    "Threshold",
                ],
                required=False,
            )

            if threshold_column is not None:
                threshold_match = thresholds.loc[
                    thresholds[
                        "Model"
                    ]
                    .astype(
                        str
                    )
                    .str.strip()
                    .eq(
                        model_name
                    )
                ]

                if not threshold_match.empty:
                    threshold = finite_float(
                        threshold_match.iloc[
                            0
                        ][
                            threshold_column
                        ]
                    )

        raw_row[
            "Locked_Threshold"
        ] = threshold

        for count_column in [
            "TN",
            "FP",
            "FN",
            "TP",
        ]:
            raw_row[
                count_column
            ] = performance_value(
                row,
                [
                    count_column,
                ],
            )

        raw_rows.append(
            raw_row
        )

    raw_table = pd.DataFrame(
        raw_rows
    )

    metric_specs = [
        (
            "AUROC",
            "AUROC",
            False,
            3,
        ),
        (
            "AUPRC",
            "AUPRC",
            False,
            3,
        ),
        (
            "Brier score",
            "Brier",
            False,
            3,
        ),
        (
            "Accuracy",
            "Accuracy",
            True,
            1,
        ),
        (
            "Balanced accuracy",
            "Balanced_Accuracy",
            True,
            1,
        ),
        (
            "Sensitivity",
            "Recall_Sensitivity",
            True,
            1,
        ),
        (
            "Specificity",
            "Specificity",
            True,
            1,
        ),
        (
            "Positive predictive value",
            "Precision_PPV",
            True,
            1,
        ),
        (
            "Negative predictive value",
            "NPV",
            True,
            1,
        ),
        (
            "F1 score",
            "F1",
            False,
            3,
        ),
        (
            "Matthews correlation coefficient",
            "MCC",
            False,
            3,
        ),
        (
            "Calibration intercept",
            "Calibration_Intercept",
            False,
            3,
        ),
        (
            "Calibration slope",
            "Calibration_Slope",
            False,
            3,
        ),
        (
            "Observed/expected ratio",
            "Observed_to_Expected_Ratio",
            False,
            3,
        ),
        (
            "Quantile calibration error",
            "Quantile_Calibration_Error",
            False,
            3,
        ),
        (
            "Locked operating threshold",
            "Locked_Threshold",
            False,
            3,
        ),
        (
            "True positives",
            "TP",
            False,
            0,
        ),
        (
            "False positives",
            "FP",
            False,
            0,
        ),
        (
            "False negatives",
            "FN",
            False,
            0,
        ),
        (
            "True negatives",
            "TN",
            False,
            0,
        ),
    ]

    formatted_rows = []

    for display_name, metric, is_percent, digits in (
        metric_specs
    ):
        output_row = {
            "Metric": display_name,
        }

        for model_name in MODEL_ORDER:
            model_row = raw_table.loc[
                raw_table[
                    "Model"
                ].eq(
                    model_name
                )
            ].iloc[
                0
            ]

            estimate = model_row.get(
                metric,
                np.nan,
            )

            lower = model_row.get(
                f"{metric}_CI_Lower",
                np.nan,
            )

            upper = model_row.get(
                f"{metric}_CI_Upper",
                np.nan,
            )

            if metric in {
                "TN",
                "FP",
                "FN",
                "TP",
            }:
                output_row[
                    model_name
                ] = (
                    f"{int(round(estimate)):,}"
                    if np.isfinite(
                        estimate
                    )
                    else "NA"
                )

            elif metric in {
                "Calibration_Intercept",
                "Calibration_Slope",
                "Observed_to_Expected_Ratio",
                "Quantile_Calibration_Error",
                "Locked_Threshold",
            }:
                output_row[
                    model_name
                ] = format_number(
                    estimate,
                    digits=digits,
                )

            else:
                output_row[
                    model_name
                ] = format_estimate_ci(
                    estimate,
                    lower,
                    upper,
                    digits=digits,
                    percent=is_percent,
                )

        formatted_rows.append(
            output_row
        )

    return (
        raw_table,
        pd.DataFrame(
            formatted_rows
        ),
    )


# =============================================================================
# 8. TABLE 3A — PAIRED MODEL COMPARISONS
# =============================================================================

def build_table3a() -> pd.DataFrame:
    table = normalize_columns(
        pd.read_csv(
            require_file(
                PAIRED_DIFFERENCE_FILE
            )
        )
    )

    required = [
        "Comparison",
        "Metric",
        "Difference",
        "CI_95_Lower",
        "CI_95_Upper",
    ]

    missing = [
        column
        for column
        in required
        if column
        not in table.columns
    ]

    if missing:
        raise KeyError(
            f"Missing paired-comparison columns: {missing}"
        )

    output = table.copy()

    output[
        "Difference (95% CI)"
    ] = output.apply(
        lambda row: format_estimate_ci(
            row[
                "Difference"
            ],
            row[
                "CI_95_Lower"
            ],
            row[
                "CI_95_Upper"
            ],
            digits=3,
        ),
        axis=1,
    )

    keep_columns = [
        "Comparison",
        "Metric",
        "Difference (95% CI)",
    ]

    if "Interpretation" in output.columns:
        keep_columns.append(
            "Interpretation"
        )

    return output[
        keep_columns
    ]


# =============================================================================
# 9. TABLE 3B — DOMAIN ABLATION
# =============================================================================

DOMAIN_DISPLAY = {
    "Demographics": "Demographics",
    "Lifestyle": "Lifestyle",
    "Mental_Health": "Mental health",
    "Medical_Conditions": "Medical conditions",
    "Examination": "Examination",
    "Laboratory": "Laboratory",
    "Diet": "Diet",
}


def build_table3b() -> pd.DataFrame:
    table = normalize_columns(
        pd.read_csv(
            require_file(
                ABLATION_FILE
            )
        )
    )

    table = table.loc[
        table[
            "Experiment"
        ]
        .astype(
            str
        )
        .ne(
            "All_Domains"
        )
    ].copy()

    table[
        "Removed domain"
    ] = table[
        "Removed_Domain"
    ].map(
        DOMAIN_DISPLAY
    ).fillna(
        table[
            "Removed_Domain"
        ].astype(
            str
        ).str.replace(
            "_",
            " ",
        )
    )

    table[
        "ΔAUROC (95% CI)"
    ] = table.apply(
        lambda row: format_estimate_ci(
            row[
                "AUROC_Drop"
            ],
            row[
                "AUROC_Drop_CI_Lower"
            ],
            row[
                "AUROC_Drop_CI_Upper"
            ],
            digits=3,
        ),
        axis=1,
    )

    table[
        "ΔAUPRC (95% CI)"
    ] = table.apply(
        lambda row: format_estimate_ci(
            row[
                "AUPRC_Drop"
            ],
            row[
                "AUPRC_Drop_CI_Lower"
            ],
            row[
                "AUPRC_Drop_CI_Upper"
            ],
            digits=3,
        ),
        axis=1,
    )

    return (
        table[
            [
                "Removed domain",
                "ΔAUROC (95% CI)",
                "ΔAUPRC (95% CI)",
            ]
        ]
        .sort_values(
            "Removed domain"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 10. SUPPLEMENTARY TABLES
# =============================================================================

def build_shap_table() -> pd.DataFrame:
    source = (
        TOP_SHAP_FILE
        if TOP_SHAP_FILE.exists()
        else FALLBACK_TOP_SHAP_FILE
    )

    table = normalize_columns(
        pd.read_csv(
            require_file(
                source
            )
        )
    )

    if "rank" not in table.columns:
        table.insert(
            0,
            "rank",
            np.arange(
                1,
                len(
                    table
                )
                + 1,
            ),
        )

    if "clinical_feature" not in table.columns:
        table[
            "clinical_feature"
        ] = table.get(
            "base_feature",
            table.index.astype(
                str
            ),
        )

    if "mean_abs_grouped_shap" not in table.columns:
        raise KeyError(
            "mean_abs_grouped_shap was not found in the grouped SHAP table."
        )

    if "relative_grouped_importance_pct" not in table.columns:
        denominator = max(
            float(
                table[
                    "mean_abs_grouped_shap"
                ].sum()
            ),
            1e-15,
        )

        table[
            "relative_grouped_importance_pct"
        ] = (
            100.0
            * table[
                "mean_abs_grouped_shap"
            ]
            / denominator
        )

    keep = [
        column
        for column
        in [
            "rank",
            "clinical_feature",
            "base_feature",
            "domain",
            "feature_type",
            "mean_abs_grouped_shap",
            "relative_grouped_importance_pct",
        ]
        if column
        in table.columns
    ]

    return (
        table[
            keep
        ]
        .head(
            20
        )
        .copy()
    )


def build_subgroup_table() -> pd.DataFrame:
    if not SUBGROUP_FILE.exists():
        return pd.DataFrame(
            {
                "Note": [
                    (
                        "Subgroup performance file was not found."
                    )
                ]
            }
        )

    table = normalize_columns(
        pd.read_csv(
            SUBGROUP_FILE
        )
    )

    keep = [
        column
        for column
        in [
            "Subgroup_Type",
            "Subgroup",
            "N",
            "Events",
            "Prevalence",
            "Status",
            "AUROC",
            "AUROC_CI_Lower",
            "AUROC_CI_Upper",
            "AUPRC",
            "AUPRC_CI_Lower",
            "AUPRC_CI_Upper",
            "Brier",
            "Brier_CI_Lower",
            "Brier_CI_Upper",
            "Calibration_Intercept",
            "Calibration_Slope",
            "Recall_Sensitivity",
            "Specificity",
        ]
        if column
        in table.columns
    ]

    return table[
        keep
    ].copy()


def build_feature_table() -> pd.DataFrame:
    return normalize_columns(
        pd.read_csv(
            require_file(
                FINAL_FEATURE_FILE
            )
        )
    )


def build_missingness_table() -> pd.DataFrame:
    return normalize_columns(
        pd.read_csv(
            require_file(
                MISSINGNESS_FILE
            )
        )
    )


# =============================================================================
# 11. BUILD, SAVE, DISPLAY
# =============================================================================

def main() -> None:
    print_header(
        "CURRENT-RUN MANUSCRIPT TABLE BUILDER"
    )

    print(
        "Data directory  :",
        DATA_DIR,
    )

    print(
        "Run directory   :",
        RUN_DIR,
    )

    print(
        "Output directory:",
        OUT_DIR,
    )

    # Table 1.
    cohort = load_table1_cohort()

    observed_n = int(
        len(
            cohort
        )
    )

    observed_events = int(
        cohort[
            OUTCOME_COLUMN
        ].sum()
    )

    observed_nonevents = int(
        cohort[
            OUTCOME_COLUMN
        ].eq(
            0
        )
        .sum()
    )

    print(
        "\nAdult cohort:",
        f"N={observed_n:,}; "
        f"mobility disability={observed_events:,}; "
        f"no disability={observed_nonevents:,}",
    )

    table1_raw, table1_formatted = (
        build_table1(
            cohort
        )
    )

    table1_raw.to_csv(
        TABLE1_RAW,
        index=False,
    )

    table1_formatted.to_csv(
        TABLE1_FORMATTED,
        index=False,
    )

    # Table 2.
    table2_raw, table2_formatted = (
        build_table2()
    )

    table2_raw.to_csv(
        TABLE2_RAW,
        index=False,
    )

    table2_formatted.to_csv(
        TABLE2_FORMATTED,
        index=False,
    )

    # Table 3.
    table3a = build_table3a()

    table3a.to_csv(
        TABLE3A_FORMATTED,
        index=False,
    )

    table3b = build_table3b()

    table3b.to_csv(
        TABLE3B_FORMATTED,
        index=False,
    )

    # Supplementary tables.
    table_s1 = build_shap_table()

    table_s1.to_csv(
        TABLES1_SHAP,
        index=False,
    )

    table_s2 = build_subgroup_table()

    table_s2.to_csv(
        TABLES2_SUBGROUP,
        index=False,
    )

    table_s3 = build_feature_table()

    table_s3.to_csv(
        TABLES3_FEATURES,
        index=False,
    )

    table_s4 = build_missingness_table()

    table_s4.to_csv(
        TABLES4_MISSINGNESS,
        index=False,
    )

    # Footnotes.
    footnotes = f"""CURRENT-RUN TABLE FOOTNOTES

Table 1
- Adult analytical cohort: N={observed_n:,}.
- Mobility disability: n={observed_events:,}.
- No mobility disability: n={observed_nonevents:,}.
- Continuous variables are shown as median [interquartile range].
- Categorical variables are shown as unweighted n (% of nonmissing values).
- Absolute standardized mean differences are descriptive and are not P values.
- Age is top-coded at 80 years in the public NHANES file.
- Income-to-poverty ratio is top-coded at 5.0.
- PHQ-9 total follows the current revised pipeline derivation.
- Missing values are not included in percentage denominators.

Table 2
- Performance was evaluated in the locked held-out adult test cohort.
- Probabilities were calibrated using Platt scaling fitted to development
  out-of-fold predictions.
- Operating thresholds were selected using development out-of-fold calibrated
  probabilities and locked before test evaluation.
- Parentheses indicate bootstrap 95% confidence intervals where available.
- The prevalence-only reference model is not included in the main manuscript
  model-comparison table.

Table 3A
- Differences are calculated as XGBoost minus the comparator.
- Positive AUROC/AUPRC differences favor XGBoost.
- Negative Brier-score differences favor XGBoost.

Table 3B
- Positive ΔAUROC or ΔAUPRC indicates lower performance after removing the
  domain.
- Negative values indicate a slightly higher point estimate after removal.
- Confidence intervals are paired bootstrap 95% confidence intervals.
- Domain SHAP attribution and domain ablation answer different questions:
  attribution within the fitted model versus reliance on unique information.
"""

    NOTES_FILE.write_text(
        footnotes,
        encoding="utf-8",
    )

    # Combined Markdown.
    markdown_sections = [
        "# Current-Run Manuscript Tables",
        "",
        "## Table 1. Adult cohort characteristics by mobility-disability status",
        "",
        markdown_table(
            table1_formatted
        ),
        "",
        (
            "*Continuous variables are median [IQR]; categorical variables "
            "are unweighted n (% of nonmissing values).*"
        ),
        "",
        "## Table 2. Internal-validation model performance",
        "",
        markdown_table(
            table2_formatted
        ),
        "",
        "## Table 3A. Paired model comparisons",
        "",
        markdown_table(
            table3a
        ),
        "",
        "## Table 3B. Domain ablation",
        "",
        markdown_table(
            table3b
        ),
        "",
        "## Supplementary Table S1. Top 20 grouped SHAP features",
        "",
        markdown_table(
            table_s1
        ),
        "",
        "## Supplementary Table S2. Subgroup performance",
        "",
        markdown_table(
            table_s2
        ),
    ]

    MARKDOWN_FILE.write_text(
        "\n".join(
            markdown_sections
        ),
        encoding="utf-8",
    )

    # Display.
    display_table(
        "TABLE 1 — ADULT COHORT CHARACTERISTICS",
        table1_formatted,
    )

    display_table(
        "TABLE 2 — LOCKED-TEST MODEL PERFORMANCE",
        table2_formatted,
    )

    display_table(
        "TABLE 3A — PAIRED MODEL COMPARISONS",
        table3a,
    )

    display_table(
        "TABLE 3B — DOMAIN ABLATION",
        table3b,
    )

    display_table(
        "TABLE S1 — TOP 20 GROUPED SHAP FEATURES",
        table_s1,
    )

    display_table(
        "TABLE S2 — SUBGROUP PERFORMANCE",
        table_s2,
    )

    print_header(
        "MANUSCRIPT TABLES COMPLETE"
    )

    for path in [
        TABLE1_RAW,
        TABLE1_FORMATTED,
        TABLE2_RAW,
        TABLE2_FORMATTED,
        TABLE3A_FORMATTED,
        TABLE3B_FORMATTED,
        TABLES1_SHAP,
        TABLES2_SUBGROUP,
        TABLES3_FEATURES,
        TABLES4_MISSINGNESS,
        MARKDOWN_FILE,
        NOTES_FILE,
    ]:
        print(
            " -",
            path,
        )


if __name__ == "__main__":
    main()
