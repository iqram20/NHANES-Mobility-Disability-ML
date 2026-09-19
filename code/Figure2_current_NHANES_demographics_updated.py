#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIGURE 2 — CURRENT NHANES COHORT CHARACTERISTICS AND MOBILITY-DISABILITY PATTERNS
================================================================================

Completed current run
---------------------
/athena/madelab/scratch/iqh4001/Disability/Results/
20260603_164343_NHANES_2017_2018_Mobility_Disability_XGB

NHANES source data
------------------
/athena/madelab/scratch/iqh4001/Disability/Data/NHANES_2017_2018

Main panels
-----------
A. Age distribution by mobility-disability status.
B. BMI distribution by mobility-disability status.
C. Income-to-poverty ratio distribution by mobility-disability status.
D. Sex distribution in the analytical cohort.
E. Race/ethnicity distribution in the analytical cohort.
F. Education distribution among adults aged ≥20 years.
G. Mobility-disability prevalence by age group.
H. Mobility-disability prevalence by sex.
I. Mobility-disability prevalence by race/ethnicity.

Current-run reconstruction
--------------------------
The script:

1. Loads all NHANES 2017–2018 XPT modules in the current data directory.
2. Uses DEMO_J as the merge base and merges other modules by SEQN.
3. Reconstructs the current analytical outcome from DLQ050:
       1 = mobility disability
       2 = no mobility disability
4. Retains participants with a nonmissing binary mobility-disability outcome.
5. Audits the expected current-run cohort:
       N = 8,053
       mobility disability = 993
       no mobility disability = 7,060
6. Builds descriptive figures from the complete analytical cohort rather than
   only the held-out model test set.

Interpretation
--------------
The main figure is an unweighted description of the analytical modeling
cohort. Wilson 95% confidence intervals are used for subgroup prevalence.
The estimates should not be described as nationally representative survey
estimates.

Run in Jupyter
--------------
%matplotlib inline
%run Figure2_current_NHANES_demographics.py

Use %run rather than !python when inline display is required.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

warnings.filterwarnings("ignore")


# =============================================================================
# CURRENT-RUN PATHS AND SETTINGS
# =============================================================================

DATA_DIR = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/Disability/Data/"
    "NHANES_2017_2018"
)

RUN_DIR = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/Disability/Results/"
    "20260603_164343_NHANES_2017_2018_Mobility_Disability_XGB"
)

OUTPUT_DIR = (
    RUN_DIR
    / "figures_revised_manuscript"
    / "figure2_demographic_characteristics"
)

FIGURE_BASENAME = (
    "Figure2_Current_NHANES_Demographic_Characteristics_Updated"
)

CMAP_NAME = "plasma"
OUTPUT_DPI = 600

EXPECTED_COHORT_N = 8053
EXPECTED_EVENT_N = 993
EXPECTED_NONEVENT_N = 7060

OUTCOME_COLUMN = "mobility_disability"
OUTCOME_LABEL = {
    0: "No mobility disability",
    1: "Mobility disability",
}

RACE_ORDER = [
    "Mexican American",
    "Other Hispanic",
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Non-Hispanic Asian",
    "Other/Multiracial",
    "Missing/Other",
]

SEX_ORDER = [
    "Female",
    "Male",
    "Missing/Other",
]

EDUCATION_ORDER = [
    "Less than high school",
    "High school/GED",
    "Some college/AA",
    "College graduate or above",
    "Missing",
]


RACE_DISPLAY_MAP = {
    "Mexican American": "Mexican American",
    "Other Hispanic": "Other Hispanic",
    "Non-Hispanic White": "NH White",
    "Non-Hispanic Black": "NH Black",
    "Non-Hispanic Asian": "NH Asian",
    "Other/Multiracial": "Other/Multiracial",
    "Missing/Other": "Missing/Other",
}


# =============================================================================
# FILE AND OUTPUT HELPERS
# =============================================================================

def require_directory(
    path: pathlib.Path,
) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired directory was not found:\n{path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"\nExpected a directory but found:\n{path}"
        )

    return path


def save_json(
    data: Mapping[str, Any],
    path: pathlib.Path,
) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, pathlib.Path):
            return str(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        raise TypeError(
            f"Cannot serialize object of type {type(value)}"
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            default=convert,
        )


def save_figure_all_formats(
    figure: plt.Figure,
    png_path: pathlib.Path,
) -> None:
    figure.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    figure.savefig(
        png_path.with_suffix(".svg"),
        bbox_inches="tight",
    )

    try:
        figure.savefig(
            png_path.with_suffix(".tiff"),
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


def find_xpt_files(
    data_dir: pathlib.Path,
) -> List[pathlib.Path]:
    candidates = list(
        data_dir.glob("*.xpt")
    ) + list(
        data_dir.glob("*.XPT")
    )

    unique_files = sorted(
        set(
            path.resolve()
            for path in candidates
        )
    )

    if not unique_files:
        raise FileNotFoundError(
            f"No XPT files were found in:\n{data_dir}"
        )

    return [
        pathlib.Path(path)
        for path in unique_files
    ]


# =============================================================================
# NUMERICAL HELPERS
# =============================================================================

def clean_numeric(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


def wilson_interval(
    events: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan

    proportion = events / n

    denominator = (
        1.0
        + z**2 / n
    )

    center = (
        proportion
        + z**2
        / (
            2.0 * n
        )
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            proportion
            * (
                1.0
                - proportion
            )
            / n
            + z**2
            / (
                4.0
                * n**2
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center - half_width,
        ),
        min(
            1.0,
            center + half_width,
        ),
    )


def ordered_table(
    table: pd.DataFrame,
    order: Sequence[str],
    column: str = "group",
) -> pd.DataFrame:
    order_map = {
        value: index
        for index, value
        in enumerate(order)
    }

    output = table.copy()

    output["_order"] = (
        output[column]
        .map(order_map)
        .fillna(999)
    )

    output = (
        output
        .sort_values(
            [
                "_order",
                column,
            ]
        )
        .drop(
            columns="_order"
        )
        .reset_index(drop=True)
    )

    return output


# =============================================================================
# NHANES XPT LOADING AND CURRENT COHORT RECONSTRUCTION
# =============================================================================

def read_xpt(
    path: pathlib.Path,
) -> pd.DataFrame:
    dataframe = pd.read_sas(
        path,
        format="xport",
    )

    dataframe.columns = [
        str(column).upper()
        for column
        in dataframe.columns
    ]

    return dataframe


def load_nhanes_modules(
    data_dir: pathlib.Path,
) -> Dict[str, pd.DataFrame]:
    datasets: Dict[
        str,
        pd.DataFrame,
    ] = {}

    print("\nLoading NHANES XPT files...")

    for path in find_xpt_files(
        data_dir
    ):
        dataset_name = (
            path.stem.upper()
        )

        dataframe = read_xpt(
            path
        )

        datasets[
            dataset_name
        ] = dataframe

        print(
            f"Loaded {dataset_name:12s}: "
            f"{dataframe.shape}"
        )

    return datasets


def merge_nhanes_modules(
    datasets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if "DEMO_J" not in datasets:
        raise KeyError(
            "DEMO_J is required as the demographic merge base."
        )

    merged = (
        datasets[
            "DEMO_J"
        ]
        .copy()
    )

    for dataset_name, dataframe in (
        datasets.items()
    ):
        if dataset_name == "DEMO_J":
            continue

        if "SEQN" not in dataframe.columns:
            print(
                f"Skipping {dataset_name}: SEQN not found."
            )
            continue

        overlapping_columns = [
            column
            for column
            in dataframe.columns
            if (
                column in merged.columns
                and column != "SEQN"
            )
        ]

        merge_frame = (
            dataframe
            .drop(
                columns=overlapping_columns,
                errors="ignore",
            )
        )

        merged = merged.merge(
            merge_frame,
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

    return merged


def reconstruct_current_analytical_cohort(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    if "DLQ050" not in merged.columns:
        raise KeyError(
            "DLQ050 was not found after merging NHANES modules."
        )

    cohort = merged.copy()

    cohort[
        OUTCOME_COLUMN
    ] = np.where(
        cohort["DLQ050"].eq(1),
        1,
        np.where(
            cohort["DLQ050"].eq(2),
            0,
            np.nan,
        ),
    )

    cohort = (
        cohort
        .dropna(
            subset=[
                OUTCOME_COLUMN,
            ]
        )
        .copy()
    )

    cohort[
        OUTCOME_COLUMN
    ] = (
        cohort[
            OUTCOME_COLUMN
        ]
        .astype(int)
    )

    cohort["SEQN"] = (
        pd.to_numeric(
            cohort["SEQN"],
            errors="raise",
        )
    )

    if cohort["SEQN"].duplicated().any():
        raise RuntimeError(
            "Duplicate SEQN values were found in the analytical cohort."
        )

    observed_counts = {
        "cohort_n": int(
            len(cohort)
        ),
        "mobility_disability_n": int(
            cohort[
                OUTCOME_COLUMN
            ].sum()
        ),
        "no_mobility_disability_n": int(
            (
                cohort[
                    OUTCOME_COLUMN
                ]
                == 0
            ).sum()
        ),
    }

    expected_counts = {
        "cohort_n": EXPECTED_COHORT_N,
        "mobility_disability_n": EXPECTED_EVENT_N,
        "no_mobility_disability_n": EXPECTED_NONEVENT_N,
    }

    if observed_counts != expected_counts:
        raise RuntimeError(
            "The reconstructed current-run cohort has unexpected counts.\n"
            f"Expected: {expected_counts}\n"
            f"Observed: {observed_counts}"
        )

    print("\nCurrent-run cohort count audit: PASSED")
    print(
        f"Analytical cohort: {len(cohort):,}"
    )
    print(
        "Mobility disability:",
        f"{int(cohort[OUTCOME_COLUMN].sum()):,}",
    )
    print(
        "No mobility disability:",
        f"{int((cohort[OUTCOME_COLUMN] == 0).sum()):,}",
    )

    return cohort.reset_index(
        drop=True
    )


# =============================================================================
# DEMOGRAPHIC VARIABLE NORMALIZATION
# =============================================================================

def normalize_sex(
    series: pd.Series,
) -> pd.Series:
    numeric = clean_numeric(
        series
    )

    output = pd.Series(
        "Missing/Other",
        index=series.index,
        dtype=object,
    )

    output.loc[
        numeric.eq(1)
    ] = "Male"

    output.loc[
        numeric.eq(2)
    ] = "Female"

    return output


def normalize_race_ethnicity(
    series: pd.Series,
) -> pd.Series:
    numeric = clean_numeric(
        series
    )

    mapping = {
        1.0: "Mexican American",
        2.0: "Other Hispanic",
        3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black",
        6.0: "Non-Hispanic Asian",
        7.0: "Other/Multiracial",
    }

    output = numeric.map(
        mapping
    )

    return output.fillna(
        "Missing/Other"
    )


def normalize_education(
    series: pd.Series,
) -> pd.Series:
    numeric = clean_numeric(
        series
    )

    output = pd.Series(
        "Missing",
        index=series.index,
        dtype=object,
    )

    output.loc[
        numeric.isin(
            [
                1,
                2,
            ]
        )
    ] = "Less than high school"

    output.loc[
        numeric.eq(3)
    ] = "High school/GED"

    output.loc[
        numeric.eq(4)
    ] = "Some college/AA"

    output.loc[
        numeric.eq(5)
    ] = "College graduate or above"

    return output


def build_age_groups(
    age: pd.Series,
) -> Tuple[pd.Series, List[str]]:
    valid_age = age.dropna()

    include_under_18 = bool(
        valid_age.lt(18).any()
    )

    group = pd.Series(
        "Missing",
        index=age.index,
        dtype=object,
    )

    if include_under_18:
        group.loc[
            age.lt(18)
        ] = "<18"

    group.loc[
        age.ge(18)
        & age.lt(40)
    ] = "18–39"

    group.loc[
        age.ge(40)
        & age.lt(60)
    ] = "40–59"

    group.loc[
        age.ge(60)
        & age.lt(70)
    ] = "60–69"

    group.loc[
        age.ge(70)
    ] = "≥70"

    order = (
        [
            "<18",
            "18–39",
            "40–59",
            "60–69",
            "≥70",
        ]
        if include_under_18
        else [
            "18–39",
            "40–59",
            "60–69",
            "≥70",
        ]
    )

    return (
        group,
        order,
    )


# =============================================================================
# SOURCE TABLE BUILDERS
# =============================================================================

def build_count_table(
    group: pd.Series,
    outcome: pd.Series,
    variable: str,
    order: Sequence[str],
) -> pd.DataFrame:
    temporary = pd.DataFrame(
        {
            "group": (
                group.astype(str)
            ),
            "outcome": (
                outcome.astype(int)
            ),
        }
    )

    table = (
        temporary
        .groupby(
            "group",
            observed=True,
        )
        .agg(
            n=(
                "outcome",
                "size",
            ),
            mobility_disability_cases=(
                "outcome",
                "sum",
            ),
        )
        .reset_index()
    )

    table[
        "percent_of_analytical_cohort"
    ] = (
        100.0
        * table["n"]
        / len(
            temporary
        )
    )

    table[
        "mobility_disability_prevalence"
    ] = (
        table[
            "mobility_disability_cases"
        ]
        / table["n"].clip(
            lower=1
        )
    )

    table["variable"] = variable

    return ordered_table(
        table,
        order,
    )


def build_prevalence_table(
    group: pd.Series,
    outcome: pd.Series,
    variable: str,
    order: Sequence[str],
    exclude: Sequence[str] = (
        "Missing",
        "Missing/Other",
    ),
) -> pd.DataFrame:
    temporary = pd.DataFrame(
        {
            "group": (
                group.astype(str)
            ),
            "outcome": (
                outcome.astype(int)
            ),
        }
    )

    temporary = temporary.loc[
        ~temporary[
            "group"
        ].isin(
            exclude
        )
    ].copy()

    grouped = (
        temporary
        .groupby(
            "group",
            observed=True,
        )
        .agg(
            n=(
                "outcome",
                "size",
            ),
            cases=(
                "outcome",
                "sum",
            ),
        )
        .reset_index()
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for row in grouped.itertuples():
        lower, upper = wilson_interval(
            int(
                row.cases
            ),
            int(
                row.n
            ),
        )

        rows.append(
            {
                "variable": variable,
                "group": str(
                    row.group
                ),
                "n": int(
                    row.n
                ),
                "cases": int(
                    row.cases
                ),
                "prevalence": (
                    row.cases
                    / row.n
                ),
                "ci_lower_95": lower,
                "ci_upper_95": upper,
            }
        )

    table = pd.DataFrame(
        rows
    )

    if table.empty:
        return table

    return ordered_table(
        table,
        order,
    )


def build_continuous_summary(
    variable_name: str,
    values: pd.Series,
    outcome: pd.Series,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    groups = [
        (
            "Overall",
            pd.Series(
                True,
                index=values.index,
            ),
        ),
        (
            OUTCOME_LABEL[0],
            outcome.eq(0),
        ),
        (
            OUTCOME_LABEL[1],
            outcome.eq(1),
        ),
    ]

    for group_name, mask in groups:
        group_values = (
            values.loc[
                mask
            ]
            .dropna()
        )

        if group_values.empty:
            continue

        rows.append(
            {
                "variable": variable_name,
                "outcome_group": group_name,
                "n_nonmissing": int(
                    len(
                        group_values
                    )
                ),
                "mean": float(
                    group_values.mean()
                ),
                "sd": float(
                    group_values.std(
                        ddof=1
                    )
                ),
                "median": float(
                    group_values.median()
                ),
                "q1": float(
                    group_values.quantile(
                        0.25
                    )
                ),
                "q3": float(
                    group_values.quantile(
                        0.75
                    )
                ),
                "minimum": float(
                    group_values.min()
                ),
                "maximum": float(
                    group_values.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_outcome_histogram(
    axis: plt.Axes,
    values: pd.Series,
    outcome: pd.Series,
    title: str,
    xlabel: str,
    bins: np.ndarray,
    no_disability_color,
    disability_color,
    *,
    show_nonmissing_n: bool = False,
) -> None:
    no_disability_values = (
        values.loc[
            outcome.eq(0)
        ]
        .dropna()
    )

    disability_values = (
        values.loc[
            outcome.eq(1)
        ]
        .dropna()
    )

    denominator_label = (
        "nonmissing n"
        if show_nonmissing_n
        else "n"
    )

    axis.hist(
        no_disability_values,
        bins=bins,
        density=True,
        alpha=0.52,
        color=no_disability_color,
        label=(
            "No mobility disability "
            f"({denominator_label}={len(no_disability_values):,})"
        ),
    )

    axis.hist(
        disability_values,
        bins=bins,
        density=True,
        alpha=0.68,
        color=disability_color,
        label=(
            "Mobility disability "
            f"({denominator_label}={len(disability_values):,})"
        ),
    )

    if not no_disability_values.empty:
        axis.axvline(
            no_disability_values.median(),
            color=no_disability_color,
            linestyle="--",
            linewidth=1.3,
        )

    if not disability_values.empty:
        axis.axvline(
            disability_values.median(),
            color=disability_color,
            linestyle="--",
            linewidth=1.3,
        )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        "Density"
    )

    axis.set_title(
        title,
        fontweight="bold",
        loc="left",
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    axis.legend(
        frameon=False,
        fontsize=8.1,
    )


def plot_count_bars(
    axis: plt.Axes,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    cmap,
    *,
    rotate: float = 0.0,
    display_label_map: Optional[Mapping[str, str]] = None,
) -> None:
    raw_labels = (
        table[
            "group"
        ]
        .astype(str)
        .tolist()
    )

    labels = [
        (
            display_label_map.get(label, label)
            if display_label_map is not None
            else label
        )
        for label in raw_labels
    ]

    values = (
        table["n"]
        .to_numpy(
            dtype=float
        )
    )

    positions = np.arange(
        len(table)
    )

    colors = [
        cmap(value)
        for value
        in np.linspace(
            0.15,
            0.88,
            max(
                len(table),
                1,
            ),
        )
    ]

    bars = axis.bar(
        positions,
        values,
        color=colors,
        edgecolor="none",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        labels,
        rotation=rotate,
        ha=(
            "right"
            if rotate
            else "center"
        ),
    )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        "Participants"
    )

    axis.set_title(
        title,
        fontweight="bold",
        loc="left",
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    maximum = max(
        float(
            values.max()
        ),
        1.0,
    )

    axis.set_ylim(
        0.0,
        maximum * 1.22,
    )

    for bar, row in zip(
        bars,
        table.itertuples(),
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height()
            + maximum
            * 0.025,
            (
                f"{int(row.n):,}\n"
                f"({row.percent_of_analytical_cohort:.1f}%)"
            ),
            ha="center",
            va="bottom",
            fontsize=7.8,
        )


def plot_prevalence_bars(
    axis: plt.Axes,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    cmap,
    *,
    rotate: float = 0.0,
    display_label_map: Optional[Mapping[str, str]] = None,
) -> None:
    if table.empty:
        axis.text(
            0.5,
            0.5,
            "Data unavailable",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )

        axis.set_title(
            title,
            fontweight="bold",
            loc="left",
        )

        return

    raw_labels = (
        table[
            "group"
        ]
        .astype(str)
        .tolist()
    )

    labels = [
        (
            display_label_map.get(label, label)
            if display_label_map is not None
            else label
        )
        for label in raw_labels
    ]

    prevalence_percent = (
        100.0
        * table[
            "prevalence"
        ]
        .to_numpy(
            dtype=float
        )
    )

    lower_percent = (
        100.0
        * table[
            "ci_lower_95"
        ]
        .to_numpy(
            dtype=float
        )
    )

    upper_percent = (
        100.0
        * table[
            "ci_upper_95"
        ]
        .to_numpy(
            dtype=float
        )
    )

    error = np.vstack(
        [
            prevalence_percent
            - lower_percent,
            upper_percent
            - prevalence_percent,
        ]
    )

    positions = np.arange(
        len(table)
    )

    colors = [
        cmap(value)
        for value
        in np.linspace(
            0.15,
            0.90,
            len(table),
        )
    ]

    bars = axis.bar(
        positions,
        prevalence_percent,
        yerr=error,
        capsize=3,
        color=colors,
        edgecolor="none",
        error_kw={
            "elinewidth": 1.0,
            "capthick": 1.0,
        },
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        labels,
        rotation=rotate,
        ha=(
            "right"
            if rotate
            else "center"
        ),
    )

    axis.set_xlabel(
        xlabel
    )

    axis.set_ylabel(
        "Mobility disability prevalence (%)"
    )

    axis.set_title(
        title,
        fontweight="bold",
        loc="left",
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    maximum = max(
        float(
            upper_percent.max()
        ),
        1.0,
    )

    axis.set_ylim(
        0.0,
        maximum * 1.32,
    )

    for bar, row, prevalence_value in zip(
        bars,
        table.itertuples(),
        prevalence_percent,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            100.0
            * row.ci_upper_95
            + maximum
            * 0.035,
            (
                f"{prevalence_value:.1f}%\n"
                f"{int(row.cases)}/{int(row.n)}"
            ),
            ha="center",
            va="bottom",
            fontsize=7.6,
        )


# =============================================================================
# MAIN FIGURE
# =============================================================================

def build_figure2() -> None:
    require_directory(
        DATA_DIR
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_style()

    print("=" * 100)
    print("FIGURE 2 — CURRENT NHANES DEMOGRAPHIC CHARACTERISTICS")
    print("=" * 100)
    print("Data directory  :", DATA_DIR)
    print("Run directory   :", RUN_DIR)
    print("Output directory:", OUTPUT_DIR)

    datasets = load_nhanes_modules(
        DATA_DIR
    )

    merged = merge_nhanes_modules(
        datasets
    )

    print(
        "\nMerged shape:",
        merged.shape,
    )

    cohort = (
        reconstruct_current_analytical_cohort(
            merged
        )
    )

    outcome = (
        cohort[
            OUTCOME_COLUMN
        ]
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # Core demographic variables
    # -------------------------------------------------------------------------
    required_columns = [
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
    ]

    missing_required = [
        column
        for column
        in required_columns
        if column not in cohort.columns
    ]

    if missing_required:
        raise KeyError(
            "Required demographic columns are missing:\n"
            f"{missing_required}"
        )

    age = clean_numeric(
        cohort["RIDAGEYR"]
    )

    bmi = (
        clean_numeric(
            cohort["BMXBMI"]
        )
        if "BMXBMI" in cohort.columns
        else pd.Series(
            np.nan,
            index=cohort.index,
        )
    )

    bmi = bmi.where(
        bmi.between(
            10,
            80,
            inclusive="both",
        )
    )

    income_to_poverty = clean_numeric(
        cohort["INDFMPIR"]
    )

    income_to_poverty = income_to_poverty.where(
        income_to_poverty.between(
            0,
            5,
            inclusive="both",
        )
    )

    sex = normalize_sex(
        cohort["RIAGENDR"]
    )

    race = normalize_race_ethnicity(
        cohort["RIDRETH3"]
    )

    age_group, age_group_order = (
        build_age_groups(
            age
        )
    )

    education_eligible = age.ge(
        20
    )

    education = normalize_education(
        cohort["DMDEDUC2"]
    )

    # -------------------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------------------
    sex_distribution = build_count_table(
        sex,
        outcome,
        "Sex",
        SEX_ORDER,
    )

    race_distribution = build_count_table(
        race,
        outcome,
        "Race/Ethnicity",
        RACE_ORDER,
    )

    education_distribution = (
        build_count_table(
            education.loc[
                education_eligible
            ],
            outcome.loc[
                education_eligible
            ],
            "Education among adults aged ≥20 years",
            EDUCATION_ORDER,
        )
    )

    mortality_by_age = (
        build_prevalence_table(
            age_group,
            outcome,
            "Age group",
            age_group_order,
        )
    )

    mortality_by_sex = (
        build_prevalence_table(
            sex,
            outcome,
            "Sex",
            SEX_ORDER,
        )
    )

    mortality_by_race = (
        build_prevalence_table(
            race,
            outcome,
            "Race/Ethnicity",
            RACE_ORDER,
        )
    )

    demographic_distribution_table = (
        pd.concat(
            [
                sex_distribution,
                race_distribution,
                education_distribution,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    prevalence_table = pd.concat(
        [
            mortality_by_age,
            mortality_by_sex,
            mortality_by_race,
        ],
        ignore_index=True,
        sort=False,
    )

    continuous_summary = pd.concat(
        [
            build_continuous_summary(
                "Age",
                age,
                outcome,
            ),
            build_continuous_summary(
                "Body Mass Index",
                bmi,
                outcome,
            ),
            build_continuous_summary(
                "Income-to-Poverty Ratio",
                income_to_poverty,
                outcome,
            ),
        ],
        ignore_index=True,
    )

    cohort_summary = pd.DataFrame(
        [
            {
                "analytical_cohort_n": int(
                    len(cohort)
                ),
                "mobility_disability_n": int(
                    outcome.sum()
                ),
                "no_mobility_disability_n": int(
                    outcome.eq(0).sum()
                ),
                "mobility_disability_prevalence": float(
                    outcome.mean()
                ),
                "education_panel_eligible_age": "≥20 years",
                "education_panel_denominator": int(
                    education_eligible.sum()
                ),
                "figure_estimates_weighted": False,
                "confidence_interval_method": (
                    "Wilson 95% confidence interval"
                ),
            }
        ]
    )

    # -------------------------------------------------------------------------
    # Main 3×3 figure
    # -------------------------------------------------------------------------
    cmap = cm.get_cmap(
        CMAP_NAME
    )

    no_disability_color = cmap(
        0.18
    )

    disability_color = cmap(
        0.82
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(19, 16.5),
    )

    (
        (
            axis_a,
            axis_b,
            axis_c,
        ),
        (
            axis_d,
            axis_e,
            axis_f,
        ),
        (
            axis_g,
            axis_h,
            axis_i,
        ),
    ) = axes

    figure.suptitle(
        "Cohort Characteristics and Mobility Disability Prevalence",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    figure.text(
        0.5,
        0.958,
        (
            f"NHANES 2017–2018 analytical cohort: "
            f"N={len(cohort):,}; "
            f"mobility disability={int(outcome.sum()):,} "
            f"({100.0 * outcome.mean():.1f}%)"
        ),
        ha="center",
        va="center",
        fontsize=11.0,
    )

    # Panel A — Age distribution
    age_valid = age.dropna()

    if not age_valid.empty:
        age_lower = max(
            0.0,
            np.floor(
                age_valid.quantile(
                    0.005
                )
                / 5.0
            )
            * 5.0,
        )

        age_upper = (
            np.ceil(
                age_valid.quantile(
                    0.995
                )
                / 5.0
            )
            * 5.0
        )

        age_bins = np.arange(
            age_lower,
            age_upper + 5.0,
            5.0,
        )

        plot_outcome_histogram(
            axis_a,
            age,
            outcome,
            "Age distribution by mobility-disability status",
            "Age (years; 80 represents ≥80)",
            age_bins,
            no_disability_color,
            disability_color,
        )

        # NHANES public-use age is top-coded at 80 years.
        current_ticks = axis_a.get_xticks()
        current_labels = [
            "80+"
            if np.isclose(tick, 80.0)
            else f"{tick:g}"
            for tick in current_ticks
        ]
        axis_a.set_xticks(current_ticks)
        axis_a.set_xticklabels(current_labels)

    # Panel B — BMI distribution
    bmi_valid = bmi.dropna()

    if not bmi_valid.empty:
        bmi_lower = max(
            10.0,
            np.floor(
                bmi_valid.quantile(
                    0.005
                )
                / 2.0
            )
            * 2.0,
        )

        bmi_upper = min(
            80.0,
            np.ceil(
                bmi_valid.quantile(
                    0.995
                )
                / 2.0
            )
            * 2.0,
        )

        bmi_bins = np.arange(
            bmi_lower,
            bmi_upper + 2.0,
            2.0,
        )

        plot_outcome_histogram(
            axis_b,
            bmi,
            outcome,
            "BMI distribution by mobility-disability status",
            "Body mass index (kg/m²)",
            bmi_bins,
            no_disability_color,
            disability_color,
            show_nonmissing_n=True,
        )

    else:
        axis_b.text(
            0.5,
            0.5,
            "BMI unavailable",
            transform=axis_b.transAxes,
            ha="center",
            va="center",
        )

    # Panel C — Income-to-poverty ratio distribution
    pir_bins = np.linspace(
        0,
        5,
        21,
    )

    plot_outcome_histogram(
        axis_c,
        income_to_poverty,
        outcome,
        "Income-to-poverty ratio by mobility-disability status",
        "Family income-to-poverty ratio (5 represents ≥5)",
        pir_bins,
        no_disability_color,
        disability_color,
        show_nonmissing_n=True,
    )

    # Panels D–F — Cohort composition
    plot_count_bars(
        axis_d,
        sex_distribution,
        "Sex distribution",
        "Sex",
        cmap,
    )

    plot_count_bars(
        axis_e,
        race_distribution,
        "Race/ethnicity distribution",
        "Race/Ethnicity",
        cmap,
        rotate=30.0,
        display_label_map=RACE_DISPLAY_MAP,
    )

    axis_e.tick_params(
        axis="x",
        labelsize=8.5,
    )

    plot_count_bars(
        axis_f,
        education_distribution,
        "Education distribution among adults aged ≥20 years",
        "Educational attainment",
        cmap,
        rotate=24.0,
    )

    # Panels G–I — Mobility-disability prevalence
    plot_prevalence_bars(
        axis_g,
        mortality_by_age,
        "Prevalence of mobility disability by age group",
        "Age group",
        cmap,
    )

    plot_prevalence_bars(
        axis_h,
        mortality_by_sex,
        "Prevalence of mobility disability by sex",
        "Sex",
        cmap,
    )

    plot_prevalence_bars(
        axis_i,
        mortality_by_race,
        "Prevalence of mobility disability by race/ethnicity",
        "Race/Ethnicity",
        cmap,
        rotate=30.0,
        display_label_map=RACE_DISPLAY_MAP,
    )

    axis_i.tick_params(
        axis="x",
        labelsize=8.5,
    )

    for axis, label in zip(
        axes.flatten(),
        list(
            "ABCDEFGHI"
        ),
    ):
        add_panel_label(
            axis,
            label,
        )

    figure.text(
        0.03,
        0.015,
        (
            "Counts and prevalence estimates describe the unweighted analytical "
            "cohort. Error bars indicate Wilson 95% confidence intervals. "
            "Participants aged ≥80 years are coded as 80, and income-to-poverty "
            "ratios ≥5.0 are coded as 5.0. Education is summarized among adults "
            "aged ≥20 years."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="dimgray",
    )

    figure.tight_layout(
        rect=(
            0.03,
            0.045,
            0.99,
            0.94,
        ),
        h_pad=2.8,
        w_pad=2.2,
    )

    figure_path = (
        OUTPUT_DIR
        / f"{FIGURE_BASENAME}.png"
    )

    save_figure_all_formats(
        figure,
        figure_path,
    )

    plt.show()
    plt.close(
        figure
    )

    # -------------------------------------------------------------------------
    # Save numerical source data
    # -------------------------------------------------------------------------
    cohort_summary_path = (
        OUTPUT_DIR
        / "Figure2_demographic_cohort_summary.csv"
    )

    continuous_summary_path = (
        OUTPUT_DIR
        / "Figure2_continuous_characteristics.csv"
    )

    distribution_path = (
        OUTPUT_DIR
        / "Figure2_demographic_distributions.csv"
    )

    prevalence_path = (
        OUTPUT_DIR
        / (
            "Figure2_mobility_disability_"
            "prevalence_with_Wilson_CIs.csv"
        )
    )

    participant_source_path = (
        OUTPUT_DIR
        / "Figure2_participant_level_source_data.csv"
    )

    manuscript_path = (
        OUTPUT_DIR
        / "Figure2_manuscript_numerics.txt"
    )

    summary_path = (
        OUTPUT_DIR
        / "Figure2_demographic_summary.json"
    )

    cohort_summary.to_csv(
        cohort_summary_path,
        index=False,
    )

    continuous_summary.to_csv(
        continuous_summary_path,
        index=False,
    )

    demographic_distribution_table.to_csv(
        distribution_path,
        index=False,
    )

    prevalence_table.to_csv(
        prevalence_path,
        index=False,
    )

    participant_source = pd.DataFrame(
        {
            "SEQN": cohort["SEQN"],
            "mobility_disability": outcome,
            "mobility_disability_label": (
                outcome.map(
                    OUTCOME_LABEL
                )
            ),
            "age_years": age,
            "age_group": age_group,
            "body_mass_index": bmi,
            "income_to_poverty_ratio": (
                income_to_poverty
            ),
            "sex": sex,
            "race_ethnicity": race,
            "education_age_20_or_older": np.where(
                education_eligible,
                education,
                "Not eligible (<20 years)",
            ),
        }
    )

    participant_source.to_csv(
        participant_source_path,
        index=False,
    )

    manuscript_lines = [
        "FIGURE 2 — CURRENT NHANES MANUSCRIPT NUMERICAL RESULTS",
        "=" * 82,
        "",
        (
            f"The analytical cohort included {len(cohort):,} participants, "
            f"including {int(outcome.sum()):,} with mobility disability "
            f"({100.0 * outcome.mean():.1f}%)."
        ),
        "",
        "CONTINUOUS CHARACTERISTICS",
    ]

    for row in continuous_summary.itertuples():
        manuscript_lines.append(
            (
                f"{row.variable}, {row.outcome_group}: "
                f"N={int(row.n_nonmissing):,}; "
                f"mean ± SD {row.mean:.2f} ± {row.sd:.2f}; "
                f"median (IQR) {row.median:.2f} "
                f"({row.q1:.2f}–{row.q3:.2f})."
            )
        )

    manuscript_lines.extend(
        [
            "",
            "MOBILITY-DISABILITY PREVALENCE",
        ]
    )

    for row in prevalence_table.itertuples():
        manuscript_lines.append(
            (
                f"{row.variable}, {row.group}: "
                f"{100.0 * row.prevalence:.1f}% "
                f"(95% CI {100.0 * row.ci_lower_95:.1f}–"
                f"{100.0 * row.ci_upper_95:.1f}); "
                f"{int(row.cases)}/{int(row.n)} participants."
            )
        )

    manuscript_path.write_text(
        "\n".join(
            manuscript_lines
        ),
        encoding="utf-8",
    )

    summary = {
        "analysis": (
            "NHANES 2017-2018 mobility-disability "
            "analytical-cohort demographic characteristics"
        ),
        "data_directory": str(
            DATA_DIR
        ),
        "run_directory": str(
            RUN_DIR
        ),
        "output_directory": str(
            OUTPUT_DIR
        ),
        "analytical_cohort_n": int(
            len(cohort)
        ),
        "mobility_disability_n": int(
            outcome.sum()
        ),
        "no_mobility_disability_n": int(
            outcome.eq(0).sum()
        ),
        "mobility_disability_prevalence": float(
            outcome.mean()
        ),
        "figure_estimates_weighted": False,
        "confidence_interval_method": (
            "Wilson 95% confidence interval"
        ),
        "education_eligibility": (
            "Adults aged 20 years or older"
        ),
        "age_top_coding": (
            "RIDAGEYR=80 represents participants aged 80 years or older"
        ),
        "income_to_poverty_top_coding": (
            "INDFMPIR=5.0 represents ratios of 5.0 or higher"
        ),
        "education_panel_denominator": int(
            education_eligible.sum()
        ),
        "main_figure": str(
            figure_path
        ),
        "cohort_summary_table": str(
            cohort_summary_path
        ),
        "continuous_characteristics_table": str(
            continuous_summary_path
        ),
        "demographic_distribution_table": str(
            distribution_path
        ),
        "prevalence_table": str(
            prevalence_path
        ),
        "participant_source_data": str(
            participant_source_path
        ),
        "manuscript_numerics": str(
            manuscript_path
        ),
        "interpretation_note": (
            "The figure describes the unweighted analytical modeling cohort "
            "and should not be interpreted as a nationally representative "
            "NHANES population estimate."
        ),
    }

    save_json(
        summary,
        summary_path,
    )

    print("\n" + "=" * 100)
    print("CURRENT NHANES FIGURE 2 COMPLETE")
    print("=" * 100)
    print("Main figure:", figure_path)
    print("Output directory:", OUTPUT_DIR)

    print("\nContinuous characteristics:")
    print(
        continuous_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print("\nMobility-disability prevalence:")
    print(
        prevalence_table.round(
            6
        ).to_string(
            index=False
        )
    )

    print("\nManuscript-ready numerics:")
    print(
        "\n".join(
            manuscript_lines
        )
    )


if __name__ == "__main__":
    build_figure2()
