#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NHANES 2017–2018 ADULT MOBILITY-DISABILITY CLASSIFICATION
NATIVE TREESHAP DOMAIN EXPLAINABILITY — FULL INTERNAL VALIDATION
STANDALONE 1 CELL / PYTHON SCRIPT
==========================================================================

Uses
----
1. Patient-level grouped domain SHAP importance
2. Fixed-split domain ablation with bootstrap 95% confidence intervals
3. Relative domain composition
4. Agreement between domain SHAP attribution and domain ablation

Current run
-----------
/athena/madelab/scratch/iqh4001/Disability/Results/
20260727_004939_NHANES_2017_2018_Adult_
Mobility_Disability_Classification_Revised

Current-run inputs
------------------
shap/domain_shap_importance.csv
domain_ablation/domain_ablation_results.csv
tables/test_model_performance.csv

Interpretation
--------------
Domain SHAP measures attribution within the fitted XGBoost model.
Domain ablation measures the model's reliance on unique information from each
domain. These quantities are related but are not expected to be identical,
because domains can contain overlapping or redundant information.

Jupyter
-------
%matplotlib inline
%run Figure_current_run_domain_explainability.py

Use %run rather than !python when inline display is required.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
ABLATION_DIR = RUN_DIR / "domain_ablation"

OUT_DIR = (
    RUN_DIR
    / "figures_reviewer_revision_native_treeshap"
    / "domain_explainability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOMAIN_SHAP_FILE = (
    SHAP_DIR
    / "domain_shap_importance.csv"
)

ABLATION_FILE = (
    ABLATION_DIR
    / "domain_ablation_results.csv"
)

PERFORMANCE_FILE = (
    TABLE_DIR
    / "test_model_performance.csv"
)

OUTPUT_BASENAME = (
    "Figure4_Domain_Explainability_NativeTreeSHAP_FULL1171"
)

OUTPUT_PNG = (
    OUT_DIR
    / f"{OUTPUT_BASENAME}.png"
)

OUTPUT_PDF = (
    OUT_DIR
    / f"{OUTPUT_BASENAME}.pdf"
)

OUTPUT_SVG = (
    OUT_DIR
    / f"{OUTPUT_BASENAME}.svg"
)

OUTPUT_TIFF = (
    OUT_DIR
    / f"{OUTPUT_BASENAME}.tiff"
)

SOURCE_TABLE_FILE = (
    OUT_DIR
    / "Figure_Current_Run_Domain_Explainability_Source_Data.csv"
)

SUMMARY_FILE = (
    OUT_DIR
    / "Figure_Current_Run_Domain_Explainability_Summary.json"
)


# =============================================================================
# 2. STYLE AND FIGURE SETTINGS
# =============================================================================

CMAP_NAME = "plasma"
OUTPUT_DPI = 600

FIGURE_WIDTH = 14.0
FIGURE_HEIGHT = 11.5

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# 3. DISPLAY LABELS
# =============================================================================

DOMAIN_DISPLAY_MAP = {
    "Demographics": "Demographics",
    "Lifestyle": "Lifestyle",
    "Mental_Health": "Mental Health",
    "Medical_Conditions": "Medical Conditions",
    "Examination": "Examination",
    "Laboratory": "Laboratory",
    "Diet": "Diet",
}


def pretty_domain(
    domain: str,
) -> str:
    return DOMAIN_DISPLAY_MAP.get(
        str(
            domain
        ),
        str(
            domain
        ).replace(
            "_",
            " ",
        ),
    )


# =============================================================================
# 4. HELPERS
# =============================================================================

def require_file(
    path: Path,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired current-run file was not found:\n{path}"
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


def require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    source: Path,
) -> None:
    missing = [
        column
        for column
        in columns
        if column
        not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"\nMissing columns in {source.name}:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{dataframe.columns.tolist()}"
        )


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


def errorbar_components(
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            np.maximum(
                estimate
                - lower,
                0.0,
            ),
            np.maximum(
                upper
                - estimate,
                0.0,
            ),
        ]
    )


# =============================================================================
# 5. LOAD CURRENT-RUN FILES
# =============================================================================

print("=" * 100)
print(
    "NHANES 2017–2018 ADULT MOBILITY-DISABILITY CLASSIFICATION"
)
print(
    "NATIVE TREESHAP DOMAIN EXPLAINABILITY — FULL INTERNAL VALIDATION"
)
print("=" * 100)
print("Run directory   :", RUN_DIR)
print("Domain SHAP file:", DOMAIN_SHAP_FILE)
print("Ablation file   :", ABLATION_FILE)
print("Performance file:", PERFORMANCE_FILE)
print("Output directory:", OUT_DIR)

domain_shap = normalize_columns(
    pd.read_csv(
        require_file(
            DOMAIN_SHAP_FILE
        )
    )
)

ablation = normalize_columns(
    pd.read_csv(
        require_file(
            ABLATION_FILE
        )
    )
)

performance = normalize_columns(
    pd.read_csv(
        require_file(
            PERFORMANCE_FILE
        )
    )
)


# =============================================================================
# 6. VALIDATE INPUTS
# =============================================================================

require_columns(
    domain_shap,
    [
        "domain",
        "mean_abs_domain_shap",
        "relative_importance",
        "relative_importance_pct",
    ],
    DOMAIN_SHAP_FILE,
)

require_columns(
    ablation,
    [
        "Experiment",
        "Removed_Domain",
        "AUROC",
        "AUPRC",
        "AUROC_Drop",
        "AUPRC_Drop",
        "AUROC_Drop_CI_Lower",
        "AUROC_Drop_CI_Upper",
        "AUPRC_Drop_CI_Lower",
        "AUPRC_Drop_CI_Upper",
    ],
    ABLATION_FILE,
)

require_columns(
    performance,
    [
        "Model",
        "AUROC",
        "AUPRC",
        "Brier",
    ],
    PERFORMANCE_FILE,
)


# =============================================================================
# 7. PREPARE CURRENT-RUN DATA
# =============================================================================

domain_shap = (
    domain_shap
    .copy()
)

domain_shap[
    "domain"
] = (
    domain_shap[
        "domain"
    ]
    .astype(str)
    .str.strip()
)

domain_shap[
    "Domain_Label"
] = domain_shap[
    "domain"
].map(
    pretty_domain
)

ablation = (
    ablation.loc[
        ablation[
            "Experiment"
        ]
        .astype(str)
        .ne(
            "All_Domains"
        )
    ]
    .copy()
)

ablation[
    "domain"
] = (
    ablation[
        "Removed_Domain"
    ]
    .astype(str)
    .str.strip()
)

ablation[
    "Domain_Label"
] = ablation[
    "domain"
].map(
    pretty_domain
)

# Keep the domain ordering based on SHAP attribution.
domain_order = (
    domain_shap
    .sort_values(
        "relative_importance",
        ascending=False,
    )[
        "domain"
    ]
    .astype(str)
    .tolist()
)

all_domains = list(
    dict.fromkeys(
        domain_order
        + ablation[
            "domain"
        ]
        .astype(str)
        .tolist()
    )
)

domain_table = pd.DataFrame(
    {
        "domain": all_domains
    }
)

domain_table = domain_table.merge(
    domain_shap[
        [
            "domain",
            "mean_abs_domain_shap",
            "relative_importance",
            "relative_importance_pct",
        ]
    ],
    on="domain",
    how="left",
)

domain_table = domain_table.merge(
    ablation[
        [
            "domain",
            "AUROC",
            "AUPRC",
            "AUROC_Drop",
            "AUPRC_Drop",
            "AUROC_Drop_CI_Lower",
            "AUROC_Drop_CI_Upper",
            "AUPRC_Drop_CI_Lower",
            "AUPRC_Drop_CI_Upper",
        ]
    ],
    on="domain",
    how="left",
)

numeric_columns = [
    "mean_abs_domain_shap",
    "relative_importance",
    "relative_importance_pct",
    "AUROC",
    "AUPRC",
    "AUROC_Drop",
    "AUPRC_Drop",
    "AUROC_Drop_CI_Lower",
    "AUROC_Drop_CI_Upper",
    "AUPRC_Drop_CI_Lower",
    "AUPRC_Drop_CI_Upper",
]

for column in numeric_columns:
    domain_table[
        column
    ] = pd.to_numeric(
        domain_table[
            column
        ],
        errors="coerce",
    )

domain_table[
    "relative_importance"
] = domain_table[
    "relative_importance"
].fillna(
    0.0
)

domain_table[
    "relative_importance_pct"
] = domain_table[
    "relative_importance_pct"
].fillna(
    0.0
)

domain_table[
    "AUROC_Drop"
] = domain_table[
    "AUROC_Drop"
].fillna(
    0.0
)

domain_table[
    "AUPRC_Drop"
] = domain_table[
    "AUPRC_Drop"
].fillna(
    0.0
)

domain_table[
    "Domain_Label"
] = domain_table[
    "domain"
].map(
    pretty_domain
)

domain_table[
    "_order"
] = domain_table[
    "domain"
].map(
    {
        domain: index
        for index, domain
        in enumerate(
            domain_order
        )
    }
).fillna(
    999
)

domain_table = (
    domain_table
    .sort_values(
        "_order"
    )
    .drop(
        columns="_order"
    )
    .reset_index(
        drop=True
    )
)

xgboost_rows = performance.loc[
    performance[
        "Model"
    ]
    .astype(str)
    .str.strip()
    .eq(
        "XGBoost"
    )
]

if xgboost_rows.empty:
    raise ValueError(
        "XGBoost was not found in test_model_performance.csv."
    )

xgboost_row = xgboost_rows.iloc[
    0
]

xgboost_auroc = float(
    xgboost_row[
        "AUROC"
    ]
)

xgboost_auprc = float(
    xgboost_row[
        "AUPRC"
    ]
)

xgboost_brier = float(
    xgboost_row[
        "Brier"
    ]
)


# =============================================================================
# 8. PRINT RESULTS
# =============================================================================

print("\n" + "=" * 100)
print(
    "DOMAIN SHAP ATTRIBUTION"
)
print("=" * 100)

print(
    domain_table[
        [
            "Domain_Label",
            "relative_importance",
            "relative_importance_pct",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print(
    "DOMAIN ABLATION"
)
print("=" * 100)

print(
    domain_table[
        [
            "Domain_Label",
            "AUROC_Drop",
            "AUROC_Drop_CI_Lower",
            "AUROC_Drop_CI_Upper",
            "AUPRC_Drop",
            "AUPRC_Drop_CI_Lower",
            "AUPRC_Drop_CI_Upper",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# 9. DOMAIN COLORS
# =============================================================================

domains = (
    domain_table[
        "domain"
    ]
    .astype(str)
    .tolist()
)

cmap = mpl.colormaps[
    CMAP_NAME
]

domain_colors: Dict[
    str,
    tuple,
] = {
    domain: cmap(
        position
    )
    for domain, position
    in zip(
        domains,
        np.linspace(
            0.08,
            0.92,
            max(
                len(
                    domains
                ),
                1,
            ),
        ),
    )
}


# =============================================================================
# 10. FIGURE LAYOUT
# =============================================================================

figure = plt.figure(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    )
)

grid = figure.add_gridspec(
    3,
    2,
    height_ratios=[
        5,
        4,
        5,
    ],
    width_ratios=[
        1,
        1,
    ],
    hspace=0.48,
    wspace=0.36,
)

axis_a = figure.add_subplot(
    grid[
        0,
        0,
    ]
)

axis_b = figure.add_subplot(
    grid[
        0,
        1,
    ]
)

axis_c = figure.add_subplot(
    grid[
        1,
        0,
    ]
)

axis_legend = figure.add_subplot(
    grid[
        1,
        1,
    ]
)

axis_d = figure.add_subplot(
    grid[
        2,
        :,
    ]
)


# =============================================================================
# 11. PANEL A — DOMAIN SHAP ATTRIBUTION
# =============================================================================

panel_a_table = (
    domain_table
    .sort_values(
        "relative_importance",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)

axis_a.barh(
    panel_a_table[
        "Domain_Label"
    ],
    panel_a_table[
        "relative_importance"
    ],
    color=[
        domain_colors[
            domain
        ]
        for domain
        in panel_a_table[
            "domain"
        ]
    ],
    edgecolor="none",
)

axis_a.set_xlabel(
    "Relative domain SHAP attribution"
)

axis_a.set_title(
    "A. Domain SHAP attribution",
    loc="left",
)

axis_a.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.22,
)

maximum_shap = max(
    float(
        panel_a_table[
            "relative_importance"
        ].max()
    ),
    0.01,
)

axis_a.set_xlim(
    0.0,
    maximum_shap
    * 1.23,
)

for position, row in enumerate(
    panel_a_table.itertuples()
):
    axis_a.text(
        row.relative_importance
        + maximum_shap
        * 0.018,
        position,
        f"{row.relative_importance_pct:.1f}%",
        va="center",
        fontsize=9,
    )


# =============================================================================
# 12. PANEL B — DOMAIN ABLATION WITH 95% CI
# =============================================================================

panel_b_table = (
    domain_table
    .sort_values(
        "AUROC_Drop",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)

auroc_drop = panel_b_table[
    "AUROC_Drop"
].to_numpy(
    dtype=float
)

auroc_lower = panel_b_table[
    "AUROC_Drop_CI_Lower"
].to_numpy(
    dtype=float
)

auroc_upper = panel_b_table[
    "AUROC_Drop_CI_Upper"
].to_numpy(
    dtype=float
)

axis_b.barh(
    panel_b_table[
        "Domain_Label"
    ],
    auroc_drop,
    xerr=errorbar_components(
        auroc_drop,
        auroc_lower,
        auroc_upper,
    ),
    color=[
        domain_colors[
            domain
        ]
        for domain
        in panel_b_table[
            "domain"
        ]
    ],
    edgecolor="none",
    capsize=3,
    error_kw={
        "elinewidth": 1.0,
        "capthick": 1.0,
        "ecolor": "black",
    },
)

axis_b.axvline(
    0.0,
    color="black",
    linewidth=0.9,
)

axis_b.set_xlabel(
    "AUROC reduction after domain removal"
)

axis_b.set_title(
    "B. Domain ablation (ΔAUROC)",
    loc="left",
)

axis_b.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.22,
)

combined_limits = np.concatenate(
    [
        auroc_lower[
            np.isfinite(
                auroc_lower
            )
        ],
        auroc_upper[
            np.isfinite(
                auroc_upper
            )
        ],
        auroc_drop[
            np.isfinite(
                auroc_drop
            )
        ],
    ]
)

if len(
    combined_limits
) > 0:
    minimum_x = float(
        combined_limits.min()
    )

    maximum_x = float(
        combined_limits.max()
    )

else:
    minimum_x = -0.01
    maximum_x = 0.01

span_x = max(
    maximum_x
    - minimum_x,
    0.01,
)

axis_b.set_xlim(
    minimum_x
    - span_x
    * 0.12,
    maximum_x
    + span_x
    * 0.28,
)

for position, row in enumerate(
    panel_b_table.itertuples()
):
    value = float(
        row.AUROC_Drop
    )

    text_x = (
        float(
            row.AUROC_Drop_CI_Upper
        )
        + span_x
        * 0.025
        if value
        >= 0
        else float(
            row.AUROC_Drop_CI_Lower
        )
        - span_x
        * 0.025
    )

    axis_b.text(
        text_x,
        position,
        f"{value:.3f}",
        va="center",
        ha=(
            "left"
            if value
            >= 0
            else "right"
        ),
        fontsize=8.8,
    )

axis_b.text(
    0.99,
    0.02,
    "Error bars indicate paired bootstrap 95% CIs",
    transform=axis_b.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.5,
    color="dimgray",
)


# =============================================================================
# 13. PANEL C — RELATIVE DOMAIN COMPOSITION
# =============================================================================

composition = (
    domain_table[
        "relative_importance"
    ]
    .clip(
        lower=0.0
    )
)

composition_total = float(
    composition.sum()
)

if composition_total <= 0:
    raise ValueError(
        "Domain SHAP relative importance sums to zero."
    )

composition = (
    composition
    / composition_total
)

wedges, _ = axis_c.pie(
    composition.to_numpy(
        dtype=float
    ),
    colors=[
        domain_colors[
            domain
        ]
        for domain
        in domain_table[
            "domain"
        ]
    ],
    startangle=90,
    counterclock=False,
    wedgeprops={
        "width": 0.53,
        "edgecolor": "white",
        "linewidth": 1.0,
    },
)

axis_c.text(
    0.0,
    0.0,
    "Domain\nattribution",
    ha="center",
    va="center",
    fontsize=10,
)

axis_c.set_title(
    "C. Relative domain composition",
    loc="left",
)


# =============================================================================
# 14. LEGEND
# =============================================================================

axis_legend.axis(
    "off"
)

axis_legend.legend(
    wedges,
    [
        (
            f"{row.Domain_Label} "
            f"({100.0 * proportion:.1f}%)"
        )
        for row, proportion
        in zip(
            domain_table.itertuples(),
            composition,
        )
    ],
    loc="center",
    frameon=False,
    fontsize=9.5,
)


# =============================================================================
# 15. PANEL D — SHAP ATTRIBUTION VS ABLATION
# =============================================================================

x_values = domain_table[
    "relative_importance"
].to_numpy(
    dtype=float
)

y_values = domain_table[
    "AUROC_Drop"
].to_numpy(
    dtype=float
)

y_lower = domain_table[
    "AUROC_Drop_CI_Lower"
].to_numpy(
    dtype=float
)

y_upper = domain_table[
    "AUROC_Drop_CI_Upper"
].to_numpy(
    dtype=float
)

bubble_sizes = (
    220.0
    + 1500.0
    * (
        x_values
        / max(
            float(
                np.nanmax(
                    x_values
                )
            ),
            1e-12,
        )
    )
)

axis_d.axhline(
    0.0,
    color="dimgray",
    linestyle="--",
    linewidth=1.0,
)

for index, row in enumerate(
    domain_table.itertuples()
):
    axis_d.errorbar(
        x_values[
            index
        ],
        y_values[
            index
        ],
        yerr=np.asarray(
            [
                [
                    max(
                        y_values[
                            index
                        ]
                        - y_lower[
                            index
                        ],
                        0.0,
                    )
                ],
                [
                    max(
                        y_upper[
                            index
                        ]
                        - y_values[
                            index
                        ],
                        0.0,
                    )
                ],
            ]
        ),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=3,
        alpha=0.70,
        zorder=1,
    )

    axis_d.scatter(
        x_values[
            index
        ],
        y_values[
            index
        ],
        s=bubble_sizes[
            index
        ],
        color=domain_colors[
            row.domain
        ],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
        zorder=2,
    )

    # Domain-specific offsets prevent nearby labels from merging.
    label_offsets = {
        "Medical Conditions": (-12, 13),
        "Lifestyle": (12, -13),
        "Examination": (11, 11),
        "Laboratory": (-12, -13),
        "Mental Health": (10, 9),
        "Demographics": (10, 9),
        "Diet": (10, 9),
    }

    offset_x, offset_y = label_offsets.get(
        row.Domain_Label,
        (9, 7),
    )

    horizontal_alignment = (
        "right"
        if offset_x < 0
        else "left"
    )

    vertical_alignment = (
        "top"
        if offset_y < 0
        else "bottom"
    )

    axis_d.annotate(
        row.Domain_Label,
        xy=(
            x_values[
                index
            ],
            y_values[
                index
            ],
        ),
        xytext=(
            offset_x,
            offset_y,
        ),
        textcoords="offset points",
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=8.6,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "dimgray",
            "linewidth": 0.65,
            "shrinkA": 2,
            "shrinkB": 4,
        },
        zorder=3,
    )

axis_d.set_xlabel(
    "Relative domain SHAP attribution"
)

axis_d.set_ylabel(
    "AUROC reduction after removing domain"
)

axis_d.set_title(
    "D. Domain attribution versus ablation effect",
    loc="left",
)

axis_d.grid(
    linestyle="--",
    linewidth=0.7,
    alpha=0.25,
)

# Add margins so offset labels remain inside the axes.
axis_d.margins(
    x=0.07,
    y=0.16,
)

axis_d.text(
    0.99,
    0.02,
    (
        "SHAP quantifies attribution within the fitted model; "
        "ablation quantifies reliance on unique domain information"
    ),
    transform=axis_d.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.7,
    color="dimgray",
)


# =============================================================================
# 16. MAIN TITLE
# =============================================================================

figure.suptitle(
    (
        "Domain-Level Explainability of Adult Mobility-Disability Classification\n"
        f"NHANES 2017–2018 | XGBoost AUROC={xgboost_auroc:.3f}, "
        f"AUPRC={xgboost_auprc:.3f}"
    ),
    fontsize=15,
    fontweight="bold",
    y=0.985,
)

figure.tight_layout(
    rect=(
        0.02,
        0.02,
        0.99,
        0.955,
    )
)


# =============================================================================
# 17. SAVE AND DISPLAY
# =============================================================================

save_figure_all_formats(
    figure
)

plt.show()
plt.close(
    figure
)


# =============================================================================
# 18. SAVE SOURCE DATA
# =============================================================================

source_table = domain_table[
    [
        "domain",
        "Domain_Label",
        "mean_abs_domain_shap",
        "relative_importance",
        "relative_importance_pct",
        "AUROC",
        "AUPRC",
        "AUROC_Drop",
        "AUROC_Drop_CI_Lower",
        "AUROC_Drop_CI_Upper",
        "AUPRC_Drop",
        "AUPRC_Drop_CI_Lower",
        "AUPRC_Drop_CI_Upper",
    ]
].copy()

source_table.to_csv(
    SOURCE_TABLE_FILE,
    index=False,
)


# =============================================================================
# 19. SUMMARY JSON
# =============================================================================

summary = {
    "analysis": (
        "NHANES 2017-2018 adult mobility-disability classification "
        "domain-level explainability"
    ),
    "run_directory": str(
        RUN_DIR
    ),
    "domain_shap_file": str(
        DOMAIN_SHAP_FILE
    ),
    "domain_ablation_file": str(
        ABLATION_FILE
    ),
    "performance_file": str(
        PERFORMANCE_FILE
    ),
    "model": "XGBoost",
    "xgboost_auroc": (
        xgboost_auroc
    ),
    "xgboost_auprc": (
        xgboost_auprc
    ),
    "xgboost_brier": (
        xgboost_brier
    ),
    "domain_count": int(
        len(
            domain_table
        )
    ),
    "interpretation": (
        "Domain SHAP measures attribution within the fitted model, whereas "
        "domain ablation measures reliance on unique information. Differences "
        "between these measures can reflect redundancy across domains."
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
    "source_table": str(
        SOURCE_TABLE_FILE
    ),
    "domains": source_table.to_dict(
        orient="records"
    ),
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )


# =============================================================================
# 20. FINAL CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print(
    "NATIVE TREESHAP DOMAIN EXPLAINABILITY — FULL INTERNAL VALIDATION FIGURE COMPLETE"
)
print("=" * 100)

print("\nSaved figures:")
print(" -", OUTPUT_PNG)
print(" -", OUTPUT_PDF)
print(" -", OUTPUT_SVG)
print(" -", OUTPUT_TIFF)

print("\nSaved source table:")
print(" -", SOURCE_TABLE_FILE)

print("\nSaved summary:")
print(" -", SUMMARY_FILE)

print("\nTop domain SHAP attribution:")

print(
    domain_table[
        [
            "Domain_Label",
            "relative_importance_pct",
        ]
    ]
    .sort_values(
        "relative_importance_pct",
        ascending=False,
    )
    .round(
        3
    )
    .to_string(
        index=False
    )
)

print("\nLargest AUROC reductions:")

print(
    domain_table[
        [
            "Domain_Label",
            "AUROC_Drop",
            "AUROC_Drop_CI_Lower",
            "AUROC_Drop_CI_Upper",
        ]
    ]
    .sort_values(
        "AUROC_Drop",
        ascending=False,
    )
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print("\nLargest AUPRC reductions:")

print(
    domain_table[
        [
            "Domain_Label",
            "AUPRC_Drop",
            "AUPRC_Drop_CI_Lower",
            "AUPRC_Drop_CI_Upper",
        ]
    ]
    .sort_values(
        "AUPRC_Drop",
        ascending=False,
    )
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print("=" * 100)
