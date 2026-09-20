# Analysis Code

This directory contains the analysis code for:

**Interpretable Machine Learning for Classifying Concurrent Mobility Disability in Adults: A Cross-Sectional NHANES Study**

## Final analysis pipeline

The revised analysis uses corrected handling of valid zero values imported from SAS XPORT files. Known XPORT floating-point representations of zero were restored to `0` before subsequent data cleaning and feature derivation.

The final analytical cohort included **5,853 adults**, and outcome-independent feature-quality screening retained **331 predictors across seven domains**.

### Main scripts

- `NHANES_publication_revised_mobility_pipeline_fixed_v3_xportzero.py`  
  Main analysis pipeline: cohort construction, preprocessing, model development, calibration, threshold selection, and evaluation.

- `Build_XPORTZeroFixed_manuscript_tables.py`  
  Generates manuscript and supplementary tables.

- `Figure1_XPORTZeroFixed.py`  
  Generates the cohort-characteristics figure.

- `Figure3_XPORTZeroFixed.py`  
  Generates grouped SHAP-based feature attribution.

- `Figure4_XPORTZeroFixed.py`  
  Generates domain-level SHAP-based attribution and domain-ablation analyses.

Older analysis scripts are retained for provenance but are not required to reproduce the revised manuscript results.

## Data

NHANES 2017–2018 public-use data are available from the National Center for Health Statistics:

https://wwwn.cdc.gov/nchs/nhanes/

NHANES source data and participant-level analysis outputs are not redistributed in this repository.
