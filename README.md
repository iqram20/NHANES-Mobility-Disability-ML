# NHANES Mobility Disability Machine Learning

Code repository for the manuscript:

**Interpretable Machine Learning for Classifying Concurrent Mobility Disability in Adults: A Cross-Sectional NHANES Study**

**Published in:** *Journal of Disability Research*  
**Article DOI:** https://doi.org/10.57197/JDR-2026-0984  
**Supplementary materials:** [Supplementary_Informations_v1.pdf](Supplementary_Informations_v1.pdf)

## Overview

This study develops and internally evaluates interpretable machine-learning models for concurrent classification of self-reported mobility disability in adults using the 2017–2018 National Health and Nutrition Examination Survey (NHANES).

The analytical cohort included 5,853 adults, including 966 participants reporting serious difficulty walking or climbing stairs. After outcome-independent feature-quality screening, 331 predictors were retained across seven domains:

- Demographics
- Lifestyle
- Mental health
- Medical conditions
- Examination
- Laboratory
- Diet

The evaluated models were XGBoost, logistic regression, and random forest. Model interpretation included grouped SHAP-based attribution, patient-level domain attribution, domain-size-normalized attribution, and leave-one-domain-out retraining.

## Data

This repository does **not** redistribute NHANES data.

NHANES 2017–2018 public-use data and documentation are available from the National Center for Health Statistics:

https://wwwn.cdc.gov/nchs/nhanes/

The mobility-disability outcome was based on NHANES variable **DLQ050**: serious difficulty walking or climbing stairs.

## Analysis design

- Final analytical cohort: 5,853 adults
- Development cohort: 4,682 participants
- Held-out internal-validation cohort: 1,171 participants
- Mobility-disability cases in internal validation: 193
- Random seed: 42
- Five-fold stratified cross-validation within the development cohort
- Platt calibration based on out-of-fold development predictions
- Operating thresholds selected from development data
- SHAP-based feature attribution evaluated in all 1,171 internal-validation participants using the native XGBoost contribution algorithm

## Software

The primary analysis used:

- Python 3.10.6
- NumPy 1.26.4
- pandas 2.2.3
- scikit-learn 1.1.3
- XGBoost 1.7.6
- SHAP 0.49.1
- SciPy 1.15.2
- matplotlib 3.10.9
- joblib 1.5.3

## Repository structure

The final reproducibility release will contain the analysis scripts used for:

1. Cohort construction and feature-quality screening
2. Model development, calibration, and evaluation
3. Survey-weight sensitivity analysis
4. Native XGBoost TreeSHAP analysis
5. Domain-level attribution and ablation
6. Subgroup analyses
7. Manuscript figures and tables

## Reproducibility

NHANES data must be downloaded separately from the official NHANES website. Local data and result directories are intentionally excluded from version control.

## Citation

If you use this code or supporting materials, please cite the published article:

> *Interpretable Machine Learning for Classifying Concurrent Mobility Disability in Adults: A Cross-Sectional NHANES Study.* **Journal of Disability Research**. https://doi.org/10.57197/JDR-2026-0984

[View published article](https://doi.org/10.57197/JDR-2026-0984) · [View supplementary materials](Supplementary_Informations_v1.pdf)

The article DOI is provided as the permanent publication identifier. Refer to the journal article for its full author and bibliographic citation.

## Contact

Iqram Hussain, PhD  
Department of Anesthesiology, Weill Cornell Medicine  
New York, NY, USA


## SAS XPORT zero handling

Known SAS XPORT floating-point representations of valid zero introduced during file import were restored to 0 before subsequent data cleaning and feature derivation. The corrected analysis retained 331 original predictors across seven domains.
