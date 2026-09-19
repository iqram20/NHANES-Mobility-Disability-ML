# Analysis scripts

This directory is intended to contain the final scripts used for the manuscript analyses.

Recommended files:

- `NHANES_publication_revised_mobility_pipeline_fixed_v2.py` — cohort construction, preprocessing, model development, calibration, threshold selection, and primary evaluation
- `NHANES_postrun_performance_metrics.py` — post-run performance summaries and sensitivity analyses
- `NHANES_postrun_native_TreeSHAP_FULL1171.py` — native XGBoost TreeSHAP analysis in the complete internal-validation cohort
- `Build_current_run_manuscript_tables.py` — manuscript and supplementary table generation
- `Figure1_current_run_adult_cohort_characteristics.py` — cohort-characteristics figure
- `Figure2_current_NHANES_demographics_updated.py` — model-performance figure
- `Figure3_native_TreeSHAP_FULL1171.py` — grouped native TreeSHAP figure
- `Figure4_native_TreeSHAP_FULL1171_domain_explainability.py` — domain explainability and ablation figure

Do not commit NHANES source data, participant-level outputs, or large serialized model files to this repository.
