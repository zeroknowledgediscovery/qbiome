# Clinical SHAP

Note: clinical data is currently not shared publicly (may be shared later)

- `shap-clinical-factors.ipynb`: To assess impact of clinical factors on risk, we train a random forest regressor to predict risk scores for subjects from the available clinical factors, then perform a SHAP analysis on the trained regressor.

- `median_risk.csv`: Risk scores for each subject in the UChicago cohort (median risk over 20 runs computed at each time point from 25-36 weeks PMA)