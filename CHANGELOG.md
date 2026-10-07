## v4.3 — Prediction Intelligence UI Refresh

- Redesigned prediction result as a visual risk-assessment experience.
- Added churn-probability ring, threshold marker and probability meter.
- Added visual feature-impact bars and clearer retention actions.
- Increased product-level typography and spacing while preserving the existing palette.
- No ML model, dataset, database schema or API behavior changes.

# Changelog

## v4.0 — Clean Analytics UI

- Rebuilt the Flask UI with a customer-risk intelligence visual system.
- Replaced external Chart.js dependency with lightweight in-app SVG charts.
- Added dataset-grounded churn, contract, tenure, charge and service analytics.
- Separated observed churn outcomes from persisted MySQL prediction risk.
- Added model CV, held-out test and XGBoost tuning visualizations.
- Improved Customer 360 financial, service and prediction-history views.
- Added responsive layouts and clearer visual hierarchy.
- Fixed pytest import resolution for the `app.py` / `app/` project layout.
- No changes to the trained XGBoost model or stored benchmark metrics.

# Changelog

## 3.0-XGBoost — 2026-10-05
- Replaced the previous Gradient Boosting production path with XGBoost.
- Added controlled XGBoost hyperparameter tuning using stratified 5-fold PR-AUC.
- Added validation-based threshold optimization.
- Added class-imbalance weighting with `scale_pos_weight`.
- Added service/tenure/charge feature engineering.
- Added batch CSV prediction and downloadable results.
- Kept final test data outside model and threshold selection.

# Changelog

## 3.1.0 — Flask + REST API

- Replaced the FastAPI/Uvicorn application layer with Flask.
- Added a single `python app.py` launcher.
- Added a REST API under `/api/*` for health, authentication, customers, predictions, model metrics, and retention actions.
- Removed API documentation from the business UI; REST endpoints remain available for programmatic integration.
- Preserved the existing scikit-learn prediction pipeline and MySQL persistence layer.
- Pinned scikit-learn to `1.8.0` to match the serialized production model and prevent the previous 1.8.0/1.9.1 warning.
- Added Flask session configuration and API authentication using the same local admin account.
- Updated Docker and Windows launch scripts.
- Updated tests for Flask routes and REST endpoints.

## 2.1.0

- Improved model evaluation and cross-validation.
- Added PR-AUC based model selection.
- Added model card and metrics artifacts.
- Added customer profile and prediction workflows.
