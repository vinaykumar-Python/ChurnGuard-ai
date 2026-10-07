# ChurnGuard AI — Clean Analytics Production Version

Customer Churn Prediction & Retention Intelligence Platform using Flask, REST API, XGBoost and MySQL.

## ML pipeline
- XGBoost is the sole production model.
- Stratified 5-fold cross-validation with PR-AUC as the primary metric.
- Six controlled XGBoost configurations are tuned on the training data.
- Validation-set threshold optimization; the final test set remains untouched for benchmarking.
- Class imbalance is handled with XGBoost `scale_pos_weight`.
- Production artifact is the complete preprocessing + XGBoost pipeline.

## Measured benchmark
- Dataset: 7,043 records
- Test set: 1,409 records
- PR-AUC: 0.6649
- ROC-AUC: 0.8474
- Precision: 0.5888
- Recall: 0.6738
- F1: 0.6284
- Accuracy: 0.7885
- Optimized classification threshold: 0.63

These metrics are specific to the supplied dataset and evaluation setup; they do not guarantee future customer outcomes.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure `.env` from `.env.example`, make sure local MySQL is running, then:

```bash
python train_model.py
python app.py
```

Open `http://localhost:8000`.

## Main features
- Single-customer churn prediction
- Batch CSV prediction and CSV export
- MySQL customer and prediction history
- Customer 360 profile
- Risk distribution dashboard
- Prediction history
- Explainability view
- Retention recommendations and action tracking
- Model Lab with XGBoost evaluation
- REST API under `/api/*`

## Analytics & Visual Intelligence — v4 UI

The dashboard now includes data-driven visualizations built from the supplied customer dataset and saved MySQL prediction history:

- Churned vs retained customer distribution
- Saved prediction risk distribution
- Contract type vs churn outcome
- Tenure band vs observed churn rate
- Monthly-charge groups vs observed churn rate
- Active service count vs observed churn rate
- Model metric visualization (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
- Visual confusion matrix using the stored test-set counts
- Customer 360 financial and service charts
- Customer prediction-probability history from MySQL

Charts are rendered with lightweight in-app SVG JavaScript, so the analytics UI does not depend on an external chart CDN. The ML pipeline, model artifact, thresholds, and stored evaluation metrics are unchanged.
