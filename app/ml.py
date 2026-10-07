from pathlib import Path
import json
import math
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, make_scorer,
)

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / 'data' / 'customer_churn.csv'
MODEL_DIR = BASE / 'models'
MODEL_PATH = MODEL_DIR / 'churn_pipeline.pkl'
METRICS_PATH = MODEL_DIR / 'metrics.json'
FEATURES_PATH = MODEL_DIR / 'feature_columns.json'
MODEL_CARD_PATH = MODEL_DIR / 'model_card.json'

ID_COL = 'CustomerID'
TARGET = 'Churn'
RANDOM_STATE = 42


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df:
        raise ValueError('Dataset must contain a Churn column.')
    target = df[TARGET].astype(str).str.strip().map({
        'Yes': 1, 'No': 0, '1': 1, '0': 0, 'True': 1, 'False': 0,
    })
    df[TARGET] = pd.to_numeric(target, errors='coerce')
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    if ID_COL in df:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()
    for col in ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'TotalCharges' in df and 'Tenure' in df:
        safe_tenure = df['Tenure'].replace(0, np.nan)
        df['AvgMonthlySpend'] = (df['TotalCharges'] / safe_tenure).fillna(df.get('MonthlyCharges'))
        df['ChargesPerTenureMonth'] = (df['TotalCharges'] / safe_tenure).replace([np.inf, -np.inf], np.nan)
    if 'Tenure' in df:
        df['IsNewCustomer'] = (df['Tenure'] <= 6).astype(int)
        df['IsLongTenure'] = (df['Tenure'] >= 48).astype(int)
    if 'MonthlyCharges' in df and 'Tenure' in df:
        df['EstimatedCustomerValue'] = df['MonthlyCharges'] * (df['Tenure'] + 1)
    service_cols = [c for c in ['PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in df.columns]
    if service_cols:
        yes_no = df[service_cols].astype(str).apply(lambda col: col.str.lower().eq('yes').astype(int))
        df['ActiveServiceCount'] = yes_no.sum(axis=1)
    if 'ContractType' in df:
        df['IsMonthToMonth'] = df['ContractType'].astype(str).str.lower().eq('month-to-month').astype(int)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(df.drop(columns=[TARGET, ID_COL], errors='ignore'))


def make_preprocessor(X: pd.DataFrame):
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('num', numeric_pipe, numerical),
        ('cat', categorical_pipe, categorical),
    ])


def model_zoo(X_train: pd.DataFrame, y_train: pd.Series):
    positive = max(int(y_train.sum()), 1)
    negative = max(int(len(y_train) - y_train.sum()), 1)
    scale_pos_weight = negative / positive
    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def xgb_search_space():
    return [
        {'n_estimators': 220, 'max_depth': 3, 'learning_rate': 0.04, 'min_child_weight': 2, 'subsample': 0.85, 'colsample_bytree': 0.85},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_weight': 2, 'subsample': 0.90, 'colsample_bytree': 0.85},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'min_child_weight': 2, 'subsample': 0.85, 'colsample_bytree': 0.85},
        {'n_estimators': 350, 'max_depth': 4, 'learning_rate': 0.04, 'min_child_weight': 3, 'subsample': 0.90, 'colsample_bytree': 0.90},
        {'n_estimators': 260, 'max_depth': 5, 'learning_rate': 0.04, 'min_child_weight': 3, 'subsample': 0.85, 'colsample_bytree': 0.80},
        {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.035, 'min_child_weight': 3, 'subsample': 0.90, 'colsample_bytree': 0.90},
    ]

def make_pipeline(X: pd.DataFrame, estimator):
    return Pipeline([
        ('preprocessor', make_preprocessor(X)),
        ('model', estimator),
    ])


def choose_threshold(y_true, probabilities):
    """Choose a probability threshold using only the training validation split."""
    thresholds = np.arange(0.30, 0.71, 0.01)
    best = {'threshold': 0.50, 'F1': -1.0, 'Recall': 0.0, 'Precision': 0.0}
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        # Prefer F1, with recall as a deterministic tie-breaker.
        candidate = (f1, recall, precision)
        current = (best['F1'], best['Recall'], best['Precision'])
        if candidate > current:
            best = {
                'threshold': round(float(threshold), 2),
                'F1': round(float(f1), 4),
                'Recall': round(float(recall), 4),
                'Precision': round(float(precision), 4),
            }
    return best


def evaluate_predictions(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': round(float(accuracy_score(y_true, pred)), 4),
        'Precision': round(float(precision_score(y_true, pred, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_true, pred, zero_division=0)), 4),
        'F1': round(float(f1_score(y_true, pred, zero_division=0)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_true, probabilities)), 4),
        'PR-AUC': round(float(average_precision_score(y_true, probabilities)), 4),
        'Brier': round(float(brier_score_loss(y_true, probabilities)), 4),
        'TrueNegative': int(tn), 'FalsePositive': int(fp),
        'FalseNegative': int(fn), 'TruePositive': int(tp),
    }


def train_models():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = prepare_features(df)
    y = df[TARGET]

    # Final test set remains untouched until every training decision is complete.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuning_results = []
    base_model = model_zoo(X_tune, y_tune)

    for params in xgb_search_space():
        estimator = XGBClassifier(**{**base_model.get_params(), **params})
        pipe = make_pipeline(X_tune, estimator)
        scores = cross_validate(
            pipe, X_tune, y_tune, cv=cv, scoring={'pr_auc':'average_precision','roc_auc':'roc_auc','f1':'f1','recall':'recall','precision':'precision'},
            n_jobs=-1, return_train_score=False
        )
        tuning_results.append({
            'params': params,
            'PR-AUC_mean': round(float(np.mean(scores['test_pr_auc'])), 4),
            'PR-AUC_std': round(float(np.std(scores['test_pr_auc'])), 4),
            'ROC-AUC_mean': round(float(np.mean(scores['test_roc_auc'])), 4),
            'F1_mean': round(float(np.mean(scores['test_f1'])), 4),
            'Recall_mean': round(float(np.mean(scores['test_recall'])), 4),
            'Precision_mean': round(float(np.mean(scores['test_precision'])), 4),
        })

    best_tuning = max(tuning_results, key=lambda item: item['PR-AUC_mean'])
    selected_params = best_tuning['params']
    tuned_model = XGBClassifier(**{**base_model.get_params(), **selected_params})
    tuned_pipe = make_pipeline(X_tune, tuned_model)
    tuned_pipe.fit(X_tune, y_tune)

    # Optimize the business classification threshold using validation data only.
    val_prob = tuned_pipe.predict_proba(X_val)[:, 1]
    threshold_info = choose_threshold(y_val, val_prob)

    # Final benchmark: test data has never been used for model/threshold selection.
    benchmark_pipe = make_pipeline(X_train, XGBClassifier(**{**model_zoo(X_train, y_train).get_params(), **selected_params}))
    benchmark_pipe.fit(X_train, y_train)
    test_prob = benchmark_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_prob, threshold_info['threshold'])
    test_metrics['Threshold'] = threshold_info['threshold']

    # Produce a clean 5-fold CV summary on the complete training partition using the selected parameters.
    final_cv_pipe = make_pipeline(X_train, XGBClassifier(**{**model_zoo(X_train, y_train).get_params(), **selected_params}))
    final_scores = cross_validate(
        final_cv_pipe, X_train, y_train, cv=cv, scoring={'pr_auc':'average_precision','roc_auc':'roc_auc','f1':'f1','recall':'recall','precision':'precision'},
        n_jobs=-1, return_train_score=False
    )
    cv_summary = {
        'PR-AUC_mean': round(float(np.mean(final_scores['test_pr_auc'])), 4),
        'PR-AUC_std': round(float(np.std(final_scores['test_pr_auc'])), 4),
        'ROC-AUC_mean': round(float(np.mean(final_scores['test_roc_auc'])), 4),
        'F1_mean': round(float(np.mean(final_scores['test_f1'])), 4),
        'Recall_mean': round(float(np.mean(final_scores['test_recall'])), 4),
        'Precision_mean': round(float(np.mean(final_scores['test_precision'])), 4),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    production_model = make_pipeline(X, XGBClassifier(**{**model_zoo(X, y).get_params(), **selected_params}))
    production_model.fit(X, y)
    joblib.dump(production_model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(X.columns.tolist(), indent=2), encoding='utf-8')

    generated = datetime.now(timezone.utc).isoformat()
    metrics = {
        'best_model': 'XGBoost',
        'selection_metric': '5-fold stratified PR-AUC mean',
        'prediction_threshold': threshold_info['threshold'],
        'threshold_selection': 'Validation-set F1 optimization with recall/precision tie-breakers',
        'cv_folds': 5,
        'cv_results': {'XGBoost': cv_summary},
        'tuning_results': tuning_results,
        'selected_params': selected_params,
        'results': {'XGBoost': test_metrics},
        'training_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'dataset_rows': int(len(df)),
        'positive_rate': round(float(y.mean()), 4),
        'generated_at_utc': generated,
        'feature_engineering': ['AvgMonthlySpend','ChargesPerTenureMonth','IsNewCustomer','IsLongTenure','EstimatedCustomerValue','ActiveServiceCount','IsMonthToMonth'],
        'model_version': '3.0-XGBoost',
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    model_card = {
        'project': 'ChurnGuard AI', 'dataset': 'data/customer_churn.csv', 'target': TARGET,
        'selected_model': 'XGBoost', 'selection_metric': metrics['selection_metric'],
        'prediction_threshold': threshold_info['threshold'], 'threshold_selection': metrics['threshold_selection'],
        'training_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
        'feature_engineering': metrics['feature_engineering'], 'model_version': metrics['model_version'],
        'generated_at_utc': generated,
        'notes': [
            'XGBoost is the sole production model.',
            'Six controlled XGBoost configurations are evaluated with stratified 5-fold cross-validation using PR-AUC.',
            'The final test partition is kept completely outside model and threshold selection.',
            'The prediction threshold is tuned on a validation split for F1, with recall and precision tie-breakers.',
            'The final production pipeline is refit on all labeled data after the benchmark is recorded.',
            'XGBoost uses scale_pos_weight derived from the training class distribution.',
            'Metrics describe the supplied evaluation data and do not guarantee future customer outcomes.',
        ],
    }
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2), encoding='utf-8')
    return metrics

def load_assets():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not MODEL_CARD_PATH.exists():
        train_models()
    df = clean_data(pd.read_csv(DATA_PATH))
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    return df, model, metrics


def predict_customer(model, customer: dict):
    frame = pd.DataFrame([customer])
    x = prepare_features(frame)
    probability = float(model.predict_proba(x)[:, 1][0])
    if not math.isfinite(probability):
        raise ValueError('Model returned an invalid probability.')
    return max(0.0, min(1.0, probability))
