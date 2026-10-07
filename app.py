from __future__ import annotations

from pathlib import Path
import math
import os
import io
from functools import wraps

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for, abort, send_file

from app.ml import load_assets, predict_customer
from app import db
from app.security import verify_password

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

DATA_PATH = BASE / "data" / "customer_churn.csv"
HIGH_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))
MEDIUM_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.45"))

try:
    DF, MODEL, METRICS = load_assets()
    MODEL_ERROR = None
except Exception as exc:
    DF, MODEL, METRICS = pd.DataFrame(), None, {}
    MODEL_ERROR = str(exc)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SECRET_KEY=os.getenv("APP_SECRET_KEY", "change-this-secret-in-production"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=False,
        JSON_SORT_KEYS=False,
    )

    # Initialize the local MySQL schema if MySQL is available. A database failure
    # must not prevent the UI from opening; the status banner explains the issue.
    try:
        db.ensure_database()
        app.config["DB_STARTUP_ERROR"] = None
    except Exception as exc:
        app.config["DB_STARTUP_ERROR"] = str(exc)

    register_routes(app)
    return app


def risk_level(probability: float) -> str:
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def recommendation(row: dict, probability: float) -> list[str]:
    actions: list[str] = []
    if probability >= HIGH_THRESHOLD:
        actions.append("Prioritize a retention review and contact the customer.")
    elif probability >= MEDIUM_THRESHOLD:
        actions.append("Monitor the customer and review possible service or plan concerns.")
    else:
        actions.append("Continue normal engagement and monitor future risk changes.")

    if str(row.get("ContractType", "")) == "Month-to-month":
        actions.append("Review whether a suitable longer-term plan or loyalty benefit is appropriate.")
    if float(row.get("Tenure", 0) or 0) <= 6:
        actions.append("Consider an onboarding check-in for early customer friction.")
    if "MonthlyCharges" in DF and float(row.get("MonthlyCharges", 0) or 0) >= float(DF["MonthlyCharges"].median()):
        actions.append("Review price/value fit and available service options.")
    if str(row.get("TechSupport", "")) == "No":
        actions.append("Check whether technical support could address the customer's needs.")
    return actions


FORM_NUMERIC_DEFAULTS = {
    "SeniorCitizen": 0,
    "Tenure": 0,
    "MonthlyCharges": 70.0,
    "TotalCharges": 0.0,
}

def fields_for_form(values=None, new_customer=False):
    values = values or {}
    fields = []
    if DF.empty:
        return fields
    for col in DF.columns:
        if col in ("CustomerID", "Churn"):
            continue
        if pd.api.types.is_numeric_dtype(DF[col]):
            median = DF[col].median()
            default = FORM_NUMERIC_DEFAULTS.get(col, 0 if pd.isna(median) else float(median)) if new_customer else values.get(col, 0 if pd.isna(median) else float(median))
            fields.append({"name": col, "type": "number", "value": default})
        else:
            options = sorted(DF[col].dropna().astype(str).unique().tolist())
            default = values.get(col, options[0] if options else "")
            fields.append({"name": col, "type": "select", "options": options, "value": str(default)})
    return fields


def parse_customer(source):
    customer = {}
    errors = []
    for field in fields_for_form():
        raw = source.get(field["name"], "")
        if field["type"] == "number":
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
                if field["name"] in ("Tenure", "SeniorCitizen"):
                    value = int(value)
                customer[field["name"]] = value
            except Exception:
                errors.append(f"{field['name']} must be a valid number.")
        else:
            value = str(raw).strip()
            if value not in field["options"]:
                errors.append(f"{field['name']} has an invalid value.")
            customer[field["name"]] = value
    # Domain validation prevents impossible values from reaching the model.
    if "SeniorCitizen" in customer and customer["SeniorCitizen"] not in (0, 1):
        errors.append("SeniorCitizen must be 0 or 1.")
    if "Tenure" in customer and not 0 <= customer["Tenure"] <= 100:
        errors.append("Tenure must be between 0 and 100 months.")
    if "MonthlyCharges" in customer and customer["MonthlyCharges"] < 0:
        errors.append("MonthlyCharges cannot be negative.")
    if "TotalCharges" in customer and customer["TotalCharges"] < 0:
        errors.append("TotalCharges cannot be negative.")
    return customer, errors


def logged_in() -> bool:
    return bool(session.get("user"))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not logged_in():
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    return wrapped


def api_login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not logged_in():
            return jsonify({"error": "Authentication required."}), 401
        return view(*args, **kwargs)
    return wrapped


def base_context(**kwargs):
    status = db.db_status()
    return {
        "user": session.get("user"),
        "db_status": status,
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "model_version": METRICS.get("model_version", "3.0-XGBoost"),
        "metrics_prediction_threshold": METRICS.get("prediction_threshold", 0.50),
        **kwargs,
    }


def explanation(customer: dict, base_probability: float):
    if MODEL is None or DF.empty:
        return []
    raw_cols = [c for c in DF.columns if c not in ("CustomerID", "Churn")]
    base = pd.DataFrame([customer])
    results = []
    for col in raw_cols:
        altered = base.copy()
        if pd.api.types.is_numeric_dtype(DF[col]):
            altered[col] = DF[col].median()
        else:
            mode = DF[col].dropna().mode()
            if not mode.empty:
                altered[col] = mode.iloc[0]
        try:
            from app.ml import prepare_features
            probability = float(MODEL.predict_proba(prepare_features(altered))[:, 1][0])
            results.append({"feature": col, "impact": round(float(base_probability - probability), 4)})
        except Exception:
            continue
    results.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return results[:6]


def customer_db_payload(customer_id: str, values: dict) -> dict:
    mapping = {
        "Gender": "gender", "SeniorCitizen": "senior_citizen", "Partner": "partner",
        "Dependents": "dependents", "Tenure": "tenure", "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines", "ServiceType": "service_type",
        "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection", "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
        "ContractType": "contract_type", "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method", "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
    }
    return {"customer_id": customer_id, **{db_col: values.get(csv_col) for csv_col, db_col in mapping.items()}}


def predict_and_store(customer_id: str, values: dict) -> dict:
    if MODEL is None:
        raise RuntimeError(f"Model unavailable: {MODEL_ERROR}")
    probability = predict_customer(MODEL, values)
    risk = risk_level(probability)
    actions = recommendation(values, probability)
    rec = " ".join(actions)
    db.save_customer(customer_db_payload(customer_id, values))
    prediction_id = db.save_prediction(customer_id, probability, risk, rec)
    return {
        "prediction_id": prediction_id,
        "customer_id": customer_id,
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(probability * 100, 2),
        "risk_level": risk,
        "recommendations": actions,
        "explanation": explanation(values, probability),
    }


def analytics_payload():
    """Build analytics from real dataset outcomes plus persisted prediction counts."""
    empty = {
        "dataset_total": 0, "churned": 0, "retained": 0, "churned_percent": 0,
        "churn": [], "prediction_risk": [], "contract": [], "tenure": [],
        "charges": [], "services": []
    }
    if DF.empty:
        return empty

    frame = DF.copy()
    outcome = frame["Churn"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    ) if "Churn" in frame else pd.Series(dtype="float64")

    churned = int((outcome == 1).sum())
    retained = int((outcome == 0).sum())
    total = int(len(frame))
    churn = [
        {"label": "Churned", "value": churned},
        {"label": "Retained", "value": retained},
    ]

    prediction_risk = []
    try:
        prediction_stats = db.prediction_stats() or {}
        prediction_risk = [
            {"label": "High", "value": int(prediction_stats.get("high_risk") or 0)},
            {"label": "Medium", "value": int(prediction_stats.get("medium_risk") or 0)},
            {"label": "Low", "value": int(prediction_stats.get("low_risk") or 0)},
        ]
    except Exception:
        # Dashboard remains useful when MySQL is unavailable; no fake prediction data is shown.
        prediction_risk = []

    contract = []
    if "ContractType" in frame and "Churn" in frame:
        work = frame.assign(_y=outcome)
        for name, group in work.groupby("ContractType", dropna=False):
            contract.append({
                "label": str(name),
                "churned": int((group["_y"] == 1).sum()),
                "retained": int((group["_y"] == 0).sum()),
            })

    tenure = []
    if "Tenure" in frame and "Churn" in frame:
        bins = [-1, 6, 12, 24, 36, 48, 60, 120]
        labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61+"]
        work = frame.assign(
            _y=outcome,
            _tenure_bucket=pd.cut(pd.to_numeric(frame["Tenure"], errors="coerce"), bins=bins, labels=labels),
        )
        for label, group in work.groupby("_tenure_bucket", observed=False):
            if len(group):
                tenure.append({
                    "label": str(label),
                    "churn_rate": round(float(group["_y"].mean() * 100), 2),
                    "customers": int(len(group)),
                })

    charges = []
    if "MonthlyCharges" in frame and "Churn" in frame:
        numeric = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
        work = frame.assign(_y=outcome, _charge_bucket=pd.qcut(numeric, q=5, duplicates="drop"))
        for index, (label, group) in enumerate(work.groupby("_charge_bucket", observed=False), start=1):
            if len(group):
                charges.append({
                    "label": f"Group {index}",
                    "churn_rate": round(float(group["_y"].mean() * 100), 2),
                    "customers": int(len(group)),
                    "range": str(label),
                })

    services = []
    service_cols = [c for c in [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ] if c in frame]
    if service_cols and "Churn" in frame:
        active = frame[service_cols].astype(str).apply(lambda col: col.str.strip().str.lower().eq("yes")).sum(axis=1)
        work = frame.assign(_y=outcome, _active=active)
        for count, group in work.groupby("_active"):
            if len(group):
                services.append({
                    "label": str(int(count)),
                    "churn_rate": round(float(group["_y"].mean() * 100), 2),
                    "customers": int(len(group)),
                })

    return {
        "dataset_total": total,
        "churned": churned,
        "retained": retained,
        "churned_percent": round((churned / total) * 100, 2) if total else 0,
        "churn": churn,
        "prediction_risk": prediction_risk,
        "contract": contract,
        "tenure": tenure,
        "charges": charges,
        "services": services,
    }

def register_routes(app: Flask):
    # ------------------------- UI -------------------------
    @app.get("/login")
    def login_page():
        if logged_in():
            return redirect(url_for("dashboard"))
        return render_template("login.html", **base_context(error=None))

    @app.post("/login")
    def login():
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        try:
            status = db.db_status()
            if not status.get("ok"):
                db.ensure_database()
            user = db.authenticate(username, password, verify_password)
        except Exception as exc:
            return render_template("login.html", **base_context(error=f"Database connection failed: {exc}")), 500
        if not user:
            return render_template("login.html", **base_context(error="Invalid username or password.")), 401
        session.clear()
        session["user"] = {"id": user["id"], "username": user["username"], "role": user["role"]}
        return redirect(url_for("dashboard"))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login_page"))

    @app.get("/")
    @login_required
    def dashboard():
        stats = {"total": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0, "avg_probability": 0}
        rows, error = [], None
        try:
            stats = db.prediction_stats() or stats
            rows = db.recent_predictions(8)
        except Exception as exc:
            error = str(exc)
        return render_template(
            "dashboard.html",
            **base_context(
                heading="Dashboard",
                stats={
                    "total": int(stats.get("total") or 0),
                    "high": int(stats.get("high_risk") or 0),
                    "medium": int(stats.get("medium_risk") or 0),
                    "low": int(stats.get("low_risk") or 0),
                    "avg": round(float(stats.get("avg_probability") or 0) * 100, 1),
                },
                rows=rows,
                analytics=analytics_payload(),
                error=error,
                model=METRICS.get("best_model", "Unknown"),
                metrics_selection=METRICS.get("selection_metric", "Cross-validation"),
            ),
        )

    @app.get("/predict")
    @login_required
    def predict_page():
        selected_id = request.args.get("customer_id", "").strip()
        mode = request.args.get("mode", "new").strip().lower()
        if mode not in {"new", "existing"}:
            mode = "new"
        values = {}
        if mode == "existing" and selected_id and "CustomerID" in DF:
            match = DF[DF["CustomerID"].astype(str).eq(selected_id)]
            if not match.empty:
                values = match.iloc[0].drop(labels=["CustomerID", "Churn"], errors="ignore").to_dict()
        return render_template("predict.html", **base_context(
            heading="Predict Customer",
            fields=fields_for_form(values, new_customer=(mode == "new")),
            error=None, selected_customer_id=selected_id, mode=mode,
        ))

    @app.post("/predict")
    @login_required
    def predict_form():
        customer_id = request.form.get("customer_id", "").strip()
        mode = request.form.get("mode", "new").strip().lower()
        if mode not in {"new", "existing"}:
            mode = "new"
        if not customer_id:
            return render_template("predict.html", **base_context(
                heading="Predict Customer", fields=fields_for_form(new_customer=(mode == "new")),
                error="Customer ID is required.", selected_customer_id="", mode=mode
            )), 400
        customer, errors = parse_customer(request.form)
        if errors:
            return render_template("predict.html", **base_context(
                heading="Predict Customer", fields=fields_for_form(customer, new_customer=(mode == "new")),
                error=" ".join(errors), selected_customer_id=customer_id, mode=mode
            )), 422
        try:
            result = predict_and_store(customer_id, customer)
            result["customer_type"] = "New customer" if mode == "new" else "Existing customer"
            return render_template("predict_result.html", **base_context(heading="Prediction Result", result=result))
        except Exception as exc:
            return render_template("predict.html", **base_context(
                heading="Predict Customer", fields=fields_for_form(customer, new_customer=(mode == "new")),
                error=f"Prediction failed: {exc}", selected_customer_id=customer_id, mode=mode
            )), 500

    @app.get("/predict-result")
    @login_required
    def predict_result_page():
        return render_template("predict_result.html", **base_context(heading="Prediction Result", result=None))

    @app.get("/batch-predict")
    @login_required
    def batch_predict_page():
        return render_template("batch_predict.html", **base_context(heading="Batch Prediction", error=None, result=None))

    @app.post("/batch-predict")
    @login_required
    def batch_predict():
        upload = request.files.get("file")
        if not upload or not upload.filename.lower().endswith(".csv"):
            return render_template("batch_predict.html", **base_context(heading="Batch Prediction", error="Please upload a CSV file.", result=None)), 400
        try:
            raw = upload.read()
            if len(raw) > 5 * 1024 * 1024:
                raise ValueError("CSV file must be 5 MB or smaller.")
            batch = pd.read_csv(io.BytesIO(raw))
            if "CustomerID" not in batch.columns:
                raise ValueError("CSV must contain a CustomerID column.")
            output = []
            errors = []
            for idx, row in batch.iterrows():
                customer_id = str(row.get("CustomerID", "")).strip()
                values = {k: row[k] for k in batch.columns if k not in {"CustomerID", "Churn"}}
                valid, validation_errors = parse_customer(values)
                if not customer_id or validation_errors:
                    errors.append({"row": int(idx) + 2, "customer_id": customer_id or "(missing)", "error": "; ".join(validation_errors) or "CustomerID is required."})
                    continue
                result = predict_and_store(customer_id, valid)
                output.append({
                    "CustomerID": customer_id,
                    "ChurnProbability": result["churn_probability_percent"],
                    "RiskLevel": result["risk_level"],
                    "Recommendation": " ".join(result["recommendations"]),
                })
            result = {"processed": len(output), "errors": errors, "rows": output}
            return render_template("batch_predict.html", **base_context(heading="Batch Prediction", error=None, result=result))
        except Exception as exc:
            return render_template("batch_predict.html", **base_context(heading="Batch Prediction", error=f"Batch prediction failed: {exc}", result=None)), 500

    @app.post("/batch-predict/download")
    @login_required
    def batch_predict_download():
        payload = request.form.get("rows", "")
        if not payload:
            return redirect(url_for("batch_predict_page"))
        import json as _json
        rows = _json.loads(payload)
        frame = pd.DataFrame(rows)
        stream = io.BytesIO()
        frame.to_csv(stream, index=False)
        stream.seek(0)
        return send_file(stream, mimetype="text/csv", as_attachment=True, download_name="churnguard_batch_predictions.csv")

    @app.get("/history")
    @login_required
    def history():
        q = request.args.get("q", "")
        risk = request.args.get("risk", "")
        try:
            rows = db.search_predictions(q.strip(), risk.strip())
            error = None
        except Exception as exc:
            rows, error = [], str(exc)
        history_chart = [{"label": str(i+1), "risk": round(float(r.get('churn_probability') or 0) * 100, 2)} for i, r in enumerate(reversed(rows[:20]))]
        return render_template("history.html", **base_context(heading="Prediction History", rows=rows, q=q, risk=risk, error=error, history_chart=history_chart))

    @app.get("/retention")
    @login_required
    def retention():
        try:
            actions, error = db.recent_actions(), None
        except Exception as exc:
            actions, error = [], str(exc)
        status_counts = {}
        for action in actions:
            key = str(action.get('status') or 'Planned')
            status_counts[key] = status_counts.get(key, 0) + 1
        retention_chart = [{"label": k, "value": v} for k, v in sorted(status_counts.items())]
        return render_template("retention.html", **base_context(heading="Retention Actions", actions=actions, error=error, retention_chart=retention_chart))

    @app.post("/retention")
    @login_required
    def retention_action():
        try:
            db.save_retention_action(
                request.form.get("customer_id", "").strip(),
                request.form.get("action", "").strip(),
                request.form.get("notes", "").strip(),
            )
        except Exception:
            pass
        return redirect(url_for("retention"))

    @app.get("/model")
    @login_required
    def model_page():
        # Model Lab reuses the same saved-prediction statistics shown on the dashboard.
        # Keep the template context explicit so a database issue cannot leave Jinja
        # with an undefined `stats` object.
        stats = {"total": 0, "high": 0, "medium": 0, "low": 0, "avg": 0}
        error = None
        try:
            raw = db.prediction_stats() or {}
            stats = {
                "total": int(raw.get("total") or 0),
                "high": int(raw.get("high_risk") or 0),
                "medium": int(raw.get("medium_risk") or 0),
                "low": int(raw.get("low_risk") or 0),
                "avg": round(float(raw.get("avg_probability") or 0) * 100, 1),
            }
        except Exception as exc:
            error = str(exc)
        return render_template(
            "model.html",
            **base_context(
                heading="Model Lab",
                metrics=METRICS,
                stats=stats,
                model=METRICS.get("best_model", "Unknown"),
                metrics_selection=METRICS.get("selection_metric", "Cross-validation"),
                error=error,
            ),
        )

    @app.get("/explain")
    @login_required
    def explain_page():
        return render_template("explain.html", **base_context(heading="Explainability", error=None))

    @app.get("/customers")
    @login_required
    def customers():
        q = request.args.get("q", "").strip()
        error = None
        try:
            saved = db.list_customers(q, 100)
        except Exception as exc:
            saved, error = [], str(exc)
        return render_template("customers.html", **base_context(
            heading="Customers", customers=saved, q=q, total=len(saved), error=error
        ))

    @app.get("/customers/<customer_id>")
    @login_required
    def customer_profile(customer_id: str):
        customer = None
        source = "Application customer"
        try:
            customer = db.get_customer(customer_id)
        except Exception:
            customer = None
        if customer is None and not DF.empty:
            match = DF[DF["CustomerID"].astype(str).eq(customer_id)]
            if not match.empty:
                customer = match.iloc[0].to_dict()
                source = "Training dataset customer"
        if customer is None:
            abort(404, description="Customer not found.")
        try:
            latest = db.latest_prediction(customer_id)
            history = db.prediction_history(customer_id, 30)
        except Exception:
            latest, history = None, []
        customer_values = {}
        for key, value in customer.items():
            customer_values[str(key)] = value
        return render_template("customer_profile.html", **base_context(
            heading="Customer Profile", customer=customer, customer_id=customer_id, source=source, latest=latest,
            history=history, customer_values=customer_values
        ))

    # ------------------------- REST API -------------------------
    @app.get("/api/health")
    def api_health():
        status = db.db_status()
        model_ok = MODEL is not None
        db_ok = bool(status.get("ok"))
        overall = "ok" if model_ok and db_ok else "degraded" if model_ok else "error"
        return jsonify({
            "status": overall,
            "service": "ChurnGuard AI REST API",
            "model_loaded": model_ok,
            "model_version": METRICS.get("model_version"),
            "best_model": METRICS.get("best_model"),
            "database": status,
        })

    @app.post("/api/login")
    def api_login():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        try:
            if not db.db_status().get("ok"):
                db.ensure_database()
            user = db.authenticate(username, password, verify_password)
        except Exception as exc:
            return jsonify({"error": f"Database connection failed: {exc}"}), 503
        if not user:
            return jsonify({"error": "Invalid username or password"}), 401
        session.clear()
        session["user"] = {"id": user["id"], "username": user["username"], "role": user["role"]}
        return jsonify({"message": "Authenticated", "user": session["user"]})

    @app.post("/api/logout")
    @api_login_required
    def api_logout():
        session.clear()
        return jsonify({"message": "Logged out"})

    @app.get("/api/stats")
    @api_login_required
    def api_stats():
        return jsonify(db.prediction_stats() or {})

    @app.get("/api/customers")
    @api_login_required
    def api_customers():
        q = request.args.get("q", "").strip()
        limit = min(max(int(request.args.get("limit", 50)), 1), 100)
        rows = db.list_customers(q, limit)
        return jsonify({"count": len(rows), "customers": rows})

    @app.get("/api/customers/<customer_id>")
    @api_login_required
    def api_customer(customer_id: str):
        row = db.get_customer(customer_id)
        if row is None:
            return jsonify({"error": "Customer not found"}), 404
        return jsonify(row)

    @app.post("/api/predict")
    @api_login_required
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        customer_id = str(payload.get("customer_id", "")).strip()
        values = payload.get("values")
        if not customer_id or not isinstance(values, dict):
            return jsonify({"error": "customer_id and values are required."}), 400
        customer = dict(values)
        customer.pop("CustomerID", None)
        customer.pop("Churn", None)
        valid, errors = parse_customer(customer)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 422
        try:
            return jsonify(predict_and_store(customer_id, valid))
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.get("/api/predictions")
    @api_login_required
    def api_predictions():
        q = request.args.get("q", "")
        risk = request.args.get("risk", "")
        limit = min(max(int(request.args.get("limit", 50)), 1), 100)
        return jsonify({"predictions": db.search_predictions(q.strip(), risk.strip(), limit)})

    @app.get("/api/model")
    @api_login_required
    def api_model():
        return jsonify(METRICS)

    @app.get("/api/retention")
    @api_login_required
    def api_retention():
        limit = min(max(int(request.args.get("limit", 50)), 1), 100)
        return jsonify({"actions": db.recent_actions(limit)})

    @app.post("/api/retention")
    @api_login_required
    def api_retention_create():
        payload = request.get_json(silent=True) or {}
        customer_id = str(payload.get("customer_id", "")).strip()
        action = str(payload.get("action", "")).strip()
        notes = str(payload.get("notes", "")).strip()
        status = str(payload.get("status", "Planned")).strip() or "Planned"
        if not customer_id or not action:
            return jsonify({"error": "customer_id and action are required."}), 400
        db.save_retention_action(customer_id, action, notes, status)
        return jsonify({"message": "Retention action saved"}), 201

    @app.get("/health")
    def health_alias():
        return api_health()

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": getattr(error, "description", "Not found")}), 404
        return "<h1>404</h1><p>Page not found.</p><p><a href='/'>Return to dashboard</a></p>", 404


app = create_app()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", "8000")), debug=True)
