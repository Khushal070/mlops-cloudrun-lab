"""Cloud Run prediction service.

- Loads a versioned model from GCS at startup (MODEL_PATH env var), or from
  LOCAL_MODEL_DIR for local testing.
- POST /predict returns predictions and logs one row per request to BigQuery.
- GET /stats summarizes the last 24h of predictions from BigQuery.
- Logs are single-line JSON, which Cloud Logging parses into structured fields.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, jsonify, request

BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_PATH = os.environ.get("MODEL_PATH")            # e.g. models/20261015-120000
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR")  # e.g. model (local testing only)
BQ_TABLE = os.environ.get("BQ_TABLE")                # e.g. my-project.mlops_lab.predictions
MAX_INSTANCES = 100

app = Flask(__name__)


def log(severity, message, **fields):
    print(json.dumps({"severity": severity, "message": message, **fields}), flush=True)


def load_model():
    if LOCAL_MODEL_DIR:
        model_dir = LOCAL_MODEL_DIR
    else:
        if not (BUCKET_NAME and MODEL_PATH):
            raise RuntimeError("Set BUCKET_NAME and MODEL_PATH (or LOCAL_MODEL_DIR for local runs)")
        from google.cloud import storage

        model_dir = "/tmp/model"
        os.makedirs(model_dir, exist_ok=True)
        bucket = storage.Client().bucket(BUCKET_NAME)
        for name in ("model.joblib", "metadata.json"):
            bucket.blob(f"{MODEL_PATH}/{name}").download_to_filename(os.path.join(model_dir, name))

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    return model, meta


t0 = time.perf_counter()
MODEL, META = load_model()
log("INFO", "model loaded", model_version=META["version"],
    load_ms=round((time.perf_counter() - t0) * 1000, 1))

_bq_client = None


def bq():
    global _bq_client
    if _bq_client is None:
        from google.cloud import bigquery

        _bq_client = bigquery.Client()
    return _bq_client


@app.get("/")
def info():
    return jsonify(service="breast-cancer-classifier", model_version=META["version"],
                   metrics=META["metrics"], endpoints=["/health", "/predict", "/stats"])


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=META["version"])


@app.post("/predict")
def predict():
    start = time.perf_counter()
    body = request.get_json(silent=True)
    if not body or "instances" not in body:
        return jsonify(error='Body must be JSON: {"instances": [[30 floats], ...]}'), 400

    try:
        X = np.asarray(body["instances"], dtype=float)
    except (TypeError, ValueError):
        return jsonify(error="instances must contain only numbers"), 400
    if X.ndim != 2 or X.shape[1] != META["n_features"]:
        return jsonify(error=f"each instance needs exactly {META['n_features']} features"), 400
    if not 1 <= X.shape[0] <= MAX_INSTANCES:
        return jsonify(error=f"send between 1 and {MAX_INSTANCES} instances"), 400

    preds = MODEL.predict(X)
    proba_malignant = MODEL.predict_proba(X)[:, 0]  # class 0 = malignant
    results = [
        {"label": META["class_names"][int(p)], "prob_malignant": round(float(pm), 4)}
        for p, pm in zip(preds, proba_malignant)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    request_id = str(uuid.uuid4())
    row = {
        "request_id": request_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_version": META["version"],
        "n_instances": int(X.shape[0]),
        "n_malignant": int((preds == 0).sum()),
        "mean_prob_malignant": round(float(proba_malignant.mean()), 4),
        "latency_ms": latency_ms,
    }
    log("INFO", "prediction", **row)

    # Prediction logging must never break serving: a BigQuery failure only logs a warning.
    if BQ_TABLE:
        try:
            errors = bq().insert_rows_json(BQ_TABLE, [row])
            if errors:
                log("WARNING", "bigquery insert errors", errors=str(errors))
        except Exception as exc:  # noqa: BLE001
            log("WARNING", "bigquery insert failed", error=str(exc))

    return jsonify(request_id=request_id, model_version=META["version"],
                   predictions=results, latency_ms=latency_ms)


@app.get("/stats")
def stats():
    if not BQ_TABLE:
        return jsonify(error="BQ_TABLE is not configured"), 503
    query = f"""
        SELECT
          model_version,
          COUNT(*) AS requests,
          SUM(n_instances) AS instances,
          ROUND(SAFE_DIVIDE(SUM(n_malignant), SUM(n_instances)), 4) AS malignant_rate,
          ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
          ROUND(APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)], 2) AS p95_latency_ms
        FROM `{BQ_TABLE}`
        WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        GROUP BY model_version
        ORDER BY requests DESC
    """
    rows = [dict(r) for r in bq().query(query).result()]
    return jsonify(window="last 24h", by_model_version=rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))