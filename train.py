"""Train the breast-cancer GradientBoosting model, enforce the F1 quality gate,
and optionally upload the versioned model to GCS.

Usage:
  python train.py                 # train + gate, save to ./model/
  python train.py --upload        # also upload to gs://$BUCKET_NAME/models/<version>/
Exits with code 1 if F1 is below F1_THRESHOLD, so CI stops before deploying.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

F1_THRESHOLD = float(os.environ.get("F1_THRESHOLD", "0.90"))
OUT_DIR = "model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="upload model + metrics to GCS")
    args = parser.parse_args()

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    version = os.environ.get("MODEL_VERSION") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    meta = {
        "version": version,
        "model": "GradientBoostingClassifier",
        "dataset": "sklearn breast_cancer",
        "n_features": int(X_train.shape[1]),
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),  # index 0 = malignant, 1 = benign
        "f1_threshold": F1_THRESHOLD,
        "metrics": metrics,
    }
    print(json.dumps(meta["metrics"], indent=2))

    # Quality gate: a model below threshold never gets saved, uploaded or deployed.
    if metrics["f1"] < F1_THRESHOLD:
        print(f"QUALITY GATE FAILED: F1 {metrics['f1']} < {F1_THRESHOLD}", file=sys.stderr)
        sys.exit(1)
    print(f"Quality gate passed: F1 {metrics['f1']} >= {F1_THRESHOLD}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUT_DIR, "model.joblib"))
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Two real test rows, so the deployed API can be smoke-tested with curl.
    with open("sample_request.json", "w") as f:
        json.dump({"instances": X_test[:2].round(5).tolist()}, f)

    if args.upload:
        from google.cloud import storage  # imported lazily so local runs need no GCP libs

        bucket_name = os.environ["BUCKET_NAME"]
        bucket = storage.Client().bucket(bucket_name)
        for name in ("model.joblib", "metadata.json"):
            blob = bucket.blob(f"models/{version}/{name}")
            blob.upload_from_filename(os.path.join(OUT_DIR, name))
        print(f"Uploaded to gs://{bucket_name}/models/{version}/")

    # Lets GitHub Actions read the version in later steps.
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"version={version}\n")


if __name__ == "__main__":
    main()