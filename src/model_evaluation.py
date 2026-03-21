"""
model_evaluation.py
-------------------
Evaluates trained model on test set, logs metrics to MLflow,
writes reports/eval_metrics.json and confusion matrix CSV.
"""
import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(params: dict) -> None:
    mlf_cfg  = params["mlflow"]
    eval_cfg = params["evaluation"]

    os.makedirs("reports", exist_ok=True)

    # ── Load artefacts ────────────────────────────────────────────
    with open("models/model.pkl",      "rb") as f: model      = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    with open("data/features/X_test.pkl", "rb") as f: X_test = pickle.load(f)
    with open("data/features/y_test.pkl", "rb") as f: y_test = pickle.load(f)

    # ── Predict ───────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Core metrics ──────────────────────────────────────────────
    acc        = accuracy_score(y_test, y_pred)
    f1_w       = f1_score(y_test, y_pred, average="weighted")
    prec_w     = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w      = recall_score(y_test, y_pred, average="weighted",  zero_division=0)

    # Per-class recall
    classes    = sorted(np.unique(y_test))
    rec_per    = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    rec_map    = dict(zip([str(c) for c in classes], rec_per.tolist()))

    logger.info("Accuracy : %.4f", acc)
    logger.info("F1 (w)   : %.4f", f1_w)
    logger.info("Recall/-1: %.4f", rec_map.get("-1", 0.0))
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

    # ── Threshold checks ──────────────────────────────────────────
    passed = True
    if acc < eval_cfg["accuracy_threshold"]:
        logger.warning("FAIL: accuracy %.4f < threshold %.4f", acc, eval_cfg["accuracy_threshold"])
        passed = False
    if f1_w < eval_cfg["f1_threshold"]:
        logger.warning("FAIL: F1 %.4f < threshold %.4f", f1_w, eval_cfg["f1_threshold"])
        passed = False
    if float(rec_map.get("-1", 0.0)) < eval_cfg["recall_neg_threshold"]:
        logger.warning("FAIL: recall(-1) %.4f < threshold %.4f",
                       float(rec_map.get("-1", 0.0)), eval_cfg["recall_neg_threshold"])
        passed = False

    # ── Confusion matrix CSV ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                              columns=[f"pred_{c}" for c in classes])
    cm_df.to_csv("reports/confusion_matrix.csv")

    # ── Save eval metrics ─────────────────────────────────────────
    eval_metrics = {
        "accuracy"        : round(acc,   4),
        "f1_weighted"     : round(f1_w,  4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted" : round(rec_w,  4),
        "recall_per_class": {k: round(v, 4) for k, v in rec_map.items()},
        "passed_thresholds": passed,
    }
    with open("reports/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # ── Log to MLflow ─────────────────────────────────────────────
    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "eval_accuracy"    : acc,
            "eval_f1_weighted" : f1_w,
            "eval_recall_neg"  : float(rec_map.get("-1", 0.0)),
        })
        mlflow.log_artifact("reports/eval_metrics.json")
        mlflow.log_artifact("reports/confusion_matrix.csv")

    if not passed:
        logger.error("Model did NOT pass evaluation thresholds. Check reports/eval_metrics.json")
        raise SystemExit(1)

    logger.info("Model PASSED all evaluation thresholds ✓")


if __name__ == "__main__":
    params = load_params()
    evaluate(params)
