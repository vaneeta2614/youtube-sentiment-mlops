"""
model_training.py
-----------------
Trains the classifier and logs everything to MLflow.
Supports: LightGBM, XGBoost, Random Forest, Logistic Regression.
"""
import os
import json
import pickle
import logging
import yaml
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_score, recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, class_weights_dict: dict | None = None):
    algo = cfg.get("algorithm", "lightgbm")

    if algo == "lightgbm":
        from lightgbm import LGBMClassifier
        logger.info("Building LightGBM model…")
        return LGBMClassifier(
            n_estimators      = cfg["n_estimators"],
            learning_rate     = cfg["learning_rate"],
            max_depth         = cfg["max_depth"],
            num_leaves        = cfg["num_leaves"],
            min_child_samples = cfg["min_child_samples"],
            subsample         = cfg["subsample"],
            colsample_bytree  = cfg["colsample_bytree"],
            reg_alpha         = cfg["reg_alpha"],
            reg_lambda        = cfg["reg_lambda"],
            class_weight      = class_weights_dict,
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )

    elif algo == "xgboost":
        from xgboost import XGBClassifier
        logger.info("Building XGBoost model…")
        # XGBoost expects 0-indexed classes → map {-1,0,1} → {0,1,2}
        return XGBClassifier(
            n_estimators      = cfg["n_estimators"],
            learning_rate     = cfg["learning_rate"],
            max_depth         = cfg["max_depth"],
            subsample         = cfg["subsample"],
            colsample_bytree  = cfg["colsample_bytree"],
            reg_alpha         = cfg["reg_alpha"],
            reg_lambda        = cfg["reg_lambda"],
            use_label_encoder = False,
            eval_metric       = "mlogloss",
            random_state      = 42,
            n_jobs            = -1,
        )

    elif algo == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        logger.info("Building Random Forest model…")
        return RandomForestClassifier(
            n_estimators = cfg["n_estimators"],
            max_depth    = cfg.get("max_depth", None),
            class_weight = class_weights_dict,
            random_state = 42,
            n_jobs       = -1,
        )

    elif algo == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        logger.info("Building Logistic Regression model…")
        return LogisticRegression(
            max_iter     = 1000,
            class_weight = class_weights_dict,
            C            = 1.0,
            random_state = 42,
            n_jobs       = -1,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def train(params: dict) -> None:
    model_cfg = params["model"]
    mlf_cfg   = params["mlflow"]
    imb_cfg   = params["imbalance"]

    os.makedirs("models",  exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ── Load features ─────────────────────────────────────────────
    with open("data/features/X_train.pkl", "rb") as f: X_train = pickle.load(f)
    with open("data/features/y_train.pkl", "rb") as f: y_train = pickle.load(f)

    # ── Class weights ─────────────────────────────────────────────
    class_weights_dict = None
    if imb_cfg.get("strategy") == "class_weight":
        logger.info("Computing balanced class weights…")
        classes        = np.unique(y_train)
        weights        = sklearn_class_weight("balanced", classes=classes, y=y_train)
        class_weights_dict = dict(zip(classes.tolist(), weights.tolist()))
        logger.info("Class weights: %s", class_weights_dict)

    # ── MLflow ────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    mlflow.set_experiment(mlf_cfg["experiment_name"])

    with mlflow.start_run(run_name=f"train_{model_cfg['algorithm']}") as run:
        mlflow.log_params({
            "algorithm":    model_cfg["algorithm"],
            "n_estimators": model_cfg["n_estimators"],
            "learning_rate":model_cfg["learning_rate"],
            "max_depth":    model_cfg["max_depth"],
            "imbalance":    imb_cfg["strategy"],
            "fe_method":    params["feature_engineering"]["method"],
        })

        # ── Build & fit ───────────────────────────────────────────
        model = build_model(model_cfg, class_weights_dict)
        model.fit(X_train, y_train)
        logger.info("Model training complete.")

        # ── Train metrics ─────────────────────────────────────────
        y_pred = model.predict(X_train)
        acc    = accuracy_score(y_train, y_pred)
        f1     = f1_score(y_train, y_pred, average="weighted")
        mlflow.log_metric("train_accuracy", acc)
        mlflow.log_metric("train_f1",       f1)
        logger.info("Train accuracy: %.4f | Train F1: %.4f", acc, f1)

        # ── Save model ────────────────────────────────────────────
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ── Write train metrics json ──────────────────────────────
        train_metrics = {"train_accuracy": round(acc, 4), "train_f1": round(f1, 4)}
        with open("reports/train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)

        logger.info("Run ID: %s", run.info.run_id)


# stdlib shim for class_weight
from sklearn.utils.class_weight import compute_class_weight as sklearn_class_weight


if __name__ == "__main__":
    params = load_params()
    train(params)
    logger.info("Model training pipeline complete.")
