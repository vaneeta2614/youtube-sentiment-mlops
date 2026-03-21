"""
register_model.py
-----------------
Pushes trained model + vectorizer to MLflow Model Registry (Staging).
"""
import json
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def register(params: dict) -> None:
    mlf_cfg    = params["mlflow"]
    model_name = mlf_cfg["registered_model_name"]

    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    mlflow.set_experiment(mlf_cfg["experiment_name"])

    with open("models/model.pkl",      "rb") as f: model      = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    with open("reports/eval_metrics.json")    as f: metrics    = json.load(f)

    logger.info("Registering model '%s' to MLflow…", model_name)

    with mlflow.start_run(run_name="register_model") as run:
        mlflow.sklearn.log_model(
            sk_model       = model,
            artifact_path  = "model",
            registered_model_name = model_name,
        )
        mlflow.log_artifact("models/vectorizer.pkl", artifact_path="vectorizer")
        mlflow.log_metrics({
            "accuracy"   : metrics.get("accuracy",     0),
            "f1_weighted": metrics.get("f1_weighted",  0),
            "recall_neg" : metrics.get("recall_per_class", {}).get("-1", 0),
        })
        run_id = run.info.run_id

    # ── Move new version to Staging ───────────────────────────────
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(int(v.version) for v in versions)

    client.transition_model_version_stage(
        name    = model_name,
        version = str(latest),
        stage   = "Staging",
        archive_existing_versions = False,
    )
    logger.info("Model version %d set to Staging ✓", latest)


if __name__ == "__main__":
    params = load_params()
    register(params)
