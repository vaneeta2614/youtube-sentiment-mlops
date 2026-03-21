"""
promote_model.py
----------------
Transitions the latest Staging model version to Production in
the MLflow Model Registry, archiving the previous Production version.

Run standalone:  python scripts/promote_model.py
Also called by the CI/CD pipeline after all tests pass.
"""
import json
import logging
import sys
import yaml
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def promote() -> None:
    params     = load_params()
    mlf_cfg    = params["mlflow"]
    model_name = mlf_cfg["registered_model_name"]

    mlflow.set_tracking_uri(mlf_cfg["tracking_uri"])
    client = MlflowClient()

    # ── Find latest Staging version ──────────────────────────────
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        logger.error("No model version found in Staging for '%s'. Aborting.", model_name)
        sys.exit(1)

    staging_version = staging_versions[0]
    new_version     = staging_version.version
    run_id          = staging_version.run_id

    logger.info("Found Staging version: %s  (run_id: %s)", new_version, run_id)

    # ── Fetch eval metrics from the run ──────────────────────────
    run_data = client.get_run(run_id).data
    acc      = run_data.metrics.get("eval_accuracy",    run_data.metrics.get("accuracy",    None))
    f1       = run_data.metrics.get("eval_f1_weighted", run_data.metrics.get("f1_weighted", None))
    rec_neg  = run_data.metrics.get("eval_recall_neg",  run_data.metrics.get("recall_neg",  None))

    logger.info("Metrics — accuracy: %s  |  f1: %s  |  recall(-1): %s", acc, f1, rec_neg)

    # ── Compare against existing Production (optional guard) ─────
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        prev_version = prod_versions[0]
        prev_run     = client.get_run(prev_version.run_id).data
        prev_acc     = prev_run.metrics.get("eval_accuracy", 0.0)
        logger.info(
            "Current Production version %s has accuracy %.4f. "
            "New version %s has accuracy %s.",
            prev_version.version, prev_acc, new_version, acc
        )

    # ── Promote ───────────────────────────────────────────────────
    client.transition_model_version_stage(
        name     = model_name,
        version  = new_version,
        stage    = "Production",
        archive_existing_versions = True,   # auto-archive old Production
    )
    logger.info(
        "✓  Model '%s' version %s promoted to Production.",
        model_name, new_version
    )

    # ── Write promotion record ────────────────────────────────────
    record = {
        "model_name"     : model_name,
        "promoted_version": new_version,
        "run_id"         : run_id,
        "metrics"        : {"accuracy": acc, "f1_weighted": f1, "recall_neg": rec_neg},
    }
    with open("reports/promotion_record.json", "w") as f:
        json.dump(record, f, indent=2)
    logger.info("Promotion record saved to reports/promotion_record.json")


if __name__ == "__main__":
    promote()
