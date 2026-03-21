"""
data_ingestion.py
-----------------
Downloads / copies raw dataset into data/raw/.
Kaggle Reddit comments dataset (multi-class sentiment).
"""
import os
import sys
import logging
import yaml
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ingest(params: dict) -> None:
    raw_path: str = params["data"]["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # ── Option A: Kaggle API ──────────────────────────────────────
    # Uncomment and set KAGGLE_USERNAME / KAGGLE_KEY env vars.
    #
    # import kaggle
    # kaggle.api.authenticate()
    # kaggle.api.dataset_download_files(
    #     "cosmos98/twitter-and-reddit-sentimental-analysis-dataset",
    #     path="data/raw/",
    #     unzip=True,
    # )
    # logger.info("Dataset downloaded from Kaggle.")

    # ── Option B: Local CSV already present ──────────────────────
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        logger.info("Raw dataset already present at %s — rows: %d", raw_path, len(df))
        return

    # ── Option C: Minimal synthetic fallback (dev/testing only) ──
    logger.warning(
        "Raw dataset not found and Kaggle download disabled. "
        "Creating tiny synthetic dataset for testing."
    )
    sample = pd.DataFrame({
        "clean_comment": [
            "This video is absolutely amazing and helpful",
            "Terrible content, wasted my time completely",
            "Ok video, nothing special about it",
            "Great explanation, very clear and concise",
            "Horrible quality, stop making videos please",
            "Decent tutorial for beginners I suppose",
        ],
        "category": [1, -1, 0, 1, -1, 0],
    })
    sample.to_csv(raw_path, index=False)
    logger.info("Synthetic dataset saved to %s", raw_path)


if __name__ == "__main__":
    params = load_params()
    ingest(params)
    logger.info("Data ingestion complete.")
