"""
data_preprocessing.py
---------------------
Cleans raw comments, splits into train/test, saves to data/processed/.
"""
import os
import re
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_special_chars(text: str) -> str:
    """Keep alphanumeric, spaces, basic punctuation."""
    return re.sub(r"[^a-zA-Z0-9\s!?.,']", " ", text)


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_comment(text: str, cfg: dict) -> str:
    text = str(text).lower()
    if cfg.get("remove_urls", True):
        text = remove_urls(text)
    text = remove_html(text)
    if not cfg.get("remove_emojis", False):
        pass   # keep emojis intact
    text = remove_special_chars(text)
    text = normalise_whitespace(text)
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def preprocess(params: dict) -> None:
    raw_path       = params["data"]["raw_path"]
    processed_path = params["data"]["processed_path"]
    test_size      = params["data"]["test_size"]
    random_state   = params["data"]["random_state"]
    min_length     = params["preprocessing"]["min_comment_length"]
    prep_cfg       = params["preprocessing"]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)

    # Rename columns to standard names if needed
    col_map = {}
    for col in df.columns:
        if "comment" in col.lower() or "text" in col.lower():
            col_map[col] = "comment"
        elif "category" in col.lower() or "label" in col.lower() or "sentiment" in col.lower():
            col_map[col] = "label"
    df.rename(columns=col_map, inplace=True)

    if "comment" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected 'comment' and 'label' columns. Got: {df.columns.tolist()}")

    logger.info("Raw shape: %s", df.shape)
    logger.info("Label distribution:\n%s", df["label"].value_counts())

    # Drop nulls
    df.dropna(subset=["comment", "label"], inplace=True)

    # Clean text
    logger.info("Cleaning comments…")
    df["comment"] = df["comment"].apply(lambda t: clean_comment(t, prep_cfg))

    # Drop too-short comments
    df = df[df["comment"].str.split().str.len() >= min_length]

    # Ensure labels are int in {-1, 0, 1}
    df["label"] = df["label"].astype(int)
    valid_labels = {-1, 0, 1}
    df = df[df["label"].isin(valid_labels)]

    logger.info("After cleaning shape: %s", df.shape)
    df.to_csv(processed_path, index=False)

    # Train / test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv",   index=False)

    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))


if __name__ == "__main__":
    params = load_params()
    preprocess(params)
    logger.info("Preprocessing complete.")
