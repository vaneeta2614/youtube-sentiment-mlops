"""
feature_engineering.py
-----------------------
Vectorises text (TF-IDF / BoW / Word2Vec) and handles class imbalance.
Saves feature arrays and vectorizer to disk.
"""
import os
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_vectorizer(cfg: dict):
    method      = cfg.get("method", "tfidf")
    max_feat    = cfg.get("max_features", 50000)
    ngram_range = tuple(cfg.get("ngram_range", [1, 2]))
    sublinear   = cfg.get("sublinear_tf", True)

    if method == "tfidf":
        logger.info("Using TF-IDF vectorizer (max_features=%d, ngram=%s)", max_feat, ngram_range)
        return TfidfVectorizer(
            max_features=max_feat,
            ngram_range=ngram_range,
            sublinear_tf=sublinear,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
            min_df=2,
        )
    elif method == "bow":
        logger.info("Using Bag-of-Words vectorizer (max_features=%d, ngram=%s)", max_feat, ngram_range)
        return CountVectorizer(
            max_features=max_feat,
            ngram_range=ngram_range,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
            min_df=2,
        )
    else:
        raise ValueError(f"Unsupported feature method: {method}. Choose tfidf or bow.")


def engineer(params: dict) -> None:
    fe_cfg       = params["feature_engineering"]
    imb_cfg      = params["imbalance"]
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("models",        exist_ok=True)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    X_train_raw = train_df["comment"].fillna("").astype(str)
    y_train     = train_df["label"].values.astype(int)
    X_test_raw  = test_df["comment"].fillna("").astype(str)
    y_test      = test_df["label"].values.astype(int)

    # ── Vectorise ──────────────────────────────────────────────────
    vectorizer = build_vectorizer(fe_cfg)
    X_train    = vectorizer.fit_transform(X_train_raw)
    X_test     = vectorizer.transform(X_test_raw)

    logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)

    # ── Imbalance handling (SMOTE / ADASYN only if using dense arrays) ─
    strategy = imb_cfg.get("strategy", "class_weight")
    if strategy in ("smote", "adasyn"):
        logger.info("Applying %s oversampling…", strategy.upper())
        try:
            if strategy == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=42)
            else:
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            logger.info("After resampling — X_train shape: %s", X_train.shape)
        except ImportError:
            logger.warning("imbalanced-learn not installed. Falling back to class_weight.")
    elif strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    # ── Save artefacts ─────────────────────────────────────────────
    with open("data/features/X_train.pkl", "wb") as f: pickle.dump(X_train, f)
    with open("data/features/X_test.pkl",  "wb") as f: pickle.dump(X_test,  f)
    with open("data/features/y_train.pkl", "wb") as f: pickle.dump(y_train, f)
    with open("data/features/y_test.pkl",  "wb") as f: pickle.dump(y_test,  f)
    with open("models/vectorizer.pkl",     "wb") as f: pickle.dump(vectorizer, f)

    logger.info("Features and vectorizer saved.")


if __name__ == "__main__":
    params = load_params()
    engineer(params)
    logger.info("Feature engineering complete.")
