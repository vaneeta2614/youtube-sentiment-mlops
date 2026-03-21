"""
app.py — SentimentScope Flask API
----------------------------------
Endpoints:
  POST /analyze          → fetch YT comments + run sentiment inference
  GET  /health           → liveness probe
  GET  /model/info       → current production model version
"""

import os
import re
import pickle
import logging
import hashlib
from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Any

import yaml
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from googleapiclient.discovery import build

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow Chrome extension origin

# ── Config ─────────────────────────────────────────────────────────
def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

PARAMS        = load_params()
MLF_CFG       = PARAMS["mlflow"]
YT_API_KEY    = os.environ.get("YOUTUBE_API_KEY", "")
MAX_COMMENTS  = int(os.environ.get("MAX_COMMENTS", "200"))
SENTIMENT_MAP = {1: "positive", -1: "negative", 0: "neutral"}

# ── Model loading (lazy, cached) ───────────────────────────────────
_model      = None
_vectorizer = None


def load_production_model():
    global _model, _vectorizer
    model_name = MLF_CFG["registered_model_name"]
    tracking_uri = MLF_CFG["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    client   = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if not versions:
        raise RuntimeError(f"No Production model found for '{model_name}'")

    version   = versions[0].version
    run_id    = versions[0].run_id
    model_uri = f"models:/{model_name}/{version}"
    logger.info("Loading model %s version %s…", model_name, version)

    _model = mlflow.sklearn.load_model(model_uri)

    local_path  = client.download_artifacts(run_id, "vectorizer/vectorizer.pkl")
    with open(local_path, "rb") as f:
        _vectorizer = pickle.load(f)

    logger.info("Model + vectorizer loaded ✓  (version %s)", version)
    return version


def get_model_and_vectorizer():
    global _model, _vectorizer
    if _model is None:
        load_production_model()
    return _model, _vectorizer


# ── YouTube comment fetching ───────────────────────────────────────

def fetch_comments(video_id: str) -> list[dict]:
    """
    Fetch up to MAX_COMMENTS comments from the YouTube Data API v3.
    Returns list of dicts: {text, author, published_at}
    """
    if not YT_API_KEY:
        raise RuntimeError(
            "YOUTUBE_API_KEY env var is not set. "
            "Set it to a valid YouTube Data API v3 key."
        )

    yt = build("youtube", "v3", developerKey=YT_API_KEY)
    comments = []
    next_page = None

    while len(comments) < MAX_COMMENTS:
        resp = yt.commentThreads().list(
            part            = "snippet",
            videoId         = video_id,
            maxResults      = min(100, MAX_COMMENTS - len(comments)),
            pageToken       = next_page,
            textFormat      = "plainText",
            order           = "relevance",
        ).execute()

        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text"        : snippet.get("textDisplay", ""),
                "author"      : snippet.get("authorDisplayName", ""),
                "published_at": snippet.get("publishedAt", ""),
                "like_count"  : snippet.get("likeCount", 0),
            })

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    logger.info("Fetched %d comments for video %s", len(comments), video_id)
    return comments


# ── Sentiment inference ───────────────────────────────────────────

def predict_sentiment(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model, vectorizer = get_model_and_vectorizer()
    X     = vectorizer.transform(texts)
    preds = model.predict(X)

    # Probability scores → scale to [0, 10]
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba(X)
        # Use max class probability as confidence
        max_proba = proba.max(axis=1)
        # Map: positive → high score, negative → low score
        class_idx = {c: i for i, c in enumerate(model.classes_)}
        pos_idx   = class_idx.get(1, 0)
        neg_idx   = class_idx.get(-1, 1)
        scores = (proba[:, pos_idx] - proba[:, neg_idx] + 1) / 2 * 10
    else:
        scores = np.where(preds == 1, 8.0, np.where(preds == -1, 2.0, 5.0))

    return preds, scores


# ── Analysis helpers ──────────────────────────────────────────────

def sentiment_counts(preds: np.ndarray) -> dict:
    c = Counter(preds.tolist())
    return {
        "positive": c.get(1,  0),
        "negative": c.get(-1, 0),
        "neutral" : c.get(0,  0),
    }


def build_trend(comments: list[dict], preds: np.ndarray) -> list[dict]:
    """Group sentiments by year-month."""
    from collections import defaultdict
    monthly: dict[str, dict] = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})

    for comment, pred in zip(comments, preds.tolist()):
        try:
            dt    = datetime.fromisoformat(comment["published_at"].replace("Z", "+00:00"))
            key   = dt.strftime("%Y-%m")
            label = SENTIMENT_MAP.get(pred, "neutral")
            monthly[key][label] += 1
        except Exception:
            pass

    return [{"month": k, **v} for k, v in sorted(monthly.items())]


def word_frequency(texts: list[str], top_n: int = 80) -> dict[str, int]:
    STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "and", "or",
        "but", "i", "me", "my", "you", "your", "we", "he", "she", "they",
        "this", "that", "for", "of", "with", "be", "was", "are", "were",
        "have", "has", "had", "not", "so", "do", "did", "if", "from",
        "by", "as", "about", "just", "very", "like", "can", "will",
        "no", "yes", "up", "out", "what", "how", "also",
    }
    counter: Counter = Counter()
    for text in texts:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        for w in words:
            if w not in STOPWORDS:
                counter[w] += 1
    return dict(counter.most_common(top_n))


def compute_stats(comments: list[dict], preds: np.ndarray, scores: np.ndarray) -> dict:
    texts   = [c["text"] for c in comments]
    authors = set(c["author"] for c in comments)
    avg_len = round(sum(len(t.split()) for t in texts) / max(len(texts), 1), 1)
    avg_score = round(float(scores.mean()), 2)

    return {
        "total"              : len(comments),
        "unique"             : len(authors),
        "avg_length"         : avg_len,
        "avg_sentiment_score": avg_score,
    }


# ── Routes ────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "SentimentScope API"}), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    try:
        model_name = MLF_CFG["registered_model_name"]
        mlflow.set_tracking_uri(MLF_CFG["tracking_uri"])
        client   = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return jsonify({"error": "No production model"}), 404
        v = versions[0]
        return jsonify({
            "model"  : model_name,
            "version": v.version,
            "run_id" : v.run_id,
            "stage"  : v.current_stage,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    video_id = body.get("video_id", "").strip()

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400
    if not re.match(r"^[A-Za-z0-9_\-]{11}$", video_id):
        return jsonify({"error": "Invalid YouTube video ID format"}), 400

    try:
        # 1. Fetch comments
        comments = fetch_comments(video_id)
        if not comments:
            return jsonify({"error": "No comments found for this video"}), 404

        # 2. Predict
        texts        = [c["text"] for c in comments]
        preds, scores = predict_sentiment(texts)

        # 3. Assemble response
        top_25 = [
            {
                "comment"  : c["text"],
                "author"   : c["author"],
                "sentiment": int(p),
                "score"    : round(float(s), 3),
            }
            for c, p, s in zip(comments[:25], preds[:25], scores[:25])
        ]

        result = {
            "video_id"       : video_id,
            "stats"          : compute_stats(comments, preds, scores),
            "sentiment_counts": sentiment_counts(preds),
            "trend"          : build_trend(comments, preds),
            "word_freq"      : word_frequency(texts),
            "top_comments"   : top_25,
        }
        return jsonify(result), 200

    except RuntimeError as e:
        logger.error("RuntimeError: %s", e)
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Unhandled error in /analyze")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info("Starting SentimentScope API on port %d…", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
