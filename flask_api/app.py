"""
app.py — SentimentScope Flask API (Render deployment version)
loads model directly from pickle files instead of MLflow
"""

import os
import re
import pickle
import logging
from collections import Counter
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────
YT_API_KEY   = os.environ.get("YOUTUBE_API_KEY", "")
MAX_COMMENTS = int(os.environ.get("MAX_COMMENTS", "500"))
SENTIMENT_MAP = {1: "positive", -1: "negative", 0: "neutral"}

# ── Load model directly from files ────────────────────
MODEL_PATH      = os.environ.get("MODEL_PATH", "models/model.pkl")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "models/vectorizer.pkl")

logger.info("Loading model from %s", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

logger.info("Loading vectorizer from %s", VECTORIZER_PATH)
with open(VECTORIZER_PATH, "rb") as f:
    VECTORIZER = pickle.load(f)

logger.info("Model and vectorizer loaded successfully!")

# ── YouTube comment fetching ───────────────────────────
def fetch_comments(video_id: str) -> list[dict]:
    if not YT_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY environment variable is not set.")

    yt = build("youtube", "v3", developerKey=YT_API_KEY)
    comments = []
    next_page = None

    while len(comments) < MAX_COMMENTS:
        resp = yt.commentThreads().list(
            part        = "snippet",
            videoId     = video_id,
            maxResults  = min(100, MAX_COMMENTS - len(comments)),
            pageToken   = next_page,
            textFormat  = "plainText",
            order       = "relevance",
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


# ── Sentiment prediction ───────────────────────────────
def predict_sentiment(texts: list[str]):
    X     = VECTORIZER.transform(texts)
    preds = MODEL.predict(X)

    if hasattr(MODEL, "predict_proba"):
        proba   = MODEL.predict_proba(X)
        classes = MODEL.classes_
        pos_idx = list(classes).index(1)  if 1  in classes else 0
        neg_idx = list(classes).index(-1) if -1 in classes else 1
        scores  = (proba[:, pos_idx] - proba[:, neg_idx] + 1) / 2 * 10
    else:
        scores = np.where(preds == 1, 8.0, np.where(preds == -1, 2.0, 5.0))

    return preds, scores


# ── Analysis helpers ───────────────────────────────────
def sentiment_counts(preds):
    c = Counter(preds.tolist())
    return {
        "positive": c.get(1,  0),
        "negative": c.get(-1, 0),
        "neutral" : c.get(0,  0),
    }


def build_trend(comments, preds):
    from collections import defaultdict
    monthly = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for comment, pred in zip(comments, preds.tolist()):
        try:
            dt    = datetime.fromisoformat(comment["published_at"].replace("Z", "+00:00"))
            key   = dt.strftime("%Y-%m")
            label = SENTIMENT_MAP.get(pred, "neutral")
            monthly[key][label] += 1
        except Exception:
            pass
    return [{"month": k, **v} for k, v in sorted(monthly.items())]


def word_frequency(texts, top_n=80):
    STOPWORDS = {
        "the","a","an","is","it","in","on","at","to","and","or","but","i",
        "me","my","you","your","we","he","she","they","this","that","for",
        "of","with","be","was","are","were","have","has","had","not","so",
        "do","did","if","from","by","as","about","just","very","like","can",
        "will","no","yes","up","out","what","how","also",
    }
    counter = Counter()
    for text in texts:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        for w in words:
            if w not in STOPWORDS:
                counter[w] += 1
    return dict(counter.most_common(top_n))


def compute_stats(comments, preds, scores):
    texts     = [c["text"] for c in comments]
    authors   = set(c["author"] for c in comments)
    avg_len   = round(sum(len(t.split()) for t in texts) / max(len(texts), 1), 1)
    avg_score = round(float(scores.mean()), 2)
    return {
        "total"              : len(comments),
        "unique"             : len(authors),
        "avg_length"         : avg_len,
        "avg_sentiment_score": avg_score,
    }


# ── Routes ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "SentimentScope API"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    body     = request.get_json(force=True)
    video_id = body.get("video_id", "").strip()

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400
    if not re.match(r"^[A-Za-z0-9_\-]{11}$", video_id):
        return jsonify({"error": "Invalid YouTube video ID"}), 400

    try:
        comments      = fetch_comments(video_id)
        if not comments:
            return jsonify({"error": "No comments found"}), 404

        texts         = [c["text"] for c in comments]
        preds, scores = predict_sentiment(texts)

        top_25 = [
            {
                "comment"  : c["text"],
                "author"   : c["author"],
                "sentiment": int(p),
                "score"    : round(float(s), 3),
            }
            for c, p, s in zip(comments[:25], preds[:25], scores[:25])
        ]

        return jsonify({
            "video_id"        : video_id,
            "stats"           : compute_stats(comments, preds, scores),
            "sentiment_counts": sentiment_counts(preds),
            "trend"           : build_trend(comments, preds),
            "word_freq"       : word_frequency(texts),
            "top_comments"    : top_25,
        }), 200

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Error in /analyze")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ── Entry point ────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info("Starting SentimentScope API on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
