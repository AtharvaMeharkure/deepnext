import numpy as np
import joblib
import os
import cv2
import base64
import json
from PIL import Image
import io

from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = os.getenv("MODELS_DIR", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))


def predict_ml(feature_vector: np.ndarray) -> dict:
    """Run ML model inference."""
    from pipeline.normalizer import transform_features
    from pipeline.trainer import get_model_meta

    if not os.path.exists(BEST_MODEL_PATH):
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "flags": ["Error: Model not trained yet. Please train the model first."],
            "model_used": "None",
            "source": "error",
        }

    clf = joblib.load(BEST_MODEL_PATH)
    meta = get_model_meta()

    X = feature_vector.reshape(1, -1)

    try:
        X_t = transform_features(X)
        proba = clf.predict_proba(X_t)[0]
    except Exception as e:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "flags": [f"Error during ML inference: {e}. Feature shape mismatch: {X.shape}. Ensure your ML model is trained on the correct feature dimension."],
            "model_used": "None",
            "source": "error",
        }

    # Use 0.45 threshold instead of 0.5 — corrects bias toward FAKE
    # (model needs 45%+ fake probability to call it FAKE, otherwise REAL)
    FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.50"))
    fake_prob = float(proba[1])
    label = "FAKE" if fake_prob >= FAKE_THRESHOLD else "REAL"
    label_idx = 1 if label == "FAKE" else 0
    confidence = fake_prob if label == "FAKE" else float(proba[0])

    flags = []
    if confidence < CONFIDENCE_THRESHOLD:
        flags.append("⚠️ Low confidence — recommend human review")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "flags": flags,
        "model_used": meta.get("best_model", "ML Ensemble"),
        "source": "ml",
    }


def predict(video_path: str, feature_vector: np.ndarray) -> dict:
    """
    Primary prediction function.
    Now solely relies on the local ML model.
    """
    return predict_ml(feature_vector)
