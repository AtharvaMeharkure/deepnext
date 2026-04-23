import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

MODELS_DIR = os.getenv("MODELS_DIR", "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
SELECTOR_PATH = os.path.join(MODELS_DIR, "selector.joblib")
TOP_K = 200  # ANOVA top-K features (increased for enhanced feature set)


def fit_normalizer(X: np.ndarray, y: np.ndarray):
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Clip outliers at 3-sigma per feature
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    X = np.clip(X, mu - 3 * sigma, mu + 3 * sigma)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(TOP_K, X_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selector, SELECTOR_PATH)
    return X_selected


def transform_features(X: np.ndarray) -> np.ndarray:
    """Apply saved scaler + selector to new feature vector(s)."""
    if not os.path.exists(SCALER_PATH) or not os.path.exists(SELECTOR_PATH):
        return X  # No normalizer fitted yet — return raw

    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip at pre-fitted bounds
    mu = scaler.mean_
    sigma = scaler.scale_
    X = np.clip(X, mu - 3 * sigma, mu + 3 * sigma)
    X_scaled = scaler.transform(X)
    return selector.transform(X_scaled)
