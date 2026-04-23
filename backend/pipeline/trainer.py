import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from pipeline.normalizer import fit_normalizer

MODELS_DIR = os.getenv("MODELS_DIR", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "meta.json")


def _get_classifiers(scale_weight=1.0):
    clfs = {
        "SVM": SVC(kernel="rbf", probability=True, C=10, gamma="scale",
                    class_weight="balanced"),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000,
                             early_stopping=True, random_state=42,
                             learning_rate="adaptive"),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42,
                                                n_jobs=-1, class_weight="balanced",
                                                max_depth=20, min_samples_split=5),
    }
    if XGBOOST_AVAILABLE:
        clfs["XGBoost"] = XGBClassifier(max_depth=8, n_estimators=500,
                                        learning_rate=0.03,
                                        subsample=0.8, colsample_bytree=0.8,
                                        scale_pos_weight=scale_weight,
                                        eval_metric="logloss",
                                        random_state=42)
    if LGBM_AVAILABLE:
        clfs["LightGBM"] = LGBMClassifier(num_leaves=63, n_estimators=500,
                                           learning_rate=0.03,
                                           scale_pos_weight=scale_weight,
                                           random_state=42, verbose=-1)
    return clfs


def _tune_xgboost(X, y, skf, scale_weight):
    """Focused grid search around known-good XGBoost params."""
    print("  [GridSearch] Tuning XGBoost (focused grid)...")
    param_grid = {
        "max_depth": [6, 8, 10],
        "n_estimators": [500, 800],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [1, 3],
        "scale_pos_weight": [scale_weight],
    }
    base = XGBClassifier(eval_metric="logloss", random_state=42)
    gs = GridSearchCV(base, param_grid, cv=skf, scoring="accuracy",
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X, y)
    print(f"  [GridSearch] Best params: {gs.best_params_}")
    print(f"  [GridSearch] Best CV accuracy: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_score_


def _build_voting_ensemble(X, y, skf, scale_weight):
    """Soft-voting ensemble of XGBoost + SVM + MLP."""
    print("  [Voting] Building soft-vote ensemble of XGBoost + SVM + MLP...")
    estimators = [
        ("xgb", XGBClassifier(max_depth=8, n_estimators=500, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=scale_weight,
                               eval_metric="logloss", random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, C=10, gamma="scale",
                     class_weight="balanced")),
        ("mlp", MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000,
                              early_stopping=True, random_state=42,
                              learning_rate="adaptive")),
    ]
    vote = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    scores = cross_val_score(vote, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    mean_score = float(scores.mean())
    print(f"  [Voting] CV accuracy: {mean_score:.4f} (±{scores.std():.4f})")
    return vote, mean_score


def train(X_raw: np.ndarray, y: np.ndarray) -> dict:
    """
    Full training pipeline:
    1. Normalize + feature selection
    2. Compute class weight ratio for imbalance handling
    3. 5-fold CV on all classifiers + XGBoost GridSearch + Voting ensemble
    4. Save best model
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if len(np.unique(y)) < 2:
        return {"error": "Need both real and fake samples to train."}

    # Step 1: Normalize
    X = fit_normalizer(X_raw, y)

    # Step 2: Compute class weight ratio (no SMOTE — just cost-sensitive learning)
    n_real = int((y == 0).sum())
    n_fake = int((y == 1).sum())
    # scale_pos_weight: ratio of negative to positive class for XGBoost/LightGBM
    scale_weight = n_fake / max(n_real, 1)
    print(f"  [Balance] Real={n_real}, Fake={n_fake}, scale_pos_weight={scale_weight:.2f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = _get_classifiers(scale_weight)
    results = {}
    best_name, best_score, best_clf = None, -1, None

    # Step 3a: Evaluate all individual classifiers
    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
            mean_score = float(scores.mean())
            results[name] = {
                "cv_mean_accuracy": round(mean_score, 4),
                "cv_std": round(float(scores.std()), 4),
            }
            print(f"  ✅ {name}: {mean_score:.4f} (±{scores.std():.4f})")
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_clf = clf
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  ❌ {name}: {e}")

    # Step 3b: XGBoost GridSearchCV
    if XGBOOST_AVAILABLE:
        try:
            tuned_xgb, tuned_score = _tune_xgboost(X, y, skf, scale_weight)
            results["XGBoost_Tuned"] = {
                "cv_mean_accuracy": round(tuned_score, 4),
                "cv_std": 0.0,
            }
            if tuned_score > best_score:
                best_score = tuned_score
                best_name = "XGBoost_Tuned"
                best_clf = tuned_xgb
        except Exception as e:
            results["XGBoost_Tuned"] = {"error": str(e)}

    # Step 3c: Voting ensemble
    try:
        vote_clf, vote_score = _build_voting_ensemble(X, y, skf, scale_weight)
        results["Voting_Ensemble"] = {
            "cv_mean_accuracy": round(vote_score, 4),
            "cv_std": 0.0,
        }
        if vote_score > best_score:
            best_score = vote_score
            best_name = "Voting_Ensemble"
            best_clf = vote_clf
    except Exception as e:
        results["Voting_Ensemble"] = {"error": str(e)}

    # Step 4: Final fit on all data with best model
    best_clf.fit(X, y)
    joblib.dump(best_clf, BEST_MODEL_PATH)

    import json
    meta = {"best_model": best_name, "best_cv_accuracy": round(best_score, 4),
            "n_samples": len(y), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    return {"results": results, "best_model": best_name, "best_cv_accuracy": round(best_score, 4)}


def get_model_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    import json
    with open(META_PATH) as f:
        return json.load(f)


def is_trained() -> bool:
    return os.path.exists(BEST_MODEL_PATH)
