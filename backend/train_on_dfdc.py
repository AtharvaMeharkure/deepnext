import os
import glob
import numpy as np
import random
import sys
import cv2
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from pipeline.feature_extractor import extract_frame_features, EXPECTED_VECTOR_DIM
from pipeline.trainer import train as trainer_train
from pipeline.normalizer import fit_normalizer, transform_features

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_ROOT = r"c:\Users\Atharva Meharkure\Desktop\New folder (2)\archive (5)\DFDC\train"
# Strictly equal classes — prevents bias entirely
BALANCE_MODE = "equal"   # "equal" = cap both classes to minority count


def get_grouped_frames(directory):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(directory, ext)))

    groups = defaultdict(list)
    for path in all_files:
        basename = os.path.basename(path)
        parts = basename.split("_frame")
        vid_id = parts[0] if len(parts) == 2 else os.path.splitext(basename)[0]
        groups[vid_id].append(path)

    for vid_id in groups:
        def sort_key(x):
            name = os.path.basename(x)
            parts = name.split("_frame")
            if len(parts) == 2:
                try:
                    return int(parts[1].split(".")[0])
                except ValueError:
                    return 0
            return 0
        groups[vid_id].sort(key=sort_key)
    return groups


def extract_features_for_group(vid, frames_paths):
    """Extract 545-dim feature vector for a video group."""
    frame_features = []
    prev_gray = None
    for frame_path in frames_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        fv, _, fg = extract_frame_features(frame, prev_gray)
        frame_features.append(fv)
        prev_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))

    if not frame_features:
        return None

    arr = np.array(frame_features, dtype=np.float32)
    stats = np.concatenate([
        arr.mean(axis=0), arr.std(axis=0),
        arr.min(axis=0), arr.max(axis=0),
        skew(arr, axis=0)
    ])
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if stats.shape != (EXPECTED_VECTOR_DIM,) or np.sum(np.abs(stats)) == 0:
        return None
    return stats


def process_videos():
    real_dir = os.path.join(DATASET_ROOT, "real")
    fake_dir = os.path.join(DATASET_ROOT, "fake")

    real_groups = get_grouped_frames(real_dir)
    fake_groups = get_grouped_frames(fake_dir)

    real_vids = list(real_groups.keys())
    fake_vids = list(fake_groups.keys())

    print(f"Found {len(real_vids)} real groups | {len(fake_vids)} fake groups")

    random.seed(42)
    random.shuffle(real_vids)
    random.shuffle(fake_vids)

    # ── STRICT EQUAL SAMPLING ───────────────────────────────────────────────
    if BALANCE_MODE == "equal":
        n = min(len(real_vids), len(fake_vids))
        real_vids = real_vids[:n]
        fake_vids = fake_vids[:n]
        print(f"Balanced: {n} real | {n} fake  (strict equal sampling)")

    all_vids = [(vid, 0, real_groups[vid]) for vid in real_vids] + \
               [(vid, 1, fake_groups[vid]) for vid in fake_vids]

    X_raw, y = [], []
    skipped = 0

    print(f"\nExtracting features for {len(all_vids)} video groups...")
    for vid, label, frames_paths in tqdm(all_vids, total=len(all_vids)):
        try:
            feats = extract_features_for_group(vid, frames_paths)
            if feats is not None:
                X_raw.append(feats)
                y.append(label)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Successful: {len(y)} | Skipped: {skipped}")
    print(f"Real: {np.sum(y==0)} | Fake: {np.sum(y==1)}")
    print(f"X shape: {X_raw.shape}")
    print(f"{'='*60}\n")

    if len(np.unique(y)) < 2:
        print("Error: Not enough valid samples for both classes.")
        sys.exit(1)

    # ── HOLD-OUT TEST SET FOR CONFUSION MATRIX ─────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {len(y_trainval)} | Test set: {len(y_test)}")
    print(f"  Train → Real: {np.sum(y_trainval==0)} | Fake: {np.sum(y_trainval==1)}")
    print(f"  Test  → Real: {np.sum(y_test==0)} | Fake: {np.sum(y_test==1)}\n")

    # ── TRAIN ───────────────────────────────────────────────────────────────
    print("Training model...")
    result = trainer_train(X_trainval, y_trainval)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Model  : {result.get('best_model')}")
    print(f"Best CV Acc : {result.get('best_cv_accuracy')}")
    print(f"\nAll CV Results:")
    for name, res in result.get('results', {}).items():
        if 'error' in res:
            print(f"  ❌ {name}: {res['error']}")
        else:
            print(f"  ✅ {name}: {res['cv_mean_accuracy']:.4f} (±{res['cv_std']:.4f})")

    # ── CONFUSION MATRIX ON HOLD-OUT TEST SET ──────────────────────────────
    import joblib
    import os as _os
    MODELS_DIR = _os.getenv("MODELS_DIR", "models")
    clf = joblib.load(_os.path.join(MODELS_DIR, "best_model.joblib"))
    X_test_t = transform_features(X_test)

    y_pred = clf.predict(X_test_t)
    y_proba = clf.predict_proba(X_test_t)[:, 1]  # probability of FAKE

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS  (n={len(y_test)})")
    print(f"{'='*60}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"                 Predicted REAL   Predicted FAKE")
    print(f"  Actual REAL  :    {tn:>6}            {fp:>6}   (missed as FAKE)")
    print(f"  Actual FAKE  :    {fn:>6}            {tp:>6}   (correctly caught)")
    print(f"\n  True Positive Rate (Sensitivity/Recall FAKE) : {tp/(tp+fn):.2%}")
    print(f"  True Negative Rate (Specificity/Recall REAL) : {tn/(tn+fp):.2%}")
    print(f"  Precision REAL : {tn/(tn+fn):.2%}")
    print(f"  Precision FAKE : {tp/(tp+fp):.2%}")
    print(f"{'='*60}")

    # ── THRESHOLD ANALYSIS ──────────────────────────────────────────────────
    print(f"\nThreshold Analysis (default=0.5):")
    for thresh in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
        y_pred_t = (y_proba >= thresh).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        acc = accuracy_score(y_test, y_pred_t)
        real_recall = tn_t/(tn_t+fp_t) if (tn_t+fp_t) > 0 else 0
        fake_recall = tp_t/(tp_t+fn_t) if (tp_t+fn_t) > 0 else 0
        print(f"  thresh={thresh:.2f} | acc={acc:.3f} | real_recall={real_recall:.2%} | fake_recall={fake_recall:.2%}")


if __name__ == "__main__":
    process_videos()
