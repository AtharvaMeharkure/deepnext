"""
CSV Loader — trains ML models directly from deepfake_detection_5000.csv
Maps pre-computed feature columns to our training pipeline.
Run: python csv_loader.py
"""
import pandas as pd
import numpy as np
import os, json, sys

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'deepfake_detection_5000.csv')

# Numeric feature columns extracted from the CSV
FEATURE_COLS = [
    'face_confidence_score',
    'noise_level_db',
    'brightness_score',
    'sharpness_score',
    'contrast_score',
    'optical_flow_score',
    'texture_consistency_score',
    'lip_sync_score',
    'eye_blink_rate_per_min',
    'head_pose_variation',
    'frequency_domain_score',
    'dct_coefficient_variance',
    'ela_score',
    'cnn_feature_score',
    'model_confidence_pct',
]

CATEGORICAL_COLS = {
    'artifact_severity': {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3},
    'environment': None,        # will be label-encoded
    'lighting_condition': None,
    'compression_codec': None,
}


def load_and_prepare(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)

    # Binary label: REAL=0, FAKE=1
    df['label_int'] = (df['label'].str.upper() == 'FAKE').astype(int)

    # Encode artifact_severity
    df['artifact_severity_enc'] = df['artifact_severity'].map(
        lambda x: {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}.get(str(x), 0)
    )

    # Label-encode other categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in ['environment', 'lighting_condition', 'compression_codec']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].fillna('Unknown'))

    feature_cols = FEATURE_COLS + [
        'artifact_severity_enc',
        'environment_enc',
        'lighting_condition_enc',
        'compression_codec_enc',
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['label_int'].values.astype(np.int32)

    print(f"[CSV Loader] Loaded {len(X)} samples: {(y==0).sum()} real, {(y==1).sum()} fake")
    print(f"[CSV Loader] Feature shape: {X.shape}")
    return X, y


def train_from_csv(csv_path: str = CSV_PATH):
    """Full pipeline: load CSV → normalize → train all models → save best."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    X, y = load_and_prepare(csv_path)

    from pipeline.normalizer import fit_normalizer
    from pipeline.trainer import train

    result = train(X, y)
    print(f"[CSV Loader] Training complete: {result}")

    # Also populate feature store so /status shows sample counts
    from pipeline.store import save_features
    for i in range(len(X)):
        save_features(f"csv_sample_{i}", int(y[i]), X[i])
        if i % 500 == 0:
            print(f"  Stored {i}/{len(X)} samples...")

    print("[CSV Loader] All done.")
    return result


if __name__ == '__main__':
    result = train_from_csv()
    print(json.dumps(result, indent=2))
