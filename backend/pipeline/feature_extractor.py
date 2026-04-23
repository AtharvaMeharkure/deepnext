import cv2
import numpy as np
from scipy.fft import dct
from skimage.feature import local_binary_pattern
import os

try:
    # mediapipe >= 0.10.30 uses Tasks API
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp
    _USE_TASKS_API = True
except Exception:
    import mediapipe as mp
    _USE_TASKS_API = False

# Fallback: try legacy solutions API
if not _USE_TASKS_API:
    _mp_face_mesh_module = mp.solutions.face_mesh

# Key landmark indices for geometric analysis
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

MAX_FRAMES = int(os.getenv("MAX_FRAMES", 30))
FACE_SIZE = 128
NUM_DCT_COEFFS = 64


def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _lip_aspect_ratio(landmarks, lip_indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in lip_indices])
    height = np.linalg.norm(pts[2] - pts[-2])
    width = np.linalg.norm(pts[0] - pts[10])
    return height / (width + 1e-6)


def _bilateral_symmetry(landmarks, w, h):
    """Mirror left landmarks vs right landmarks across vertical axis."""
    left_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE])
    right_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE])
    # Flip right side horizontally around face center
    cx = w / 2.0
    right_mirrored = right_pts.copy()
    right_mirrored[:, 0] = 2 * cx - right_mirrored[:, 0]
    n = min(len(left_pts), len(right_mirrored))
    diffs = np.linalg.norm(left_pts[:n] - right_mirrored[:n], axis=1)
    return [diffs.mean(), diffs.std(), diffs.max(), diffs.min()]


def _dct_features(face_gray):
    """Compute DCT on grayscale face patch."""
    resized = cv2.resize(face_gray, (FACE_SIZE, FACE_SIZE)).astype(np.float32)
    dct_block = dct(dct(resized, axis=0, norm='ortho'), axis=1, norm='ortho')
    flat = dct_block.flatten()
    top = flat[:NUM_DCT_COEFFS]
    high_freq_ratio = np.abs(flat[NUM_DCT_COEFFS:]).mean() / (np.abs(flat[:NUM_DCT_COEFFS]).mean() + 1e-6)
    return list(top) + [high_freq_ratio]


def _fft_features(face_gray):
    """FFT magnitude spectrum statistics."""
    f = np.fft.fft2(cv2.resize(face_gray, (FACE_SIZE, FACE_SIZE)).astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    return [mag.mean(), mag.std(), mag.max(), mag.min(),
            mag[:FACE_SIZE//2, :FACE_SIZE//2].mean()]  # low-freq quadrant


def _texture_features(face_gray):
    """Texture analysis: Laplacian blur, LBP histogram, noise estimation."""
    resized = cv2.resize(face_gray, (FACE_SIZE, FACE_SIZE))
    feats = []

    # Laplacian variance (blur detection — deepfakes often have unnatural blur)
    lap = cv2.Laplacian(resized, cv2.CV_64F)
    feats.append(lap.var())

    # Local Binary Pattern histogram (10-bin) — captures micro-texture
    lbp = local_binary_pattern(resized, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(lbp_hist.tolist())

    # Noise estimation (high-pass filter residual)
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    noise = resized.astype(np.float64) - blur.astype(np.float64)
    feats.append(np.std(noise))
    feats.append(np.mean(np.abs(noise)))

    return feats  # 13 features


def _color_features(face_bgr):
    """Color space statistics — GANs often leave subtle color distribution traces."""
    resized = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    feats = []
    for ch in range(3):  # H, S, V channels
        feats.append(np.mean(hsv[:, :, ch]))
        feats.append(np.std(hsv[:, :, ch]))
    return feats  # 6 features


def _edge_features(face_gray):
    """Edge analysis — deepfakes may have different edge characteristics."""
    resized = cv2.resize(face_gray, (FACE_SIZE, FACE_SIZE))
    feats = []

    # Canny edge density
    edges = cv2.Canny(resized, 50, 150)
    feats.append(np.mean(edges > 0))  # edge pixel ratio

    # Sobel gradient statistics
    sobx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    soby = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobx**2 + soby**2)
    feats.append(np.mean(mag))
    feats.append(np.std(mag))
    feats.append(np.max(mag))

    return feats  # 4 features


def _optical_flow_features(prev_gray, curr_gray):
    """Farneback optical flow magnitude stats."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return [mag.mean(), mag.std(), mag.max()]


def _run_face_mesh(frame_bgr):
    """Run face mesh detection, returning landmark list or None. Handles both API versions."""
    if _USE_TASKS_API:
        import mediapipe as mp
        detector = _get_detector()
        if detector is None:
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.face_landmarks:
                return result.face_landmarks[0]
        except Exception as e:
            print(f'[DeepNext] Face detection error: {e}')
        return None
    else:
        # Legacy solutions API
        with _mp_face_mesh_module.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None


_FACE_MODEL_PATH = None
_DETECTOR = None  # cached global detector instance

def _get_face_model_path():
    """Download face_landmarker.task model if not present."""
    global _FACE_MODEL_PATH
    if _FACE_MODEL_PATH and os.path.exists(_FACE_MODEL_PATH):
        return _FACE_MODEL_PATH
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        print(f'[DeepDetect] Downloading face_landmarker.task model...')
        urllib.request.urlretrieve(url, model_path)
        print(f'[DeepDetect] Model downloaded to {model_path}')
    _FACE_MODEL_PATH = model_path
    return model_path


def _get_detector():
    """Return a cached FaceLandmarker detector (initialised once per process)."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    try:
        base_options = mp_tasks.BaseOptions(model_asset_path=_get_face_model_path())
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _DETECTOR = mp_vision.FaceLandmarker.create_from_options(options)
        print('[DeepNext] FaceLandmarker detector initialised.')
    except Exception as e:
        print(f'[DeepNext] FaceLandmarker init failed: {e}')
        _DETECTOR = None
    return _DETECTOR

def extract_frame_features(frame_bgr, prev_gray=None):
    """
    Extract a feature vector from a single BGR frame.
    Returns (feature_vector: np.ndarray, face_gray: np.ndarray | None, frame_gray: np.ndarray).
    """
    h, w = frame_bgr.shape[:2]
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    features = []

    lm = _run_face_mesh(frame_bgr)

    face_gray = None

    if lm is not None:

        # --- Facial Geometry ---
        nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
        chin = np.array([lm[CHIN].x * w, lm[CHIN].y * h])
        l_cheek = np.array([lm[LEFT_CHEEK].x * w, lm[LEFT_CHEEK].y * h])
        r_cheek = np.array([lm[RIGHT_CHEEK].x * w, lm[RIGHT_CHEEK].y * h])

        face_height = np.linalg.norm(nose - chin)
        face_width = np.linalg.norm(l_cheek - r_cheek)
        ear_l = _eye_aspect_ratio(lm, LEFT_EYE, w, h)
        ear_r = _eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        lar = _lip_aspect_ratio(lm, LIPS, w, h)

        geom = [face_height, face_width, face_height / (face_width + 1e-6),
                ear_l, ear_r, abs(ear_l - ear_r), lar,
                np.linalg.norm(nose - l_cheek), np.linalg.norm(nose - r_cheek)]
        features.extend(geom)

        # --- Bilateral Symmetry ---
        features.extend(_bilateral_symmetry(lm, w, h))

        # --- Face bounding box for freq domain ---
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
        y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))
        face_crop = frame_gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame_gray
        face_bgr_crop = frame_bgr[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame_bgr
        face_gray = face_crop

        # --- Frequency Domain ---
        features.extend(_dct_features(face_crop))
        features.extend(_fft_features(face_crop))

        # --- Texture Analysis (LBP, blur, noise) ---
        features.extend(_texture_features(face_crop))

        # --- Color Space Analysis ---
        features.extend(_color_features(face_bgr_crop))

        # --- Edge Analysis ---
        features.extend(_edge_features(face_crop))

    else:
        # No face detected — zero pad geometry + symmetry + freq + texture + color + edge
        features.extend([0.0] * (9 + 4 + NUM_DCT_COEFFS + 1 + 5 + 13 + 6 + 4))

    # --- Optical Flow ---
    if prev_gray is not None:
        try:
            curr_resized = cv2.resize(frame_gray, (320, 240))
            prev_resized = cv2.resize(prev_gray, (320, 240))
            features.extend(_optical_flow_features(prev_resized, curr_resized))
        except Exception:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32), face_gray, frame_gray


def extract_video_features(media_path: str) -> np.ndarray:
    """
    Process a video or image file and return a fixed-length statistical feature vector.
    Returns zero vector on failure.
    """
    ext = os.path.splitext(media_path)[1].lower()
    is_image = ext in ['.jpg', '.jpeg', '.png']

    frame_features = []
    prev_gray = None

    if is_image:
        frame = cv2.imread(media_path)
        if frame is not None:
            try:
                fv, _, _ = extract_frame_features(frame, None)
                frame_features.append(fv)
            except Exception:
                pass
    else:
        cap = cv2.VideoCapture(media_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // MAX_FRAMES)
            idx = 0
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx * step)
                ret, frame = cap.read()
                if not ret or idx >= MAX_FRAMES:
                    break
                try:
                    fv, _, _ = extract_frame_features(frame, prev_gray)
                    frame_features.append(fv)
                    prev_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
                except Exception:
                    pass
                idx += 1
            cap.release()

    if not frame_features:
        return np.zeros(EXPECTED_VECTOR_DIM, dtype=np.float32)

    arr = np.array(frame_features, dtype=np.float32)  # shape (N, D)

    # Statistical aggregation across frames: mean, std, min, max, skewness
    from scipy.stats import skew
    stats = np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
        skew(arr, axis=0)
    ])
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return stats.astype(np.float32)


# Single-frame feature dim: 9 geom + 4 sym + 65 dct + 5 fft + 13 texture + 6 color + 4 edge + 3 flow = 109
SINGLE_FRAME_DIM = 9 + 4 + (NUM_DCT_COEFFS + 1) + 5 + 13 + 6 + 4 + 3  # 109
EXPECTED_VECTOR_DIM = SINGLE_FRAME_DIM * 5  # 5 statistical aggregates = 545
