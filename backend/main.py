from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles
import os
import tempfile
import shutil
from dotenv import load_dotenv

load_dotenv()

from pipeline.feature_extractor import extract_video_features
from pipeline.predictor import predict
from pipeline.store import save_features, load_all_features, save_prediction, get_history, get_stats
from pipeline.trainer import train, is_trained, get_model_meta
from pipeline.csv_loader import train_from_csv, CSV_PATH


app = FastAPI(title="DeepFake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("store", exist_ok=True)


@app.get("/")
def root():
    return {"message": "DeepFake Detection API is running", "status": "ok"}


@app.get("/status")
def status():
    """Get system status: model info + dataset stats."""
    stats = get_stats()
    meta = get_model_meta()
    return {
        "model_trained": is_trained(),
        "model_info": meta,
        "dataset": stats,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "")),
    }


@app.post("/predict")
async def predict_video(
    file: UploadFile = File(...),
    label: int = Form(default=-1),  # -1 = no label (pure inference), 0 = real, 1 = fake
):
    """
    Upload a video for deepfake detection.
    Optionally provide label (0=real, 1=fake) to contribute to training dataset.
    """
    allowed_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".webp", ".heic")
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail=f"Unsupported format '{file.filename}'. Please use JPG/PNG or MP4/AVI.")

    # Save uploaded file temporarily
    tmp_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(tmp_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        # Step 1-4: Extract & aggregate features
        feature_vector = extract_video_features(tmp_path)

        # If labeled — save to feature store for future training
        if label in (0, 1):
            save_features(file.filename, label, feature_vector)

        # Step 5-7: Predict
        result = predict(tmp_path, feature_vector)
        result["filename"] = file.filename

        # Save prediction to history
        save_prediction(
            filename=file.filename,
            label=result["label"],
            confidence=result["confidence"],
            model_used=result.get("model_used", "unknown"),
            source=result.get("source", "unknown"),
            flags=result.get("flags", []),
        )

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/train-from-csv")
def train_from_csv_endpoint():
    """
    Train ML models directly from the bundled deepfake_detection_5000.csv.
    This gives immediate high-accuracy training without uploading any videos.
    """
    import os
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail=f"CSV not found at: {CSV_PATH}")
    try:
        result = train_from_csv()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def trigger_training():
    """Trigger ML model training on all stored labeled features."""
    X, y = load_all_features()
    if len(X) == 0:
        raise HTTPException(status_code=400, detail="No labeled samples found. Upload labeled videos first.")
    if len(X) < 10:
        raise HTTPException(status_code=400, detail=f"Need at least 10 samples to train. Have {len(X)}.")

    result = train(X, y)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/history")
def prediction_history(limit: int = 50):
    """Get recent prediction history."""
    return {"predictions": get_history(limit)}


@app.get("/health")
def health():
    return {"status": "healthy"}
