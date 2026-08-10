"""Live demo: upload an image or short video, get a real/fake prediction.

Runs on CPU deliberately -- the GPU is busy with the background
preprocessing/training jobs for the actual study, and a single demo
inference is fast enough on CPU that it doesn't need to contend for it.

Until a trained checkpoint exists at runs/<model>_all_c23/best.pt, this
serves predictions from the ImageNet-pretrained backbone with a random/
untrained classification head -- clearly flagged as such in the API
response so the frontend can warn the user the number is meaningless.
Once training finishes, it auto-picks up the checkpoint on next server
start (or call /api/reload).

Usage:
  python -m webapp.backend
  # then open http://127.0.0.1:8000
"""
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocess import crop_face, get_detector, sample_frame_indices
from src.data.dataset import build_transforms
from src.models.factory import create_model, model_img_size

DEVICE = "cpu"
RUNS_DIR = ROOT / "runs"
FRAMES_PER_VIDEO = 12
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="Deepfake Detection Demo")
_detector = None
_model_cache = {}


def get_face_detector():
    global _detector
    if _detector is None:
        _detector = get_detector(DEVICE)
    return _detector


def load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    ckpt = RUNS_DIR / f"{name}_all_c23" / "best.pt"
    model = create_model(name, pretrained=True).to(DEVICE).eval()
    trained = False
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state["model"])
        trained = True
    entry = {"model": model, "trained": trained,
             "checkpoint": str(ckpt) if trained else None}
    _model_cache[name] = entry
    return entry


def available_models():
    names = ["xception", "cnn_vit"]
    return {n: {"trained": (RUNS_DIR / f"{n}_all_c23" / "best.pt").exists()}
            for n in names}


def predict_frames(frames_bgr, model_name: str):
    entry = load_model(model_name)
    model = entry["model"]
    img_size = model_img_size(model_name)
    transform = build_transforms(img_size, train=False)
    detector = get_face_detector()

    per_frame = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb)
        if boxes is None or len(boxes) == 0:
            continue
        crop = crop_face(bgr, boxes[0], img_size)
        if crop is None:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        x = transform(Image.fromarray(crop_rgb)).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        per_frame.append(prob)

    if not per_frame:
        raise HTTPException(422, "No face detected in the uploaded media.")

    mean_prob = float(np.mean(per_frame))
    return {
        "model": model_name,
        "trained": entry["trained"],
        "checkpoint": entry["checkpoint"],
        "frames_used": len(per_frame),
        "per_frame_fake_prob": per_frame,
        "fake_probability": mean_prob,
        "label": "fake" if mean_prob >= 0.5 else "real",
    }


@app.get("/api/models")
def api_models():
    return available_models()


@app.post("/api/reload")
def api_reload():
    _model_cache.clear()
    return available_models()


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...), model: str = "xception"):
    if model not in ("xception", "cnn_vit"):
        raise HTTPException(400, f"Unknown model '{model}'")
    ext = Path(file.filename or "").suffix.lower()
    data = await file.read()

    if ext in IMAGE_EXTS or (not ext and len(data) > 0 and ext not in VIDEO_EXTS):
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "Could not decode image.")
        frames = [bgr]
    elif ext in VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total, FRAMES_PER_VIDEO)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
        Path(tmp_path).unlink(missing_ok=True)
        if not frames:
            raise HTTPException(400, "Could not read any frames from video.")
    else:
        raise HTTPException(400, f"Unsupported file type: {ext or '(none)'}")

    return predict_frames(frames, model)


static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
