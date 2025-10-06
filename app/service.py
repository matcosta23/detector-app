from __future__ import annotations
from pydantic import BaseModel
import io, os
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import numpy as np
import httpx
import cv2

from .config import load_config
from .yolo_runner import YOLODetector
from .schemas import Box, DetectResponse

app = FastAPI(title="Retail Product Detector", version="0.1.0")

# Serve annotated images and json files
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

cfg = load_config()
detector = YOLODetector(cfg)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None),
    iou: Optional[float] = Query(None),
    imgsz: Optional[int] = Query(None),
    save_vis: bool = Query(True)
):
    # Optional overrides
    if conf is not None: detector.cfg.conf = conf
    if iou is not None: detector.cfg.iou = iou
    if imgsz is not None: detector.cfg.imgsz = imgsz

    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    dets, img_out_path, json_path, _ = detector.detect_image(img, save_vis=save_vis, out_dir=OUTPUTS_DIR, basename=file.filename)

    boxes = [Box(**d) for d in dets]
    public_img = f"/outputs/{os.path.basename(img_out_path)}" if img_out_path else None
    public_json = f"/outputs/{os.path.basename(json_path)}" if json_path else None
    return DetectResponse(num_detections=len(boxes), detections=boxes, image_path=public_img, json_path=public_json)

class BatchRequest(BaseModel):
    paths: List[str]

@app.post("/detect-batch")
async def detect_batch(body: BatchRequest):
    # Basic implementation that loads local files or URLs (URLs via cv2)
    results = []
    for p in body.paths:
        if p.startswith("http"):
            img = cv2.imdecode(np.frombuffer((await (await httpx.AsyncClient().get(p)).aread()), np.uint8), cv2.IMREAD_COLOR)  # lazy URL fetch
            name = os.path.basename(p).split("?")[0] or "remote.jpg"
        else:
            img = cv2.imread(p)
            name = os.path.basename(p)
        dets, img_out_path, json_path, _ = detector.detect_image(img, save_vis=True, out_dir=OUTPUTS_DIR, basename=name)
        results.append({
            "path": p,
            "num_detections": len(dets),
            "image_path": f"/outputs/{os.path.basename(img_out_path)}" if img_out_path else None,
            "json_path": f"/outputs/{os.path.basename(json_path)}" if json_path else None,
            "detections": dets
        })
    return {"status": "ok", "results": results}