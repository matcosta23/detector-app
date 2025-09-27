from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO

from .config import DetectorConfig
from .postproc import apply_class_map

class YOLODetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        # Allow model to be .pt, .onnx, or .engine; Ultralytics handles backends
        model_path = self.cfg.model_path
        if not Path(model_path).exists():
            # let Ultralytics auto-download if it's a well-known alias (e.g., 'yolov8n.pt')
            pass
        self.model = YOLO(model_path)

    def _predict(self, image: np.ndarray):
        # Run one image through the model
        res = self.model.predict(
            source=image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            max_det=self.cfg.max_det,
            verbose=False,
            stream=False
        )
        return res

    def detect_image(self, image: np.ndarray, save_vis: bool = True, out_dir: str | os.PathLike = "outputs", basename: str | None = None):
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        results = self._predict(image)
        elapsed = (time.time() - t0) * 1000.0

        # Ultralytics returns a Results or list; ensure we have one
        r = results[0] if isinstance(results, list) else results

        dets: List[Dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                dets.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": float(c),
                    "cls": int(k),
                    "label": None
                })

        dets = apply_class_map(dets, self.cfg.classes_map)

        img_out_path = None
        if save_vis:
            plot = r.plot()
            name = basename or f"pred_{int(time.time()*1000)}.jpg"
            img_out_path = os.path.join(out_dir, name)
            cv2.imwrite(img_out_path, plot)

        # Save JSON
        name_json = (basename or Path(img_out_path).stem if img_out_path else f"pred_{int(time.time()*1000)}") + ".json"
        json_path = os.path.join(out_dir, name_json)
        with open(json_path, "w") as f:
            json.dump({"dets": dets, "elapsed_ms": elapsed}, f, indent=2)

        return dets, img_out_path, json_path, elapsed