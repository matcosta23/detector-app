import os
import typer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import load_config
from .yolo_runner import YOLODetector

app = typer.Typer(add_completion=False, help="Retail detector CLI")

@app.callback()
def main():
    """Root command group (enables subcommands even if only one)."""
    pass

@app.command()
def detect(
    source: str = typer.Option(..., "--source", "-s", help="Image/dir/URL to run detection on"),
    save_vis: bool = typer.Option(True, help="Save annotated images"),
    imgsz: int = typer.Option(None, help="Image size override"),
    conf: float = typer.Option(None, help="Confidence threshold override"),
    iou: float = typer.Option(None, help="IoU threshold override")
):
    cfg = load_config()
    if imgsz is not None: cfg.imgsz = imgsz
    if conf is not None: cfg.conf = conf
    if iou is not None: cfg.iou = iou

    detector = YOLODetector(cfg)

    # If source is a directory, iterate; otherwise run single
    src_path = Path(source)
    outputs_dir = os.environ.get("OUTPUTS_DIR", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if src_path.is_dir():
        images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")) + list(src_path.glob("*.jpeg"))
        for p in images:
            img = cv2.imread(str(p))
            dets, img_out, json_out, ms = detector.detect_image(img, save_vis=save_vis, out_dir=outputs_dir, basename=p.name)
            typer.echo(f"{p.name}: {len(dets)} dets, {ms:.1f} ms, vis={img_out}")
    else:
        # Let Ultralytics handle URLs and videos as well
        model = YOLO(cfg.model_path)
        results = model(source, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, device=cfg.device, stream=True, verbose=False)
        for r in results:
            name = os.path.basename(getattr(r, 'path', 'frame.jpg'))
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1,y1,x2,y2), c, k in zip(xyxy, confs, clss):
                    dets.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(c), "cls": int(k)})
            plot = r.plot()
            out_img = os.path.join(outputs_dir, name)
            cv2.imwrite(out_img, plot)
            typer.echo(f"{name}: {len(dets)} dets, saved {out_img}")

if __name__ == "__main__":
    app()