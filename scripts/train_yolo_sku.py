#!/usr/bin/env python
"""
Fine-tune YOLOv8n on SKU-110K (or a local dataset YAML) and then evaluate.
- Uses your configs/dataset.yaml to decide "builtin" (SKU-110K.yaml) vs "local".
- Writes runs under Ultralytics' standard runs/detect/train*/val* folders.
"""

from __future__ import annotations
from pathlib import Path
import argparse, yaml, os

from ultralytics import YOLO

# ---- dataset config loader (generic) ----
def load_dataset_cfg(path: str = "configs/dataset.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def resolve_data_arg(ds_cfg: dict) -> str:
    mode = ds_cfg.get("mode", "builtin")
    if mode == "builtin":
        ultra_yaml = ds_cfg.get("ultra_yaml", "SKU-110K.yaml")
        return ultra_yaml                      # triggers Ultralytics auto-download + CSV→YOLO conversion on first use
    elif mode == "local":
        yml = Path(ds_cfg["yaml_path"]).expanduser().resolve()
        if not yml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {yml}")
        return str(yml)
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights/yolov8n.pt", help="Starting weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=None, help="Override dataset imgsz")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")      # e.g., "0" or "cpu"
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default=None, help="Ultralytics project dir")
    p.add_argument("--name", default="sku110k-v8n")
    p.add_argument("--rect", action="store_true", help="Rectangular training")
    p.add_argument("--coslr", action="store_true", help="Use cosine LR")
    p.add_argument("--multiscale", action="store_true", help="Enable multi-scale training")
    p.add_argument("--single-cls", action="store_true", help="Treat dataset as single class")
    p.add_argument("--cache", default=None, choices=["ram", "disk"], help="Cache images for speed")
    p.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in last N epochs")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    args = p.parse_args()

    ds = load_dataset_cfg()
    data_arg = resolve_data_arg(ds)
    split = ds.get("split", "val")
    default_imgsz = ds.get("imgsz", 640)
    imgsz = args.imgsz or default_imgsz

    # Build and train
    model = YOLO(args.model)  # v8n checkpoint defines arch+weights
    model.train(
        data=data_arg,
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        rect=args.rect,                 # keep aspect ratio if desired (dense shelves)
        cos_lr=args.coslr,              # cosine LR scheduler
        multi_scale=args.multiscale,    # random scale per batch
        single_cls=args.single_cls,     # forces single-class training if needed
        cache=args.cache,               # "ram" for faster epochs if memory allows
        close_mosaic=args.close_mosaic, # stop mosaic near the end to stabilize
        patience=args.patience,         # early stopping
        # you can also tune lr0/lrf/momentum/weight_decay here; see docs
    )

    # Evaluate (e.g., on test split if provided in your YAML)
    model.val(
        data=data_arg,
        split=ds.get("split", "test"),  # "val" or "test"
        imgsz=imgsz,
        save_json=True
    )

if __name__ == "__main__":
    main()
