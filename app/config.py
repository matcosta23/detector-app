from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os
import yaml

class DetectorConfig(BaseSettings):
    model_path: str = Field(default="weights/yolov8n.pt")
    imgsz: int = Field(default=640)
    conf: float = Field(default=0.25)
    iou: float = Field(default=0.45)
    device: str = Field(default="auto")
    max_det: int = Field(default=300)
    save_txt: bool = Field(default=False)
    save_json: bool = Field(default=True)
    save_vis: bool = Field(default=True)
    classes_map: dict[int, str] | None = None

    @staticmethod
    def from_yaml(path: str | os.PathLike) -> "DetectorConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return DetectorConfig(**data)

def load_config() -> DetectorConfig:
    # Prefer YAML if present, else env defaults
    cfg_path = os.environ.get("DETECTOR_YAML", "configs/detector.yaml")
    if Path(cfg_path).exists():
        return DetectorConfig.from_yaml(cfg_path)
    return DetectorConfig()