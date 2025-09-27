from pydantic import BaseModel, Field
from typing import List, Optional

class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    label: Optional[str] = None

class DetectResponse(BaseModel):
    status: str = Field(default="ok")
    num_detections: int
    detections: List[Box]
    image_path: Optional[str] = None
    json_path: Optional[str] = None