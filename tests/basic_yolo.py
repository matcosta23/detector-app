from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")
    metrics = model.val(data='SKU-110K.yaml', split='val', imgsz=640)
    print("DONE!")