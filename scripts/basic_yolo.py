from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/sku110k-v8n5/weights/best.pt")
    metrics = model.val(
        data='SKU-110K.yaml',
        split='val',
        imgsz=1280,
        save_json=True
    )
    print("DONE!")