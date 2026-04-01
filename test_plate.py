from ultralytics import YOLO

model = YOLO('models/yolo11_plate.pt')
results = model.predict('creta.jpg', save=True, conf=0.25)

for r in results:
    print(f"Detections: {len(r.boxes)}")
    for box in r.boxes:
        print(f"  Confidence: {float(box.conf[0]):.2f}")
        print(f"  Box: {box.xyxy[0].tolist()}")