from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolov8n.pt")

results = model.train(data="classification.yaml", epochs=60)
