from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolov8n.yaml")

results = model.train(data="classification.yaml", epochs=1)