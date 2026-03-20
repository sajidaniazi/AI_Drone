from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolov8n-cls.pt")

results = model.train(data=r"C:/Users/C00282704/PycharmProjects/PythonProject2/data/images", epochs=1, plots=False)