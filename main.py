from ultralytics import YOLO


# # Load a pretrained YOLO26n model
# model = YOLO("yolov8n.pt")

# results = model.train(data="classification.yaml", epochs=60, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)


# Load a pretrained YOLO26n model
model = YOLO("C:/Users/C00282704/Project/AI_Drone/runs/detect/train7/weights/last.pt")

results = model.train(resume=True)
