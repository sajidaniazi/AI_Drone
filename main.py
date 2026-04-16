from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolov8n.pt")

results = model.train(data="dataset.yaml", epochs=60, batch=16, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, flipud=0.1, degrees=15, translate=0.1, scale=0.5, shear=5.0,mosaic=1.0, mixup=0.1, copy_paste=0.1, name="red_lego_detector")
