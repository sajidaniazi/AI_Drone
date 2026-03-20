from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("C:\\Users\\C00282704\\PycharmProjects\\PythonProject2\\runs\\classify\\train12\\weights\\best.pt")

classNames = ["Red_Lego"]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        probs = r.probs  # Classification probabilities

        if probs is not None:
            top_class_index = int(probs.top1)          # Index of top predicted class
            confidence = round(float(probs.top1conf), 2)  # Confidence score

            label = f"{classNames[top_class_index]}: {confidence}"
            print(label)

            # Display label on frame
            cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()