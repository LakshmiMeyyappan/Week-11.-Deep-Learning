import cv2
import time
import os
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# List of messy objects to check
messy_objects = ['cup', 'bottle', 'chair', 'book', 'backpack', 'suitcase', 'teddy bear', 'tv', 'remote']

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("[INFO] Capturing image... Please position your room")
time.sleep(2)  # Wait for webcam to adjust

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image from webcam.")
    exit()

# Run YOLO on the image
results = model.predict(source=frame, conf=0.4, verbose=False)
boxes = results[0].boxes
names = model.names

# Messy object detection
detected_messy_items = []
for box in boxes:
    cls_id = int(box.cls[0])
    label = names[cls_id]
    if label in messy_objects:
        detected_messy_items.append(label)

# Annotate the image
annotated = results[0].plot()

if detected_messy_items:
    y_offset = 30
    cv2.putText(annotated, "Detected Messy Items:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for i, item in enumerate(set(detected_messy_items)):
        cv2.putText(annotated, f"- {item}", (20, y_offset + 25 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
else:
    # If no messy items found
    cv2.putText(annotated, "Room is Clean", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("CapturedImages", exist_ok=True)
output_path = f"CapturedImages/detected_{timestamp}.jpg"
cv2.imwrite(output_path, annotated)

# Display the annotated image
cv2.imshow("Room Mess Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"[INFO] Image saved to {output_path}")
