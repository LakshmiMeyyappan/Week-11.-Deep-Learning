import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define messy items to look for
messy_objects = ['cup', 'bottle', 'chair', 'book', 'backpack', 'suitcase', 'sports ball', 'tv', 'remote']

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Video writer settings
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_messy_room.avi', fourcc, 20.0, (640, 480))

# Track all seen items and messy items
all_detected_items = Counter()
all_messy_items = Counter()
final_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Frame not received.")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame, conf=0.3, verbose=False)
    boxes = results[0].boxes
    names = model.names

    current_items = []
    current_messy = []

    for box in boxes:
        cls_id = int(box.cls[0].item())
        name = names[cls_id]
        current_items.append(name)
        if name in messy_objects:
            current_messy.append(name)

    all_detected_items.update(current_items)
    all_messy_items.update(current_messy)

    # Annotate frame
    annotated = results[0].plot()

    # Mess score (arbitrary)
    mess_score = max(0, 100 - len(current_messy) * 10)
    cv2.putText(annotated, f"Mess Score: {mess_score}/100", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show all detected items
    cv2.putText(annotated, "Detected Items:", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset = 85
    for i, item in enumerate(sorted(set(current_items))):
        color = (0, 255, 255) if item in current_messy else (180, 180, 180)
        cv2.putText(annotated, f"- {item}", (30, y_offset + (25 * i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show "Room is Clean" if no messy items
    if not current_messy:
        cv2.putText(annotated, "Room is Clean", (20, y_offset + 30 + len(current_items) * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    final_frame = annotated.copy()
    out.write(annotated)
    cv2.imshow("Messy Room Detector", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Video stopped by user.")
        break

# Show final summary
if final_frame is not None:
    summary = final_frame.copy()
    cv2.putText(summary, "FINAL SUMMARY", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    y_offset = 70
    cv2.putText(summary, "All Items Seen:", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, (item, count) in enumerate(sorted(all_detected_items.items())):
        color = (0, 255, 255) if item in all_messy_items else (180, 180, 180)
        cv2.putText(summary, f"- {item}: {count}x", (30, y_offset + 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not all_messy_items:
        cv2.putText(summary, "Room was always clean!", (20, y_offset + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Final Summary", summary)
    cv2.waitKey(5000)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video saved as 'output_messy_room.avi'")
