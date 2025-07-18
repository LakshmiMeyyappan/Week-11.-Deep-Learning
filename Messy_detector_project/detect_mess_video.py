# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 22:12:49 2025
@author: admin
"""

import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# Load the YOLOv8 model
model_names = YOLO('yolov8n.pt')

# Load the trained MobileNetV2 classifier
print("[INFO] loading model...")
model = load_model("messy_clean_detector.h5")

# Start webcam video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

# Define known messy objects from YOLO COCO classes
messy_objects = ['cup', 'bottle', 'chair', 'book', 'backpack', 'suitcase', 'teddy bear', 'tv', 'remote']

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Clean_output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model_names.predict(source=frame, conf=0.4, verbose=False)
    boxes = results[0].boxes
    names = model_names.names

    detected_messy_items = []
    mess_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        if label in messy_objects:
            mess_count += 1
            detected_messy_items.append(label)

    # Estimate mess score
    mess_score = max(0, 100 - mess_count * 10)

    # Create annotated frame with boxes
    annotated_frame = results[0].plot()

    # Add mess score text
    cv2.putText(annotated_frame, f'Mess Score: {mess_score}/100', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display messy object names on the screen
    #y_offset = 70
    #cv2.putText(annotated_frame, "Detected Messy Items:", (20, y_offset),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    #for i, item in enumerate(set(detected_messy_items)):
        #cv2.putText(annotated_frame, f"- {item}", (30, y_offset + (25 * (i + 1))),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display messy object names or "Clean" status on the screen
    y_offset = 70
    if detected_messy_items:
        messy_list = list(set(detected_messy_items))
        cv2.putText(annotated_frame, f"Messy Items ({len(messy_list)}):", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for i, item in enumerate(messy_list):
            cv2.putText(annotated_frame, f"- {item}", (30, y_offset + (25 * (i + 1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(annotated_frame, "No messy items detected - Clean", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize and preprocess the frame for MobileNet
    frame_resized = imutils.resize(frame, width=400)
    image = cv2.resize(frame_resized, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

        # Predict using MobileNetV2 (messy or clean)
    (pred_clean, pred_messy) = model.predict(image)[0]  # <-- FIXED ORDER
    label = "Messy" if pred_messy > pred_clean else "Clean"
    confidence = max(pred_messy, pred_clean) * 100
    
    # Optional Debug Print
    print(f"MobileNet Predictions => Clean: {pred_clean:.4f}, Messy: {pred_messy:.4f}")
    
    # Draw the label and bounding box on MobileNet frame
    color = (0, 0, 255) if label == "Messy" else (0, 255, 0)
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(frame_resized, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame_resized, (5, 5), (395, 295), color, 2)
# Write to output video file
    out.write(annotated_frame)
    # Show both outputs
    cv2.imshow("YOLOv8 - Messy Room Detector", annotated_frame)
    cv2.imshow("Room Status - MobileNetV2", frame_resized)
    
    
   
    
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
vs.release()
out.release()
cv2.destroyAllWindows()
