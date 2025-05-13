import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model for person detection
yolo_model = YOLO(r"C:\Users\sk.hasna\Desktop\women_safety_project\yolov8n.pt")
  # Ensure the path is correct

# Function to detect available camera indexes
def find_camera_index():
    for i in range(5):  # Check indexes 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1  # No camera found

# Try to find an available camera index
camera_index = find_camera_index()
if camera_index == -1:
    print("🚨 Error: No accessible camera found!")
    exit()

# Open Video Capture
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("🚨 Error: Unable to access the camera!")
    exit()

def detect_people(frame):
    results = yolo_model(frame)
    persons = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            persons.append((x1, y1, x2, y2))
    return persons

# Dummy Gender Classification Model (Replace with a trained model)
def classify_gender(face):
    return np.random.choice(["Male", "Female"], p=[0.7, 0.3])

def detect_anomalies(gender_data, frame_time):
    males = sum(1 for g in gender_data if g == "Male")
    females = sum(1 for g in gender_data if g == "Female")
    alerts = []

    if females == 1 and males >= 3:
        alerts.append("🚨 ALERT: Woman surrounded by men!")
    if females == 1 and frame_time >= 20:  # 8 PM or later
        alerts.append("🚨 ALERT: Lone woman detected at night!")
    return alerts

# Start Video Stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🚨 Error: Frame not captured")
        break

    persons = detect_people(frame)
    gender_data = []
    
    for (x1, y1, x2, y2) in persons:
        if y1 < y2 and x1 < x2:  # Ensure valid face region
            face = frame[y1:y2, x1:x2]
            gender = classify_gender(face)
            gender_data.append(gender)
            color = (0, 255, 0) if gender == "Female" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    current_hour = time.localtime().tm_hour
    anomalies = detect_anomalies(gender_data, current_hour)
    
    for i, alert in enumerate(anomalies):
        cv2.putText(frame, alert, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Women Safety Analytics", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
