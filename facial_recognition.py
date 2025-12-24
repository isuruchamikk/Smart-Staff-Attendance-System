import face_recognition
import cv2
import numpy as np
import time
import pickle
import RPi.GPIO as GPIO
import csv
from datetime import datetime

# --- Load pre-trained encodings ---
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# --- GPIO Setup ---
SERVO_PIN = 18
BUTTON_PIN = 23  # GPIO pin for push button
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button input with pull-up

# --- Servo Setup ---
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

# --- Initialize Raspberry Pi Camera ---
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(3, 1280)
camera.set(4, 720)

# --- Variables ---
cv_scaler = 4
face_locations, face_encodings, face_names = [], [], []
frame_count = 0
start_time = time.time()
fps = 0

servo_state = False
last_seen_time = 0
SERVO_TIMEOUT = 5
BUTTON_PRESS_TIMEOUT = 5  # seconds door stays open on button press
last_button_press = 0

last_log_time = time.time()
LOG_INTERVAL = 5 * 60
detections = {}

# --- CSV Setup ---
CSV_FILE = "detections_log.csv"
with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "name", "confidence", "detected_duration", "day"])

# --- Helper Functions ---
def open_servo():
    global servo_state, last_seen_time
    pwm.ChangeDutyCycle(7.5)  # open position
    servo_state = True
    last_seen_time = time.time()
    print("[SERVO] Door Opened")

def close_servo():
    global servo_state
    pwm.ChangeDutyCycle(2.5)  # closed position
    servo_state = False
    print("[SERVO] Door Closed")

def process_frame(frame):
    resized_frame = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb_resized = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_resized)
    face_encodings = face_recognition.face_encodings(rgb_resized, face_locations, model='large')

    names, confidences = [], []
    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        confidence = round((1 - distances[best_match_index]) * 100, 2)
        name = known_face_names[best_match_index] if confidence > 50 else "Unknown"
        names.append(name)
        confidences.append(confidence)

    return face_locations, names, confidences

def draw_results(frame, locations, names, confidences):
    global servo_state, last_seen_time, detections
    current_time = time.time()
    detected_staff = False

    for (top, right, bottom, left), name, conf in zip(locations, names, confidences):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left, top - 35), (right, top), (244, 42, 3), cv2.FILLED)
        cv2.putText(frame, f"{name} {conf:.1f}%", (left + 6, top - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        staff = ["dinith", "isuru", "dulaj", "nimesh"]
        if name.lower() in staff:
            detected_staff = True
            last_seen_time = current_time

        if name != "Unknown":
            if name not in detections:
                detections[name] = {"start": current_time, "last_seen": current_time, "confidence": conf}
            else:
                detections[name]["last_seen"] = current_time
                detections[name]["confidence"] = conf

    # --- Face detection-based control ---
    if detected_staff and not servo_state:
        open_servo()

    elif servo_state and (current_time - last_seen_time > SERVO_TIMEOUT) and not is_button_pressed():
        close_servo()

def log_detections():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for name, info in detections.items():
        duration = round(info["last_seen"] - info["start"], 2)
        rows.append([now, name, info["confidence"], duration, day])

    if rows:
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"[LOG] {len(rows)} detections logged")

    detections.clear()

def is_button_pressed():
    return GPIO.input(BUTTON_PIN) == GPIO.LOW

# --- Main Loop ---
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Camera not found.")
            break

        # --- Face recognition ---
        locations, names, confidences = process_frame(frame)
        draw_results(frame, locations, names, confidences)

        # --- Button control ---
        if is_button_pressed():
            current_time = time.time()
            if current_time - last_button_press > 0.5:  # debounce
                last_button_press = current_time
                print("[BUTTON] Manual open triggered")
                open_servo()

        # --- Auto close after button timeout ---
        if servo_state and (time.time() - last_seen_time > BUTTON_PRESS_TIMEOUT) and not is_button_pressed():
            close_servo()

        # --- FPS display ---
        fps = (fps * 0.9) + (1 / (time.time() - start_time)) * 0.1
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # --- Logging every 5 minutes ---
        if time.time() - last_log_time >= LOG_INTERVAL:
            log_detections()
            last_log_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    log_detections()
    camera.release()
    cv2.destroyAllWindows()
    close_servo()
    pwm.stop()
    GPIO.cleanup()
    print("[INFO] System stopped and cleaned up successfully.")
