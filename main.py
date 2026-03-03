import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import pyttsx3
import threading
import math
import requests
from collections import deque

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.5
DWELL_FRAMES_REQUIRED = 12   
TOLERANCE_FRAMES = 8
COOLDOWN_SECONDS = 3.5
MAGNET_DISTANCE = 180       
SMOOTHING_FACTOR = 0.2       

# Dashboard URL
DWEET_IO_URL = "https://dweet.io/dweet/for/neuro-look-demo-123"

# Default Sensitivity
sensitivity = 2.8

# Narrative Mappings
OBJECT_MAP = {
    'bottle': "Water",
    'cup': "Medicine",
    'cell phone': "Call Doctor",
    'book': "Read Report"
}

# --- VISUAL STYLING (The "Cool" Factor) ---
MESH_COLOR = (100, 100, 100) # Grey sci-fi lines
IRIS_COLOR = (0, 255, 255)   # Yellow/Cyan for eyes
HUD_COLOR = (0, 255, 0)      # Matrix Green

# --- INITIALIZATION ---
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except:
    print("Audio Error: Driver not found.")

print("Loading Core Systems...")
model = YOLO('yolov8n.pt') 

# Initialize MediaPipe with Drawing Utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Critical for IRIS tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# --- STATE VARIABLES ---
dwell_counter = 0
miss_counter = 0  
last_object_id = None
locked_object_label = None 
last_spoken_time = 0
is_locked_on = False 
visuals_enabled = True # Toggle for the Face Mesh

# Calibration & Anchor
calibrated_center_x = None
calibrated_center_y = None
anchor_warning = False

# Smoothing
curr_gaze_x = 0
curr_gaze_y = 0

# --- ASYNC ALERT SYSTEM ---
def trigger_alert(message):
    def _worker():
        try:
            requests.get(f"{DWEET_IO_URL}?alert={message}&timestamp={time.time()}")
            engine.say(message)
            engine.runAndWait()
        except:
            pass
    thread = threading.Thread(target=_worker)
    thread.start()

# --- DRAWING HELPERS ---
def draw_cyberpunk_hud(frame, w, h, battery, fps):
    # 1. Vignette / Letterbox
    cv2.rectangle(frame, (0,0), (w, 50), (10, 10, 10), -1)
    cv2.rectangle(frame, (0, h-40), (w, h), (10, 10, 10), -1)
    
    # 2. Header Text
    cv2.putText(frame, "NEURO-LOOK [VISION CORE ONLINE]", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 2)
    
    # 3. System Stats (Right Side)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLOR, 1)
    cv2.putText(frame, f"BAT: {battery}%", (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLOR, 1)

    # 4. Status Light
    status_color = (0, 0, 255) if anchor_warning else (0, 255, 0)
    cv2.circle(frame, (w - 30, 25), 5, status_color, -1)
    
    # 5. Anchor Warning Overlay
    if anchor_warning:
        cv2.putText(frame, "WARNING: HEAD ALIGNMENT DRIFT", (w//2 - 200, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# --- MAIN LOOP ---
print("SYSTEM LIVE.")
print("'v' to Toggle Face Mesh Visuals")
print("'c' to Calibrate")

start_time = time.time()
frame_count = 0
fps = 0
prev_frame_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    # FPS Calculation
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # Mirror & Setup
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape
    screen_center_x, screen_center_y = img_w // 2, img_h // 2
    
    # Battery Sim
    battery_sim = max(100 - (int(time.time() - start_time) // 60), 20) 

    # 1. OBJECT DETECTION
    results = model(frame, stream=True, verbose=False, classes=[39, 41, 67, 73]) 
    
    detected_objects = []
    if results:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                raw_label = model.names[cls]
                display_label = OBJECT_MAP.get(raw_label, raw_label.upper())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detected_objects.append((x1, y1, x2, y2, cx, cy, display_label))

    # 2. GAZE & MESH TRACKING
    mesh_results = face_mesh.process(rgb_frame)
    raw_gaze_x, raw_gaze_y = None, None
    anchor_warning = False

    if mesh_results.multi_face_landmarks:
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        # --- A. DRAW THE COOL MESH (IF ENABLED) ---
        if visuals_enabled:
            # Draw the Tessellation (The Spiderweb)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Draw the Irises (The Eyes)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # --- B. TRACKING LOGIC ---
        landmark_4 = face_landmarks.landmark[4] # Nose Tip
        raw_x = int(landmark_4.x * img_w)
        raw_y = int(landmark_4.y * img_h)

        if calibrated_center_x is None:
            cv2.circle(frame, (img_w//2, img_h//2), 5, (0, 0, 255), -1)
            cv2.putText(frame, "LOOK CENTER & PRESS 'C'", (img_w//2 - 150, img_h//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            calibrated_center_x, calibrated_center_y = raw_x, raw_y
        
        # Anchor Check
        dist_from_anchor = math.sqrt((raw_x - calibrated_center_x)**2 + (raw_y - calibrated_center_y)**2)
        if dist_from_anchor > 80:
            anchor_warning = True
            # Red Line back to center
            cv2.line(frame, (raw_x, raw_y), (calibrated_center_x, calibrated_center_y), (0, 0, 255), 2)
        else:
            # Green Anchor Ring (Safe Zone)
            cv2.circle(frame, (calibrated_center_x, calibrated_center_y), 80, (0, 50, 0), 1)

        # Calculate Cursor Position
        offset_x = raw_x - calibrated_center_x
        offset_y = raw_y - calibrated_center_y
        target_x = screen_center_x + (offset_x * sensitivity)
        target_y = screen_center_y + (offset_y * sensitivity)

        # Smooth it
        if curr_gaze_x == 0: curr_gaze_x, curr_gaze_y = target_x, target_y
        curr_gaze_x = (target_x * SMOOTHING_FACTOR) + (curr_gaze_x * (1 - SMOOTHING_FACTOR))
        curr_gaze_y = (target_y * SMOOTHING_FACTOR) + (curr_gaze_y * (1 - SMOOTHING_FACTOR))
        raw_gaze_x, raw_gaze_y = int(curr_gaze_x), int(curr_gaze_y)

    # 3. AIM ASSIST
    final_gaze_x, final_gaze_y = raw_gaze_x, raw_gaze_y
    snapped_label = None
    
    if raw_gaze_x is not None and not anchor_warning:
        closest_dist = MAGNET_DISTANCE
        best_candidate = None 
        for obj in detected_objects:
            dist = math.sqrt((raw_gaze_x - obj[4])**2 + (raw_gaze_y - obj[5])**2)
            if dist < closest_dist:
                closest_dist = dist
                best_candidate = obj

        if best_candidate:
            cx, cy = best_candidate[4], best_candidate[5]
            snap_label = best_candidate[6]
            if snap_label != locked_object_label:
                locked_object_label = snap_label
                dwell_counter = 0 
            final_gaze_x, final_gaze_y = cx, cy
            snapped_label = snap_label

    # 4. RENDER UI
    draw_cyberpunk_hud(frame, img_w, img_h, battery_sim, fps)

    # Draw Cursor & Interaction
    if raw_gaze_x and not anchor_warning:
        # Ghost Cursor (Where you are looking)
        cv2.circle(frame, (raw_gaze_x, raw_gaze_y), 5, (150, 150, 150), -1)
        
        if final_gaze_x:
            # Main Cursor (Snapped)
            color = (0, 255, 255) if snapped_label else (0, 255, 0)
            
            # Sci-Fi Crosshair
            cv2.line(frame, (final_gaze_x-20, final_gaze_y), (final_gaze_x+20, final_gaze_y), color, 2)
            cv2.line(frame, (final_gaze_x, final_gaze_y-20), (final_gaze_x, final_gaze_y+20), color, 2)
            cv2.circle(frame, (final_gaze_x, final_gaze_y), 15, color, 2)
            
            # Tether Line
            if raw_gaze_x != final_gaze_x:
                cv2.line(frame, (raw_gaze_x, raw_gaze_y), (final_gaze_x, final_gaze_y), (0, 255, 255), 1)

    # Trigger Logic
    hit_this_frame = False
    if snapped_label and not anchor_warning:
        hit_this_frame = True
        dwell_counter += 1
        
        # Find Box & Animate
        for obj in detected_objects:
            if obj[6] == snapped_label:
                # Highlight Box
                cv2.rectangle(frame, (obj[0], obj[1]), (obj[2], obj[3]), (0, 255, 0), 2)
                
                # Draw "Scanning" Text
                cv2.putText(frame, f"TARGET LOCKED: {snapped_label}", (obj[0], obj[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Circular Loading Bar around cursor
                angle = int((dwell_counter / DWELL_FRAMES_REQUIRED) * 360)
                cv2.ellipse(frame, (final_gaze_x, final_gaze_y), (35, 35), 0, 0, angle, (0, 255, 255), 3)

                # TRIGGER
                if dwell_counter >= DWELL_FRAMES_REQUIRED and (current_time - last_spoken_time > COOLDOWN_SECONDS):
                    trigger_alert(f"I need {snapped_label}")
                    last_spoken_time = current_time
                    dwell_counter = 0
                    cv2.rectangle(frame, (0,0), (img_w, img_h), (0, 255, 0), 10) # Screen Flash
                break

    # Decay
    if not hit_this_frame:
        miss_counter += 1
        if miss_counter > TOLERANCE_FRAMES:
            locked_object_label = None
            dwell_counter = 0
            miss_counter = 0
    else:
        miss_counter = 0

    cv2.imshow('Neuro-Look: CYBERPUNK EDITION', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        if mesh_results.multi_face_landmarks:
            landmark_4 = mesh_results.multi_face_landmarks[0].landmark[4]
            calibrated_center_x = int(landmark_4.x * img_w)
            calibrated_center_y = int(landmark_4.y * img_h)
    elif key == ord('v'): # Toggle Visuals
        visuals_enabled = not visuals_enabled
    # Failsafes
    elif key == ord('1'): trigger_alert("I need Water")
    elif key == ord('2'): trigger_alert("Call Doctor")
    elif key == ord('3'): trigger_alert("Read Report")

cap.release()
cv2.destroyAllWindows()