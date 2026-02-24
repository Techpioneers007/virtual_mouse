import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import urllib.request
import os
import math # NEW: Required for calculating thumb distance

# 1. Setup Models and API
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Setup Webcam & Screen Size
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size() 

# 3. Control Variables
smooth_scroll_y = None
smooth_cursor_x = None
smooth_cursor_y = None
smoothing_factor = 0.2  
top_boundary = 0.45     
bottom_boundary = 0.55  
pyautogui.PAUSE = 0 
pyautogui.FAILSAFE = False 

# NEW: Prevents the mouse from clicking a million times a second
is_clicked = False 

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    
    cv2.line(image, (0, int(img_h * top_boundary)), (img_w, int(img_h * top_boundary)), (0, 255, 255), 2)
    cv2.line(image, (0, int(img_h * bottom_boundary)), (img_w, int(img_h * bottom_boundary)), (0, 255, 255), 2)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]
        
        # --- FINGER STATES ---
        index_extended = landmarks[8].y < landmarks[6].y
        middle_extended = landmarks[12].y < landmarks[10].y
        ring_extended = landmarks[16].y < landmarks[14].y
        pinky_extended = landmarks[20].y < landmarks[18].y
        
        index_curled = landmarks[8].y > landmarks[6].y
        middle_curled = landmarks[12].y > landmarks[10].y
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y

        # NEW: THUMB LOGIC (Calculate distance from thumb tip to index knuckle)
        thumb_tip = landmarks[4]
        index_knuckle = landmarks[5]
        thumb_distance = math.hypot(thumb_tip.x - index_knuckle.x, thumb_tip.y - index_knuckle.y)
        
        # If distance is large, thumb is popped out. If small, it's tucked in.
        thumb_popped_out = thumb_distance > 0.08 

        # --- 1. OPEN PALM -> PAUSE ---
        if index_extended and middle_extended and ring_extended and pinky_extended:
            cv2.putText(image, "PALM: Paused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            smooth_scroll_y = None 
            smooth_cursor_x = None
            smooth_cursor_y = None
            is_clicked = False

        # --- 2. FIST -> MOVE CURSOR & CLICK ---
        elif index_curled and middle_curled and ring_curled and pinky_curled:
            raw_x = landmarks[9].x
            raw_y = landmarks[9].y
            
            if smooth_cursor_x is None or smooth_cursor_y is None:
                smooth_cursor_x = raw_x
                smooth_cursor_y = raw_y
            else:
                smooth_cursor_x = (smoothing_factor * raw_x) + ((1 - smoothing_factor) * smooth_cursor_x)
                smooth_cursor_y = (smoothing_factor * raw_y) + ((1 - smoothing_factor) * smooth_cursor_y)

            screen_x = int(smooth_cursor_x * screen_w)
            screen_y = int(smooth_cursor_y * screen_h)
            
            pyautogui.moveTo(screen_x, screen_y)
            cv2.circle(image, (int(raw_x * img_w), int(raw_y * img_h)), 10, (0, 255, 0), -1)
            
            # --- THE CLICK INSTRUCTOR ---
            if thumb_popped_out:
                if not is_clicked:
                    pyautogui.click() # Triggers the actual OS click
                    is_clicked = True
                cv2.putText(image, "LEFT CLICK!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                is_clicked = False # Reset when thumb is tucked back in
                cv2.putText(image, "FIST: Moving Mouse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            smooth_scroll_y = None 

        # --- 3. POINTING -> SCROLL MODE ---
        elif index_extended and middle_curled and ring_curled and pinky_curled:
            raw_y = landmarks[8].y
            
            if smooth_scroll_y is None:
                smooth_scroll_y = raw_y
            else:
                smooth_scroll_y = (smoothing_factor * raw_y) + ((1 - smoothing_factor) * smooth_scroll_y)
            
            cv2.circle(image, (int(landmarks[8].x * img_w), int(smooth_scroll_y * img_h)), 10, (255, 0, 0), -1)

            if smooth_scroll_y < top_boundary: 
                pyautogui.scroll(40) 
                cv2.putText(image, "SCROLLING UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif smooth_scroll_y > bottom_boundary:
                pyautogui.scroll(-50) 
                cv2.putText(image, "SCROLLING DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(image, "POINTING: Ready to Scroll", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            smooth_cursor_x = None
            smooth_cursor_y = None
            is_clicked = False

        else:
            smooth_scroll_y = None
            smooth_cursor_x = None
            smooth_cursor_y = None
            is_clicked = False

    cv2.imshow('Virtual Mouse & Scroll', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()