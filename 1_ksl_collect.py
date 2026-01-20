import cv2
import numpy as np
import os
import mediapipe as mp
import time

# --- QT IMPORTS ---
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPainter, QFont, QColor, QFontDatabase

# --- 1. CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data') 

# Actions to record
actions = np.array(['chumreapsour', 'orkun', 'trov', 'nothing', 'howru', 'mineyte', 'deaf', 'soursdey', 'WC', 'i dont understand', 'zero']) 

# Dictionary to convert "chumreapsour" -> "ជម្រាបសួរ"
KHMER_MAP = {
    'chumreapsour': 'ជម្រាបសួរ', 'orkun': 'អរគុណ', 'trov': 'ត្រឹមត្រូវ',
    'howru': 'សុខសប្បាយទេ', 'mineyte': 'មិនអីទេ', 'deaf': 'មនុស្សថ្លង់',
    'soursdey': 'សួស្តី', 'nothing': '', 'WC': 'បង្គន់', 'i dont understand': 'ខ្ញុំមិនយល់'
}

# Config
no_sequences_to_add = 20  
sequence_length = 30
FONT_PATH = os.path.join('font', 'KhmerUI.ttf') 

# --- 2. SETUP QT FONT ENGINE ---
app = QApplication([]) 

if not os.path.exists(FONT_PATH):
    print(f"⚠️ ERROR: Font not found at {FONT_PATH}")
    FONT_FAMILY = "Arial" 
else:
    font_id = QFontDatabase.addApplicationFont(FONT_PATH)
    if font_id != -1:
        FONT_FAMILY = QFontDatabase.applicationFontFamilies(font_id)[0]
    else:
        FONT_FAMILY = "Arial"

# --- 3. ROBUST QT RENDERER ---
def draw_khmer_text(img, text, x, y, size=32, color=(0, 255, 0)):
    height, width, channels = img.shape
    bytes_per_line = 3 * width
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    qimg = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.TextAntialiasing)
    
    font = QFont(FONT_FAMILY, size)
    painter.setFont(font)
    
    # FIX: OpenCV uses BGR (Blue, Green, Red). Qt uses RGB.
    # We must swap the color channels here to avoid "weird" colors.
    # color[0]=B, color[1]=G, color[2]=R  -->  QColor(R, G, B)
    painter.setPen(QColor(color[2], color[1], color[0]))
    
    painter.drawText(int(x), int(y), text)
    painter.end()

    ptr = qimg.bits()
    ptr.setsize(height * width * 3)
    img_array = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# --- 4. MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- 5. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('KSL Data Collector', cv2.WINDOW_NORMAL)

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    
    for action in actions:
        try: os.makedirs(os.path.join(DATA_PATH, action))
        except: pass

    # LOOP THROUGH CLASSES
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        existing_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        
        if len(existing_files) == 0:
            start_sequence = 0
        else:
            files_ints = [int(f.split('.')[0]) for f in existing_files]
            start_sequence = max(files_ints) + 1

        end_sequence = start_sequence + no_sequences_to_add
        
        # --- PHASE 1: WAITING ROOM ---
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # --- DISPLAY INFO IN KHMER ---
            khmer_action = KHMER_MAP.get(action, action) # Get translation
            
            # Draw Status
            image = draw_khmer_text(image, f'COLLECTING: {khmer_action}', 50, 50, size=35, color=(0, 255, 255)) # Yellow
            image = draw_khmer_text(image, f'Existing: {len(existing_files)} | Adding: {no_sequences_to_add}', 50, 90, size=25, color=(255, 255, 255))
            
            image = draw_khmer_text(image, '[SPACE] Start Recording', 50, 150, size=30, color=(0, 255, 0)) # Green
            image = draw_khmer_text(image, '[S] Skip Class', 50, 400, size=30, color=(255, 255, 0)) # Cyan
            image = draw_khmer_text(image, '[Q] Quit', 50, 450, size=30, color=(0, 0, 255)) # Red

            cv2.imshow('KSL Data Collector', image)

            key = cv2.waitKey(10) & 0xFF
            if key == 32: # SPACE
                break
            if key == ord('s'): # SKIP
                start_sequence = end_sequence 
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        # --- PHASE 2: RECORDING LOOP ---
        for sequence in range(start_sequence, end_sequence):
            
            # --- REST PERIOD ---
            for countdown in [3, 2, 1]: 
                start_time = time.time()
                while time.time() - start_time < 1.0: 
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    khmer_action = KHMER_MAP.get(action, action)
                    msg = f'Recording: {khmer_action} ({sequence})'
                    image = draw_khmer_text(image, msg, 20, 40, size=30, color=(0, 255, 255))
                    
                    cv2.putText(image, str(countdown), (280, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                    cv2.imshow('KSL Data Collector', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

            # --- CAPTURE FRAMES ---
            frames = []
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Recording Indicator
                cv2.circle(image, (50, 50), 20, (0, 0, 255), -1) 
                image = draw_khmer_text(image, 'REC', 80, 60, size=25, color=(0, 0, 255))
                
                status_text = f'{action} file: {sequence}.npy'
                image = draw_khmer_text(image, status_text, 50, 450, size=30, color=(0, 255, 0))
                
                cv2.imshow('KSL Data Collector', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                keypoints = extract_keypoints(results)
                frames.append(keypoints)
            
            npy_path = os.path.join(DATA_PATH, action, str(sequence))
            np.save(npy_path, np.array(frames))
            print(f"Captured {action} sequence {sequence}")

    cap.release()
    cv2.destroyAllWindows()