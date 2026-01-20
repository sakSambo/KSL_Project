import cv2
import numpy as np
import os
import mediapipe as mp
import torch
import torch.nn as nn
import time

# --- QT IMPORTS ---
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPainter, QFont, QColor, QFontDatabase, QFontMetrics

# --- 1. CONFIGURATION ---
ACTIONS = np.array(['chumreapsour', 'orkun', 'trov', 'nothing', 'howru', 'mineyte', 'deaf', 'soursdey', 'WC', 'i dont understand', 'zero'])

KHMER_MAP = {
    'chumreapsour': 'ជម្រាបសួរ', 'orkun': 'អរគុណ', 'trov': 'ត្រឹមត្រូវ',
    'howru': 'សុខសប្បាយទេ', 'mineyte': 'មិនអីទេ', 'deaf': 'មនុស្សថ្លង់',
    'soursdey': 'សួស្តី', 'nothing': '', 'WC': 'បង្គន់', 'i dont understand': 'ខ្ញុំមិនយល់', 'zero': 'សូន្យ'
}

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
def draw_khmer_text(img, text, x, y, size=35, color=(255, 255, 255)):
    height, width, channels = img.shape
    bytes_per_line = 3 * width
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Copy data to Qt memory
    qimg = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.TextAntialiasing)
    
    # --- AUTO-SHRINK LOGIC ---
    font = QFont(FONT_FAMILY, size)
    fm = QFontMetrics(font)
    max_width = width - x - 10
    
    current_width = fm.width(text)
    while current_width > max_width and size > 10:
        size -= 2 
        font = QFont(FONT_FAMILY, size)
        fm = QFontMetrics(font)
        current_width = fm.width(text)
    # -------------------------

    painter.setFont(font)
    painter.setPen(QColor(color[0], color[1], color[2]))
    painter.drawText(int(x), int(y), text)
    painter.end()

    # Convert back to Numpy
    ptr = qimg.bits()
    ptr.setsize(height * width * 3)
    img_array = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# --- 4. MODEL ---
class KSLModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=3, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)

# --- 5. LOAD ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KSLModel(258, len(ACTIONS)).to(device)

if os.path.exists('ksl_model.pth'):
    model.load_state_dict(torch.load('ksl_model.pth', map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded on {device}")
else:
    print("ERROR: ksl_model.pth not found!"); exit()

# --- 6. MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# --- 7. MAIN LOOP ---
sequence, sentence = [], []
threshold = 0.9

# Initialize FPS variables
prev_time = 0 
curr_time = 0

cap = cv2.VideoCapture(0)
cv2.namedWindow('KSL Translator', cv2.WINDOW_NORMAL)

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            input_data = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            with torch.no_grad():
                prediction = model(input_data)
                probs = torch.softmax(prediction, dim=1)[0]
                confidence, idx = torch.max(probs, 0)

                if confidence.item() > threshold:
                    word = ACTIONS[idx.item()]
                    if word != 'nothing':
                        if not sentence or word != sentence[-1]:
                            sentence.append(word)
                            if len(sentence) > 5: sentence = sentence[-5:]

                        # BOTTOM UI
                        cv2.rectangle(image, (0, h - 90), (w, h), (0, 0, 0), -1)
                        image = draw_khmer_text(image, KHMER_MAP[word], 20, h - 30, size=40)
                        cv2.putText(image, f"{confidence.item()*100:.0f}%", (w-80, h-20), 0, 0.5, (100,255,100), 1)

        # TOP UI
        cv2.rectangle(image, (0, 0), (w, 85), (45, 45, 45), -1)
        sentence_kh = [KHMER_MAP[s] for s in sentence]
        full_sentence = "Sentence: " + " ".join(sentence_kh)
        image = draw_khmer_text(image, full_sentence, 10, 40, size=30, color=(0, 255, 255))
        
        cv2.putText(image, "[C] Clear Memory    [Q] Quit", (15, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # --- FPS LOGIC (Must be before imshow) ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the final image
        cv2.imshow('KSL Translator', image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('ឆ'): break 
        elif key == ord('c'): sentence.clear()

cap.release()
cv2.destroyAllWindows()