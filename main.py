import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pyttsx3
import time

# ------------------------
# Config
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS_PATH = "model_weights.pth"
SEQ_LEN = 30
USE_TWO_HANDS = True
SMOOTHING_WINDOW = 5
CLASS_NAMES = [chr(i) for i in range(65, 91)]  # A–Z
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------
# Speech setup
# ------------------------
engine = pyttsx3.init(driverName='sapi5')  # force Windows voice
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

def speak_letter(letter):
    """Speaks the detected letter"""
    try:
        engine.say(letter)
        engine.runAndWait()
    except Exception as e:
        print(f"[Speech Error] {e}")

# ------------------------
# Model
# ------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_size=126, num_classes=26, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 30, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 30, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc
        x = self.transformer_encoder(x)
        return self.classifier(x.reshape(x.size(0), -1))

# ------------------------
# MediaPipe utilities
# ------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(results, width, height):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.multi_handedness and results.multi_hand_landmarks:
        for idx, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            lm = results.multi_hand_landmarks[idx]
            arr = np.array([[p.x * width, p.y * height, p.z] for p in lm.landmark], dtype=np.float32)
            if label == "Left":
                left = arr
            else:
                right = arr
    return left, right

def normalize_and_flatten(left, right, use_two_hands=True):
    combined = np.vstack([left, right])
    valid = combined[np.any(combined != 0, axis=1)]
    if valid.size == 0:
        return np.zeros(((21 * 3) * (2 if use_two_hands else 1),), dtype=np.float32)
    center = valid.mean(axis=0)
    scale = max(1e-6, np.max(np.abs(valid - center)))
    left = (left - center) / scale
    right = (right - center) / scale
    if use_two_hands:
        feat = np.concatenate([left.flatten(), right.flatten()])
    else:
        feat = right.flatten() if np.any(right != 0) else left.flatten()
    return feat.astype(np.float32)

# ------------------------
# Load model
# ------------------------
def load_model(path, seq_len, use_two_hands):
    input_size = (21 * 3) * (2 if use_two_hands else 1)
    model = TransformerClassifier(input_size=input_size, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    try:
        state = torch.load(path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[INFO] ✅ Loaded weights from: {path}")
    except Exception as e:
        print(f"[WARN] ⚠️ Could not load weights: {e}")
        print("[WARN] Running with random weights.")
    model.eval()
    return model

# ------------------------
# Main loop
# ------------------------
def main():
    model = load_model(MODEL_WEIGHTS_PATH, SEQ_LEN, USE_TWO_HANDS)
    seq_buffer = deque(maxlen=SEQ_LEN)
    pred_smoothing = deque(maxlen=SMOOTHING_WINDOW)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Webcam not detected.")
        return

    last_spoken = ""
    last_time = time.time()
    cooldown = 0.7  # seconds between voices

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            left, right = extract_hand_landmarks(results, w, h)
            feat = normalize_and_flatten(left, right, USE_TWO_HANDS)
            seq_buffer.append(feat)

            label, prob = "--", 0.0
            if len(seq_buffer) == SEQ_LEN:
                seq = np.expand_dims(np.stack(seq_buffer, axis=0), axis=0)
                seq_t = torch.from_numpy(seq).float().to(DEVICE)
                with torch.no_grad():
                    logits = model(seq_t)
                    scores = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    idx = int(np.argmax(scores))
                    label = CLASS_NAMES[idx]
                    prob = float(scores[idx])
                    pred_smoothing.append((label, prob))

                votes = {}
                for l, p in pred_smoothing:
                    votes[l] = votes.get(l, 0) + p
                label = max(votes, key=votes.get)
                prob = votes[label] / len(pred_smoothing)

                # Speak when the label changes
                if label != last_spoken and time.time() - last_time > cooldown:
                    print(f"🔊 Speaking: {label}")
                    speak_letter(label)
                    last_spoken = label
                    last_time = time.time()

            cv2.putText(frame, f"Pred: {label} ({prob*100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ISL Real-time Demo (with Speech)", frame)

            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
