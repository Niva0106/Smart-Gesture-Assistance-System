import cv2
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import joblib
import numpy as np
import requests
import time
import threading
import winsound
from collections import Counter, deque

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "models/svm.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
LANDMARKER_PATH = "hand_landmarker.task"
SERVER_URL = "http://127.0.0.1:5000/api/gesture"
CHECKIN_STATUS_URL = "http://127.0.0.1:5000/api/check-in/current"

CONFIDENCE_THRESHOLD = 0.60
HISTORY_SIZE = 5
STABLE_COUNT = 3
REQUEST_COOLDOWN = 5
CHECKIN_POLL_INTERVAL = 0.5
CHECKIN_RESPONSE_DISPLAY_SECONDS = 7

# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading ML model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✓ SVM model loaded")
print("✓ Scaler loaded")

# ============================================================
# GESTURE NAMES
# ============================================================
gesture_names = {
    "call_nurse": "CALL NURSE",
    "okay": "I'M OKAY",
    "need_food_water": "NEED FOOD / WATER",
    "emergency": "EMERGENCY"
}

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ============================================================
# VARIABLES
# ============================================================
prediction_history = deque(maxlen=HISTORY_SIZE)
last_sent_gesture = None
last_sent_time = 0
stable_gesture = None
last_checkin_poll = 0
checkin_state = {"status": "none"}
last_checkin_id = None
checkin_responded_at = None

# ============================================================
# PATIENT CHECK-IN SOUND: 1-SECOND BEEP
# ============================================================
def play_patient_checkin_beep():
    """Play a one-second beep on the patient-side Windows computer."""
    try:
        winsound.Beep(1000, 1000)
    except (RuntimeError, OSError):
        pass

# ============================================================
# SEND GESTURE TO FLASK
# ============================================================
def send_gesture(gesture, confidence):
    global last_sent_gesture, last_sent_time
    current_time = time.time()

    if (
        gesture == last_sent_gesture
        and current_time - last_sent_time < REQUEST_COOLDOWN
        and not (gesture == "okay" and checkin_state.get("status") == "pending")
    ):
        return

    try:
        response = requests.post(
            SERVER_URL,
            json={"gesture": gesture, "confidence": round(confidence * 100, 2)},
            timeout=0.5
        )
        if response.status_code == 200:
            print(
                f"\n✓ Sent to dashboard: {gesture_names.get(gesture, gesture)} "
                f"({confidence * 100:.1f}%)"
            )
            last_sent_gesture = gesture
            last_sent_time = current_time
    except requests.exceptions.RequestException:
        pass

# ============================================================
# START WEBCAM
# ============================================================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("\nERROR: Could not open webcam.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n==============================================")
print(" SMART PATIENT GESTURE ASSISTANCE SYSTEM")
print("==============================================")
print("\nGestures:")
print("✊  CALL NURSE")
print("👍  I'M OKAY")
print("☝️  NEED FOOD / WATER")
print("🖐️  EMERGENCY")
print("\nDashboard:")
print("http://127.0.0.1:5000")
print("\nPress Q to quit.")
print("==============================================\n")

# ============================================================
# MAIN MEDIAPIPE LOOP
# ============================================================
try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)

            # Poll Flask for nurse check-ins. A new pending check-in
            # triggers exactly one one-second beep on this computer.
            if time.time() - last_checkin_poll >= CHECKIN_POLL_INTERVAL:
                last_checkin_poll = time.time()
                try:
                    response = requests.get(CHECKIN_STATUS_URL, timeout=0.2)
                    if response.ok:
                        new_state = response.json()
                        new_checkin_id = new_state.get("id")

                        if (
                            new_state.get("status") == "pending"
                            and new_checkin_id is not None
                            and new_checkin_id != last_checkin_id
                        ):
                            threading.Thread(
                                target=play_patient_checkin_beep,
                                daemon=True
                            ).start()

                        if new_checkin_id is not None:
                            last_checkin_id = new_checkin_id

                        if (
                            new_state.get("status") == "responded"
                            and checkin_state.get("status") != "responded"
                        ):
                            checkin_responded_at = time.time()

                        checkin_state = new_state
                except requests.exceptions.RequestException:
                    pass

            # Convert frame to RGB and detect hand
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            prediction_text = "NO HAND"
            confidence = 0.0
            current_prediction = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand])
                landmarks = landmarks - landmarks[0].copy()  # wrist normalization
                features = landmarks.flatten().reshape(1, -1)
                features_scaled = scaler.transform(features)

                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities))

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(prediction)

                if len(prediction_history) >= STABLE_COUNT:
                    candidate, count = Counter(prediction_history).most_common(1)[0]
                    if count >= STABLE_COUNT:
                        stable_gesture = candidate
                        current_prediction = candidate
                        prediction_text = gesture_names.get(candidate, candidate)
                    else:
                        prediction_text = "CHECKING..."
                else:
                    prediction_text = "CHECKING..."

                if current_prediction is not None:
                    send_gesture(current_prediction, confidence)

                # Draw hand landmarks and connections
                for lm in hand:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                connections = [
                    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)
                ]
                for start, end in connections:
                    x1, y1 = int(hand[start].x * frame.shape[1]), int(hand[start].y * frame.shape[0])
                    x2, y2 = int(hand[end].x * frame.shape[1]), int(hand[end].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                prediction_history.clear()
                stable_gesture = None
                prediction_text = "NO HAND DETECTED"

            # Camera status panel
            cv2.rectangle(frame, (10, 10), (670, 120), (0, 0, 0), -1)
            cv2.putText(frame, "Prediction: " + prediction_text, (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if confidence > 0:
                cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (25, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Dashboard: CONNECTED", (25, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display nurse check-in status on patient camera
            checkin_status = checkin_state.get("status")
            if checkin_status == "pending":
                cv2.rectangle(frame, (10, 130), (625, 180), (0, 90, 180), -1)
                cv2.putText(frame, "NURSE CHECK-IN: ARE YOU OKAY?", (20, 163),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif checkin_status == "responded" and checkin_responded_at is not None:
                if time.time() - checkin_responded_at < CHECKIN_RESPONSE_DISPLAY_SECONDS:
                    cv2.rectangle(frame, (10, 130), (625, 180), (30, 130, 60), -1)
                    cv2.putText(frame, "THANK YOU! RESPONSE RECEIVED", (20, 163),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Patient gesture guide panel
            panel_width = 330
            panel = np.full((frame.shape[0], panel_width, 3), (245, 245, 245), dtype=np.uint8)
            panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            panel_image = Image.fromarray(panel_rgb)
            draw = ImageDraw.Draw(panel_image)

            def load_font(size, emoji=False):
                candidates = (
                    [r"C:\Windows\Fonts\seguiemj.ttf", "seguiemj.ttf"]
                    if emoji else [r"C:\Windows\Fonts\arial.ttf", "arial.ttf"]
                )
                for font_path in candidates:
                    try:
                        return ImageFont.truetype(font_path, size)
                    except OSError:
                        continue
                return ImageFont.load_default()

            title_font = load_font(18)
            subtitle_font = load_font(12)
            label_font = load_font(16)
            meaning_font = load_font(12)
            emoji_font = load_font(35, emoji=True)

            draw.text((15, 14), "PATIENT GESTURE GUIDE", font=title_font, fill=(35, 35, 35))
            draw.text((15, 43), "Show your hand clearly to the camera", font=subtitle_font, fill=(75, 75, 75))
            draw.line((15, 67, panel_width - 15, 67), fill=(190, 190, 190), width=1)
            guide_items = [
                ("✊", "1. CALL NURSE", "Close your hand into a fist", (150, 20, 20)),
                ("👍", "2. I'M OKAY", "Show a thumbs-up", (20, 120, 20)),
                ("☝️", "3. NEED WATER", "Raise one finger", (160, 100, 20)),
                ("🖐️", "4. EMERGENCY", "Show all five fingers open", (180, 20, 20)),
            ]
            y_positions = [82, 150, 218, 286]
            for (emoji, label, meaning, label_color), y in zip(guide_items, y_positions):
                draw.text((18, y), emoji, font=emoji_font, embedded_color=True)
                draw.text((78, y + 2), label, font=label_font, fill=label_color)
                draw.text((78, y + 28), meaning, font=meaning_font, fill=(55, 55, 55))
            draw.line((15, 354, panel_width - 15, 354), fill=(190, 190, 190), width=1)
            draw.text((15, 365), "Hold the gesture steady", font=subtitle_font, fill=(45, 45, 45))
            draw.text((15, 385), "Press Q to close camera", font=subtitle_font, fill=(75, 75, 75))
            panel = cv2.cvtColor(np.array(panel_image), cv2.COLOR_RGB2BGR)
            patient_window = np.hstack((frame, panel))
            cv2.imshow("Smart Patient Gesture Assistance", patient_window)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("\nReal-time prediction stopped.")
