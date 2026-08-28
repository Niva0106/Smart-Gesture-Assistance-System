import cv2
import mediapipe as mp
import joblib
import numpy as np
import requests
import time
from collections import Counter, deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
LANDMARKER_PATH = "hand_landmarker.task"

SERVER_URL = "http://127.0.0.1:5000/api/gesture"

CONFIDENCE_THRESHOLD = 0.65

# Number of recent predictions used for stability
HISTORY_SIZE = 7

# Minimum number of matching predictions
STABLE_COUNT = 4

# Prevent sending the same request repeatedly
REQUEST_COOLDOWN = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ML model...")

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

print("✓ Random Forest model loaded")
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

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARKER_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=HISTORY_SIZE
)


# ============================================================
# VARIABLES
# ============================================================

last_sent_gesture = None

last_sent_time = 0

stable_gesture = None


# ============================================================
# SEND GESTURE TO FLASK
# ============================================================

def send_gesture(gesture, confidence):

    global last_sent_gesture
    global last_sent_time

    current_time = time.time()

    # Prevent duplicate requests
    if (
        gesture == last_sent_gesture
        and
        current_time - last_sent_time < REQUEST_COOLDOWN
    ):
        return


    try:

        response = requests.post(
            SERVER_URL,
            json={
                "gesture": gesture,
                "confidence": round(
                    confidence * 100,
                    2
                )
            },
            timeout=2
        )


        if response.status_code == 200:

            print(
                f"\n✓ Sent to dashboard:"
                f" {gesture_names.get(gesture, gesture)}"
                f" ({confidence * 100:.1f}%)"
            )

            last_sent_gesture = gesture

            last_sent_time = current_time

        else:

            print(
                "Dashboard error:",
                response.status_code
            )


    except requests.exceptions.RequestException:

        print(
            "\n⚠ Could not connect to Flask server."
        )

        print(
            "Make sure app.py is running."
        )


# ============================================================
# START WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("\nERROR: Could not open webcam.")

    exit()


print("\n==============================================")
print(" SMART PATIENT GESTURE ASSISTANCE SYSTEM")
print("==============================================")

print("\nGestures:")

print("✋  CALL NURSE")
print("👍  I'M OKAY")
print("☝️  NEED FOOD / WATER")
print("✌️  EMERGENCY")

print("\nDashboard:")
print("http://127.0.0.1:5000")

print("\nPress Q to quit.")

print("==============================================\n")


# ============================================================
# MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        success, frame = camera.read()


        if not success:

            print("Could not read webcam.")

            break


        # Mirror camera

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # CONVERT TO RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )


        prediction_text = "NO HAND"

        confidence = 0

        current_prediction = None


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # EXTRACT 21 LANDMARKS
            # ------------------------------------------------

            landmarks = []


            for landmark in hand:

                landmarks.append([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])


            landmarks = np.array(
                landmarks
            )


            # ------------------------------------------------
            # WRIST NORMALIZATION
            # ------------------------------------------------

            wrist = landmarks[0].copy()

            landmarks = landmarks - wrist


            # ------------------------------------------------
            # 21 × 3 = 63 FEATURES
            # ------------------------------------------------

            features = landmarks.flatten()

            features = features.reshape(
                1,
                -1
            )


            # ------------------------------------------------
            # SCALE FEATURES
            # ------------------------------------------------

            features_scaled = scaler.transform(
                features
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                features_scaled
            )[0]


            probabilities = model.predict_proba(
                features_scaled
            )[0]


            confidence = float(
                np.max(probabilities)
            )


            # ------------------------------------------------
            # STABILITY FILTER
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    prediction
                )


            if len(prediction_history) >= 3:

                most_common = Counter(
                    prediction_history
                ).most_common(1)[0]


                candidate = most_common[0]

                count = most_common[1]


                if count >= STABLE_COUNT:

                    stable_gesture = candidate

                    current_prediction = candidate

                    prediction_text = gesture_names.get(
                        candidate,
                        candidate
                    )

                else:

                    prediction_text = "CHECKING..."


            else:

                prediction_text = "CHECKING..."


            # ------------------------------------------------
            # SEND STABLE GESTURE
            # ------------------------------------------------

            if current_prediction is not None:

                send_gesture(
                    current_prediction,
                    confidence
                )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for landmark in hand:

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )


                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # HAND CONNECTIONS
            # ------------------------------------------------

            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (5, 9),
                (9, 13),
                (13, 17)
            ]


            for start, end in connections:

                x1 = int(
                    hand[start].x *
                    frame.shape[1]
                )

                y1 = int(
                    hand[start].y *
                    frame.shape[0]
                )

                x2 = int(
                    hand[end].x *
                    frame.shape[1]
                )

                y2 = int(
                    hand[end].y *
                    frame.shape[0]
                )


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


        else:

            # No hand detected

            prediction_history.clear()

            stable_gesture = None

            prediction_text = "NO HAND DETECTED"


        # ====================================================
        # DISPLAY BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (670, 120),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            "Prediction: " + prediction_text,
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        if confidence > 0:

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        cv2.putText(
            frame,
            "Dashboard: CONNECTED",
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


        # ====================================================
        # QUIT
        # ====================================================

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                20,
                frame.shape[0] - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Smart Patient Gesture Assistance",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nReal-time prediction stopped.")