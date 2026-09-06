import cv2
import mediapipe as mp
import csv
import os
import time

# ==========================================
# SETTINGS
# ==========================================

DATA_FILE = "data/raw/gesture_data.csv"

SAMPLES_PER_GESTURE = 500
SAMPLE_INTERVAL = 0.15  # seconds

GESTURES = {
    "1": "call_nurse",
    "2": "okay",
    "3": "need_food_water",
    "4": "emergency"
}

# ==========================================
# CREATE DATA FOLDER
# ==========================================

os.makedirs("data/raw", exist_ok=True)

# ==========================================
# MediaPipe setup
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ==========================================
# CREATE NEW CSV
# ==========================================
# "w" means overwrite the old dataset.
# This prevents old gestures from mixing with new ones.

csv_file = open(
    DATA_FILE,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

header = []

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

header.append("gesture")

writer.writerow(header)

# ==========================================
# Webcam
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    csv_file.close()

    exit()

print("\n========================================")
print(" SMART PATIENT GESTURE DATASET")
print("========================================")
print("1 → Call Nurse ✊ (closed fist)")
print("2 → I'm Okay 👍")
print("3 → Need Water / Food ☝️")
print("4 → Emergency 🖐️ (five fingers)")
print("Q → Quit")
print("========================================")

selected_gesture = None
sample_count = 0
last_sample_time = 0

# ==========================================
# MediaPipe
# ==========================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read webcam.")

            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        # ==========================================
        # KEYBOARD INPUT
        # ==========================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        # Select gesture
        if chr(key) in GESTURES:

            selected_gesture = GESTURES[chr(key)]

            sample_count = 0

            last_sample_time = 0

            print(
                f"\nCollecting: {selected_gesture}"
            )

        # ==========================================
        # HAND DETECTED
        # ==========================================

        if result.hand_landmarks and selected_gesture:

            hand = result.hand_landmarks[0]

            current_time = time.time()

            # ==========================================
            # COLLECT ONLY UP TO 500 SAMPLES
            # ==========================================

            if (
                sample_count < SAMPLES_PER_GESTURE
                and
                current_time - last_sample_time
                >= SAMPLE_INTERVAL
            ):

                features = []

                for landmark in hand:

                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                features.append(selected_gesture)

                writer.writerow(features)

                csv_file.flush()

                sample_count += 1

                last_sample_time = current_time

                print(
                    f"\r{selected_gesture}: "
                    f"{sample_count}/{SAMPLES_PER_GESTURE}",
                    end=""
                )

            # ==========================================
            # DRAW LANDMARKS
            # ==========================================

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # ==========================================
        # DISPLAY INFORMATION
        # ==========================================

        if selected_gesture:

            cv2.putText(
                frame,
                f"Gesture: {selected_gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {sample_count}/{SAMPLES_PER_GESTURE}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if sample_count < SAMPLES_PER_GESTURE:

                cv2.putText(
                    frame,
                    "Move hand slowly",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "DONE - Press another number",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "Press 1-4 to select gesture",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Gesture Dataset Collection",
            frame
        )

# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

csv_file.close()

print("\n\n========================================")
print(" DATASET COLLECTION STOPPED")
print("========================================")
print(f"Dataset saved at: {DATA_FILE}")
print("========================================")