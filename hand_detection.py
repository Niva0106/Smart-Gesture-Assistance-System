import cv2
import mediapipe as mp

# Create MediaPipe Hand Landmarker
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

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam started. Press Q to quit.")

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read frame")
            break

        # Flip camera for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert OpenCV frame to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        # Draw landmarks
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                for landmark in hand:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                connections = [
                    (0,1), (1,2), (2,3), (3,4),
                    (0,5), (5,6), (6,7), (7,8),
                    (5,9), (9,10), (10,11), (11,12),
                    (9,13), (13,14), (14,15), (15,16),
                    (13,17), (17,18), (18,19), (19,20),
                    (0,17)
                ]

                for start, end in connections:

                    x1 = int(hand[start].x * frame.shape[1])
                    y1 = int(hand[start].y * frame.shape[0])

                    x2 = int(hand[end].x * frame.shape[1])
                    y2 = int(hand[end].y * frame.shape[0])

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        cv2.putText(
            frame,
            "Hand Detection - Press Q to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Smart Patient - Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()