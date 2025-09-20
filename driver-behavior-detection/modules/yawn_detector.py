import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time

class YawnDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Increased threshold - mouth needs to open wider to be considered yawning
        self.MOUTH_AR_THRESH = 0.8  # Changed from 0.5 to 0.8
        # Increased frames - mouth needs to stay open longer (about 1-1.5 seconds at 20-30 fps)
        self.MOUTH_AR_CONSEC_FRAMES = 30  # Changed from 15 to 30
        self.COUNTER = 0
        self.yawn_detected = False

        # Additional parameters to reduce false positives
        self.mar_history = []  # Store MAR values to check for pattern
        self.MAX_HISTORY = 60  # Store last 2-3 seconds of data

        self.MOUTH_POINTS = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            324, 318, 402, 317, 14, 87, 178, 88, 95
        ]

        self.UPPER_LIP = [61, 84, 17, 314, 405, 320, 307, 308, 324, 318]
        self.LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

        self.yawn_start_time = None
        self.total_yawn_duration = 0
        self.yawn_count = 0

    def calculate_mar(self, mouth_points):
        # Calculate vertical distances (mouth opening)
        # Upper center (13) to lower center (14)
        vertical_1 = dist.euclidean(mouth_points[2], mouth_points[6])  # center vertical
        # Additional vertical measurements for accuracy
        vertical_2 = dist.euclidean(mouth_points[1], mouth_points[7])  # left vertical
        vertical_3 = dist.euclidean(mouth_points[3], mouth_points[5])  # right vertical

        # Calculate horizontal distance (mouth width)
        # Left corner (78) to right corner (308)
        horizontal = dist.euclidean(mouth_points[0], mouth_points[4])

        # MAR = average vertical distance / horizontal distance
        avg_vertical = (vertical_1 + vertical_2 + vertical_3) / 3.0
        mar = avg_vertical / horizontal if horizontal != 0 else 0
        return mar

    def get_mouth_landmarks(self, landmarks):
        # Using more accurate mouth landmarks for better detection
        # Upper lip: 13 (top center)
        # Lower lip: 14 (bottom center)
        # Left corner: 78
        # Right corner: 308
        # Additional points for better calculation
        mouth = []
        # Order: left_corner, upper_left, upper_center, upper_right, right_corner,
        #        lower_right, lower_center, lower_left, close_loop
        for idx in [78, 81, 13, 311, 308, 415, 14, 87, 78]:
            landmark = landmarks.landmark[idx]
            mouth.append([landmark.x, landmark.y])
        return np.array(mouth)

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        status = "Normal"
        mar_value = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            mouth = self.get_mouth_landmarks(landmarks)

            h, w = frame.shape[:2]
            mouth_pixels = (mouth * [w, h]).astype(int)

            mar_value = self.calculate_mar(mouth_pixels)

            # Store MAR history for pattern analysis
            self.mar_history.append(mar_value)
            if len(self.mar_history) > self.MAX_HISTORY:
                self.mar_history.pop(0)

            # Check if this is likely a yawn (gradual opening and closing)
            # Yawns typically have a smooth increase then decrease pattern
            is_yawn_pattern = False
            if len(self.mar_history) >= 10:
                recent_values = self.mar_history[-10:]
                # Check for smooth increase (not rapid changes like talking)
                differences = [recent_values[i+1] - recent_values[i] for i in range(len(recent_values)-1)]
                avg_change = sum(abs(d) for d in differences) / len(differences)
                # Yawns have smoother changes, talking has more rapid variations
                is_yawn_pattern = avg_change < 0.05  # Smooth change threshold

            if mar_value > self.MOUTH_AR_THRESH and is_yawn_pattern:
                self.COUNTER += 1

                if self.COUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                    if not self.yawn_detected:
                        self.yawn_detected = True
                        self.yawn_start_time = time.time()
                        self.yawn_count += 1
                    status = "YAWNING"
                else:
                    status = "Mouth Opening"
            else:
                if self.yawn_detected and self.yawn_start_time:
                    self.total_yawn_duration += time.time() - self.yawn_start_time
                    self.yawn_start_time = None

                self.COUNTER = 0
                self.yawn_detected = False
                status = "Normal"

            mouth_hull = cv2.convexHull(mouth_pixels)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)

            for point in mouth_pixels:
                cv2.circle(frame, tuple(point), 2, (255, 255, 0), -1)

        return status, mar_value, frame

    def get_statistics(self):
        return {
            'total_yawn_count': self.yawn_count,
            'total_yawn_duration': self.total_yawn_duration,
            'current_counter': self.COUNTER
        }