import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 15
        self.COUNTER = 0
        self.drowsy_detected = False

        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

        self.drowsy_start_time = None
        self.total_drowsy_duration = 0
        self.drowsy_count = 0

    def calculate_ear(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])

        ear = (A + B) / (2.0 * C) if C != 0 else 0
        return ear

    def get_eye_landmarks(self, landmarks):
        left_eye = []
        right_eye = []

        for idx in self.LEFT_EYE_POINTS:
            landmark = landmarks.landmark[idx]
            left_eye.append([landmark.x, landmark.y])

        for idx in self.RIGHT_EYE_POINTS:
            landmark = landmarks.landmark[idx]
            right_eye.append([landmark.x, landmark.y])

        return np.array(left_eye), np.array(right_eye)

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        status = "Normal"
        ear_value = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            left_eye, right_eye = self.get_eye_landmarks(landmarks)

            h, w = frame.shape[:2]
            left_eye_pixels = (left_eye * [w, h]).astype(int)
            right_eye_pixels = (right_eye * [w, h]).astype(int)

            left_ear = self.calculate_ear(left_eye_pixels)
            right_ear = self.calculate_ear(right_eye_pixels)

            ear_value = (left_ear + right_ear) / 2.0

            if ear_value < self.EYE_AR_THRESH:
                self.COUNTER += 1

                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.drowsy_detected:
                        self.drowsy_detected = True
                        self.drowsy_start_time = time.time()
                        self.drowsy_count += 1
                    status = "DROWSY"
                else:
                    status = "Eyes Closing"
            else:
                if self.drowsy_detected and self.drowsy_start_time:
                    self.total_drowsy_duration += time.time() - self.drowsy_start_time
                    self.drowsy_start_time = None

                self.COUNTER = 0
                self.drowsy_detected = False
                status = "Normal"

            left_eye_hull = cv2.convexHull(left_eye_pixels)
            right_eye_hull = cv2.convexHull(right_eye_pixels)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            for idx in self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        return status, ear_value, frame

    def get_statistics(self):
        return {
            'total_drowsy_count': self.drowsy_count,
            'total_drowsy_duration': self.total_drowsy_duration,
            'current_counter': self.COUNTER
        }