import cv2
import mediapipe as mp
import numpy as np
import time

class SmokingDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.SMOKING_POSITION_THRESH = 0.1
        self.SMOKING_CONSEC_FRAMES = 15
        self.smoking_counter = 0
        self.smoking_detected = False

        self.smoking_start_time = None
        self.total_smoking_duration = 0
        self.smoking_count = 0

        self.MOUTH_LANDMARKS = [13, 14, 78, 308]
        self.LIPS_LANDMARKS = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]

    def detect_hand_near_mouth(self, hand_landmarks, face_landmarks, img_shape):
        if not hand_landmarks or not face_landmarks:
            return False, None

        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        mouth_points = []
        for idx in self.MOUTH_LANDMARKS:
            mouth_point = face_landmarks.landmark[idx]
            mouth_points.append([mouth_point.x, mouth_point.y])

        mouth_center = np.mean(mouth_points, axis=0)

        fingers = [
            [index_finger_tip.x, index_finger_tip.y],
            [middle_finger_tip.x, middle_finger_tip.y],
            [thumb_tip.x, thumb_tip.y]
        ]

        min_distance = float('inf')
        closest_finger = None

        for i, finger in enumerate(fingers):
            distance = np.sqrt((finger[0] - mouth_center[0])**2 +
                             (finger[1] - mouth_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_finger = i

        if min_distance < self.SMOKING_POSITION_THRESH:
            return True, closest_finger

        return False, None

    def detect_cigarette_gesture(self, hand_landmarks):
        if not hand_landmarks:
            return False

        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y

        fingers_distance = np.sqrt((index_tip.x - middle_tip.x)**2 +
                                  (index_tip.y - middle_tip.y)**2)

        if (index_extended and middle_extended and fingers_distance < 0.05):
            return True

        return False

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        status = "Normal"
        smoking_type = ""

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            for hand_landmarks in hand_results.multi_hand_landmarks:
                near_mouth, finger_idx = self.detect_hand_near_mouth(
                    hand_landmarks,
                    face_landmarks,
                    frame.shape
                )

                cigarette_gesture = self.detect_cigarette_gesture(hand_landmarks)

                if near_mouth and cigarette_gesture:
                    self.smoking_counter += 1

                    if self.smoking_counter >= self.SMOKING_CONSEC_FRAMES:
                        status = "SMOKING DETECTED"
                        smoking_type = "Cigarette Smoking"

                        if not self.smoking_detected:
                            self.smoking_detected = True
                            self.smoking_start_time = time.time()
                            self.smoking_count += 1
                else:
                    self.smoking_counter = max(0, self.smoking_counter - 1)

                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            h, w = frame.shape[:2]
            for idx in self.MOUTH_LANDMARKS:
                mouth_point = face_landmarks.landmark[idx]
                x = int(mouth_point.x * w)
                y = int(mouth_point.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            mouth_points = []
            for idx in self.LIPS_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                mouth_points.append([int(landmark.x * w), int(landmark.y * h)])
            mouth_points = np.array(mouth_points, dtype=np.int32)
            cv2.polylines(frame, [mouth_points], True, (0, 255, 255), 1)

        else:
            self.smoking_counter = max(0, self.smoking_counter - 1)
            if self.smoking_counter == 0:
                if self.smoking_detected and self.smoking_start_time:
                    self.total_smoking_duration += time.time() - self.smoking_start_time
                    self.smoking_start_time = None
                self.smoking_detected = False

        return status, smoking_type, frame

    def get_statistics(self):
        return {
            'total_smoking_count': self.smoking_count,
            'total_smoking_duration': self.total_smoking_duration,
            'current_counter': self.smoking_counter
        }