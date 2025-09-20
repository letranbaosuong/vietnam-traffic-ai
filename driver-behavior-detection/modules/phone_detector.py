import cv2
import mediapipe as mp
import numpy as np
import time

class PhoneDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.PHONE_POSITION_THRESH = 0.15
        self.PHONE_CONSEC_FRAMES = 10
        self.phone_counter = 0
        self.phone_detected = False

        self.phone_start_time = None
        self.total_phone_duration = 0
        self.phone_usage_count = 0

        self.hand_near_ear = False
        self.hand_in_front = False

    def detect_hand_position(self, hand_landmarks, img_shape):
        if not hand_landmarks:
            return None, None

        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        avg_y = (wrist.y + thumb_tip.y + index_tip.y + middle_tip.y + pinky_tip.y) / 5
        avg_x = (wrist.x + thumb_tip.x + index_tip.x + middle_tip.x + pinky_tip.x) / 5

        hand_height = abs(max([wrist.y, thumb_tip.y, index_tip.y, middle_tip.y, pinky_tip.y]) -
                         min([wrist.y, thumb_tip.y, index_tip.y, middle_tip.y, pinky_tip.y]))

        hand_width = abs(max([wrist.x, thumb_tip.x, index_tip.x, middle_tip.x, pinky_tip.x]) -
                        min([wrist.x, thumb_tip.x, index_tip.x, middle_tip.x, pinky_tip.x]))

        return (avg_x, avg_y), (hand_width, hand_height)

    def check_phone_usage(self, hand_position, pose_landmarks):
        if not hand_position or not pose_landmarks:
            return False, ""

        (hand_x, hand_y), (hand_w, hand_h) = hand_position

        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]

        dist_to_left_ear = np.sqrt((hand_x - left_ear.x)**2 + (hand_y - left_ear.y)**2)
        dist_to_right_ear = np.sqrt((hand_x - right_ear.x)**2 + (hand_y - right_ear.y)**2)

        if dist_to_left_ear < self.PHONE_POSITION_THRESH or dist_to_right_ear < self.PHONE_POSITION_THRESH:
            return True, "Phone Call"

        eye_level = (left_eye.y + right_eye.y) / 2
        if abs(hand_y - eye_level) < 0.2 and hand_y > eye_level:
            if hand_w / hand_h > 0.5 and hand_w / hand_h < 2:
                return True, "Texting/Browsing"

        if hand_y < nose.y and abs(hand_x - nose.x) < 0.3:
            return True, "Looking at Phone"

        return False, ""

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        status = "Normal"
        phone_usage_type = ""

        if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_position = self.detect_hand_position(hand_landmarks, frame.shape)

                is_phone_usage, usage_type = self.check_phone_usage(
                    hand_position,
                    pose_results.pose_landmarks
                )

                if is_phone_usage:
                    self.phone_counter += 1
                    if self.phone_counter >= self.PHONE_CONSEC_FRAMES:
                        status = "PHONE USAGE"
                        phone_usage_type = usage_type

                        if not self.phone_detected:
                            self.phone_detected = True
                            self.phone_start_time = time.time()
                            self.phone_usage_count += 1
                else:
                    self.phone_counter = max(0, self.phone_counter - 1)

                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

            pose_points = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_EAR,
                self.mp_pose.PoseLandmark.RIGHT_EAR,
                self.mp_pose.PoseLandmark.LEFT_EYE,
                self.mp_pose.PoseLandmark.RIGHT_EYE
            ]

            h, w = frame.shape[:2]
            for point in pose_points:
                landmark = pose_results.pose_landmarks.landmark[point]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        else:
            self.phone_counter = max(0, self.phone_counter - 1)
            if self.phone_counter == 0:
                if self.phone_detected and self.phone_start_time:
                    self.total_phone_duration += time.time() - self.phone_start_time
                    self.phone_start_time = None
                self.phone_detected = False

        return status, phone_usage_type, frame

    def get_statistics(self):
        return {
            'total_phone_usage_count': self.phone_usage_count,
            'total_phone_duration': self.total_phone_duration,
            'current_counter': self.phone_counter
        }