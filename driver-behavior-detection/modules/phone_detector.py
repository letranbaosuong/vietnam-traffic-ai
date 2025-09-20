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
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.5
        )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Improved thresholds based on research
        self.PHONE_POSITION_THRESH = 0.12  # Tighter threshold for ear distance
        self.PHONE_CONSEC_FRAMES = 20  # Increased to reduce false positives
        self.phone_counter = 0
        self.phone_detected = False

        self.phone_start_time = None
        self.total_phone_duration = 0
        self.phone_usage_count = 0

        # Track hand positions over time
        self.hand_history = []
        self.MAX_HISTORY = 30

        # Phone usage patterns
        self.phone_call_pattern = False
        self.texting_pattern = False

    def detect_hand_position(self, hand_landmarks, img_shape):
        if not hand_landmarks:
            return None, None, None

        # Get all key landmarks
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # Calculate center of palm (more stable than average of fingertips)
        palm_center_x = (index_mcp.x + middle_mcp.x + pinky_mcp.x + wrist.x) / 4
        palm_center_y = (index_mcp.y + middle_mcp.y + pinky_mcp.y + wrist.y) / 4

        # Calculate hand dimensions
        hand_points_x = [wrist.x, thumb_tip.x, index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x]
        hand_points_y = [wrist.y, thumb_tip.y, index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y]

        hand_width = max(hand_points_x) - min(hand_points_x)
        hand_height = max(hand_points_y) - min(hand_points_y)

        # Detect hand orientation (important for phone holding detection)
        hand_angle = np.arctan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)

        return (palm_center_x, palm_center_y), (hand_width, hand_height), hand_angle

    def check_phone_usage(self, hand_position, pose_landmarks, face_landmarks=None):
        if not hand_position or not pose_landmarks:
            return False, ""

        (hand_x, hand_y), (hand_w, hand_h), hand_angle = hand_position

        # Get face landmarks
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate distances
        dist_to_left_ear = np.sqrt((hand_x - left_ear.x)**2 + (hand_y - left_ear.y)**2)
        dist_to_right_ear = np.sqrt((hand_x - right_ear.x)**2 + (hand_y - right_ear.y)**2)

        # Phone Call Detection - hand near ear with proper angle
        if dist_to_left_ear < self.PHONE_POSITION_THRESH or dist_to_right_ear < self.PHONE_POSITION_THRESH:
            # Check if hand is raised (above shoulder level)
            shoulder_level = (left_shoulder.y + right_shoulder.y) / 2
            if hand_y < shoulder_level:
                # Check hand angle for phone holding position
                if abs(hand_angle) > 0.5:  # Hand tilted (typical phone holding angle)
                    return True, "Phone Call"

        # Texting/Browsing Detection - hand in front at chest/stomach level
        eye_level = (left_eye.y + right_eye.y) / 2
        shoulder_level = (left_shoulder.y + right_shoulder.y) / 2

        # Check if hand is below eye level but above waist (texting position)
        if eye_level < hand_y < shoulder_level + 0.2:
            # Check if hand is in front of body
            face_center_x = (left_eye.x + right_eye.x) / 2
            if abs(hand_x - face_center_x) < 0.25:  # Hand roughly centered
                # Check hand size ratio (phone-like shape)
                if 0.4 < hand_w / hand_h < 1.8:  # Reasonable aspect ratio for holding phone
                    # Store position for pattern analysis
                    self.hand_history.append((hand_x, hand_y, time.time()))
                    if len(self.hand_history) > self.MAX_HISTORY:
                        self.hand_history.pop(0)

                    # Check for stable hand position (texting requires steady hand)
                    if len(self.hand_history) >= 10:
                        recent_positions = self.hand_history[-10:]
                        x_variance = np.var([p[0] for p in recent_positions])
                        y_variance = np.var([p[1] for p in recent_positions])

                        # Low variance means stable position (texting/browsing)
                        if x_variance < 0.01 and y_variance < 0.01:
                            return True, "Texting/Browsing"

        # Looking at Phone Detection - hand raised near face
        if hand_y < nose.y + 0.1:  # Hand at face level or higher
            face_center_x = (left_eye.x + right_eye.x) / 2
            # Check if hand is in front of face
            if abs(hand_x - face_center_x) < 0.2:
                # Check if hand is at reading distance
                dist_to_nose = np.sqrt((hand_x - nose.x)**2 + (hand_y - nose.y)**2)
                if 0.15 < dist_to_nose < 0.35:  # Typical phone viewing distance
                    return True, "Looking at Phone"

        return False, ""

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process all detections
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        status = "Normal"
        phone_usage_type = ""

        if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
            # Analyze each detected hand
            phone_detected_in_frame = False
            detected_usage_type = ""

            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_position = self.detect_hand_position(hand_landmarks, frame.shape)

                is_phone_usage, usage_type = self.check_phone_usage(
                    hand_position,
                    pose_results.pose_landmarks,
                    face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
                )

                if is_phone_usage:
                    phone_detected_in_frame = True
                    detected_usage_type = usage_type
                    break  # One hand with phone is enough

            # Update counter based on detection
            if phone_detected_in_frame:
                self.phone_counter += 1
                if self.phone_counter >= self.PHONE_CONSEC_FRAMES:
                    status = "PHONE USAGE"
                    phone_usage_type = detected_usage_type

                    if not self.phone_detected:
                        self.phone_detected = True
                        self.phone_start_time = time.time()
                        self.phone_usage_count += 1
            else:
                # Gradual decrease to handle brief occlusions
                self.phone_counter = max(0, self.phone_counter - 2)

            # Draw hand landmarks
            for hand_landmarks in hand_results.multi_hand_landmarks:
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
            # No hands or pose detected
            self.phone_counter = max(0, self.phone_counter - 2)
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