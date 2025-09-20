import cv2
import numpy as np

class HeadPoseLite:
    def __init__(self):
        """
        Lightweight head pose estimation using facial landmarks
        Optimized for Raspberry Pi without heavy ML models
        """
        # Load face landmark detector (68 points)
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Thresholds
        self.YAW_THRESHOLD = 0.3  # Left/Right threshold
        self.PITCH_THRESHOLD = 0.3  # Up/Down threshold
        self.DISTRACTION_FRAMES = 10

        # Counters
        self.left_counter = 0
        self.right_counter = 0
        self.up_counter = 0
        self.down_counter = 0

        # Status
        self.distracted = False
        self.distraction_type = ""

    def estimate_head_pose(self, face_roi):
        """Estimate head pose using face symmetry and position"""
        if face_roi is None or face_roi.size == 0:
            return 0, 0, "Unknown"

        h, w = face_roi.shape[:2]

        # Simple pose estimation based on face position
        # Divide face into regions
        left_region = face_roi[:, :w//3]
        center_region = face_roi[:, w//3:2*w//3]
        right_region = face_roi[:, 2*w//3:]

        # Calculate intensity of each region
        left_intensity = np.mean(cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY))
        center_intensity = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
        right_intensity = np.mean(cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY))

        # Estimate yaw (left/right)
        yaw_ratio = (left_intensity - right_intensity) / (center_intensity + 1)

        # Divide face horizontally for pitch
        upper_region = face_roi[:h//3, :]
        middle_region = face_roi[h//3:2*h//3, :]
        lower_region = face_roi[2*h//3:, :]

        upper_intensity = np.mean(cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY))
        middle_intensity = np.mean(cv2.cvtColor(middle_region, cv2.COLOR_BGR2GRAY))
        lower_intensity = np.mean(cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY))

        # Estimate pitch (up/down)
        pitch_ratio = (upper_intensity - lower_intensity) / (middle_intensity + 1)

        # Determine direction
        direction = "Forward"

        if abs(yaw_ratio) > self.YAW_THRESHOLD:
            if yaw_ratio > 0:
                direction = "Looking Left"
                self.left_counter += 1
                self.right_counter = max(0, self.right_counter - 1)
            else:
                direction = "Looking Right"
                self.right_counter += 1
                self.left_counter = max(0, self.left_counter - 1)
        else:
            self.left_counter = max(0, self.left_counter - 1)
            self.right_counter = max(0, self.right_counter - 1)

        if abs(pitch_ratio) > self.PITCH_THRESHOLD:
            if pitch_ratio > 0:
                direction = "Looking Up" if direction == "Forward" else direction + " & Up"
                self.up_counter += 1
                self.down_counter = max(0, self.down_counter - 1)
            else:
                direction = "Looking Down" if direction == "Forward" else direction + " & Down"
                self.down_counter += 1
                self.up_counter = max(0, self.up_counter - 1)
        else:
            self.up_counter = max(0, self.up_counter - 1)
            self.down_counter = max(0, self.down_counter - 1)

        return yaw_ratio, pitch_ratio, direction

    def detect_distraction(self):
        """Check if driver is distracted based on head pose"""
        status = "Normal"
        distraction_detail = ""

        # Check for sustained distraction
        if self.left_counter >= self.DISTRACTION_FRAMES:
            status = "DISTRACTED"
            distraction_detail = "Nhìn TRÁI quá lâu"
            self.distracted = True
        elif self.right_counter >= self.DISTRACTION_FRAMES:
            status = "DISTRACTED"
            distraction_detail = "Nhìn PHẢI quá lâu"
            self.distracted = True
        elif self.up_counter >= self.DISTRACTION_FRAMES:
            status = "DISTRACTED"
            distraction_detail = "Nhìn LÊN quá lâu"
            self.distracted = True
        elif self.down_counter >= self.DISTRACTION_FRAMES:
            status = "DISTRACTED"
            distraction_detail = "Nhìn XUỐNG quá lâu"
            self.distracted = True
        else:
            self.distracted = False

        self.distraction_type = distraction_detail
        return status, distraction_detail

    def draw_indicators(self, frame, face_bbox, yaw, pitch, direction):
        """Draw visual indicators for head pose"""
        if face_bbox is None:
            return frame

        x, y, w, h = face_bbox

        # Draw arrow indicating head direction
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate arrow endpoint
        arrow_length = 50
        end_x = int(center_x + arrow_length * yaw * 3)
        end_y = int(center_y - arrow_length * pitch * 3)

        # Color based on distraction
        color = (0, 0, 255) if self.distracted else (0, 255, 0)

        # Draw arrow
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 3)

        # Draw direction text
        cv2.putText(frame, direction, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw counters if in debug mode
        debug_text = f"L:{self.left_counter} R:{self.right_counter} U:{self.up_counter} D:{self.down_counter}"
        cv2.putText(frame, debug_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def get_statistics(self):
        """Get head pose statistics"""
        return {
            'distracted': self.distracted,
            'distraction_type': self.distraction_type,
            'left_counter': self.left_counter,
            'right_counter': self.right_counter,
            'up_counter': self.up_counter,
            'down_counter': self.down_counter
        }

    def reset(self):
        """Reset counters"""
        self.left_counter = 0
        self.right_counter = 0
        self.up_counter = 0
        self.down_counter = 0
        self.distracted = False
        self.distraction_type = ""