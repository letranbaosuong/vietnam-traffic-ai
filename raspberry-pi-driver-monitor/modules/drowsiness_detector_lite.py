import cv2
import numpy as np
import time

class DrowsinessDetectorLite:
    def __init__(self):
        """
        Lightweight drowsiness detector using simple computer vision
        No heavy ML models - optimized for Raspberry Pi
        """
        # Eye cascade for detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # Parameters
        self.EYE_CLOSED_THRESH = 5  # Number of frames eyes must be closed
        self.eye_closed_counter = 0
        self.drowsy_detected = False

        # Blink detection
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.blinks_per_minute = 0

        # Eye aspect ratio (simplified)
        self.MIN_EYE_HEIGHT = 10  # Minimum eye height in pixels
        self.eye_heights = []

        # Statistics
        self.drowsy_count = 0
        self.total_drowsy_duration = 0
        self.drowsy_start_time = None

    def detect_eyes(self, face_roi):
        """Detect eyes in face region"""
        if face_roi is None or face_roi.size == 0:
            return []

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        return eyes

    def calculate_eye_openness(self, eye_roi):
        """Calculate if eye is open using simple edge detection"""
        if eye_roi is None or eye_roi.size == 0:
            return 0

        # Convert to grayscale
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Count edge pixels (more edges = eye open)
        edge_density = np.sum(edges > 0) / edges.size

        return edge_density

    def detect_drowsiness(self, face_roi):
        """Main drowsiness detection function"""
        status = "Normal"
        eye_status = "Unknown"

        if face_roi is None or face_roi.size == 0:
            return status, eye_status

        # Detect eyes
        eyes = self.detect_eyes(face_roi)

        # Check if eyes are detected
        if len(eyes) >= 2:
            # Eyes detected - likely open
            eye_status = "Eyes Open"

            # Calculate average eye height
            avg_eye_height = np.mean([h for (x, y, w, h) in eyes])
            self.eye_heights.append(avg_eye_height)

            # Keep only recent measurements
            if len(self.eye_heights) > 30:
                self.eye_heights.pop(0)

            # Check for significant height reduction (drowsiness indicator)
            if len(self.eye_heights) > 10:
                recent_avg = np.mean(self.eye_heights[-5:])
                baseline_avg = np.mean(self.eye_heights[:-5])

                if recent_avg < baseline_avg * 0.7:  # 30% reduction
                    self.eye_closed_counter += 1
                else:
                    self.eye_closed_counter = max(0, self.eye_closed_counter - 1)

            # Reset closed counter
            if self.eye_closed_counter > 0:
                self.eye_closed_counter = max(0, self.eye_closed_counter - 1)

        elif len(eyes) == 1:
            # Only one eye detected - possible blink or partial closure
            eye_status = "One Eye Detected"
            self.eye_closed_counter += 0.5  # Half increment

        else:
            # No eyes detected - possibly closed
            eye_status = "Eyes Closed"
            self.eye_closed_counter += 1

        # Check for drowsiness
        if self.eye_closed_counter >= self.EYE_CLOSED_THRESH:
            status = "DROWSY"
            eye_status = "Eyes Closed Too Long"

            if not self.drowsy_detected:
                self.drowsy_detected = True
                self.drowsy_start_time = time.time()
                self.drowsy_count += 1

        else:
            if self.drowsy_detected and self.drowsy_start_time:
                self.total_drowsy_duration += time.time() - self.drowsy_start_time
                self.drowsy_start_time = None
            self.drowsy_detected = False

        return status, eye_status

    def draw_eyes(self, frame, face_roi, face_offset):
        """Draw eye detection results"""
        if face_roi is None or face_roi.size == 0:
            return frame

        eyes = self.detect_eyes(face_roi)
        x_offset, y_offset = face_offset

        for (ex, ey, ew, eh) in eyes:
            # Draw eye rectangle
            cv2.rectangle(frame,
                         (x_offset + ex, y_offset + ey),
                         (x_offset + ex + ew, y_offset + ey + eh),
                         (0, 255, 0), 2)

        return frame

    def calculate_blink_rate(self):
        """Calculate blinks per minute"""
        current_time = time.time()
        time_diff = current_time - self.last_blink_time

        if time_diff >= 60:  # Calculate every minute
            self.blinks_per_minute = self.blink_counter
            self.blink_counter = 0
            self.last_blink_time = current_time

        # High blink rate can indicate fatigue
        if self.blinks_per_minute > 20:
            return "High Blink Rate"
        elif self.blinks_per_minute < 10:
            return "Low Blink Rate"
        else:
            return "Normal Blink Rate"

    def get_statistics(self):
        """Get detection statistics"""
        return {
            'drowsy_count': self.drowsy_count,
            'total_drowsy_duration': self.total_drowsy_duration,
            'eye_closed_counter': self.eye_closed_counter,
            'blinks_per_minute': self.blinks_per_minute,
            'current_status': 'Drowsy' if self.drowsy_detected else 'Alert'
        }

    def reset(self):
        """Reset detector state"""
        self.eye_closed_counter = 0
        self.drowsy_detected = False
        self.eye_heights = []
        self.blink_counter = 0