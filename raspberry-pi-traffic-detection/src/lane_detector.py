#!/usr/bin/env python3
"""
Lane Detection cho Raspberry Pi 4
Sử dụng OpenCV và Computer Vision techniques
"""

import cv2
import numpy as np
import math

class LaneDetector:
    def __init__(self):
        """Initialize lane detector with optimized parameters for Pi 4"""
        # Canny edge detection parameters
        self.canny_low = 50
        self.canny_high = 150

        # Hough line parameters
        self.hough_rho = 2
        self.hough_theta = np.pi / 180
        self.hough_threshold = 100
        self.hough_min_line_length = 40
        self.hough_max_line_gap = 5

        # ROI parameters (Region of Interest)
        self.roi_bottom_width = 0.85
        self.roi_top_width = 0.08
        self.roi_height = 0.62

        # Lane line filtering
        self.slope_threshold = 0.5
        self.lane_history = []
        self.history_length = 5

    def create_roi_mask(self, image):
        """
        Tạo ROI mask cho lane detection
        Focus vào khu vực đường phía trước
        """
        height, width = image.shape[:2]

        # Define ROI vertices
        bottom_left = [int(width * (1 - self.roi_bottom_width) / 2), height]
        bottom_right = [int(width * (1 + self.roi_bottom_width) / 2), height]
        top_left = [int(width * (1 - self.roi_top_width) / 2), int(height * (1 - self.roi_height))]
        top_right = [int(width * (1 + self.roi_top_width) / 2), int(height * (1 - self.roi_height))]

        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        # Create mask with same dimensions as image
        if len(image.shape) == 3:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)

        return mask

    def detect_edges(self, image):
        """
        Detect edges using Canny edge detector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        return edges

    def detect_lines(self, edges):
        """
        Detect lines using Hough Line Transform
        """
        lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        return lines

    def classify_lines(self, lines, image_width):
        """
        Phân loại lines thành left lane và right lane
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Vertical line
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter based on slope threshold
            if abs(slope) < self.slope_threshold:
                continue

            # Classify based on slope and position
            if slope < 0 and x1 < image_width / 2:  # Left lane
                left_lines.append(line[0])
            elif slope > 0 and x1 > image_width / 2:  # Right lane
                right_lines.append(line[0])

        return left_lines, right_lines

    def average_lines(self, lines):
        """
        Average multiple lines into single line
        """
        if not lines:
            return None

        lines = np.array(lines)

        # Calculate average
        x1_avg = np.mean(lines[:, 0])
        y1_avg = np.mean(lines[:, 1])
        x2_avg = np.mean(lines[:, 2])
        y2_avg = np.mean(lines[:, 3])

        return [int(x1_avg), int(y1_avg), int(x2_avg), int(y2_avg)]

    def extrapolate_line(self, line, y_min, y_max):
        """
        Extrapolate line to full lane length
        """
        if line is None:
            return None

        x1, y1, x2, y2 = line

        # Calculate slope and intercept
        if x2 - x1 == 0:
            return None

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Calculate extrapolated points
        x_min = int((y_min - intercept) / slope)
        x_max = int((y_max - intercept) / slope)

        return [x_min, y_min, x_max, y_max]

    def smooth_lanes(self, left_lane, right_lane):
        """
        Smooth lane detection using history
        """
        current_lanes = {'left': left_lane, 'right': right_lane}

        # Add to history
        self.lane_history.append(current_lanes)

        # Keep only recent history
        if len(self.lane_history) > self.history_length:
            self.lane_history.pop(0)

        # Calculate smoothed lanes
        smoothed_left = None
        smoothed_right = None

        # Average left lanes
        left_lanes = [frame['left'] for frame in self.lane_history if frame['left'] is not None]
        if left_lanes:
            left_array = np.array(left_lanes)
            smoothed_left = np.mean(left_array, axis=0).astype(int).tolist()

        # Average right lanes
        right_lanes = [frame['right'] for frame in self.lane_history if frame['right'] is not None]
        if right_lanes:
            right_array = np.array(right_lanes)
            smoothed_right = np.mean(right_array, axis=0).astype(int).tolist()

        return smoothed_left, smoothed_right

    def calculate_lane_curvature(self, left_lane, right_lane, image_height):
        """
        Tính độ cong của làn đường
        """
        if left_lane is None or right_lane is None:
            return None, "Straight"

        # Calculate midpoint at bottom and top
        left_bottom = (left_lane[0], left_lane[1])
        left_top = (left_lane[2], left_lane[3])
        right_bottom = (right_lane[0], right_lane[1])
        right_top = (right_lane[2], right_lane[3])

        # Lane center points
        center_bottom = ((left_bottom[0] + right_bottom[0]) // 2, image_height)
        center_top = ((left_top[0] + right_top[0]) // 2, left_top[1])

        # Calculate deviation from straight line
        deviation = center_bottom[0] - center_top[0]

        # Determine direction
        if abs(deviation) < 20:
            direction = "Straight"
        elif deviation > 0:
            direction = "Curve Right"
        else:
            direction = "Curve Left"

        return abs(deviation), direction

    def detect_lanes(self, image):
        """
        Main lane detection function
        """
        height, width = image.shape[:2]

        # Step 1: Create ROI mask
        roi_mask = self.create_roi_mask(image)

        # Step 2: Detect edges
        edges = self.detect_edges(image)

        # Step 3: Apply ROI mask to edges
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # Step 4: Detect lines
        lines = self.detect_lines(masked_edges)

        # Step 5: Classify lines
        left_lines, right_lines = self.classify_lines(lines, width)

        # Step 6: Average lines
        left_lane = self.average_lines(left_lines)
        right_lane = self.average_lines(right_lines)

        # Step 7: Extrapolate lines
        y_min = int(height * (1 - self.roi_height))
        y_max = height

        if left_lane:
            left_lane = self.extrapolate_line(left_lane, y_min, y_max)
        if right_lane:
            right_lane = self.extrapolate_line(right_lane, y_min, y_max)

        # Step 8: Smooth using history
        left_lane, right_lane = self.smooth_lanes(left_lane, right_lane)

        # Step 9: Calculate curvature
        curvature, direction = self.calculate_lane_curvature(left_lane, right_lane, height)

        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'curvature': curvature,
            'direction': direction,
            'roi_mask': roi_mask,
            'edges': masked_edges
        }

    def draw_lanes(self, image, lane_data):
        """
        Vẽ lane detection lên image
        """
        result = image.copy()

        left_lane = lane_data['left_lane']
        right_lane = lane_data['right_lane']

        # Draw lanes
        if left_lane:
            cv2.line(result, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 8)

        if right_lane:
            cv2.line(result, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 8)

        # Fill lane area if both lanes detected
        if left_lane and right_lane:
            lane_points = np.array([
                [left_lane[0], left_lane[1]],
                [left_lane[2], left_lane[3]],
                [right_lane[2], right_lane[3]],
                [right_lane[0], right_lane[1]]
            ], dtype=np.int32)

            # Create transparent overlay
            overlay = result.copy()
            cv2.fillPoly(overlay, [lane_points], (0, 255, 255))
            result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)

        # Add text information
        direction = lane_data['direction']
        curvature = lane_data['curvature']

        cv2.putText(result, f"Direction: {direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if curvature:
            cv2.putText(result, f"Curvature: {curvature:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result

def main():
    """Test lane detection"""
    detector = LaneDetector()

    # Test with video
    cap = cv2.VideoCapture("video/sample.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect lanes
        lane_data = detector.detect_lanes(frame)

        # Draw results
        result = detector.draw_lanes(frame, lane_data)

        # Show result
        cv2.imshow('Lane Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()