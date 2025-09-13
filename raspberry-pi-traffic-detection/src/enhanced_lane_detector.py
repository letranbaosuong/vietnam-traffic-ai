#!/usr/bin/env python3
"""
Enhanced Lane Detection with Departure Warning
Phát hiện làn đường nâng cao với cảnh báo chệch làn
Inspired by Udacity's Lane Detection Project
"""

import cv2
import numpy as np
import math
import time
from pathlib import Path

class EnhancedLaneDetector:
    def __init__(self):
        """Initialize enhanced lane detector with departure warning"""
        # Basic lane detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_rho = 2
        self.hough_theta = np.pi / 180
        self.hough_threshold = 100
        self.hough_min_line_length = 40
        self.hough_max_line_gap = 5

        # ROI parameters
        self.roi_bottom_width = 0.85
        self.roi_top_width = 0.08
        self.roi_height = 0.62

        # Lane departure parameters
        self.lane_width_pixels = 0  # Will be calculated
        self.vehicle_center = 0.5   # Assume vehicle center is at 50% of image width
        self.departure_threshold = 0.15  # 15% deviation triggers warning
        self.confidence_threshold = 0.7  # Minimum confidence for valid detection

        # History for smoothing
        self.lane_history = []
        self.departure_history = []
        self.history_length = 10

        # Warning states
        self.warning_active = False
        self.warning_type = None
        self.warning_level = 0  # 0: None, 1: Mild, 2: Moderate, 3: Severe

        # Vietnamese warning messages
        self.warning_messages = {
            'left_departure': 'CẢNH BÁO: Chệch trái làn đường!',
            'right_departure': 'CẢNH BÁO: Chệch phải làn đường!',
            'no_lanes': 'CẢNH BÁO: Không phát hiện làn đường!',
            'curve_ahead': 'CHÚ Ý: Đường cong phía trước',
            'lane_change': 'Đang chuyển làn đường'
        }

    def create_roi_mask(self, image):
        """Create region of interest mask"""
        height, width = image.shape[:2]

        # Define trapezoid vertices
        bottom_left = [int(width * (1 - self.roi_bottom_width) / 2), height]
        bottom_right = [int(width * (1 + self.roi_bottom_width) / 2), height]
        top_left = [int(width * (1 - self.roi_top_width) / 2), int(height * (1 - self.roi_height))]
        top_right = [int(width * (1 + self.roi_top_width) / 2), int(height * (1 - self.roi_height))]

        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)

        return mask

    def detect_edges(self, image):
        """Enhanced edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        return edges

    def detect_lines(self, edges):
        """Detect lines using Hough Transform"""
        lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        return lines

    def classify_and_average_lines(self, lines, image_width):
        """Classify lines into left and right lanes and calculate averages"""
        if lines is None:
            return None, None, 0.0

        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter based on slope threshold (ignore nearly horizontal lines)
            if abs(slope) < 0.5:
                continue

            # Classify based on slope and position
            if slope < 0 and x1 < image_width / 2:  # Left lane
                left_lines.append([x1, y1, x2, y2, slope])
            elif slope > 0 and x1 > image_width / 2:  # Right lane
                right_lines.append([x1, y1, x2, y2, slope])

        # Calculate average lines
        left_lane = self.calculate_average_line(left_lines)
        right_lane = self.calculate_average_line(right_lines)

        # Calculate confidence based on number of lines detected
        confidence = min(1.0, (len(left_lines) + len(right_lines)) / 10.0)

        return left_lane, right_lane, confidence

    def calculate_average_line(self, lines):
        """Calculate average line from multiple line segments"""
        if not lines:
            return None

        lines = np.array(lines)

        # Calculate weighted average (weight by line length)
        weights = []
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            weights.append(length)

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Weighted average
        avg_slope = np.average(lines[:, 4], weights=weights)
        avg_x = np.average((lines[:, 0] + lines[:, 2]) / 2, weights=weights)
        avg_y = np.average((lines[:, 1] + lines[:, 3]) / 2, weights=weights)

        return [avg_x, avg_y, avg_slope]

    def extrapolate_line(self, line_data, y_min, y_max):
        """Extrapolate line to full lane length"""
        if line_data is None:
            return None

        avg_x, avg_y, slope = line_data

        # Calculate intercept: b = y - mx
        intercept = avg_y - slope * avg_x

        # Calculate x coordinates for y_min and y_max
        x_min = int((y_min - intercept) / slope)
        x_max = int((y_max - intercept) / slope)

        return [x_min, y_min, x_max, y_max]

    def calculate_lane_departure(self, left_lane, right_lane, image_width, image_height):
        """Calculate lane departure metrics"""
        if left_lane is None and right_lane is None:
            return {
                'departure_ratio': 0.0,
                'warning_type': 'no_lanes',
                'warning_level': 3,
                'lane_center': image_width // 2,
                'vehicle_position': image_width // 2,
                'lane_width': 0
            }

        # Calculate lane center and vehicle position
        vehicle_x = int(image_width * self.vehicle_center)

        if left_lane is not None and right_lane is not None:
            # Both lanes detected - ideal case
            left_x = left_lane[2]  # x at bottom
            right_x = right_lane[2]  # x at bottom

            lane_center = (left_x + right_x) // 2
            lane_width = abs(right_x - left_x)

            # Calculate departure ratio
            departure_pixels = abs(vehicle_x - lane_center)
            departure_ratio = departure_pixels / (lane_width / 2) if lane_width > 0 else 0

            # Determine warning type
            if departure_ratio < self.departure_threshold:
                warning_type = None
                warning_level = 0
            elif vehicle_x < lane_center:
                warning_type = 'left_departure'
                warning_level = min(3, int(departure_ratio / self.departure_threshold))
            else:
                warning_type = 'right_departure'
                warning_level = min(3, int(departure_ratio / self.departure_threshold))

        elif left_lane is not None:
            # Only left lane detected
            left_x = left_lane[2]
            estimated_lane_width = 300  # Estimate based on typical lane width
            lane_center = left_x + estimated_lane_width // 2
            lane_width = estimated_lane_width

            departure_pixels = abs(vehicle_x - lane_center)
            departure_ratio = departure_pixels / (lane_width / 2)

            if vehicle_x < left_x:
                warning_type = 'left_departure'
                warning_level = 2
            else:
                warning_type = None
                warning_level = 1

        elif right_lane is not None:
            # Only right lane detected
            right_x = right_lane[2]
            estimated_lane_width = 300
            lane_center = right_x - estimated_lane_width // 2
            lane_width = estimated_lane_width

            departure_pixels = abs(vehicle_x - lane_center)
            departure_ratio = departure_pixels / (lane_width / 2)

            if vehicle_x > right_x:
                warning_type = 'right_departure'
                warning_level = 2
            else:
                warning_type = None
                warning_level = 1

        return {
            'departure_ratio': departure_ratio,
            'warning_type': warning_type,
            'warning_level': warning_level,
            'lane_center': lane_center,
            'vehicle_position': vehicle_x,
            'lane_width': lane_width
        }

    def smooth_lane_detection(self, left_lane, right_lane, departure_info):
        """Smooth lane detection using history"""
        # Add current detection to history
        current_data = {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'departure_info': departure_info
        }

        self.lane_history.append(current_data)
        self.departure_history.append(departure_info['warning_level'])

        # Keep only recent history
        if len(self.lane_history) > self.history_length:
            self.lane_history.pop(0)
        if len(self.departure_history) > self.history_length:
            self.departure_history.pop(0)

        # Calculate smoothed warning level
        if len(self.departure_history) >= 3:
            avg_warning_level = np.mean(self.departure_history[-3:])
            departure_info['warning_level'] = int(avg_warning_level)

        return left_lane, right_lane, departure_info

    def detect_enhanced_lanes(self, image):
        """Main enhanced lane detection function"""
        height, width = image.shape[:2]

        # Step 1: Create ROI mask
        roi_mask = self.create_roi_mask(image)

        # Step 2: Detect edges
        edges = self.detect_edges(image)

        # Step 3: Apply ROI mask
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # Step 4: Detect lines
        lines = self.detect_lines(masked_edges)

        # Step 5: Classify and average lines
        left_lane_data, right_lane_data, confidence = self.classify_and_average_lines(lines, width)

        # Step 6: Extrapolate lines
        y_min = int(height * (1 - self.roi_height))
        y_max = height

        left_lane = self.extrapolate_line(left_lane_data, y_min, y_max) if left_lane_data else None
        right_lane = self.extrapolate_line(right_lane_data, y_min, y_max) if right_lane_data else None

        # Step 7: Calculate lane departure
        departure_info = self.calculate_lane_departure(left_lane, right_lane, width, height)

        # Step 8: Smooth detection
        left_lane, right_lane, departure_info = self.smooth_lane_detection(
            left_lane, right_lane, departure_info
        )

        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'departure_info': departure_info,
            'confidence': confidence,
            'roi_mask': roi_mask,
            'edges': masked_edges
        }

    def draw_enhanced_lanes(self, image, lane_data):
        """Draw enhanced lane detection with warnings"""
        result = image.copy()

        left_lane = lane_data['left_lane']
        right_lane = lane_data['right_lane']
        departure_info = lane_data['departure_info']
        confidence = lane_data['confidence']

        # Draw lane lines
        if left_lane:
            cv2.line(result, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]),
                    (0, 255, 0), 8)

        if right_lane:
            cv2.line(result, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]),
                    (0, 255, 0), 8)

        # Fill lane area
        if left_lane and right_lane:
            lane_points = np.array([
                [left_lane[0], left_lane[1]],
                [left_lane[2], left_lane[3]],
                [right_lane[2], right_lane[3]],
                [right_lane[0], right_lane[1]]
            ], dtype=np.int32)

            overlay = result.copy()
            cv2.fillPoly(overlay, [lane_points], (0, 255, 255))
            result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)

        # Draw vehicle center line
        height, width = result.shape[:2]
        vehicle_x = int(width * self.vehicle_center)
        cv2.line(result, (vehicle_x, height-50), (vehicle_x, height-150), (255, 0, 0), 3)

        # Draw lane center
        if departure_info['lane_center']:
            lane_center_x = departure_info['lane_center']
            cv2.line(result, (lane_center_x, height-50), (lane_center_x, height-150), (0, 255, 255), 3)

        # Draw departure warning
        self.draw_departure_warning(result, departure_info, confidence)

        # Draw status information
        self.draw_status_info(result, departure_info, confidence)

        return result

    def draw_departure_warning(self, image, departure_info, confidence):
        """Draw departure warning on screen"""
        warning_type = departure_info['warning_type']
        warning_level = departure_info['warning_level']

        if warning_level == 0 or confidence < self.confidence_threshold:
            return

        # Choose warning color based on severity
        if warning_level == 1:
            color = (0, 255, 255)  # Yellow
        elif warning_level == 2:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)    # Red

        # Get warning message
        message = self.warning_messages.get(warning_type, 'CẢNH BÁO!')

        # Draw warning background
        height, width = image.shape[:2]
        warning_y = 50
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]

        # Flashing effect for severe warnings
        if warning_level >= 3:
            flash = int(time.time() * 4) % 2  # Flash every 0.25 seconds
            if flash:
                cv2.rectangle(image, (0, 0), (width, warning_y + 60), color, -1)

        # Warning background rectangle
        cv2.rectangle(image, (10, warning_y - 30), (20 + text_size[0], warning_y + 20), color, -1)

        # Warning text
        cv2.putText(image, message, (15, warning_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Warning level indicator
        level_text = f"Mức độ: {warning_level}/3"
        cv2.putText(image, level_text, (15, warning_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_status_info(self, image, departure_info, confidence):
        """Draw status information"""
        height, width = image.shape[:2]

        status_info = [
            f"Confidence: {confidence:.2f}",
            f"Lane Width: {departure_info['lane_width']}px",
            f"Departure Ratio: {departure_info['departure_ratio']:.2f}",
            f"Vehicle Position: {departure_info['vehicle_position']}px",
            f"Lane Center: {departure_info['lane_center']}px"
        ]

        # Draw status background
        status_bg_height = len(status_info) * 25 + 20
        cv2.rectangle(image, (width - 350, height - status_bg_height),
                     (width - 10, height - 10), (0, 0, 0), -1)

        # Draw status text
        for i, info in enumerate(status_info):
            y_pos = height - status_bg_height + 25 + i * 25
            cv2.putText(image, info, (width - 340, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    """Test enhanced lane detection"""
    detector = EnhancedLaneDetector()

    # Test with video
    video_path = "video/sample.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    print("🛣️ Enhanced Lane Detection with Departure Warning")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Detect enhanced lanes
        lane_data = detector.detect_enhanced_lanes(frame)

        # Draw results
        result = detector.draw_enhanced_lanes(frame, lane_data)

        # Display frame
        cv2.imshow('Enhanced Lane Detection with Departure Warning', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()