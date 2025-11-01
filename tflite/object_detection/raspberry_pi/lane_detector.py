"""
Lightweight Lane Detection Module for Raspberry Pi
Optimized for real-time performance using OpenCV
"""

import cv2
import numpy as np


class LaneDetector:
    """Lightweight lane detector optimized for Raspberry Pi"""

    def __init__(self, img_height=480, img_width=640):
        """
        Initialize lane detector with image dimensions

        Args:
            img_height: Frame height in pixels
            img_width: Frame width in pixels
        """
        self.img_height = img_height
        self.img_width = img_width

        # ROI parameters (only process bottom 60% of frame)
        self.roi_height_ratio = 0.6

        # Canny edge detection parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150

        # Hough Transform parameters
        self.hough_threshold = 50
        self.hough_min_line_length = 100
        self.hough_max_line_gap = 50

        # Visualization parameters
        self.lane_color = (0, 255, 0)  # Green
        self.lane_thickness = 3
        self.roi_color = (255, 0, 0)  # Blue
        self.roi_thickness = 2

    def get_roi_mask(self, img_shape):
        """
        Create Region of Interest (ROI) mask
        Focus on the lower part of the image where lanes are

        Args:
            img_shape: Shape of the input image (height, width)

        Returns:
            Binary mask for ROI
        """
        height, width = img_shape[:2]

        # Define trapezoidal ROI vertices
        # Bottom left, Top left, Top right, Bottom right
        vertices = np.array([
            [
                (0, height),
                (width * 0.4, height * self.roi_height_ratio),
                (width * 0.6, height * self.roi_height_ratio),
                (width, height)
            ]
        ], dtype=np.int32)

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)

        return mask, vertices

    def preprocess_frame(self, frame):
        """
        Preprocess frame for lane detection

        Args:
            frame: Input BGR frame

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        return blur

    def detect_edges(self, preprocessed_frame):
        """
        Detect edges using Canny edge detection

        Args:
            preprocessed_frame: Grayscale blurred frame

        Returns:
            Edge detected frame
        """
        edges = cv2.Canny(
            preprocessed_frame,
            self.canny_low_threshold,
            self.canny_high_threshold
        )

        return edges

    def detect_lines(self, edges):
        """
        Detect lines using Hough Transform

        Args:
            edges: Edge detected frame

        Returns:
            List of detected lines
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        return lines

    def separate_left_right_lines(self, lines):
        """
        Separate detected lines into left and right lanes
        Based on line slope

        Args:
            lines: Array of detected lines

        Returns:
            Tuple of (left_lines, right_lines)
        """
        if lines is None:
            return [], []

        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Vertical line, skip
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter out horizontal lines (slope close to 0)
            if abs(slope) < 0.5:
                continue

            # Classify based on slope
            if slope < 0:  # Left lane (negative slope)
                left_lines.append(line[0])
            else:  # Right lane (positive slope)
                right_lines.append(line[0])

        return left_lines, right_lines

    def average_lines(self, lines):
        """
        Average multiple lines into a single representative line

        Args:
            lines: List of lines

        Returns:
            Single averaged line or None
        """
        if not lines:
            return None

        # Extract all points
        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        # Fit a line using polyfit
        if len(x_coords) < 2:
            return None

        poly = np.polyfit(x_coords, y_coords, 1)

        # Calculate line endpoints
        y1 = self.img_height
        y2 = int(self.img_height * self.roi_height_ratio)

        x1 = int((y1 - poly[1]) / poly[0])
        x2 = int((y2 - poly[1]) / poly[0])

        return [x1, y1, x2, y2]

    def detect(self, frame):
        """
        Main lane detection pipeline

        Args:
            frame: Input BGR frame

        Returns:
            Dictionary containing:
                - left_lane: Left lane line coordinates
                - right_lane: Right lane line coordinates
                - raw_lines: All detected lines
        """
        # Store original dimensions
        self.img_height, self.img_width = frame.shape[:2]

        # Preprocess
        preprocessed = self.preprocess_frame(frame)

        # Detect edges
        edges = self.detect_edges(preprocessed)

        # Apply ROI mask
        mask, _ = self.get_roi_mask(frame.shape)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines
        lines = self.detect_lines(masked_edges)

        # Separate left and right lines
        left_lines, right_lines = self.separate_left_right_lines(lines)

        # Average lines
        left_lane = self.average_lines(left_lines)
        right_lane = self.average_lines(right_lines)

        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'raw_lines': lines
        }

    def visualize(self, frame, detection_result, show_roi=False):
        """
        Draw detected lanes on frame

        Args:
            frame: Input frame
            detection_result: Result from detect() method
            show_roi: Whether to show ROI region

        Returns:
            Frame with lanes drawn
        """
        # Create a copy to draw on
        output = frame.copy()

        # Draw ROI if requested
        if show_roi:
            _, vertices = self.get_roi_mask(frame.shape)
            cv2.polylines(
                output,
                vertices,
                True,
                self.roi_color,
                self.roi_thickness
            )

        # Draw left lane
        if detection_result['left_lane'] is not None:
            x1, y1, x2, y2 = detection_result['left_lane']
            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                self.lane_color,
                self.lane_thickness
            )

        # Draw right lane
        if detection_result['right_lane'] is not None:
            x1, y1, x2, y2 = detection_result['right_lane']
            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                self.lane_color,
                self.lane_thickness
            )

        # Fill lane area with semi-transparent overlay
        if (detection_result['left_lane'] is not None and
            detection_result['right_lane'] is not None):

            left_line = detection_result['left_lane']
            right_line = detection_result['right_lane']

            # Create polygon for lane area
            lane_polygon = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)

            # Create overlay
            overlay = output.copy()
            cv2.fillPoly(overlay, [lane_polygon], (0, 255, 0))

            # Blend with original
            cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)

        return output
