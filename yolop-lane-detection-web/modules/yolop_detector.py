"""
YOLOP Detection Module
Based on https://github.com/hustvl/YOLOP
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class YOLOPDetector:
    def __init__(self, model_path=None, use_gpu=False):
        """
        Initialize YOLOP detector
        For demo, using simplified detection
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Model configuration
        self.input_size = (640, 384)  # YOLOP input size

        # For demo purposes, using OpenCV methods
        # In production, load actual YOLOP model:
        # self.model = torch.load(model_path)
        # self.model.to(self.device)
        # self.model.eval()

        # Lane detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 50

        # Object detection (simplified)
        self.vehicle_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_car.xml'
        ) if hasattr(cv2.data, 'haarcascades') else None

    def detect(self, frame):
        """
        Run YOLOP detection on frame
        Returns: dict with lanes, drivable_area, and objects
        """
        h, w = frame.shape[:2]

        # Prepare result
        result = {
            'lanes': [],
            'drivable_area': None,
            'objects': []
        }

        # 1. Lane Detection
        lanes = self.detect_lanes(frame)
        result['lanes'] = lanes

        # 2. Drivable Area Segmentation
        drivable_area = self.segment_drivable_area(frame)
        result['drivable_area'] = drivable_area

        # 3. Vehicle Detection
        vehicles = self.detect_vehicles(frame)
        result['objects'] = vehicles

        return result

    def detect_lanes(self, frame):
        """
        Detect lane lines using traditional CV
        In production, use YOLOP lane detection head
        """
        h, w = frame.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Region of interest (lower half of image)
        mask = np.zeros_like(edges)
        vertices = np.array([
            [(0, h),
             (w * 0.45, h * 0.6),
             (w * 0.55, h * 0.6),
             (w, h)]
        ], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # Group lines into lanes
        lanes = []
        if lines is not None:
            left_lines = []
            right_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate slope
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)

                    # Filter by slope
                    if abs(slope) > 0.5:  # Ignore horizontal lines
                        if slope < 0:
                            left_lines.append(line[0])
                        else:
                            right_lines.append(line[0])

            # Fit polynomial to left lane
            if len(left_lines) > 0:
                left_points = []
                for x1, y1, x2, y2 in left_lines:
                    left_points.extend([(x1, y1), (x2, y2)])
                left_points = np.array(left_points)

                # Generate lane points
                if len(left_points) > 2:
                    left_lane = self.fit_lane(left_points, h)
                    if left_lane is not None:
                        lanes.append(left_lane)

            # Fit polynomial to right lane
            if len(right_lines) > 0:
                right_points = []
                for x1, y1, x2, y2 in right_lines:
                    right_points.extend([(x1, y1), (x2, y2)])
                right_points = np.array(right_points)

                # Generate lane points
                if len(right_points) > 2:
                    right_lane = self.fit_lane(right_points, h)
                    if right_lane is not None:
                        lanes.append(right_lane)

        return lanes

    def fit_lane(self, points, img_height):
        """Fit polynomial to lane points"""
        try:
            # Fit second-order polynomial
            x = points[:, 0]
            y = points[:, 1]

            # Polynomial fit
            z = np.polyfit(y, x, 2)
            p = np.poly1d(z)

            # Generate lane curve
            y_points = np.linspace(img_height * 0.6, img_height, 10)
            x_points = p(y_points)

            # Create lane points
            lane_points = [(int(x), int(y)) for x, y in zip(x_points, y_points)]

            return lane_points
        except:
            return None

    def segment_drivable_area(self, frame):
        """
        Segment drivable area
        In production, use YOLOP segmentation head
        """
        h, w = frame.shape[:2]

        # Create mask for drivable area (simplified)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Define drivable area polygon (road region)
        vertices = np.array([
            [(w * 0.1, h),
             (w * 0.4, h * 0.6),
             (w * 0.6, h * 0.6),
             (w * 0.9, h)]
        ], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)

        # Apply color-based segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Road color range (gray/black)
        lower_road = np.array([0, 0, 50])
        upper_road = np.array([180, 30, 200])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)

        # Combine masks
        drivable_mask = cv2.bitwise_and(mask, road_mask)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)
        drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_OPEN, kernel)

        return drivable_mask

    def detect_vehicles(self, frame):
        """
        Detect vehicles in frame
        In production, use YOLOP detection head
        """
        vehicles = []

        # Use cascade classifier for demo
        if self.vehicle_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect vehicles
            cars = self.vehicle_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            for (x, y, w, h) in cars:
                vehicles.append({
                    'bbox': [x, y, x + w, y + h],
                    'class': 'vehicle',
                    'confidence': 0.75  # Demo confidence
                })

        # In production, use YOLO detection:
        # detections = self.model(frame)
        # for det in detections:
        #     if det.class in ['car', 'truck', 'bus', 'motorcycle']:
        #         vehicles.append({
        #             'bbox': det.bbox,
        #             'class': det.class_name,
        #             'confidence': det.confidence
        #         })

        return vehicles

    def preprocess_image(self, image):
        """
        Preprocess image for YOLOP model
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)

        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize
        normalized = rgb.astype(np.float32) / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return tensor