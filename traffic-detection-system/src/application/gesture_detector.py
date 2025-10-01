"""Hand gesture detection for traffic scenarios using MediaPipe."""

import asyncio
import logging
from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils

from ..domain.entities import (
    GestureDetection,
    GestureType,
    Frame,
    BoundingBox
)
from ..domain.interfaces import IGestureDetector


logger = logging.getLogger(__name__)


class TrafficGestureDetector(IGestureDetector):
    """Hand gesture detector for traffic control scenarios."""

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0  # 0 for lite model (faster)
    ):
        """
        Initialize gesture detector.

        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            model_complexity: Model complexity (0=lite, 1=full)
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.hands: Optional[mp_hands.Hands] = None
        self.mp_drawing = drawing_utils
        self.mp_hands = mp_hands

    async def initialize(self) -> None:
        """Initialize the gesture detector."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_mediapipe)
            logger.info("MediaPipe Hands initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def _initialize_mediapipe(self) -> None:
        """Initialize MediaPipe Hands model."""
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity
        )

    async def detect(self, frame: Frame) -> List[GestureDetection]:
        """
        Detect hand gestures in a frame.

        Args:
            frame: Input frame

        Returns:
            List of detected gestures
        """
        if self.hands is None:
            await self.initialize()

        if not frame.is_valid:
            logger.warning("Invalid frame provided for gesture detection")
            return []

        try:
            # Run detection in thread pool
            loop = asyncio.get_event_loop()
            gestures = await loop.run_in_executor(
                None,
                self._detect_sync,
                frame.data,
                frame.width,
                frame.height
            )
            return gestures
        except Exception as e:
            logger.error(f"Gesture detection failed: {e}")
            return []

    def _detect_sync(
        self,
        image: np.ndarray,
        width: int,
        height: int
    ) -> List[GestureDetection]:
        """
        Synchronous gesture detection.

        Args:
            image: Input image
            width: Image width
            height: Image height

        Returns:
            List of gesture detections
        """
        if self.hands is None:
            raise RuntimeError("MediaPipe not initialized")

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image[:, :, ::-1].copy()
        else:
            rgb_image = image

        # Process the image
        results = self.hands.process(rgb_image)

        gestures = []
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract hand side information
                hand_side = None
                if results.multi_handedness:
                    hand_info = results.multi_handedness[hand_idx]
                    hand_side = hand_info.classification[0].label.lower()

                # Convert landmarks to pixel coordinates
                landmarks = []
                xs, ys = [], []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmarks.append((x, y))
                    xs.append(x)
                    ys.append(y)

                # Calculate bounding box
                bbox = BoundingBox(
                    x1=float(min(xs)),
                    y1=float(min(ys)),
                    x2=float(max(xs)),
                    y2=float(max(ys))
                )

                # Classify the gesture
                gesture_type = self._classify_gesture(hand_landmarks, hand_side)

                # Create gesture detection
                gesture = GestureDetection(
                    class_name="hand_gesture",
                    confidence=0.9,  # MediaPipe doesn't provide confidence
                    bounding_box=bbox,
                    gesture_type=gesture_type,
                    landmarks=landmarks,
                    hand_side=hand_side,
                    metadata={
                        "detector": "mediapipe",
                        "hand_index": hand_idx
                    }
                )
                gestures.append(gesture)

        return gestures

    def _classify_gesture(
        self,
        hand_landmarks: Any,
        hand_side: Optional[str]
    ) -> GestureType:
        """
        Classify hand gesture based on landmarks.

        Args:
            hand_landmarks: MediaPipe hand landmarks
            hand_side: Which hand (left/right)

        Returns:
            Detected gesture type
        """
        # Get landmark positions
        landmarks = hand_landmarks.landmark

        # Extract key points
        wrist_y = landmarks[self.WRIST].y
        thumb_tip_y = landmarks[self.THUMB_TIP].y
        index_tip_y = landmarks[self.INDEX_TIP].y
        middle_tip_y = landmarks[self.MIDDLE_TIP].y
        ring_tip_y = landmarks[self.RING_TIP].y
        pinky_tip_y = landmarks[self.PINKY_TIP].y

        thumb_tip_x = landmarks[self.THUMB_TIP].x
        index_tip_x = landmarks[self.INDEX_TIP].x
        wrist_x = landmarks[self.WRIST].x

        # Check if fingers are raised
        fingers_up = []
        # Thumb (special case - check x coordinate)
        if hand_side == "right":
            fingers_up.append(thumb_tip_x > landmarks[self.THUMB_TIP - 1].x)
        else:
            fingers_up.append(thumb_tip_x < landmarks[self.THUMB_TIP - 1].x)

        # Other fingers (check y coordinate)
        for tip_idx in [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]:
            pip_idx = tip_idx - 2  # PIP joint
            fingers_up.append(landmarks[tip_idx].y < landmarks[pip_idx].y)

        # Count raised fingers
        raised_count = sum(fingers_up)

        # STOP gesture: Open palm (all fingers raised)
        if raised_count >= 5:
            return GestureType.STOP

        # GO gesture: Pointing forward (index finger raised)
        if fingers_up[1] and raised_count == 1:
            return GestureType.GO

        # TURN gestures: Check hand orientation
        if raised_count >= 3:
            # Calculate hand orientation
            hand_angle = np.arctan2(
                index_tip_y - wrist_y,
                index_tip_x - wrist_x
            )
            angle_degrees = np.degrees(hand_angle)

            # Determine turn direction based on angle
            if -45 < angle_degrees < 45:
                if hand_side == "right":
                    return GestureType.TURN_RIGHT
                else:
                    return GestureType.TURN_LEFT
            elif 135 < angle_degrees or angle_degrees < -135:
                if hand_side == "right":
                    return GestureType.TURN_LEFT
                else:
                    return GestureType.TURN_RIGHT

        # SLOW DOWN: Palm facing down
        if raised_count == 0 and wrist_y < middle_tip_y:
            return GestureType.SLOW_DOWN

        # Default
        return GestureType.UNKNOWN

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
            logger.info("MediaPipe Hands cleaned up")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the gesture detector."""
        return {
            "detector": "mediapipe",
            "max_hands": self.max_num_hands,
            "detection_confidence": self.min_detection_confidence,
            "tracking_confidence": self.min_tracking_confidence,
            "model_complexity": self.model_complexity,
            "status": "initialized" if self.hands else "not_initialized"
        }