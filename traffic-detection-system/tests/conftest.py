"""Pytest configuration and fixtures."""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List
import cv2

from src.domain.entities import (
    Frame,
    Detection,
    GestureDetection,
    BoundingBox,
    DetectionSource,
    ObjectClass,
    GestureType
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    # Create a dummy image (640x480 RGB)
    image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    return Frame(
        data=image_data,
        width=640,
        height=480,
        frame_number=1,
        source=DetectionSource.CAMERA
    )


@pytest.fixture
def sample_detection():
    """Create a sample detection."""
    return Detection(
        class_name=ObjectClass.CAR.value,
        confidence=0.85,
        bounding_box=BoundingBox(x1=100, y1=100, x2=200, y2=200)
    )


@pytest.fixture
def sample_gesture_detection():
    """Create a sample gesture detection."""
    return GestureDetection(
        class_name="hand_gesture",
        confidence=0.9,
        bounding_box=BoundingBox(x1=150, y1=150, x2=250, y2=250),
        gesture_type=GestureType.STOP,
        landmarks=[(175, 175), (200, 200), (225, 225)],
        hand_side="right"
    )


@pytest.fixture
def sample_detections() -> List[Detection]:
    """Create multiple sample detections."""
    return [
        Detection(
            class_name=ObjectClass.CAR.value,
            confidence=0.9,
            bounding_box=BoundingBox(x1=50, y1=50, x2=150, y2=150)
        ),
        Detection(
            class_name=ObjectClass.PERSON.value,
            confidence=0.85,
            bounding_box=BoundingBox(x1=200, y1=100, x2=280, y2=300)
        ),
        Detection(
            class_name=ObjectClass.TRAFFIC_LIGHT.value,
            confidence=0.75,
            bounding_box=BoundingBox(x1=400, y1=50, x2=450, y2=150)
        )
    ]


@pytest.fixture
def test_image():
    """Create a test image with known objects."""
    # Create a simple test image with colored rectangles
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw some rectangles to simulate objects
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(image, (300, 150), (400, 250), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(image, (450, 50), (550, 150), (0, 0, 255), -1)   # Red rectangle

    return image


@pytest.fixture
def test_video_path(tmp_path):
    """Create a temporary test video file."""
    video_path = tmp_path / "test_video.mp4"

    # Create a simple video with 10 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))

    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add moving rectangle
        x = 50 + i * 30
        cv2.rectangle(frame, (x, 200), (x + 100, 300), (255, 255, 255), -1)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def mock_yolo_model(mocker):
    """Mock YOLO model for testing."""
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = [
        mocker.MagicMock(
            boxes=mocker.MagicMock(
                xyxy=np.array([[100, 100, 200, 200]]),
                conf=np.array([0.9]),
                cls=np.array([2]),  # Car class
                cpu=mocker.MagicMock(
                    return_value=mocker.MagicMock(
                        numpy=mocker.MagicMock(
                            return_value=mocker.MagicMock(
                                xyxy=np.array([[100, 100, 200, 200]]),
                                conf=np.array([0.9]),
                                cls=np.array([2])
                            )
                        )
                    )
                )
            )
        )
    ]
    return mock_model


@pytest.fixture
def mock_mediapipe_hands(mocker):
    """Mock MediaPipe Hands for testing."""
    mock_hands = mocker.MagicMock()

    # Create mock hand landmarks
    mock_landmark = mocker.MagicMock()
    mock_landmark.x = 0.5
    mock_landmark.y = 0.5

    mock_landmarks = mocker.MagicMock()
    mock_landmarks.landmark = [mock_landmark] * 21  # 21 hand landmarks

    mock_result = mocker.MagicMock()
    mock_result.multi_hand_landmarks = [mock_landmarks]
    mock_result.multi_handedness = [
        mocker.MagicMock(
            classification=[
                mocker.MagicMock(label="Right")
            ]
        )
    ]

    mock_hands.process.return_value = mock_result
    return mock_hands


@pytest.fixture
def settings_mock(mocker):
    """Mock application settings."""
    mock_settings = mocker.MagicMock()
    mock_settings.yolo_model = "yolov8n.pt"
    mock_settings.yolo_confidence = 0.5
    mock_settings.yolo_iou = 0.45
    mock_settings.gesture_confidence = 0.7
    mock_settings.camera_width = 640
    mock_settings.camera_height = 480
    mock_settings.camera_fps = 30
    mock_settings.platform_name = "Darwin"
    mock_settings.is_raspberry_pi = False
    mock_settings.use_gpu = False
    mock_settings.models_dir = Path("/tmp/models")
    mock_settings.data_dir = Path("/tmp/data")
    return mock_settings