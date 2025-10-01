"""Unit tests for detection service."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from src.application.detection_service import DetectionService
from src.domain.entities import Frame, Detection, GestureDetection, DetectionSource


class TestDetectionService:
    """Test DetectionService."""

    @pytest.fixture
    def mock_object_detector(self, mocker):
        """Create mock object detector."""
        detector = mocker.MagicMock()
        detector.initialize = AsyncMock()
        detector.detect = AsyncMock(return_value=[
            Detection(class_name="car", confidence=0.9)
        ])
        detector.cleanup = AsyncMock()
        return detector

    @pytest.fixture
    def mock_gesture_detector(self, mocker):
        """Create mock gesture detector."""
        detector = mocker.MagicMock()
        detector.initialize = AsyncMock()
        detector.detect = AsyncMock(return_value=[
            GestureDetection(class_name="hand", confidence=0.85)
        ])
        detector.cleanup = AsyncMock()
        return detector

    @pytest.fixture
    def mock_performance_monitor(self, mocker):
        """Create mock performance monitor."""
        monitor = mocker.MagicMock()
        monitor.record_frame_processing = MagicMock()
        monitor.record_detection = MagicMock()
        return monitor

    @pytest.fixture
    def detection_service(self, mock_object_detector, mock_gesture_detector, mock_performance_monitor):
        """Create detection service with mocks."""
        return DetectionService(
            object_detector=mock_object_detector,
            gesture_detector=mock_gesture_detector,
            performance_monitor=mock_performance_monitor
        )

    @pytest.mark.asyncio
    async def test_initialize(self, detection_service, mock_object_detector, mock_gesture_detector):
        """Test service initialization."""
        await detection_service.initialize()

        assert detection_service._initialized
        mock_object_detector.initialize.assert_called_once()
        mock_gesture_detector.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_frame(self, detection_service, sample_frame):
        """Test processing a single frame."""
        await detection_service.initialize()
        result = await detection_service.process_frame(sample_frame)

        assert result is not None
        assert result.frame_id == sample_frame.id
        assert result.frame_number == sample_frame.frame_number
        assert len(result.object_detections) == 1
        assert len(result.gesture_detections) == 1

    @pytest.mark.asyncio
    async def test_process_frame_object_only(
        self,
        detection_service,
        sample_frame,
        mock_gesture_detector
    ):
        """Test processing frame with object detection only."""
        await detection_service.initialize()
        result = await detection_service.process_frame(
            sample_frame,
            detect_objects=True,
            detect_gestures=False
        )

        assert len(result.object_detections) == 1
        assert len(result.gesture_detections) == 0
        mock_gesture_detector.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_frame_gesture_only(
        self,
        detection_service,
        sample_frame,
        mock_object_detector
    ):
        """Test processing frame with gesture detection only."""
        await detection_service.initialize()
        result = await detection_service.process_frame(
            sample_frame,
            detect_objects=False,
            detect_gestures=True
        )

        assert len(result.object_detections) == 0
        assert len(result.gesture_detections) == 1
        mock_object_detector.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_frame_skip(self, detection_service, sample_frame):
        """Test frame skipping functionality."""
        detection_service.frame_skip = 2  # Skip 2 out of 3 frames
        await detection_service.initialize()

        # Process 3 frames
        results = []
        for i in range(3):
            sample_frame.frame_number = i
            result = await detection_service.process_frame(sample_frame)
            results.append(result)

        # Only first frame should be processed
        assert results[0].metadata.get("skipped") is None
        assert results[1].metadata.get("skipped") is True
        assert results[2].metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_parallel_processing(
        self,
        mock_object_detector,
        mock_gesture_detector,
        mock_performance_monitor,
        sample_frame
    ):
        """Test parallel processing of detectors."""
        service = DetectionService(
            object_detector=mock_object_detector,
            gesture_detector=mock_gesture_detector,
            performance_monitor=mock_performance_monitor,
            parallel_processing=True
        )

        await service.initialize()
        result = await service.process_frame(sample_frame)

        assert result is not None
        # Both detectors should have been called
        mock_object_detector.detect.assert_called_once()
        mock_gesture_detector.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_processing(
        self,
        mock_object_detector,
        mock_gesture_detector,
        mock_performance_monitor,
        sample_frame
    ):
        """Test sequential processing of detectors."""
        service = DetectionService(
            object_detector=mock_object_detector,
            gesture_detector=mock_gesture_detector,
            performance_monitor=mock_performance_monitor,
            parallel_processing=False
        )

        await service.initialize()
        result = await service.process_frame(sample_frame)

        assert result is not None
        mock_object_detector.detect.assert_called_once()
        mock_gesture_detector.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(
        self,
        detection_service,
        mock_object_detector,
        mock_gesture_detector
    ):
        """Test service cleanup."""
        await detection_service.initialize()
        await detection_service.cleanup()

        assert not detection_service._initialized
        mock_object_detector.cleanup.assert_called_once()
        mock_gesture_detector.cleanup.assert_called_once()

    def test_update_settings(self, detection_service):
        """Test updating service settings."""
        detection_service.update_settings({
            "frame_skip": 3,
            "parallel_processing": False
        })

        assert detection_service.frame_skip == 3
        assert detection_service.parallel_processing is False

    def test_get_status(self, detection_service):
        """Test getting service status."""
        status = detection_service.get_status()

        assert status["initialized"] is False
        assert status["object_detector"] is True
        assert status["gesture_detector"] is True
        assert status["performance_monitor"] is True
        assert status["frame_skip"] == 0
        assert status["parallel_processing"] is True

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        detection_service,
        sample_frame,
        mock_object_detector
    ):
        """Test error handling during detection."""
        # Make detector raise an exception
        mock_object_detector.detect = AsyncMock(
            side_effect=Exception("Detection failed")
        )

        await detection_service.initialize()
        result = await detection_service.process_frame(sample_frame)

        # Should still return a result, but with no detections
        assert result is not None
        assert len(result.object_detections) == 0
        assert len(result.gesture_detections) == 1  # Gesture should still work

    @pytest.mark.asyncio
    async def test_process_stream(self, detection_service, mocker):
        """Test processing a stream of frames."""
        # Create mock frame source
        mock_source = mocker.MagicMock()
        mock_source.start = AsyncMock()
        mock_source.stop = AsyncMock()

        # Create async generator for frames
        async def frame_generator():
            for i in range(3):
                frame = Frame(
                    data=np.zeros((480, 640, 3)),
                    width=640,
                    height=480,
                    frame_number=i
                )
                yield frame

        mock_source.get_frames = frame_generator

        await detection_service.initialize()

        # Process stream
        results = []
        async for result in detection_service.process_stream(mock_source):
            results.append(result)

        assert len(results) == 3
        mock_source.start.assert_called_once()
        mock_source.stop.assert_called_once()