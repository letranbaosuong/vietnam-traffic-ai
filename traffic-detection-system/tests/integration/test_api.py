"""Integration tests for FastAPI endpoints."""

import pytest
import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import cv2
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.presentation.api import app
from src.domain.entities import Detection, GestureDetection, Frame


class TestAPIEndpoints:
    """Test API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "platform" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data

    @pytest.mark.asyncio
    async def test_detect_image(self, async_client, test_image, mocker):
        """Test image detection endpoint."""
        # Mock detection service
        mock_service = mocker.patch("src.presentation.api.detection_service")
        mock_result = mocker.MagicMock()
        mock_result.to_dict.return_value = {
            "frame_id": "test",
            "object_detections": [],
            "gesture_detections": [],
            "processing_time_ms": 25.0
        }
        mock_service.process_frame = AsyncMock(return_value=mock_result)

        # Convert test image to bytes
        _, buffer = cv2.imencode('.jpg', test_image)
        image_bytes = buffer.tobytes()

        # Send request
        files = {"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        response = await async_client.post(
            "/detect/image",
            files=files,
            params={"visualize": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert "frame_id" in data
        assert "processing_time_ms" in data

    def test_get_metrics(self, client, mocker):
        """Test metrics endpoint."""
        # Mock performance monitor
        mock_monitor = mocker.patch("src.presentation.api.performance_monitor")
        mock_metrics = mocker.MagicMock()
        mock_metrics.to_dict.return_value = {
            "avg_fps": 30.0,
            "total_frames_processed": 100
        }
        mock_monitor.get_metrics.return_value = mock_metrics

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "avg_fps" in data

    def test_get_settings(self, client):
        """Test get settings endpoint."""
        response = client.get("/settings")
        assert response.status_code == 200
        data = response.json()
        assert "yolo_confidence" in data
        assert "platform" in data

    def test_update_settings(self, client):
        """Test update settings endpoint."""
        settings_data = {
            "yolo_confidence": 0.6,
            "frame_skip": 2
        }

        response = client.post("/settings", json=settings_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"

    @pytest.mark.asyncio
    async def test_camera_start(self, async_client, mocker):
        """Test camera start endpoint."""
        # Mock camera source
        mock_camera = mocker.MagicMock()
        mock_camera.is_available.return_value = True
        mock_camera.get_properties.return_value = {
            "status": "initialized",
            "resolution": "640x480"
        }
        mock_camera.start = AsyncMock()

        mocker.patch("src.presentation.api.CameraSource", return_value=mock_camera)

        response = await async_client.get("/camera/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["started", "already_running"]

    @pytest.mark.asyncio
    async def test_camera_stop(self, async_client, mocker):
        """Test camera stop endpoint."""
        # Mock camera source
        mock_camera = mocker.MagicMock()
        mock_camera.stop = AsyncMock()
        mocker.patch("src.presentation.api.camera_source", mock_camera)

        response = await async_client.get("/camera/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["stopped", "not_running"]

    @pytest.mark.asyncio
    async def test_process_video(self, async_client, test_video_path):
        """Test video processing endpoint."""
        request_data = {
            "video_path": str(test_video_path),
            "detect_objects": True,
            "detect_gestures": True,
            "save_output": False
        }

        response = await async_client.post("/detect/video", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "processing"

    @pytest.mark.asyncio
    async def test_websocket_stream(self, mocker):
        """Test WebSocket streaming endpoint."""
        from fastapi.testclient import TestClient

        # Mock camera and detection service
        mock_camera = mocker.MagicMock()
        mock_camera.start = AsyncMock()
        mock_camera.stop = AsyncMock()

        async def mock_frames():
            for i in range(2):
                yield Frame(
                    data=np.zeros((480, 640, 3)),
                    width=640,
                    height=480,
                    frame_number=i
                )

        mock_camera.get_frames = mock_frames

        mock_service = mocker.patch("src.presentation.api.detection_service")

        async def mock_stream(source):
            for i in range(2):
                result = mocker.MagicMock()
                result.to_dict.return_value = {
                    "frame_id": f"frame_{i}",
                    "frame_number": i,
                    "object_detections": [],
                    "gesture_detections": []
                }
                yield result

        mock_service.process_stream = mock_stream

        mocker.patch("src.presentation.api.CameraSource", return_value=mock_camera)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/stream") as websocket:
                # Receive first frame result
                data = websocket.receive_json()
                assert "frame_id" in data
                assert data["frame_id"] == "frame_0"

    def test_download_models(self, client, mocker):
        """Test model download endpoint."""
        # Mock YOLO download
        mock_yolo = mocker.patch("src.presentation.api.YOLO")
        mock_model = mocker.MagicMock()
        mock_yolo.return_value = mock_model

        response = client.get("/models/download")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["downloaded", "exists"]


class TestAPIErrorHandling:
    """Test API error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_detect_invalid_image(self, client):
        """Test detection with invalid image."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/detect/image", files=files)
        assert response.status_code == 400

    def test_process_nonexistent_video(self, client):
        """Test processing non-existent video."""
        request_data = {
            "video_path": "/nonexistent/video.mp4",
            "detect_objects": True,
            "detect_gestures": True
        }

        response = client.post("/detect/video", json=request_data)
        assert response.status_code == 404

    def test_camera_capture_no_camera(self, client, mocker):
        """Test camera capture when camera is not available."""
        mocker.patch("src.presentation.api.camera_source", None)
        mock_camera_class = mocker.patch("src.presentation.api.CameraSource")
        mock_camera = mocker.MagicMock()
        mock_camera.is_available.return_value = False
        mock_camera.start = AsyncMock(side_effect=RuntimeError("Camera not found"))
        mock_camera_class.return_value = mock_camera

        response = client.get("/camera/capture")
        assert response.status_code == 503