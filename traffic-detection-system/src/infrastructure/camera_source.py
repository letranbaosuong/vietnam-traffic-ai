"""Camera source implementation for capturing frames."""

import asyncio
import logging
import platform
from typing import Optional, AsyncIterator, Dict, Any
from datetime import datetime
import cv2
import numpy as np

from ..domain.entities import Frame, DetectionSource
from ..domain.interfaces import IFrameSource


logger = logging.getLogger(__name__)


class CameraSource(IFrameSource):
    """Camera source for capturing frames from webcam or USB camera."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
        backend: Optional[int] = None
    ):
        """
        Initialize camera source.

        Args:
            camera_index: Camera device index
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            buffer_size: Frame buffer size
            backend: OpenCV backend (CAP_ANY, CAP_V4L2, CAP_AVFOUNDATION, etc.)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_counter = 0
        self._running = False

        # Select backend based on platform if not specified
        if backend is None:
            system = platform.system()
            if system == "Linux":
                self.backend = cv2.CAP_V4L2
            elif system == "Darwin":  # macOS
                self.backend = cv2.CAP_AVFOUNDATION
            elif system == "Windows":
                self.backend = cv2.CAP_DSHOW
            else:
                self.backend = cv2.CAP_ANY
        else:
            self.backend = backend

    async def start(self) -> None:
        """Start the camera source."""
        if self._running:
            return

        try:
            # Initialize camera in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_camera)
            self._running = True
            logger.info(f"Camera source started (index: {self.camera_index})")
        except Exception as e:
            logger.error(f"Failed to start camera source: {e}")
            raise

    def _initialize_camera(self) -> None:
        """Initialize the camera (sync operation)."""
        self.cap = cv2.VideoCapture(self.camera_index, self.backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Read actual properties (may differ from requested)
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera initialized - Resolution: {self.actual_width}x{self.actual_height}, "
            f"FPS: {self.actual_fps}"
        )

    async def stop(self) -> None:
        """Stop the camera source."""
        if not self._running:
            return

        self._running = False

        if self.cap is not None:
            # Release camera in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._release_camera)

        logger.info("Camera source stopped")

    def _release_camera(self) -> None:
        """Release the camera (sync operation)."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    async def get_frame(self) -> Optional[Frame]:
        """
        Get a single frame from the camera.

        Returns:
            Frame object or None if no frame available
        """
        if not self._running or self.cap is None:
            return None

        try:
            # Capture frame in thread pool
            loop = asyncio.get_event_loop()
            frame_data = await loop.run_in_executor(None, self._capture_frame)

            if frame_data is not None:
                self.frame_counter += 1
                frame = Frame(
                    data=frame_data,
                    width=self.actual_width,
                    height=self.actual_height,
                    frame_number=self.frame_counter,
                    source=DetectionSource.CAMERA,
                    metadata={
                        "camera_index": self.camera_index,
                        "backend": self.backend
                    }
                )
                return frame

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")

        return None

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera (sync operation)."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    async def get_frames(self) -> AsyncIterator[Frame]:
        """
        Get frames as an async iterator.

        Yields:
            Frame objects
        """
        while self._running:
            frame = await self.get_frame()
            if frame is not None:
                yield frame
            else:
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)

    def is_available(self) -> bool:
        """Check if camera is available."""
        return self.cap is not None and self.cap.isOpened()

    def get_properties(self) -> Dict[str, Any]:
        """Get camera properties."""
        if self.cap is None:
            return {"status": "not_initialized"}

        return {
            "status": "initialized" if self._running else "stopped",
            "camera_index": self.camera_index,
            "backend": self.backend,
            "requested_resolution": f"{self.width}x{self.height}",
            "actual_resolution": f"{self.actual_width}x{self.actual_height}",
            "requested_fps": self.fps,
            "actual_fps": self.actual_fps,
            "buffer_size": self.buffer_size,
            "frames_captured": self.frame_counter
        }


class PiCameraSource(IFrameSource):
    """Camera source for Raspberry Pi Camera (using picamera2)."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        sensor_mode: int = 0
    ):
        """
        Initialize Pi Camera source.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            sensor_mode: Camera sensor mode
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.sensor_mode = sensor_mode
        self.picam = None
        self.frame_counter = 0
        self._running = False

        # Only import picamera2 on Raspberry Pi
        if platform.system() == "Linux" and platform.machine().startswith("arm"):
            try:
                from picamera2 import Picamera2
                self.Picamera2 = Picamera2
                self.is_pi = True
            except ImportError:
                logger.warning("picamera2 not available, falling back to regular camera")
                self.is_pi = False
        else:
            self.is_pi = False

    async def start(self) -> None:
        """Start the Pi camera source."""
        if not self.is_pi:
            raise RuntimeError("PiCamera is not available on this platform")

        if self._running:
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_picamera)
            self._running = True
            logger.info("Pi Camera source started")
        except Exception as e:
            logger.error(f"Failed to start Pi Camera: {e}")
            raise

    def _initialize_picamera(self) -> None:
        """Initialize the Pi Camera."""
        self.picam = self.Picamera2()

        # Configure camera
        config = self.picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps}
        )
        self.picam.configure(config)
        self.picam.start()

    async def stop(self) -> None:
        """Stop the Pi camera source."""
        if not self._running:
            return

        self._running = False

        if self.picam is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._stop_picamera)

        logger.info("Pi Camera source stopped")

    def _stop_picamera(self) -> None:
        """Stop the Pi Camera."""
        if self.picam is not None:
            self.picam.stop()
            self.picam.close()
            self.picam = None

    async def get_frame(self) -> Optional[Frame]:
        """Get a single frame from Pi Camera."""
        if not self._running or self.picam is None:
            return None

        try:
            loop = asyncio.get_event_loop()
            frame_data = await loop.run_in_executor(None, self._capture_frame)

            if frame_data is not None:
                self.frame_counter += 1
                frame = Frame(
                    data=frame_data,
                    width=self.width,
                    height=self.height,
                    frame_number=self.frame_counter,
                    source=DetectionSource.CAMERA,
                    metadata={"camera_type": "picamera"}
                )
                return frame

        except Exception as e:
            logger.error(f"Failed to capture frame from Pi Camera: {e}")

        return None

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from Pi Camera."""
        if self.picam is None:
            return None

        array = self.picam.capture_array("main")
        # Convert RGB to BGR for OpenCV compatibility
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    async def get_frames(self) -> AsyncIterator[Frame]:
        """Get frames as an async iterator."""
        while self._running:
            frame = await self.get_frame()
            if frame is not None:
                yield frame
            else:
                await asyncio.sleep(0.001)

    def is_available(self) -> bool:
        """Check if Pi Camera is available."""
        return self.is_pi and self.picam is not None

    def get_properties(self) -> Dict[str, Any]:
        """Get Pi Camera properties."""
        return {
            "status": "initialized" if self._running else "stopped",
            "camera_type": "picamera",
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "sensor_mode": self.sensor_mode,
            "frames_captured": self.frame_counter,
            "is_available": self.is_pi
        }