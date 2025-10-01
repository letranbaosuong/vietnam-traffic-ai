"""Video file source implementation for processing video files."""

import asyncio
import logging
from typing import Optional, AsyncIterator, Dict, Any
from pathlib import Path
import cv2
import numpy as np

from ..domain.entities import Frame, DetectionSource
from ..domain.interfaces import IFrameSource


logger = logging.getLogger(__name__)


class VideoFileSource(IFrameSource):
    """Video file source for processing pre-recorded videos."""

    def __init__(
        self,
        video_path: Path,
        loop: bool = False,
        target_fps: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ):
        """
        Initialize video file source.

        Args:
            video_path: Path to video file
            loop: Whether to loop the video
            target_fps: Target FPS (None to use original)
            start_frame: Starting frame number
            end_frame: Ending frame number (None for entire video)
        """
        self.video_path = Path(video_path)
        self.loop = loop
        self.target_fps = target_fps
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_counter = 0
        self.total_frames = 0
        self.original_fps = 0.0
        self._running = False

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    async def start(self) -> None:
        """Start the video source."""
        if self._running:
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_video)
            self._running = True
            logger.info(f"Video source started: {self.video_path.name}")
        except Exception as e:
            logger.error(f"Failed to start video source: {e}")
            raise

    def _initialize_video(self) -> None:
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set start position if specified
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.frame_counter = self.start_frame

        # Determine actual FPS
        self.actual_fps = self.target_fps if self.target_fps else self.original_fps

        logger.info(
            f"Video initialized - Resolution: {self.width}x{self.height}, "
            f"FPS: {self.original_fps}, Total frames: {self.total_frames}"
        )

    async def stop(self) -> None:
        """Stop the video source."""
        if not self._running:
            return

        self._running = False

        if self.cap is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._release_video)

        logger.info("Video source stopped")

    def _release_video(self) -> None:
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    async def get_frame(self) -> Optional[Frame]:
        """
        Get a single frame from the video.

        Returns:
            Frame object or None if no frame available
        """
        if not self._running or self.cap is None:
            return None

        try:
            # Check if we've reached the end frame
            if self.end_frame and self.frame_counter >= self.end_frame:
                if self.loop:
                    # Reset to start frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    self.frame_counter = self.start_frame
                else:
                    return None

            # Capture frame in thread pool
            loop = asyncio.get_event_loop()
            frame_data = await loop.run_in_executor(None, self._capture_frame)

            if frame_data is not None:
                self.frame_counter += 1
                frame = Frame(
                    data=frame_data,
                    width=self.width,
                    height=self.height,
                    frame_number=self.frame_counter,
                    source=DetectionSource.VIDEO_FILE,
                    metadata={
                        "video_file": self.video_path.name,
                        "original_fps": self.original_fps,
                        "progress": f"{self.frame_counter}/{self.total_frames}"
                    }
                )
                return frame
            elif self.loop:
                # Reset to start if looping
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self.frame_counter = self.start_frame
                return await self.get_frame()

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")

        return None

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the video."""
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
        # Calculate frame delay for target FPS
        if self.actual_fps > 0:
            frame_delay = 1.0 / self.actual_fps
        else:
            frame_delay = 0.033  # Default to ~30 FPS

        while self._running:
            frame = await self.get_frame()
            if frame is not None:
                yield frame
                # Add delay to match target FPS
                await asyncio.sleep(frame_delay)
            elif not self.loop:
                # End of video and not looping
                break
            else:
                # Small delay before retrying
                await asyncio.sleep(0.001)

    def is_available(self) -> bool:
        """Check if video is available."""
        return self.cap is not None and self.cap.isOpened()

    def get_properties(self) -> Dict[str, Any]:
        """Get video properties."""
        if self.cap is None:
            return {"status": "not_initialized"}

        current_position = 0
        if self.cap:
            current_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        return {
            "status": "playing" if self._running else "stopped",
            "video_file": str(self.video_path),
            "resolution": f"{self.width}x{self.height}",
            "original_fps": self.original_fps,
            "target_fps": self.actual_fps,
            "total_frames": self.total_frames,
            "current_frame": current_position,
            "progress_percent": (current_position / self.total_frames * 100)
            if self.total_frames > 0
            else 0,
            "loop": self.loop,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame
        }

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if successful
        """
        if self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_counter = frame_number
            return True
        except Exception as e:
            logger.error(f"Failed to seek to frame {frame_number}: {e}")
            return False


class StreamSource(IFrameSource):
    """Source for network streams (RTSP, HTTP, etc.)."""

    def __init__(
        self,
        stream_url: str,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 5.0,
        buffer_size: int = 1
    ):
        """
        Initialize stream source.

        Args:
            stream_url: URL of the stream
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
            buffer_size: Frame buffer size
        """
        self.stream_url = stream_url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_counter = 0
        self._running = False

    async def start(self) -> None:
        """Start the stream source."""
        if self._running:
            return

        for attempt in range(self.reconnect_attempts):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._initialize_stream)
                self._running = True
                logger.info(f"Stream source started: {self.stream_url}")
                return
            except Exception as e:
                logger.warning(
                    f"Failed to connect to stream (attempt {attempt + 1}/"
                    f"{self.reconnect_attempts}): {e}"
                )
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay)

        raise RuntimeError(f"Failed to connect to stream after {self.reconnect_attempts} attempts")

    def _initialize_stream(self) -> None:
        """Initialize stream capture."""
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open stream: {self.stream_url}")

        # Set buffer size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Stream initialized - Resolution: {self.width}x{self.height}, FPS: {self.fps}"
        )

    async def stop(self) -> None:
        """Stop the stream source."""
        if not self._running:
            return

        self._running = False

        if self.cap is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._release_stream)

        logger.info("Stream source stopped")

    def _release_stream(self) -> None:
        """Release stream capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    async def get_frame(self) -> Optional[Frame]:
        """Get a single frame from the stream."""
        if not self._running or self.cap is None:
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
                    source=DetectionSource.STREAM,
                    metadata={
                        "stream_url": self.stream_url,
                        "fps": self.fps
                    }
                )
                return frame

        except Exception as e:
            logger.error(f"Failed to capture frame from stream: {e}")
            # Try to reconnect
            await self._reconnect()

        return None

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the stream."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the stream."""
        logger.info("Attempting to reconnect to stream...")
        await self.stop()
        await asyncio.sleep(self.reconnect_delay)
        try:
            await self.start()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def get_frames(self) -> AsyncIterator[Frame]:
        """Get frames as an async iterator."""
        while self._running:
            frame = await self.get_frame()
            if frame is not None:
                yield frame
            else:
                await asyncio.sleep(0.001)

    def is_available(self) -> bool:
        """Check if stream is available."""
        return self.cap is not None and self.cap.isOpened()

    def get_properties(self) -> Dict[str, Any]:
        """Get stream properties."""
        return {
            "status": "connected" if self._running else "disconnected",
            "stream_url": self.stream_url,
            "resolution": f"{self.width}x{self.height}" if self.cap else "unknown",
            "fps": self.fps if self.cap else 0,
            "frames_captured": self.frame_counter,
            "buffer_size": self.buffer_size
        }