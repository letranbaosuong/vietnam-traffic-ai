"""FastAPI application and API endpoints."""

import asyncio
import io
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

from ..config.settings import settings
from ..domain.entities import Frame, DetectionSource
from ..application.yolo_detector import YOLODetector
from ..application.gesture_detector import TrafficGestureDetector
from ..application.detection_service import DetectionService
from ..infrastructure.camera_source import CameraSource, PiCameraSource
from ..infrastructure.video_source import VideoFileSource
from ..infrastructure.performance_monitor import PerformanceMonitor
from ..infrastructure.visualization import DetectionVisualizer


logger = logging.getLogger(__name__)


# Pydantic models for API
class DetectionRequest(BaseModel):
    """Request model for detection."""
    detect_objects: bool = True
    detect_gestures: bool = True
    visualize: bool = True


class DetectionSettings(BaseModel):
    """Model for updating detection settings."""
    yolo_confidence: Optional[float] = None
    yolo_iou: Optional[float] = None
    gesture_confidence: Optional[float] = None
    frame_skip: Optional[int] = None
    parallel_processing: Optional[bool] = None


class VideoProcessRequest(BaseModel):
    """Request model for video processing."""
    video_path: str
    detect_objects: bool = True
    detect_gestures: bool = True
    save_output: bool = False
    output_path: Optional[str] = None


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
object_detector: Optional[YOLODetector] = None
gesture_detector: Optional[TrafficGestureDetector] = None
detection_service: Optional[DetectionService] = None
performance_monitor: Optional[PerformanceMonitor] = None
visualizer: Optional[DetectionVisualizer] = None
camera_source: Optional[CameraSource] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global object_detector, gesture_detector, detection_service
    global performance_monitor, visualizer

    logger.info("Initializing Traffic Detection System...")

    # Initialize detectors
    if settings.enable_object_detection:
        object_detector = YOLODetector(
            model_name=settings.yolo_model,
            confidence_threshold=settings.yolo_confidence,
            iou_threshold=settings.yolo_iou,
            use_half_precision=settings.use_half_precision,
            max_detections=settings.max_detections_per_frame
        )

    if settings.enable_gesture_detection:
        gesture_detector = TrafficGestureDetector(
            min_detection_confidence=settings.gesture_confidence,
            min_tracking_confidence=settings.gesture_tracking_confidence,
            model_complexity=0  # Use lite model
        )

    # Initialize performance monitor
    if settings.enable_performance_monitoring:
        performance_monitor = PerformanceMonitor(
            window_size=settings.performance_window_size
        )

    # Initialize detection service
    detection_service = DetectionService(
        object_detector=object_detector,
        gesture_detector=gesture_detector,
        performance_monitor=performance_monitor,
        frame_skip=settings.frame_skip,
        parallel_processing=settings.parallel_processing
    )

    # Initialize visualizer
    visualizer = DetectionVisualizer()

    # Initialize the service
    await detection_service.initialize()

    logger.info("Traffic Detection System initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global detection_service, camera_source

    logger.info("Shutting down Traffic Detection System...")

    if camera_source:
        await camera_source.stop()

    if detection_service:
        await detection_service.cleanup()

    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "platform": settings.platform_name,
        "is_raspberry_pi": settings.is_raspberry_pi,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "object_detection": object_detector is not None,
            "gesture_detection": gesture_detector is not None,
            "performance_monitoring": performance_monitor is not None
        }
    }


@app.post("/detect/image")
async def detect_in_image(
    file: UploadFile = File(...),
    detect_objects: bool = True,
    detect_gestures: bool = True,
    visualize: bool = True
):
    """
    Detect objects and gestures in an uploaded image.

    Args:
        file: Image file
        detect_objects: Whether to detect objects
        detect_gestures: Whether to detect gestures
        visualize: Whether to return visualized image

    Returns:
        Detection results and optionally visualized image
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Create frame
        frame = Frame(
            data=image,
            width=image.shape[1],
            height=image.shape[0],
            source=DetectionSource.IMAGE_FILE
        )

        # Process frame
        result = await detection_service.process_frame(
            frame,
            detect_objects=detect_objects,
            detect_gestures=detect_gestures
        )

        response_data = result.to_dict()

        # Visualize if requested
        if visualize and visualizer:
            output_image = image.copy()
            output_image = visualizer.draw_detections(output_image, result.object_detections)
            output_image = visualizer.draw_gestures(output_image, result.gesture_detections)

            if performance_monitor:
                metrics = performance_monitor.get_metrics()
                output_image = visualizer.draw_metrics(output_image, metrics)

            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', output_image)
            image_bytes = buffer.tobytes()

            # Return image with detections drawn
            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type="image/jpeg",
                headers={
                    "X-Detection-Count": str(result.total_detections),
                    "X-Processing-Time": f"{result.processing_time_ms:.2f}ms"
                }
            )

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a video file for detections.

    Args:
        request: Video processing request

    Returns:
        Task ID for tracking progress
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")

    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    # Generate task ID
    task_id = f"video_{int(time.time())}"

    # Process video in background
    background_tasks.add_task(
        process_video_task,
        task_id,
        video_path,
        request.detect_objects,
        request.detect_gestures,
        request.save_output,
        request.output_path
    )

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Video processing started"
    }


async def process_video_task(
    task_id: str,
    video_path: Path,
    detect_objects: bool,
    detect_gestures: bool,
    save_output: bool,
    output_path: Optional[str]
):
    """Background task for video processing."""
    try:
        video_source = VideoFileSource(video_path)
        await video_source.start()

        output_writer = None
        if save_output and output_path:
            # Setup video writer
            props = video_source.get_properties()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                video_source.original_fps,
                (video_source.width, video_source.height)
            )

        # Process video
        async for result in detection_service.process_stream(
            video_source,
            detect_objects=detect_objects,
            detect_gestures=detect_gestures
        ):
            if save_output and output_writer and visualizer:
                # Get frame and visualize
                frame = await video_source.get_frame()
                if frame and frame.data is not None:
                    output_frame = visualizer.draw_detections(frame.data, result.object_detections)
                    output_frame = visualizer.draw_gestures(output_frame, result.gesture_detections)
                    output_writer.write(output_frame)

        if output_writer:
            output_writer.release()

        await video_source.stop()
        logger.info(f"Video processing completed: {task_id}")

    except Exception as e:
        logger.error(f"Error processing video {task_id}: {e}")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera stream processing.

    Streams detection results in real-time.
    """
    await websocket.accept()
    global camera_source

    try:
        # Initialize camera if not already done
        if camera_source is None:
            if settings.use_pi_camera and settings.is_raspberry_pi:
                camera_source = PiCameraSource(
                    width=settings.camera_width,
                    height=settings.camera_height,
                    fps=settings.camera_fps
                )
            else:
                camera_source = CameraSource(
                    camera_index=settings.camera_index,
                    width=settings.camera_width,
                    height=settings.camera_height,
                    fps=settings.camera_fps,
                    buffer_size=settings.camera_buffer_size
                )

        await camera_source.start()

        # Process stream
        async for result in detection_service.process_stream(camera_source):
            # Send results to client
            await websocket.send_json(result.to_dict())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1000)
    finally:
        if camera_source:
            await camera_source.stop()


@app.get("/camera/start")
async def start_camera():
    """Start camera capture."""
    global camera_source

    if camera_source and camera_source.is_available():
        return {"status": "already_running"}

    try:
        if settings.use_pi_camera and settings.is_raspberry_pi:
            camera_source = PiCameraSource(
                width=settings.camera_width,
                height=settings.camera_height,
                fps=settings.camera_fps
            )
        else:
            camera_source = CameraSource(
                camera_index=settings.camera_index,
                width=settings.camera_width,
                height=settings.camera_height,
                fps=settings.camera_fps
            )

        await camera_source.start()
        return {"status": "started", "properties": camera_source.get_properties()}

    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/camera/stop")
async def stop_camera():
    """Stop camera capture."""
    global camera_source

    if camera_source:
        await camera_source.stop()
        camera_source = None
        return {"status": "stopped"}

    return {"status": "not_running"}


@app.get("/camera/capture")
async def capture_frame():
    """Capture a single frame from camera."""
    global camera_source

    if not camera_source or not camera_source.is_available():
        # Try to start camera
        await start_camera()

    if camera_source:
        frame = await camera_source.get_frame()
        if frame and frame.data is not None:
            # Process frame
            result = await detection_service.process_frame(frame)

            # Visualize
            output_image = frame.data
            if visualizer:
                output_image = visualizer.draw_detections(output_image, result.object_detections)
                output_image = visualizer.draw_gestures(output_image, result.gesture_detections)

            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', output_image)
            image_bytes = buffer.tobytes()

            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type="image/jpeg"
            )

    raise HTTPException(status_code=503, detail="Camera not available")


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    if not performance_monitor:
        return {"status": "monitoring_disabled"}

    metrics = performance_monitor.get_metrics()
    return metrics.to_dict()


@app.post("/settings")
async def update_settings(settings_update: DetectionSettings):
    """Update detection settings."""
    global detection_service, object_detector, gesture_detector

    try:
        # Update YOLO settings
        if object_detector:
            object_detector.update_thresholds(
                confidence_threshold=settings_update.yolo_confidence,
                iou_threshold=settings_update.yolo_iou
            )

        # Update service settings
        if detection_service:
            service_settings = {}
            if settings_update.frame_skip is not None:
                service_settings["frame_skip"] = settings_update.frame_skip
            if settings_update.parallel_processing is not None:
                service_settings["parallel_processing"] = settings_update.parallel_processing

            if service_settings:
                detection_service.update_settings(service_settings)

        return {"status": "updated", "settings": settings_update.dict(exclude_none=True)}

    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/settings")
async def get_settings():
    """Get current settings."""
    current_settings = {
        "yolo_confidence": settings.yolo_confidence,
        "yolo_iou": settings.yolo_iou,
        "gesture_confidence": settings.gesture_confidence,
        "frame_skip": settings.frame_skip,
        "parallel_processing": settings.parallel_processing,
        "platform": settings.platform_name,
        "is_raspberry_pi": settings.is_raspberry_pi,
        "use_gpu": settings.use_gpu
    }

    if object_detector:
        current_settings["yolo_model_info"] = object_detector.get_model_info()

    if gesture_detector:
        current_settings["gesture_model_info"] = gesture_detector.get_model_info()

    return current_settings


@app.get("/models/download")
async def download_models():
    """Download required models."""
    try:
        # Download YOLO model if not exists
        from ultralytics import YOLO
        model_path = settings.models_dir / settings.yolo_model
        if not model_path.exists():
            logger.info(f"Downloading {settings.yolo_model}...")
            model = YOLO(settings.yolo_model)
            logger.info("Model downloaded successfully")
            return {"status": "downloaded", "model": settings.yolo_model}
        else:
            return {"status": "exists", "model": settings.yolo_model}

    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        raise HTTPException(status_code=500, detail=str(e))