"""YOLO-based object detection implementation."""

import asyncio
import time
from typing import List, Optional, Dict, Any
import logging
import numpy as np
from pathlib import Path

from ultralytics import YOLO
import torch

from ..domain.entities import Detection, Frame, BoundingBox, ObjectClass
from ..domain.interfaces import IObjectDetector


logger = logging.getLogger(__name__)


class YOLODetector(IObjectDetector):
    """YOLO-based object detector optimized for traffic scenarios."""

    # Traffic-related COCO classes
    TRAFFIC_CLASSES = {
        0: ObjectClass.PERSON,
        1: ObjectClass.BICYCLE,
        2: ObjectClass.CAR,
        3: ObjectClass.MOTORCYCLE,
        5: ObjectClass.BUS,
        7: ObjectClass.TRUCK,
        9: ObjectClass.TRAFFIC_LIGHT,
        11: ObjectClass.STOP_SIGN,
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        use_half_precision: bool = False,
        max_detections: int = 100,
        model_path: Optional[Path] = None
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLO model name (yolov8n.pt for nano version)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to use (cpu, cuda, mps, or None for auto)
            use_half_precision: Use FP16 for inference (faster on GPU)
            max_detections: Maximum number of detections per frame
            model_path: Custom model path
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_half_precision = use_half_precision
        self.max_detections = max_detections
        self.model_path = model_path
        self.model: Optional[YOLO] = None

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"YOLODetector initialized with device: {self.device}")

    async def initialize(self) -> None:
        """Initialize the detector and load model."""
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            logger.info(f"YOLO model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def _load_model(self) -> None:
        """Load YOLO model (sync operation)."""
        model_path = self.model_path or self.model_name
        self.model = YOLO(model_path)

        # Move model to device
        if self.device != "cpu":
            self.model.to(self.device)

        # Enable half precision if requested and on GPU
        if self.use_half_precision and self.device in ["cuda", "mps"]:
            self.model.model.half()

        # Warm up the model with a dummy inference
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(
            dummy_img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

    async def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame

        Returns:
            List of detected objects
        """
        if self.model is None:
            await self.initialize()

        if not frame.is_valid:
            logger.warning("Invalid frame provided for detection")
            return []

        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                None,
                self._detect_sync,
                frame.data
            )
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _detect_sync(self, image: np.ndarray) -> List[Detection]:
        """
        Synchronous detection method.

        Args:
            image: Input image as numpy array

        Returns:
            List of detections
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
            stream=False
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes)):
                # Extract detection information
                xyxy = boxes.xyxy[i]
                conf = boxes.conf[i]
                cls = int(boxes.cls[i])

                # Filter for traffic-related classes
                if cls in self.TRAFFIC_CLASSES:
                    object_class = self.TRAFFIC_CLASSES[cls]
                else:
                    # Skip non-traffic related detections
                    continue

                # Create detection object
                detection = Detection(
                    class_name=object_class.value,
                    confidence=float(conf),
                    bounding_box=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3])
                    ),
                    metadata={
                        "model": self.model_name,
                        "class_id": cls
                    }
                )
                detections.append(detection)

        return detections

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model is not None:
            # Clear model from memory
            del self.model
            self.model = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

            logger.info("YOLO detector cleaned up")

    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes."""
        return [cls.value for cls in self.TRAFFIC_CLASSES.values()]

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> None:
        """
        Update detection thresholds.

        Args:
            confidence_threshold: New confidence threshold
            iou_threshold: New IOU threshold
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            logger.info(f"Updated IOU threshold to {iou_threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "half_precision": self.use_half_precision,
            "supported_classes": self.get_supported_classes()
        }