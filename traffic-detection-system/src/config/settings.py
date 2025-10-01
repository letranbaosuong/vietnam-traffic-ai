"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional, List
from pathlib import Path
import platform


class AppSettings(BaseSettings):
    """Main application settings."""

    # Application info
    app_name: str = "Traffic Detection System"
    app_version: str = "1.0.0"
    debug: bool = False

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

    # Detection Settings
    enable_object_detection: bool = True
    enable_gesture_detection: bool = True
    parallel_processing: bool = True
    frame_skip: int = 0  # Process every frame by default

    # Model Settings
    yolo_model: str = "yolov8n.pt"  # Nano version for speed
    yolo_confidence: float = 0.5
    yolo_iou: float = 0.45
    gesture_confidence: float = 0.7
    gesture_tracking_confidence: float = 0.5
    max_detections_per_frame: int = 100

    # Platform-specific Settings
    platform_name: str = Field(default_factory=platform.system)
    is_raspberry_pi: bool = Field(
        default_factory=lambda: platform.system() == "Linux"
        and platform.machine().startswith("arm")
    )
    use_gpu: bool = True
    use_half_precision: bool = False  # FP16 for faster inference

    # Camera Settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_buffer_size: int = 1
    use_pi_camera: bool = False  # Use Pi Camera module if available

    # Performance Settings
    performance_window_size: int = 100
    enable_performance_monitoring: bool = True
    log_performance_interval: int = 30  # seconds

    # Cache Settings
    enable_model_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 5  # Maximum cached models

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "models")
    upload_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "uploads")
    cache_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "cache")

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[Path] = None

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        self._apply_platform_optimizations()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.models_dir, self.upload_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _apply_platform_optimizations(self):
        """Apply platform-specific optimizations."""
        if self.is_raspberry_pi:
            # Raspberry Pi optimizations
            self.use_gpu = False  # No GPU on Pi
            self.frame_skip = 1  # Skip every other frame for performance
            self.yolo_model = "yolov8n.pt"  # Ensure nano model
            self.camera_width = 640
            self.camera_height = 480
            self.max_detections_per_frame = 50
            if platform.machine() == "armv7l":
                # Pi 4 or older
                self.parallel_processing = False  # Save CPU
        elif self.platform_name == "Darwin":
            # macOS optimizations
            import torch
            if torch.backends.mps.is_available():
                self.use_gpu = True
                # MPS doesn't support half precision well
                self.use_half_precision = False
        elif self.platform_name == "Linux":
            # Linux optimizations
            import torch
            if torch.cuda.is_available():
                self.use_gpu = True
                self.use_half_precision = True  # Use FP16 on CUDA


class RaspberryPiSettings(BaseSettings):
    """Raspberry Pi specific settings."""

    # GPIO Settings
    enable_gpio_indicators: bool = False
    gpio_detection_pin: int = 17  # Pin for detection indicator LED
    gpio_gesture_pin: int = 27    # Pin for gesture indicator LED

    # Pi Camera Settings
    picamera_sensor_mode: int = 0
    picamera_exposure_mode: str = "auto"
    picamera_awb_mode: str = "auto"

    # Power Management
    enable_throttling_monitor: bool = True
    throttle_temp_threshold: float = 70.0  # Celsius
    reduce_fps_on_throttle: bool = True

    model_config = ConfigDict(
        env_prefix="RPI_",
        case_sensitive=False
    )


class DevelopmentSettings(AppSettings):
    """Development environment settings."""

    debug: bool = True
    log_level: str = "DEBUG"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]


class ProductionSettings(AppSettings):
    """Production environment settings."""

    debug: bool = False
    log_level: str = "WARNING"
    use_half_precision: bool = True
    enable_performance_monitoring: bool = True
    cors_origins: List[str] = []  # Configure based on deployment


def get_settings(env: str = "development") -> AppSettings:
    """
    Get settings based on environment.

    Args:
        env: Environment name (development, production, etc.)

    Returns:
        Settings instance
    """
    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    else:
        return AppSettings()


# Global settings instance
settings = get_settings()