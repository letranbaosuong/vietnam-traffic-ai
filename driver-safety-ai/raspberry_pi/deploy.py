#!/usr/bin/env python3
"""
Deploy script cho Raspberry Pi 4
Triển khai hệ thống giám sát lái xe
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Import detection module
from src.detection.driver_monitor import DriverMonitor, BehaviorType

# GPIO pins configuration
LED_PINS = {
    'green': 17,   # Normal status
    'yellow': 27,  # Warning
    'red': 22      # Danger
}
BUZZER_PIN = 23


class RaspberryPiDeployment:
    """
    Deployment class cho Raspberry Pi
    """

    def __init__(
        self,
        model_path: str,
        use_picamera: bool = True,
        enable_gpio: bool = True,
        log_file: str = "driver_monitor.log"
    ):
        """
        Initialize deployment

        Args:
            model_path: Path to TFLite model
            use_picamera: Use Pi Camera instead of USB camera
            enable_gpio: Enable GPIO alerts
            log_file: Log file path
        """
        self.model_path = model_path
        self.use_picamera = use_picamera
        self.enable_gpio = enable_gpio

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.monitor = DriverMonitor(model_path=model_path)
        self.camera = None
        self.setup_camera()

        if self.enable_gpio:
            self.setup_gpio()

        self.logger.info("Raspberry Pi Deployment initialized")

    def setup_camera(self):
        """Setup camera (PiCamera or USB)"""
        if self.use_picamera:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.logger.info("PiCamera initialized")
            except Exception as e:
                self.logger.error(f"PiCamera init failed: {e}")
                self.logger.info("Falling back to USB camera")
                self.use_picamera = False
                self.camera = cv2.VideoCapture(0)
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.logger.info("USB camera initialized")

    def setup_gpio(self):
        """Setup GPIO pins for alerts"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Setup LED pins
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # Setup buzzer
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        self.buzzer_pwm = GPIO.PWM(BUZZER_PIN, 1000)

        self.logger.info("GPIO initialized")

    def update_alerts(self, behavior: BehaviorType, should_alert: bool):
        """
        Update LED and buzzer based on detection

        Args:
            behavior: Detected behavior
            should_alert: Alert flag
        """
        if not self.enable_gpio:
            return

        # Reset all LEDs
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)

        # Set appropriate LED
        if should_alert:
            GPIO.output(LED_PINS['red'], GPIO.HIGH)
            # Sound buzzer
            self.buzzer_pwm.start(50)  # 50% duty cycle
            time.sleep(0.5)
            self.buzzer_pwm.stop()
        elif behavior != BehaviorType.NORMAL:
            GPIO.output(LED_PINS['yellow'], GPIO.HIGH)
        else:
            GPIO.output(LED_PINS['green'], GPIO.HIGH)

    def get_frame(self) -> np.ndarray:
        """Get frame from camera"""
        if self.use_picamera:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.error("Failed to get frame from camera")
                return None
        return frame

    def run(self, headless: bool = False):
        """
        Run the monitoring system

        Args:
            headless: Run without display
        """
        self.logger.info("Starting driver monitoring system")

        fps_time = time.time()
        fps_counter = 0
        current_fps = 0

        # Statistics
        total_frames = 0
        dangerous_frames = 0
        session_start = time.time()

        try:
            while True:
                # Get frame
                frame = self.get_frame()
                if frame is None:
                    continue

                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # Analyze frame
                result = self.monitor.analyze_frame(frame)
                detection = result['detection']
                should_alert = result['should_alert']

                # Update statistics
                total_frames += 1
                if detection.is_dangerous:
                    dangerous_frames += 1

                # Update alerts
                self.update_alerts(detection.behavior, should_alert)

                # Log dangerous behavior
                if should_alert:
                    self.logger.warning(
                        f"ALERT: {detection.behavior.value} detected "
                        f"(confidence: {detection.confidence:.2%}, "
                        f"duration: {result['behavior_duration']:.1f}s)"
                    )

                # Display if not headless
                if not headless:
                    display_frame = self.monitor.draw_overlay(
                        frame, detection, should_alert
                    )

                    # Add system info
                    info_text = f"FPS: {current_fps} | "
                    info_text += f"Danger Rate: {(dangerous_frames/max(total_frames,1))*100:.1f}%"
                    cv2.putText(display_frame, info_text,
                              (10, frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 1)

                    cv2.imshow('Driver Monitoring', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"screenshot_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        self.logger.info(f"Screenshot saved: {filename}")

                # Log statistics every minute
                if total_frames % (30 * 60) == 0:  # Assuming 30 FPS
                    session_duration = time.time() - session_start
                    self.logger.info(
                        f"Session stats - Duration: {session_duration/60:.1f}m, "
                        f"Frames: {total_frames}, "
                        f"Danger rate: {(dangerous_frames/total_frames)*100:.1f}%"
                    )

        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")

        if self.use_picamera and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()

        if self.enable_gpio:
            # Turn off all LEDs
            for pin in LED_PINS.values():
                GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup()

        cv2.destroyAllWindows()
        self.logger.info("Cleanup complete")

    def run_benchmark(self):
        """Run performance benchmark"""
        self.logger.info("Running benchmark...")

        # Test model performance
        benchmark_results = self.monitor.benchmark(100)

        self.logger.info("Benchmark Results:")
        for key, value in benchmark_results.items():
            self.logger.info(f"  {key}: {value:.2f}")

        # Test camera FPS
        self.logger.info("\nTesting camera FPS...")
        start_time = time.time()
        frame_count = 0

        while frame_count < 100:
            frame = self.get_frame()
            if frame is not None:
                frame_count += 1

        elapsed = time.time() - start_time
        camera_fps = frame_count / elapsed

        self.logger.info(f"Camera FPS: {camera_fps:.2f}")

        # System info
        self.logger.info("\nSystem Information:")

        # CPU temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read()) / 1000
                self.logger.info(f"CPU Temperature: {cpu_temp:.1f}°C")
        except:
            pass

        # Memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.logger.info(f"Memory Usage: {memory.percent:.1f}%")
            self.logger.info(f"Available Memory: {memory.available / 1024 / 1024:.1f} MB")
        except ImportError:
            pass


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Driver Monitoring System for Raspberry Pi'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to TFLite model file'
    )
    parser.add_argument(
        '--usb-camera',
        action='store_true',
        help='Use USB camera instead of Pi Camera'
    )
    parser.add_argument(
        '--no-gpio',
        action='store_true',
        help='Disable GPIO alerts'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark and exit'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='driver_monitor.log',
        help='Log file path'
    )

    args = parser.parse_args()

    # Check model file
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Initialize deployment
    deployment = RaspberryPiDeployment(
        model_path=args.model,
        use_picamera=not args.usb_camera,
        enable_gpio=not args.no_gpio,
        log_file=args.log
    )

    if args.benchmark:
        deployment.run_benchmark()
    else:
        deployment.run(headless=args.headless)


if __name__ == "__main__":
    main()