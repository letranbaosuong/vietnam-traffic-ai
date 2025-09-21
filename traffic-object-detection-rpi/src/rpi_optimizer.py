import os
import psutil
import subprocess
from typing import Dict, Any
import yaml


class RPiOptimizer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.is_rpi = self.detect_raspberry_pi()

    def detect_raspberry_pi(self) -> bool:
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Raspberry Pi' in model
        except:
            return False

    def optimize_system(self):
        if not self.is_rpi:
            print("Not running on Raspberry Pi, skipping system optimization")
            return

        os.environ['OMP_NUM_THREADS'] = str(self.config['rpi_optimization']['num_threads'])
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.config['rpi_optimization']['num_threads'])
        os.environ['MKL_NUM_THREADS'] = str(self.config['rpi_optimization']['num_threads'])

        try:
            subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'],
                         check=False, capture_output=True)
        except:
            pass

        try:
            gpu_mem = subprocess.run(['vcgencmd', 'get_mem', 'gpu'],
                                   capture_output=True, text=True)
            print(f"GPU Memory: {gpu_mem.stdout.strip()}")
        except:
            pass

    def get_system_info(self) -> Dict[str, Any]:
        info = {
            'is_rpi': self.is_rpi,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3),
            'memory_available': psutil.virtual_memory().available / (1024**3),
            'memory_percent': psutil.virtual_memory().percent
        }

        if self.is_rpi:
            try:
                temp = subprocess.run(['vcgencmd', 'measure_temp'],
                                    capture_output=True, text=True)
                info['temperature'] = temp.stdout.strip()
            except:
                pass

            try:
                throttled = subprocess.run(['vcgencmd', 'get_throttled'],
                                         capture_output=True, text=True)
                info['throttled'] = throttled.stdout.strip()
            except:
                pass

        return info

    def monitor_resources(self) -> Dict[str, float]:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }

    def check_throttling(self) -> bool:
        if not self.is_rpi:
            return False

        try:
            result = subprocess.run(['vcgencmd', 'get_throttled'],
                                  capture_output=True, text=True)
            throttled_value = int(result.stdout.split('=')[1], 16)

            if throttled_value != 0:
                print("WARNING: Raspberry Pi is throttling!")
                if throttled_value & 0x1:
                    print("  - Under-voltage detected")
                if throttled_value & 0x2:
                    print("  - ARM frequency capped")
                if throttled_value & 0x4:
                    print("  - Currently throttled")
                if throttled_value & 0x8:
                    print("  - Soft temperature limit active")
                return True
        except:
            pass

        return False

    def optimize_opencv(self):
        cv2_threads = self.config['rpi_optimization']['num_threads']
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

        import cv2
        cv2.setNumThreads(cv2_threads)
        cv2.setUseOptimized(True)

        print(f"OpenCV optimizations enabled:")
        print(f"  - Threads: {cv2_threads}")
        print(f"  - Optimized: {cv2.useOptimized()}")

    def get_optimization_tips(self) -> list:
        tips = []

        info = self.get_system_info()

        if info['memory_percent'] > 80:
            tips.append("High memory usage detected. Consider:")
            tips.append("  - Reducing video resolution")
            tips.append("  - Decreasing batch size")
            tips.append("  - Using smaller model (YOLOv8n)")

        if info.get('cpu_percent', 0) > 90:
            tips.append("High CPU usage detected. Consider:")
            tips.append("  - Lowering FPS")
            tips.append("  - Using ONNX Runtime")
            tips.append("  - Enabling quantization")

        if self.is_rpi and self.check_throttling():
            tips.append("Throttling detected. Consider:")
            tips.append("  - Improving cooling (fan/heatsink)")
            tips.append("  - Using official power supply (5V 3A)")
            tips.append("  - Reducing workload")

        return tips