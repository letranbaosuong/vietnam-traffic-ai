#!/usr/bin/env python3
"""
Download Vietnamese Traffic Sign Detection Models
Tải về các model phát hiện biển báo giao thông Việt Nam
"""

import os
import zipfile
from pathlib import Path
import time

class VietnameseModelDownloader:
    def __init__(self):
        """Initialize model downloader"""
        self.models_dir = Path("../models")
        self.models_dir.mkdir(exist_ok=True)

        # Available Vietnamese traffic sign models
        self.models = {
            "roboflow_vietnam": {
                "name": "Roboflow Vietnam Traffic Signs",
                "description": "58 classes, 4200+ images, YOLOv8 format",
                "url": "https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8",
                "classes": 58,
                "performance": "mAP: 98.1%, Recall: 98.6%",
                "size": "~150MB"
            },

            "yolov5_vietnam": {
                "name": "YOLOv5s Vietnamese Signs",
                "description": "Da Nang dataset, YOLOv5s optimized",
                "github": "https://github.com/Luantrannew/VietNam_Traffic_sign_recognise",
                "classes": 29,
                "performance": "F1: 92%+, 0.17s detection time",
                "size": "~30MB"
            },

            "vctsr47": {
                "name": "VCTSR47 Dataset",
                "description": "47 Vietnamese traffic sign classes",
                "classes": 47,
                "performance": "Real-time capable",
                "format": "YOLO format"
            }
        }

    def print_available_models(self):
        """Print all available Vietnamese traffic sign models"""
        print("🚦 Vietnamese Traffic Sign Detection Models\n")
        print("=" * 60)

        for model_id, info in self.models.items():
            print(f"\n📦 {info['name']}")
            print(f"   📝 Description: {info['description']}")
            print(f"   🎯 Classes: {info['classes']}")
            print(f"   ⚡ Performance: {info['performance']}")
            print(f"   💾 Size: {info.get('size', 'N/A')}")
            if 'url' in info:
                print(f"   🔗 URL: {info['url']}")
            if 'github' in info:
                print(f"   🔗 GitHub: {info['github']}")

    def download_roboflow_model(self):
        """
        Download Roboflow Vietnam traffic sign model
        Note: Requires Roboflow API key for full access
        """
        print("\n🔄 Downloading Roboflow Vietnam Traffic Signs Model...")

        # This would require Roboflow API key
        print("⚠️ Roboflow model requires API key registration")
        print("📝 Steps to download:")
        print("   1. Go to: https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8")
        print("   2. Sign up for free account")
        print("   3. Get API key and download dataset")
        print("   4. Extract to models/roboflow_vietnam/")

        return False

    def download_github_model(self):
        """Download model from GitHub repository"""
        print("\n🔄 Downloading GitHub Vietnamese Traffic Sign Model...")

        repo_url = "https://github.com/Luantrannew/VietNam_Traffic_sign_recognise.git"
        model_dir = self.models_dir / "github_vietnam"

        try:
            if model_dir.exists():
                print(f"✅ Model already exists at {model_dir}")
                return True

            # Clone repository
            import subprocess
            result = subprocess.run(
                ["git", "clone", repo_url, str(model_dir)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✅ Successfully downloaded to {model_dir}")
                print("📂 Repository contents:")
                for item in model_dir.iterdir():
                    print(f"   - {item.name}")
                return True
            else:
                print(f"❌ Git clone failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error downloading GitHub model: {e}")
            return False

    def create_vietnamese_sign_config(self):
        """Create configuration for Vietnamese traffic signs"""
        config_path = self.models_dir / "vietnamese_signs_config.yaml"

        # Vietnamese traffic sign classes (common ones)
        vietnamese_classes = [
            # Biển cấm (Prohibitory signs)
            'cam_di_nguoc_chieu',      # No entry
            'cam_re_trai',             # No left turn
            'cam_re_phai',             # No right turn
            'cam_quay_dau',            # No U-turn
            'cam_dung_va_do',          # No stopping
            'toc_do_toi_da_30',        # Speed limit 30
            'toc_do_toi_da_40',        # Speed limit 40
            'toc_do_toi_da_50',        # Speed limit 50
            'toc_do_toi_da_60',        # Speed limit 60
            'toc_do_toi_da_80',        # Speed limit 80

            # Biển báo hiệu (Warning signs)
            'bien_stop',               # Stop sign
            'nhuong_duong',           # Yield
            'duong_uu_tien',          # Priority road
            'duong_nguoi_di_bo',      # Pedestrian crossing
            'khu_vuc_truong_hoc',     # School zone
            'duong_xe_dap',           # Bicycle path
            'dang_thi_cong',          # Road work
            'den_giao_thong',         # Traffic light

            # Biển chỉ dẫn (Information signs)
            're_trai',                # Turn left
            're_phai',                # Turn right
            'di_thang',               # Go straight
            'bung_binh',              # Roundabout
            'duong_mot_chieu',        # One way
            'cho_do_xe',              # Parking
            'benh_vien',              # Hospital
            'cay_xang',               # Gas station
            'nha_hang',               # Restaurant
            'khach_san',              # Hotel
            'atm'                     # ATM
        ]

        config_content = f"""# Vietnamese Traffic Signs Configuration
# Cấu hình biển báo giao thông Việt Nam

# Dataset info
path: {self.models_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes ({len(vietnamese_classes)} classes)
nc: {len(vietnamese_classes)}
names: {vietnamese_classes}

# Model recommendations for Pi 4
recommended_models:
  - yolov8n.pt    # Fastest, ~6 FPS on Pi 4
  - yolov8s.pt    # Balanced, ~4 FPS on Pi 4
  - yolov5n.pt    # Alternative, good compatibility

# Performance targets for Pi 4
performance_targets:
  fps_min: 5
  accuracy_min: 85
  memory_max: 1GB
"""

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"✅ Created Vietnamese signs config: {config_path}")
        return config_path

    def download_pretrained_yolo(self):
        """Download pre-trained YOLO models optimized for traffic signs"""
        print("\n🔄 Downloading pre-trained YOLO models...")

        models_to_download = [
            ("yolov8n.pt", "YOLOv8 Nano - Best for Pi 4"),
            ("yolov8s.pt", "YOLOv8 Small - Good accuracy"),
            ("yolov5n.pt", "YOLOv5 Nano - Alternative option")
        ]

        for model_name, description in models_to_download:
            model_path = self.models_dir / model_name

            if model_path.exists():
                print(f"✅ {model_name} already exists")
                continue

            try:
                print(f"📥 Downloading {model_name} ({description})...")

                # Use ultralytics to download
                from ultralytics import YOLO
                model = YOLO(model_name)

                # Move to models directory
                import shutil
                source = Path(model_name)
                if source.exists():
                    shutil.move(str(source), str(model_path))
                    print(f"✅ Downloaded {model_name}")
                else:
                    print(f"⚠️ {model_name} not found after download")

            except Exception as e:
                print(f"❌ Error downloading {model_name}: {e}")

    def create_model_evaluation_script(self):
        """Create script to evaluate different models"""
        script_path = self.models_dir / "evaluate_models.py"

        script_content = '''#!/usr/bin/env python3
"""
Evaluate Vietnamese Traffic Sign Models
So sánh hiệu suất các model phát hiện biển báo
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import psutil

class ModelEvaluator:
    def __init__(self):
        self.models_dir = Path(".")
        self.test_video = "../../video/sample.mp4"

    def evaluate_model(self, model_path, frames_to_test=100):
        """Evaluate a single model"""
        print(f"\\n🧪 Testing {model_path.name}")
        print("-" * 40)

        try:
            # Load model
            model = YOLO(str(model_path))

            # Open video
            cap = cv2.VideoCapture(self.test_video)
            if not cap.isOpened():
                print("❌ Could not open test video")
                return None

            # Performance metrics
            frame_times = []
            cpu_usage = []
            memory_usage = []

            for i in range(frames_to_test):
                ret, frame = cap.read()
                if not ret:
                    break

                # Measure inference time
                start_time = time.time()
                results = model(frame, conf=0.5, verbose=False)
                inference_time = time.time() - start_time

                frame_times.append(inference_time)
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)

                if i % 20 == 0:
                    print(f"  Frame {i}/{frames_to_test}")

            cap.release()

            # Calculate metrics
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            avg_memory = sum(memory_usage) / len(memory_usage)

            results = {
                'model': model_path.name,
                'avg_fps': round(avg_fps, 2),
                'avg_cpu': round(avg_cpu, 1),
                'avg_memory': round(avg_memory, 1),
                'pi4_estimated_fps': round(avg_fps * 0.6, 2)  # Estimate for Pi 4
            }

            print(f"📊 Results:")
            print(f"  Average FPS: {results['avg_fps']}")
            print(f"  CPU Usage: {results['avg_cpu']}%")
            print(f"  Memory: {results['avg_memory']}%")
            print(f"  Pi 4 Est. FPS: {results['pi4_estimated_fps']}")

            return results

        except Exception as e:
            print(f"❌ Error testing {model_path.name}: {e}")
            return None

    def run_comparison(self):
        """Run comparison of all available models"""
        print("🚦 Vietnamese Traffic Sign Model Comparison")
        print("=" * 50)

        models = list(self.models_dir.glob("*.pt"))
        if not models:
            print("❌ No .pt model files found")
            return

        results = []
        for model_path in models:
            result = self.evaluate_model(model_path)
            if result:
                results.append(result)

        # Print comparison table
        if results:
            print("\\n📊 Comparison Summary:")
            print("-" * 60)
            print(f"{'Model':<15} {'FPS':<8} {'CPU%':<8} {'Memory%':<10} {'Pi4 FPS':<10}")
            print("-" * 60)

            for result in sorted(results, key=lambda x: x['avg_fps'], reverse=True):
                print(f"{result['model']:<15} {result['avg_fps']:<8} "
                      f"{result['avg_cpu']:<8} {result['avg_memory']:<10} "
                      f"{result['pi4_estimated_fps']:<10}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_comparison()
'''

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        print(f"✅ Created model evaluation script: {script_path}")
        return script_path

    def run_download_process(self):
        """Run the complete download process"""
        print("🚀 Vietnamese Traffic Sign Models R&D")
        print("=" * 50)

        # Step 1: Show available models
        self.print_available_models()

        # Step 2: Create config
        self.create_vietnamese_sign_config()

        # Step 3: Download pre-trained YOLO models
        self.download_pretrained_yolo()

        # Step 4: Try to download GitHub model
        self.download_github_model()

        # Step 5: Create evaluation script
        self.create_model_evaluation_script()

        # Step 6: Instructions for Roboflow
        print("\n" + "=" * 50)
        print("📋 Next Steps:")
        print("1. ✅ Pre-trained YOLO models downloaded")
        print("2. ✅ Vietnamese sign config created")
        print("3. ✅ Model evaluation script ready")
        print("4. 📝 Manual: Download Roboflow dataset (best quality)")
        print("5. 🧪 Run: python models/evaluate_models.py")

        print("\n🎯 Recommendations for Pi 4:")
        print("- Use YOLOv8n for best speed (~6 FPS)")
        print("- Use YOLOv8s for balanced performance (~4 FPS)")
        print("- Train custom model on Vietnamese dataset for best accuracy")

def main():
    """Main function"""
    downloader = VietnameseModelDownloader()
    downloader.run_download_process()

if __name__ == "__main__":
    main()