#!/usr/bin/env python3
"""
Setup script để cài đặt môi trường cho Vietnam Traffic AI System
"""

import subprocess
import sys
import os
from pathlib import Path
import urllib.request
import yaml

def run_command(command, description):
    """Run shell command with description"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} hoàn thành!")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi {description.lower()}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Tương thích")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Cần Python 3.8+")
        return False

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment đã tồn tại")
        return True
    
    result = run_command("python -m venv venv", "Tạo virtual environment")
    return result is not None

def install_requirements():
    """Install required packages"""
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Cập nhật pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        result = run_command(f"{pip_cmd} install -r requirements.txt", 
                           "Cài đặt dependencies")
        return result is not None
    else:
        print("❌ Không tìm thấy requirements.txt")
        return False

def download_yolo_model():
    """Download YOLOv8 model"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "yolov8n.pt"
    
    if model_path.exists():
        print("📁 YOLOv8 model đã tồn tại")
        return True
    
    print("📥 Tải YOLOv8 model...")
    try:
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        urllib.request.urlretrieve(url, str(model_path))
        print("✅ YOLOv8 model đã tải xong")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return False

def setup_data_directories():
    """Setup data directory structure"""
    directories = [
        "data/raw/videos",
        "data/raw/images", 
        "data/processed",
        "data/annotations",
        "models",
        "demo/outputs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep for empty directories
        gitkeep_path = Path(dir_path) / ".gitkeep"
        if not any(Path(dir_path).iterdir()):  # Empty directory
            gitkeep_path.touch()
    
    print("✅ Cấu trúc thư mục đã được tạo")
    return True

def verify_installation():
    """Verify installation by testing imports"""
    print("🔍 Kiểm tra cài đặt...")
    
    # Determine python command based on OS
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Linux/Mac  
        python_cmd = "venv/bin/python"
    
    # Create test script file to avoid shell escaping issues
    test_script_content = """import sys
try:
    import cv2
    print("OK OpenCV:", cv2.__version__)
except ImportError:
    print("ERROR OpenCV not installed")
    sys.exit(1)

try:
    import ultralytics
    print("OK Ultralytics available")
except ImportError:
    print("ERROR Ultralytics not installed")
    sys.exit(1)

try:
    import mediapipe as mp
    print("OK MediaPipe available")
except ImportError:
    print("ERROR MediaPipe not installed")
    sys.exit(1)

try:
    import easyocr
    print("OK EasyOCR available")
except ImportError:
    print("ERROR EasyOCR not installed")
    sys.exit(1)

print("SUCCESS All dependencies installed!")
"""
    
    # Write test script to file
    with open('temp_test.py', 'w') as f:
        f.write(test_script_content)
    
    result = subprocess.run(f'{python_cmd} temp_test.py', 
                          shell=True, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists('temp_test.py'):
        os.remove('temp_test.py')
    
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("❌ Một số dependencies chưa được cài đặt đúng:")
        print(result.stdout)
        print(result.stderr)
        return False

def create_run_scripts():
    """Create convenience run scripts"""
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        python_cmd = "venv\\Scripts\\python"
        
        # Windows batch script
        run_script = f'''@echo off
echo 🚀 Starting Vietnam Traffic AI System...
call {activate_cmd}
{python_cmd} main.py %*
'''
        with open("run.bat", 'w') as f:
            f.write(run_script)
        print("✅ Tạo run.bat script")
        
    else:  # Linux/Mac
        activate_cmd = "source venv/bin/activate"
        python_cmd = "venv/bin/python"
        
        # Bash script
        run_script = f'''#!/bin/bash
echo "🚀 Starting Vietnam Traffic AI System..."
{activate_cmd}
{python_cmd} main.py "$@"
'''
        with open("run.sh", 'w') as f:
            f.write(run_script)
        
        # Make executable
        os.chmod("run.sh", 0o755)
        print("✅ Tạo run.sh script")
    
    # Streamlit run script
    if os.name == 'nt':
        streamlit_script = f'''@echo off
echo 🌐 Starting Streamlit Web Demo...
call {activate_cmd}
{python_cmd} -m streamlit run demo/streamlit_app.py
'''
        with open("run_demo.bat", 'w') as f:
            f.write(streamlit_script)
    else:
        streamlit_script = f'''#!/bin/bash
echo "🌐 Starting Streamlit Web Demo..."
{activate_cmd}
{python_cmd} -m streamlit run demo/streamlit_app.py
'''
        with open("run_demo.sh", 'w') as f:
            f.write(streamlit_script)
        os.chmod("run_demo.sh", 0o755)
    
    print("✅ Tạo demo run scripts")

def main():
    print("🚀 Khởi động setup Vietnam Traffic AI System...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("💡 Hãy cài đặt Python 3.8 trở lên và thử lại")
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download models
    if not download_yolo_model():
        print("⚠️  Model download thất bại, nhưng có thể tiếp tục")
    
    # Setup directories
    setup_data_directories()
    
    # Create run scripts
    create_run_scripts()
    
    # Verify installation
    if not verify_installation():
        return False
    
    print("=" * 50)
    print("🎉 Setup hoàn thành!")
    print()
    print("🚀 Cách chạy:")
    
    if os.name == 'nt':  # Windows
        print("  • Demo:     run_demo.bat")
        print("  • Webcam:   run.bat --mode webcam")
        print("  • Video:    run.bat --mode video --input video.mp4")
    else:  # Linux/Mac
        print("  • Demo:     ./run_demo.sh")
        print("  • Webcam:   ./run.sh --mode webcam") 
        print("  • Video:    ./run.sh --mode video --input video.mp4")
    
    print()
    print("📖 Đọc README.md để biết thêm chi tiết!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)