#!/bin/bash
# Setup script cho macOS
# Cài đặt môi trường cho Driver Safety AI trên macOS

echo "==================================="
echo "Driver Safety AI - macOS Setup"
echo "==================================="

# Check Python version
echo "[1/7] Checking Python version..."
python3 --version

# Create virtual environment
echo "[2/7] Creating Python virtual environment..."
python3 -m venv ~/driver_safety_env
source ~/driver_safety_env/bin/activate

# Upgrade pip
echo "[3/7] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install NumPy (latest compatible version)
echo "[4/7] Installing NumPy..."
pip install numpy

# Install OpenCV
echo "[5/7] Installing OpenCV..."
pip install opencv-contrib-python

# Install TensorFlow Lite Runtime or TensorFlow
echo "[6/7] Installing TensorFlow..."
# For macOS, use full TensorFlow as tflite_runtime may not be available
pip install tensorflow

# Install other Python packages
echo "[7/7] Installing other dependencies..."
pip install \
    pillow \
    scipy \
    scikit-learn \
    imutils \
    psutil \
    flask \
    flask-cors

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source ~/driver_safety_env/bin/activate"
echo "2. Test the system: python raspberry_pi/deploy.py --model <model-path>"
echo ""
echo "Note: Some Raspberry Pi specific features (GPIO, Pi Camera) won't work on macOS."
echo "Use USB webcam or built-in camera instead."