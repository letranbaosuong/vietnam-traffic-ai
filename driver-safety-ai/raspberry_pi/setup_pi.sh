#!/bin/bash
# Setup script cho Raspberry Pi 4
# Cài đặt môi trường cho Driver Safety AI

echo "==================================="
echo "Driver Safety AI - Raspberry Pi Setup"
echo "==================================="

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and essential tools
echo "[2/8] Installing Python and development tools..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    cmake \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    python3-pyqt5

# Install camera libraries
echo "[3/8] Installing camera libraries..."
sudo apt-get install -y \
    libcamera-dev \
    libcamera-apps \
    python3-picamera2 \
    v4l-utils

# Create virtual environment
echo "[4/8] Creating Python virtual environment..."
python3 -m venv ~/driver_safety_env
source ~/driver_safety_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install NumPy first (required for OpenCV)
echo "[5/8] Installing NumPy..."
pip install numpy==1.24.3

# Install OpenCV (pre-compiled for ARM)
echo "[6/8] Installing OpenCV..."
pip install opencv-contrib-python==4.8.1.78

# Install TensorFlow Lite Runtime
echo "[7/8] Installing TensorFlow Lite Runtime..."
# For 64-bit Raspberry Pi OS
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Alternative if above doesn't work:
# wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
# pip install tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl

# Install other Python packages
echo "[8/8] Installing other dependencies..."
pip install \
    pillow==10.1.0 \
    scipy==1.11.3 \
    scikit-learn==1.3.1 \
    imutils==0.5.4 \
    RPi.GPIO==0.7.1 \
    psutil \
    flask \
    flask-cors

# Enable camera
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Increase GPU memory split (for better camera performance)
echo "Configuring GPU memory..."
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Setup systemd service (optional)
echo "Creating systemd service file..."
cat << EOF | sudo tee /etc/systemd/system/driver-monitor.service
[Unit]
Description=Driver Safety Monitoring System
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/driver-safety-ai
Environment="PATH=/home/pi/driver_safety_env/bin"
ExecStart=/home/pi/driver_safety_env/bin/python /home/pi/driver-safety-ai/raspberry_pi/deploy.py --model /home/pi/driver-safety-ai/models/model_int8.tflite --headless
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Reboot your Raspberry Pi: sudo reboot"
echo "2. Activate virtual environment: source ~/driver_safety_env/bin/activate"
echo "3. Clone the project: git clone <your-repo-url> ~/driver-safety-ai"
echo "4. Copy your trained model to ~/driver-safety-ai/models/"
echo "5. Test the system: python ~/driver-safety-ai/raspberry_pi/deploy.py --model <model-path>"
echo ""
echo "To enable auto-start on boot:"
echo "sudo systemctl enable driver-monitor.service"
echo "sudo systemctl start driver-monitor.service"
echo ""
echo "Check service status:"
echo "sudo systemctl status driver-monitor.service"