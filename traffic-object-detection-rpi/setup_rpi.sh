#!/bin/bash

echo "=== Traffic Object Detection - Raspberry Pi Setup ==="

echo "1. Updating system..."
sudo apt update && sudo apt upgrade -y

echo "2. Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    git \
    htop

echo "3. Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "4. Creating directories..."
mkdir -p models
mkdir -p data/{videos,images,outputs}

echo "5. Optimizing Raspberry Pi..."

read -p "Enable performance governor? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo apt install -y cpufrequtils
    echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
    sudo systemctl restart cpufrequtils
fi

read -p "Increase GPU memory split to 128MB? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo sed -i 's/^gpu_mem=.*/gpu_mem=128/' /boot/config.txt
    if ! grep -q "gpu_mem=" /boot/config.txt; then
        echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    fi
    echo "GPU memory set to 128MB. Reboot required."
fi

read -p "Increase swap to 2GB? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo dphys-swapfile swapoff
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    echo "Swap increased to 2GB"
fi

echo "6. Testing installation..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "from ultralytics import YOLO; print('Ultralytics OK')"

echo "7. System info..."
echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "RAM: $(free -h | grep '^Mem' | awk '{print $2}')"
echo "Storage: $(df -h / | tail -1 | awk '{print $4}' ) available"

if [ -f /usr/bin/vcgencmd ]; then
    echo "Temperature: $(vcgencmd measure_temp)"
    echo "Throttled: $(vcgencmd get_throttled)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Reboot if GPU memory was changed: sudo reboot"
echo "2. Download test video to data/videos/"
echo "3. Run demo: python3 demo_simple.py"
echo "4. Run benchmark: python3 main.py --mode benchmark"
echo ""
echo "For camera test:"
echo "  python3 main.py --mode camera"
echo ""