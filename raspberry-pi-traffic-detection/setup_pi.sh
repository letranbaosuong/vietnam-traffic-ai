#!/bin/bash
# Setup script cho Raspberry Pi 4 Traffic Detection

echo "🚀 Setting up Raspberry Pi 4 for Traffic Detection"

# Update system
echo "📦 Updating system..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing dependencies..."
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    git cmake build-essential \
    libopencv-dev python3-opencv \
    libhdf5-dev libatlas-base-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module \
    libblas-dev liblapack-dev liblapacke-dev \
    gfortran pkg-config

# Tối ưu hóa Pi 4
echo "⚡ Optimizing Pi 4 performance..."

# GPU memory split
sudo raspi-config nonint do_memory_split 128

# CPU governor
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable hciuart

# Add to /boot/config.txt for better performance
sudo tee -a /boot/config.txt << EOF

# Optimizations for object detection
gpu_mem=128
arm_freq=1750
over_voltage=4
temp_limit=75

# Camera support
start_x=1
gpu_mem=128
EOF

# Virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch cho Pi 4 (ARM64)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt

# Download models
echo "📥 Downloading models..."
cd src
python download_models.py
cd ..

# Create sample systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/traffic-detection.service << EOF
[Unit]
Description=Traffic Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/raspberry-pi-traffic-detection
Environment=PATH=/home/pi/raspberry-pi-traffic-detection/venv/bin
ExecStart=/home/pi/raspberry-pi-traffic-detection/venv/bin/python src/traffic_detector.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
chmod +x src/*.py

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Reboot Pi: sudo reboot"
echo "2. Test detection: cd raspberry-pi-traffic-detection && source venv/bin/activate && python src/traffic_detector.py"
echo "3. Run benchmark: python src/benchmark.py"
echo "4. Enable service: sudo systemctl enable traffic-detection.service"
echo ""
echo "🎯 Performance tips:"
echo "- Use 720p or lower resolution for better FPS"
echo "- Monitor temperature: vcgencmd measure_temp"
echo "- Check CPU: htop"
echo "- GPU memory: vcgencmd get_mem gpu"