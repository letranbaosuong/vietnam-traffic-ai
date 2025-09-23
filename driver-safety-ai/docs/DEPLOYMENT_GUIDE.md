# Hướng Dẫn Triển Khai Driver Safety AI

## 📱 Triển Khai Trên Raspberry Pi 4

### Yêu Cầu Phần Cứng

- **Raspberry Pi 4** (4GB RAM tối thiểu, khuyến nghị 8GB)
- **Camera Module** (Pi Camera v2 hoặc USB Webcam)
- **MicroSD Card** (32GB+, Class 10)
- **Nguồn điện** 5V/3A USB-C
- **Tản nhiệt** (quan trọng cho inference liên tục)
- **Tùy chọn**: AI Accelerator (Coral USB, Intel NCS2, Hailo-8L)

### Cài Đặt Hệ Thống

#### 1. Cài Raspberry Pi OS

```bash
# Download Raspberry Pi Imager
# Flash Raspberry Pi OS 64-bit lên SD card
# Enable SSH và WiFi trong cấu hình ban đầu
```

#### 2. Chạy Script Setup Tự Động

```bash
# Clone project
git clone https://github.com/your-repo/driver-safety-ai.git
cd driver-safety-ai

# Chạy setup script
chmod +x raspberry_pi/setup_pi.sh
./raspberry_pi/setup_pi.sh
```

#### 3. Download Model Đã Train

```bash
# Tải model từ Google Drive hoặc server
wget https://your-server.com/models/model_int8.tflite -O models/model_int8.tflite
```

### Chạy Hệ Thống

#### Mode Cơ Bản

```bash
# Activate virtual environment
source ~/driver_safety_env/bin/activate

# Chạy với camera
python raspberry_pi/deploy.py --model models/model_int8.tflite

# Chạy headless (không hiển thị)
python raspberry_pi/deploy.py --model models/model_int8.tflite --headless
```

#### Mode Nâng Cao với Config

```bash
# Sử dụng file config
python raspberry_pi/deploy.py --config configs/config.yaml
```

### Tự Động Khởi Động

```bash
# Enable service
sudo systemctl enable driver-monitor.service
sudo systemctl start driver-monitor.service

# Check status
sudo systemctl status driver-monitor.service

# View logs
journalctl -u driver-monitor.service -f
```

## 🚀 Tối Ưu Hiệu Suất

### 1. Chọn Model Phù Hợp

| Model | FPS | RAM | Độ Chính Xác | Khuyến Nghị |
|-------|-----|-----|--------------|-------------|
| INT8 Quantized | 20-25 | 150MB | 90% | ✅ Pi 4 không accelerator |
| FP16 | 15-18 | 250MB | 93% | Pi 4 + tản nhiệt tốt |
| Full Model | 8-10 | 400MB | 95% | Pi 4 + Coral USB |

### 2. Sử Dụng Hardware Acceleration

#### Google Coral USB

```bash
# Install Edge TPU runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

# Convert model cho Edge TPU
edgetpu_compiler models/model_edgetpu.tflite

# Run với Coral
python raspberry_pi/deploy.py --model models/model_edgetpu_edgetpu.tflite --use-tpu
```

#### Intel Neural Compute Stick 2

```bash
# Install OpenVINO
wget https://download.01.org/opencv/2021/openvinotoolkit/2021.4/l_openvino_toolkit_runtime_raspbian_p_2021.4.689.tgz
tar -xf l_openvino_toolkit_runtime_raspbian_p_2021.4.689.tgz
sudo ./l_openvino_toolkit_runtime_raspbian_p_2021.4.689/install_openvino_dependencies.sh

# Setup environment
source /opt/intel/openvino_2021/bin/setupvars.sh

# Run với NCS2
python raspberry_pi/deploy.py --model models/model.xml --use-ncs
```

### 3. Tối Ưu Camera

```python
# Trong config.yaml
camera:
  resolution: [640, 480]  # Giảm resolution để tăng FPS
  fps: 15                 # Giảm FPS nếu cần

performance:
  num_threads: 4         # Sử dụng tất cả CPU cores
```

## 🎯 Training Model Tùy Chỉnh

### 1. Thu Thập Dữ Liệu

```bash
# Sử dụng tool thu thập data
python tools/collect_data.py --output data/custom_dataset --duration 60
```

### 2. Training trên Google Colab

1. Upload notebook `training/train_on_colab.ipynb` lên Colab
2. Upload dataset lên Google Drive
3. Chạy training với GPU miễn phí
4. Download model đã optimize

### 3. Test Model Mới

```bash
# Benchmark model
python raspberry_pi/optimize_model.py --model model.h5 --benchmark

# Test accuracy
python tests/test_accuracy.py --model models/model_int8.tflite --dataset data/test
```

## 🔧 Troubleshooting

### Camera Không Hoạt Động

```bash
# Check camera
vcgencmd get_camera
v4l2-ctl --list-devices

# Enable camera
sudo raspi-config nonint do_camera 0
sudo reboot
```

### Model Chạy Chậm

1. Kiểm tra nhiệt độ CPU:
```bash
vcgencmd measure_temp
```

2. Tăng GPU memory split:
```bash
sudo nano /boot/config.txt
# Add: gpu_mem=128
```

3. Sử dụng model quantized nhẹ hơn

### Out of Memory

```bash
# Check memory
free -h

# Tăng swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 📊 Monitoring & Logging

### Dashboard Web (Tùy chọn)

```bash
# Enable web dashboard
python webapp/app.py

# Access: http://raspberry-pi-ip:5000
```

### Metrics Collection

```python
# Trong config.yaml
monitoring:
  enable: true
  metrics:
    - fps
    - cpu_temp
    - memory_usage
    - detection_rate
```

### Remote Monitoring

```bash
# Setup ngrok cho remote access
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-arm.zip
unzip ngrok-stable-linux-arm.zip
./ngrok authtoken YOUR_TOKEN
./ngrok http 5000
```

## 🔒 Bảo Mật

### 1. Mã Hóa Dữ Liệu

```bash
# Encrypt sensitive data
openssl enc -aes-256-cbc -in config.yaml -out config.enc
```

### 2. Secure Boot

```bash
# Enable firewall
sudo ufw enable
sudo ufw allow 22  # SSH
sudo ufw allow 5000  # Web interface
```

### 3. Update Định Kỳ

```bash
# Auto update script
sudo crontab -e
# Add: 0 2 * * 0 apt-get update && apt-get upgrade -y
```

## 📝 Best Practices

1. **Test kỹ trước triển khai thực tế**
2. **Backup model và config định kỳ**
3. **Monitor nhiệt độ và hiệu suất**
4. **Cập nhật firmware camera**
5. **Sử dụng UPS để tránh mất điện đột ngột**

## 🆘 Support

- GitHub Issues: [Link to issues]
- Documentation: [Link to docs]
- Community Forum: [Link to forum]