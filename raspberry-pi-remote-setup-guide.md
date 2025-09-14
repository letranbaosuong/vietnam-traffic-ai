# Hướng dẫn cấu hình Remote Access cho Raspberry Pi 4

## Mục lục
1. [Cấu hình mạng và SSH](#cấu-hình-mạng-và-ssh)
2. [Cài đặt TeamViewer](#cài-đặt-teamviewer)
3. [Cấu hình VNC](#cấu-hình-vnc)
4. [Khắc phục sự cố](#khắc-phục-sự-cố)
5. [Phương án thay thế](#phương-án-thay-thế)

---

## Cấu hình mạng và SSH

### 1. Kiểm tra kết nối mạng

#### Tìm IP của Raspberry Pi:
```bash
# Quét mạng để tìm Raspberry Pi
nmap -sn 192.168.1.0/24

# Hoặc tìm theo MAC address của Pi
arp -a | grep -i b8:27:eb

# Ping để kiểm tra kết nối
ping raspberrypi.local
# hoặc
ping 192.168.1.xxx  # thay xxx bằng IP thực tế
```

### 2. Khắc phục lỗi SSH "No supported authentication methods"

#### Bật password authentication:
```bash
# Nếu truy cập được Pi qua màn hình trực tiếp:
sudo nano /etc/ssh/sshd_config

# Tìm và sửa các dòng sau:
PasswordAuthentication yes
ChallengeResponseAuthentication yes
UsePAM yes

# Khởi động lại SSH service
sudo systemctl restart ssh
```

#### Nếu không truy cập được Pi trực tiếp:
1. Tắt Pi, lấy thẻ SD card cắm vào máy tính
2. Tạo/chỉnh sửa file trong partition boot
3. Khởi động lại Pi

---

## Cài đặt TeamViewer

### 1. Cập nhật hệ thống
```bash
# SSH vào Pi
ssh pi@192.168.1.170

# Cập nhật
sudo apt update && sudo apt upgrade -y
```

### 2. Tải và cài đặt TeamViewer Host
```bash
# Tải file cài đặt cho ARM
wget https://download.teamviewer.com/download/linux/teamviewer-host_armhf.deb

# Cài đặt package
sudo dpkg -i teamviewer-host_armhf.deb

# Fix dependencies nếu có lỗi
sudo apt install -f
```

### 3. Cấu hình TeamViewer
```bash
# Khởi động TeamViewer daemon
sudo teamviewer --daemon enable
sudo teamviewer --daemon start

# Lấy ID và password tạm thời
sudo teamviewer --info
```

### 4. Liên kết tài khoản TeamViewer (Khuyến nghị)
```bash
# Cấu hình tự động update
sudo teamviewer --configure enable_updates

# Đăng nhập tài khoản (thay email/password thật)
sudo teamviewer --configure account email=your_email@gmail.com password=your_password
```

### 5. Kết nối từ máy tính
1. Tải TeamViewer cho Windows/Mac từ teamviewer.com
2. Đăng nhập cùng tài khoản đã liên kết
3. Tìm Pi trong "My Computers" và click "Remote control"

**Lưu ý:** TeamViewer Host chỉ nhận kết nối, không thể kết nối ra ngoài. Miễn phí cho cá nhân.

---

## Cấu hình VNC

### 1. Bật VNC trên Raspberry Pi
```bash
# Sử dụng raspi-config
sudo raspi-config
# Interface Options > VNC > Enable
```

### 2. Cài đặt VNC Server (nếu chưa có)
```bash
sudo apt update
sudo apt install realvnc-vnc-server realvnc-vnc-viewer
sudo systemctl enable vncserver-x11-serviced
sudo systemctl start vncserver-x11-serviced
```

### 3. Kết nối qua RealVNC Viewer
- Tải RealVNC Viewer từ: https://www.realvnc.com/en/connect/download/viewer/
- Nhập địa chỉ: `raspberrypi.local:5900` hoặc `192.168.1.xxx:5900`
- Username: `pi`
- Password: password của user Pi

---

## Khắc phục sự cố

### Lỗi VNC "Cannot currently show the desktop"

#### Bước 1: Kiểm tra dung lượng đĩa
```bash
df -h
# Nếu đầy đĩa, dọn dẹp log:
sudo journalctl --vacuum-time=7d
```

#### Bước 2: Cấu hình resolution cho VNC
```bash
sudo raspi-config
# Advanced Options > Resolution > chọn 1920x1080 hoặc 1280x720
# Finish > Reboot
```

#### Bước 3: Đảm bảo boot vào desktop
```bash
sudo raspi-config
# System Options > Boot/Auto Login > Desktop Autologin
# Finish > Reboot
```

#### Bước 4: Fix lỗi Wayland (Raspberry Pi OS Bookworm)
```bash
sudo raspi-config
# Advanced Options > Wayland > X11
# Reboot
```

#### Bước 5: Force HDMI output
```bash
sudo nano /boot/config.txt
# Thêm các dòng:
hdmi_force_hotplug=1
hdmi_group=2
hdmi_mode=82
hdmi_drive=2

sudo reboot
```

#### Bước 6: Cài đặt desktop environment (nếu thiếu)
```bash
sudo apt update
sudo apt install raspberrypi-ui-mods lxde-core
```

#### Bước 7: Khởi động lại VNC service
```bash
sudo systemctl stop vncserver-x11-serviced
sudo systemctl start vncserver-x11-serviced
sudo systemctl enable vncserver-x11-serviced
```

### Lỗi GTK Pixbuf
```bash
# Cập nhật pixbuf loaders
sudo gdk-pixbuf-query-loaders --update-cache

# Cài đặt lại GTK themes
sudo apt install --reinstall adwaita-icon-theme gtk2-engines-pixbuf

# Rebuild icon cache
sudo gtk-update-icon-cache -f -t /usr/share/icons/Adwaita
sudo gtk-update-icon-cache -f -t /usr/share/icons/hicolor

# Cài đặt pixbuf loaders thiếu
sudo apt install librsvg2-common gdk-pixbuf2.0-dev

# Khởi động lại
sudo reboot
```

---

## Phương án thay thế

### 1. SSH với X11 Forwarding
```bash
# Từ máy tính (Linux/macOS)
ssh -X pi@192.168.1.170

# Chạy app GUI
lxterminal
```

### 2. NoMachine (Nhanh hơn VNC)
```bash
# Trên Pi
wget https://download.nomachine.com/download/7.10/Raspberry/nomachine_7.10.1_1_armhf.deb
sudo dpkg -i nomachine_7.10.1_1_armhf.deb

# Tải NoMachine client cho máy tính từ nomachine.com
```

### 3. Raspberry Pi Connect (Miễn phí, chính thức)
```bash
# Cài đặt trên Pi
sudo apt update
sudo apt install rpi-connect

# Truy cập qua web browser tại connect.raspberrypi.com
```

### 4. Các lệnh SSH hữu ích
```bash
# Copy file từ Pi về máy
scp pi@192.168.1.170:/home/pi/file.txt ./

# Copy file từ máy lên Pi
scp ./file.txt pi@192.168.1.170:/home/pi/

# Mount Pi filesystem qua SSHFS
sshfs pi@192.168.1.170:/home/pi ~/pi_mount
```

---

## Checklist cài đặt

### Chuẩn bị
- [ ] Raspberry Pi 4 đã cài OS
- [ ] Kết nối WiFi/Ethernet
- [ ] SSH đã bật
- [ ] Biết IP address của Pi

### TeamViewer
- [ ] Cập nhật hệ thống
- [ ] Tải và cài TeamViewer Host
- [ ] Liên kết tài khoản
- [ ] Test kết nối từ máy tính

### VNC (tùy chọn)
- [ ] Bật VNC trong raspi-config
- [ ] Cấu hình resolution
- [ ] Cài VNC Viewer trên máy tính
- [ ] Test kết nối

### Backup & Security
- [ ] Đổi password mặc định
- [ ] Backup SD card
- [ ] Cập nhật thường xuyên
- [ ] Chỉ mở port cần thiết

---

**Lưu ý bảo mật:**
- Luôn đổi password mặc định
- Sử dụng SSH key thay vì password khi có thể
- Cập nhật hệ thống thường xuyên
- Chỉ mở các port cần thiết trong router

**Tài liệu tham khảo:**
- TeamViewer: https://www.teamviewer.com/en/download/raspberry-pi/
- RealVNC: https://www.realvnc.com/en/connect/download/viewer/
- Raspberry Pi Documentation: https://www.raspberrypi.org/documentation/