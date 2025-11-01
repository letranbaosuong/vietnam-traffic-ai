#!/bin/bash

echo "================================================================"
echo "🚗 TRAFFIC DETECTION WEB APPLICATION"
echo "================================================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check Flask
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

# Create required directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p outputs
mkdir -p templates
mkdir -p static

# Clear old files (optional)
read -p "Xóa video đã xử lý cũ? [y/N]: " clear_old
if [[ "$clear_old" =~ ^[Yy]$ ]]; then
    echo "Clearing old files..."
    rm -f uploads/*
    rm -f outputs/*
    echo "✅ Cleared!"
fi

echo ""
echo "================================================================"
echo "🌟 STARTING WEB SERVER"
echo "================================================================"
echo ""
echo "Features:"
echo "  ✅ Lane Detection"
echo "  ✅ Driver Gesture Detection"
echo "  ✅ Combined Detection"
echo ""
echo "📱 Access the web app at:"
echo "  http://localhost:5000"
echo "  http://$(hostname).local:5000"
echo "  http://$(ipconfig getifaddr en0 2>/dev/null || echo "your-ip"):5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================================"
echo ""

# Start Flask app
python3 web_app.py
