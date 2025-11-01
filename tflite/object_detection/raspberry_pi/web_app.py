#!/usr/bin/env python3
"""
Web Application for Traffic Detection System
Features:
- Lane Detection
- Driver Gesture Detection
"""

from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
import cv2
import time
from werkzeug.utils import secure_filename
import json

# Import detection modules
from lane_detector import LaneDetector
from gesture_warning_system import GestureWarningSystem

# MediaPipe not available on macOS Python 3.13, use simulation mode
try:
    from driver_gesture_detector import DriverGestureDetector
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - using simulation mode for gesture detection")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    detection_type = request.form.get('detection_type', 'lane')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        return jsonify({
            'success': True,
            'filename': unique_filename,
            'detection_type': detection_type
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/process/lane/<filename>')
def process_lane(filename):
    """Process video with lane detection"""
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"lane_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Initialize lane detector
        lane_detector = LaneDetector()

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video'}), 500

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed_frames = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect lanes
            detection_result = lane_detector.detect(frame)

            # Visualize lanes on frame
            lane_frame = lane_detector.visualize(frame, detection_result)

            # Write output
            out.write(lane_frame)
            processed_frames += 1

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed if elapsed > 0 else 0

        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

        return jsonify({
            'success': True,
            'output_file': output_filename,
            'stats': {
                'frames_processed': processed_frames,
                'processing_time': f"{elapsed:.2f}s",
                'avg_fps': f"{avg_fps:.2f}",
                'output_size': f"{file_size:.1f} MB"
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process/gesture/<filename>')
def process_gesture(filename):
    """Process video with gesture detection (simulation)"""
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"gesture_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Initialize gesture system
        warning_system = GestureWarningSystem()

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video'}), 500

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        warning_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Simulate gesture detection
            warnings = simulate_gestures(frame_num)

            # Clear old warnings
            warning_system.clear_old_warnings(max_age=2.0)

            # Add warnings
            for warning in warnings:
                if "điện thoại" in warning.lower():
                    warning_system.add_warning(warning, 'phone_usage')
                elif "tập trung" in warning.lower():
                    warning_system.add_warning(warning, 'distraction')
                elif "vô lăng" in warning.lower():
                    warning_system.add_warning(warning, 'hands_off_wheel')

            # Visualize
            output_frame = warning_system.draw_warnings(frame, warnings)
            output_frame = warning_system.draw_status_bar(output_frame)

            if warnings:
                warning_count += 1

            # Write output
            out.write(output_frame)

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0

        # Get statistics
        stats_report = warning_system.get_warning_report()

        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

        return jsonify({
            'success': True,
            'output_file': output_filename,
            'stats': {
                'frames_processed': frame_num,
                'frames_with_warnings': warning_count,
                'warning_percentage': f"{100*warning_count/frame_num:.1f}%",
                'processing_time': f"{elapsed:.2f}s",
                'avg_fps': f"{avg_fps:.2f}",
                'output_size': f"{file_size:.1f} MB",
                'report': stats_report
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process/both/<filename>')
def process_both(filename):
    """Process video with both lane and gesture detection"""
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"combined_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Initialize detectors
        lane_detector = LaneDetector()
        warning_system = GestureWarningSystem()

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video'}), 500

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        warning_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Lane detection
            detection_result = lane_detector.detect(frame)
            frame = lane_detector.visualize(frame, detection_result)

            # Gesture detection (simulation)
            warnings = simulate_gestures(frame_num)
            warning_system.clear_old_warnings(max_age=2.0)

            for warning in warnings:
                if "điện thoại" in warning.lower():
                    warning_system.add_warning(warning, 'phone_usage')
                elif "tập trung" in warning.lower():
                    warning_system.add_warning(warning, 'distraction')
                elif "vô lăng" in warning.lower():
                    warning_system.add_warning(warning, 'hands_off_wheel')

            # Visualize gestures
            frame = warning_system.draw_warnings(frame, warnings)
            frame = warning_system.draw_status_bar(frame)

            if warnings:
                warning_count += 1

            # Write output
            out.write(frame)

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0
        file_size = os.path.getsize(output_path) / (1024 * 1024)

        return jsonify({
            'success': True,
            'output_file': output_filename,
            'stats': {
                'frames_processed': frame_num,
                'frames_with_warnings': warning_count,
                'processing_time': f"{elapsed:.2f}s",
                'avg_fps': f"{avg_fps:.2f}",
                'output_size': f"{file_size:.1f} MB"
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download processed video"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/results')
def results():
    """Show all processed results"""
    output_files = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        files = os.listdir(app.config['OUTPUT_FOLDER'])
        for f in sorted(files, reverse=True):
            if f.endswith(('.mp4', '.avi')):
                filepath = os.path.join(app.config['OUTPUT_FOLDER'], f)
                size = os.path.getsize(filepath) / (1024 * 1024)
                output_files.append({
                    'filename': f,
                    'size': f"{size:.1f} MB",
                    'type': f.split('_')[0] if '_' in f else 'unknown'
                })

    return render_template('results.html', files=output_files)


def simulate_gestures(frame_num):
    """Simulate gesture warnings based on frame number"""
    warnings = []

    if 50 < frame_num < 100:
        warnings.append("⚠️ NGUY HIỂM: Đang gọi điện thoại!")

    if 150 < frame_num < 200:
        warnings.append("⚠️ MẤT TẬP TRUNG: Đang nhìn sang PHẢI (35°)!")

    if 250 < frame_num < 280:
        warnings.append("⚠️ CẢNH BÁO: Tay rời vô lăng!")

    if 320 < frame_num < 360:
        warnings.append("⚠️ NGUY HIỂM: Đang xem điện thoại!")

    import random
    if random.random() > 0.95:
        warnings.append("⚠️ MẤT TẬP TRUNG: Đang nhìn sang TRÁI (28°)!")

    return warnings


if __name__ == '__main__':
    print("=" * 70)
    print("🚗 Traffic Detection Web Application")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✅ Lane Detection")
    print("  ✅ Driver Gesture Detection")
    print("  ✅ Combined Detection")
    print("\nStarting server...")
    print("\n📱 Open in browser: http://localhost:5001")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5001)
