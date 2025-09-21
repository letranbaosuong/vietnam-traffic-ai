"""
YOLOP-based Lane Detection Web Application
Vietnamese Traffic Safety System
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import json
import threading
from queue import Queue

# Import detection modules
from modules.yolop_detector import YOLOPDetector
from modules.lane_departure_warning import LaneDepartureWarning
from modules.video_processor import VideoProcessor

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
for folder in ['uploads', 'outputs', 'static/results']:
    os.makedirs(folder, exist_ok=True)

# Initialize detectors
yolop_detector = YOLOPDetector()
lane_warning = LaneDepartureWarning()
video_processor = VideoProcessor()

# Processing queue
processing_queue = Queue()
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'Không tìm thấy file video'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    if file and allowed_file(file.filename):
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate job ID
        job_id = f"job_{timestamp}_{np.random.randint(1000, 9999)}"

        # Add to processing queue
        processing_status[job_id] = {
            'status': 'queued',
            'progress': 0,
            'input_file': filepath,
            'output_file': None,
            'stats': None,
            'started_at': datetime.now().isoformat()
        }

        # Start processing in background
        thread = threading.Thread(target=process_video_task, args=(job_id, filepath))
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video đã được tải lên và đang xử lý'
        })

    return jsonify({'error': 'File không hợp lệ'}), 400

def process_video_task(job_id, input_path):
    """Process video in background"""
    try:
        processing_status[job_id]['status'] = 'processing'

        # Output path
        output_filename = f"processed_{os.path.basename(input_path)}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'lane_departures': 0,
            'vehicles_detected': [],
            'dangerous_situations': []
        }

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Update progress
            progress = int((frame_count / total_frames) * 100)
            processing_status[job_id]['progress'] = progress

            # Run YOLOP detection
            detection_result = yolop_detector.detect(frame)

            # Draw detection results
            processed_frame = draw_detections(frame, detection_result)

            # Check lane departure
            lane_status = lane_warning.check_departure(detection_result['lanes'])
            if lane_status['departed']:
                stats['lane_departures'] += 1
                stats['dangerous_situations'].append({
                    'frame': frame_count,
                    'type': 'lane_departure',
                    'message': lane_status['message']
                })

                # Draw warning
                processed_frame = draw_warning(processed_frame, lane_status['message'])

            # Count vehicles
            vehicles = detection_result.get('objects', [])
            for vehicle in vehicles:
                vehicle_type = vehicle.get('class', 'unknown')
                if vehicle_type not in stats['vehicles_detected']:
                    stats['vehicles_detected'].append(vehicle_type)

            # Write frame
            out.write(processed_frame)

            stats['processed_frames'] = frame_count

        # Release resources
        cap.release()
        out.release()

        # Update status
        processing_status[job_id].update({
            'status': 'completed',
            'progress': 100,
            'output_file': output_path,
            'stats': stats,
            'completed_at': datetime.now().isoformat()
        })

    except Exception as e:
        processing_status[job_id].update({
            'status': 'failed',
            'error': str(e)
        })

def draw_detections(frame, detection_result):
    """Draw detection results on frame"""
    h, w = frame.shape[:2]

    # Draw drivable area (green overlay)
    if 'drivable_area' in detection_result:
        drivable_mask = detection_result['drivable_area']
        overlay = frame.copy()
        overlay[drivable_mask > 0] = [0, 255, 0]  # Green for drivable area
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw lanes (yellow lines)
    if 'lanes' in detection_result:
        for lane in detection_result['lanes']:
            if len(lane) > 1:
                pts = np.array(lane, dtype=np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 255), 3)

    # Draw vehicles (bounding boxes)
    if 'objects' in detection_result:
        for obj in detection_result['objects']:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['class']}: {obj['confidence']:.2f}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def draw_warning(frame, message):
    """Draw warning message on frame"""
    h, w = frame.shape[:2]

    # Create warning overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 255), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw warning text
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 60

    cv2.putText(frame, message, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id in processing_status:
        return jsonify(processing_status[job_id])
    return jsonify({'error': 'Job không tồn tại'}), 404

@app.route('/download/<job_id>')
def download_result(job_id):
    """Download processed video"""
    if job_id in processing_status:
        job = processing_status[job_id]
        if job['status'] == 'completed' and job['output_file']:
            return send_file(job['output_file'], as_attachment=True)
    return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/api/demo', methods=['POST'])
def demo_detection():
    """Quick demo with single frame"""
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy ảnh'}), 400

    file = request.files['image']

    # Read image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run detection
    result = yolop_detector.detect(image)

    # Process result
    processed = draw_detections(image, result)

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', processed)
    image_base64 = buffer.tobytes().hex()

    return jsonify({
        'success': True,
        'image': image_base64,
        'detections': {
            'lanes': len(result.get('lanes', [])),
            'vehicles': len(result.get('objects', [])),
            'has_drivable_area': 'drivable_area' in result
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)