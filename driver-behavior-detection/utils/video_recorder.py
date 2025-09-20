import cv2
import os
from datetime import datetime
import threading

class VideoRecorder:
    def __init__(self, output_dir="logs/recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.is_recording = False
        self.video_writer = None
        self.current_filename = None
        self.frame_queue = []
        self.recording_thread = None

    def start_recording(self, frame_width, frame_height, fps=20, event_type="general"):
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(
            self.output_dir,
            f"{event_type}_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_filename,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        self.is_recording = True
        self.frame_queue = []

        self.recording_thread = threading.Thread(target=self._write_frames)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        return self.current_filename

    def add_frame(self, frame):
        if self.is_recording and frame is not None:
            self.frame_queue.append(frame.copy())

    def _write_frames(self):
        while self.is_recording or len(self.frame_queue) > 0:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.pop(0)
                if self.video_writer:
                    self.video_writer.write(frame)

    def stop_recording(self):
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join(timeout=2)

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        saved_file = self.current_filename
        self.current_filename = None

        return saved_file

    def save_snapshot(self, frame, event_type="snapshot"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.output_dir,
            f"{event_type}_{timestamp}.jpg"
        )
        cv2.imwrite(filename, frame)
        return filename

class EventLogger:
    def __init__(self, log_file="logs/events.csv"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("timestamp,event_type,duration,severity,details\n")

    def log_event(self, event_type, duration=0, severity="LOW", details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{event_type},{duration},{severity},{details}\n")

    def get_statistics(self):
        if not os.path.exists(self.log_file):
            return {}

        stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {}
        }

        with open(self.log_file, 'r') as f:
            lines = f.readlines()[1:]
            stats['total_events'] = len(lines)

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    event_type = parts[1]
                    severity = parts[3]

                    if event_type not in stats['events_by_type']:
                        stats['events_by_type'][event_type] = 0
                    stats['events_by_type'][event_type] += 1

                    if severity not in stats['events_by_severity']:
                        stats['events_by_severity'][severity] = 0
                    stats['events_by_severity'][severity] += 1

        return stats