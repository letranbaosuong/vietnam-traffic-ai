import cv2
import numpy as np
from pathlib import Path
import time
from typing import Optional, Tuple
from tqdm import tqdm
import yaml


class VideoProcessor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_path = self.config['video']['input_path']
        self.output_path = self.config['video']['output_path']
        self.target_fps = self.config['video']['fps']
        self.codec = self.config['video']['codec']

    def process_video(self, detector, input_path: Optional[str] = None,
                     output_path: Optional[str] = None, display: bool = False):
        if input_path:
            self.input_path = input_path
        if output_path:
            self.output_path = output_path

        cap = cv2.VideoCapture(self.input_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_skip = max(1, int(original_fps / self.target_fps))

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (width, height))

        frame_count = 0
        processed_count = 0
        total_inference_time = 0

        stats = {
            'total_detections': 0,
            'class_counts': {},
            'avg_fps': 0
        }

        pbar = tqdm(total=total_frames, desc="Processing video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update(1)

            if frame_count % frame_skip == 0:
                start_time = time.time()

                detections = detector.detect(frame)

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                processed_frame = detector.draw_detections(frame, detections)

                fps_text = f"FPS: {1/inference_time:.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                count_text = f"Objects: {len(detections)}"
                cv2.putText(processed_frame, count_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                out.write(processed_frame)

                stats['total_detections'] += len(detections)
                for det in detections:
                    class_name = det['class_name']
                    stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1

                if display:
                    cv2.imshow('Traffic Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                processed_count += 1

            frame_count += 1

        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if processed_count > 0:
            stats['avg_fps'] = processed_count / total_inference_time

        return stats

    def process_camera(self, detector, camera_id: int = 0, display: bool = True,
                      save_output: bool = False):
        cap = cv2.VideoCapture(camera_id)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if save_output:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (width, height))

        frame_count = 0
        total_inference_time = 0
        fps_history = []

        print("Press 'q' to quit, 's' to save snapshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            detections = detector.detect(frame)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            current_fps = 1 / inference_time if inference_time > 0 else 0
            fps_history.append(current_fps)

            if len(fps_history) > 30:
                fps_history.pop(0)

            avg_fps = sum(fps_history) / len(fps_history)

            processed_frame = detector.draw_detections(frame, detections)

            cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(processed_frame, f"Objects: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(detections) > 0:
                y_offset = 90
                for class_name in set(d['class_name'] for d in detections):
                    count = sum(1 for d in detections if d['class_name'] == class_name)
                    cv2.putText(processed_frame, f"{class_name}: {count}",
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 25

            if display:
                cv2.imshow('Traffic Detection - Camera', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snapshot_path = f"data/outputs/snapshot_{frame_count}.jpg"
                    cv2.imwrite(snapshot_path, processed_frame)
                    print(f"Snapshot saved: {snapshot_path}")

            if save_output and out is not None:
                out.write(processed_frame)

            frame_count += 1

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        print(f"\nAverage FPS: {avg_fps:.2f}")
        print(f"Total frames processed: {frame_count}")

    def benchmark(self, detector, num_frames: int = 100):
        cap = cv2.VideoCapture(self.input_path)

        inference_times = []
        frame_count = 0

        print(f"Running benchmark on {num_frames} frames...")

        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            _ = detector.detect(frame)
            inference_time = time.time() - start_time

            inference_times.append(inference_time)
            frame_count += 1

        cap.release()

        if inference_times:
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)

            print("\nBenchmark Results:")
            print(f"Frames processed: {frame_count}")
            print(f"Average inference time: {avg_time*1000:.2f}ms")
            print(f"Std deviation: {std_time*1000:.2f}ms")
            print(f"Min time: {min_time*1000:.2f}ms")
            print(f"Max time: {max_time*1000:.2f}ms")
            print(f"Average FPS: {1/avg_time:.2f}")

            return {
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'avg_fps': 1/avg_time
            }

        return None