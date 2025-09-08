import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
import time
from pathlib import Path

class VideoProcessor:
    """
    Utility class for processing videos with AI modules
    """
    
    def __init__(self, output_fps: int = 30):
        self.output_fps = output_fps
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'processing_time': 0,
            'fps': 0
        }
    
    def process_video(self, input_path: str, output_path: str, 
                     processor_func: Callable[[np.ndarray], np.ndarray],
                     show_progress: bool = True) -> Dict:
        """
        Process video file with given processor function
        
        Args:
            input_path: Input video file path
            output_path: Output video file path  
            processor_func: Function that takes a frame and returns processed frame
            show_progress: Whether to show processing progress
            
        Returns:
            Dictionary with processing statistics
        """
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.output_fps, (width, height))
        
        self.stats['total_frames'] = total_frames
        start_time = time.time()
        
        frame_count = 0
        
        if show_progress:
            print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = processor_func(frame)
            
            # Write frame
            out.write(processed_frame)
            
            frame_count += 1
            self.stats['processed_frames'] = frame_count
            
            # Show progress
            if show_progress and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"FPS: {current_fps:.1f}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        self.stats['processing_time'] = total_time
        self.stats['fps'] = frame_count / total_time if total_time > 0 else 0
        
        if show_progress:
            print(f"Processing complete!")
            print(f"Processed {frame_count} frames in {total_time:.2f}s")
            print(f"Average FPS: {self.stats['fps']:.2f}")
        
        return self.stats.copy()
    
    def process_webcam(self, processor_func: Callable[[np.ndarray], np.ndarray],
                      window_name: str = "Vietnam Traffic AI",
                      save_output: bool = False,
                      output_path: Optional[str] = None) -> None:
        """
        Process webcam feed in real-time
        
        Args:
            processor_func: Function that takes a frame and returns processed frame
            window_name: Name of the display window
            save_output: Whether to save processed video
            output_path: Path to save output video (if save_output=True)
        """
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Setup video writer if saving
        out = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.output_fps, (1280, 720))
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print(f"Starting webcam processing...")
        print(f"Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            start_process_time = time.time()
            processed_frame = processor_func(frame)
            process_time = time.time() - start_process_time
            
            # Add FPS and processing info
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:  # Update FPS every second
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Add info text to frame
            info_text = f"FPS: {current_fps:.1f} | Process: {process_time*1000:.1f}ms"
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow(window_name, processed_frame)
            
            # Save frame if requested
            if out is not None:
                out.write(processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                save_path = f"captured_frame_{timestamp}.jpg"
                cv2.imwrite(save_path, processed_frame)
                print(f"Frame saved: {save_path}")
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print("Webcam processing stopped")
    
    def batch_process_images(self, input_dir: str, output_dir: str,
                           processor_func: Callable[[np.ndarray], np.ndarray],
                           image_extensions: List[str] = ['.jpg', '.jpeg', '.png']):
        """
        Batch process images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            processor_func: Function that takes a frame and returns processed frame
            image_extensions: List of image file extensions to process
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"Failed to load: {image_file}")
                    continue
                
                # Process image
                processed_image = processor_func(image)
                
                # Save processed image
                output_file = output_path / image_file.name
                cv2.imwrite(str(output_file), processed_image)
                
                # Show progress
                if (i + 1) % 10 == 0:
                    progress = ((i + 1) / len(image_files)) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(image_files)})")
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"Batch processing complete! Results saved to {output_dir}")
    
    def create_comparison_video(self, original_path: str, processed_path: str, 
                              output_path: str) -> None:
        """
        Create side-by-side comparison video
        
        Args:
            original_path: Path to original video
            processed_path: Path to processed video  
            output_path: Path to save comparison video
        """
        
        cap_orig = cv2.VideoCapture(original_path)
        cap_proc = cv2.VideoCapture(processed_path)
        
        if not cap_orig.isOpened() or not cap_proc.isOpened():
            raise ValueError("Cannot open one of the input videos")
        
        # Get video properties
        fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
        width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video (double width for side-by-side)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        print("Creating comparison video...")
        
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_proc, frame_proc = cap_proc.read()
            
            if not ret_orig or not ret_proc:
                break
            
            # Create side-by-side frame
            comparison_frame = np.hstack([frame_orig, frame_proc])
            
            # Add labels
            cv2.putText(comparison_frame, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison_frame, "Processed", (width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(comparison_frame)
        
        # Cleanup
        cap_orig.release()
        cap_proc.release()
        out.release()
        
        print(f"Comparison video saved: {output_path}")
    
    def extract_statistics(self, video_path: str, 
                         analyzer_func: Callable[[np.ndarray], Dict]) -> Dict:
        """
        Extract statistics from video using analyzer function
        
        Args:
            video_path: Path to video file
            analyzer_func: Function that takes a frame and returns statistics dict
            
        Returns:
            Aggregated statistics from all frames
        """
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        aggregated_stats = {}
        
        print(f"Analyzing {total_frames} frames for statistics...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame statistics
            frame_stats = analyzer_func(frame)
            
            # Aggregate statistics
            for key, value in frame_stats.items():
                if key not in aggregated_stats:
                    aggregated_stats[key] = []
                aggregated_stats[key].append(value)
            
            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Analysis progress: {progress:.1f}%")
        
        cap.release()
        
        # Calculate final statistics (averages, totals, etc.)
        final_stats = {
            'total_frames_analyzed': frame_count,
            'video_path': video_path
        }
        
        for key, values in aggregated_stats.items():
            if isinstance(values[0], (int, float)):
                final_stats[f"avg_{key}"] = sum(values) / len(values)
                final_stats[f"max_{key}"] = max(values)
                final_stats[f"min_{key}"] = min(values)
                final_stats[f"total_{key}"] = sum(values)
            elif isinstance(values[0], dict):
                # Handle nested dictionaries
                final_stats[key] = values  # Keep all frame data
        
        print("Video analysis complete!")
        return final_stats