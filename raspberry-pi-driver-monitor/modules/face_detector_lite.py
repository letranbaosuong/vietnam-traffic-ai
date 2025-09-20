import cv2
import numpy as np
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
    print("Warning: Using full TensorFlow Lite, consider installing tflite_runtime for better performance")

class FaceDetectorLite:
    def __init__(self, model_path=None, use_coral=False):
        """
        Lightweight face detector for Raspberry Pi
        Uses TFLite models for efficient inference
        """
        # Default to built-in Haar Cascade if no model provided
        if model_path is None:
            self.use_tflite = False
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Using OpenCV Haar Cascade for face detection")
        else:
            self.use_tflite = True
            self.load_tflite_model(model_path, use_coral)

        # Detection parameters
        self.min_confidence = 0.5
        self.last_detection_time = 0
        self.detection_interval = 0.1  # Detect every 100ms max

    def load_tflite_model(self, model_path, use_coral):
        """Load TFLite model with optional Coral support"""
        try:
            if use_coral:
                # Load delegate for Coral USB Accelerator
                delegates = [tflite.load_delegate('libedgetpu.so.1')]
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=delegates
                )
                print("Loaded model with Coral USB Accelerator")
            else:
                # Standard TFLite interpreter
                self.interpreter = tflite.Interpreter(model_path=model_path)
                # Use multiple threads for better performance
                self.interpreter.set_num_threads(4)
                print(f"Loaded TFLite model: {model_path}")

            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]

        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            print("Falling back to Haar Cascade")
            self.use_tflite = False
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade (fallback method)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Convert to normalized coordinates
        h, w = frame.shape[:2]
        detected_faces = []
        for (x, y, face_w, face_h) in faces:
            detected_faces.append({
                'bbox': [x/w, y/h, (x+face_w)/w, (y+face_h)/h],
                'confidence': 0.8  # Default confidence for Haar
            })

        return detected_faces

    def detect_faces_tflite(self, frame):
        """Detect faces using TFLite model"""
        # Resize frame for model input
        input_frame = cv2.resize(frame, (self.input_width, self.input_height))

        # Normalize if needed (depends on model)
        if self.input_details[0]['dtype'] == np.float32:
            input_frame = input_frame.astype(np.float32) / 255.0

        # Add batch dimension
        input_data = np.expand_dims(input_frame, axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        # Filter detections
        detected_faces = []
        for i in range(len(scores)):
            if scores[i] > self.min_confidence and classes[i] == 0:  # 0 = face class
                detected_faces.append({
                    'bbox': boxes[i],  # [ymin, xmin, ymax, xmax]
                    'confidence': scores[i]
                })

        return detected_faces

    def detect(self, frame):
        """Main detection function with rate limiting"""
        current_time = time.time()

        # Rate limiting to reduce CPU usage
        if current_time - self.last_detection_time < self.detection_interval:
            return []

        self.last_detection_time = current_time

        # Detect faces using appropriate method
        if self.use_tflite:
            faces = self.detect_faces_tflite(frame)
        else:
            faces = self.detect_faces_haar(frame)

        return faces

    def draw_faces(self, frame, faces):
        """Draw bounding boxes around detected faces"""
        h, w = frame.shape[:2]

        for face in faces:
            bbox = face['bbox']
            conf = face.get('confidence', 0.5)

            # Convert normalized coords to pixel coords
            if self.use_tflite:
                x1 = int(bbox[1] * w)  # xmin
                y1 = int(bbox[0] * h)  # ymin
                x2 = int(bbox[3] * w)  # xmax
                y2 = int(bbox[2] * h)  # ymax
            else:
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)

            # Draw rectangle
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw confidence
            label = f"Face: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def get_face_roi(self, frame, face):
        """Extract face region of interest"""
        h, w = frame.shape[:2]
        bbox = face['bbox']

        if self.use_tflite:
            x1 = max(0, int(bbox[1] * w))
            y1 = max(0, int(bbox[0] * h))
            x2 = min(w, int(bbox[3] * w))
            y2 = min(h, int(bbox[2] * h))
        else:
            x1 = max(0, int(bbox[0] * w))
            y1 = max(0, int(bbox[1] * h))
            x2 = min(w, int(bbox[2] * w))
            y2 = min(h, int(bbox[3] * h))

        return frame[y1:y2, x1:x2]