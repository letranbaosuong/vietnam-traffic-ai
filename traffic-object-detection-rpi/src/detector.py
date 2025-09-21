import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import time
from typing import List, Tuple, Dict


class TrafficObjectDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.traffic_class_ids = []
        self.load_model()

    def load_model(self):
        model_name = self.config['model']['name']

        if self.config['rpi_optimization']['use_onnx']:
            model_path = f"models/{model_name}.onnx"
            if not Path(model_path).exists():
                print(f"Converting {model_name} to ONNX format...")
                yolo_model = YOLO(f"{model_name}.pt")
                yolo_model.export(format='onnx', imgsz=self.config['model']['input_size'][0])
                Path(f"{model_name}.onnx").rename(model_path)

            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config['rpi_optimization']['num_threads']
            sess_options.inter_op_num_threads = self.config['rpi_optimization']['num_threads']

            self.model = ort.InferenceSession(model_path, sess_options, providers=providers)
        else:
            self.model = YOLO(f"{model_name}.pt")

        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        for cls_name in self.config['traffic_classes']:
            if cls_name in coco_classes:
                self.traffic_class_ids.append(coco_classes.index(cls_name))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        input_size = tuple(self.config['model']['input_size'])
        resized = cv2.resize(image, input_size)

        if self.config['rpi_optimization']['use_onnx']:
            blob = cv2.dnn.blobFromImage(resized, 1/255.0, input_size, swapRB=True, crop=False)
            return blob
        else:
            return resized

    def detect_onnx(self, image: np.ndarray) -> List[Dict]:
        preprocessed = self.preprocess(image)

        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: preprocessed})

        predictions = outputs[0][0]

        h, w = image.shape[:2]
        input_h, input_w = self.config['model']['input_size']

        detections = []
        conf_threshold = self.config['model']['confidence_threshold']

        for pred in predictions.T:
            x, y, width, height = pred[:4]
            scores = pred[4:]

            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and class_id in self.traffic_class_ids:
                x1 = int((x - width/2) * w / input_w)
                y1 = int((y - height/2) * h / input_h)
                x2 = int((x + width/2) * w / input_w)
                y2 = int((y + height/2) * h / input_h)

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': self.get_class_name(class_id)
                })

        return self.apply_nms(detections)

    def detect_pytorch(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image, conf=self.config['model']['confidence_threshold'])

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in self.traffic_class_ids:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(box.conf[0]),
                            'class_id': class_id,
                            'class_name': self.get_class_name(class_id)
                        })

        return detections

    def detect(self, image: np.ndarray) -> List[Dict]:
        if self.config['rpi_optimization']['use_onnx']:
            return self.detect_onnx(image)
        else:
            return self.detect_pytorch(image)

    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config['model']['confidence_threshold'],
            self.config['model']['nms_threshold']
        )

        if indices is not None and len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return []

    def get_class_name(self, class_id: int) -> str:
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img_copy = image.copy()

        colors = {
            'person': (255, 0, 0),
            'car': (0, 255, 0),
            'motorcycle': (0, 0, 255),
            'bus': (255, 255, 0),
            'truck': (255, 0, 255),
            'bicycle': (0, 255, 255),
            'traffic light': (128, 0, 128),
            'stop sign': (255, 128, 0)
        }

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            color = colors.get(class_name, (128, 128, 128))

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)

            cv2.putText(img_copy, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_copy