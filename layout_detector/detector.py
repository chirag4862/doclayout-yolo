"""
DetectFunction: YOLO ONNX-based page layout detector.
Handles preprocessing, inference, postprocessing, and visualization.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class DetectFunction:
    """
    Performs layout detection on document images using a YOLO ONNX model.

    Args:
        model_path: Path to the .onnx model weights.
        class_mapping_path: Path to the metadata.yaml with class names.
        original_size: (width, height) of the input image.
        score_threshold: Minimum score for NMS filtering.
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        device: Inference device — "CPU" or "CUDA".
    """

    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        original_size: Tuple[int, int] = (1280, 720),
        score_threshold: float = 0.1,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.4,
        device: str = "CPU",
    ) -> None:
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.device = device
        self.score_threshold = score_threshold
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = original_size

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, img: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
        """Resize and normalize image to model input format (NCHW, float32, 0-1)."""
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (input_width, input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def postprocess(
        self,
        outputs: np.ndarray,
        input_width: int,
        input_height: int,
        classes: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Filter, rescale, and NMS the raw ONNX outputs.

        Returns:
            List of dicts with keys: class_index, confidence, box, class_name.
        """
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)

        indices = cv2.dnn.NMSBoxes(
            boxes, scores,
            score_threshold=self.score_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections = []
        for bbox, score, label in zip(
            self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
        ):
            detections.append({
                "class_index": int(label),
                "confidence": float(score),
                "box": bbox,
                "class_name": classes[label],
            })
        return detections

    # ------------------------------------------------------------------
    # Full inference pipeline
    # ------------------------------------------------------------------

    def detect(self, img: np.ndarray, session_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run end-to-end detection on a BGR image.

        Args:
            img: Input image (BGR numpy array).
            session_args: Dict returned by `create_session()` containing the
                          ONNX session, I/O names, dimensions, and class list.

        Returns:
            List of detection dicts.
        """
        input_tensor = self.preprocess(img, session_args["input_width"], session_args["input_height"])
        outputs = session_args["session"].run(
            session_args["output_names"],
            {session_args["input_names"][0]: input_tensor},
        )[0]
        return self.postprocess(
            outputs,
            session_args["input_width"],
            session_args["input_height"],
            session_args["classes"],
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw_detections(
        self,
        img: np.ndarray,
        detections: List[Dict[str, Any]],
        color_palette: np.ndarray,
        classes: List[str],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of the image.

        Returns:
            Annotated image (BGR numpy array).
        """
        annotated = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"].astype(int)
            class_id = det["class_index"]
            confidence = det["confidence"]
            color = color_palette[class_id].tolist()

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x, label_y = x1, y1 - 10 if y1 - 10 > lh else y1 + 10
            cv2.rectangle(annotated, (label_x, label_y - lh), (label_x + lw, label_y + lh), color, cv2.FILLED)
            cv2.putText(annotated, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated
