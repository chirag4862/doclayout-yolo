"""
OCR extraction helpers and structured JSON output builder.
"""

import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Optional

# Classes that are purely visual — skip OCR for these
_SKIP_OCR_CLASS_IDS = {1, 8, 9}  # Table, Picture, Formula


def extract_text_from_region(
    image: np.ndarray,
    box: np.ndarray,
    class_index: int,
    ocr,
    padding: int = 10,
) -> Optional[str]:
    """
    Crop a detected region and run PaddleOCR on it.

    Args:
        image: Full BGR image.
        box: (x1, y1, x2, y2) bounding box.
        class_index: Detected class index (some classes skip OCR).
        ocr: Initialized PaddleOCR instance.
        padding: Pixels of padding to add around the crop.

    Returns:
        Extracted text string, or None for visual-only regions.
    """
    if class_index in _SKIP_OCR_CLASS_IDS:
        return None

    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]

    # Clamp with padding
    x1c = max(0, x1 - padding)
    y1c = max(0, y1 - padding)
    x2c = min(w, x2 + padding)
    y2c = min(h, y2 + padding)

    crop = image[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return ""

    result = ocr.ocr(crop)
    if not result or result == [None]:
        return ""

    lines = []
    for block in result:
        if block:
            for line in block:
                lines.append(line[1][0])
    return "\n".join(lines)


def build_output_structure(
    image: np.ndarray,
    image_path: str,
    detections: List[Dict[str, Any]],
    ocr,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert raw detections into the structured JSON format, grouping by class.

    Each entry has:
        - coordinates: {l, t, r, b} as floats
        - accuracy: confidence * 100
        - text: OCR result (None for visual classes)

    Returns:
        Dict[class_name -> list of detection records], sorted top-to-bottom.
    """
    output: Dict[str, List[Dict[str, Any]]] = {}

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_name = det["class_name"]
        class_index = det["class_index"]

        text = extract_text_from_region(image, det["box"], class_index, ocr)

        record = {
            "coordinates": {"l": float(x1), "t": float(y1), "r": float(x2), "b": float(y2)},
            "accuracy": det["confidence"] * 100,
            "text": text,
        }

        output.setdefault(class_name, []).append(record)

    # Sort each class group top-to-bottom, left-to-right
    for key in output:
        output[key] = sorted(output[key], key=lambda x: (x["coordinates"]["t"], x["coordinates"]["l"]))

    return output


def save_output(data: Dict[str, Any], output_path: str) -> None:
    """Write the structured output dict to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
