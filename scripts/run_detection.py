"""
run_detection.py — CLI script to run DocLayout-YOLO on a single image.

Usage:
    python scripts/run_detection.py --image path/to/doc.png [options]

Example:
    python scripts/run_detection.py \
        --image examples/singleDocImg.png \
        --model best.onnx \
        --output outputs/result.json \
        --save-image outputs/annotated.png
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layout_detector import DetectFunction, create_session, build_output_structure, save_output


def parse_args():
    parser = argparse.ArgumentParser(description="DocLayout-YOLO: Document Layout Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="best.onnx", help="Path to ONNX model weights")
    parser.add_argument("--classes", default="config/metadata.yaml", help="Path to class mapping YAML")
    parser.add_argument("--output", default="outputs/result.json", help="Path to save JSON output")
    parser.add_argument("--save-image", default=None, help="Path to save annotated image (optional)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold for NMS")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR text extraction")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"[INFO] Image loaded: {args.image} ({w}x{h})")

    # Build session
    print(f"[INFO] Loading model: {args.model}")
    session_args = create_session(args.model, args.classes)

    # Detect
    detector = DetectFunction(
        model_path=args.model,
        class_mapping_path=args.classes,
        original_size=(w, h),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    detections = detector.detect(image, session_args)
    print(f"[INFO] Detected {len(detections)} regions")

    # OCR
    if not args.no_ocr:
        from paddleocr import PaddleOCR
        print("[INFO] Running OCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        structured = build_output_structure(image, args.image, detections, ocr)
    else:
        # Build without OCR text
        structured = {}
        for det in detections:
            name = det["class_name"]
            x1, y1, x2, y2 = det["box"]
            record = {
                "coordinates": {"l": float(x1), "t": float(y1), "r": float(x2), "b": float(y2)},
                "accuracy": det["confidence"] * 100,
                "text": None,
            }
            structured.setdefault(name, []).append(record)

    # Save JSON
    save_output(structured, args.output)
    print(f"[INFO] JSON saved to: {args.output}")

    # Save annotated image
    if args.save_image:
        annotated = detector.draw_detections(
            image, detections, session_args["color_palette"], session_args["classes"]
        )
        os.makedirs(os.path.dirname(args.save_image) or ".", exist_ok=True)
        cv2.imwrite(args.save_image, annotated)
        print(f"[INFO] Annotated image saved to: {args.save_image}")

    # Print summary
    print("\n── Detection Summary ──")
    for cls_name, items in structured.items():
        print(f"  {cls_name}: {len(items)} region(s)")


if __name__ == "__main__":
    main()
