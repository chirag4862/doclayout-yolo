"""
app.py — Streamlit demo for DocLayout-YOLO
Run: streamlit run app.py
"""

import io
import json
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import urllib.request

from layout_detector import DetectFunction, create_session, build_output_structure

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocLayout-YOLO",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "best.onnx"
MODEL_URL = "https://github.com/chirag4862/doclayout-yolo/releases/download/v1.0.0/best.onnx"

@st.cache_resource(show_spinner="Downloading model weights…")
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights (~your_size MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return True


CLASSES_PATH = "config/metadata.yaml"

CLASS_COLORS = {
    "Title":          "#e74c3c",
    "Section-header": "#e67e22",
    "Text":           "#3498db",
    "Table":          "#2ecc71",
    "List-item":      "#9b59b6",
    "Caption":        "#1abc9c",
    "Page-header":    "#f39c12",
    "Page-footer":    "#95a5a6",
    "Footnote":       "#7f8c8d",
    "Picture":        "#e91e63",
    "Formula":        "#00bcd4",
}

# ── Session state ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_session():
    if not os.path.exists(MODEL_PATH):
        return None
    return create_session(MODEL_PATH, CLASSES_PATH)

@st.cache_resource(show_spinner="Loading OCR engine…")
def load_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/YOLO-Layout%20Detection-blue?style=for-the-badge")
    st.title("⚙️ Settings")

    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    iou_threshold  = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.4, 0.05)
    enable_ocr     = st.checkbox("Enable OCR text extraction", value=True)

    st.divider()
    st.markdown("**Detected classes**")
    for cls, color in CLASS_COLORS.items():
        st.markdown(f"<span style='color:{color}'>■</span> {cls}", unsafe_allow_html=True)

    st.divider()
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-black?logo=github)](https://github.com/your-username/doclayout-yolo)")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📄 DocLayout-YOLO")
st.markdown(
    "Upload a document page image to detect its layout regions — "
    "tables, headers, text blocks, figures, and more."
)

model_available = os.path.exists(MODEL_PATH)
if not model_available:
    st.warning(
        "⚠️ **Model weights not found.**  \n"
        "Place `best.onnx` in the project root to enable inference.  \n"
        "See the README for download instructions."
    )

uploaded = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg", "webp", "tiff"],
    disabled=not model_available,
)

if uploaded:
    # Decode uploaded image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w       = image_bgr.shape[:2]

    col_orig, col_anno = st.columns(2)
    with col_orig:
        st.subheader("Original")
        st.image(image_rgb, use_container_width=True)

    # ── Run detection ──────────────────────────────────────────────────────────
    with st.spinner("Running layout detection…"):
        session_args = load_session()
        detector = DetectFunction(
            model_path=MODEL_PATH,
            class_mapping_path=CLASSES_PATH,
            original_size=(w, h),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        detections = detector.detect(image_bgr, session_args)

    # ── OCR ────────────────────────────────────────────────────────────────────
    structured = {}
    if enable_ocr and detections:
        with st.spinner("Extracting text with OCR…"):
            ocr = load_ocr()
            structured = build_output_structure(image_bgr, "", detections, ocr)
    else:
        for det in detections:
            name = det["class_name"]
            x1, y1, x2, y2 = det["box"]
            structured.setdefault(name, []).append({
                "coordinates": {"l": float(x1), "t": float(y1), "r": float(x2), "b": float(y2)},
                "accuracy": det["confidence"] * 100,
                "text": None,
            })

    # ── Annotated image ────────────────────────────────────────────────────────
    annotated_bgr = detector.draw_detections(
        image_bgr, detections, session_args["color_palette"], session_args["classes"]
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with col_anno:
        st.subheader(f"Detected ({len(detections)} regions)")
        st.image(annotated_rgb, use_container_width=True)

    # ── Download buttons ───────────────────────────────────────────────────────
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        _, buf = cv2.imencode(".png", annotated_bgr)
        st.download_button(
            "⬇️ Download annotated image",
            data=buf.tobytes(),
            file_name="annotated.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        json_str = json.dumps(structured, indent=2)
        st.download_button(
            "⬇️ Download JSON output",
            data=json_str,
            file_name="layout_output.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Structured results ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Structured Output")

    # Summary metrics
    metric_cols = st.columns(min(len(structured), 5))
    for i, (cls_name, items) in enumerate(structured.items()):
        with metric_cols[i % 5]:
            color = CLASS_COLORS.get(cls_name, "#888")
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:8px;border-radius:4px;margin-bottom:8px'>"
                f"<b style='color:{color}'>{cls_name}</b><br>"
                f"<span style='font-size:1.4em'>{len(items)}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Per-class expandable sections
    for cls_name, items in structured.items():
        color = CLASS_COLORS.get(cls_name, "#888")
        with st.expander(f"**{cls_name}** — {len(items)} region(s)", expanded=False):
            for idx, item in enumerate(items):
                coords = item["coordinates"]
                st.markdown(
                    f"**Region {idx+1}** &nbsp; "
                    f"<span style='color:gray;font-size:0.85em'>"
                    f"({int(coords['l'])}, {int(coords['t'])}) → ({int(coords['r'])}, {int(coords['b'])})"
                    f" | conf: {item['accuracy']:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                if item.get("text"):
                    st.code(item["text"], language=None)
                elif item["text"] is None:
                    st.caption("_(visual region — OCR skipped)_")
                else:
                    st.caption("_(no text detected)_")

    # Raw JSON viewer
    with st.expander("🔍 Raw JSON", expanded=False):
        st.json(structured)

else:
    # Landing placeholder
    st.info("👆 Upload a document image to get started.")
    st.markdown("""
    **What this model detects:**
    | Class | Description |
    |---|---|
    | Title | Document title |
    | Section-header | Section headings |
    | Text | Body text paragraphs |
    | Table | Tabular data |
    | List-item | Bulleted / numbered items |
    | Caption | Figure/table captions |
    | Page-header | Running header |
    | Page-footer | Running footer |
    | Footnote | Footnote text |
    | Picture | Images and figures |
    | Formula | Mathematical formulas |
    """)
