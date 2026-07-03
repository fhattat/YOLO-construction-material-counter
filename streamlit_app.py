"""
Construction Site Material Counter — Streamlit UI
=================================================
Upload a photo -> SAHI sliced detection -> per-class counts + annotated image.

Run locally:
    C:\\Users\\pc\\insaat-sayim\\.venv\\Scripts\\streamlit.exe run streamlit_app.py

Defaults were tuned on field tests (128/132 = 97%):
confidence=0.2, slice=512, overlap=0.3 + auto-upscaling of small images.
"""

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Resolve best.pt next to this file regardless of working directory
MODEL_PATH = str(Path(__file__).parent / "best.pt")

st.set_page_config(page_title="Construction Material Counter", page_icon="🏗️", layout="wide")
st.title("🏗️ Construction Site Material Counter")
st.markdown(
    "<p style='font-size:22px; color:#5c5c5c; margin-top:-8px;'>"
    "YOLO26 + SAHI — sliced inference for dense, overlapping objects</p>",
    unsafe_allow_html=True,
)

# --- What this app does + example image ---
intro_col, img_col = st.columns([3, 2])
with intro_col:
    st.markdown(
        """
        #### What does this app do?
        Counting stacked **PVC pipes** on a construction site by hand is slow and
        error-prone. This app does it from a single photo:

        1. **Upload** a photo of a pipe stack
        2. A **YOLO26** model trained on pipe imagery scans it **slice by slice (SAHI)**,
           so even small, distant pipe ends are detected
        3. You get the **total count** plus an annotated image where every
           detected pipe is marked

        Field accuracy: **~97%** on dense stacks of 130+ pipes.
        """
    )
with img_col:
    sample = Path(__file__).parent / "test.png"
    if sample.exists():
        st.image(str(sample), caption="Example input: stacked PVC pipes", use_container_width=True)

st.markdown("---")


@st.cache_resource
def load_model(path, conf):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=path,
        confidence_threshold=conf,
        device="cpu",  # set to "cuda:0" on a GPU machine
    )


# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.2, 0.05)
    slice_px = st.select_slider("Slice size (px)", [384, 512, 640, 768], value=512)
    overlap = st.slider("Slice overlap", 0.1, 0.4, 0.3, 0.05)
    st.markdown("---")
    st.markdown(
        "**Tips:** If pipes look small, reduce the slice size (384). "
        "If you see false detections, raise the confidence threshold (0.3)."
    )

uploaded = st.file_uploader("Upload a site photo", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Small images hurt counting: upscale to give SAHI room to slice
    # (field test: a 469px image jumped from 95 to 128 detections at 3x)
    if max(img.size) < 1500:
        scale = 3 if max(img.size) < 800 else 2
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        st.info(f"Image was small — upscaled {scale}x to {img.width}×{img.height}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        img_path = tmp.name

    with st.spinner("Running sliced detection..."):
        model = load_model(MODEL_PATH, conf)
        result = get_sliced_prediction(
            img_path,
            model,
            slice_height=slice_px,
            slice_width=slice_px,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
        )
        preds = result.object_prediction_list
        counts = Counter(p.category.name for p in preds)

        # annotate
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for p in preds:
            x1, y1, x2, y2 = map(int, p.bbox.to_xyxy())
            cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # --- results ---
    st.subheader("Counting results")
    if counts:
        cols = st.columns(len(counts))
        for col, (cls, n) in zip(cols, counts.items()):
            col.metric(cls, n)
        st.metric("TOTAL", sum(counts.values()))
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", use_container_width=True)
    c2.image(arr, caption="Annotated", use_container_width=True)
else:
    st.info("Upload a site photo to get started.")
