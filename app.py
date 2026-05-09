import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
import tempfile
import os
import glob

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Philippine Money Detector",
    page_icon="💵",
    layout="centered",
)

st.title("💵 Philippine Money Detector")
st.markdown(
    "Upload a photo of Philippine banknotes and the model will detect "
    "and label each denomination."
)

# ─────────────────────────────────────────────
# Load model (cached so it only loads once)
# ─────────────────────────────────────────────
MODEL_PATH = "best.pt"   # put your trained best.pt next to app.py

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(
            f"Model file **{path}** not found. "
            "Please place your trained `best.pt` in the same folder as `app.py`."
        )
        st.stop()
    return YOLO(path)

model = load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# Sidebar settings
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider(
    "Confidence threshold", min_value=0.10, max_value=1.00, value=0.75, step=0.05
)
imgsz = st.sidebar.selectbox("Inference image size (px)", [320, 416, 512, 640], index=2)

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    # Pre-process exactly like the notebook
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original")
        st.image(image, use_container_width=True)

    # Run inference
    with st.spinner("Detecting banknotes…"):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.jpg")
            image.save(img_path, quality=95)

            results = model.predict(
                source=img_path,
                imgsz=imgsz,
                conf=confidence,
                save=True,
                project=tmpdir,
                name="pred",
            )

            # Grab saved annotated image
            saved = (
                glob.glob(os.path.join(tmpdir, "pred", "*.jpg"))
                + glob.glob(os.path.join(tmpdir, "pred", "*.png"))
            )

            with col2:
                st.subheader("🔍 Detections")
                if saved:
                    annotated = Image.open(saved[0])
                    st.image(annotated, use_container_width=True)
                else:
                    st.info("No annotated image saved — check model output.")

    # ── Detection summary ──────────────────────────────────────
    st.subheader("📊 Detection Summary")
    result = results[0]

    if result.boxes and len(result.boxes) > 0:
        boxes = result.boxes
        names = model.names          # class id → label

        counts: dict[str, int] = {}
        rows = []
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            conf_val = float(box.conf[0])
            counts[label] = counts.get(label, 0) + 1
            rows.append({"Label": label, "Confidence": f"{conf_val:.0%}"})

        # Table of every detected box
        st.dataframe(rows, use_container_width=True)

        # Count per denomination
        st.markdown("**Count per denomination:**")
        for label, cnt in sorted(counts.items()):
            st.write(f"- **{label}**: {cnt}")
    else:
        st.info(
            "No banknotes detected at the current confidence threshold. "
            "Try lowering the threshold in the sidebar."
        )

else:
    st.info("👆 Upload an image to get started.")