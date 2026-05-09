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
st.markdown("Upload a photo of Philippine banknotes and click **Detect** to identify each denomination.")

# ─────────────────────────────────────────────
# Load model (cached — only loads once)
# ─────────────────────────────────────────────
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file **{path}** not found. Place your trained `best.pt` next to `app.py`.")
        st.stop()
    return YOLO(path)

model = load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ── Predict button ──────────────────────────────────────────
    if st.button("🔍 Detect", use_container_width=True, type="primary"):
        with st.spinner("Detecting banknotes…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                img_path = os.path.join(tmpdir, "input.jpg")
                image.save(img_path, quality=95)

                results = model.predict(
                    source=img_path,
                    imgsz=320,       # fast inference
                    conf=0.75,
                    save=True,
                    project=tmpdir,
                    name="pred",
                )

                saved = (
                    glob.glob(os.path.join(tmpdir, "pred", "*.jpg"))
                    + glob.glob(os.path.join(tmpdir, "pred", "*.png"))
                )

                # ── Annotated image ─────────────────────────────
                st.subheader("🔍 Detection Result")
                if saved:
                    annotated = Image.open(saved[0])
                    st.image(annotated, use_container_width=True)
                else:
                    st.info("No annotated image saved.")

                # ── Summary ─────────────────────────────────────
                result = results[0]

                if result.boxes and len(result.boxes) > 0:
                    st.subheader("📊 Detection Summary")
                    names = model.names
                    counts: dict[str, int] = {}
                    rows = []

                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        label = names[cls_id]
                        conf_val = float(box.conf[0])
                        counts[label] = counts.get(label, 0) + 1
                        rows.append({"Label": label, "Confidence": f"{conf_val:.0%}"})

                    st.dataframe(rows, use_container_width=True)

                    st.markdown("**Count per denomination:**")
                    for label, cnt in sorted(counts.items()):
                        st.write(f"- **{label}**: {cnt}")
                else:
                    st.warning("No banknotes detected. Try a clearer photo with better lighting.")

else:
    st.info("👆 Upload an image to get started.")
