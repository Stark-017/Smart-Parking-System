from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from parking_core import build_class_map, draw_detections, load_model_info, parse_ultralytics_result

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "models" / "best.pt"
DEFAULT_INFO = APP_DIR / "models" / "model_info.json"
MAX_IMAGE_MB = 20
MAX_MODEL_MB = 200
Image.MAX_IMAGE_PIXELS = 60_000_000

st.set_page_config(page_title="Smart Parking Detection", page_icon="🅿️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {font-family:'Barlow',system-ui,sans-serif;}
h1,h2,h3 {font-family:'Barlow Condensed','Barlow',sans-serif !important;}
.block-container {padding-top:2rem;max-width:1200px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained model…")
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def save_uploaded_model(uploaded) -> Path:
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()[:20]
    target = Path(tempfile.gettempdir()) / f"parking_model_{digest}.pt"
    if not target.exists():
        target.write_bytes(raw)
    return target


def show_header(free="—", occupied="—", note="Upload a parking-lot photo to detect free and occupied spaces."):
    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem">
      <div style="flex:1 1 360px"><h1 style="font-size:2.7rem;margin:0">Smart Parking Detection</h1><p>{note}</p></div>
      <div style="background:#2B2D2F;border-left:8px solid #E8C547;color:white;padding:1rem 1.4rem;display:flex;gap:1.5rem;align-items:center">
        <div style="font-size:2rem;background:#1F5FAD;padding:.1rem .6rem;border-radius:6px">P</div>
        <div><div style="font-size:2rem;color:#5FD38D;font-weight:700">{free}</div><div>Free bays</div></div>
        <div><div style="font-size:2rem;color:#FF7A6E;font-weight:700">{occupied}</div><div>Occupied</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Model & settings")
    st.caption("Upload your trained YOLO .pt weights here, or include models/best.pt in the GitHub repository.")
    uploaded_model = st.file_uploader("Trained model weights (.pt)", type=["pt"], help="Upload the best.pt file exported after training. Dataset ZIPs are not model weights.")
    uploaded_info = st.file_uploader("Optional model metadata (.json)", type=["json"])
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("Overlap (IoU) threshold", 0.10, 0.90, 0.50, 0.05)
    show_labels = st.checkbox("Show labels on boxes", value=False)

model_path = None
if uploaded_model is not None:
    if uploaded_model.size > MAX_MODEL_MB * 1024 * 1024:
        st.error(f"Model file is larger than {MAX_MODEL_MB} MB. Use external model storage or commit it using Git LFS.")
        st.stop()
    model_path = save_uploaded_model(uploaded_model)
elif DEFAULT_MODEL.exists():
    model_path = DEFAULT_MODEL

show_header()

if model_path is None:
    st.warning("No trained model is loaded yet. Upload your trained `best.pt` file in the sidebar, or add it to the repository at `models/best.pt`.")
    st.info("A dataset ZIP is used to train a model; it cannot be used directly for predictions. Train the model in Jupyter/Colab first, then upload the exported `best.pt` here.")
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.caption("Check that this is a valid Ultralytics YOLO .pt file and that the package versions are compatible.")
    st.stop()

if uploaded_info is not None:
    try:
        info = json.loads(uploaded_info.getvalue().decode("utf-8"))
    except Exception:
        info = {}
        st.warning("The optional model_info.json could not be read; using model defaults.")
elif DEFAULT_INFO.exists():
    info = load_model_info(DEFAULT_INFO.parent)
else:
    info = {}

names_raw = model.names or {}
names = {int(k): str(v) for k, v in names_raw.items()} if isinstance(names_raw, dict) else {i: str(v) for i, v in enumerate(names_raw)}
cmap = build_class_map(names)
task = getattr(model, "task", "detect")
imgsz = int(info.get("imgsz", 224 if task == "classify" else 640))

total_bays = None
if task == "detect" and cmap["mode"] == "vehicle_only":
    with st.sidebar:
        st.markdown("---")
        st.caption("This model detects vehicles only. Enter the total visible bays to estimate free spaces.")
        total_input = st.number_input("Total bays in image", min_value=0, value=0, step=1)
        total_bays = int(total_input) if total_input > 0 else None

uploaded_image = st.file_uploader("Upload a parking-lot image", type=["jpg", "jpeg", "png"])
if uploaded_image is None:
    st.caption("Supported formats: JPG, JPEG and PNG. Upload one image to run inference.")
    st.stop()
if uploaded_image.size > MAX_IMAGE_MB * 1024 * 1024:
    st.error(f"Image exceeds the {MAX_IMAGE_MB} MB upload limit.")
    st.stop()
try:
    raw = uploaded_image.getvalue()
    Image.open(io.BytesIO(raw)).verify()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
    st.error(f"Could not read that image ({type(exc).__name__}). Try another JPG or PNG.")
    st.stop()

with st.spinner("Detecting parking spaces…"):
    try:
        result_raw = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, max_det=1000, verbose=False)[0]
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

result = parse_ultralytics_result(result_raw, names, cmap)
if result.task == "classify":
    label = (result.top_label or "Unknown").replace("_", " ")
    group = result.top_group or "other"
    verdict = "EMPTY" if group == "empty" else "OCCUPIED" if group == "occupied" else label.upper()
    show_header("1" if group == "empty" else "0", "1" if group == "occupied" else "0", "Image classification result.")
    c1, c2 = st.columns(2)
    c1.image(image, caption="Uploaded image", use_container_width=True)
    with c2:
        st.subheader(f"Prediction: {verdict}")
        st.metric("Confidence", f"{(result.top_conf or 0):.1%}")
        st.bar_chart({k: [v] for k, v in result.probs.items()})
else:
    n_free, n_occupied = result.n_empty, result.n_occupied
    if cmap["mode"] == "vehicle_only":
        free_display = str(max(total_bays - n_occupied, 0)) if total_bays else "?"
        note = "Vehicles detected. Free spaces are estimated only when you enter the total number of bays."
    else:
        free_display = str(n_free)
        note = "Green boxes indicate free spaces; red boxes indicate occupied spaces."
    show_header(free_display, str(n_occupied), note)
    if cmap["mode"] == "occupancy" and n_free + n_occupied > 0:
        st.success(f"{n_free} of {n_free+n_occupied} detected bays are free ({n_occupied/(n_free+n_occupied):.0%} occupied)." if n_free else "No free bays detected in this image.")
    elif not result.detections:
        st.warning("No parking spaces were detected. Try a clear overhead/elevated parking-lot image or lower the confidence threshold.")
    annotated = draw_detections(image, result, show_labels=show_labels)
    c1, c2 = st.columns(2)
    c1.image(image, caption="Original image", use_container_width=True)
    c2.image(annotated, caption="Detection result", use_container_width=True)
    out = io.BytesIO()
    annotated.save(out, format="PNG")
    st.download_button("Download annotated image", out.getvalue(), file_name="parking_prediction.png", mime="image/png")
    if result.detections:
        st.caption(f"{len(result.detections)} detections · mean confidence {sum(d.conf for d in result.detections)/len(result.detections):.2f}")
        with st.expander("Detection details"):
            st.dataframe([{"class": d.name, "status": d.group, "confidence": round(d.conf, 3), "x1": int(d.box[0]), "y1": int(d.box[1]), "x2": int(d.box[2]), "y2": int(d.box[3])} for d in result.detections], use_container_width=True)

with st.expander("About this model"):
    st.write(f"**Task:** {task}")
    st.write(f"**Model:** {info.get('base_model', 'Ultralytics YOLO')}")
    st.write("**Classes:** " + ", ".join(names.values()))
    if info.get("test_metrics"):
        st.write("**Test metrics:**")
        st.json(info["test_metrics"])
