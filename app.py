import streamlit as st
import torch
from PIL import Image
from app.inference import BreastHistoModel
import time

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "models/resnet50_bh_e1_ts_probs.pt"
CLASS_NAMES = ["benign", "malignant"]

# ------------------------------
# PAGE SETTINGS
# ------------------------------
st.set_page_config(
    page_title="Breast Histopathology Classifier",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return BreastHistoModel(model_path=MODEL_PATH, class_names=CLASS_NAMES)

model = load_model()

# ------------------------------
# HEADER
# ------------------------------
st.title("🩺 Breast Histopathology Classifier")
st.write("""
Upload one or more histopathology images (magnification 400x) and the model will
analyze them to determine the likelihood of **benign** or **malignant** tissue.  
This tool uses a ResNet50 model fine-tuned on breast histopathology image datasets.
""")

# ------------------------------
# INFO SECTION
# ------------------------------
with st.expander("ℹ️ How It Works", expanded=False):
    st.write("""
    1. **Upload Images** – You can upload multiple `.jpg`, `.jpeg`, or `.png` files at once.  
    2. **Model Prediction** – The model processes each image individually and calculates probabilities.  
    3. **Final Aggregate** – If multiple images are uploaded, the results are averaged to provide one overall prediction.  
    4. **Confidence Score** – Probabilities show how confident the model is in its prediction.  
    """)
    st.warning("This tool is for research & educational purposes only. Not for clinical diagnosis.")

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_files = st.file_uploader(
    "📤 Upload histopathology image(s)...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# ------------------------------
# PREDICTION
# ------------------------------
if uploaded_files:
    progress = st.progress(0)
    status_text = st.empty()

    all_probs = []
    num_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = model.transform(image).unsqueeze(0).to(model.device)
        probs, _, _ = model.predict(img_tensor, return_label=True)
        all_probs.append(torch.tensor(probs))

        # Simulate loading
        progress.progress((idx + 1) / num_files)
        status_text.text(f"Processing {idx+1}/{num_files} images...")
        time.sleep(0.2)

    # Final aggregate
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    final_class_idx = torch.argmax(avg_probs).item()
    final_label = CLASS_NAMES[final_class_idx]

    # ------------------------------
    # RESULTS
    # ------------------------------
    st.subheader("🧾 Final Prediction")
    st.success(f"**{final_label.upper()}**")

    st.subheader("Confidence Scores")
    st.write({
        CLASS_NAMES[i]: f"{avg_probs[i]:.4f}" for i in range(len(CLASS_NAMES))
    })

    # Optional: Show images in a grid
    st.subheader("Uploaded Images")
    cols = st.columns(3)
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 3]:
            st.image(uploaded_file, use_container_width=True)

    progress.empty()
    status_text.empty()


