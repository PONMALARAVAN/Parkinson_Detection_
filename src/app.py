import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import sys
import os

# Allow importing other src files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_image
from gradcam import gradcam
from utils import overlay_heatmap


# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "../saved_models/cnn_model.h5"     # model outside src folder

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Parkinson's Spiral Detection", layout="centered")
st.title("🌀 Parkinson's Disease Detection using Spiral Drawings")
st.write("Upload a spiral drawing to predict if the patient has Parkinson’s disease.")


uploaded_file = st.file_uploader("Upload Spiral Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Spiral Drawing", width=300)

    # Preprocess
    processed_img = preprocess_image(img, img_size=224)
    expanded_img = np.expand_dims(processed_img, axis=0)

    # Prediction
    pred = model.predict(expanded_img)[0][0]
    label = "Parkinson" if pred > 0.5 else "Healthy"
    confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"### Result: **{label}**")
    st.write(f"### Confidence: **{confidence:.2f}%**")

    # Grad-CAM
    st.markdown("---")
    st.subheader("🔥 Grad-CAM Visualization")

    heatmap = gradcam(model, expanded_img)

    gray_img = processed_img.squeeze()
    combined = overlay_heatmap(gray_img, heatmap)

    st.image(combined, caption="Grad-CAM Heatmap", width=300)

st.markdown("---")
st.write("Developed by **Pon Malaravan** — Parkinson’s Spiral Detection Project")
