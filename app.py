import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model.h5")
labels = ["Metal", "Paper", "Plastic"]

st.title("♻️ Smart Waste Detector")

uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    resized = cv2.resize(img, (128, 128)) / 255.0
    reshaped = np.expand_dims(resized, axis=0)
    
    # Predict
    prediction = model.predict(reshaped)
    label = labels[np.argmax(prediction)]

    st.success(f"Detected: {label}")
