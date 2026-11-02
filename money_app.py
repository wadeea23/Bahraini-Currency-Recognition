import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av

st.title("Bahraini Currency Recognition")

# Load model
model = YOLO("best.pt")

# Upload Image
st.header("Upload Image")

file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")

    result = model(img)[0]

    st.image(result.plot(), caption="Prediction")


# Live Camera
st.header("Live Camera")

def camera(frame):
    img = frame.to_ndarray(format="bgr24")

    result = model(img)[0]

    return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")


webrtc_streamer(
    key="camera",
    video_frame_callback=camera
)
