import streamlit as st
import av
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8m-pose.pt")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_image()  # Convert to PIL Image

        # Convert PIL Image to numpy array
        img_np = np.array(img)

        # Process frame with YOLOv8 model
        results = self.model(img_np, conf=0.3)
        annotated_frame = results[0].plot()

        # Convert numpy array back to PIL Image
        annotated_img = Image.fromarray(annotated_frame)

        return av.VideoFrame.from_image(annotated_img)

st.title("Real-time YOLOv8 Object Detection with Streamlit")
st.text("Using YOLOv8 model with Streamlit and streamlit-webrtc")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor, mode=WebRtcMode.SENDRECV, media_stream_constraints={"video": True, "audio": False})
