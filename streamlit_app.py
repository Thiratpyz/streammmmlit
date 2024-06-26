import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8m-pose.pt")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process frame with YOLOv8 model
        results = self.model(img, conf=0.3)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("Real-time YOLOv8 Object Detection with Streamlit")
st.text("Using YOLOv8 model with Streamlit and streamlit-webrtc")

# Configuration for the STUN/TURN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": "stun:stun.relay.metered.ca:80"},
        {"urls": "turn:sg.relay.metered.ca:80", "username": "60af5bc4849e559a1aaf1dbe", "credential": "WPw2tHgvHBnuHSUi"},
        {"urls": "turn:sg.relay.metered.ca:80?transport=tcp", "username": "60af5bc4849e559a1aaf1dbe", "credential": "WPw2tHgvHBnuHSUi"},
        {"urls": "turn:sg.relay.metered.ca:443", "username": "60af5bc4849e559a1aaf1dbe", "credential": "WPw2tHgvHBnuHSUi"},
        {"urls": "turns:sg.relay.metered.ca:443?transport=tcp", "username": "60af5bc4849e559a1aaf1dbe", "credential": "WPw2tHgvHBnuHSUi"}
    ]
})

# Initialize the webrtc_streamer and retrieve the context
ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

# Set the model for the video processor if it exists
if ctx and ctx.video_processor:
    ctx.video_processor.model = model
