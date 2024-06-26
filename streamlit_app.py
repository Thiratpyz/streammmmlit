import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolov8m-pose.pt")

st.title("Real-time YOLOv8 Object Detection")
st.text("Using YOLOv8 model with Streamlit")

# Function to process each frame
def process_frame(frame):
    results = model(frame, conf=0.3)
    annotated_frame = results[0].plot()  # Get the annotated frame
    return annotated_frame

# Start video capture
cap = cv2.VideoCapture(0)

stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = process_frame(frame)

    stframe.image(processed_frame, channels="RGB")

cap.release()
