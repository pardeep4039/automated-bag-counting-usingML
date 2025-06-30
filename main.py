import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import torch
from collections import defaultdict
import supervision as sv

# Load the trained YOLOv8 model
model = YOLO('best.pt')  

#Streamlit UI
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("AUTOMATED BAG COUNTING USING MACHINE LEARNING")

# Header buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Home"):
        st.session_state.page = "Home"

with col2:
    if st.button("About Us"):
        st.session_state.page = "About Us"

with col3:
    if st.button("Detection"):
        st.session_state.page = "Detection"

# Default page setup
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Page content
if st.session_state.page == "Home":
    st.subheader("Welcome to the Home Page")
    st.markdown("We propose a novel approach to automate the process of counting bags loaded on trucks using machine learning techniques with an accuracy of 94%. \n\n Our proposed solution leverages computer vision algorithms to analyze images or video footage captured during the loading process. By employing object detection and recognition methods, coupled with deep learning models, we aim to accurately detect and track and count the bags as they are being loaded onto trucks. \n\n The key objectives of this project include the development of a robust machine learning model capable of accurately counting bags in real-time, the integration of the model into a user- friendly software interface")

elif st.session_state.page == "About Us":
    st.subheader("About Us")
    st.markdown("In this study, we implemented YOLOv8 for automated bag counting and detection. \n\n Extensive evaluations indicate that YOLOv8 significantly enhances detection accuracy and computational efficiency. \n\n A comprehensive video dataset was compiled from a local grain godown, where active loading and unloading processes took place. \n\n Notably, the video includes sequences from both daytime and nighttime, thereby addressing potential challenges associated with varying lighting conditions. \n\n Object counting is performed using a line-crossing detection algorithm, where objects are counted only when they cross a predefined virtual line in the frame.Objects are counted only if their confidence score exceeds 0.3. The total count is dynamically updated and displayed on each frame. The processed video is shown with bounding boxes, track IDs, and count annotations for analysis. \n\n This method ensures accurate and efficient counting of objects, making it suitable for applications like inventory management, traffic monitoring, and crowd analysis.")

elif st.session_state.page == "Detection":
    st.subheader("Detection Page")

    #Sidebar
    st.sidebar.title("Upload")
    option = st.sidebar.selectbox("Select Input Type:", ["Image", "Video", "Webcam"])

    # Image input
    if option == "Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Run Detection"):
                with st.spinner("Detecting..."):
                    results = model.predict(image, conf=0.25)
                    result_image = results[0].plot()
                    st.image(result_image, caption="Detected", use_container_width=True)

    # Video input
    elif option == "Video":
        uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                video_path = tmp_video.name

            cap = cv2.VideoCapture(video_path)

            START = sv.Point(15, 160)
            END = sv.Point(1414, 950)

            track_history = defaultdict(lambda: [])
            crossed_objects = {}

            stframe = st.empty()
            st.info("Running Detection and Counting...")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = model.track(frame, classes=[15], conf=0.3, iou=0.5, persist=True, tracker="bytetrack.yaml")

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                annotated_frame = results[0].plot()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    if START.x < x < END.x and abs(y - START.y) < 5:
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True

                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)
                count_text = f"Objects crossed: {len(crossed_objects)}"
                cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
            st.success("Processing complete.")
            st.metric("Total Count of bags: ", len(crossed_objects))

    #Webcam input
    if option == "Webcam":

        START = sv.Point(15, 160)
        END = sv.Point(1414, 950)

        track_history = defaultdict(lambda: [])
        crossed_objects = {}

        stframe = st.empty()
        st.info("Starting Webcam...")

        cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, classes=[15], conf=0.3, iou=0.5, persist=True, tracker="bytetrack.yaml")

            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                if START.x < x < END.x and abs(y - START.y) < 5:
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = True

                    cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)
            count_text = f"Objects crossed: {len(crossed_objects)}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Webcam processing complete.")
        st.metric("Total Count", len(crossed_objects))
