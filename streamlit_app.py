import cv2
import numpy as np
import streamlit as st
import threading
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
import subprocess

from fish_classifier import FishClassifier
from fish_counter import FishCounter

class FishClassificationApp:
    def __init__(self):
        self.is_running = False
        self.use_webcam = True
        self.video_path = None
        self.cap = None
        self.classifier = None
        self.detector = None
        self.counter = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = 0
        self.webcam_index = 0
        
        # Initialize session state
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.is_running = False
            st.session_state.frame_placeholder = None
            st.session_state.current_frame = None
            
        self.load_model()
    
    def load_model(self):
        try:
            # Load classification model
            if os.path.exists('fish_classifier.pt'):
                self.classifier = FishClassifier('fish_classifier.pt', 'class_names.txt', confidence_threshold=0.5)
                st.success("Classification model loaded successfully.")
            else:
                st.warning("Classification model not found.")
                self.classifier = None
            
            # Load YOLOv8 detection model
            try:
                if os.path.exists('yolov8n.pt'):
                    self.detector = YOLO('yolov8n.pt')
                    st.success("YOLOv8 detection model loaded successfully.")
                else:
                    st.info("Downloading YOLOv8 detection model...")
                    self.detector = YOLO('yolov8n.pt')
            except Exception as detector_error:
                st.error(f"Could not load detector: {detector_error}")
                self.detector = None
            
            # Initialize counter
            try:
                self.counter = FishCounter()
                st.success("Fish counter initialized successfully.")
            except ImportError as import_error:
                st.warning(f"Counter dependency missing: {import_error}")
                st.info("Install supervision with: uv add supervision")
                self.counter = None
            except Exception as counter_error:
                import traceback
                st.error(f"Could not initialize counter: {counter_error}")
                st.error(f"Traceback: {traceback.format_exc()}")
                self.counter = None
            
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            self.detector = None
            self.classifier = None
            self.counter = None
    
    def detect_fish(self, frame, confidence_threshold):
        detections = []
        
        # Check if detector is available
        if self.detector is None:
            return detections
        
        # Use YOLOv8 to detect objects
        results = self.detector(frame, conf=confidence_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Extract region of interest (ROI)
                roi = frame[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Classify ROI if classifier is available
                if self.classifier:
                    species, confidence = self.classifier.classify(roi)
                else:
                    # Use class predicted by YOLOv8 if no specific classifier
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    species = result.names[class_id]
                
                if species:
                    bbox = (x, y, w, h)
                    detections.append((bbox, species, confidence))
        
        return detections
    
    def draw_detections(self, frame, detections, tracked_fish):
        for detection in detections:
            bbox, species, confidence = detection
            x, y, w, h = bbox
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display species and confidence
            label = f"{species}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw IDs of tracked fish
        for fish_id, (centroid, species, confidence) in tracked_fish.items():
            cx, cy = centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(fish_id), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def process_frame(self, frame, confidence_threshold):
        # Detect fish
        detections = self.detect_fish(frame, confidence_threshold)
        
        # Update counter if available
        if self.counter is not None:
            tracked_fish = self.counter.update(detections, frame)
        else:
            tracked_fish = {}
        
        # Draw detections
        processed_frame = self.draw_detections(frame.copy(), detections, tracked_fish)
        
        return processed_frame, len(detections)

def main():
    st.set_page_config(
        page_title="Fish Classification System",
        layout="wide"
    )
    
    st.title("Fish Classification System")
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = FishClassificationApp()
    
    app = st.session_state.app
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Video source selection
    video_source = st.sidebar.radio(
        "Video Source",
        ["Webcam", "Upload Video"]
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Update classifier confidence if available
    if app.classifier:
        app.classifier.confidence_threshold = confidence_threshold
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Feed")
        
        # Video display placeholder
        frame_placeholder = st.empty()
        
        # Show placeholder when not running
        if not st.session_state.get('is_running', False):
            placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder_image.fill(64)  # Dark gray background
            
            # Add text to placeholder
            cv2.putText(placeholder_image, "Video Feed", (250, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(placeholder_image, "Click 'Start' to begin", (200, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
            
            frame_placeholder.image(placeholder_image, channels="BGR", width=640)
        
        # Control buttons
        button_col1, button_col2, button_col3, button_col4 = st.columns(4)
        
        with button_col1:
            start_button = st.button("Start", disabled=st.session_state.get('is_running', False))
        
        with button_col2:
            stop_button = st.button("Stop", disabled=not st.session_state.get('is_running', False))
        
        with button_col3:
            reset_button = st.button("Reset Counters")
        
        with button_col4:
            clear_csv_button = st.button("Clear CSV")
    
    with col2:
        st.header("Statistics")
        
        # Counts display
        if app.counter:
            counts = app.counter.get_counts()
            if counts:
                counts_df = pd.DataFrame(list(counts.items()), columns=['Species', 'Count'])
                st.dataframe(counts_df, use_container_width=True)
            else:
                st.info("No detections yet")
        else:
            st.warning("Counter not initialized")
        
        # Info display
        info_placeholder = st.empty()
    
    # Handle video source
    if video_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov']
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            app.video_path = "temp_video.mp4"
            app.use_webcam = False
        else:
            app.video_path = None
    else:
        app.use_webcam = True
    
    # Handle button clicks
    if start_button and not st.session_state.get('is_running', False):
        # Start processing
        if app.use_webcam:
            app.cap = cv2.VideoCapture(0)
            app.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            app.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            if app.video_path:
                app.cap = cv2.VideoCapture(app.video_path)
            else:
                st.error("Please select a video file first.")
                st.stop()
        
        if not app.cap.isOpened():
            st.error("Could not open video source.")
            st.stop()
        
        st.session_state.is_running = True
        app.is_running = True
        app.frame_count = 0
        app.start_time = time.time()
        st.rerun()
    
    if stop_button and st.session_state.get('is_running', False):
        # Stop processing
        st.session_state.is_running = False
        app.is_running = False
        if app.cap:
            app.cap.release()
        st.rerun()
    
    if reset_button:
        if app.counter:
            app.counter.reset()
            st.rerun()
    
    if clear_csv_button:
        if app.counter:
            app.counter.clear_csv()
    
    # Process video frames if running
    if st.session_state.get('is_running', False) and app.cap:
        # Initialize frame tracking
        if 'frame_counter' not in st.session_state:
            st.session_state.frame_counter = 0
        
        # Read and process frame
        ret, frame = app.cap.read()
        
        if ret:
            st.session_state.frame_counter += 1
            
            # Update frame count and FPS
            app.frame_count += 1
            elapsed_time = time.time() - app.start_time
            if elapsed_time > 0:
                app.fps = app.frame_count / elapsed_time
            
            # Process frame
            processed_frame, num_detections = app.process_frame(frame, confidence_threshold)
            
            # Convert to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame with stable settings
            frame_placeholder.image(
                processed_frame_rgb,
                channels="RGB",
                width=640
            )
            
            # Update info less frequently to reduce flicker
            if st.session_state.frame_counter % 10 == 0:  # Update every 10 frames
                info_placeholder.text(f"FPS: {app.fps:.1f} | Detections: {num_detections}")
            
            # Control frame rate
            time.sleep(0.05)  # 20 FPS to reduce processing load
            st.rerun()
        else:
            # End of video or error
            if not app.use_webcam:
                # Loop video
                app.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                st.rerun()
            else:
                st.session_state.is_running = False
                app.is_running = False
                if app.cap:
                    app.cap.release()
    
    # Metrics report button
    if st.sidebar.button("Generate Report"):
        try:
            subprocess.Popen(["python", "metrics.py"])
            st.success("Report generation started!")
        except Exception as e:
            st.error(f"Could not start report generation: {e}")

if __name__ == "__main__":
    main()