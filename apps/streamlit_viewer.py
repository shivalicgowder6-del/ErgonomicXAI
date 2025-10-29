"""
Streamlit app for ergonomic risk assessment with logging and summary features.

This application scans for media, performs analysis, and now logs each result
to a CSV file. It also displays a summary dashboard of all past analyses.

Run with:
  streamlit run apps/streamlit_viewer.py
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt # --- FIX: Import matplotlib here ---

# --- Path Setup ---
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import your project modules
from src.pose_extraction import PoseExtractor
from src.risk_calculation import RiskCalculator
from src.temporal_model import ErgonomicTemporalModel
from src.explainability import ExplainabilityModule
from src.visualize import VisualizationModule

# --- Page Configuration ---
st.set_page_config(
    page_title="ErgonomicXAI Analyzer",
    page_icon="🦾",
    layout="wide"
)

# --- Helper Functions for Logging and Risk Categorization ---
def get_risk_level(score):
    """Categorizes a REBA score into Low, Medium, or High risk levels."""
    if score > 6:
        return "High Risk"
    elif score >= 3:
        return "Medium Risk"
    else:
        return "Low Risk"

def log_result(log_path, result_data):
    """Appends a new result to the results.csv file."""
    header = ["File Name", "Avg REBA Score", "Peak REBA Score", "Risk Level", "Advice"]
    
    # Create a DataFrame from the new result
    new_log_df = pd.DataFrame([result_data])
    
    try:
        if not log_path.exists():
            # If file doesn't exist, create it with a header
            new_log_df.to_csv(log_path, index=False, header=header)
        else:
            # If file exists, append without the header
            new_log_df.to_csv(log_path, mode='a', index=False, header=False)
    except Exception as e:
        st.error(f"Failed to log results to {log_path}. Error: {e}")

# --- Caching for expensive models ---
@st.cache_resource
def load_models():
    """Load and cache AI models."""
    pose_extractor = PoseExtractor()
    temporal_model = ErgonomicTemporalModel()
    temporal_model.train_with_dummy_data(epochs=5)
    risk_calculator = RiskCalculator()
    return pose_extractor, temporal_model, risk_calculator

# --- Main App ---
st.title("🦾 ErgonomicXAI: Posture & Task Analyzer")
st.markdown("Select an image or video from the sidebar to analyze its posture for ergonomic risks.")

# Load models
with st.spinner("Loading AI models, please wait..."):
    pose_extractor, temporal_model, risk_calculator = load_models()

# --- Sidebar for Media Selection ---
with st.sidebar:
    st.header("Select Media File")
    
    media_dir = project_root / "data" / "images" / "manufacturing"
    media_files = []
    if media_dir.exists():
        supported_types = ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.mov", "*.avi")
        for file_type in supported_types:
            media_files.extend([f.name for f in media_dir.glob(file_type)])

    if media_files:
        selected_media_name = st.selectbox("Choose a file to analyze:", options=["-"] + sorted(media_files))
        selected_media_path = media_dir / selected_media_name if selected_media_name != "-" else None
    else:
        st.error("No compatible media found in `data/images/manufacturing` folder.")
        selected_media_path = None

output_placeholder = st.container()

# --- Analysis and Display Logic ---
if selected_media_path and selected_media_path.exists():
    with output_placeholder:
        file_extension = selected_media_path.suffix.lower()
        st.header(f"Analysis for: `{selected_media_path.name}`")

        # Define path for results log
        results_csv_path = project_root / "output" / "results.csv"
        os.makedirs(project_root / "output", exist_ok=True)

        # --- IMAGE ANALYSIS ---
        if file_extension in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(str(selected_media_path))
            with st.spinner("Analyzing image..."):
                pose_results, _ = pose_extractor.extract_pose(image)
                if not pose_results.pose_landmarks:
                    st.error("Could not detect a person in the selected image.")
                else:
                    reba_score, score_breakdown = risk_calculator.calculate_reba_score(pose_results.pose_world_landmarks)
                    keypoints_flat = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]).flatten()
                    temporal_prediction = temporal_model.predict(keypoints_flat)
                    explainer = ExplainabilityModule(temporal_model, risk_calculator, score_breakdown)
                    shap_contributions, explanation_text = explainer.generate_explanation(keypoints_flat)
                    risk_level = get_risk_level(reba_score)
                    
                    # Display results
                    color = "red" if risk_level == "High Risk" else ("orange" if risk_level == "Medium Risk" else "green")
                    col1, col2 = st.columns([1, 1.2])
                    with col1:
                        st.subheader("Pose Visualization")
                        viz_module = VisualizationModule(image, pose_results, reba_score, score_breakdown, temporal_prediction, shap_contributions, explanation_text)
                        annotated_image = viz_module._draw_pose_skeleton()
                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Pose Skeleton", use_container_width=True)
                    with col2:
                        st.subheader("Ergonomic Report")
                        st.metric(label="REBA Score", value=f"{reba_score:.1f}", delta=risk_level)
                        st.markdown(f"The score indicates a **:{color}[{risk_level}]**.")
                        st.subheader("Parts Importance (XAI)")
                        st.bar_chart(pd.DataFrame.from_dict(shap_contributions, orient='index', columns=['importance']))
                        st.subheader("Actionable Advice (XAI)")
                        st.info(explanation_text.replace("**", ""))

                    # Log the results
                    log_data = {
                        "File Name": selected_media_name,
                        "Avg REBA Score": f"{reba_score:.2f}",
                        "Peak REBA Score": f"{reba_score:.2f}", # For images, avg and peak are the same
                        "Risk Level": risk_level,
                        "Advice": explanation_text.replace("\n", " ").replace("**", "")
                    }
                    log_result(results_csv_path, log_data)
                    st.success(f"Results for `{selected_media_name}` logged to `output/results.csv`.")

        # --- VIDEO ANALYSIS ---
        elif file_extension in ['.mp4', '.mov', '.avi']:
            col1, col2 = st.columns([2, 1])
            with col1: st.subheader("Video Processing"); stframe = st.empty()
            with col2: st.subheader("Live Analysis"); stchart = st.empty(); summary_placeholder = st.empty()

            cap = cv2.VideoCapture(str(selected_media_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            risk_scores_over_time = []
            progress_bar = st.progress(0, "Analyzing video frames...")

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % 10 == 0: # Process every 10th frame for performance
                    pose_results, _ = pose_extractor.extract_pose(frame)
                    score = 0
                    if pose_results.pose_landmarks:
                        score, _ = risk_calculator.calculate_reba_score(pose_results.pose_world_landmarks)
                        viz_module = VisualizationModule(frame, pose_results, 0, {}, np.array([[0]]), {}, "")
                        annotated_frame = viz_module._draw_pose_skeleton()
                        MAX_WIDTH = 800; h, w, _ = annotated_frame.shape
                        ratio = MAX_WIDTH / w if w > MAX_WIDTH else 1
                        resized_frame = cv2.resize(annotated_frame, (int(w*ratio), int(h*ratio)))
                        stframe.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}/{total_frames}")
                    
                    risk_scores_over_time.append({'frame': frame_idx, 'REBA Score': score})
                    with stchart.container(): st.line_chart(pd.DataFrame(risk_scores_over_time).set_index('frame'))
                
                progress_bar.progress((frame_idx + 1) / total_frames, f"Analyzing frame {frame_idx+1}/{total_frames}")
            
            cap.release()
            progress_bar.progress(1.0, "Analysis complete.")

            if risk_scores_over_time:
                with summary_placeholder.container():
                    st.subheader("Task Summary")
                    summary_df = pd.DataFrame(risk_scores_over_time)
                    avg_score = summary_df[summary_df['REBA Score'] > 0]['REBA Score'].mean() # Exclude frames with no detection
                    max_score = summary_df['REBA Score'].max()
                    risk_level = get_risk_level(avg_score)
                    color = "red" if risk_level == "High Risk" else ("orange" if risk_level == "Medium Risk" else "green")
                    st.metric(label="Average REBA Score", value=f"{avg_score:.1f}", delta=risk_level)
                    st.metric(label="Peak REBA Score", value=f"{max_score:.1f}")

                    # Log video results
                    log_data = {
                        "File Name": selected_media_name,
                        "Avg REBA Score": f"{avg_score:.2f}",
                        "Peak REBA Score": f"{max_score:.2f}",
                        "Risk Level": risk_level,
                        "Advice": "Review temporal risk curve for high-risk moments."
                    }
                    log_result(results_csv_path, log_data)
                    st.success(f"Results for `{selected_media_name}` logged to `output/results.csv`.")
else:
    with output_placeholder:
        st.info("Select a media file from the sidebar to begin analysis.")

# --- Summary Dashboard (Always Visible) ---
st.divider()
st.header("📊 Project Summary Dashboard")
results_csv_path = project_root / "output" / "results.csv"
if results_csv_path.exists():
    log_df = pd.read_csv(results_csv_path)
    risk_counts = log_df["Risk Level"].value_counts()
    
    # Ensure all risk categories are present for the pie chart
    for level in ["Low Risk", "Medium Risk", "High Risk"]:
        if level not in risk_counts:
            risk_counts[level] = 0
            
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Risk Level Distribution")
        fig, ax = plt.subplots()
        ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, colors=['#d4edda', '#ffeeba', '#f5c6cb'])
        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
    with col2:
        st.subheader("Logged Analyses")
        st.dataframe(log_df)
else:
    st.info("No results have been logged yet. Analyze an image or video to generate the first log entry.")

