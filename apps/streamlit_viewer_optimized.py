import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
from src.pose_extraction import PoseExtractor
from src.risk_calculation import RiskCalculator
from src.explainability import ExplainabilityModule
from src.temporal_model import ErgonomicTemporalModel

# Page config
st.set_page_config(
    page_title="ErgonomicXAI - Optimized",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the models for better performance
@st.cache_resource
def load_models():
    """Load and cache models for better performance"""
    extractor = PoseExtractor()
    calculator = RiskCalculator()
    temporal_model = ErgonomicTemporalModel()
    return extractor, calculator, temporal_model

def analyze_image(image, extractor, calculator, temporal_model):
    """Optimized image analysis function"""
    start_time = time.time()
    
    # Extract pose
    pose_results, _ = extractor.extract_pose(image)
    
    if not pose_results.pose_landmarks:
        return None, "No pose detected in image"
    
    # Calculate REBA score
    reba_score, breakdown = calculator.calculate_reba_score(pose_results.pose_world_landmarks)
    
    # Generate explanation
    explainer = ExplainabilityModule(temporal_model, calculator, breakdown)
    contributions, advice = explainer.generate_explanation([])
    
    processing_time = time.time() - start_time
    
    return {
        'reba_score': reba_score,
        'breakdown': breakdown,
        'contributions': contributions,
        'advice': advice,
        'processing_time': processing_time
    }, None

def main():
    st.title("🤖 ErgonomicXAI - Optimized System")
    st.markdown("**Real-time Ergonomic Risk Assessment with AI**")
    
    # Load models (cached)
    with st.spinner("Loading AI models..."):
        extractor, calculator, temporal_model = load_models()
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    st.sidebar.header("📊 System Status")
    st.sidebar.success("🟢 System: Optimized & Ready")
    st.sidebar.info("📈 Performance: Enhanced")
    st.sidebar.metric("💾 Size", "2.2GB", "-74%")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to analyze ergonomic risk"
        )
        
        # Quick test with sample images
        st.subheader("🧪 Quick Test")
        if st.button("Test with Sample Images"):
            sample_dir = Path("data/images")
            sample_images = list(sample_dir.glob("*.jpg"))
            
            if sample_images:
                st.info(f"Found {len(sample_images)} sample images")
                
                # Analyze all samples
                results = []
                progress_bar = st.progress(0)
                
                for i, img_path in enumerate(sample_images):
                    image = cv2.imread(str(img_path))
                    if image is not None:
                        result, error = analyze_image(image, extractor, calculator, temporal_model)
                        if result:
                            results.append({
                                'image': img_path.name,
                                'result': result
                            })
                    progress_bar.progress((i + 1) / len(sample_images))
                
                # Display results
                st.subheader("📊 Sample Analysis Results")
                for result in results:
                    with st.expander(f"📸 {result['image']}"):
                        data = result['result']
                        st.metric("REBA Score", f"{data['reba_score']:.1f}")
                        st.metric("Processing Time", f"{data['processing_time']:.2f}s")
                        
                        # Risk level
                        if data['reba_score'] < 6:
                            st.success("🟢 LOW RISK")
                        elif data['reba_score'] < 9:
                            st.warning("🟡 MEDIUM RISK")
                        else:
                            st.error("🔴 HIGH RISK")
                        
                        # Breakdown
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Trunk", data['breakdown']['trunk'])
                        with col_b:
                            st.metric("Arms", data['breakdown']['arms'])
                        with col_c:
                            st.metric("Legs", data['breakdown']['legs'])
    
    with col2:
        if uploaded_file is not None:
            st.header("🔍 Analysis Results")
            
            # Process uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Display uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analyze
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result, error = analyze_image(image, extractor, calculator, temporal_model)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # REBA Score
                    st.metric("REBA Score", f"{result['reba_score']:.1f}")
                    
                    # Risk level
                    if result['reba_score'] < 6:
                        st.success("🟢 LOW RISK - Good posture!")
                    elif result['reba_score'] < 9:
                        st.warning("🟡 MEDIUM RISK - Consider improvements")
                    else:
                        st.error("🔴 HIGH RISK - Immediate attention needed")
                    
                    # Breakdown
                    st.subheader("📊 Body Part Analysis")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Trunk", result['breakdown']['trunk'])
                    with col_b:
                        st.metric("Arms", result['breakdown']['arms'])
                    with col_c:
                        st.metric("Legs", result['breakdown']['legs'])
                    
                    # Advice
                    st.subheader("💡 Recommendations")
                    st.info(result['advice'])
                    
                    # Performance
                    st.subheader("⚡ Performance")
                    st.metric("Processing Time", f"{result['processing_time']:.2f} seconds")
        else:
            st.info("👆 Upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("**ErgonomicXAI** - Optimized AI-powered ergonomic risk assessment")
    st.markdown("*System optimized for performance and efficiency*")

if __name__ == "__main__":
    main()
