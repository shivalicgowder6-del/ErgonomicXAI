import cv2
import numpy as np
from pathlib import Path
from src.pose_extraction import PoseExtractor
from src.risk_calculation import RiskCalculator
from src.temporal_model import ErgonomicTemporalModel
from src.explainability import ExplainabilityModule
from src.visualize import VisualizationModule
import os

def main():
    """
    Main function to run the full ErgonomicXAI pipeline.
    """
    print("--- Starting ErgonomicXAI Prototype Pipeline ---")
    
    # --- 1. Setup Paths ---
    root = Path(__file__).parent
    data_path = root / "data"
    image_dir = data_path / "images"
    output_dir = root / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get a list of image paths from the data directory
    image_paths = [p for p in image_dir.glob("*.jpg")]
    if not image_paths:
        print(f"Error: No .jpg images found in {image_dir}. Please add some images to process.")
        return
        
    print(f"Found {len(image_paths)} images to process.\n")

    # --- 2. Initialize Modules ---
    print("Initializing modules...")
    pose_extractor = PoseExtractor()
    risk_calculator = RiskCalculator()
    temporal_model = ErgonomicTemporalModel()
    
    # Train the temporal model with dummy data for this prototype
    temporal_model.train_with_dummy_data(epochs=5)
    
    # The explainer needs the trained model and score breakdown (for context)
    # This is a conceptual link; we'll pass dummy breakdown for now.
    dummy_score_breakdown = {'trunk': 0, 'arms': 0, 'legs': 0}
    explainer = ExplainabilityModule(temporal_model, dummy_score_breakdown)
    print("Modules initialized successfully.\n")

    # --- 3. Process Each Image ---
    for image_path in image_paths:
        print(f"--- Processing: {image_path.name} ---")
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path.name}. Skipping.")
                continue

            # a. Pose Extraction
            pose_results, keypoints_df = pose_extractor.extract_pose(image)
            if not pose_results.pose_landmarks:
                print(f"Warning: No pose detected in {image_path.name}. Skipping.")
                continue
            
            # b. Risk Calculation
            reba_score, score_breakdown = risk_calculator.calculate_reba_score(pose_results.pose_world_landmarks)

            # c. Temporal Prediction
            # Flatten keypoints for the temporal model and explainer
            keypoints_flat = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]).flatten()
            temporal_prediction = temporal_model.predict(keypoints_flat)
            
            # d. Explainability
            shap_contributions, explanation_text = explainer.generate_explanation(keypoints_flat)

            # e. Visualization
            viz = VisualizationModule(
                image,
                pose_results,
                reba_score,
                score_breakdown,
                temporal_prediction,
                shap_contributions,
                explanation_text
            )
            output_path = output_dir / f"{image_path.stem}_analysis.png"
            viz.generate_visualization(str(output_path))

        except Exception as e:
            print(f"An error occurred while processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * (len(image_path.name) + 20) + "\n")

if __name__ == "__main__":
    main()

    

