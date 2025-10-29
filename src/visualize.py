import cv2
import numpy as np
import matplotlib
# Force Matplotlib to use a non-GUI backend to avoid Tkinter/icon errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapipe as mp

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class VisualizationModule:
    """
    Handles the creation of detailed output visualizations by combining pose
    skeletons, risk scores, and SHAP-based explanation charts.
    """
    def __init__(self, image, pose_results, reba_score, score_breakdown, temporal_prediction, shap_contributions, explanation_text):
        """
        Initializes the visualization with all necessary data components.

        Args:
            image (np.ndarray): The original input image.
            pose_results (mediapipe.python.solution_base.SolutionOutputs): The landmark results from MediaPipe.
            reba_score (float): The calculated REBA score for the current posture.
            score_breakdown (dict): A dictionary of risk scores per body part.
            temporal_prediction (np.ndarray): The predicted future risk from the LSTM model.
            shap_contributions (dict): A dictionary of SHAP values for each body part.
            explanation_text (str): The final, human-readable actionable advice.
        """
        self.image = image
        self.pose_results = pose_results
        self.reba_score = reba_score
        self.score_breakdown = score_breakdown
        self.temporal_prediction = temporal_prediction
        self.shap_contributions = shap_contributions
        self.explanation_text = explanation_text

    def _draw_pose_skeleton(self):
        """Draws the pose landmarks and connections on a copy of the image."""
        annotated_image = self.image.copy()
        if self.pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                self.pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        return annotated_image

    def _create_shap_barchart(self):
        """Creates a horizontal bar chart from SHAP contributions and returns it as an image."""
        fig, ax = plt.subplots(figsize=(5, 3))
        parts = list(self.shap_contributions.keys())
        values = list(self.shap_contributions.values())
        
        # Color bars based on whether they increase (positive) or decrease (negative) risk
        colors = ['#ff7f0e' if v > 0 else '#1f77b4' for v in values]
        
        ax.barh(parts, values, color=colors)
        ax.set_title('Risk Contribution (SHAP Values)', fontsize=10)
        ax.set_xlabel('Contribution to Risk Score', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.axvline(0, color='grey', linewidth=0.8) # Add a zero line for reference
        fig.tight_layout()
        
        # Convert the Matplotlib plot to a NumPy array (image)
        fig.canvas.draw()
        try:
            # Preferred: direct RGB string
            plot_buf = fig.canvas.tostring_rgb()
            plot_img = np.frombuffer(plot_buf, dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except Exception:
            try:
                # Fallback: buffer_rgba() (returns an object supporting buffer protocol)
                buf = fig.canvas.buffer_rgba()
                plot_img = np.frombuffer(buf, dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Drop alpha channel
                plot_img = plot_img[:, :, :3]
            except Exception:
                try:
                    # Final fallback: print_to_buffer() returns (buf, (w,h))
                    buf, (w, h) = fig.canvas.print_to_buffer()
                    plot_img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
                except Exception as e:
                    plt.close(fig)
                    raise RuntimeError(f"Could not convert Matplotlib figure to image: {e}")

        plt.close(fig)
        return plot_img

    def generate_visualization(self, output_path):
        """
        Combines all visual elements (skeleton, text, charts) into a single
        output image and saves it to the specified path.
        """
        # 1. Create the base image with the pose skeleton drawn on it
        pose_image = self._draw_pose_skeleton()
        h, w, _ = pose_image.shape

        # 2. Generate the SHAP bar chart as an image
        shap_chart = self._create_shap_barchart()
        sh, sw, _ = shap_chart.shape

        # 3. Create a white sidebar to hold all the text and chart information
        sidebar_width = 400
        combined_width = w + sidebar_width
        combined_height = max(h, 650) # Ensure a minimum height for all content
        
        # Pad the original pose image if it's shorter than the combined height
        if h < combined_height:
            padding = np.ones((combined_height - h, w, 3), dtype=np.uint8) * 255
            pose_image = np.vstack((pose_image, padding))
        
        # Create the final output canvas
        output_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        output_image[0:combined_height, 0:w] = cv2.resize(pose_image, (w, combined_height))

        # 4. Add all text information to the sidebar
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(output_image, "Ergonomic Risk Analysis", (w + 20, 40), font, 0.9, (0, 0, 0), 2)

        # Display REBA Score and determine risk level/color
        reba_text = f"REBA Score: {self.reba_score:.1f}"
        risk_level, risk_color = ("Low", (0, 150, 0))
        if self.reba_score > 7:
            risk_level, risk_color = ("High", (0, 0, 255))
        elif self.reba_score > 4:
            risk_level, risk_color = ("Medium", (0, 165, 255))
        
        cv2.putText(output_image, reba_text, (w + 20, 90), font, 0.7, (0, 0, 0), 2)
        cv2.putText(output_image, f"Risk Level: {risk_level}", (w + 20, 120), font, 0.7, risk_color, 2)
        
        temporal_text = f"Predicted Temporal Risk: {self.temporal_prediction[0][0]:.1f}"
        cv2.putText(output_image, temporal_text, (w + 20, 170), font, 0.7, (0, 0, 0), 2)

        # 5. Place the SHAP chart onto the sidebar
        chart_y_start = 220
        scale_ratio = (sidebar_width - 40) / sw
        new_sh, new_sw = int(sh * scale_ratio), int(sw * scale_ratio)
        shap_chart_resized = cv2.resize(shap_chart, (new_sw, new_sh))
        output_image[chart_y_start:chart_y_start + new_sh, w + 20:w + 20 + new_sw] = shap_chart_resized

        # 6. Add the final actionable explanation text
        explanation_y_start = chart_y_start + new_sh + 40
        cv2.putText(output_image, "Actionable Insights (XAI)", (w + 20, explanation_y_start - 10), font, 0.7, (0, 0, 0), 2)

        # Add logic to wrap long lines of text for clean display
        y0, dy = explanation_y_start + 20, 25
        for i, line in enumerate(self.explanation_text.split('\n')):
            y = y0 + i * dy
            # Handle markdown-style bolding for emphasis
            if "**" in line:
                parts = line.split("**")
                cv2.putText(output_image, parts[0], (w + 30, y), font, 0.5, (0,0,0), 1)
                x_offset = cv2.getTextSize(parts[0], font, 0.5, 2)[0][0]
                cv2.putText(output_image, parts[1], (w + 30 + x_offset, y), font, 0.5, (0,0,200), 2)
                x_offset += cv2.getTextSize(parts[1], font, 0.5, 2)[0][0]
                cv2.putText(output_image, parts[2], (w + 30 + x_offset, y), font, 0.5, (0,0,0), 1)
            else:
                 cv2.putText(output_image, line, (w + 30, y), font, 0.5, (0,0,0), 1)

        # 7. Save the final composed image to disk
        try:
            cv2.imwrite(output_path, output_image)
            print(f"Successfully saved visualization to: {output_path}")
        except Exception as e:
            print(f"Error: Could not save the visualization image. {e}")

# Example usage for direct testing of this script
if __name__ == '__main__':
    # Create a dummy blank image to test the layout
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 220
    
    # Create mock data objects to simulate input
    class DummyPose:
        pose_landmarks = None
    
    dummy_results = DummyPose()
    dummy_reba = 8.5
    dummy_breakdown = {'trunk': 4, 'arms': 3, 'legs': 1.5}
    dummy_temporal = np.array([[9.2]])
    dummy_shap = {'trunk': 0.8, 'arms': 0.5, 'legs': -0.1}
    dummy_explanation = "High risk primarily due to posture of the **TRUNK**.\nActionable Advice: Focus on straightening your back."

    # Initialize and run the visualization module
    viz = VisualizationModule(
        test_image,
        dummy_results,
        dummy_reba,
        dummy_breakdown,
        dummy_temporal,
        dummy_shap,
        dummy_explanation
    )
    
    viz.generate_visualization("test_visualization.png")
    print("Test visualization 'test_visualization.png' created.")

