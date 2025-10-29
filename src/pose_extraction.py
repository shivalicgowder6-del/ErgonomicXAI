import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

class PoseExtractor:
    """
    A class to handle human pose extraction from images using MediaPipe.
    """
    def __init__(self, static_image_mode=True, model_complexity=2, min_detection_confidence=0.5):
        """
        Initializes the MediaPipe Pose model.

        Args:
            static_image_mode (bool): Whether to treat the input images as a batch of static
                                      images or a video stream.
            model_complexity (int): Complexity of the pose landmark model: 0, 1, or 2.
            min_detection_confidence (float): Minimum confidence value for the person detection.
        """
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )

    def extract_pose(self, image):
        """
        Processes an image to detect and extract human pose keypoints.

        Args:
            image (np.ndarray): The input image in BGR format (from OpenCV).

        Returns:
            tuple: A tuple containing:
                - results (mediapipe.python.solution_base.SolutionOutputs): The full results object from MediaPipe.
                - keypoints_df (pd.DataFrame): A DataFrame containing the x, y, z, and visibility
                                               of each landmark. Returns an empty DataFrame if no pose is detected.
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect the pose
        results = self.pose.process(image_rgb)

        keypoints_list = []
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints_list.append({
                    'landmark': i,
                    'name': mp_pose.PoseLandmark(i).name,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        
        keypoints_df = pd.DataFrame(keypoints_list)
        
        # --- FIX ---
        # The original code returned three values. We only need two for main.py.
        return results, keypoints_df

    def close(self):
        """Releases the MediaPipe Pose model resources."""
        self.pose.close()

# Example usage (for testing this script directly)
if __name__ == '__main__':
    # Create a dummy blank image for testing
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Initialize and run the extractor
    extractor = PoseExtractor()
    pose_results, df = extractor.extract_pose(test_image)
    
    if not df.empty:
        print("PoseExtractor Test: Successfully extracted keypoints.")
        print("First 5 rows of keypoints DataFrame:")
        print(df.head())
    else:
        print("PoseExtractor Test: No pose detected in the dummy image (as expected).")
        
    extractor.close()

