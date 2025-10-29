"""
Debug REBA calculation to understand why scores are consistently high
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_reba_calculation():
    """Debug the REBA calculation step by step"""
    print("🔍 DEBUGGING REBA CALCULATION")
    print("="*50)
    
    try:
        from src.pose_extraction import PoseExtractor
        from src.risk_calculation import RiskCalculator
        
        extractor = PoseExtractor()
        calculator = RiskCalculator()
        
        # Test with one image
        image_dir = project_root / "data" / "images" / "manufacturing"
        test_image = list(image_dir.glob("*.jpg"))[0]
        
        print(f"Testing with: {test_image.name}")
        image = cv2.imread(str(test_image))
        
        # Extract pose
        pose_results, _ = extractor.extract_pose(image)
        
        if not pose_results.pose_landmarks:
            print("❌ No pose detected")
            return
        
        landmarks = pose_results.pose_landmarks.landmark
        print(f"✅ Detected {len(landmarks)} landmarks")
        
        # Debug the REBA calculation
        print("\n🔍 ANALYZING POSE CHARACTERISTICS:")
        
        # Get key landmarks
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Extract specific landmarks
        p = {lm.name: [landmarks[lm.value].x, landmarks[lm.value].y, landmarks[lm.value].z] 
             for lm in mp_pose.PoseLandmark}
        
        # Analyze trunk angle
        mid_shoulder = np.mean([p['LEFT_SHOULDER'], p['RIGHT_SHOULDER']], axis=0)
        mid_hip = np.mean([p['LEFT_HIP'], p['RIGHT_HIP']], axis=0)
        
        print(f"Shoulder position: {mid_shoulder}")
        print(f"Hip position: {mid_hip}")
        
        # Calculate trunk angle manually
        trunk_vector = mid_shoulder - mid_hip
        vertical_vector = np.array([0, -1, 0])
        
        # Calculate angle
        dot_product = np.dot(trunk_vector, vertical_vector)
        magnitude = np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector)
        trunk_angle = np.degrees(np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0)))
        
        print(f"Trunk angle: {trunk_angle:.1f}°")
        
        # Analyze arm positions
        left_arm_angle = np.degrees(np.arccos(np.clip(
            np.dot(p['LEFT_SHOULDER'] - p['LEFT_ELBOW'], p['LEFT_ELBOW'] - p['LEFT_WRIST']) /
            (np.linalg.norm(p['LEFT_SHOULDER'] - p['LEFT_ELBOW']) * np.linalg.norm(p['LEFT_ELBOW'] - p['LEFT_WRIST'])), -1.0, 1.0)))
        
        print(f"Left arm angle: {left_arm_angle:.1f}°")
        
        # Check if landmarks are reasonable
        print(f"\n🔍 LANDMARK QUALITY CHECK:")
        for i, landmark in enumerate(landmarks[:5]):  # Check first 5 landmarks
            print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}, vis={landmark.visibility:.3f}")
        
        # Now run the actual REBA calculation
        print(f"\n🔍 RUNNING REBA CALCULATION:")
        reba_score, score_breakdown = calculator.calculate_reba_score(pose_results.pose_world_landmarks)
        
        print(f"REBA Score: {reba_score}")
        print(f"Breakdown: {score_breakdown}")
        
        # Analyze why score is high
        print(f"\n🔍 ANALYSIS:")
        if reba_score > 8:
            print("❌ HIGH RISK SCORE DETECTED")
            print("Possible causes:")
            print("1. Trunk angle too extreme")
            print("2. Arm positions awkward")
            print("3. Leg positions problematic")
            print("4. REBA calculation logic issues")
        
        # Check if the issue is in the calculation logic
        print(f"\n🔍 CHECKING CALCULATION LOGIC:")
        
        # The issue might be in the REBA calculation itself
        # Let's check if the landmarks are being interpreted correctly
        print("Landmark coordinate ranges:")
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        z_coords = [lm.z for lm in landmarks]
        
        print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        print(f"Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
        
        # Check if coordinates are normalized (should be 0-1)
        if max(x_coords) > 1.0 or max(y_coords) > 1.0:
            print("⚠️ WARNING: Coordinates not properly normalized!")
        
        extractor.close()
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_reba_calculation()
