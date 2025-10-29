"""
Debug the fixed REBA calculation to see why scores are 0.0
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_fixed_reba():
    """Debug the fixed REBA calculation"""
    print("🔍 DEBUGGING FIXED REBA CALCULATION")
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
        
        # Debug step by step
        print("\n🔍 STEP-BY-STEP DEBUG:")
        
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Extract specific landmarks
        p = {lm.name: [landmarks[lm.value].x, landmarks[lm.value].y, landmarks[lm.value].z] 
             for lm in mp_pose.PoseLandmark}
        
        print(f"Left shoulder: {p['LEFT_SHOULDER']}")
        print(f"Right shoulder: {p['RIGHT_SHOULDER']}")
        print(f"Left hip: {p['LEFT_HIP']}")
        print(f"Right hip: {p['RIGHT_HIP']}")
        
        # Test trunk calculation
        mid_shoulder = np.mean([p['LEFT_SHOULDER'], p['RIGHT_SHOULDER']], axis=0)
        mid_hip = np.mean([p['LEFT_HIP'], p['RIGHT_HIP']], axis=0)
        print(f"Mid shoulder: {mid_shoulder}")
        print(f"Mid hip: {mid_hip}")
        
        trunk_vector = mid_shoulder - mid_hip
        vertical_vector = np.array([0, -1, 0])
        print(f"Trunk vector: {trunk_vector}")
        print(f"Vertical vector: {vertical_vector}")
        
        trunk_angle = calculator._calculate_angle(trunk_vector, vertical_vector)
        print(f"Trunk angle: {trunk_angle:.1f}°")
        
        # Test trunk scoring
        trunk_score = 1
        if 15 < trunk_angle <= 30: trunk_score = 2
        elif 30 < trunk_angle <= 45: trunk_score = 3
        elif trunk_angle > 45: trunk_score = 4
        print(f"Trunk score: {trunk_score}")
        
        # Test neck calculation
        neck_vector = p['NOSE'] - mid_shoulder
        neck_angle = calculator._calculate_angle(neck_vector, vertical_vector)
        print(f"Neck angle: {neck_angle:.1f}°")
        
        neck_score = 1
        if neck_angle > 20: neck_score = 2
        print(f"Neck score: {neck_score}")
        
        # Test leg calculation
        left_knee_angle = calculator._calculate_angle(p['LEFT_HIP'], p['LEFT_KNEE'], p['LEFT_ANKLE'])
        right_knee_angle = calculator._calculate_angle(p['RIGHT_HIP'], p['RIGHT_KNEE'], p['RIGHT_ANKLE'])
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        print(f"Left knee angle: {left_knee_angle:.1f}°")
        print(f"Right knee angle: {right_knee_angle:.1f}°")
        print(f"Average knee angle: {knee_angle:.1f}°")
        
        leg_score = 1
        if 120 < knee_angle < 160: leg_score = 2
        elif 100 < knee_angle <= 120: leg_score = 3
        elif knee_angle <= 100: leg_score = 4
        print(f"Leg score: {leg_score}")
        
        # Test arm calculation
        left_arm_vector = p['LEFT_ELBOW'] - p['LEFT_SHOULDER']
        right_arm_vector = p['RIGHT_ELBOW'] - p['RIGHT_SHOULDER']
        print(f"Left arm vector: {left_arm_vector}")
        print(f"Right arm vector: {right_arm_vector}")
        
        left_arm_angle = calculator._calculate_angle(left_arm_vector, vertical_vector)
        right_arm_angle = calculator._calculate_angle(right_arm_vector, vertical_vector)
        arm_angle = max(left_arm_angle, right_arm_angle)
        print(f"Left arm angle: {left_arm_angle:.1f}°")
        print(f"Right arm angle: {right_arm_angle:.1f}°")
        print(f"Max arm angle: {arm_angle:.1f}°")
        
        arm_score = 1
        if 30 < arm_angle <= 60: arm_score = 2
        elif 60 < arm_angle <= 90: arm_score = 3
        elif arm_angle > 90: arm_score = 4
        print(f"Arm score: {arm_score}")
        
        # Test elbow calculation
        left_elbow_angle = calculator._calculate_angle(p['LEFT_SHOULDER'], p['LEFT_ELBOW'], p['LEFT_WRIST'])
        right_elbow_angle = calculator._calculate_angle(p['RIGHT_SHOULDER'], p['RIGHT_ELBOW'], p['RIGHT_WRIST'])
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        print(f"Left elbow angle: {left_elbow_angle:.1f}°")
        print(f"Right elbow angle: {right_elbow_angle:.1f}°")
        print(f"Average elbow angle: {elbow_angle:.1f}°")
        
        lower_arm_score = 1
        if not (80 < elbow_angle < 120): lower_arm_score = 2
        print(f"Lower arm score: {lower_arm_score}")
        
        # Now run the actual calculation
        print(f"\n🔍 RUNNING ACTUAL REBA CALCULATION:")
        reba_score, score_breakdown = calculator.calculate_reba_score(pose_results.pose_world_landmarks)
        
        print(f"REBA Score: {reba_score}")
        print(f"Breakdown: {score_breakdown}")
        
        extractor.close()
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fixed_reba()
