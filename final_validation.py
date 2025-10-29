"""
Final comprehensive validation of the entire ErgonomicXAI system
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pose_extraction():
    """Test pose extraction functionality"""
    print("🔍 Testing Pose Extraction...")
    
    try:
        from src.pose_extraction import PoseExtractor
        
        extractor = PoseExtractor()
        image_dir = project_root / "data" / "images" / "manufacturing"
        test_images = list(image_dir.glob("*.jpg"))[:3]
        
        results = []
        for img_path in test_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            start_time = time.time()
            pose_results, keypoints_df = extractor.extract_pose(image)
            extraction_time = time.time() - start_time
            
            if pose_results.pose_landmarks:
                results.append({
                    'success': True,
                    'time': extraction_time,
                    'landmarks': len(pose_results.pose_landmarks.landmark)
                })
            else:
                results.append({'success': False, 'time': extraction_time, 'landmarks': 0})
        
        extractor.close()
        
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_time = np.mean([r['time'] for r in results])
        
        print(f"   ✅ Success Rate: {success_rate:.1%}")
        print(f"   ⏱️ Average Time: {avg_time:.2f}s")
        
        return success_rate > 0.8, results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, []

def test_risk_calculation():
    """Test risk calculation with score variation"""
    print("\n🔍 Testing Risk Calculation...")
    
    try:
        from src.pose_extraction import PoseExtractor
        from src.risk_calculation import RiskCalculator
        
        extractor = PoseExtractor()
        calculator = RiskCalculator()
        
        image_dir = project_root / "data" / "images" / "manufacturing"
        test_images = list(image_dir.glob("*.jpg"))[:5]
        
        results = []
        for img_path in test_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            pose_results, _ = extractor.extract_pose(image)
            if not pose_results.pose_landmarks:
                continue
            
            start_time = time.time()
            reba_score, score_breakdown = calculator.calculate_reba_score(pose_results.pose_world_landmarks)
            calc_time = time.time() - start_time
            
            results.append({
                'image': img_path.name,
                'reba_score': reba_score,
                'breakdown': score_breakdown,
                'time': calc_time
            })
        
        extractor.close()
        
        if not results:
            print("   ❌ No valid results")
            return False, []
        
        reba_scores = [r['reba_score'] for r in results]
        score_range = max(reba_scores) - min(reba_scores)
        avg_score = np.mean(reba_scores)
        
        print(f"   📊 Score Range: {min(reba_scores):.1f} - {max(reba_scores):.1f}")
        print(f"   📈 Average Score: {avg_score:.1f}")
        print(f"   📊 Score Variation: {score_range:.1f}")
        
        # Check for good variation
        has_variation = score_range > 1.0
        has_low_scores = min(reba_scores) < 6.0
        has_high_scores = max(reba_scores) > 8.0
        
        print(f"   ✅ Has Variation: {has_variation}")
        print(f"   ✅ Has Low Scores: {has_low_scores}")
        print(f"   ✅ Has High Scores: {has_high_scores}")
        
        return has_variation and (has_low_scores or has_high_scores), results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, []

def test_temporal_model():
    """Test temporal model functionality"""
    print("\n🔍 Testing Temporal Model...")
    
    try:
        from src.temporal_model import ErgonomicTemporalModel
        
        model = ErgonomicTemporalModel()
        
        # Test training
        start_time = time.time()
        model.train_with_dummy_data(epochs=2)
        train_time = time.time() - start_time
        
        # Test prediction
        test_keypoints = np.random.rand(99)
        start_time = time.time()
        prediction = model.predict(test_keypoints)
        pred_time = time.time() - start_time
        
        print(f"   ✅ Training Time: {train_time:.2f}s")
        print(f"   ✅ Prediction: {prediction[0][0]:.2f}")
        print(f"   ⏱️ Prediction Time: {pred_time:.3f}s")
        
        return True, {
            'train_time': train_time,
            'prediction': prediction[0][0],
            'pred_time': pred_time
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, {}

def test_explainability():
    """Test explainability module"""
    print("\n🔍 Testing Explainability...")
    
    try:
        from src.explainability import ExplainabilityModule
        from src.temporal_model import ErgonomicTemporalModel
        from src.risk_calculation import RiskCalculator
        
        temporal_model = ErgonomicTemporalModel()
        temporal_model.train_with_dummy_data(epochs=1)
        risk_calculator = RiskCalculator()
        
        explainer = ExplainabilityModule(temporal_model, risk_calculator, {'trunk': 2, 'arms': 1, 'legs': 1})
        
        test_keypoints = np.random.rand(99)
        start_time = time.time()
        contributions, explanation = explainer.generate_explanation(test_keypoints)
        explain_time = time.time() - start_time
        
        print(f"   ✅ Contributions: {contributions}")
        print(f"   📝 Explanation Length: {len(explanation)} chars")
        print(f"   ⏱️ Generation Time: {explain_time:.3f}s")
        
        return True, {
            'contributions': contributions,
            'explanation': explanation,
            'time': explain_time
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, {}

def test_streamlit_app():
    """Test if Streamlit app is accessible"""
    print("\n🔍 Testing Streamlit App...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:8504", timeout=5)
        if response.status_code == 200:
            print("   ✅ Streamlit app is accessible")
            print("   🌐 URL: http://localhost:8504")
            return True
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        return False

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🧪 COMPREHENSIVE ERGONOMICXAI VALIDATION")
    print("="*60)
    
    # Run all tests
    pose_success, pose_results = test_pose_extraction()
    risk_success, risk_results = test_risk_calculation()
    temporal_success, temporal_results = test_temporal_model()
    explain_success, explain_results = test_explainability()
    streamlit_success = test_streamlit_app()
    
    # Overall assessment
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE VALIDATION RESULTS")
    print("="*60)
    
    tests = [
        ("Pose Extraction", pose_success),
        ("Risk Calculation", risk_success),
        ("Temporal Model", temporal_success),
        ("Explainability", explain_success),
        ("Streamlit App", streamlit_success)
    ]
    
    passed_tests = sum(1 for _, success in tests if success)
    total_tests = len(tests)
    
    print(f"\n🎯 TEST RESULTS:")
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 OVERALL SCORE: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
    
    # Detailed analysis
    if risk_success and risk_results:
        reba_scores = [r['reba_score'] for r in risk_results]
        print(f"\n🔍 DETAILED ANALYSIS:")
        print(f"   REBA Score Range: {min(reba_scores):.1f} - {max(reba_scores):.1f}")
        print(f"   Score Variation: {max(reba_scores) - min(reba_scores):.1f}")
        
        # Check if we have good distribution
        low_scores = sum(1 for s in reba_scores if s < 6)
        medium_scores = sum(1 for s in reba_scores if 6 <= s < 9)
        high_scores = sum(1 for s in reba_scores if s >= 9)
        
        print(f"   Low Risk (1-5): {low_scores} images")
        print(f"   Medium Risk (6-8): {medium_scores} images")
        print(f"   High Risk (9+): {high_scores} images")
    
    # Final verdict
    if passed_tests >= 4:
        print(f"\n🎉 SYSTEM STATUS: ✅ WORKING WELL")
        print(f"   The ErgonomicXAI system is functioning properly!")
        print(f"   🌐 Access your app at: http://localhost:8504")
    else:
        print(f"\n⚠️ SYSTEM STATUS: ❌ ISSUES FOUND")
        print(f"   {total_tests - passed_tests} tests failed - system needs fixes")
    
    return passed_tests >= 4

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
