"""
Comprehensive test of ErgonomicXAI system to identify real issues
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
    """Test pose extraction with real images"""
    print("🔍 Testing Pose Extraction...")
    
    try:
        from src.pose_extraction import PoseExtractor
        extractor = PoseExtractor()
        
        # Test with a real image
        image_dir = project_root / "data" / "images" / "manufacturing"
        test_images = list(image_dir.glob("*.jpg"))[:3]
        
        results = []
        for img_path in test_images:
            print(f"  Testing: {img_path.name}")
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    ❌ Could not read image")
                continue
                
            start_time = time.time()
            pose_results, keypoints_df = extractor.extract_pose(image)
            extraction_time = time.time() - start_time
            
            if pose_results.pose_landmarks:
                print(f"    ✅ Pose detected in {extraction_time:.2f}s")
                print(f"    📊 Landmarks: {len(pose_results.pose_landmarks.landmark)}")
                results.append({
                    'image': img_path.name,
                    'landmarks': len(pose_results.pose_landmarks.landmark),
                    'time': extraction_time,
                    'success': True
                })
            else:
                print(f"    ❌ No pose detected")
                results.append({
                    'image': img_path.name,
                    'landmarks': 0,
                    'time': extraction_time,
                    'success': False
                })
        
        extractor.close()
        return results
        
    except Exception as e:
        print(f"❌ Pose extraction error: {e}")
        return []

def test_risk_calculation():
    """Test risk calculation with pose data"""
    print("\n🔍 Testing Risk Calculation...")
    
    try:
        from src.risk_calculation import RiskCalculator
        from src.pose_extraction import PoseExtractor
        
        calculator = RiskCalculator()
        extractor = PoseExtractor()
        
        # Test with real images
        image_dir = project_root / "data" / "images" / "manufacturing"
        test_images = list(image_dir.glob("*.jpg"))[:5]
        
        results = []
        for img_path in test_images:
            print(f"  Testing: {img_path.name}")
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            pose_results, _ = extractor.extract_pose(image)
            if not pose_results.pose_landmarks:
                print(f"    ❌ No pose for risk calculation")
                continue
            
            start_time = time.time()
            reba_score, score_breakdown = calculator.calculate_reba_score(pose_results.pose_world_landmarks)
            calc_time = time.time() - start_time
            
            print(f"    ✅ REBA Score: {reba_score:.2f} (calculated in {calc_time:.3f}s)")
            print(f"    📊 Breakdown: {score_breakdown}")
            
            results.append({
                'image': img_path.name,
                'reba_score': reba_score,
                'breakdown': score_breakdown,
                'time': calc_time
            })
        
        extractor.close()
        return results
        
    except Exception as e:
        print(f"❌ Risk calculation error: {e}")
        return []

def test_temporal_model():
    """Test temporal model"""
    print("\n🔍 Testing Temporal Model...")
    
    try:
        from src.temporal_model import ErgonomicTemporalModel
        
        model = ErgonomicTemporalModel()
        
        # Test training
        print("  Training model...")
        start_time = time.time()
        model.train_with_dummy_data(epochs=2)
        train_time = time.time() - start_time
        print(f"  ✅ Training completed in {train_time:.2f}s")
        
        # Test prediction
        test_keypoints = np.random.rand(99)  # 33 landmarks * 3 coords
        start_time = time.time()
        prediction = model.predict(test_keypoints)
        pred_time = time.time() - start_time
        
        print(f"  ✅ Prediction: {prediction[0][0]:.2f} (calculated in {pred_time:.3f}s)")
        
        return {
            'train_time': train_time,
            'prediction': prediction[0][0],
            'pred_time': pred_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Temporal model error: {e}")
        return {'success': False, 'error': str(e)}

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
        
        print(f"  ✅ Contributions: {contributions}")
        print(f"  📝 Explanation: {explanation[:100]}...")
        print(f"  ⏱️ Generated in {explain_time:.3f}s")
        
        return {
            'contributions': contributions,
            'explanation': explanation,
            'time': explain_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Explainability error: {e}")
        return {'success': False, 'error': str(e)}

def analyze_results(pose_results, risk_results, temporal_results, explain_results):
    """Analyze all test results and identify issues"""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Pose extraction analysis
    if pose_results:
        success_rate = sum(1 for r in pose_results if r['success']) / len(pose_results)
        avg_time = np.mean([r['time'] for r in pose_results])
        print(f"\n🎯 POSE EXTRACTION:")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Average Time: {avg_time:.2f}s")
        
        if success_rate < 0.8:
            print("   ⚠️ ISSUE: Low pose detection success rate")
        if avg_time > 2.0:
            print("   ⚠️ ISSUE: Slow pose extraction")
    
    # Risk calculation analysis
    if risk_results:
        reba_scores = [r['reba_score'] for r in risk_results]
        score_range = max(reba_scores) - min(reba_scores)
        avg_score = np.mean(reba_scores)
        
        print(f"\n🎯 RISK CALCULATION:")
        print(f"   Score Range: {min(reba_scores):.1f} - {max(reba_scores):.1f}")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Score Variation: {score_range:.1f}")
        
        if score_range < 2.0:
            print("   ⚠️ ISSUE: Low score variation - may be generating similar scores")
        if avg_score > 8.0:
            print("   ⚠️ ISSUE: Consistently high risk scores")
        if avg_score < 2.0:
            print("   ⚠️ ISSUE: Consistently low risk scores")
    
    # Temporal model analysis
    if temporal_results and temporal_results.get('success'):
        print(f"\n🎯 TEMPORAL MODEL:")
        print(f"   Training Time: {temporal_results['train_time']:.2f}s")
        print(f"   Prediction: {temporal_results['prediction']:.2f}")
        
        if temporal_results['train_time'] > 10:
            print("   ⚠️ ISSUE: Slow model training")
        if abs(temporal_results['prediction'] - 5.0) < 0.5:
            print("   ⚠️ ISSUE: Predictions may be too centered around 5.0")
    
    # Explainability analysis
    if explain_results and explain_results.get('success'):
        print(f"\n🎯 EXPLAINABILITY:")
        print(f"   Generation Time: {explain_results['time']:.3f}s")
        print(f"   Contributions: {explain_results['contributions']}")
        
        if explain_results['time'] > 1.0:
            print("   ⚠️ ISSUE: Slow explanation generation")
    
    # Overall system issues
    print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")
    
    issues = []
    
    if pose_results and sum(1 for r in pose_results if r['success']) / len(pose_results) < 0.5:
        issues.append("❌ Poor pose detection - many images fail to detect poses")
    
    if risk_results:
        reba_scores = [r['reba_score'] for r in risk_results]
        if max(reba_scores) - min(reba_scores) < 1.0:
            issues.append("❌ Low score variation - system may not be differentiating postures properly")
        
        if np.mean(reba_scores) > 7.0:
            issues.append("❌ Consistently high risk scores - may be overestimating risk")
        elif np.mean(reba_scores) < 3.0:
            issues.append("❌ Consistently low risk scores - may be underestimating risk")
    
    if not temporal_results.get('success'):
        issues.append("❌ Temporal model not working properly")
    
    if not explain_results.get('success'):
        issues.append("❌ Explainability module not working properly")
    
    if not issues:
        print("✅ No critical issues found - system appears to be working well!")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    return issues

def main():
    """Run comprehensive test suite"""
    print("🧪 COMPREHENSIVE ERGONOMICXAI TEST SUITE")
    print("="*60)
    
    # Run all tests
    pose_results = test_pose_extraction()
    risk_results = test_risk_calculation()
    temporal_results = test_temporal_model()
    explain_results = test_explainability()
    
    # Analyze results
    issues = analyze_results(pose_results, risk_results, temporal_results, explain_results)
    
    print(f"\n📋 SUMMARY:")
    print(f"   Tests Run: 4")
    print(f"   Critical Issues: {len(issues)}")
    print(f"   System Status: {'❌ Issues Found' if issues else '✅ Working Well'}")
    
    return issues

if __name__ == "__main__":
    main()
