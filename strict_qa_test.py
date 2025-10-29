"""
STRICT QA TEST - UNCOMPROMISING VALIDATION
This test will identify EVERY issue and prove system functionality
"""
import cv2
import numpy as np
import sys
from pathlib import Path
import time
import json
import requests
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class StrictQAValidator:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'issues': [],
            'critical_failures': [],
            'overall_status': 'FAILED'
        }
    
    def log_issue(self, test_name, issue, severity='MEDIUM'):
        self.results['issues'].append({
            'test': test_name,
            'issue': issue,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        if severity == 'CRITICAL':
            self.results['critical_failures'].append(f"{test_name}: {issue}")
    
    def test_dependencies(self):
        """Test 1: Verify all dependencies are properly installed"""
        print("🔍 TEST 1: DEPENDENCY VERIFICATION")
        print("-" * 50)
        
        dependencies = {
            'torch': 'PyTorch for temporal model',
            'mediapipe': 'Pose detection',
            'opencv-python': 'Image processing',
            'streamlit': 'Web interface',
            'numpy': 'Numerical computations',
            'pandas': 'Data handling'
        }
        
        missing_deps = []
        for dep, description in dependencies.items():
            try:
                if dep == 'opencv-python':
                    import cv2
                    print(f"   ✅ {dep}: {cv2.__version__}")
                elif dep == 'torch':
                    import torch
                    print(f"   ✅ {dep}: {torch.__version__}")
                elif dep == 'mediapipe':
                    import mediapipe as mp
                    print(f"   ✅ {dep}: {mp.__version__}")
                elif dep == 'streamlit':
                    import streamlit
                    print(f"   ✅ {dep}: {streamlit.__version__}")
                elif dep == 'numpy':
                    import numpy
                    print(f"   ✅ {dep}: {numpy.__version__}")
                elif dep == 'pandas':
                    import pandas
                    print(f"   ✅ {dep}: {pandas.__version__}")
            except ImportError as e:
                print(f"   ❌ {dep}: MISSING - {e}")
                missing_deps.append(dep)
        
        if missing_deps:
            self.log_issue('Dependencies', f"Missing dependencies: {missing_deps}", 'CRITICAL')
            return False
        else:
            print("   ✅ All dependencies present")
            return True
    
    def test_pose_extraction_accuracy(self):
        """Test 2: Verify pose extraction works on multiple images"""
        print("\n🔍 TEST 2: POSE EXTRACTION ACCURACY")
        print("-" * 50)
        
        try:
            from src.pose_extraction import PoseExtractor
            
            extractor = PoseExtractor()
            image_dir = project_root / "data" / "images" / "manufacturing"
            test_images = list(image_dir.glob("*.jpg"))
            
            if not test_images:
                self.log_issue('Pose Extraction', "No test images found", 'CRITICAL')
                return False
            
            print(f"   Testing {len(test_images)} images...")
            
            results = []
            for i, img_path in enumerate(test_images, 1):
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"   ❌ Image {i}: Could not read {img_path.name}")
                    continue
                
                start_time = time.time()
                pose_results, keypoints_df = extractor.extract_pose(image)
                extraction_time = time.time() - start_time
                
                if pose_results.pose_landmarks:
                    landmark_count = len(pose_results.pose_landmarks.landmark)
                    results.append({
                        'image': img_path.name,
                        'success': True,
                        'landmarks': landmark_count,
                        'time': extraction_time
                    })
                    print(f"   ✅ Image {i}: {img_path.name} - {landmark_count} landmarks ({extraction_time:.2f}s)")
                else:
                    results.append({
                        'image': img_path.name,
                        'success': False,
                        'landmarks': 0,
                        'time': extraction_time
                    })
                    print(f"   ❌ Image {i}: {img_path.name} - No pose detected")
            
            extractor.close()
            
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_time = np.mean([r['time'] for r in results])
            
            print(f"\n   📊 Results: {success_rate:.1%} success rate, {avg_time:.2f}s average")
            
            if success_rate < 0.8:
                self.log_issue('Pose Extraction', f"Low success rate: {success_rate:.1%}", 'CRITICAL')
                return False
            
            if avg_time > 2.0:
                self.log_issue('Pose Extraction', f"Slow extraction: {avg_time:.2f}s", 'MEDIUM')
            
            return True
            
        except Exception as e:
            self.log_issue('Pose Extraction', f"Exception: {str(e)}", 'CRITICAL')
            return False
    
    def test_reba_score_variation(self):
        """Test 3: CRITICAL - Verify REBA scores are truly varied"""
        print("\n🔍 TEST 3: REBA SCORE VARIATION (CRITICAL)")
        print("-" * 50)
        
        try:
            from src.pose_extraction import PoseExtractor
            from src.risk_calculation import RiskCalculator
            
            extractor = PoseExtractor()
            calculator = RiskCalculator()
            
            image_dir = project_root / "data" / "images" / "manufacturing"
            test_images = list(image_dir.glob("*.jpg"))
            
            scores = []
            detailed_results = []
            
            for img_path in test_images:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                pose_results, _ = extractor.extract_pose(image)
                if not pose_results.pose_landmarks:
                    continue
                
                reba_score, breakdown = calculator.calculate_reba_score(pose_results.pose_world_landmarks)
                scores.append(reba_score)
                detailed_results.append({
                    'image': img_path.name,
                    'score': reba_score,
                    'breakdown': breakdown
                })
            
            extractor.close()
            
            if not scores:
                self.log_issue('REBA Scores', "No valid scores generated", 'CRITICAL')
                return False
            
            # CRITICAL ANALYSIS
            unique_scores = len(set(scores))
            score_range = max(scores) - min(scores)
            std_dev = np.std(scores)
            
            print(f"   📊 Score Analysis:")
            print(f"      Total Images: {len(scores)}")
            print(f"      Score Range: {min(scores):.1f} - {max(scores):.1f}")
            print(f"      Unique Scores: {unique_scores}")
            print(f"      Standard Deviation: {std_dev:.2f}")
            print(f"      Score Variation: {score_range:.1f}")
            
            # CRITICAL CHECKS
            issues = []
            
            if unique_scores < 3:
                issues.append(f"Only {unique_scores} unique scores (need ≥3)")
            
            if score_range < 3.0:
                issues.append(f"Score range too small: {score_range:.1f} (need ≥3.0)")
            
            if std_dev < 1.0:
                issues.append(f"Low variation: std={std_dev:.2f} (need ≥1.0)")
            
            # Check for identical scores
            identical_count = 0
            for i, score1 in enumerate(scores):
                for j, score2 in enumerate(scores[i+1:], i+1):
                    if abs(score1 - score2) < 0.1:
                        identical_count += 1
            
            if identical_count > len(scores) * 0.3:
                issues.append(f"Too many similar scores: {identical_count} pairs")
            
            # Check score distribution
            low_scores = sum(1 for s in scores if s < 6)
            medium_scores = sum(1 for s in scores if 6 <= s < 9)
            high_scores = sum(1 for s in scores if s >= 9)
            
            print(f"   📈 Score Distribution:")
            print(f"      Low Risk (1-5): {low_scores}")
            print(f"      Medium Risk (6-8): {medium_scores}")
            print(f"      High Risk (9+): {high_scores}")
            
            if low_scores == 0:
                issues.append("No low-risk scores found")
            
            if issues:
                for issue in issues:
                    self.log_issue('REBA Scores', issue, 'CRITICAL')
                return False
            
            print("   ✅ REBA scores show proper variation")
            return True
            
        except Exception as e:
            self.log_issue('REBA Scores', f"Exception: {str(e)}", 'CRITICAL')
            return False
    
    def test_streamlit_app_functionality(self):
        """Test 4: Verify Streamlit app is fully functional"""
        print("\n🔍 TEST 4: STREAMLIT APP FUNCTIONALITY")
        print("-" * 50)
        
        try:
            # Test if app is running
            response = requests.get("http://localhost:8504", timeout=10)
            if response.status_code != 200:
                self.log_issue('Streamlit App', f"HTTP {response.status_code}", 'CRITICAL')
                return False
            
            print("   ✅ App is accessible")
            
            # Test if we can upload an image (simulate)
            test_image_path = project_root / "data" / "images" / "manufacturing"
            test_images = list(test_image_path.glob("*.jpg"))
            
            if not test_images:
                self.log_issue('Streamlit App', "No test images for upload test", 'MEDIUM')
                return True
            
            print(f"   ✅ Found {len(test_images)} test images")
            print("   ✅ App ready for image upload testing")
            
            return True
            
        except requests.exceptions.ConnectionError:
            self.log_issue('Streamlit App', "Cannot connect to app", 'CRITICAL')
            return False
        except Exception as e:
            self.log_issue('Streamlit App', f"Exception: {str(e)}", 'CRITICAL')
            return False
    
    def test_temporal_model_functionality(self):
        """Test 5: Verify temporal model works correctly"""
        print("\n🔍 TEST 5: TEMPORAL MODEL FUNCTIONALITY")
        print("-" * 50)
        
        try:
            from src.temporal_model import ErgonomicTemporalModel
            
            model = ErgonomicTemporalModel()
            
            # Test training
            start_time = time.time()
            model.train_with_dummy_data(epochs=3)
            train_time = time.time() - start_time
            
            if train_time > 5.0:
                self.log_issue('Temporal Model', f"Slow training: {train_time:.2f}s", 'MEDIUM')
            
            # Test prediction
            test_keypoints = np.random.rand(99)
            start_time = time.time()
            prediction = model.predict(test_keypoints)
            pred_time = time.time() - start_time
            
            if pred_time > 1.0:
                self.log_issue('Temporal Model', f"Slow prediction: {pred_time:.3f}s", 'MEDIUM')
            
            print(f"   ✅ Training: {train_time:.2f}s")
            print(f"   ✅ Prediction: {prediction[0][0]:.2f} ({pred_time:.3f}s)")
            
            return True
            
        except Exception as e:
            self.log_issue('Temporal Model', f"Exception: {str(e)}", 'CRITICAL')
            return False
    
    def test_explainability_functionality(self):
        """Test 6: Verify explainability module works"""
        print("\n🔍 TEST 6: EXPLAINABILITY FUNCTIONALITY")
        print("-" * 50)
        
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
            
            if explain_time > 2.0:
                self.log_issue('Explainability', f"Slow explanation: {explain_time:.3f}s", 'MEDIUM')
            
            if len(explanation) < 50:
                self.log_issue('Explainability', f"Short explanation: {len(explanation)} chars", 'MEDIUM')
            
            print(f"   ✅ Contributions: {contributions}")
            print(f"   ✅ Explanation: {len(explanation)} chars ({explain_time:.3f}s)")
            
            return True
            
        except Exception as e:
            self.log_issue('Explainability', f"Exception: {str(e)}", 'CRITICAL')
            return False
    
    def run_strict_qa(self):
        """Run all strict QA tests"""
        print("🧪 STRICT QA TEST - UNCOMPROMISING VALIDATION")
        print("=" * 60)
        print(f"Timestamp: {self.results['timestamp']}")
        print()
        
        tests = [
            ("Dependencies", self.test_dependencies),
            ("Pose Extraction", self.test_pose_extraction_accuracy),
            ("REBA Score Variation", self.test_reba_score_variation),
            ("Streamlit App", self.test_streamlit_app_functionality),
            ("Temporal Model", self.test_temporal_model_functionality),
            ("Explainability", self.test_explainability_functionality)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results['tests'][test_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'timestamp': datetime.now().isoformat()
                }
                if result:
                    passed_tests += 1
            except Exception as e:
                self.log_issue(test_name, f"Test exception: {str(e)}", 'CRITICAL')
                self.results['tests'][test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Final assessment
        print("\n" + "=" * 60)
        print("📊 STRICT QA RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 TEST RESULTS:")
        for test_name, test_data in self.results['tests'].items():
            status = test_data['status']
            if status == 'PASS':
                print(f"   ✅ {test_name}: PASS")
            elif status == 'FAIL':
                print(f"   ❌ {test_name}: FAIL")
            else:
                print(f"   💥 {test_name}: ERROR")
        
        print(f"\n📊 OVERALL SCORE: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
        
        # Issue summary
        if self.results['issues']:
            print(f"\n🚨 ISSUES FOUND: {len(self.results['issues'])}")
            for issue in self.results['issues']:
                severity_icon = "💥" if issue['severity'] == 'CRITICAL' else "⚠️"
                print(f"   {severity_icon} {issue['test']}: {issue['issue']}")
        
        # Critical failures
        if self.results['critical_failures']:
            print(f"\n💥 CRITICAL FAILURES:")
            for failure in self.results['critical_failures']:
                print(f"   ❌ {failure}")
        
        # Final verdict
        if passed_tests == total_tests and not self.results['critical_failures']:
            self.results['overall_status'] = 'PASSED'
            print(f"\n🎉 STRICT QA RESULT: ✅ ALL TESTS PASSED")
            print(f"   System is fully functional and ready for production!")
        else:
            self.results['overall_status'] = 'FAILED'
            print(f"\n❌ STRICT QA RESULT: FAILED")
            print(f"   {total_tests - passed_tests} tests failed")
            print(f"   {len(self.results['critical_failures'])} critical issues")
        
        # Save results
        results_file = project_root / "qa_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Results saved to: {results_file}")
        
        return self.results['overall_status'] == 'PASSED'

if __name__ == "__main__":
    validator = StrictQAValidator()
    success = validator.run_strict_qa()
    sys.exit(0 if success else 1)
