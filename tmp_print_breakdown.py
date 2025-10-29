from validate import create_mock_landmarks
from src.risk_calculation import RiskCalculator

for p in ['upright','slouched','reaching']:
    mock = create_mock_landmarks(p)
    rc = RiskCalculator()
    total, breakdown = rc.calculate_reba_score(mock.pose_world_landmarks)
    print(p, 'total=', total, 'breakdown=', breakdown)
