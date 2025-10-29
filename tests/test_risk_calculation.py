import unittest
from src import risk_calculation


class TestRiskCalculation(unittest.TestCase):
    def test_compute_angle_straight(self):
        # points in a straight line: angle at b should be ~180
        a = (0.0, 0.0)
        b = (0.5, 0.0)
        c = (1.0, 0.0)
        ang = risk_calculation.compute_angle(a, b, c)
        self.assertAlmostEqual(ang, 180.0, places=1)

    def test_compute_angle_right(self):
        a = (0.0, 0.0)
        b = (0.0, 1.0)
        c = (1.0, 1.0)
        ang = risk_calculation.compute_angle(a, b, c)
        self.assertAlmostEqual(ang, 90.0, places=1)

    def test_angles_to_risk_bounds(self):
        # all zeros -> minimal posture -> low risk but >=1
        angs = {"neck": 0.0, "back": 0.0, "shoulder": 0.0, "elbow": 180.0, "knee": 0.0}
        score = risk_calculation.angles_to_risk(angs)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 10.0)


if __name__ == "__main__":
    unittest.main()
