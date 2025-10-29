"""Deterministic advice generator for image-level REBA outputs.

This module converts a numeric REBA score, the breakdown, part contributions,
and optional landmarks into a short, deterministic, actionable advice string.
It complements SHAP-based advice (which explains model attributions) with a
clear, rule-based instruction that's directly grounded in measurable posture
features computed from the landmarks.
"""
from typing import Dict, Any

def _severity_phrase(score: float) -> str:
    if score >= 11:
        return 'Critically high risk — act immediately.'
    if score >= 8:
        return 'High risk — address promptly.'
    if score >= 5:
        return 'Moderate risk — correct posture soon.'
    return 'Low risk — maintain good posture.'


def generate_advice(reba: float, breakdown: Dict[str, float], parts: Dict[str, float], landmarks: Any = None) -> str:
    """Return concise, deterministic, image-specific advice.

    This version gives side-specific hints (when landmarks available), uses a
    severity phrase based on REBA, and prefers the deterministic breakdown
    to choose the target area. The wording is split into a 1-line summary
    and 1-2 short, prescriptive actions.
    """
    primary = None
    if breakdown and isinstance(breakdown, dict):
        try:
            primary = max(breakdown, key=breakdown.get)
        except Exception:
            primary = None
    if primary is None and parts:
        primary = max(parts, key=parts.get)

    sev = _severity_phrase(reba)
    header = f"REBA {reba:.1f} — {sev} Focus: {primary if primary else 'general posture'}."

    actions = []
    if primary == 'trunk':
        actions.append('Stand upright: reduce forward bending and avoid twisting.')
        # side-specific: detect strong forward lean or head-forward
        try:
            if landmarks is not None and hasattr(landmarks, 'landmark'):
                lm = landmarks.landmark
                nose_x = float(lm[0].x)
                hip_mid_x = (float(lm[23].x) + float(lm[24].x)) / 2.0
                if nose_x - hip_mid_x > 0.06:
                    actions.append('Head/torso appears forward — pull your shoulders back and tuck your chin slightly.')
        except Exception:
            pass

    elif primary == 'arms':
        actions.append('Bring items closer and support your forearms; avoid reaching overhead or far forward.')
        try:
            if landmarks is not None and hasattr(landmarks, 'landmark'):
                lm = landmarks.landmark
                # approximate forward reach: wrist x vs shoulder x
                l_wx = float(lm[15].x); r_wx = float(lm[16].x)
                l_sx = float(lm[11].x); r_sx = float(lm[12].x)
                if l_wx - l_sx > 0.12:
                    actions.append('Left arm appears extended — lower the load or use arm support.')
                if r_wx - r_sx > 0.12:
                    actions.append('Right arm appears extended — lower the load or use arm support.')
        except Exception:
            pass

    elif primary == 'legs':
        actions.append('Adopt a stable stance and bend knees slightly when lifting; avoid prolonged single-leg support.')
        try:
            if landmarks is not None and hasattr(landmarks, 'landmark'):
                lm = landmarks.landmark
                l_knee_y = float(lm[25].y); r_knee_y = float(lm[26].y)
                l_hip_y = float(lm[23].y); r_hip_y = float(lm[24].y)
                # if knees are high (straight), suggest bend
                if (abs(l_hip_y - l_knee_y) < 0.03) or (abs(r_hip_y - r_knee_y) < 0.03):
                    actions.append('Knees look straight — add slight knee flexion to reduce back load.')
        except Exception:
            pass

    else:
        actions.append('Maintain neutral spine, relaxed shoulders, and avoid excessive reaching.')

    # Compose final advice: header + up to 2 action sentences
    final = ' '.join([header] + actions[:2])
    return final
