import numpy as np

class ExplainabilityModule:
    """
    Generates dynamic, context-aware, and actionable advice based on a detailed
    analysis of the REBA score breakdown. This is the definitive version.
    """
    def __init__(self, temporal_model, risk_calculator, score_breakdown):
        """
        Initializes the module with all necessary components.
        
        Args:
            temporal_model (ErgonomicTemporalModel): Not used for advice, but kept for compatibility.
            risk_calculator (RiskCalculator): The risk calculator instance, needed to access the final score.
            score_breakdown (dict): The dictionary of per-part REBA component scores.
        """
        self.risk_calculator = risk_calculator
        self.score_breakdown = score_breakdown if score_breakdown else {}

    def generate_explanation(self, keypoints_flat_sequence):
        """
        Generates a conceptual SHAP chart for visualization and, more importantly,
        creates highly relevant, rule-based advice from the REBA score breakdown
        for the specific posture being analyzed.

        Returns:
            tuple: A tuple containing a dictionary for the chart and the final advice string.
        """
        # The "Parts Importance" chart is a conceptual visualization of risk contribution.
        part_contributions_for_chart = self._simulate_importance_from_reba()

        # The advice is generated from a more detailed, intelligent analysis of the score breakdown.
        explanation_text = self._create_intelligent_advice()
        
        return part_contributions_for_chart, explanation_text

    def _simulate_importance_from_reba(self):
        """
        Simulates logical values for the "Parts Importance" bar chart based on the
        proportional risk contribution of each body part from the REBA breakdown.
        """
        total_score = sum(self.score_breakdown.values())
        part_contributions = {'trunk': 0.0, 'arms': 0.0, 'legs': 0.0}

        if total_score > 0:
            # Distribute a total positive effect (normalized to 1.0) proportionally to risk scores
            for part, score in self.score_breakdown.items():
                part_contributions[part] = score / total_score if total_score > 0 else 0
        else: # If risk is very low, show small, almost neutral contributions
            part_contributions = {'trunk': 0.1, 'arms': 0.05, 'legs': 0.02}
        
        return part_contributions

    def _create_intelligent_advice(self):
        """
        Creates a specific, context-aware text explanation directly from the
        `self.score_breakdown` and the final score calculated by the RiskCalculator.
        """
        # The final REBA score determines the overall risk level and the need for advice.
        # This uses the `last_landmarks` stored in the risk_calculator to get the definitive score.
        final_reba_score, _ = self.risk_calculator.calculate_reba_score(self.risk_calculator.last_landmarks)

        # If the risk score is low, provide positive reinforcement.
        if not self.score_breakdown or final_reba_score < 4:
            return "Risk level is low. The current posture is acceptable and poses minimal ergonomic risk."

        # Sort the body parts by their individual risk score, from highest to lowest
        sorted_parts = sorted(self.score_breakdown.items(), key=lambda item: item[1], reverse=True)
        
        primary_risk_part, primary_score = sorted_parts[0]
        
        # Define specific, actionable advice for each body part
        advice_map = {
            'trunk': "focus on straightening your back and avoid twisting your torso.",
            'arms': "try to keep your elbows closer to your body and avoid over-reaching.",
            'legs': "maintain a more stable stance, perhaps by bending your knees slightly."
        }
        
        # --- Generate the final, context-aware advice string ---
        risk_level_text = 'High' if final_reba_score > 7 else 'Medium'
        advice = f"The posture indicates a **{risk_level_text} Risk**."
        advice += f" This is primarily driven by the awkward position of the **{primary_risk_part.upper()}**."

        # Check if a second body part is also a significant contributor to the risk
        if len(sorted_parts) > 1:
            secondary_risk_part, secondary_score = sorted_parts[1]
            # If the second part's score is high and close to the primary, it's also a key factor
            if secondary_score > 2 and secondary_score > primary_score * 0.6:
                advice += f" The position of the **{secondary_risk_part.upper()}** is also a significant factor."

        advice += f"\n\n**Actionable Advice:** To reduce strain, {advice_map.get(primary_risk_part, 'adjust your overall posture')}."

        return advice

