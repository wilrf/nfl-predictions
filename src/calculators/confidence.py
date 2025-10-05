"""
Confidence Calculator - 50 to 90 scale
50: Minimum viable edge (2%)
65: Standard confidence (3-4% edge)
80: Strong confidence (5-6% edge)
90: Exceptional (7%+ edge, very rare)

FAIL FAST - No edges below 2%
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ConfidenceError(Exception):
    """Custom exception for confidence calculation"""
    pass


class ConfidenceCalculator:
    """
    Calculate confidence score on 50-90 scale
    Incorporates edge, model certainty, and correlation penalties
    """

    # Confidence mapping based on edge percentage
    EDGE_TO_CONFIDENCE = {
        0.020: 50,   # 2.0% edge = minimum
        0.025: 55,   # 2.5% edge
        0.030: 60,   # 3.0% edge
        0.035: 65,   # 3.5% edge = standard
        0.040: 70,   # 4.0% edge
        0.045: 73,   # 4.5% edge
        0.050: 76,   # 5.0% edge = strong
        0.055: 79,   # 5.5% edge
        0.060: 82,   # 6.0% edge
        0.070: 85,   # 7.0% edge = exceptional
        0.080: 88,   # 8.0% edge
        0.100: 90    # 10%+ edge = maximum
    }

    def __init__(self):
        """Initialize calculator"""
        self.min_edge = 0.02  # 2% minimum
        self.max_confidence = 90
        self.min_confidence = 50

    def calculate(self,
                 edge: float,
                 model_probability: float,
                 market_probability: float,
                 model_certainty: float = 1.0,
                 correlation_penalty: float = 0.0) -> float:
        """
        Calculate confidence score

        Args:
            edge: Betting edge (e.g., 0.03 for 3%)
            model_probability: Our model's probability (0-1)
            market_probability: Market implied probability (0-1)
            model_certainty: How certain the model is (0-1)
            correlation_penalty: Penalty for correlated bets (0-1)

        Returns:
            Confidence score between 50-90
        """
        # FAIL FAST - Edge below minimum
        if edge < self.min_edge:
            raise ConfidenceError(f"Edge {edge:.1%} below minimum {self.min_edge:.0%}")

        # FAIL FAST - Invalid probabilities
        if not (0 < model_probability < 1):
            raise ConfidenceError(f"Invalid model probability: {model_probability}")

        if not (0 < market_probability < 1):
            raise ConfidenceError(f"Invalid market probability: {market_probability}")

        if not (0 <= model_certainty <= 1):
            raise ConfidenceError(f"Invalid model certainty: {model_certainty}")

        if not (0 <= correlation_penalty <= 1):
            raise ConfidenceError(f"Invalid correlation penalty: {correlation_penalty}")

        # Calculate base confidence from edge
        base_confidence = self._edge_to_confidence(edge)

        # Apply model certainty adjustment
        # High certainty = full confidence, low certainty = reduced
        certainty_multiplier = 0.7 + (0.3 * model_certainty)  # Range: 0.7-1.0
        adjusted_confidence = base_confidence * certainty_multiplier

        # Apply correlation penalty
        # High correlation reduces confidence
        correlation_multiplier = 1.0 - (correlation_penalty * 0.3)  # Max 30% reduction
        adjusted_confidence *= correlation_multiplier

        # Additional adjustments based on probability differential
        prob_diff = abs(model_probability - market_probability)

        # Small probability differences get slight penalty
        if prob_diff < 0.02:  # Less than 2% difference
            adjusted_confidence *= 0.95  # 5% penalty

        # Large probability differences get slight boost
        elif prob_diff > 0.10:  # More than 10% difference
            adjusted_confidence *= 1.05  # 5% boost

        # Ensure within bounds
        final_confidence = max(self.min_confidence, min(self.max_confidence, adjusted_confidence))

        logger.debug(f"Confidence calculation: edge={edge:.1%}, base={base_confidence:.0f}, "
                    f"final={final_confidence:.0f}")

        return round(final_confidence, 1)

    def _edge_to_confidence(self, edge: float) -> float:
        """Convert edge percentage to base confidence score"""
        # Find the appropriate confidence level
        for edge_threshold, confidence in sorted(self.EDGE_TO_CONFIDENCE.items()):
            if edge <= edge_threshold:
                return confidence

        # Edge above all thresholds
        return self.max_confidence

    def calculate_model_certainty(self,
                                 calibration_error: float,
                                 sample_size: int,
                                 feature_importance: Dict[str, float]) -> float:
        """
        Calculate how certain we are about the model's prediction

        Args:
            calibration_error: Model calibration error (0-1, lower is better)
            sample_size: Number of similar historical games
            feature_importance: Importance of features used

        Returns:
            Certainty score (0-1)
        """
        # FAIL FAST - Invalid inputs
        if calibration_error < 0 or calibration_error > 1:
            raise ConfidenceError(f"Invalid calibration error: {calibration_error}")

        if sample_size < 0:
            raise ConfidenceError(f"Invalid sample size: {sample_size}")

        # Base certainty from calibration
        # Perfect calibration = 1.0, poor calibration = lower
        calibration_certainty = 1.0 - calibration_error

        # Sample size factor
        # More historical examples = higher certainty
        if sample_size < 10:
            sample_factor = 0.5
        elif sample_size < 50:
            sample_factor = 0.7
        elif sample_size < 100:
            sample_factor = 0.85
        else:
            sample_factor = 1.0

        # Feature reliability factor
        # If using unreliable features, reduce certainty
        reliable_features = ['spread_movement', 'sharp_consensus', 'historical_matchup']
        feature_reliability = 0.0

        for feature, importance in feature_importance.items():
            if feature in reliable_features:
                feature_reliability += importance

        # Combine factors
        certainty = (calibration_certainty * 0.5 +
                    sample_factor * 0.3 +
                    feature_reliability * 0.2)

        return min(1.0, max(0.0, certainty))

    def adjust_for_timing(self, base_confidence: float, hours_until_game: float) -> float:
        """
        Adjust confidence based on how far out the game is

        Args:
            base_confidence: Initial confidence score
            hours_until_game: Hours until kickoff

        Returns:
            Adjusted confidence
        """
        if hours_until_game < 0:
            raise ConfidenceError("Game has already started")

        # Early in week = lower confidence (lines may move)
        if hours_until_game > 120:  # More than 5 days
            timing_multiplier = 0.9
        elif hours_until_game > 72:  # 3-5 days
            timing_multiplier = 0.95
        elif hours_until_game > 24:  # 1-3 days
            timing_multiplier = 1.0
        else:  # Less than 24 hours
            timing_multiplier = 1.05  # Slight boost - less time for news to break

        adjusted = base_confidence * timing_multiplier

        return max(self.min_confidence, min(self.max_confidence, adjusted))

    def explain_confidence(self, confidence: float) -> str:
        """
        Provide human-readable explanation of confidence level

        Args:
            confidence: Confidence score (50-90)

        Returns:
            Text explanation
        """
        if confidence < 50:
            return "Below minimum threshold"
        elif confidence < 60:
            return "Minimum edge - bet only if no better options"
        elif confidence < 70:
            return "Standard confidence - reasonable bet"
        elif confidence < 80:
            return "Good confidence - solid value"
        elif confidence < 85:
            return "Strong confidence - priority bet"
        else:
            return "Exceptional confidence - rare opportunity"