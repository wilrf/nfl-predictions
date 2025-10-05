"""
Margin Calculator - 0 to 30 scale
0-10: Low expected value
10-20: Standard expected value
20-30: High expected value

Based on expected value relative to bet size
FAIL FAST - No negative EV bets
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MarginError(Exception):
    """Custom exception for margin calculation"""
    pass


class MarginCalculator:
    """
    Calculate margin score on 0-30 scale based on expected value
    """

    def __init__(self, bankroll: float = 10000):
        """
        Initialize calculator

        Args:
            bankroll: Total bankroll for context
        """
        if bankroll <= 0:
            raise MarginError(f"Invalid bankroll: {bankroll}")

        self.bankroll = bankroll
        self.min_margin = 0
        self.max_margin = 30

    def calculate(self,
                 edge: float,
                 kelly_fraction: float,
                 odds: int,
                 confidence: float) -> float:
        """
        Calculate margin score based on expected value

        Args:
            edge: Betting edge (e.g., 0.03 for 3%)
            kelly_fraction: Recommended bet size as fraction of bankroll
            odds: American odds for the bet
            confidence: Confidence score (50-90)

        Returns:
            Margin score between 0-30
        """
        # FAIL FAST - Negative edge
        if edge <= 0:
            raise MarginError(f"Negative or zero edge: {edge}")

        # FAIL FAST - Invalid Kelly
        if kelly_fraction <= 0 or kelly_fraction > 0.25:
            raise MarginError(f"Invalid Kelly fraction: {kelly_fraction}")

        # FAIL FAST - Invalid confidence
        if confidence < 50 or confidence > 90:
            raise MarginError(f"Invalid confidence: {confidence}")

        # Calculate expected value percentage
        ev_percentage = edge

        # Convert American odds to decimal for payout calculation
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Calculate bet amount
        bet_amount = self.bankroll * kelly_fraction

        # Calculate expected value in dollars
        expected_value = bet_amount * edge

        # Calculate EV as percentage of bet
        ev_as_pct_of_bet = (expected_value / bet_amount) * 100

        # Base margin from EV percentage
        # 2% EV = 10 margin, 4% EV = 20 margin, 6%+ EV = 30 margin
        base_margin = min(30, ev_as_pct_of_bet * 5)

        # Adjust for confidence level
        # Higher confidence = higher margin
        confidence_multiplier = 0.5 + (confidence / 90) * 0.5  # Range: 0.5-1.0
        adjusted_margin = base_margin * confidence_multiplier

        # Adjust for Kelly size
        # Larger Kelly = more conviction = higher margin
        if kelly_fraction > 0.15:  # Large bet
            kelly_boost = 1.1
        elif kelly_fraction > 0.10:  # Medium bet
            kelly_boost = 1.05
        elif kelly_fraction > 0.05:  # Standard bet
            kelly_boost = 1.0
        else:  # Small bet
            kelly_boost = 0.95

        adjusted_margin *= kelly_boost

        # Ensure within bounds
        final_margin = max(self.min_margin, min(self.max_margin, adjusted_margin))

        logger.debug(f"Margin calculation: edge={edge:.1%}, EV%={ev_as_pct_of_bet:.1f}, "
                    f"base={base_margin:.1f}, final={final_margin:.1f}")

        return round(final_margin, 1)

    def calculate_expected_value(self,
                                model_prob: float,
                                odds: int) -> tuple[float, float]:
        """
        Calculate expected value and edge from probability and odds

        Args:
            model_prob: Our model's probability (0-1)
            odds: American odds

        Returns:
            Tuple of (expected_value_pct, edge)
        """
        # FAIL FAST - Invalid probability
        if not (0 < model_prob < 1):
            raise MarginError(f"Invalid probability: {model_prob}")

        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Calculate implied probability from odds
        implied_prob = 1 / decimal_odds

        # Calculate edge
        edge = model_prob - implied_prob

        if edge <= 0:
            raise MarginError(f"No positive edge: model={model_prob:.1%}, implied={implied_prob:.1%}")

        # Calculate expected value
        # EV = (P(win) * profit) - (P(lose) * loss)
        profit = decimal_odds - 1  # Profit on $1 bet
        ev = (model_prob * profit) - ((1 - model_prob) * 1)

        # EV as percentage
        ev_pct = ev * 100

        return ev_pct, edge

    def calculate_kelly_fraction(self,
                                edge: float,
                                model_prob: float,
                                max_fraction: float = 0.25) -> float:
        """
        Calculate Kelly fraction for bet sizing

        Args:
            edge: Betting edge
            model_prob: Win probability
            max_fraction: Maximum fraction (25% for safety)

        Returns:
            Kelly fraction (capped at max_fraction)
        """
        # FAIL FAST - Invalid inputs
        if edge <= 0:
            raise MarginError(f"Non-positive edge: {edge}")

        if not (0 < model_prob < 1):
            raise MarginError(f"Invalid probability: {model_prob}")

        # Kelly formula: f = edge / b
        # Where b = decimal odds - 1
        # Simplified: f = edge / (1/model_prob - 1)

        loss_prob = 1 - model_prob
        if loss_prob == 0:
            raise MarginError("Cannot calculate Kelly with 100% win probability")

        # Calculate full Kelly
        full_kelly = edge * model_prob / loss_prob

        # Apply fractional Kelly (25% max for safety)
        fractional_kelly = min(full_kelly * 0.25, max_fraction)

        # Additional safety: never more than 5% on a single bet
        safe_kelly = min(fractional_kelly, 0.05)

        return round(safe_kelly, 4)

    def explain_margin(self, margin: float) -> str:
        """
        Provide human-readable explanation of margin level

        Args:
            margin: Margin score (0-30)

        Returns:
            Text explanation
        """
        if margin < 0:
            return "Negative expected value - DO NOT BET"
        elif margin < 5:
            return "Minimal value - consider passing"
        elif margin < 10:
            return "Low value - small position only"
        elif margin < 15:
            return "Standard value - normal position"
        elif margin < 20:
            return "Good value - solid position"
        elif margin < 25:
            return "High value - priority position"
        else:
            return "Exceptional value - maximum position"

    def calculate_breakeven_probability(self, odds: int) -> float:
        """
        Calculate breakeven probability for given odds

        Args:
            odds: American odds

        Returns:
            Breakeven probability (0-1)
        """
        # Convert to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1

        # Breakeven = 1 / decimal_odds
        breakeven = 1 / decimal_odds

        return round(breakeven, 4)