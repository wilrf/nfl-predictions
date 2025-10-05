"""
Correlation Warning Engine
Identifies and warns about correlated bets
Does NOT eliminate bets - only provides warnings

Based on research:
- Same game: 0.45 correlation
- Favorite + Over: 0.73 correlation
- Underdog + Under: 0.68 correlation
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CorrelationError(Exception):
    """Custom exception for correlation operations"""
    pass


@dataclass
class Bet:
    """Represents a betting opportunity"""
    game_id: str
    bet_type: str  # 'spread', 'total', 'moneyline'
    selection: str  # 'home', 'away', 'over', 'under'
    team: str
    line: float
    is_favorite: bool
    confidence: float
    margin: float


@dataclass
class CorrelationWarning:
    """Warning about correlated bets"""
    bet1: Bet
    bet2: Bet
    correlation_type: str
    correlation_value: float
    warning_level: str  # 'high', 'moderate', 'low'
    message: str
    visual_indicator: str  # 🔴, 🟡, 🟢


class CorrelationEngine:
    """
    Detects and warns about correlated betting opportunities
    WARNINGS ONLY - does not eliminate bets
    """

    # Research-based correlation values
    CORRELATIONS = {
        'same_game_spread_total': 0.45,
        'same_game_spread_ml': 0.85,  # Very high - same outcome
        'same_game_total_ml': 0.35,
        'favorite_over': 0.73,
        'underdog_under': 0.68,
        'same_team_different_games': 0.25,
        'division_rivals': 0.20,
        'weather_affected': 0.30,  # Under bets in bad weather games
        'primetime_overs': 0.15  # Slight correlation in primetime games
    }

    # Warning thresholds
    HIGH_CORRELATION = 0.70
    MODERATE_CORRELATION = 0.40

    def __init__(self):
        """Initialize correlation engine"""
        self.warnings = []

    def check_correlations(self, bets: List[Bet]) -> List[CorrelationWarning]:
        """
        Check all bets for correlations

        Args:
            bets: List of betting opportunities

        Returns:
            List of correlation warnings
        """
        if not bets:
            raise CorrelationError("No bets provided")

        warnings = []

        # Check each pair of bets
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                bet1 = bets[i]
                bet2 = bets[j]

                # Check for various correlation types
                correlation = self._calculate_correlation(bet1, bet2)

                if correlation['value'] > 0:
                    warning = self._create_warning(bet1, bet2, correlation)
                    warnings.append(warning)

        # Sort by correlation value (highest first)
        warnings.sort(key=lambda w: w.correlation_value, reverse=True)

        self.warnings = warnings
        return warnings

    def _calculate_correlation(self, bet1: Bet, bet2: Bet) -> Dict:
        """Calculate correlation between two bets"""
        correlation_type = None
        correlation_value = 0

        # Same game correlations
        if bet1.game_id == bet2.game_id:
            if bet1.bet_type == 'spread' and bet2.bet_type == 'total':
                correlation_type = 'same_game_spread_total'
                correlation_value = self.CORRELATIONS['same_game_spread_total']

                # Adjust for favorite/over correlation
                if bet1.is_favorite and bet2.selection == 'over':
                    correlation_type = 'same_game_favorite_over'
                    correlation_value = self.CORRELATIONS['favorite_over']
                elif not bet1.is_favorite and bet2.selection == 'under':
                    correlation_type = 'same_game_underdog_under'
                    correlation_value = self.CORRELATIONS['underdog_under']

            elif bet1.bet_type == 'spread' and bet2.bet_type == 'moneyline':
                if bet1.selection == bet2.selection:  # Same team
                    correlation_type = 'same_game_spread_ml'
                    correlation_value = self.CORRELATIONS['same_game_spread_ml']

            elif bet1.bet_type == 'total' and bet2.bet_type == 'moneyline':
                correlation_type = 'same_game_total_ml'
                correlation_value = self.CORRELATIONS['same_game_total_ml']

        # Different game correlations
        else:
            # Same team in different games
            if bet1.team == bet2.team:
                correlation_type = 'same_team_different_games'
                correlation_value = self.CORRELATIONS['same_team_different_games']

            # Favorite/Over correlation across games
            elif (bet1.bet_type == 'spread' and bet2.bet_type == 'total' and
                  bet1.is_favorite and bet2.selection == 'over'):
                correlation_type = 'cross_game_favorite_over'
                correlation_value = self.CORRELATIONS['favorite_over'] * 0.5  # Reduced for different games

        return {
            'type': correlation_type,
            'value': correlation_value
        }

    def _create_warning(self, bet1: Bet, bet2: Bet, correlation: Dict) -> CorrelationWarning:
        """Create a correlation warning"""
        value = correlation['value']

        # Determine warning level
        if value >= self.HIGH_CORRELATION:
            level = 'high'
            indicator = '🔴'
        elif value >= self.MODERATE_CORRELATION:
            level = 'moderate'
            indicator = '🟡'
        else:
            level = 'low'
            indicator = '🟢'

        # Create message
        message = self._generate_warning_message(bet1, bet2, correlation)

        return CorrelationWarning(
            bet1=bet1,
            bet2=bet2,
            correlation_type=correlation['type'],
            correlation_value=value,
            warning_level=level,
            message=message,
            visual_indicator=indicator
        )

    def _generate_warning_message(self, bet1: Bet, bet2: Bet, correlation: Dict) -> str:
        """Generate human-readable warning message"""
        corr_type = correlation['type']
        value = correlation['value']
        pct = f"{value:.0%}"

        if corr_type == 'same_game_spread_total':
            return f"Same game bets ({pct} correlation): {bet1.team} spread and total"
        elif corr_type == 'same_game_favorite_over':
            return f"HIGH: Favorite + Over in same game ({pct} correlation)"
        elif corr_type == 'same_game_underdog_under':
            return f"HIGH: Underdog + Under in same game ({pct} correlation)"
        elif corr_type == 'same_game_spread_ml':
            return f"VERY HIGH: Same team spread + ML ({pct} correlation)"
        elif corr_type == 'same_team_different_games':
            return f"Same team in multiple games ({pct} correlation)"
        else:
            return f"{corr_type.replace('_', ' ').title()} ({pct} correlation)"

    def calculate_portfolio_correlation(self, bets: List[Bet]) -> float:
        """
        Calculate overall portfolio correlation

        Args:
            bets: List of bets

        Returns:
            Portfolio correlation score (0-1)
        """
        if len(bets) < 2:
            return 0.0

        # Get all pairwise correlations
        total_correlation = 0
        pair_count = 0

        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                correlation = self._calculate_correlation(bets[i], bets[j])
                total_correlation += correlation['value']
                pair_count += 1

        if pair_count == 0:
            return 0.0

        # Average correlation
        avg_correlation = total_correlation / pair_count

        return round(avg_correlation, 3)

    def apply_correlation_penalty(self, confidence: float, correlation: float) -> float:
        """
        Apply penalty to confidence based on correlation

        Args:
            confidence: Original confidence (50-90)
            correlation: Correlation value (0-1)

        Returns:
            Adjusted confidence
        """
        if correlation <= 0:
            return confidence

        # Maximum 30% penalty for high correlation
        max_penalty = 0.30
        penalty = min(correlation * max_penalty, max_penalty)

        adjusted = confidence * (1 - penalty)

        return max(50, adjusted)  # Never go below minimum

    def get_correlation_summary(self) -> Dict:
        """Get summary of all correlations"""
        if not self.warnings:
            return {
                'total_warnings': 0,
                'high': 0,
                'moderate': 0,
                'low': 0,
                'highest_correlation': 0
            }

        high_warnings = [w for w in self.warnings if w.warning_level == 'high']
        moderate_warnings = [w for w in self.warnings if w.warning_level == 'moderate']
        low_warnings = [w for w in self.warnings if w.warning_level == 'low']

        return {
            'total_warnings': len(self.warnings),
            'high': len(high_warnings),
            'moderate': len(moderate_warnings),
            'low': len(low_warnings),
            'highest_correlation': max(w.correlation_value for w in self.warnings) if self.warnings else 0,
            'warnings': self.warnings
        }

    def should_warn_user(self, correlation_value: float) -> bool:
        """Determine if correlation warrants user warning"""
        return correlation_value >= self.MODERATE_CORRELATION

    def format_warning_display(self, warning: CorrelationWarning) -> str:
        """Format warning for display"""
        return f"{warning.visual_indicator} {warning.message}"