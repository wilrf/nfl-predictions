"""
Phase 2: Market-Aware Validation
Test if markets already price in your information and discover interaction effects
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

class MarketEfficiencyTester:
    """
    Market-aware validation framework that determines if features provide
    exploitable betting value beyond what markets already price in
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_roi_threshold = 0.02  # Minimum 2% ROI for exploitability
        self.min_win_rate = 0.53      # Beat breakeven after vig
        self.efficiency_threshold = 0.8  # Market efficiency cutoff

    def convert_prediction_to_probability(self, predictions: np.ndarray,
                                       prediction_type: str = 'spread') -> np.ndarray:
        """Convert model predictions to implied probabilities"""
        if prediction_type == 'spread':
            # Convert spread predictions to win probabilities using sigmoid
            return 1 / (1 + np.exp(-predictions * 0.1))  # Scaling factor for spread
        elif prediction_type == 'total':
            # For totals, convert to over/under probabilities
            # Assuming predictions are margin from line
            return 1 / (1 + np.exp(-predictions * 0.15))
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

    def convert_odds_to_probability(self, odds: np.ndarray, odds_format: str = 'american') -> np.ndarray:
        """Convert betting odds to implied probabilities"""
        if odds_format == 'american':
            probabilities = np.where(
                odds > 0,
                100 / (odds + 100),
                -odds / (-odds + 100)
            )
        elif odds_format == 'decimal':
            probabilities = 1 / odds
        else:
            raise ValueError(f"Unknown odds format: {odds_format}")

        return probabilities

    def simulate_kelly_betting(self, edge: np.ndarray, odds: np.ndarray,
                             outcomes: np.ndarray, max_bet_size: float = 0.05) -> Dict[str, float]:
        """Simulate Kelly criterion betting with risk management"""
        bankroll = 1.0
        total_bets = 0
        winning_bets = 0

        for i in range(len(edge)):
            if abs(edge[i]) < 0.01:  # Skip bets with minimal edge
                continue

            # Kelly fraction calculation
            win_prob = 0.5 + edge[i]  # Adjust probability based on edge
            if odds[i] > 0:  # American odds
                decimal_odds = (odds[i] / 100) + 1
            else:
                decimal_odds = (100 / abs(odds[i])) + 1

            kelly_fraction = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
            kelly_fraction = max(0, min(kelly_fraction, max_bet_size))  # Risk management

            if kelly_fraction > 0.01:  # Only bet if Kelly > 1%
                bet_size = kelly_fraction * bankroll
                total_bets += 1

                if outcomes[i] == 1:  # Winning bet
                    bankroll += bet_size * (decimal_odds - 1)
                    winning_bets += 1
                else:  # Losing bet
                    bankroll -= bet_size

        if total_bets == 0:
            return {'total_roi': 0, 'win_rate': 0, 'total_bets': 0, 'final_bankroll': 1.0}

        return {
            'total_roi': bankroll - 1.0,
            'win_rate': winning_bets / total_bets,
            'total_bets': total_bets,
            'final_bankroll': bankroll
        }

    def calculate_confidence_level(self, betting_results: Dict[str, float]) -> float:
        """Calculate confidence level based on betting performance"""
        total_bets = betting_results['total_bets']
        win_rate = betting_results['win_rate']

        if total_bets < 30:
            return 0.3  # Low confidence due to small sample
        elif total_bets < 100:
            base_confidence = 0.6
        else:
            base_confidence = 0.8

        # Adjust based on win rate
        if win_rate > 0.6:
            return min(base_confidence + 0.2, 1.0)
        elif win_rate > 0.55:
            return base_confidence
        else:
            return max(base_confidence - 0.2, 0.1)

    def test_market_efficiency(self, predictions: np.ndarray, market_lines: np.ndarray,
                             outcomes: np.ndarray, data_source: str,
                             prediction_type: str = 'spread') -> Dict[str, Any]:
        """Test if market already incorporates this information"""

        # Convert predictions and market lines to implied probabilities
        our_implied_prob = self.convert_prediction_to_probability(predictions, prediction_type)
        market_implied_prob = self.convert_odds_to_probability(market_lines)

        # Calculate our theoretical edge
        theoretical_edge = our_implied_prob - market_implied_prob

        # Test actual performance
        betting_results = self.simulate_kelly_betting(
            theoretical_edge, market_lines, outcomes
        )

        actual_roi = betting_results['total_roi']
        win_rate = betting_results['win_rate']

        # Market efficiency score (1.0 = perfectly efficient, 0.0 = inefficient)
        if abs(theoretical_edge.mean()) > 0:
            efficiency_score = 1 - (abs(actual_roi) / abs(theoretical_edge.mean()))
        else:
            efficiency_score = 1.0

        efficiency_score = max(0, min(efficiency_score, 1))  # Clamp to [0,1]

        # Determine exploitability
        exploitable = (
            actual_roi > self.min_roi_threshold and     # Minimum 2% ROI
            efficiency_score < self.efficiency_threshold and  # Market not fully efficient
            win_rate > self.min_win_rate and           # Beat breakeven after vig
            len(predictions) >= 50                     # Sufficient sample size
        )

        # Recommendation logic
        if exploitable:
            recommendation = 'implement'
        elif actual_roi > 0.01:
            recommendation = 'monitor'
        else:
            recommendation = 'skip'

        return {
            'data_source': data_source,
            'theoretical_edge_mean': theoretical_edge.mean(),
            'theoretical_edge_std': theoretical_edge.std(),
            'actual_roi': actual_roi,
            'win_rate': win_rate,
            'efficiency_score': efficiency_score,
            'exploitable': exploitable,
            'recommendation': recommendation,
            'confidence_level': self.calculate_confidence_level(betting_results),
            'sample_size': len(predictions),
            'total_bets_placed': betting_results['total_bets'],
            'sharpe_ratio': self.calculate_betting_sharpe_ratio(betting_results, theoretical_edge)
        }

    def calculate_betting_sharpe_ratio(self, betting_results: Dict[str, float],
                                     edge_history: np.ndarray) -> float:
        """Calculate Sharpe ratio for betting performance"""
        if betting_results['total_bets'] == 0:
            return 0.0

        # Estimate return volatility from edge variance
        edge_volatility = np.std(edge_history) if len(edge_history) > 1 else 0.1
        if edge_volatility == 0:
            return 0.0

        return betting_results['total_roi'] / edge_volatility

    def calculate_weather_severity_score(self, weather_data: pd.DataFrame) -> np.ndarray:
        """Calculate composite weather severity score"""
        severity_score = np.zeros(len(weather_data))

        if 'temperature' in weather_data.columns:
            # Extreme temperature penalty (below 32F or above 90F)
            temp_penalty = np.where(
                (weather_data['temperature'] < 32) | (weather_data['temperature'] > 90),
                1.0,
                0.0
            )
            severity_score += temp_penalty

        if 'wind_speed' in weather_data.columns:
            # Wind speed impact (linear above 10 mph)
            wind_impact = np.maximum(0, (weather_data['wind_speed'] - 10) / 20)
            severity_score += wind_impact

        if 'precipitation' in weather_data.columns:
            # Precipitation impact
            precip_impact = weather_data['precipitation'] / 10  # Normalize
            severity_score += precip_impact

        return np.minimum(severity_score, 3.0)  # Cap at 3.0

    def test_data_source_interactions(self, weather_data: pd.DataFrame,
                                    referee_data: pd.DataFrame,
                                    injury_data: pd.DataFrame,
                                    outcomes: pd.DataFrame) -> Dict[str, Any]:
        """Discover if combining data sources creates multiplicative value"""

        interaction_tests = {}

        # Weather + Referee Interaction
        if not weather_data.empty and not referee_data.empty:
            weather_severity = self.calculate_weather_severity_score(weather_data)
            referee_penalty_tendency = referee_data.get('avg_penalties_per_game', np.zeros(len(referee_data)))

            weather_ref_interaction = weather_severity * referee_penalty_tendency

            # Test interaction effect on total penalties
            if 'total_penalties' in outcomes.columns:
                baseline_features = pd.concat([
                    weather_data[['temperature', 'wind_speed']] if 'temperature' in weather_data.columns else pd.DataFrame(),
                    referee_data[['avg_penalties_per_game']] if 'avg_penalties_per_game' in referee_data.columns else pd.DataFrame()
                ], axis=1)

                if not baseline_features.empty:
                    # Add season column for temporal validation
                    baseline_features['season'] = outcomes.get('season', 2023)

                    new_features = pd.DataFrame({'weather_ref_interaction': weather_ref_interaction})

                    from .production_data_tester import ProductionDataTester
                    tester = ProductionDataTester()

                    interaction_tests['weather_referee'] = tester.test_feature_importance_leak_free(
                        baseline_features=baseline_features,
                        new_features=new_features,
                        target=outcomes['total_penalties']
                    )

        # Injury + Weather Interaction
        if not injury_data.empty and not weather_data.empty:
            key_injury_count = injury_data.get('key_players_out', np.zeros(len(injury_data)))
            weather_severity = self.calculate_weather_severity_score(weather_data)

            injury_weather_interaction = key_injury_count * weather_severity

            # Test interaction effect on total score variance
            if 'total_score_variance' in outcomes.columns:
                baseline_features = pd.concat([
                    injury_data[['key_players_out']] if 'key_players_out' in injury_data.columns else pd.DataFrame(),
                    weather_data[['temperature', 'wind_speed']] if 'temperature' in weather_data.columns else pd.DataFrame()
                ], axis=1)

                if not baseline_features.empty:
                    baseline_features['season'] = outcomes.get('season', 2023)

                    new_features = pd.DataFrame({'injury_weather_interaction': injury_weather_interaction})

                    from .production_data_tester import ProductionDataTester
                    tester = ProductionDataTester()

                    interaction_tests['injury_weather'] = tester.test_feature_importance_leak_free(
                        baseline_features=baseline_features,
                        new_features=new_features,
                        target=outcomes['total_score_variance']
                    )

        # Team Performance + Injury Interaction
        if not injury_data.empty and 'team_epa' in outcomes.columns:
            team_performance = outcomes['team_epa']
            key_injury_count = injury_data.get('key_players_out', np.zeros(len(injury_data)))

            # Hypothesis: Injuries have more impact on high-performing teams
            performance_injury_interaction = team_performance * key_injury_count

            baseline_features = pd.DataFrame({
                'team_epa': team_performance,
                'key_players_out': key_injury_count,
                'season': outcomes.get('season', 2023)
            })

            new_features = pd.DataFrame({'performance_injury_interaction': performance_injury_interaction})

            if 'point_differential' in outcomes.columns:
                from .production_data_tester import ProductionDataTester
                tester = ProductionDataTester()

                interaction_tests['performance_injury'] = tester.test_feature_importance_leak_free(
                    baseline_features=baseline_features,
                    new_features=new_features,
                    target=outcomes['point_differential']
                )

        return interaction_tests

    def run_comprehensive_market_testing(self, data_source: str, predictions: np.ndarray,
                                       market_lines: np.ndarray, outcomes: np.ndarray,
                                       weather_data: pd.DataFrame = None,
                                       referee_data: pd.DataFrame = None,
                                       injury_data: pd.DataFrame = None,
                                       outcome_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run complete Phase 2 market validation pipeline"""

        results = {
            'data_source': data_source,
            'validation_timestamp': pd.Timestamp.now(),
            'phase': 'Phase 2: Market-Aware Validation'
        }

        # Step 1: Market efficiency testing
        efficiency_results = self.test_market_efficiency(
            predictions, market_lines, outcomes, data_source
        )
        results['market_efficiency'] = efficiency_results

        # Step 2: Interaction effect discovery (if data provided)
        if all(data is not None for data in [weather_data, referee_data, injury_data, outcome_data]):
            interaction_results = self.test_data_source_interactions(
                weather_data, referee_data, injury_data, outcome_data
            )
            results['interaction_effects'] = interaction_results
        else:
            results['interaction_effects'] = {'note': 'Insufficient data for interaction testing'}

        # Step 3: Generate overall recommendation
        if efficiency_results['exploitable']:
            if efficiency_results['confidence_level'] > 0.7:
                results['recommendation'] = 'Proceed to Phase 3: High confidence exploitable edge detected'
            else:
                results['recommendation'] = 'Proceed to Phase 3: Exploitable but monitor sample size'
        elif efficiency_results['actual_roi'] > 0.01:
            results['recommendation'] = 'Monitor closely: Marginal edge detected'
        else:
            results['recommendation'] = 'Skip implementation: No exploitable edge detected'

        return results