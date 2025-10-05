"""
Market Efficiency Testing Module
Tests whether betting markets already price in the information from new data sources
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

from scipy import stats
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class MarketEfficiencyAnalyzer:
    """
    Analyzes market efficiency to determine if new data sources provide exploitable edges.

    Key Concepts:
    - If market is efficient, our edge should translate to actual profit
    - If market already prices in our information, we won't beat it consistently
    - Efficiency score helps determine if pursuing a data source is worthwhile
    """

    def __init__(self):
        self.efficiency_thresholds = {
            'highly_efficient': 0.9,      # Market captures 90%+ of information
            'moderately_efficient': 0.7,  # Market captures 70-90% of information
            'inefficient': 0.5,           # Market captures <70% of information
            'exploitable_roi': 0.02,      # Minimum 2% ROI to be worth pursuing
            'min_sample_size': 50         # Minimum bets for reliable analysis
        }

        self.test_results = {}

    def test_market_efficiency(self,
                             our_predictions: pd.Series,
                             market_lines: pd.Series,
                             actual_outcomes: pd.Series,
                             data_source: str,
                             bet_type: str = 'spread') -> Dict[str, Any]:
        """
        Test if market already prices in our information advantage.

        Args:
            our_predictions: Our model predictions
            market_lines: Market lines/odds
            actual_outcomes: Actual game results
            data_source: Name of data source being tested
            bet_type: Type of bet ('spread', 'total', 'moneyline')

        Returns:
            Comprehensive market efficiency analysis
        """
        logger.info(f"Testing market efficiency for {data_source} on {bet_type} bets")

        # Validate inputs
        if len(our_predictions) != len(market_lines) or len(our_predictions) != len(actual_outcomes):
            raise ValueError("All inputs must have same length")

        if len(our_predictions) < self.efficiency_thresholds['min_sample_size']:
            return {
                'error': 'Insufficient sample size for reliable efficiency testing',
                'sample_size': len(our_predictions),
                'min_required': self.efficiency_thresholds['min_sample_size']
            }

        # Convert predictions and market lines to probabilities
        our_probabilities = self._convert_predictions_to_probabilities(our_predictions, bet_type)
        market_probabilities = self._convert_lines_to_probabilities(market_lines, bet_type)

        # Calculate theoretical edge
        theoretical_edge = our_probabilities - market_probabilities
        edge_magnitude = np.abs(theoretical_edge).mean()

        # Simulate betting performance
        betting_results = self._simulate_betting_performance(
            our_probabilities, market_lines, actual_outcomes, bet_type
        )

        # Calculate market efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            our_probabilities, market_probabilities, actual_outcomes
        )

        # Determine market efficiency level
        efficiency_score = efficiency_metrics['efficiency_score']

        if efficiency_score >= self.efficiency_thresholds['highly_efficient']:
            efficiency_level = 'highly_efficient'
            recommendation = 'skip'
        elif efficiency_score >= self.efficiency_thresholds['moderately_efficient']:
            efficiency_level = 'moderately_efficient'
            recommendation = 'monitor' if betting_results['roi'] < 0.01 else 'consider'
        else:
            efficiency_level = 'inefficient'
            recommendation = 'implement' if betting_results['roi'] >= self.efficiency_thresholds['exploitable_roi'] else 'monitor'

        # Check for exploitability
        exploitable = (
            betting_results['roi'] >= self.efficiency_thresholds['exploitable_roi'] and
            betting_results['win_rate'] > 0.53 and  # Beat breakeven after vig
            efficiency_score < 0.8 and  # Market not too efficient
            betting_results['confidence_interval'][0] > 0  # Lower bound of ROI is positive
        )

        results = {
            'data_source': data_source,
            'bet_type': bet_type,
            'sample_size': len(our_predictions),

            # Edge Analysis
            'theoretical_edge_mean': theoretical_edge.mean(),
            'theoretical_edge_std': theoretical_edge.std(),
            'edge_magnitude': edge_magnitude,

            # Betting Performance
            'actual_roi': betting_results['roi'],
            'win_rate': betting_results['win_rate'],
            'total_bets': betting_results['total_bets'],
            'sharpe_ratio': betting_results['sharpe_ratio'],
            'confidence_interval': betting_results['confidence_interval'],

            # Market Efficiency
            'efficiency_score': efficiency_score,
            'efficiency_level': efficiency_level,
            'our_calibration_error': efficiency_metrics['our_calibration_error'],
            'market_calibration_error': efficiency_metrics['market_calibration_error'],
            'information_coefficient': efficiency_metrics['information_coefficient'],

            # Decision Framework
            'exploitable': exploitable,
            'recommendation': recommendation,
            'confidence_level': self._calculate_confidence_level(betting_results, efficiency_metrics),

            # Risk Metrics
            'max_drawdown': betting_results.get('max_drawdown', 0),
            'volatility': betting_results.get('volatility', 0),

            'test_timestamp': datetime.now()
        }

        # Store results
        self.test_results[f"{data_source}_{bet_type}"] = results

        # Log results
        logger.info(f"{data_source} Market Efficiency Results:")
        logger.info(f"  Efficiency Level: {efficiency_level} (score: {efficiency_score:.3f})")
        logger.info(f"  Actual ROI: {betting_results['roi']:.3f}")
        logger.info(f"  Win Rate: {betting_results['win_rate']:.3f}")
        logger.info(f"  Recommendation: {recommendation}")
        logger.info(f"  Exploitable: {'Yes' if exploitable else 'No'}")

        return results

    def _convert_predictions_to_probabilities(self, predictions: pd.Series, bet_type: str) -> pd.Series:
        """Convert model predictions to implied probabilities based on bet type."""
        if bet_type == 'spread':
            # For spread bets, convert point differential to win probability
            # Using logistic transformation: P(win) = 1 / (1 + exp(-prediction/sigma))
            sigma = 14  # Typical NFL point spread standard deviation
            return 1 / (1 + np.exp(-predictions / sigma))

        elif bet_type == 'total':
            # For totals, compare prediction to line
            # This assumes predictions is actual total prediction minus line
            sigma = 10  # Typical total points standard deviation
            return 1 / (1 + np.exp(-predictions / sigma))

        elif bet_type == 'moneyline':
            # If predictions are already probabilities
            if predictions.min() >= 0 and predictions.max() <= 1:
                return predictions
            else:
                # Convert using logistic transformation
                return 1 / (1 + np.exp(-predictions / 10))
        else:
            raise ValueError(f"Unsupported bet type: {bet_type}")

    def _convert_lines_to_probabilities(self, lines: pd.Series, bet_type: str) -> pd.Series:
        """Convert betting lines to implied probabilities."""
        if bet_type in ['spread', 'total']:
            # For spread/total, assume -110 vig on both sides
            return pd.Series([0.5238] * len(lines))  # 110/210 = 0.5238

        elif bet_type == 'moneyline':
            # Convert American odds to probabilities
            probabilities = []
            for line in lines:
                if line > 0:
                    prob = 100 / (line + 100)
                else:
                    prob = abs(line) / (abs(line) + 100)
                probabilities.append(prob)
            return pd.Series(probabilities)
        else:
            raise ValueError(f"Unsupported bet type: {bet_type}")

    def _simulate_betting_performance(self,
                                    our_probabilities: pd.Series,
                                    market_lines: pd.Series,
                                    outcomes: pd.Series,
                                    bet_type: str,
                                    bankroll: float = 1000.0) -> Dict[str, Any]:
        """Simulate actual betting performance using Kelly criterion."""

        current_bankroll = bankroll
        bet_history = []
        bankroll_history = [bankroll]

        for i, (our_prob, line, outcome) in enumerate(zip(our_probabilities, market_lines, outcomes)):
            # Calculate Kelly fraction
            if bet_type in ['spread', 'total']:
                # Standard -110 lines
                decimal_odds = 1.909  # -110 in decimal
                kelly_fraction = (our_prob * decimal_odds - 1) / (decimal_odds - 1)
            else:
                # Moneyline
                if line > 0:
                    decimal_odds = (line / 100) + 1
                else:
                    decimal_odds = (100 / abs(line)) + 1
                kelly_fraction = (our_prob * decimal_odds - 1) / (decimal_odds - 1)

            # Cap Kelly fraction at 25% of bankroll for safety
            kelly_fraction = max(0, min(0.25, kelly_fraction))

            # Skip if edge is too small
            if kelly_fraction < 0.01:  # Less than 1% of bankroll
                continue

            # Calculate bet amount
            bet_amount = current_bankroll * kelly_fraction

            if bet_amount < 1:  # Minimum bet threshold
                continue

            # Determine bet outcome
            if bet_type in ['spread', 'total']:
                # For spread/total, outcome is binary (cover/not cover)
                bet_won = bool(outcome)
                payout = bet_amount * 0.909 if bet_won else -bet_amount  # -110 odds
            else:
                # Moneyline
                bet_won = bool(outcome)
                if bet_won:
                    if line > 0:
                        payout = bet_amount * (line / 100)
                    else:
                        payout = bet_amount * (100 / abs(line))
                else:
                    payout = -bet_amount

            # Update bankroll
            current_bankroll += payout
            bankroll_history.append(current_bankroll)

            bet_history.append({
                'bet_amount': bet_amount,
                'outcome': bet_won,
                'payout': payout,
                'bankroll_after': current_bankroll,
                'kelly_fraction': kelly_fraction
            })

        if not bet_history:
            return {
                'roi': 0,
                'win_rate': 0,
                'total_bets': 0,
                'sharpe_ratio': 0,
                'confidence_interval': (0, 0),
                'max_drawdown': 0,
                'volatility': 0
            }

        # Calculate performance metrics
        total_bets = len(bet_history)
        wins = sum(1 for bet in bet_history if bet['outcome'])
        win_rate = wins / total_bets

        total_profit = current_bankroll - bankroll
        roi = total_profit / bankroll

        # Calculate Sharpe ratio
        returns = [bet['payout'] / bankroll for bet in bet_history]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        peak_bankroll = bankroll
        max_drawdown = 0
        for balance in bankroll_history:
            if balance > peak_bankroll:
                peak_bankroll = balance
            drawdown = (peak_bankroll - balance) / peak_bankroll
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate volatility
        bankroll_returns = np.diff(bankroll_history) / bankroll_history[:-1]
        volatility = np.std(bankroll_returns) * np.sqrt(252) if len(bankroll_returns) > 1 else 0

        # Calculate confidence interval for ROI
        roi_std = np.std(returns)
        roi_se = roi_std / np.sqrt(len(returns))
        ci_lower, ci_upper = stats.t.interval(0.95, len(returns) - 1, roi, roi_se)

        return {
            'roi': roi,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'total_profit': total_profit,
            'sharpe_ratio': sharpe_ratio,
            'confidence_interval': (ci_lower, ci_upper),
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_bankroll': current_bankroll,
            'bet_history': bet_history
        }

    def _calculate_efficiency_metrics(self,
                                    our_probabilities: pd.Series,
                                    market_probabilities: pd.Series,
                                    outcomes: pd.Series) -> Dict[str, float]:
        """Calculate various market efficiency metrics."""

        # Convert outcomes to binary if needed
        binary_outcomes = (outcomes > 0.5).astype(int)

        # Calibration analysis
        our_calibration_error = self._calculate_calibration_error(our_probabilities, binary_outcomes)
        market_calibration_error = self._calculate_calibration_error(market_probabilities, binary_outcomes)

        # Information coefficient (correlation between probabilities and outcomes)
        our_ic = stats.pearsonr(our_probabilities, binary_outcomes)[0]
        market_ic = stats.pearsonr(market_probabilities, binary_outcomes)[0]

        # Efficiency score: How much of our information is already in the market
        # High score = market already incorporates our information
        probability_correlation = stats.pearsonr(our_probabilities, market_probabilities)[0]

        # If our predictions are well-calibrated but market isn't, efficiency is low
        # If market is well-calibrated and correlates with our predictions, efficiency is high
        calibration_ratio = market_calibration_error / (our_calibration_error + 1e-6)

        efficiency_score = (
            probability_correlation * 0.4 +           # How similar are our predictions to market
            (1 - our_calibration_error) * 0.3 +       # How well-calibrated are we
            calibration_ratio * 0.3                   # Relative calibration quality
        )

        efficiency_score = max(0, min(1, efficiency_score))  # Clamp to [0,1]

        return {
            'efficiency_score': efficiency_score,
            'our_calibration_error': our_calibration_error,
            'market_calibration_error': market_calibration_error,
            'our_information_coefficient': our_ic,
            'market_information_coefficient': market_ic,
            'probability_correlation': probability_correlation,
            'information_coefficient': max(abs(our_ic), abs(market_ic))
        }

    def _calculate_calibration_error(self, probabilities: pd.Series, outcomes: pd.Series) -> float:
        """Calculate mean calibration error (reliability of probability estimates)."""
        try:
            # Use sklearn's calibration_curve for reliable calculation
            fraction_of_positives, mean_predicted_value = calibration_curve(
                outcomes, probabilities, n_bins=10, strategy='quantile'
            )

            # Calculate mean absolute calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            return calibration_error

        except Exception as e:
            logger.warning(f"Error calculating calibration: {e}")
            # Fallback: simple binning approach
            n_bins = min(10, len(probabilities) // 5)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            calibration_error = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = outcomes[in_bin].mean()
                    avg_confidence_in_bin = probabilities[in_bin].mean()
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return calibration_error

    def _calculate_confidence_level(self, betting_results: Dict, efficiency_metrics: Dict) -> str:
        """Determine confidence level in market efficiency assessment."""

        sample_size = betting_results['total_bets']
        roi_ci_width = betting_results['confidence_interval'][1] - betting_results['confidence_interval'][0]
        efficiency_score = efficiency_metrics['efficiency_score']

        # High confidence if:
        # - Large sample size
        # - Narrow confidence interval
        # - Clear efficiency score (very high or very low)

        confidence_factors = {
            'sample_size': min(1.0, sample_size / 100),  # Scale to [0,1]
            'precision': max(0, 1 - roi_ci_width / 0.1),  # Narrower CI = higher confidence
            'clarity': 1 - 2 * abs(efficiency_score - 0.5)  # Extreme scores are clearer
        }

        overall_confidence = np.mean(list(confidence_factors.values()))

        if overall_confidence >= 0.8:
            return 'high'
        elif overall_confidence >= 0.6:
            return 'medium'
        else:
            return 'low'

    def test_data_source_interactions(self,
                                    weather_predictions: pd.Series,
                                    referee_predictions: pd.Series,
                                    injury_predictions: pd.Series,
                                    market_lines: pd.Series,
                                    outcomes: pd.Series) -> Dict[str, Any]:
        """Test market efficiency for combined data sources."""

        logger.info("Testing market efficiency for data source interactions")

        # Create interaction features
        interactions = {
            'weather_referee': weather_predictions * referee_predictions,
            'weather_injury': weather_predictions * injury_predictions,
            'referee_injury': referee_predictions * injury_predictions,
            'three_way': weather_predictions * referee_predictions * injury_predictions
        }

        interaction_results = {}

        for interaction_name, interaction_values in interactions.items():
            # Test market efficiency for each interaction
            efficiency_result = self.test_market_efficiency(
                our_predictions=interaction_values,
                market_lines=market_lines,
                actual_outcomes=outcomes,
                data_source=f"interaction_{interaction_name}",
                bet_type='total'  # Most interactions likely affect totals
            )

            interaction_results[interaction_name] = efficiency_result

        # Find best interaction
        best_interaction = max(
            interaction_results.keys(),
            key=lambda x: interaction_results[x].get('actual_roi', 0)
        )

        return {
            'individual_interactions': interaction_results,
            'best_interaction': best_interaction,
            'best_interaction_roi': interaction_results[best_interaction].get('actual_roi', 0),
            'synergy_detected': any(
                result.get('exploitable', False) for result in interaction_results.values()
            )
        }

    def generate_efficiency_report(self, data_sources: List[str]) -> str:
        """Generate comprehensive efficiency report for tested data sources."""

        report = []
        report.append("=" * 60)
        report.append("MARKET EFFICIENCY ANALYSIS REPORT")
        report.append("=" * 60)

        exploitable_sources = []
        monitor_sources = []
        skip_sources = []

        for source in data_sources:
            if source in self.test_results:
                result = self.test_results[source]

                report.append(f"\n{source.upper()}")
                report.append("-" * 40)
                report.append(f"Efficiency Level: {result['efficiency_level']}")
                report.append(f"Efficiency Score: {result['efficiency_score']:.3f}")
                report.append(f"Actual ROI: {result['actual_roi']:.3f}")
                report.append(f"Win Rate: {result['win_rate']:.3f}")
                report.append(f"Sample Size: {result['sample_size']}")
                report.append(f"Recommendation: {result['recommendation']}")

                if result['recommendation'] == 'implement':
                    exploitable_sources.append(source)
                elif result['recommendation'] in ['monitor', 'consider']:
                    monitor_sources.append(source)
                else:
                    skip_sources.append(source)

        # Summary
        report.append("\n" + "=" * 60)
        report.append("SUMMARY RECOMMENDATIONS")
        report.append("=" * 60)

        if exploitable_sources:
            report.append(f"\n✅ IMPLEMENT IMMEDIATELY: {', '.join(exploitable_sources)}")

        if monitor_sources:
            report.append(f"\n⚠️  MONITOR/CONSIDER: {', '.join(monitor_sources)}")

        if skip_sources:
            report.append(f"\n❌ SKIP: {', '.join(skip_sources)}")

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Initialize analyzer
    analyzer = MarketEfficiencyAnalyzer()

    # Create sample data
    np.random.seed(42)
    n_games = 200

    # Simulate predictions and market lines
    true_outcomes = np.random.binomial(1, 0.5, n_games)

    # Our predictions (slightly better than random)
    our_predictions = true_outcomes + np.random.normal(0, 0.3, n_games)

    # Market lines (for spread bets, assume -110 on both sides)
    market_lines = np.full(n_games, -110)

    print("Testing Market Efficiency Analyzer")
    print("=" * 50)

    # Test market efficiency
    results = analyzer.test_market_efficiency(
        our_predictions=pd.Series(our_predictions),
        market_lines=pd.Series(market_lines),
        actual_outcomes=pd.Series(true_outcomes),
        data_source='test_source',
        bet_type='spread'
    )

    print("Market Efficiency Test Results:")
    for key, value in results.items():
        if key not in ['test_timestamp']:
            print(f"  {key}: {value}")

    # Generate report
    print("\n" + analyzer.generate_efficiency_report(['test_source']))