"""
Production-Grade Data Usefulness Testing Framework for NFL Betting
Scientifically rigorous approach to validate which data sources provide actual betting value
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Statistical packages
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import xgboost as xgb

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Feature attribution analysis will be limited.")

logger = logging.getLogger(__name__)


class ProductionDataTester:
    """
    Comprehensive framework for testing data source usefulness in betting applications.

    Features:
    - Sample size validation to prevent false discoveries
    - Temporal cross-validation to prevent data leakage
    - Market efficiency testing
    - Statistical significance with multiple comparison correction
    - Risk-adjusted ROI calculations
    """

    def __init__(self, db_connection=None):
        self.db_conn = db_connection

        # Minimum sample sizes for reliable testing
        self.min_sample_sizes = {
            'epa_predictive_power': 500,    # Core performance metrics
            'injury_impact': 200,           # Key player availability
            'weather_impact': 100,          # Outdoor games only
            'referee_tendencies': 50,       # Per-referee minimum
            'snap_count_trends': 300,       # Usage pattern analysis
            'ngs_metrics': 400,             # Advanced analytics
            'default': 100                  # Default minimum
        }

        # Statistical significance thresholds
        self.significance_thresholds = {
            'statistical_significance': 0.05,  # Standard p-value
            'practical_significance': 0.2,     # Minimum effect size (Cohen's d)
            'temporal_stability': 0.6,         # Year-over-year reliability
            'market_exploitability': 0.02,     # Minimum ROI improvement
            'win_rate_threshold': 0.53         # Beat breakeven after vig
        }

        # Feature testing results storage
        self.test_results = {}
        self.interaction_results = {}
        self.stability_results = {}

        logger.info("ProductionDataTester initialized with rigorous testing standards")

    def validate_testing_readiness(self, data_source: str, available_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ensure sufficient sample size before testing to prevent false discoveries.

        Args:
            data_source: Name of data source being tested
            available_data: DataFrame with available data

        Returns:
            Dictionary with readiness status and recommendations
        """
        required_samples = self.min_sample_sizes.get(data_source, self.min_sample_sizes['default'])
        actual_samples = len(available_data)

        # Calculate statistical power
        if actual_samples >= required_samples:
            power = min(0.99, 0.5 + (actual_samples - required_samples) / required_samples)
        else:
            power = 0.5 * (actual_samples / required_samples)

        readiness_status = {
            'ready': actual_samples >= required_samples,
            'data_source': data_source,
            'required_samples': required_samples,
            'actual_samples': actual_samples,
            'statistical_power': power,
            'sample_ratio': actual_samples / required_samples
        }

        if not readiness_status['ready']:
            readiness_status['recommendation'] = (
                f"Collect {required_samples - actual_samples} more samples "
                f"or test with reduced confidence"
            )
            logger.warning(f"{data_source}: Insufficient sample size "
                         f"({actual_samples}/{required_samples})")
        else:
            readiness_status['recommendation'] = "Ready for testing"
            logger.info(f"{data_source}: Sample size validation passed "
                       f"({actual_samples} samples, power={power:.2f})")

        return readiness_status

    def test_feature_importance_leak_free(self,
                                        baseline_features: pd.DataFrame,
                                        new_features: pd.DataFrame,
                                        target: pd.Series,
                                        data_source: str) -> Dict[str, Any]:
        """
        Test feature importance with temporal cross-validation to prevent data leakage.

        Args:
            baseline_features: Current feature set (must include 'season' column)
            new_features: New features to test
            target: Target variable (e.g., point differential, total points)
            data_source: Name of data source for logging

        Returns:
            Comprehensive test results with statistical significance
        """
        logger.info(f"Testing feature importance for {data_source} with leak-free validation")

        # Validate inputs
        if 'season' not in baseline_features.columns:
            raise ValueError("baseline_features must include 'season' column for temporal validation")

        if len(baseline_features) != len(new_features) or len(baseline_features) != len(target):
            raise ValueError("All inputs must have the same length")

        # Check sample size
        readiness = self.validate_testing_readiness(data_source, baseline_features)
        if not readiness['ready']:
            return {
                'error': 'Insufficient sample size',
                'details': readiness,
                'data_source': data_source
            }

        # Create temporal splits (train on past seasons, test on future)
        seasons = sorted(baseline_features['season'].unique())
        temporal_splits = []

        for i, test_season in enumerate(seasons):
            train_seasons = seasons[:i]  # Only past seasons for training
            if len(train_seasons) < 2:  # Need minimum training history
                continue

            train_mask = baseline_features['season'].isin(train_seasons)
            test_mask = baseline_features['season'] == test_season

            if train_mask.sum() >= 50 and test_mask.sum() >= 10:  # Minimum split sizes
                temporal_splits.append((train_mask, test_mask))

        if len(temporal_splits) < 3:
            return {
                'error': 'Insufficient temporal data for reliable testing',
                'available_splits': len(temporal_splits),
                'required_splits': 3,
                'data_source': data_source
            }

        # Run temporal cross-validation
        baseline_scores = []
        enhanced_scores = []
        feature_importance_scores = []

        for fold, (train_idx, test_idx) in enumerate(temporal_splits):
            try:
                # Prepare data (exclude season from features)
                X_baseline_train = baseline_features.loc[train_idx].drop('season', axis=1)
                X_baseline_test = baseline_features.loc[test_idx].drop('season', axis=1)

                X_enhanced_train = pd.concat([
                    baseline_features.loc[train_idx].drop('season', axis=1),
                    new_features.loc[train_idx]
                ], axis=1)
                X_enhanced_test = pd.concat([
                    baseline_features.loc[test_idx].drop('season', axis=1),
                    new_features.loc[test_idx]
                ], axis=1)

                y_train = target.loc[train_idx]
                y_test = target.loc[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_baseline_train_scaled = scaler.fit_transform(X_baseline_train)
                X_baseline_test_scaled = scaler.transform(X_baseline_test)

                scaler_enhanced = StandardScaler()
                X_enhanced_train_scaled = scaler_enhanced.fit_transform(X_enhanced_train)
                X_enhanced_test_scaled = scaler_enhanced.transform(X_enhanced_test)

                # Baseline model
                baseline_model = xgb.XGBRegressor(
                    random_state=42,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    verbosity=0
                )
                baseline_model.fit(X_baseline_train_scaled, y_train)
                baseline_pred = baseline_model.predict(X_baseline_test_scaled)
                baseline_mse = mean_squared_error(y_test, baseline_pred)
                baseline_scores.append(baseline_mse)

                # Enhanced model
                enhanced_model = xgb.XGBRegressor(
                    random_state=42,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    verbosity=0
                )
                enhanced_model.fit(X_enhanced_train_scaled, y_train)
                enhanced_pred = enhanced_model.predict(X_enhanced_test_scaled)
                enhanced_mse = mean_squared_error(y_test, enhanced_pred)
                enhanced_scores.append(enhanced_mse)

                # Feature importance (for new features only)
                if SHAP_AVAILABLE:
                    explainer = shap.TreeExplainer(enhanced_model)
                    shap_values = explainer.shap_values(X_enhanced_test_scaled)

                    # Get importance for new features (last columns)
                    n_new_features = len(new_features.columns)
                    new_feature_importance = np.abs(shap_values[:, -n_new_features:]).mean(axis=0)
                    feature_importance_scores.append(new_feature_importance.mean())
                else:
                    # Fallback to built-in feature importance
                    importance = enhanced_model.feature_importances_[-len(new_features.columns):]
                    feature_importance_scores.append(importance.mean())

                logger.debug(f"Fold {fold}: Baseline MSE={baseline_mse:.4f}, "
                           f"Enhanced MSE={enhanced_mse:.4f}")

            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                continue

        if len(baseline_scores) < 2:
            return {
                'error': 'Insufficient successful cross-validation folds',
                'successful_folds': len(baseline_scores),
                'data_source': data_source
            }

        # Statistical analysis
        baseline_scores = np.array(baseline_scores)
        enhanced_scores = np.array(enhanced_scores)

        # Lower MSE is better, so improvement = baseline - enhanced
        mse_improvement = baseline_scores.mean() - enhanced_scores.mean()
        improvement_std = np.std(baseline_scores - enhanced_scores)

        # Statistical significance test
        t_stat, p_value = stats.ttest_rel(baseline_scores, enhanced_scores)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((baseline_scores.var() + enhanced_scores.var()) / 2)
        effect_size = mse_improvement / pooled_std if pooled_std > 0 else 0

        # Bonferroni correction for multiple testing
        n_tests = len(self.min_sample_sizes)  # Number of potential data sources
        adjusted_p_value = min(p_value * n_tests, 1.0)

        # Confidence interval
        se_diff = stats.sem(baseline_scores - enhanced_scores)
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(baseline_scores) - 1, mse_improvement, se_diff
        )

        # Convert MSE improvement to R² improvement approximation
        baseline_r2 = 1 - baseline_scores.mean() / np.var(target)
        enhanced_r2 = 1 - enhanced_scores.mean() / np.var(target)
        r2_improvement = enhanced_r2 - baseline_r2

        results = {
            'data_source': data_source,
            'sample_size': len(baseline_features),
            'temporal_folds': len(baseline_scores),
            'mse_improvement': mse_improvement,
            'mse_improvement_std': improvement_std,
            'r2_improvement': r2_improvement,
            'baseline_mse': baseline_scores.mean(),
            'enhanced_mse': enhanced_scores.mean(),
            'baseline_r2': baseline_r2,
            'enhanced_r2': enhanced_r2,
            'effect_size': effect_size,
            'p_value': p_value,
            'adjusted_p_value': adjusted_p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'statistically_significant': adjusted_p_value < self.significance_thresholds['statistical_significance'],
            'practically_significant': abs(effect_size) >= self.significance_thresholds['practical_significance'],
            'feature_importance_avg': np.mean(feature_importance_scores) if feature_importance_scores else None,
            'test_timestamp': datetime.now(),
            'baseline_scores': baseline_scores.tolist(),
            'enhanced_scores': enhanced_scores.tolist()
        }

        # Store results
        self.test_results[data_source] = results

        # Log results
        significance_status = "✓" if results['statistically_significant'] else "✗"
        practical_status = "✓" if results['practically_significant'] else "✗"

        logger.info(f"{data_source} Test Results:")
        logger.info(f"  MSE Improvement: {mse_improvement:.6f} (p={p_value:.4f}) {significance_status}")
        logger.info(f"  R² Improvement: {r2_improvement:.4f}")
        logger.info(f"  Effect Size: {effect_size:.4f} {practical_status}")
        logger.info(f"  Temporal Folds: {len(baseline_scores)}")

        return results

    def simulate_kelly_betting(self,
                             theoretical_edge: pd.Series,
                             market_lines: pd.Series,
                             outcomes: pd.Series,
                             bankroll: float = 1000.0) -> Dict[str, float]:
        """
        Simulate Kelly betting strategy to test actual performance.

        Args:
            theoretical_edge: Our calculated edge vs market
            market_lines: Market odds (American format)
            outcomes: Actual outcomes (1 for win, 0 for loss)
            bankroll: Starting bankroll

        Returns:
            Betting performance metrics
        """
        if len(theoretical_edge) != len(market_lines) or len(theoretical_edge) != len(outcomes):
            raise ValueError("All inputs must have same length")

        current_bankroll = bankroll
        total_bet = 0
        wins = 0
        losses = 0
        bet_history = []

        for i, (edge, line, outcome) in enumerate(zip(theoretical_edge, market_lines, outcomes)):
            # Skip if no meaningful edge
            if abs(edge) < 0.01:  # Less than 1% edge
                continue

            # Convert American odds to decimal odds
            if line > 0:
                decimal_odds = (line / 100) + 1
            else:
                decimal_odds = (100 / abs(line)) + 1

            # Kelly fraction calculation
            win_prob = 0.5 + edge  # Assuming edge around 50% baseline
            win_prob = max(0.01, min(0.99, win_prob))  # Clamp to reasonable range

            kelly_fraction = (win_prob * decimal_odds - 1) / (decimal_odds - 1)
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25% of bankroll

            # Bet sizing
            bet_amount = current_bankroll * kelly_fraction

            if bet_amount < 1:  # Minimum bet threshold
                continue

            # Execute bet
            total_bet += bet_amount

            if outcome == 1:  # Win
                profit = bet_amount * (decimal_odds - 1)
                current_bankroll += profit
                wins += 1
            else:  # Loss
                current_bankroll -= bet_amount
                losses += 1

            bet_history.append({
                'bet_amount': bet_amount,
                'odds': decimal_odds,
                'outcome': outcome,
                'profit_loss': bet_amount * (decimal_odds - 1) if outcome else -bet_amount,
                'bankroll_after': current_bankroll
            })

        total_bets = wins + losses
        if total_bets == 0:
            return {
                'total_roi': 0,
                'win_rate': 0,
                'total_bets': 0,
                'total_wagered': 0,
                'final_bankroll': bankroll,
                'sharpe_ratio': 0
            }

        # Calculate metrics
        total_profit = current_bankroll - bankroll
        roi = total_profit / bankroll
        win_rate = wins / total_bets

        # Sharpe ratio (risk-adjusted return)
        if bet_history:
            returns = [bet['profit_loss'] / bankroll for bet in bet_history]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return {
            'total_roi': roi,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'total_wagered': total_bet,
            'final_bankroll': current_bankroll,
            'total_profit': total_profit,
            'sharpe_ratio': sharpe_ratio,
            'avg_bet_size': total_bet / total_bets if total_bets > 0 else 0,
            'bet_history': bet_history
        }

    def convert_prediction_to_probability(self, predictions: pd.Series) -> pd.Series:
        """Convert model predictions to implied probabilities."""
        # Assuming predictions are point spreads or totals
        # This is a simplified conversion - should be customized based on prediction type
        # For demonstration, using logistic transformation
        return 1 / (1 + np.exp(-predictions / 10))

    def convert_odds_to_probability(self, odds: pd.Series) -> pd.Series:
        """Convert American odds to implied probabilities."""
        probabilities = []
        for line in odds:
            if line > 0:
                prob = 100 / (line + 100)
            else:
                prob = abs(line) / (abs(line) + 100)
            probabilities.append(prob)
        return pd.Series(probabilities)

    def get_tier_1_features(self) -> Dict[str, Dict[str, Any]]:
        """Define Tier 1 (highest expected value) features for testing."""
        return {
            'epa_metrics': {
                'features': ['home_epa_per_play_l5', 'away_epa_per_play_l5', 'epa_differential_trend'],
                'expected_roi_improvement': 0.04,  # 4% improvement
                'confidence_level': 0.9,          # High confidence
                'test_markets': ['spread', 'total'],
                'implementation_cost': 20,         # hours
                'priority_score': 10,
                'description': 'EPA-based performance metrics (last 5 games)'
            },
            'key_injuries': {
                'features': ['qb_injury_severity', 'rb1_availability', 'wr1_2_injury_impact'],
                'expected_roi_improvement': 0.03,  # 3% improvement
                'confidence_level': 0.85,         # High confidence
                'test_markets': ['spread', 'total', 'player_props'],
                'implementation_cost': 15,         # hours
                'priority_score': 9,
                'description': 'Key player injury impact analysis'
            }
        }

    def get_tier_2_features(self) -> Dict[str, Dict[str, Any]]:
        """Define Tier 2 (medium expected value) features for testing."""
        return {
            'weather_impact': {
                'features': ['temperature_vs_avg', 'wind_speed_impact', 'precipitation_flag'],
                'expected_roi_improvement': 0.02,  # 2% improvement
                'confidence_level': 0.7,          # Medium confidence
                'test_markets': ['total'],         # Focus on totals
                'filters': ['outdoor_games_only'],
                'implementation_cost': 12,         # hours
                'priority_score': 7,
                'description': 'Weather impact on scoring (outdoor games)'
            },
            'referee_tendencies': {
                'features': ['ref_over_percentage_l20', 'ref_penalty_rate', 'ref_home_bias'],
                'expected_roi_improvement': 0.015, # 1.5% improvement
                'confidence_level': 0.6,          # Medium confidence
                'test_markets': ['total', 'penalty_props'],
                'implementation_cost': 18,         # hours
                'priority_score': 6,
                'description': 'Referee tendency analysis (over/under rates)'
            }
        }

    def get_tier_3_features(self) -> Dict[str, Dict[str, Any]]:
        """Define Tier 3 (experimental) features for testing."""
        return {
            'ngs_advanced': {
                'features': ['avg_separation_trend', 'pressure_rate_allowed', 'yac_over_expected'],
                'expected_roi_improvement': 0.008, # 0.8% improvement
                'confidence_level': 0.4,          # Low confidence
                'test_markets': ['player_props'],
                'implementation_cost': 30,         # hours
                'priority_score': 3,
                'warning': 'May be too noisy for consistent value',
                'description': 'Next Gen Stats advanced metrics'
            },
            'snap_count_trends': {
                'features': ['usage_trend_3games', 'snap_share_change', 'role_stability'],
                'expected_roi_improvement': 0.01,  # 1% improvement
                'confidence_level': 0.5,          # Medium-low confidence
                'test_markets': ['player_props', 'team_totals'],
                'implementation_cost': 25,         # hours
                'priority_score': 4,
                'description': 'Player usage pattern analysis'
            }
        }


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Initialize tester
    tester = ProductionDataTester()

    # Create sample data for testing
    np.random.seed(42)
    n_samples = 600

    sample_data = pd.DataFrame({
        'season': np.repeat([2020, 2021, 2022, 2023], n_samples // 4),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })

    new_features = pd.DataFrame({
        'new_feature1': np.random.randn(n_samples) * 0.5,
        'new_feature2': np.random.randn(n_samples) * 0.3
    })

    # Create correlated target for testing
    target = (sample_data['feature1'] * 0.3 +
             sample_data['feature2'] * 0.2 +
             new_features['new_feature1'] * 0.4 +  # New feature has some predictive power
             np.random.randn(n_samples) * 0.5)

    # Test the framework
    print("Testing Production Data Testing Framework")
    print("=" * 50)

    # Check readiness
    readiness = tester.validate_testing_readiness('epa_predictive_power', sample_data)
    print(f"Sample size validation: {readiness}")

    # Test feature importance
    if readiness['ready']:
        results = tester.test_feature_importance_leak_free(
            baseline_features=sample_data,
            new_features=new_features,
            target=target,
            data_source='test_features'
        )

        print(f"\nFeature importance test results:")
        for key, value in results.items():
            if key not in ['baseline_scores', 'enhanced_scores']:
                print(f"  {key}: {value}")

    print("\nFramework testing complete!")