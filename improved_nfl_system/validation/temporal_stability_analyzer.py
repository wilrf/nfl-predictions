"""
Phase 3: Temporal Stability Analysis
Ensure findings are reliable across time periods and detect feature decay
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import mean_squared_error

class TemporalStabilityAnalyzer:
    """
    Validate feature reliability across multiple seasons and detect temporal patterns
    """

    def __init__(self, min_seasons: int = 3):
        self.min_seasons = min_seasons
        self.logger = logging.getLogger(__name__)
        self.reliability_thresholds = {
            'highly_reliable': 0.7,
            'moderately_reliable': 0.5,
            'monitor_closely': 0.3
        }

    def calculate_yoy_correlation(self, importance_values: pd.Series) -> float:
        """Calculate year-over-year correlation for consistency"""
        if len(importance_values) < 2:
            return 0.0

        return importance_values.autocorr(lag=1)

    def analyze_trend(self, importance_values: pd.Series) -> Tuple[float, float]:
        """Analyze trend in feature importance over time"""
        if len(importance_values) < 2:
            return 0.0, 1.0

        seasons = range(len(importance_values))
        try:
            trend_slope, intercept, r_value, p_value, std_err = stats.linregress(seasons, importance_values)
            return trend_slope, p_value
        except Exception:
            return 0.0, 1.0

    def calculate_coefficient_of_variation(self, importance_values: pd.Series) -> float:
        """Calculate coefficient of variation as stability measure"""
        if len(importance_values) < 2 or importance_values.mean() == 0:
            return 1.0  # High variation if no data or zero mean

        return importance_values.std() / abs(importance_values.mean())

    def calculate_decay_rate(self, importance_values: pd.Series) -> float:
        """Calculate feature decay rate (declining predictive power)"""
        if len(importance_values) < 4:
            return 0.0

        recent_performance = importance_values[-2:].mean()
        historical_performance = importance_values[:-2].mean()

        if historical_performance == 0:
            return 0.0

        decay_rate = (historical_performance - recent_performance) / historical_performance
        return decay_rate

    def calculate_reliability_score(self, yoy_correlation: float, trend_slope: float,
                                  coefficient_of_variation: float, decay_rate: float,
                                  mean_importance: float) -> float:
        """Calculate overall reliability score (0-1 scale)"""

        # Consistency score (how stable year-over-year)
        consistency_score = max(0, abs(yoy_correlation))

        # Magnitude score (how important is the feature)
        magnitude_score = min(1, mean_importance / 0.05)  # Normalize to 5% baseline

        # Stability score (low coefficient of variation is good)
        stability_score = max(0, 1 - coefficient_of_variation)

        # Decay penalty (feature not declining)
        decay_penalty = max(0, 1 - abs(decay_rate))

        # Weighted combination
        reliability_score = (
            consistency_score * 0.3 +     # Consistency weight
            magnitude_score * 0.3 +       # Magnitude weight
            stability_score * 0.2 +       # Stability weight
            decay_penalty * 0.2           # Decay penalty weight
        )

        return min(1.0, max(0.0, reliability_score))

    def classify_reliability(self, reliability_score: float) -> str:
        """Classify feature reliability based on score"""
        if reliability_score >= self.reliability_thresholds['highly_reliable']:
            return 'highly_reliable'
        elif reliability_score >= self.reliability_thresholds['moderately_reliable']:
            return 'moderately_reliable'
        elif reliability_score >= self.reliability_thresholds['monitor_closely']:
            return 'monitor_closely'
        else:
            return 'unreliable'

    def get_reliability_recommendation(self, classification: str, decay_rate: float) -> str:
        """Generate recommendation based on reliability classification"""
        if classification == 'highly_reliable':
            if decay_rate > 0.15:
                return 'implement_with_monitoring'
            else:
                return 'implement_immediately'
        elif classification == 'moderately_reliable':
            if decay_rate > 0.2:
                return 'monitor_before_implementing'
            else:
                return 'implement_with_caution'
        elif classification == 'monitor_closely':
            return 'requires_improvement_or_more_data'
        else:
            return 'do_not_implement'

    def test_temporal_stability(self, feature_importance_by_season: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature reliability across multiple seasons"""

        if len(feature_importance_by_season) < self.min_seasons:
            return {
                'error': f'Need at least {self.min_seasons} seasons for stability testing',
                'available_seasons': len(feature_importance_by_season),
                'required_seasons': self.min_seasons
            }

        stability_analysis = {}

        for feature in feature_importance_by_season.columns:
            importance_values = feature_importance_by_season[feature].dropna()

            if len(importance_values) < self.min_seasons:
                stability_analysis[feature] = {
                    'error': f'Insufficient data for feature {feature}',
                    'available_data_points': len(importance_values)
                }
                continue

            # Year-over-year correlation (consistency)
            yoy_correlation = self.calculate_yoy_correlation(importance_values)

            # Trend analysis (improving/declining/stable)
            trend_slope, trend_p_value = self.analyze_trend(importance_values)

            # Coefficient of variation (stability measure)
            coefficient_of_variation = self.calculate_coefficient_of_variation(importance_values)

            # Feature decay rate
            decay_rate = self.calculate_decay_rate(importance_values)

            # Overall reliability score
            reliability_score = self.calculate_reliability_score(
                yoy_correlation, trend_slope, coefficient_of_variation,
                decay_rate, importance_values.mean()
            )

            # Classification and recommendation
            classification = self.classify_reliability(reliability_score)
            recommendation = self.get_reliability_recommendation(classification, decay_rate)

            stability_analysis[feature] = {
                'yoy_correlation': yoy_correlation,
                'trend_slope': trend_slope,
                'trend_significance': trend_p_value,
                'coefficient_of_variation': coefficient_of_variation,
                'decay_rate': decay_rate,
                'mean_importance': importance_values.mean(),
                'std_importance': importance_values.std(),
                'reliability_score': reliability_score,
                'classification': classification,
                'recommendation': recommendation,
                'data_points': len(importance_values),
                'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable'
            }

        return stability_analysis

    def analyze_seasonal_patterns(self, feature_performance: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in feature performance"""

        if 'week' not in feature_performance.columns:
            return {'error': 'Week column required for seasonal pattern analysis'}

        seasonal_analysis = {}

        for feature in feature_performance.columns:
            if feature == 'week':
                continue

            performance_by_week = feature_performance.groupby('week')[feature].agg(['mean', 'std', 'count'])

            # Early season vs late season performance
            early_season = performance_by_week.loc[1:8, 'mean'].mean()  # Weeks 1-8
            late_season = performance_by_week.loc[9:17, 'mean'].mean()  # Weeks 9-17

            seasonal_diff = late_season - early_season
            seasonal_effect_size = seasonal_diff / performance_by_week['mean'].std() if performance_by_week['mean'].std() > 0 else 0

            # Week-to-week volatility
            weekly_volatility = performance_by_week['mean'].std()

            # Identify best/worst performing weeks
            best_weeks = performance_by_week.nlargest(3, 'mean').index.tolist()
            worst_weeks = performance_by_week.nsmallest(3, 'mean').index.tolist()

            seasonal_analysis[feature] = {
                'early_season_performance': early_season,
                'late_season_performance': late_season,
                'seasonal_difference': seasonal_diff,
                'seasonal_effect_size': seasonal_effect_size,
                'weekly_volatility': weekly_volatility,
                'best_performing_weeks': best_weeks,
                'worst_performing_weeks': worst_weeks,
                'consistent_across_season': abs(seasonal_effect_size) < 0.2
            }

        return seasonal_analysis

    def detect_regime_changes(self, feature_performance: pd.DataFrame) -> Dict[str, Any]:
        """Detect structural breaks or regime changes in feature performance"""

        regime_analysis = {}

        for feature in feature_performance.columns:
            if feature in ['week', 'season', 'date']:
                continue

            performance_series = feature_performance[feature].dropna()

            if len(performance_series) < 20:  # Need sufficient data
                regime_analysis[feature] = {'error': 'Insufficient data for regime analysis'}
                continue

            # Simple regime change detection using rolling mean differences
            window_size = min(10, len(performance_series) // 4)
            rolling_mean = performance_series.rolling(window=window_size, center=True).mean()

            # Calculate changes in rolling mean
            mean_changes = rolling_mean.diff().abs()
            significant_changes = mean_changes > (2 * mean_changes.std())

            # Identify potential change points
            change_points = performance_series.index[significant_changes].tolist()

            # Calculate stability periods
            if len(change_points) > 0:
                stability_periods = []
                start_idx = 0
                for change_point in change_points:
                    period_length = change_point - start_idx
                    stability_periods.append(period_length)
                    start_idx = change_point

                avg_stability_period = np.mean(stability_periods) if stability_periods else len(performance_series)
            else:
                avg_stability_period = len(performance_series)

            regime_analysis[feature] = {
                'change_points_detected': len(change_points),
                'change_point_indices': change_points,
                'average_stability_period': avg_stability_period,
                'regime_stability_score': min(1.0, avg_stability_period / len(performance_series)),
                'recent_regime_stable': len(change_points) == 0 or (len(change_points) > 0 and change_points[-1] < len(performance_series) * 0.8)
            }

        return regime_analysis

    def run_comprehensive_temporal_analysis(self, feature_importance_by_season: pd.DataFrame,
                                          feature_performance_by_week: pd.DataFrame = None) -> Dict[str, Any]:
        """Run complete Phase 3 temporal stability analysis"""

        results = {
            'validation_timestamp': pd.Timestamp.now(),
            'phase': 'Phase 3: Temporal Stability Analysis',
            'seasons_analyzed': len(feature_importance_by_season)
        }

        # Step 1: Core temporal stability testing
        stability_results = self.test_temporal_stability(feature_importance_by_season)
        results['temporal_stability'] = stability_results

        if 'error' in stability_results:
            results['recommendation'] = f"Cannot proceed: {stability_results['error']}"
            return results

        # Step 2: Seasonal pattern analysis (if weekly data provided)
        if feature_performance_by_week is not None:
            seasonal_results = self.analyze_seasonal_patterns(feature_performance_by_week)
            results['seasonal_patterns'] = seasonal_results

            # Step 3: Regime change detection
            regime_results = self.detect_regime_changes(feature_performance_by_week)
            results['regime_analysis'] = regime_results
        else:
            results['seasonal_patterns'] = {'note': 'Weekly data not provided'}
            results['regime_analysis'] = {'note': 'Weekly data not provided'}

        # Step 4: Generate overall recommendation
        reliable_features = [
            feature for feature, analysis in stability_results.items()
            if isinstance(analysis, dict) and analysis.get('classification') in ['highly_reliable', 'moderately_reliable']
        ]

        if len(reliable_features) > 0:
            results['recommendation'] = f'Proceed to Phase 4: {len(reliable_features)} features show temporal stability'
            results['reliable_features'] = reliable_features
        else:
            results['recommendation'] = 'Reconsider implementation: No temporally stable features detected'
            results['reliable_features'] = []

        # Step 5: Feature ranking by temporal stability
        feature_rankings = []
        for feature, analysis in stability_results.items():
            if isinstance(analysis, dict) and 'reliability_score' in analysis:
                feature_rankings.append({
                    'feature': feature,
                    'reliability_score': analysis['reliability_score'],
                    'classification': analysis['classification'],
                    'recommendation': analysis['recommendation']
                })

        # Sort by reliability score descending
        feature_rankings.sort(key=lambda x: x['reliability_score'], reverse=True)
        results['feature_rankings'] = feature_rankings

        return results