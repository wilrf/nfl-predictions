"""
Phase 1: Enhanced Statistical Foundation
Production-grade data validation with sample size validation and leak-free cross-validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

class ProductionDataTester:
    """
    Scientifically rigorous data validation framework that prevents false discoveries
    through insufficient data and temporal data leakage
    """

    def __init__(self):
        self.min_sample_sizes = {
            'epa_predictive_power': 500,    # Core performance metrics
            'injury_impact': 200,           # Key player availability
            'weather_impact': 100,          # Outdoor games only
            'referee_tendencies': 50,       # Per-referee minimum
            'snap_count_trends': 300,       # Usage pattern analysis
            'ngs_metrics': 400,             # Advanced analytics
        }

        self.significance_thresholds = {
            'statistical_significance': 0.05,  # Standard p-value
            'practical_significance': 0.2,     # Minimum effect size
            'temporal_stability': 0.6,         # Year-over-year reliability
            'market_exploitability': 0.02      # Minimum ROI improvement
        }

        self.logger = logging.getLogger(__name__)

    def calculate_statistical_power(self, sample_size: int, effect_size: float = 0.2,
                                  alpha: float = 0.05) -> float:
        """Calculate statistical power for given sample size and effect size"""
        try:
            from scipy.stats import norm

            # For two-sample t-test
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(0.8)  # 80% power target

            # Cohen's d calculation
            ncp = effect_size * np.sqrt(sample_size/2)
            power = 1 - norm.cdf(z_alpha - ncp)

            return min(power, 1.0)
        except Exception as e:
            self.logger.warning(f"Power calculation failed: {e}")
            return 0.5  # Conservative estimate

    def validate_testing_readiness(self, data_source: str, available_data: pd.DataFrame) -> Dict[str, Any]:
        """Ensure sufficient sample size before testing"""
        if data_source not in self.min_sample_sizes:
            return {
                'ready': False,
                'message': f"Unknown data source: {data_source}",
                'recommendation': 'Add data source to min_sample_sizes configuration'
            }

        required_samples = self.min_sample_sizes[data_source]
        actual_samples = len(available_data)

        if actual_samples < required_samples:
            return {
                'ready': False,
                'message': f"{data_source}: {actual_samples} samples (need {required_samples})",
                'recommendation': 'Collect more data or test with reduced confidence'
            }

        power = self.calculate_statistical_power(actual_samples)

        return {
            'ready': True,
            'power': power,
            'sample_size': actual_samples,
            'required_size': required_samples,
            'excess_samples': actual_samples - required_samples
        }

    def test_feature_importance_leak_free(self, baseline_features: pd.DataFrame,
                                        new_features: pd.DataFrame,
                                        target: pd.Series) -> Dict[str, Any]:
        """Temporally-aware testing that prevents future data leakage"""

        if 'season' not in baseline_features.columns:
            return {'error': 'baseline_features must contain "season" column for temporal validation'}

        # Stratify by season to prevent information leakage
        seasons = sorted(baseline_features['season'].unique())
        temporal_splits = []

        for test_season in seasons:
            train_seasons = [s for s in seasons if s < test_season]
            if len(train_seasons) < 2:  # Need minimum training history
                continue

            train_mask = baseline_features['season'].isin(train_seasons)
            test_mask = baseline_features['season'] == test_season
            temporal_splits.append((train_mask, test_mask))

        if len(temporal_splits) < 3:
            return {'error': 'Insufficient temporal data for reliable testing (need 3+ seasons)'}

        # Test with temporal validation
        baseline_scores = []
        enhanced_scores = []

        for train_idx, test_idx in temporal_splits:
            try:
                # Baseline model
                baseline_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
                baseline_train_features = baseline_features.loc[train_idx].drop('season', axis=1)
                baseline_test_features = baseline_features.loc[test_idx].drop('season', axis=1)

                baseline_model.fit(baseline_train_features, target.loc[train_idx])
                baseline_pred = baseline_model.predict(baseline_test_features)
                baseline_scores.append(mean_squared_error(target.loc[test_idx], baseline_pred))

                # Enhanced model
                enhanced_features_full = pd.concat([baseline_features, new_features], axis=1)
                enhanced_train_features = enhanced_features_full.loc[train_idx].drop('season', axis=1)
                enhanced_test_features = enhanced_features_full.loc[test_idx].drop('season', axis=1)

                enhanced_model = XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
                enhanced_model.fit(enhanced_train_features, target.loc[train_idx])
                enhanced_pred = enhanced_model.predict(enhanced_test_features)
                enhanced_scores.append(mean_squared_error(target.loc[test_idx], enhanced_pred))

            except Exception as e:
                self.logger.warning(f"Model training failed for temporal split: {e}")
                continue

        if len(baseline_scores) < 3:
            return {'error': 'Insufficient successful temporal splits for analysis'}

        # Statistical analysis
        improvement = np.mean(baseline_scores) - np.mean(enhanced_scores)  # Lower MSE = better

        if len(baseline_scores) > 1:
            t_stat, p_value = stats.ttest_rel(baseline_scores, enhanced_scores)
        else:
            t_stat, p_value = 0, 1.0

        effect_size = improvement / np.std(baseline_scores) if np.std(baseline_scores) > 0 else 0

        # Bonferroni correction for multiple testing
        adjusted_p_value = min(p_value * len(self.min_sample_sizes), 1.0)

        # Confidence interval
        if len(enhanced_scores) > 1:
            diff_scores = np.array(enhanced_scores) - np.array(baseline_scores)
            confidence_interval = stats.t.interval(
                0.95, len(enhanced_scores)-1,
                improvement,
                stats.sem(diff_scores)
            )
        else:
            confidence_interval = (improvement, improvement)

        return {
            'mse_improvement': improvement,
            'p_value': p_value,
            'adjusted_p_value': adjusted_p_value,
            'effect_size': effect_size,
            'statistically_significant': adjusted_p_value < self.significance_thresholds['statistical_significance'],
            'practically_significant': abs(effect_size) > self.significance_thresholds['practical_significance'],
            'confidence_interval': confidence_interval,
            'temporal_splits_used': len(temporal_splits),
            'baseline_mse_mean': np.mean(baseline_scores),
            'enhanced_mse_mean': np.mean(enhanced_scores),
            'baseline_mse_std': np.std(baseline_scores),
            'enhanced_mse_std': np.std(enhanced_scores)
        }

    def validate_feature_quality(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Validate feature quality before testing"""
        quality_report = {
            'missing_data_percentage': features.isnull().sum() / len(features),
            'zero_variance_features': [],
            'high_correlation_pairs': [],
            'outlier_percentage': {},
            'data_quality_score': 0.0
        }

        # Check for zero variance features
        for col in features.columns:
            if features[col].var() == 0:
                quality_report['zero_variance_features'].append(col)

        # Check for high correlation pairs
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            quality_report['high_correlation_pairs'] = high_corr_pairs

        # Check for outliers using IQR method
        for col in features.select_dtypes(include=[np.number]).columns:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((features[col] < (Q1 - 1.5 * IQR)) | (features[col] > (Q3 + 1.5 * IQR))).sum()
            quality_report['outlier_percentage'][col] = outliers / len(features)

        # Calculate overall quality score
        missing_penalty = quality_report['missing_data_percentage'].mean()
        zero_var_penalty = len(quality_report['zero_variance_features']) / len(features.columns)
        high_corr_penalty = len(quality_report['high_correlation_pairs']) / max(1, len(features.columns) * (len(features.columns) - 1) / 2)
        outlier_penalty = np.mean(list(quality_report['outlier_percentage'].values())) if quality_report['outlier_percentage'] else 0

        quality_report['data_quality_score'] = max(0, 1 - missing_penalty - zero_var_penalty - high_corr_penalty - outlier_penalty)

        return quality_report

    def run_comprehensive_validation(self, data_source: str, baseline_features: pd.DataFrame,
                                   new_features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Run complete Phase 1 validation pipeline"""

        results = {
            'data_source': data_source,
            'validation_timestamp': pd.Timestamp.now(),
            'phase': 'Phase 1: Enhanced Statistical Foundation'
        }

        # Step 1: Validate testing readiness
        readiness = self.validate_testing_readiness(data_source, baseline_features)
        results['readiness_check'] = readiness

        if not readiness['ready']:
            results['recommendation'] = 'Insufficient data for reliable testing'
            return results

        # Step 2: Validate feature quality
        baseline_quality = self.validate_feature_quality(baseline_features, target)
        new_feature_quality = self.validate_feature_quality(new_features, target)

        results['baseline_feature_quality'] = baseline_quality
        results['new_feature_quality'] = new_feature_quality

        # Step 3: Run leak-free importance testing
        importance_results = self.test_feature_importance_leak_free(baseline_features, new_features, target)
        results['importance_testing'] = importance_results

        # Step 4: Generate overall recommendation
        if 'error' in importance_results:
            results['recommendation'] = f"Testing failed: {importance_results['error']}"
        elif (importance_results['statistically_significant'] and
              importance_results['practically_significant'] and
              baseline_quality['data_quality_score'] > 0.7 and
              new_feature_quality['data_quality_score'] > 0.7):
            results['recommendation'] = 'Proceed to Phase 2: Market-Aware Validation'
        elif importance_results['statistically_significant']:
            results['recommendation'] = 'Marginal improvement detected - monitor closely'
        else:
            results['recommendation'] = 'No significant improvement - consider alternative features'

        return results