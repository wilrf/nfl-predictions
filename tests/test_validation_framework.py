"""
Comprehensive tests for the data validation framework
Tests all 5 phases and the complete pipeline
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import DataValidationFramework, ProductionDataTester, MarketEfficiencyTester
from validation import TemporalStabilityAnalyzer, ImplementationStrategy, PerformanceMonitor

class TestDataValidationFramework:
    """Test the complete data validation framework"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)  # For reproducible results

        n_samples = 300
        n_seasons = 4

        # Create baseline features
        baseline_features = pd.DataFrame({
            'season': np.repeat(range(2020, 2020 + n_seasons), n_samples // n_seasons),
            'home_score': np.random.normal(24, 7, n_samples),
            'away_score': np.random.normal(21, 6, n_samples),
            'home_yards': np.random.normal(350, 50, n_samples),
            'away_yards': np.random.normal(320, 45, n_samples)
        })

        # Create new features (with some actual predictive power)
        new_features = pd.DataFrame({
            'epa_differential': np.random.normal(0, 2, n_samples),
            'injury_impact': np.random.uniform(0, 5, n_samples)
        })

        # Create target variable (correlated with features)
        target = (
            baseline_features['home_score'] - baseline_features['away_score'] +
            new_features['epa_differential'] * 2 +
            np.random.normal(0, 3, n_samples)
        )

        # Create market data
        market_data = {
            'predictions': target + np.random.normal(0, 1, n_samples),
            'market_lines': target + np.random.normal(0, 2, n_samples),
            'outcomes': (target > 0).astype(int)
        }

        return baseline_features, new_features, target, market_data

    @pytest.fixture
    def framework(self):
        """Create framework instance for testing"""
        config = {
            'min_seasons_required': 3,
            'min_sample_size': 50,
            'enable_detailed_logging': False,
            'save_intermediate_results': False
        }
        return DataValidationFramework(config)

    def test_input_validation(self, framework, sample_data):
        """Test input data validation"""
        baseline_features, new_features, target, market_data = sample_data

        # Test valid data
        validation = framework.validate_input_data(
            baseline_features, new_features, target, market_data
        )
        assert validation['data_quality_passed'] == True

        # Test missing season column
        baseline_no_season = baseline_features.drop('season', axis=1)
        validation = framework.validate_input_data(
            baseline_no_season, new_features, target, market_data
        )
        assert validation['data_quality_passed'] == False

        # Test mismatched lengths
        short_target = target[:100]
        validation = framework.validate_input_data(
            baseline_features, new_features, short_target, market_data
        )
        assert validation['data_quality_passed'] == False

    def test_phase_1_statistical_foundation(self, framework, sample_data):
        """Test Phase 1: Statistical Foundation"""
        baseline_features, new_features, target, _ = sample_data

        results = framework.run_phase_1_statistical_foundation(
            'test_feature', baseline_features, new_features, target
        )

        assert results['phase_1_success'] == True
        assert 'readiness_check' in results
        assert 'importance_testing' in results
        assert results['readiness_check']['ready'] == True

    def test_phase_2_market_validation(self, framework, sample_data):
        """Test Phase 2: Market Validation"""
        _, _, _, market_data = sample_data

        results = framework.run_phase_2_market_validation(
            'test_feature',
            market_data['predictions'],
            market_data['market_lines'],
            market_data['outcomes']
        )

        assert results['phase_2_success'] == True
        assert 'market_efficiency' in results
        assert 'actual_roi' in results['market_efficiency']

    def test_phase_3_temporal_analysis(self, framework):
        """Test Phase 3: Temporal Analysis"""
        # Create feature importance by season data
        feature_importance = pd.DataFrame({
            'epa_differential': [0.15, 0.18, 0.16, 0.17],
            'injury_impact': [0.08, 0.09, 0.07, 0.08]
        }, index=range(2020, 2024))

        results = framework.run_phase_3_temporal_analysis(feature_importance)

        assert results['phase_3_success'] == True
        assert 'temporal_stability' in results
        assert 'feature_rankings' in results

    def test_phase_4_implementation_strategy(self, framework):
        """Test Phase 4: Implementation Strategy"""
        # Mock results from previous phases
        phase_1_results = {
            'features_tested': ['epa_differential', 'injury_impact'],
            'importance_testing': {'statistically_significant': True}
        }

        phase_2_results = {
            'market_efficiency': {'actual_roi': 0.03, 'exploitable': True}
        }

        phase_3_results = {
            'temporal_stability': {
                'epa_differential': {'reliability_score': 0.8},
                'injury_impact': {'reliability_score': 0.6}
            }
        }

        results = framework.run_phase_4_implementation_strategy(
            phase_1_results, phase_2_results, phase_3_results
        )

        assert results['phase_4_success'] == True
        assert 'testing_schedule' in results

    def test_phase_5_monitoring(self, framework):
        """Test Phase 5: Monitoring"""
        # Create performance history
        performance_history = {
            'epa_differential': pd.Series(np.random.normal(0.15, 0.02, 50)),
            'injury_impact': pd.Series(np.random.normal(0.08, 0.01, 50))
        }

        results = framework.run_phase_5_monitoring(performance_history)

        assert results['phase_5_success'] == True
        assert 'monitoring_report' in results
        assert 'health_dashboard' in results

    def test_complete_pipeline(self, framework, sample_data):
        """Test the complete validation pipeline"""
        baseline_features, new_features, target, market_data = sample_data

        # Create feature history
        feature_history = {
            'feature_importance_by_season': pd.DataFrame({
                'epa_differential': [0.15, 0.18, 0.16, 0.17],
                'injury_impact': [0.08, 0.09, 0.07, 0.08]
            }, index=range(2020, 2024)),
            'performance_history': {
                'epa_differential': pd.Series(np.random.normal(0.15, 0.02, 30)),
                'injury_impact': pd.Series(np.random.normal(0.08, 0.01, 30))
            }
        }

        results = framework.run_complete_validation_pipeline(
            data_source='test_complete',
            baseline_features=baseline_features,
            new_features=new_features,
            target=target,
            market_data=market_data,
            feature_history=feature_history
        )

        assert results['pipeline_success'] == True
        assert results['phases_completed'] >= 3
        assert 'final_recommendation' in results
        assert 'pipeline_start_time' in results

class TestProductionDataTester:
    """Test Phase 1 components"""

    @pytest.fixture
    def tester(self):
        return ProductionDataTester()

    @pytest.fixture
    def test_data(self):
        np.random.seed(42)
        n_samples = 200

        baseline_features = pd.DataFrame({
            'season': np.repeat([2020, 2021, 2022, 2023], n_samples // 4),
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(1, 0.5, n_samples)
        })

        new_features = pd.DataFrame({
            'new_feature': np.random.normal(0, 1, n_samples)
        })

        target = baseline_features['feature1'] * 2 + new_features['new_feature'] + np.random.normal(0, 0.5, n_samples)

        return baseline_features, new_features, target

    def test_sample_size_validation(self, tester):
        """Test sample size validation"""
        small_data = pd.DataFrame({'test': range(10)})
        result = tester.validate_testing_readiness('epa_predictive_power', small_data)
        assert result['ready'] == False

        large_data = pd.DataFrame({'test': range(600)})
        result = tester.validate_testing_readiness('epa_predictive_power', large_data)
        assert result['ready'] == True

    def test_leak_free_validation(self, tester, test_data):
        """Test leak-free cross-validation"""
        baseline_features, new_features, target = test_data

        results = tester.test_feature_importance_leak_free(
            baseline_features, new_features, target
        )

        assert 'mse_improvement' in results
        assert 'p_value' in results
        assert 'statistically_significant' in results
        assert 'temporal_splits_used' in results

    def test_feature_quality_validation(self, tester, test_data):
        """Test feature quality validation"""
        baseline_features, new_features, target = test_data

        quality_report = tester.validate_feature_quality(baseline_features, target)

        assert 'missing_data_percentage' in quality_report
        assert 'zero_variance_features' in quality_report
        assert 'data_quality_score' in quality_report
        assert 0 <= quality_report['data_quality_score'] <= 1

class TestMarketEfficiencyTester:
    """Test Phase 2 components"""

    @pytest.fixture
    def tester(self):
        return MarketEfficiencyTester()

    @pytest.fixture
    def market_test_data(self):
        np.random.seed(42)
        n_samples = 100

        predictions = np.random.normal(0, 1, n_samples)
        market_lines = predictions + np.random.normal(0, 0.5, n_samples)
        outcomes = (predictions > 0).astype(int)

        return predictions, market_lines, outcomes

    def test_probability_conversion(self, tester):
        """Test probability conversion methods"""
        predictions = np.array([1.0, -1.0, 0.0])
        probabilities = tester.convert_prediction_to_probability(predictions)

        assert len(probabilities) == 3
        assert all(0 <= p <= 1 for p in probabilities)

        # Test odds conversion
        odds = np.array([100, -110, 200])
        market_probs = tester.convert_odds_to_probability(odds)

        assert len(market_probs) == 3
        assert all(0 <= p <= 1 for p in market_probs)

    def test_kelly_betting_simulation(self, tester):
        """Test Kelly betting simulation"""
        edge = np.array([0.1, -0.05, 0.02])
        odds = np.array([100, -110, 150])
        outcomes = np.array([1, 0, 1])

        results = tester.simulate_kelly_betting(edge, odds, outcomes)

        assert 'total_roi' in results
        assert 'win_rate' in results
        assert 'total_bets' in results

    def test_market_efficiency_testing(self, tester, market_test_data):
        """Test market efficiency analysis"""
        predictions, market_lines, outcomes = market_test_data

        results = tester.test_market_efficiency(
            predictions, market_lines, outcomes, 'test_feature'
        )

        assert 'efficiency_score' in results
        assert 'exploitable' in results
        assert 'recommendation' in results
        assert 0 <= results['efficiency_score'] <= 1

class TestTemporalStabilityAnalyzer:
    """Test Phase 3 components"""

    @pytest.fixture
    def analyzer(self):
        return TemporalStabilityAnalyzer()

    @pytest.fixture
    def temporal_data(self):
        # Create feature importance data across seasons
        feature_importance = pd.DataFrame({
            'feature1': [0.15, 0.18, 0.16, 0.17, 0.14],
            'feature2': [0.08, 0.09, 0.07, 0.08, 0.06],
            'feature3': [0.12, 0.05, 0.03, 0.02, 0.01]  # Declining feature
        }, index=range(2019, 2024))

        return feature_importance

    def test_temporal_stability_analysis(self, analyzer, temporal_data):
        """Test temporal stability analysis"""
        results = analyzer.test_temporal_stability(temporal_data)

        assert isinstance(results, dict)
        for feature in temporal_data.columns:
            assert feature in results
            feature_analysis = results[feature]
            assert 'reliability_score' in feature_analysis
            assert 'classification' in feature_analysis
            assert 'recommendation' in feature_analysis

    def test_yoy_correlation(self, analyzer):
        """Test year-over-year correlation calculation"""
        stable_series = pd.Series([0.15, 0.16, 0.15, 0.16, 0.15])
        volatile_series = pd.Series([0.15, 0.05, 0.25, 0.10, 0.20])

        stable_corr = analyzer.calculate_yoy_correlation(stable_series)
        volatile_corr = analyzer.calculate_yoy_correlation(volatile_series)

        # Stable series should have higher correlation
        assert stable_corr > volatile_corr

    def test_decay_rate_calculation(self, analyzer):
        """Test feature decay rate calculation"""
        declining_series = pd.Series([0.20, 0.18, 0.15, 0.10, 0.08])
        stable_series = pd.Series([0.15, 0.16, 0.15, 0.16, 0.15])

        declining_decay = analyzer.calculate_decay_rate(declining_series)
        stable_decay = analyzer.calculate_decay_rate(stable_series)

        # Declining series should have higher decay rate
        assert declining_decay > stable_decay

class TestImplementationStrategy:
    """Test Phase 4 components"""

    @pytest.fixture
    def strategy(self):
        return ImplementationStrategy()

    def test_tier_configurations(self, strategy):
        """Test tier-based feature configurations"""
        tiers = strategy.tier_configurations

        assert len(tiers) == 3  # Three tiers
        for tier, features in tiers.items():
            assert len(features) > 0
            for feature_name, feature_set in features.items():
                assert hasattr(feature_set, 'expected_roi_improvement')
                assert hasattr(feature_set, 'confidence_level')
                assert hasattr(feature_set, 'implementation_cost')

    def test_risk_adjusted_roi(self, strategy):
        """Test risk-adjusted ROI calculation"""
        roi = 0.05
        confidence_interval = (0.02, 0.08)

        conservative_roi = strategy.calculate_risk_adjusted_roi(roi, confidence_interval)

        # Conservative estimate should be lower than optimistic estimate
        assert conservative_roi < roi
        assert conservative_roi > confidence_interval[0]

    def test_implementation_decision(self, strategy):
        """Test implementation decision framework"""
        test_results = {
            'test_feature': {
                'roi_improvement': 0.04,
                'confidence_interval': (0.02, 0.06),
                'statistical_confidence': 0.9,
                'temporal_stability': 0.8
            }
        }

        costs = {'test_feature': {'development': 20, 'maintenance_annual': 5}}
        market_conditions = {'test_feature': {'efficiency_stability': 0.7}}

        decisions = strategy.calculate_implementation_decision(
            test_results, costs, market_conditions
        )

        assert 'test_feature' in decisions
        decision_info = decisions['test_feature']
        assert 'decision' in decision_info
        assert 'implementation_score' in decision_info
        assert 'timeline' in decision_info

class TestPerformanceMonitor:
    """Test Phase 5 components"""

    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor()

    @pytest.fixture
    def performance_data(self):
        np.random.seed(42)
        # Create performance history with some features showing decay
        stable_feature = pd.Series(np.random.normal(0.15, 0.01, 50))
        declining_feature = pd.Series(np.concatenate([
            np.random.normal(0.20, 0.01, 25),  # Good historical performance
            np.random.normal(0.10, 0.01, 25)   # Poor recent performance
        ]))

        return {
            'stable_feature': stable_feature,
            'declining_feature': declining_feature
        }

    def test_performance_metrics_calculation(self, monitor):
        """Test performance metrics calculation"""
        performance_data = pd.Series(np.random.normal(0.15, 0.02, 40))

        metrics = monitor.calculate_performance_metrics(performance_data)

        assert 'mean_performance' in metrics
        assert 'decay_rate' in metrics
        assert 'consistency_score' in metrics
        assert 0 <= metrics['consistency_score'] <= 1

    def test_decay_detection(self, monitor, performance_data):
        """Test performance decay detection"""
        results = monitor.monitor_feature_performance_decay(performance_data)

        assert 'stable_feature' in results
        assert 'declining_feature' in results

        # Declining feature should have higher decay rate
        stable_decay = results['stable_feature']['decay_rate']
        declining_decay = results['declining_feature']['decay_rate']

        assert declining_decay > stable_decay

    def test_health_dashboard(self, monitor, performance_data):
        """Test health dashboard generation"""
        dashboard = monitor.generate_health_dashboard(performance_data)

        assert 'timestamp' in dashboard
        assert 'decay_analysis' in dashboard
        assert 'alert_summary' in dashboard
        assert 'system_health_score' in dashboard
        assert 'health_status' in dashboard
        assert 'recommendations' in dashboard

        # Health score should be between 0 and 1
        assert 0 <= dashboard['system_health_score'] <= 1

    def test_anomaly_detection(self, monitor):
        """Test anomaly detection"""
        recent_performance = {'feature1': 0.05, 'feature2': 0.25}
        historical_baselines = {'feature1': 0.15, 'feature2': 0.20}

        alerts = monitor.detect_feature_anomalies(recent_performance, historical_baselines)

        # feature1 should trigger an alert (significant deviation)
        feature1_alerts = [alert for alert in alerts if alert.feature == 'feature1']
        assert len(feature1_alerts) > 0

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])