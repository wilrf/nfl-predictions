"""
Data Validation Framework

A scientifically rigorous, market-aware approach to validate which data sources
deliver actual betting value. Combines statistical significance testing with
practical betting performance metrics while accounting for market efficiency
and temporal stability.

This framework provides a 5-phase validation process:

Phase 1: Enhanced Statistical Foundation
- Sample size validation & power analysis
- Leak-free cross-validation to prevent temporal data leakage

Phase 2: Market-Aware Validation
- Market efficiency testing to determine if markets already price in information
- Interaction effect discovery between data sources

Phase 3: Temporal Stability Analysis
- Year-over-year consistency testing
- Feature decay detection

Phase 4: Prioritized Implementation Strategy
- Tier-based testing schedule (Tier 1: Highest value, Tier 2: Medium, Tier 3: Experimental)
- Risk-adjusted decision framework

Phase 5: Continuous Monitoring Framework
- Performance decay detection
- Real-time feature health monitoring

Usage:
    from validation import DataValidationFramework

    # Initialize framework
    framework = DataValidationFramework()

    # Run complete validation pipeline
    results = framework.run_complete_validation_pipeline(
        data_source='epa_metrics',
        baseline_features=baseline_df,
        new_features=new_features_df,
        target=target_series,
        market_data={'predictions': pred_array, 'market_lines': lines_array, 'outcomes': outcomes_array}
    )

    # Check recommendation
    print(results['final_recommendation'])
"""

from .data_validation_framework import DataValidationFramework
from .production_data_tester import ProductionDataTester
from .market_efficiency_tester import MarketEfficiencyTester
from .temporal_stability_analyzer import TemporalStabilityAnalyzer
from .implementation_strategy import ImplementationStrategy, ImplementationTier, FeatureSet
from .performance_monitor import PerformanceMonitor, AlertLevel, PerformanceAlert

__version__ = "1.0.0"
__author__ = "NFL Betting System"

__all__ = [
    'DataValidationFramework',
    'ProductionDataTester',
    'MarketEfficiencyTester',
    'TemporalStabilityAnalyzer',
    'ImplementationStrategy',
    'PerformanceMonitor',
    'ImplementationTier',
    'FeatureSet',
    'AlertLevel',
    'PerformanceAlert'
]

# Expected results by implementation priority
EXPECTED_RESULTS = {
    'tier_1_immediate': {
        'epa_metrics': {'roi_improvement': '3-5%', 'confidence': '95%'},
        'key_injuries': {'roi_improvement': '2-4%', 'confidence': '85%'}
    },
    'tier_2_next_phase': {
        'weather_impact': {'roi_improvement': '1-3%', 'confidence': '70%'},
        'referee_tendencies': {'roi_improvement': '1-2%', 'confidence': '60%'}
    },
    'tier_3_experimental': {
        'ngs_advanced': {'roi_improvement': '0-1%', 'confidence': '40%'},
        'usage_metrics': {'note': 'May be already priced by market'}
    }
}

# Implementation timeline
IMPLEMENTATION_TIMELINE = {
    'week_1': 'Tier 1 Testing (EPA metrics, key injuries)',
    'week_2': 'Tier 2 Testing (weather impact, referee tendencies)',
    'week_3': 'Advanced Analysis (Tier 3, temporal stability, interactions)',
    'week_4': 'Decision & Implementation (analysis, roadmap, Tier 1 deployment)'
}