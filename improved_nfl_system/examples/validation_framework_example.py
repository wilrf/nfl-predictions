"""
Example usage of the Data Validation Framework

This example demonstrates how to use the complete 5-phase validation framework
to scientifically validate data sources for betting value.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add the parent directory to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import DataValidationFramework

def create_example_data():
    """Create realistic example data for demonstration"""
    print("Creating example NFL data...")

    np.random.seed(42)  # For reproducible results
    n_games = 400  # About 1.5 seasons of data
    n_seasons = 4

    # Create baseline features (traditional stats)
    baseline_features = pd.DataFrame({
        'season': np.repeat(range(2020, 2020 + n_seasons), n_games // n_seasons),
        'week': np.tile(range(1, 18), 30)[:n_games],  # Tile enough times to cover all games
        'home_total_yards': np.random.normal(380, 60, n_games),
        'away_total_yards': np.random.normal(360, 55, n_games),
        'home_turnovers': np.random.poisson(1.5, n_games),
        'away_turnovers': np.random.poisson(1.5, n_games),
        'home_time_of_possession': np.random.normal(30, 3, n_games)
    })
    # Calculate away_time_of_possession as complement to 60 minutes
    baseline_features['away_time_of_possession'] = 60 - baseline_features['home_time_of_possession']

    # Create new features (EPA-based metrics - should be more predictive)
    new_features = pd.DataFrame({
        'home_epa_per_play_l5': np.random.normal(0.05, 0.15, n_games),
        'away_epa_per_play_l5': np.random.normal(0.02, 0.15, n_games),
        'epa_differential_trend': np.random.normal(0, 0.3, n_games)
    })

    # Create target variable (point differential)
    # EPA should be more predictive than traditional stats
    target = (
        (baseline_features['home_total_yards'] - baseline_features['away_total_yards']) * 0.01 +
        (baseline_features['away_turnovers'] - baseline_features['home_turnovers']) * 1.5 +
        new_features['home_epa_per_play_l5'] * 15 +  # Strong EPA predictive power
        new_features['away_epa_per_play_l5'] * -15 +
        new_features['epa_differential_trend'] * 8 +
        np.random.normal(0, 7, n_games)  # Random noise
    )

    # Create market data (betting lines and outcomes)
    market_data = {
        'predictions': target + np.random.normal(0, 2, n_games),  # Our model predictions
        'market_lines': target + np.random.normal(0, 3, n_games),  # Market lines (slightly less accurate)
        'outcomes': (target > 0).astype(int)  # 1 if home team covers, 0 otherwise
    }

    # Create supplementary data for interaction testing
    supplementary_data = {
        'weather': pd.DataFrame({
            'temperature': np.random.normal(65, 20, n_games),
            'wind_speed': np.random.exponential(5, n_games),
            'precipitation': np.random.exponential(0.1, n_games)
        }),
        'referee': pd.DataFrame({
            'avg_penalties_per_game': np.random.normal(12, 3, n_games)
        }),
        'injury': pd.DataFrame({
            'key_players_out': np.random.poisson(1.2, n_games)
        }),
        'outcomes': pd.DataFrame({
            'season': baseline_features['season'],
            'total_penalties': np.random.poisson(12, n_games),
            'total_score_variance': np.random.exponential(15, n_games),
            'team_epa': new_features['home_epa_per_play_l5'],
            'point_differential': target
        })
    }

    # Create feature history for temporal analysis
    feature_history = {
        'feature_importance_by_season': pd.DataFrame({
            'home_epa_per_play_l5': [0.18, 0.22, 0.20, 0.21],
            'away_epa_per_play_l5': [0.16, 0.19, 0.17, 0.18],
            'epa_differential_trend': [0.14, 0.15, 0.13, 0.14],
            'home_total_yards': [0.08, 0.07, 0.06, 0.06],  # Declining importance
            'away_total_yards': [0.07, 0.06, 0.05, 0.05]
        }, index=range(2020, 2024)),

        'feature_performance_by_week': pd.DataFrame({
            'week': np.tile(range(1, 18), 3),
            'home_epa_per_play_l5': np.random.normal(0.18, 0.02, 51),
            'away_epa_per_play_l5': np.random.normal(0.17, 0.02, 51),
            'epa_differential_trend': np.random.normal(0.14, 0.015, 51)
        }),

        'performance_history': {
            'home_epa_per_play_l5': pd.Series(np.random.normal(0.18, 0.02, 60)),
            'away_epa_per_play_l5': pd.Series(np.random.normal(0.17, 0.02, 60)),
            'epa_differential_trend': pd.Series(np.random.normal(0.14, 0.015, 60)),
            'home_total_yards': pd.Series(np.concatenate([
                np.random.normal(0.08, 0.01, 30),  # Historical performance
                np.random.normal(0.05, 0.01, 30)   # Recent decline
            ]))
        },

        'recent_performance': {
            'home_epa_per_play_l5': 0.21,
            'away_epa_per_play_l5': 0.18,
            'epa_differential_trend': 0.15,
            'home_total_yards': 0.04  # Recent poor performance
        }
    }

    return baseline_features, new_features, target, market_data, supplementary_data, feature_history

def run_single_phase_examples():
    """Demonstrate individual phase usage"""
    print("\n=== Running Individual Phase Examples ===")

    baseline_features, new_features, target, market_data, supplementary_data, feature_history = create_example_data()

    # Initialize framework
    framework = DataValidationFramework({
        'enable_detailed_logging': True,
        'save_intermediate_results': False
    })

    print("\n--- Phase 1: Statistical Foundation ---")
    phase_1_results = framework.run_phase_1_statistical_foundation(
        'epa_metrics', baseline_features, new_features, target
    )
    print(f"Phase 1 Success: {phase_1_results['phase_1_success']}")
    print(f"Recommendation: {phase_1_results['recommendation']}")
    if 'importance_testing' in phase_1_results:
        importance = phase_1_results['importance_testing']
        print(f"Statistical Significance: {importance.get('statistically_significant', 'N/A')}")
        print(f"Effect Size: {importance.get('effect_size', 'N/A'):.3f}")

    print("\n--- Phase 2: Market Validation ---")
    phase_2_results = framework.run_phase_2_market_validation(
        'epa_metrics',
        market_data['predictions'],
        market_data['market_lines'],
        market_data['outcomes'],
        supplementary_data
    )
    print(f"Phase 2 Success: {phase_2_results['phase_2_success']}")
    if 'market_efficiency' in phase_2_results:
        market = phase_2_results['market_efficiency']
        print(f"Actual ROI: {market.get('actual_roi', 'N/A'):.3f}")
        print(f"Exploitable: {market.get('exploitable', 'N/A')}")
        print(f"Recommendation: {market.get('recommendation', 'N/A')}")

    print("\n--- Phase 3: Temporal Analysis ---")
    phase_3_results = framework.run_phase_3_temporal_analysis(
        feature_history['feature_importance_by_season'],
        feature_history['feature_performance_by_week']
    )
    print(f"Phase 3 Success: {phase_3_results['phase_3_success']}")
    print(f"Reliable Features: {len(phase_3_results.get('reliable_features', []))}")
    if 'feature_rankings' in phase_3_results:
        print("Top 3 Features by Reliability:")
        for i, feature in enumerate(phase_3_results['feature_rankings'][:3]):
            print(f"  {i+1}. {feature['feature']}: {feature['reliability_score']:.3f} ({feature['classification']})")

    print("\n--- Phase 4: Implementation Strategy ---")
    phase_4_results = framework.run_phase_4_implementation_strategy(
        phase_1_results, phase_2_results, phase_3_results
    )
    print(f"Phase 4 Success: {phase_4_results['phase_4_success']}")
    if 'implementation_roadmap' in phase_4_results:
        roadmap = phase_4_results['implementation_roadmap']
        immediate = roadmap.get('phase_1_immediate', {})
        print(f"Immediate Implementation Features: {len(immediate.get('features', []))}")
        print(f"Expected ROI: {immediate.get('total_expected_roi', 0):.3f}")

    print("\n--- Phase 5: Monitoring ---")
    phase_5_results = framework.run_phase_5_monitoring(
        feature_history['performance_history'],
        feature_history['recent_performance']
    )
    print(f"Phase 5 Success: {phase_5_results['phase_5_success']}")
    if 'health_dashboard' in phase_5_results:
        health = phase_5_results['health_dashboard']
        print(f"System Health Score: {health.get('system_health_score', 0):.3f}")
        print(f"Health Status: {health.get('health_status', 'unknown')}")
        print(f"Critical Features: {health.get('alert_summary', {}).get('critical', 0)}")

def run_complete_pipeline_example():
    """Demonstrate complete pipeline usage"""
    print("\n\n=== Running Complete Pipeline Example ===")

    baseline_features, new_features, target, market_data, supplementary_data, feature_history = create_example_data()

    # Initialize framework with custom configuration
    config = {
        'min_seasons_required': 3,
        'min_sample_size': 100,
        'significance_level': 0.05,
        'roi_threshold': 0.02,
        'enable_detailed_logging': True,
        'save_intermediate_results': False
    }

    framework = DataValidationFramework(config)

    print("Starting complete validation pipeline for EPA metrics...")

    # Run complete pipeline
    results = framework.run_complete_validation_pipeline(
        data_source='epa_metrics',
        baseline_features=baseline_features,
        new_features=new_features,
        target=target,
        market_data=market_data,
        supplementary_data=supplementary_data,
        feature_history=feature_history
    )

    # Display results
    print(f"\nPipeline Success: {results['pipeline_success']}")
    print(f"Phases Completed: {results['phases_completed']}/5")
    print(f"Total Runtime: {results.get('total_runtime', 0):.2f} seconds")
    print(f"\nFinal Recommendation: {results['final_recommendation']}")

    # Detailed phase results
    if results['phases_completed'] >= 1 and 'phase_1' in results:
        print(f"\nPhase 1 - Statistical Foundation:")
        phase1 = results['phase_1']
        if 'importance_testing' in phase1:
            imp = phase1['importance_testing']
            print(f"  MSE Improvement: {imp.get('mse_improvement', 0):.4f}")
            print(f"  P-value: {imp.get('p_value', 1):.4f}")
            print(f"  Effect Size: {imp.get('effect_size', 0):.3f}")

    if results['phases_completed'] >= 2 and 'phase_2' in results:
        print(f"\nPhase 2 - Market Validation:")
        phase2 = results['phase_2']
        if 'market_efficiency' in phase2:
            market = phase2['market_efficiency']
            print(f"  Actual ROI: {market.get('actual_roi', 0):.3f}")
            print(f"  Win Rate: {market.get('win_rate', 0):.3f}")
            print(f"  Market Efficiency Score: {market.get('efficiency_score', 0):.3f}")
            print(f"  Exploitable: {market.get('exploitable', False)}")

    if results['phases_completed'] >= 3 and 'phase_3' in results:
        print(f"\nPhase 3 - Temporal Stability:")
        phase3 = results['phase_3']
        reliable_features = phase3.get('reliable_features', [])
        print(f"  Reliable Features: {len(reliable_features)}")
        if reliable_features:
            print(f"  Top Reliable Features: {', '.join(reliable_features[:3])}")

    if results['phases_completed'] >= 4 and 'phase_4' in results:
        print(f"\nPhase 4 - Implementation Strategy:")
        phase4 = results['phase_4']
        if 'implementation_roadmap' in phase4:
            roadmap = phase4['implementation_roadmap']
            immediate = roadmap.get('phase_1_immediate', {})
            next_phase = roadmap.get('phase_2_next', {})
            print(f"  Immediate Implementation: {len(immediate.get('features', []))} features")
            print(f"  Next Phase: {len(next_phase.get('features', []))} features")
            print(f"  Total Expected ROI: {immediate.get('total_expected_roi', 0) + next_phase.get('total_expected_roi', 0):.3f}")

    if results['phases_completed'] >= 5 and 'phase_5' in results:
        print(f"\nPhase 5 - Monitoring:")
        phase5 = results['phase_5']
        if 'health_dashboard' in phase5:
            health = phase5['health_dashboard']
            print(f"  System Health Score: {health.get('system_health_score', 0):.3f}")
            print(f"  Features Monitored: {health.get('features_monitored', 0)}")
            alerts = health.get('alert_summary', {})
            print(f"  Alert Summary: {alerts.get('critical', 0)} critical, {alerts.get('warning', 0)} warning")

def demonstrate_tier_analysis():
    """Demonstrate tier-based feature analysis"""
    print("\n\n=== Tier-Based Feature Analysis Demo ===")

    from validation import ImplementationStrategy

    strategy = ImplementationStrategy()

    print("Feature Implementation Tiers:")

    for tier, features in strategy.tier_configurations.items():
        print(f"\n{tier.value.upper().replace('_', ' ')}")
        for feature_name, feature_set in features.items():
            print(f"  • {feature_set.name}")
            print(f"    Expected ROI: {feature_set.expected_roi_improvement:.1%}")
            print(f"    Confidence: {feature_set.confidence_level:.0%}")
            print(f"    Implementation Cost: {feature_set.implementation_cost} hours")
            print(f"    Priority Score: {feature_set.priority_score}/10")
            if feature_set.warning:
                print(f"    ⚠️  Warning: {feature_set.warning}")

    # Generate testing schedule
    schedule = strategy.generate_testing_schedule()
    print(f"\n4-Week Testing Schedule:")
    for week, details in schedule.items():
        print(f"\n{week.upper().replace('_', ' ')}:")
        print(f"  Focus: {details['tier']}")
        for day, task in details['tasks'].items():
            print(f"    {day.replace('_', ' ')}: {task}")

def show_expected_results():
    """Show expected results by tier"""
    print("\n\n=== Expected Results by Implementation Priority ===")

    from validation import EXPECTED_RESULTS, IMPLEMENTATION_TIMELINE

    for tier, features in EXPECTED_RESULTS.items():
        print(f"\n{tier.upper().replace('_', ' ')}:")
        for feature, expectations in features.items():
            if 'roi_improvement' in expectations:
                print(f"  • {feature}: {expectations['roi_improvement']} ROI improvement ({expectations['confidence']} confidence)")
            else:
                print(f"  • {feature}: {expectations['note']}")

    print(f"\nImplementation Timeline:")
    for week, description in IMPLEMENTATION_TIMELINE.items():
        print(f"  {week.replace('_', ' ').title()}: {description}")

def main():
    """Run the complete example demonstration"""
    print("=" * 80)
    print("Data Validation Framework - Complete Example")
    print("=" * 80)
    print("""
This framework provides a scientifically rigorous, market-aware approach
to validate which data sources deliver actual betting value. It combines
statistical significance testing with practical betting performance
metrics while accounting for market efficiency and temporal stability.
""")

    # Run examples
    run_single_phase_examples()
    run_complete_pipeline_example()
    demonstrate_tier_analysis()
    show_expected_results()

    print("\n\n=== Framework Summary ===")
    print("""
The 5-phase validation framework ensures you invest development time only
in data sources that provide:

✓ Statistical significance (Phase 1)
✓ Market exploitability (Phase 2)
✓ Temporal stability (Phase 3)
✓ Implementation value (Phase 4)
✓ Ongoing reliability (Phase 5)

This systematic approach prevents wasted effort on features that markets
already price efficiently or that show inconsistent predictive power.
""")

if __name__ == "__main__":
    main()