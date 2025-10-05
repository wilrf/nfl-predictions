"""
Complete Data Validation Framework Orchestrator
Coordinates all 5 phases of the scientifically rigorous data validation process
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json
import os

from .production_data_tester import ProductionDataTester
from .market_efficiency_tester import MarketEfficiencyTester
from .temporal_stability_analyzer import TemporalStabilityAnalyzer
from .implementation_strategy import ImplementationStrategy
from .performance_monitor import PerformanceMonitor

class DataValidationFramework:
    """
    Complete 5-phase data validation framework that ensures development time
    is invested only in data sources that provide statistically significant,
    practically meaningful, and temporally stable betting value
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()

        # Initialize all phase components
        self.phase_1 = ProductionDataTester()
        self.phase_2 = MarketEfficiencyTester()
        self.phase_3 = TemporalStabilityAnalyzer()
        self.phase_4 = ImplementationStrategy()
        self.phase_5 = PerformanceMonitor()

        self.validation_history = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the framework"""
        return {
            'min_seasons_required': 3,
            'min_sample_size': 100,
            'significance_level': 0.05,
            'roi_threshold': 0.02,
            'monitoring_window': 30,
            'output_directory': 'validation_results',
            'enable_detailed_logging': True,
            'save_intermediate_results': True
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DataValidationFramework')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config['enable_detailed_logging'] else logging.WARNING)

        return logger

    def validate_input_data(self, baseline_features: pd.DataFrame,
                          new_features: pd.DataFrame,
                          target: pd.Series,
                          market_data: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """Validate input data quality and completeness"""

        validation_results = {
            'data_quality_passed': True,
            'issues': [],
            'recommendations': []
        }

        # Check for required columns
        if 'season' not in baseline_features.columns:
            validation_results['data_quality_passed'] = False
            validation_results['issues'].append('baseline_features missing required "season" column')

        # Check data alignment
        if len(baseline_features) != len(new_features) or len(baseline_features) != len(target):
            validation_results['data_quality_passed'] = False
            validation_results['issues'].append('Data length mismatch between features and target')

        # Check for sufficient historical data
        if 'season' in baseline_features.columns:
            unique_seasons = baseline_features['season'].nunique()
            if unique_seasons < self.config['min_seasons_required']:
                validation_results['data_quality_passed'] = False
                validation_results['issues'].append(
                    f'Insufficient seasons: {unique_seasons} (need {self.config["min_seasons_required"]})'
                )

        # Check sample size
        if len(baseline_features) < self.config['min_sample_size']:
            validation_results['issues'].append(
                f'Small sample size: {len(baseline_features)} (recommended {self.config["min_sample_size"]}+)'
            )
            validation_results['recommendations'].append('Consider collecting more data for robust analysis')

        # Check for missing values
        baseline_missing = baseline_features.isnull().sum().sum()
        new_features_missing = new_features.isnull().sum().sum()
        target_missing = target.isnull().sum()

        if baseline_missing > 0 or new_features_missing > 0 or target_missing > 0:
            validation_results['issues'].append(
                f'Missing values detected: baseline({baseline_missing}), '
                f'new_features({new_features_missing}), target({target_missing})'
            )

        # Validate market data if provided
        if market_data:
            required_market_fields = ['predictions', 'market_lines', 'outcomes']
            for field in required_market_fields:
                if field not in market_data:
                    validation_results['issues'].append(f'Missing required market data field: {field}')

        return validation_results

    def run_phase_1_statistical_foundation(self, data_source: str,
                                         baseline_features: pd.DataFrame,
                                         new_features: pd.DataFrame,
                                         target: pd.Series) -> Dict[str, Any]:
        """Run Phase 1: Enhanced Statistical Foundation"""

        self.logger.info(f"Starting Phase 1: Statistical Foundation for {data_source}")

        try:
            results = self.phase_1.run_comprehensive_validation(
                data_source, baseline_features, new_features, target
            )
            results['phase_1_success'] = True

            if self.config['save_intermediate_results']:
                self._save_intermediate_results('phase_1', data_source, results)

        except Exception as e:
            self.logger.error(f"Phase 1 failed for {data_source}: {str(e)}")
            results = {
                'phase_1_success': False,
                'error': str(e),
                'recommendation': 'Fix data issues before proceeding'
            }

        return results

    def run_phase_2_market_validation(self, data_source: str,
                                    predictions: np.ndarray,
                                    market_lines: np.ndarray,
                                    outcomes: np.ndarray,
                                    supplementary_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Run Phase 2: Market-Aware Validation"""

        self.logger.info(f"Starting Phase 2: Market Validation for {data_source}")

        try:
            # Prepare supplementary data
            weather_data = supplementary_data.get('weather', pd.DataFrame()) if supplementary_data else pd.DataFrame()
            referee_data = supplementary_data.get('referee', pd.DataFrame()) if supplementary_data else pd.DataFrame()
            injury_data = supplementary_data.get('injury', pd.DataFrame()) if supplementary_data else pd.DataFrame()
            outcome_data = supplementary_data.get('outcomes', pd.DataFrame()) if supplementary_data else pd.DataFrame()

            results = self.phase_2.run_comprehensive_market_testing(
                data_source, predictions, market_lines, outcomes,
                weather_data, referee_data, injury_data, outcome_data
            )
            results['phase_2_success'] = True

            if self.config['save_intermediate_results']:
                self._save_intermediate_results('phase_2', data_source, results)

        except Exception as e:
            self.logger.error(f"Phase 2 failed for {data_source}: {str(e)}")
            results = {
                'phase_2_success': False,
                'error': str(e),
                'recommendation': 'Check market data quality and predictions format'
            }

        return results

    def run_phase_3_temporal_analysis(self, feature_importance_by_season: pd.DataFrame,
                                    feature_performance_by_week: pd.DataFrame = None) -> Dict[str, Any]:
        """Run Phase 3: Temporal Stability Analysis"""

        self.logger.info("Starting Phase 3: Temporal Stability Analysis")

        try:
            results = self.phase_3.run_comprehensive_temporal_analysis(
                feature_importance_by_season, feature_performance_by_week
            )
            results['phase_3_success'] = True

            if self.config['save_intermediate_results']:
                self._save_intermediate_results('phase_3', 'temporal_analysis', results)

        except Exception as e:
            self.logger.error(f"Phase 3 failed: {str(e)}")
            results = {
                'phase_3_success': False,
                'error': str(e),
                'recommendation': 'Ensure feature importance data spans multiple seasons'
            }

        return results

    def run_phase_4_implementation_strategy(self, phase_1_results: Dict[str, Any],
                                          phase_2_results: Dict[str, Any],
                                          phase_3_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run Phase 4: Prioritized Implementation Strategy"""

        self.logger.info("Starting Phase 4: Implementation Strategy")

        try:
            results = self.phase_4.run_comprehensive_implementation_strategy(
                phase_1_results, phase_2_results, phase_3_results
            )
            results['phase_4_success'] = True

            if self.config['save_intermediate_results']:
                self._save_intermediate_results('phase_4', 'implementation_strategy', results)

        except Exception as e:
            self.logger.error(f"Phase 4 failed: {str(e)}")
            results = {
                'phase_4_success': False,
                'error': str(e),
                'recommendation': 'Ensure previous phases completed successfully'
            }

        return results

    def run_phase_5_monitoring(self, feature_performance_history: Dict[str, pd.Series],
                             recent_performance: Dict[str, float] = None,
                             correlation_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run Phase 5: Continuous Monitoring Framework"""

        self.logger.info("Starting Phase 5: Continuous Monitoring")

        try:
            results = self.phase_5.run_comprehensive_monitoring(
                feature_performance_history, recent_performance, correlation_data
            )
            results['phase_5_success'] = True

            if self.config['save_intermediate_results']:
                self._save_intermediate_results('phase_5', 'monitoring', results)

        except Exception as e:
            self.logger.error(f"Phase 5 failed: {str(e)}")
            results = {
                'phase_5_success': False,
                'error': str(e),
                'recommendation': 'Check feature performance history data format'
            }

        return results

    def run_complete_validation_pipeline(self, data_source: str,
                                       baseline_features: pd.DataFrame,
                                       new_features: pd.DataFrame,
                                       target: pd.Series,
                                       market_data: Dict[str, np.ndarray] = None,
                                       supplementary_data: Dict[str, pd.DataFrame] = None,
                                       feature_history: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete 5-phase validation pipeline"""

        pipeline_start_time = datetime.now()
        self.logger.info(f"Starting complete validation pipeline for {data_source}")

        # Initialize pipeline results
        pipeline_results = {
            'data_source': data_source,
            'pipeline_start_time': pipeline_start_time,
            'pipeline_success': False,
            'phases_completed': 0,
            'final_recommendation': 'Pipeline incomplete'
        }

        try:
            # Input validation
            input_validation = self.validate_input_data(
                baseline_features, new_features, target, market_data
            )
            pipeline_results['input_validation'] = input_validation

            if not input_validation['data_quality_passed']:
                pipeline_results['final_recommendation'] = 'Fix data quality issues before proceeding'
                return pipeline_results

            # Phase 1: Statistical Foundation
            phase_1_results = self.run_phase_1_statistical_foundation(
                data_source, baseline_features, new_features, target
            )
            pipeline_results['phase_1'] = phase_1_results
            pipeline_results['phases_completed'] = 1

            if not phase_1_results.get('phase_1_success', False):
                pipeline_results['final_recommendation'] = phase_1_results.get('recommendation', 'Phase 1 failed')
                return pipeline_results

            # Check if we should proceed to Phase 2
            if phase_1_results.get('recommendation', '').startswith('No significant improvement'):
                pipeline_results['final_recommendation'] = 'Stop: No statistical significance detected'
                return pipeline_results

            # Phase 2: Market Validation (if market data available)
            if market_data and all(key in market_data for key in ['predictions', 'market_lines', 'outcomes']):
                phase_2_results = self.run_phase_2_market_validation(
                    data_source,
                    market_data['predictions'],
                    market_data['market_lines'],
                    market_data['outcomes'],
                    supplementary_data
                )
                pipeline_results['phase_2'] = phase_2_results
                pipeline_results['phases_completed'] = 2

                if not phase_2_results.get('phase_2_success', False):
                    pipeline_results['final_recommendation'] = phase_2_results.get('recommendation', 'Phase 2 failed')
                    return pipeline_results

                # Check market exploitability
                if not phase_2_results.get('market_efficiency', {}).get('exploitable', False):
                    pipeline_results['final_recommendation'] = 'Stop: No exploitable market edge detected'
                    return pipeline_results

            # Phase 3: Temporal Analysis (if historical data available)
            if feature_history and 'feature_importance_by_season' in feature_history:
                phase_3_results = self.run_phase_3_temporal_analysis(
                    feature_history['feature_importance_by_season'],
                    feature_history.get('feature_performance_by_week')
                )
                pipeline_results['phase_3'] = phase_3_results
                pipeline_results['phases_completed'] = 3

                if not phase_3_results.get('phase_3_success', False):
                    pipeline_results['final_recommendation'] = phase_3_results.get('recommendation', 'Phase 3 failed')
                    return pipeline_results

                # Check temporal stability
                reliable_features = phase_3_results.get('reliable_features', [])
                if len(reliable_features) == 0:
                    pipeline_results['final_recommendation'] = 'Stop: No temporally stable features detected'
                    return pipeline_results

            # Phase 4: Implementation Strategy
            if pipeline_results['phases_completed'] >= 2:
                phase_4_results = self.run_phase_4_implementation_strategy(
                    pipeline_results.get('phase_1', {}),
                    pipeline_results.get('phase_2', {}),
                    pipeline_results.get('phase_3', {})
                )
                pipeline_results['phase_4'] = phase_4_results
                pipeline_results['phases_completed'] = 4

                if phase_4_results.get('phase_4_success', False):
                    pipeline_results['final_recommendation'] = phase_4_results.get('recommendation', 'Proceed with implementation')

            # Phase 5: Monitoring Setup (if ongoing monitoring data available)
            if feature_history and 'performance_history' in feature_history:
                phase_5_results = self.run_phase_5_monitoring(
                    feature_history['performance_history'],
                    feature_history.get('recent_performance'),
                    feature_history.get('correlation_data')
                )
                pipeline_results['phase_5'] = phase_5_results
                pipeline_results['phases_completed'] = 5

            pipeline_results['pipeline_success'] = True
            pipeline_results['pipeline_end_time'] = datetime.now()
            pipeline_results['total_runtime'] = (pipeline_results['pipeline_end_time'] - pipeline_start_time).total_seconds()

        except Exception as e:
            self.logger.error(f"Pipeline failed for {data_source}: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['final_recommendation'] = f'Pipeline error: {str(e)}'

        # Save complete results
        if self.config['save_intermediate_results']:
            self._save_complete_results(data_source, pipeline_results)

        # Add to validation history
        self.validation_history.append({
            'data_source': data_source,
            'timestamp': pipeline_start_time,
            'success': pipeline_results['pipeline_success'],
            'phases_completed': pipeline_results['phases_completed'],
            'recommendation': pipeline_results['final_recommendation']
        })

        return pipeline_results

    def _save_intermediate_results(self, phase: str, data_source: str, results: Dict[str, Any]):
        """Save intermediate results to disk"""
        try:
            output_dir = self.config['output_directory']
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{phase}_{data_source}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            # Convert numpy arrays and pandas objects to serializable format
            serializable_results = self._make_json_serializable(results)

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            self.logger.info(f"Saved {phase} results to {filepath}")

        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {str(e)}")

    def _save_complete_results(self, data_source: str, results: Dict[str, Any]):
        """Save complete pipeline results"""
        try:
            output_dir = self.config['output_directory']
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"complete_validation_{data_source}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            serializable_results = self._make_json_serializable(results)

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            self.logger.info(f"Saved complete validation results to {filepath}")

        except Exception as e:
            self.logger.warning(f"Failed to save complete results: {str(e)}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and pandas objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation runs"""
        return {
            'total_validations': len(self.validation_history),
            'successful_validations': sum(1 for v in self.validation_history if v['success']),
            'validation_history': self.validation_history,
            'framework_config': self.config
        }

    def get_implementation_readiness_report(self, recent_validations: int = 5) -> Dict[str, Any]:
        """Generate implementation readiness report"""
        recent_history = self.validation_history[-recent_validations:] if self.validation_history else []

        successful_features = [v['data_source'] for v in recent_history if v['success']]
        failed_features = [v['data_source'] for v in recent_history if not v['success']]

        return {
            'ready_for_implementation': successful_features,
            'requires_more_work': failed_features,
            'success_rate': len(successful_features) / max(len(recent_history), 1),
            'recommendation': (
                'High confidence in feature pipeline' if len(successful_features) > len(failed_features)
                else 'Focus on data quality improvements'
            )
        }