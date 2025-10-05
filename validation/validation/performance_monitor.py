"""
Phase 5: Continuous Monitoring Framework
Performance decay detection and ongoing feature health monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    NORMAL = "normal"

@dataclass
class PerformanceAlert:
    feature: str
    alert_level: AlertLevel
    metric: str
    current_value: float
    threshold_value: float
    recommended_action: str
    timestamp: datetime

class PerformanceMonitor:
    """
    Continuous monitoring framework to detect when features lose predictive power
    and trigger appropriate responses
    """

    def __init__(self, rolling_window: int = 30):
        self.rolling_window = rolling_window
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {
            'critical_decay_rate': 0.2,     # 20% performance drop
            'warning_decay_rate': 0.1,      # 10% performance drop
            'min_sample_size': 20,          # Minimum for reliable analysis
            'volatility_threshold': 2.0,    # Standard deviations
            'correlation_threshold': 0.3     # Minimum correlation with outcomes
        }

    def calculate_performance_metrics(self, performance_data: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for a feature"""

        if len(performance_data) < self.alert_thresholds['min_sample_size']:
            return {'error': f'Insufficient data: {len(performance_data)} samples'}

        metrics = {}

        # Basic statistics
        metrics['mean_performance'] = performance_data.mean()
        metrics['std_performance'] = performance_data.std()
        metrics['current_performance'] = performance_data.iloc[-1] if len(performance_data) > 0 else 0

        # Rolling performance analysis
        if len(performance_data) >= self.rolling_window:
            recent_performance = performance_data[-self.rolling_window:].mean()
            historical_performance = performance_data[:-self.rolling_window].mean()

            if historical_performance != 0:
                metrics['decay_rate'] = (historical_performance - recent_performance) / historical_performance
            else:
                metrics['decay_rate'] = 0.0

            # Rolling volatility
            rolling_std = performance_data.rolling(window=min(self.rolling_window, len(performance_data))).std()
            metrics['volatility_trend'] = rolling_std.iloc[-1] - rolling_std.mean()

        else:
            # Use available data for smaller samples
            window_size = max(5, len(performance_data) // 2)
            recent_performance = performance_data[-window_size:].mean()
            historical_performance = performance_data[:-window_size].mean()

            if historical_performance != 0:
                metrics['decay_rate'] = (historical_performance - recent_performance) / historical_performance
            else:
                metrics['decay_rate'] = 0.0

            metrics['volatility_trend'] = 0.0

        # Trend analysis
        if len(performance_data) > 2:
            time_index = np.arange(len(performance_data))
            correlation = np.corrcoef(time_index, performance_data)[0, 1]
            metrics['trend_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['trend_correlation'] = 0.0

        # Performance consistency
        if metrics['std_performance'] > 0:
            metrics['consistency_score'] = 1 - (metrics['std_performance'] / abs(metrics['mean_performance']))
        else:
            metrics['consistency_score'] = 1.0

        metrics['consistency_score'] = max(0, min(1, metrics['consistency_score']))

        return metrics

    def monitor_feature_performance_decay(self, feature_performance_history: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Detect when features lose predictive power"""

        decay_alerts = {}

        for feature, performance_data in feature_performance_history.items():

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(performance_data)

            if 'error' in metrics:
                decay_alerts[feature] = {
                    'status': 'insufficient_data',
                    'message': metrics['error'],
                    'sample_size': len(performance_data)
                }
                continue

            # Determine alert level
            decay_rate = metrics['decay_rate']

            if decay_rate > self.alert_thresholds['critical_decay_rate']:
                alert_level = AlertLevel.CRITICAL
                action = 'investigate_immediately'
            elif decay_rate > self.alert_thresholds['warning_decay_rate']:
                alert_level = AlertLevel.WARNING
                action = 'monitor_closely'
            else:
                alert_level = AlertLevel.NORMAL
                action = 'continue_monitoring'

            # Additional checks for volatility and correlation
            if (metrics.get('volatility_trend', 0) > self.alert_thresholds['volatility_threshold'] and
                alert_level == AlertLevel.NORMAL):
                alert_level = AlertLevel.WARNING
                action = 'investigate_volatility'

            if (abs(metrics.get('trend_correlation', 0)) < self.alert_thresholds['correlation_threshold'] and
                alert_level != AlertLevel.CRITICAL):
                alert_level = AlertLevel.WARNING
                action = 'check_feature_relevance'

            decay_alerts[feature] = {
                'alert_level': alert_level.value,
                'decay_rate': decay_rate,
                'recent_performance': metrics.get('current_performance', 0),
                'historical_baseline': metrics.get('mean_performance', 0),
                'volatility_trend': metrics.get('volatility_trend', 0),
                'trend_correlation': metrics.get('trend_correlation', 0),
                'consistency_score': metrics.get('consistency_score', 0),
                'recommended_action': action,
                'sample_size': len(performance_data),
                'last_updated': datetime.now()
            }

        return decay_alerts

    def detect_feature_anomalies(self, recent_performance: Dict[str, float],
                                historical_baselines: Dict[str, float]) -> List[PerformanceAlert]:
        """Detect anomalous feature performance patterns"""

        alerts = []

        for feature in recent_performance:
            if feature not in historical_baselines:
                continue

            recent_value = recent_performance[feature]
            baseline_value = historical_baselines[feature]

            # Z-score based anomaly detection
            if baseline_value != 0:
                deviation = abs((recent_value - baseline_value) / baseline_value)

                if deviation > 0.5:  # 50% deviation threshold
                    alert_level = AlertLevel.CRITICAL
                    action = 'immediate_investigation_required'
                elif deviation > 0.25:  # 25% deviation threshold
                    alert_level = AlertLevel.WARNING
                    action = 'monitor_next_few_games'
                else:
                    continue  # No alert needed

                alert = PerformanceAlert(
                    feature=feature,
                    alert_level=alert_level,
                    metric='performance_deviation',
                    current_value=recent_value,
                    threshold_value=baseline_value,
                    recommended_action=action,
                    timestamp=datetime.now()
                )
                alerts.append(alert)

        return alerts

    def analyze_feature_correlation_degradation(self, feature_correlations: pd.DataFrame,
                                              target_variable: str) -> Dict[str, Any]:
        """Analyze degradation in feature correlation with target variable"""

        correlation_analysis = {}

        if target_variable not in feature_correlations.columns:
            return {'error': f'Target variable {target_variable} not found in correlation data'}

        target_correlations = feature_correlations[target_variable].drop(target_variable, errors='ignore')

        for feature in target_correlations.index:
            correlation_value = target_correlations[feature]

            # Historical correlation analysis (if available)
            if len(feature_correlations) > 1:
                historical_correlation = feature_correlations[target_variable].iloc[:-1].mean()
                correlation_degradation = historical_correlation - correlation_value
            else:
                correlation_degradation = 0.0

            # Determine significance
            if abs(correlation_value) < 0.1:
                significance = 'very_weak'
                recommendation = 'consider_removing'
            elif abs(correlation_value) < 0.3:
                significance = 'weak'
                recommendation = 'monitor_closely'
            elif abs(correlation_value) < 0.5:
                significance = 'moderate'
                recommendation = 'maintain'
            else:
                significance = 'strong'
                recommendation = 'prioritize'

            correlation_analysis[feature] = {
                'current_correlation': correlation_value,
                'correlation_degradation': correlation_degradation,
                'significance_level': significance,
                'recommendation': recommendation,
                'abs_correlation': abs(correlation_value)
            }

        return correlation_analysis

    def generate_health_dashboard(self, feature_performance_history: Dict[str, pd.Series],
                                recent_performance: Dict[str, float] = None,
                                correlation_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive feature health dashboard"""

        dashboard = {
            'timestamp': datetime.now(),
            'monitoring_window': self.rolling_window,
            'features_monitored': len(feature_performance_history)
        }

        # Performance decay analysis
        decay_analysis = self.monitor_feature_performance_decay(feature_performance_history)
        dashboard['decay_analysis'] = decay_analysis

        # Count alerts by level
        alert_counts = {'critical': 0, 'warning': 0, 'normal': 0}
        for feature_analysis in decay_analysis.values():
            if 'alert_level' in feature_analysis:
                alert_counts[feature_analysis['alert_level']] += 1

        dashboard['alert_summary'] = alert_counts

        # Anomaly detection (if recent performance provided)
        if recent_performance:
            historical_baselines = {
                feature: analysis.get('historical_baseline', 0)
                for feature, analysis in decay_analysis.items()
                if 'historical_baseline' in analysis
            }

            anomaly_alerts = self.detect_feature_anomalies(recent_performance, historical_baselines)
            dashboard['anomaly_alerts'] = [
                {
                    'feature': alert.feature,
                    'alert_level': alert.alert_level.value,
                    'deviation': abs(alert.current_value - alert.threshold_value) / alert.threshold_value,
                    'action': alert.recommended_action
                }
                for alert in anomaly_alerts
            ]
        else:
            dashboard['anomaly_alerts'] = []

        # Correlation analysis (if correlation data provided)
        if correlation_data is not None:
            # Assume first column is target variable or use 'outcome' as default
            target_var = correlation_data.columns[0] if len(correlation_data.columns) > 1 else 'outcome'
            correlation_analysis = self.analyze_feature_correlation_degradation(correlation_data, target_var)
            dashboard['correlation_analysis'] = correlation_analysis
        else:
            dashboard['correlation_analysis'] = {}

        # Overall system health score
        total_features = len(feature_performance_history)
        healthy_features = alert_counts['normal']
        warning_features = alert_counts['warning']
        critical_features = alert_counts['critical']

        if total_features > 0:
            health_score = (healthy_features * 1.0 + warning_features * 0.5 + critical_features * 0.0) / total_features
        else:
            health_score = 1.0

        dashboard['system_health_score'] = health_score
        dashboard['health_status'] = (
            'healthy' if health_score > 0.8 else
            'needs_attention' if health_score > 0.5 else
            'critical'
        )

        # Recommendations
        recommendations = []
        if critical_features > 0:
            recommendations.append(f'URGENT: {critical_features} features require immediate investigation')
        if warning_features > 0:
            recommendations.append(f'{warning_features} features need monitoring')
        if health_score < 0.6:
            recommendations.append('Consider pausing new feature deployments until health improves')
        if len(dashboard['anomaly_alerts']) > 0:
            recommendations.append(f'{len(dashboard["anomaly_alerts"])} anomalies detected in recent performance')

        dashboard['recommendations'] = recommendations

        return dashboard

    def generate_monitoring_report(self, feature_performance_history: Dict[str, pd.Series],
                                 time_period: str = 'last_30_days') -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""

        report = {
            'report_period': time_period,
            'generation_timestamp': datetime.now(),
            'features_analyzed': list(feature_performance_history.keys())
        }

        # Generate health dashboard
        dashboard = self.generate_health_dashboard(feature_performance_history)
        report['health_dashboard'] = dashboard

        # Performance trends
        performance_trends = {}
        for feature, performance_data in feature_performance_history.items():
            if len(performance_data) > 5:
                # Calculate trend over time
                time_index = np.arange(len(performance_data))
                trend_slope = np.polyfit(time_index, performance_data.values, 1)[0]

                performance_trends[feature] = {
                    'trend_slope': trend_slope,
                    'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
                    'recent_avg': performance_data[-7:].mean() if len(performance_data) >= 7 else performance_data.mean(),
                    'historical_avg': performance_data[:-7].mean() if len(performance_data) >= 14 else performance_data.mean()
                }

        report['performance_trends'] = performance_trends

        # Feature rankings by current health
        feature_rankings = []
        for feature, analysis in dashboard['decay_analysis'].items():
            if 'consistency_score' in analysis:
                ranking_score = (
                    analysis['consistency_score'] * 0.4 +
                    (1 - abs(analysis.get('decay_rate', 0))) * 0.3 +
                    abs(analysis.get('trend_correlation', 0)) * 0.3
                )

                feature_rankings.append({
                    'feature': feature,
                    'health_score': ranking_score,
                    'alert_level': analysis['alert_level'],
                    'primary_concern': analysis['recommended_action']
                })

        feature_rankings.sort(key=lambda x: x['health_score'], reverse=True)
        report['feature_health_rankings'] = feature_rankings

        # Executive summary
        total_features = len(feature_performance_history)
        healthy_count = sum(1 for r in feature_rankings if r['alert_level'] == 'normal')

        report['executive_summary'] = {
            'total_features_monitored': total_features,
            'healthy_features': healthy_count,
            'features_needing_attention': total_features - healthy_count,
            'overall_system_health': dashboard['health_status'],
            'key_recommendations': dashboard['recommendations'][:3]  # Top 3 recommendations
        }

        return report

    def run_comprehensive_monitoring(self, feature_performance_history: Dict[str, pd.Series],
                                   recent_performance: Dict[str, float] = None,
                                   correlation_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run complete Phase 5 continuous monitoring"""

        results = {
            'validation_timestamp': pd.Timestamp.now(),
            'phase': 'Phase 5: Continuous Monitoring Framework',
            'monitoring_period': f'Last {self.rolling_window} observations'
        }

        # Generate comprehensive monitoring report
        monitoring_report = self.generate_monitoring_report(feature_performance_history)
        results['monitoring_report'] = monitoring_report

        # Generate health dashboard
        health_dashboard = self.generate_health_dashboard(
            feature_performance_history, recent_performance, correlation_data
        )
        results['health_dashboard'] = health_dashboard

        # Generate recommendations based on monitoring results
        health_score = health_dashboard['system_health_score']
        critical_features = health_dashboard['alert_summary']['critical']
        warning_features = health_dashboard['alert_summary']['warning']

        if health_score > 0.8 and critical_features == 0:
            results['recommendation'] = 'System healthy: Continue normal operations with routine monitoring'
        elif health_score > 0.6:
            results['recommendation'] = f'Moderate attention needed: Monitor {warning_features + critical_features} features closely'
        else:
            results['recommendation'] = f'URGENT: System health compromised - {critical_features} critical issues require immediate action'

        return results