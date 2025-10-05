"""
Monitoring Module for NFL Betting Model
PSI drift detection, performance monitoring, and alerting system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Additional imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    # PSI thresholds
    psi_no_action_threshold: float = 0.1
    psi_warning_threshold: float = 0.2
    psi_critical_threshold: float = 0.3
    
    # Performance thresholds
    performance_degradation_threshold: float = 0.1  # 10% degradation
    roi_alert_threshold: float = -0.05  # -5% ROI
    
    # Monitoring windows
    short_window: int = 7  # days
    medium_window: int = 30  # days
    long_window: int = 90  # days
    
    # Alert settings
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ['log', 'email'])
    alert_cooldown_hours: int = 24
    
    # Reporting
    report_frequency: str = 'daily'  # 'hourly', 'daily', 'weekly'
    save_reports: bool = True
    report_path: str = 'monitoring_reports'
    
    # Feature monitoring
    n_bins_psi: int = 10
    features_to_monitor: Optional[List[str]] = None
    
    # Model monitoring
    track_feature_importance: bool = True
    track_prediction_distribution: bool = True
    track_calibration: bool = True


class PSICalculator:
    """Population Stability Index calculator for drift detection"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.baseline_distributions = {}
        self.psi_history = deque(maxlen=1000)
        
    def set_baseline(self, X: pd.DataFrame, features: Optional[List[str]] = None):
        """Set baseline distributions for PSI calculation"""
        features = features or X.columns.tolist()
        
        for feature in features:
            if feature in X.columns:
                self.baseline_distributions[feature] = self._compute_distribution(
                    X[feature].values
                )
        
        logger.info(f"Baseline set for {len(self.baseline_distributions)} features")
    
    def calculate_psi(self, X_current: pd.DataFrame) -> Dict[str, float]:
        """Calculate PSI for all monitored features"""
        psi_scores = {}
        
        for feature, baseline_dist in self.baseline_distributions.items():
            if feature in X_current.columns:
                current_dist = self._compute_distribution(X_current[feature].values)
                psi = self._compute_psi_score(baseline_dist, current_dist)
                psi_scores[feature] = psi
        
        # Store in history
        self.psi_history.append({
            'timestamp': datetime.now(),
            'psi_scores': psi_scores,
            'overall_psi': np.mean(list(psi_scores.values()))
        })
        
        return psi_scores
    
    def _compute_distribution(self, values: np.ndarray) -> np.ndarray:
        """Compute binned distribution for PSI calculation"""
        # Handle numeric features
        if np.issubdtype(values.dtype, np.number):
            # Remove NaN values
            values = values[~np.isnan(values)]
            
            # Create bins
            if len(np.unique(values)) > self.config.n_bins_psi:
                _, bin_edges = np.histogram(values, bins=self.config.n_bins_psi)
                distribution, _ = np.histogram(values, bins=bin_edges)
            else:
                # For categorical or low-cardinality features
                unique_vals, counts = np.unique(values, return_counts=True)
                distribution = counts
            
            # Normalize to probabilities
            distribution = distribution / distribution.sum()
            
            # Add small epsilon to avoid log(0)
            distribution = distribution + 1e-10
            
            return distribution
        else:
            # Handle categorical features
            unique_vals, counts = np.unique(values, return_counts=True)
            distribution = counts / counts.sum() + 1e-10
            return distribution
    
    def _compute_psi_score(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Compute PSI score between two distributions"""
        # Ensure same length
        if len(baseline) != len(current):
            # Pad shorter array with small values
            max_len = max(len(baseline), len(current))
            baseline = np.pad(baseline, (0, max_len - len(baseline)), constant_values=1e-10)
            current = np.pad(current, (0, max_len - len(current)), constant_values=1e-10)
        
        # Calculate PSI
        psi = np.sum((current - baseline) * np.log(current / baseline))
        
        return psi
    
    def get_drift_status(self, psi_scores: Dict[str, float]) -> Dict[str, str]:
        """Determine drift status based on PSI scores"""
        status = {}
        
        for feature, psi in psi_scores.items():
            if psi < self.config.psi_no_action_threshold:
                status[feature] = 'stable'
            elif psi < self.config.psi_warning_threshold:
                status[feature] = 'warning'
            elif psi < self.config.psi_critical_threshold:
                status[feature] = 'alert'
            else:
                status[feature] = 'critical'
        
        return status
    
    def get_summary(self) -> Dict[str, Any]:
        """Get PSI summary statistics"""
        if not self.psi_history:
            return {}
        
        recent_psi = [h['overall_psi'] for h in list(self.psi_history)[-10:]]
        
        return {
            'current_overall_psi': self.psi_history[-1]['overall_psi'],
            'mean_recent_psi': np.mean(recent_psi),
            'max_recent_psi': np.max(recent_psi),
            'trending_up': recent_psi[-1] > recent_psi[0] if len(recent_psi) > 1 else False,
            'features_drifting': sum(1 for f, psi in self.psi_history[-1]['psi_scores'].items() 
                                    if psi > self.config.psi_warning_threshold)
        }


class PerformanceMonitor:
    """Monitor model performance metrics over time"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = deque(maxlen=10000)
        self.baseline_metrics = {}
        self.alerts_sent = {}
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, 
              y_pred_proba: Optional[np.ndarray] = None,
              metadata: Optional[Dict] = None):
        """Update performance metrics with new predictions"""
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Add metadata
        metrics['timestamp'] = datetime.now()
        if metadata:
            metrics.update(metadata)
        
        self.metrics_history.append(metrics)
        
        # Check for performance degradation
        if self.baseline_metrics:
            self._check_performance_degradation(metrics)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            # Calibration metrics
            metrics['calibration_error'] = self._calculate_calibration_error(
                y_true, y_pred_proba
            )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        return metrics
    
    def _calculate_calibration_error(self, y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray, 
                                    n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def set_baseline(self, metrics: Optional[Dict[str, float]] = None):
        """Set baseline performance metrics"""
        if metrics:
            self.baseline_metrics = metrics
        elif len(self.metrics_history) > 0:
            # Use recent average as baseline
            recent_metrics = list(self.metrics_history)[-100:]
            self.baseline_metrics = {
                key: np.mean([m[key] for m in recent_metrics if key in m])
                for key in recent_metrics[0].keys()
                if isinstance(recent_metrics[0][key], (int, float))
            }
        
        logger.info(f"Baseline metrics set: {self.baseline_metrics}")
    
    def _check_performance_degradation(self, current_metrics: Dict[str, float]):
        """Check for performance degradation and send alerts"""
        for metric_name in ['accuracy', 'auc_roc', 'f1']:
            if metric_name in current_metrics and metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                current_value = current_metrics[metric_name]
                
                degradation = (baseline_value - current_value) / baseline_value
                
                if degradation > self.config.performance_degradation_threshold:
                    self._send_alert(
                        f"Performance degradation detected: {metric_name} "
                        f"dropped by {degradation:.1%} from baseline"
                    )
    
    def _send_alert(self, message: str, severity: str = 'warning'):
        """Send alert through configured channels"""
        if not self.config.enable_alerts:
            return
        
        # Check cooldown
        alert_key = f"{message[:50]}_{severity}"
        if alert_key in self.alerts_sent:
            last_sent = self.alerts_sent[alert_key]
            if (datetime.now() - last_sent).total_seconds() < self.config.alert_cooldown_hours * 3600:
                return
        
        # Send alerts
        if 'log' in self.config.alert_channels:
            if severity == 'critical':
                logger.error(f"ALERT: {message}")
            elif severity == 'warning':
                logger.warning(f"ALERT: {message}")
            else:
                logger.info(f"ALERT: {message}")
        
        if 'email' in self.config.alert_channels:
            # Placeholder for email notification
            pass
        
        # Update sent time
        self.alerts_sent[alert_key] = datetime.now()
    
    def get_summary(self, window_days: Optional[int] = None) -> Dict[str, Any]:
        """Get performance summary for specified window"""
        if not self.metrics_history:
            return {}
        
        if window_days:
            cutoff = datetime.now() - timedelta(days=window_days)
            recent_metrics = [m for m in self.metrics_history 
                            if m['timestamp'] > cutoff]
        else:
            recent_metrics = list(self.metrics_history)[-100:]
        
        if not recent_metrics:
            return {}
        
        summary = {}
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            values = [m[key] for m in recent_metrics if key in m]
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
        
        return summary


class BettingPerformanceMonitor:
    """Monitor betting-specific performance metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.betting_history = deque(maxlen=10000)
        self.bankroll_history = deque(maxlen=10000)
        self.current_bankroll = 10000  # Starting bankroll
        
    def update(self, bet_data: pd.DataFrame):
        """Update with betting results"""
        for _, bet in bet_data.iterrows():
            # Calculate profit/loss
            if bet['result'] == 'win':
                profit = bet['stake'] * (bet['odds'] - 1)
            else:
                profit = -bet['stake']
            
            self.current_bankroll += profit
            
            # Store bet information
            self.betting_history.append({
                'timestamp': datetime.now(),
                'game_id': bet.get('game_id'),
                'stake': bet['stake'],
                'odds': bet['odds'],
                'probability': bet.get('probability'),
                'result': bet['result'],
                'profit': profit,
                'bankroll': self.current_bankroll,
                'roi': profit / bet['stake']
            })
            
            # Store bankroll snapshot
            self.bankroll_history.append({
                'timestamp': datetime.now(),
                'bankroll': self.current_bankroll
            })
        
        # Check for alerts
        self._check_betting_alerts()
    
    def _check_betting_alerts(self):
        """Check for betting performance alerts"""
        # Check recent ROI
        if len(self.betting_history) >= 20:
            recent_bets = list(self.betting_history)[-20:]
            recent_roi = sum(bet['profit'] for bet in recent_bets) / sum(bet['stake'] for bet in recent_bets)
            
            if recent_roi < self.config.roi_alert_threshold:
                logger.warning(f"Poor betting performance: Recent ROI = {recent_roi:.1%}")
        
        # Check bankroll drawdown
        if len(self.bankroll_history) >= 50:
            recent_bankrolls = [b['bankroll'] for b in list(self.bankroll_history)[-50:]]
            max_bankroll = max(recent_bankrolls)
            current = recent_bankrolls[-1]
            drawdown = (max_bankroll - current) / max_bankroll
            
            if drawdown > 0.2:  # 20% drawdown
                logger.warning(f"Significant drawdown: {drawdown:.1%} from peak")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get betting performance summary"""
        if not self.betting_history:
            return {}
        
        all_bets = list(self.betting_history)
        total_stakes = sum(bet['stake'] for bet in all_bets)
        total_profit = sum(bet['profit'] for bet in all_bets)
        
        wins = [bet for bet in all_bets if bet['result'] == 'win']
        
        return {
            'total_bets': len(all_bets),
            'total_stakes': total_stakes,
            'total_profit': total_profit,
            'overall_roi': total_profit / total_stakes if total_stakes > 0 else 0,
            'win_rate': len(wins) / len(all_bets) if all_bets else 0,
            'average_odds': np.mean([bet['odds'] for bet in all_bets]),
            'current_bankroll': self.current_bankroll,
            'peak_bankroll': max(b['bankroll'] for b in self.bankroll_history) if self.bankroll_history else self.current_bankroll,
            'max_drawdown': self._calculate_max_drawdown()
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.bankroll_history) < 2:
            return 0.0
        
        bankrolls = [b['bankroll'] for b in self.bankroll_history]
        peak = bankrolls[0]
        max_dd = 0
        
        for bankroll in bankrolls[1:]:
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Generate performance visualization"""
        if not self.bankroll_history:
            logger.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Bankroll over time
        timestamps = [b['timestamp'] for b in self.bankroll_history]
        bankrolls = [b['bankroll'] for b in self.bankroll_history]
        
        axes[0, 0].plot(timestamps, bankrolls)
        axes[0, 0].set_title('Bankroll Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Bankroll ($)')
        axes[0, 0].grid(True)
        
        # ROI distribution
        rois = [bet['roi'] for bet in self.betting_history]
        axes[0, 1].hist(rois, bins=50, edgecolor='black')
        axes[0, 1].set_title('ROI Distribution')
        axes[0, 1].set_xlabel('ROI')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        
        # Win rate over time (rolling)
        if len(self.betting_history) > 20:
            window = 20
            win_rates = []
            for i in range(window, len(self.betting_history)):
                window_bets = list(self.betting_history)[i-window:i]
                wins = sum(1 for bet in window_bets if bet['result'] == 'win')
                win_rates.append(wins / window)
            
            axes[1, 0].plot(range(len(win_rates)), win_rates)
            axes[1, 0].set_title(f'Rolling Win Rate (Window={window})')
            axes[1, 0].set_xlabel('Bet Number')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].axhline(y=0.5, color='r', linestyle='--')
            axes[1, 0].grid(True)
        
        # Profit by odds range
        odds_ranges = [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, float('inf'))]
        profits_by_range = []
        labels = []
        
        for low, high in odds_ranges:
            range_bets = [bet for bet in self.betting_history 
                         if low <= bet['odds'] < high]
            if range_bets:
                total_profit = sum(bet['profit'] for bet in range_bets)
                profits_by_range.append(total_profit)
                labels.append(f'{low:.1f}-{high:.1f}' if high != float('inf') else f'{low:.1f}+')
        
        if profits_by_range:
            axes[1, 1].bar(labels, profits_by_range)
            axes[1, 1].set_title('Profit by Odds Range')
            axes[1, 1].set_xlabel('Odds Range')
            axes[1, 1].set_ylabel('Total Profit ($)')
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            logger.info(f"Performance plot saved to {save_path}")
        else:
            plt.show()


class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.psi_calculator = PSICalculator(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.betting_monitor = BettingPerformanceMonitor(self.config)
        self.monitoring_active = False
        self.last_report_time = datetime.now()
        
    def initialize(self, baseline_data: pd.DataFrame, 
                  baseline_predictions: Optional[np.ndarray] = None,
                  baseline_actuals: Optional[np.ndarray] = None):
        """Initialize monitoring with baseline data"""
        # Set PSI baseline
        features_to_monitor = self.config.features_to_monitor or baseline_data.columns.tolist()
        self.psi_calculator.set_baseline(baseline_data, features_to_monitor)
        
        # Set performance baseline if provided
        if baseline_predictions is not None and baseline_actuals is not None:
            self.performance_monitor.update(
                baseline_actuals, 
                (baseline_predictions > 0.5).astype(int),
                baseline_predictions
            )
            self.performance_monitor.set_baseline()
        
        self.monitoring_active = True
        logger.info("Model monitoring initialized")
    
    def update(self, X: pd.DataFrame, 
              predictions: np.ndarray,
              actuals: Optional[np.ndarray] = None,
              bet_results: Optional[pd.DataFrame] = None):
        """Update monitoring with new data"""
        if not self.monitoring_active:
            logger.warning("Monitoring not initialized. Call initialize() first.")
            return
        
        # Calculate PSI
        psi_scores = self.psi_calculator.calculate_psi(X)
        drift_status = self.psi_calculator.get_drift_status(psi_scores)
        
        # Log drift warnings
        for feature, status in drift_status.items():
            if status in ['alert', 'critical']:
                logger.warning(f"Drift detected in {feature}: PSI = {psi_scores[feature]:.3f} ({status})")
        
        # Update performance if actuals available
        if actuals is not None:
            self.performance_monitor.update(
                actuals,
                (predictions > 0.5).astype(int),
                predictions
            )
        
        # Update betting performance
        if bet_results is not None:
            self.betting_monitor.update(bet_results)
        
        # Generate report if needed
        if self._should_generate_report():
            self.generate_report()
    
    def _should_generate_report(self) -> bool:
        """Check if report should be generated"""
        if self.config.report_frequency == 'hourly':
            return (datetime.now() - self.last_report_time).seconds > 3600
        elif self.config.report_frequency == 'daily':
            return (datetime.now() - self.last_report_time).days >= 1
        elif self.config.report_frequency == 'weekly':
            return (datetime.now() - self.last_report_time).days >= 7
        return False
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'psi_summary': self.psi_calculator.get_summary(),
            'performance_summary': {
                'short_window': self.performance_monitor.get_summary(self.config.short_window),
                'medium_window': self.performance_monitor.get_summary(self.config.medium_window),
                'long_window': self.performance_monitor.get_summary(self.config.long_window)
            },
            'betting_summary': self.betting_monitor.get_summary()
        }
        
        # Add drift details
        if self.psi_calculator.psi_history:
            latest_psi = self.psi_calculator.psi_history[-1]
            report['drift_details'] = {
                'features_stable': sum(1 for f, psi in latest_psi['psi_scores'].items() 
                                      if psi < self.config.psi_no_action_threshold),
                'features_warning': sum(1 for f, psi in latest_psi['psi_scores'].items() 
                                       if self.config.psi_no_action_threshold <= psi < self.config.psi_warning_threshold),
                'features_critical': sum(1 for f, psi in latest_psi['psi_scores'].items() 
                                        if psi >= self.config.psi_critical_threshold)
            }
        
        # Save report if configured
        if self.config.save_reports:
            save_path = save_path or self.config.report_path
            self._save_report(report, save_path)
        
        self.last_report_time = datetime.now()
        logger.info("Monitoring report generated")
        
        return report
    
    def _save_report(self, report: Dict[str, Any], save_path: str):
        """Save report to disk"""
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        
        filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path / filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {path / filename}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'psi_summary': self.psi_calculator.get_summary(),
            'recent_performance': self.performance_monitor.get_summary(window_days=1),
            'betting_roi': self.betting_monitor.get_summary().get('overall_roi', 0),
            'alerts_enabled': self.config.enable_alerts
        }
    
    def create_dashboard(self, save_path: Optional[str] = None):
        """Create monitoring dashboard visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        # PSI trends
        ax1 = plt.subplot(2, 3, 1)
        if self.psi_calculator.psi_history:
            psi_values = [h['overall_psi'] for h in list(self.psi_calculator.psi_history)[-100:]]
            ax1.plot(psi_values)
            ax1.axhline(y=self.config.psi_warning_threshold, color='orange', linestyle='--', label='Warning')
            ax1.axhline(y=self.config.psi_critical_threshold, color='red', linestyle='--', label='Critical')
            ax1.set_title('PSI Trend')
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('PSI Score')
            ax1.legend()
            ax1.grid(True)
        
        # Performance metrics
        ax2 = plt.subplot(2, 3, 2)
        if self.performance_monitor.metrics_history:
            recent_metrics = list(self.performance_monitor.metrics_history)[-100:]
            metrics_to_plot = ['accuracy', 'precision', 'recall']
            for metric in metrics_to_plot:
                values = [m[metric] for m in recent_metrics if metric in m]
                if values:
                    ax2.plot(values, label=metric)
            ax2.set_title('Performance Metrics')
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Score')
            ax2.legend()
            ax2.grid(True)
        
        # ROI trend
        ax3 = plt.subplot(2, 3, 3)
        if self.betting_monitor.betting_history:
            cumulative_roi = []
            total_stake = 0
            total_profit = 0
            for bet in self.betting_monitor.betting_history:
                total_stake += bet['stake']
                total_profit += bet['profit']
                if total_stake > 0:
                    cumulative_roi.append(total_profit / total_stake)
            
            ax3.plot(cumulative_roi)
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_title('Cumulative ROI')
            ax3.set_xlabel('Bet Number')
            ax3.set_ylabel('ROI')
            ax3.grid(True)
        
        # Feature drift heatmap
        ax4 = plt.subplot(2, 3, 4)
        if self.psi_calculator.psi_history:
            # Get last 20 PSI calculations for top features
            recent_psi = list(self.psi_calculator.psi_history)[-20:]
            if recent_psi:
                features = list(recent_psi[-1]['psi_scores'].keys())[:10]  # Top 10 features
                psi_matrix = []
                for record in recent_psi:
                    row = [record['psi_scores'].get(f, 0) for f in features]
                    psi_matrix.append(row)
                
                im = ax4.imshow(np.array(psi_matrix).T, aspect='auto', cmap='YlOrRd')
                ax4.set_yticks(range(len(features)))
                ax4.set_yticklabels(features)
                ax4.set_xlabel('Time')
                ax4.set_title('Feature Drift Heatmap')
                plt.colorbar(im, ax=ax4)
        
        # Calibration plot
        ax5 = plt.subplot(2, 3, 5)
        if self.performance_monitor.metrics_history:
            recent_metrics = list(self.performance_monitor.metrics_history)[-100:]
            cal_errors = [m.get('calibration_error', 0) for m in recent_metrics]
            if cal_errors:
                ax5.plot(cal_errors)
                ax5.set_title('Calibration Error')
                ax5.set_xlabel('Sample')
                ax5.set_ylabel('ECE')
                ax5.grid(True)
        
        # Win rate by confidence
        ax6 = plt.subplot(2, 3, 6)
        if self.betting_monitor.betting_history:
            bets_with_prob = [b for b in self.betting_monitor.betting_history if 'probability' in b]
            if bets_with_prob:
                prob_bins = np.linspace(0.5, 1.0, 6)
                win_rates_by_conf = []
                conf_labels = []
                
                for i in range(len(prob_bins) - 1):
                    bin_bets = [b for b in bets_with_prob 
                               if prob_bins[i] <= b['probability'] < prob_bins[i+1]]
                    if bin_bets:
                        win_rate = sum(1 for b in bin_bets if b['result'] == 'win') / len(bin_bets)
                        win_rates_by_conf.append(win_rate)
                        conf_labels.append(f'{prob_bins[i]:.2f}-{prob_bins[i+1]:.2f}')
                
                if win_rates_by_conf:
                    ax6.bar(conf_labels, win_rates_by_conf)
                    ax6.set_title('Win Rate by Confidence')
                    ax6.set_xlabel('Probability Range')
                    ax6.set_ylabel('Win Rate')
                    ax6.axhline(y=0.5, color='red', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            logger.info(f"Dashboard saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create configuration
    config = MonitoringConfig(
        psi_warning_threshold=0.15,
        psi_critical_threshold=0.25,
        enable_alerts=True,
        report_frequency='daily'
    )
    
    # Initialize monitor
    monitor = ModelMonitor(config)
    
    # Generate synthetic baseline data
    n_samples = 1000
    n_features = 30
    
    baseline_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    baseline_predictions = np.random.uniform(0, 1, n_samples)
    baseline_actuals = (baseline_predictions + np.random.randn(n_samples) * 0.2 > 0.5).astype(int)
    
    # Initialize monitoring
    monitor.initialize(baseline_data, baseline_predictions, baseline_actuals)
    
    print("Monitoring initialized!")
    
    # Simulate production monitoring
    print("\nSimulating production monitoring...")
    
    for day in range(5):
        print(f"\nDay {day + 1}:")
        
        # Generate new data with slight drift
        drift_factor = 0.1 * (day + 1)
        new_data = pd.DataFrame(
            np.random.randn(100, n_features) + drift_factor,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        predictions = np.random.uniform(0, 1, 100)
        actuals = (predictions + np.random.randn(100) * 0.3 > 0.5).astype(int)
        
        # Create betting results
        bet_results = pd.DataFrame({
            'game_id': range(10),
            'stake': np.random.uniform(50, 200, 10),
            'odds': np.random.uniform(1.8, 2.5, 10),
            'probability': np.random.uniform(0.4, 0.7, 10),
            'result': np.random.choice(['win', 'loss'], 10, p=[0.52, 0.48])
        })
        
        # Update monitoring
        monitor.update(new_data, predictions, actuals, bet_results)
        
        # Get status
        status = monitor.get_status()
        print(f"Overall PSI: {status['psi_summary'].get('current_overall_psi', 0):.3f}")
        print(f"Recent Accuracy: {status['recent_performance'].get('accuracy_mean', 0):.3f}")
        print(f"Betting ROI: {status['betting_roi']:.3%}")
    
    # Generate report
    report = monitor.generate_report()
    print("\nMonitoring Report Generated:")
    print(f"Features with drift warning: {report['drift_details'].get('features_warning', 0)}")
    print(f"Features with critical drift: {report['drift_details'].get('features_critical', 0)}")
    
    # Create dashboard
    monitor.create_dashboard(save_path='monitoring_dashboard.png')
    print("\nMonitoring dashboard created!")
    
    # Get betting performance summary
    betting_summary = monitor.betting_monitor.get_summary()
    print("\nBetting Performance Summary:")
    for key, value in betting_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
