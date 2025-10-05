"""
NFL Betting Model Validation Framework
======================================
A comprehensive validation pipeline to prevent overfitting and ensure
real predictive power for NFL betting models.

Features:
- Time-series cross-validation with temporal gaps
- Statistical significance testing
- Regime change detection
- Betting-specific metrics
- Multiple testing correction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

# Statistical Libraries
from scipy import stats
from scipy.stats import norm, t, chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# Utilities
import json
from collections import defaultdict
import pickle
from concurrent.futures import ProcessPoolExecutor
import hashlib


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    # Time series CV parameters
    training_seasons: int = 3  # Number of seasons for training
    purge_weeks: int = 1  # Gap between train and test
    embargo_weeks: int = 2  # Embargo period after test
    min_test_games: int = 100  # Minimum games in test set
    
    # Statistical testing
    n_permutations: int = 1000  # Permutation test iterations
    bootstrap_iterations: int = 10000  # Bootstrap iterations
    confidence_level: float = 0.95  # Confidence interval level
    min_sample_size: int = 385  # For 95% confidence, 5% margin
    
    # Betting metrics
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    max_bet_size: float = 0.05  # Maximum bet as fraction of bankroll
    commission_rate: float = 0.05  # Sportsbook commission (vig)
    
    # Regime detection
    cusum_threshold: float = 3.0  # Standard deviations for CUSUM
    lookback_window: int = 100  # Games for rolling metrics
    
    # Multiple testing
    fdr_alpha: float = 0.05  # False discovery rate threshold
    n_features_tested: int = 30  # Number of features tested


class NFLTimeSeriesValidator:
    """
    Time-series cross-validation for NFL betting models with proper
    temporal handling and season boundaries.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize validator with configuration."""
        self.config = config or ValidationConfig()
        self.season_boundaries = self._define_season_boundaries()
        
    def _define_season_boundaries(self) -> Dict[int, Dict[str, pd.Timestamp]]:
        """Define NFL season boundaries from 2015-2024."""
        boundaries = {}
        for year in range(2015, 2025):
            boundaries[year] = {
                'regular_start': pd.Timestamp(f'{year}-09-01'),  # Approximate
                'regular_end': pd.Timestamp(f'{year+1}-01-10'),   # Approximate
                'playoff_start': pd.Timestamp(f'{year+1}-01-11'),
                'playoff_end': pd.Timestamp(f'{year+1}-02-15')
            }
        return boundaries
    
    def walk_forward_split(self, 
                          data: pd.DataFrame,
                          date_col: str = 'game_date') -> List[Tuple[pd.Index, pd.Index]]:
        """
        Implement walk-forward validation with proper temporal gaps.
        
        Features:
        - 3-season training windows
        - 1-week purge between train/test
        - 2-week embargo after test
        - Respects season boundaries
        """
        data = data.sort_values(date_col).copy()
        splits = []
        
        # Calculate weeks in dataset
        data['week_id'] = (data[date_col] - data[date_col].min()).dt.days // 7
        
        # Define training windows (3 seasons)
        min_training_weeks = 3 * 17  # 3 regular seasons
        
        # Start from season 4 (2018 if starting from 2015)
        start_week = min_training_weeks
        max_week = data['week_id'].max()
        
        # Generate splits with proper gaps
        current_week = start_week
        
        while current_week + self.config.purge_weeks + 10 <= max_week:  # Need room for test
            # Training set: previous 3 seasons
            train_start_week = current_week - min_training_weeks
            train_end_week = current_week
            
            # Purge period
            test_start_week = train_end_week + self.config.purge_weeks
            
            # Test set: next available games
            test_end_week = min(
                test_start_week + 4,  # 4 weeks of testing
                max_week
            )
            
            # Get indices
            train_idx = data[
                (data['week_id'] >= train_start_week) & 
                (data['week_id'] < train_end_week)
            ].index
            
            test_idx = data[
                (data['week_id'] >= test_start_week) & 
                (data['week_id'] < test_end_week)
            ].index
            
            if len(test_idx) >= self.config.min_test_games:
                splits.append((train_idx, test_idx))
            
            # Move forward with embargo
            current_week = test_end_week + self.config.embargo_weeks
        
        print(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def validate_temporal_independence(self, 
                                      predictions: pd.Series,
                                      actuals: pd.Series,
                                      dates: pd.Series) -> Dict[str, float]:
        """
        Test for temporal independence in prediction errors.
        """
        errors = predictions - actuals
        
        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(errors, lags=10, return_df=True)
        
        # Augmented Dickey-Fuller test for stationarity
        adf_test = adfuller(errors, autolag='AIC')
        
        return {
            'ljung_box_pvalue': lb_test['lb_pvalue'].min(),
            'adf_statistic': adf_test[0],
            'adf_pvalue': adf_test[1],
            'is_stationary': adf_test[1] < 0.05,
            'has_autocorrelation': lb_test['lb_pvalue'].min() < 0.05
        }


class StatisticalSignificanceTester:
    """
    Comprehensive statistical testing for betting model performance.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize tester with configuration."""
        self.config = config or ValidationConfig()
        
    def permutation_test(self,
                        predictions: np.ndarray,
                        actuals: np.ndarray,
                        metric_func: Callable,
                        n_permutations: int = None) -> Dict[str, float]:
        """
        Permutation test for statistical significance.
        
        Tests null hypothesis that predictions have no predictive power.
        """
        n_permutations = n_permutations or self.config.n_permutations
        
        # Calculate observed metric
        observed_metric = metric_func(actuals, predictions)
        
        # Generate permutation distribution
        permuted_metrics = []
        np.random.seed(42)
        
        for _ in range(n_permutations):
            # Shuffle predictions to break relationship with outcomes
            shuffled_predictions = np.random.permutation(predictions)
            permuted_metric = metric_func(actuals, shuffled_predictions)
            permuted_metrics.append(permuted_metric)
        
        permuted_metrics = np.array(permuted_metrics)
        
        # Calculate p-value (one-sided test for better performance)
        if metric_func.__name__ in ['accuracy', 'roc_auc', 'profit']:
            # Higher is better
            p_value = (permuted_metrics >= observed_metric).mean()
        else:
            # Lower is better (e.g., log loss)
            p_value = (permuted_metrics <= observed_metric).mean()
        
        # Calculate effect size (Cohen's d)
        effect_size = (observed_metric - permuted_metrics.mean()) / permuted_metrics.std()
        
        return {
            'observed_metric': observed_metric,
            'null_mean': permuted_metrics.mean(),
            'null_std': permuted_metrics.std(),
            'p_value': p_value,
            'effect_size': effect_size,
            'percentile': stats.percentileofscore(permuted_metrics, observed_metric),
            'is_significant': p_value < 0.05
        }
    
    def bootstrap_confidence_interval(self,
                                     predictions: np.ndarray,
                                     actuals: np.ndarray,
                                     metric_func: Callable,
                                     n_bootstrap: int = None) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for win rate and other metrics.
        """
        n_bootstrap = n_bootstrap or self.config.bootstrap_iterations
        n_samples = len(predictions)
        
        # Generate bootstrap samples
        bootstrap_metrics = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            boot_predictions = predictions[indices]
            boot_actuals = actuals[indices]
            
            # Calculate metric
            boot_metric = metric_func(boot_actuals, boot_predictions)
            bootstrap_metrics.append(boot_metric)
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        # Point estimate
        point_estimate = metric_func(actuals, predictions)
        
        return {
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'bootstrap_mean': bootstrap_metrics.mean(),
            'bootstrap_std': bootstrap_metrics.std(),
            'bootstrap_median': np.median(bootstrap_metrics),
            'is_significant': ci_lower > 0.5 if metric_func.__name__ == 'accuracy' else True
        }
    
    def calculate_minimum_sample_size(self,
                                     effect_size: float = 0.05,
                                     power: float = 0.8,
                                     alpha: float = 0.05) -> int:
        """
        Calculate minimum sample size for detecting significant edge.
        
        Uses power analysis for proportion testing.
        """
        from statsmodels.stats.power import NormalIndPower
        
        # For betting, we're testing if win rate > 52.38% (breakeven with -110 odds)
        p0 = 0.5238  # Null hypothesis (breakeven)
        p1 = p0 + effect_size  # Alternative hypothesis
        
        # Calculate required sample size
        power_analysis = NormalIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            alternative='larger'
        )
        
        return int(np.ceil(sample_size))
    
    def multiple_testing_correction(self,
                                   p_values: List[float],
                                   method: str = 'fdr_bh') -> Dict[str, Any]:
        """
        Apply Benjamini-Hochberg FDR correction for multiple testing.
        """
        # Apply correction
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
            p_values,
            alpha=self.config.fdr_alpha,
            method=method
        )
        
        # Calculate FDR
        fdr = (rejected.sum() * self.config.fdr_alpha) / len(p_values)
        
        return {
            'original_p_values': p_values,
            'adjusted_p_values': p_adjusted.tolist(),
            'rejected_hypotheses': rejected.tolist(),
            'n_significant': rejected.sum(),
            'fdr_rate': fdr,
            'bonferroni_alpha': alpha_bonf,
            'sidak_alpha': alpha_sidak
        }


class RegimeChangeDetector:
    """
    Detect and handle regime changes in betting model performance.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize detector with configuration."""
        self.config = config or ValidationConfig()
        self.known_regimes = self._define_known_regimes()
        
    def _define_known_regimes(self) -> List[Dict[str, Any]]:
        """Define known regime changes in NFL."""
        return [
            {
                'name': 'COVID_2020',
                'start_date': pd.Timestamp('2020-03-01'),
                'end_date': pd.Timestamp('2021-03-01'),
                'description': 'COVID-19 impact on games'
            },
            {
                'name': 'Rule_Change_2018',
                'start_date': pd.Timestamp('2018-09-01'),
                'end_date': pd.Timestamp('2018-09-30'),
                'description': 'Roughing the passer emphasis'
            },
            {
                'name': 'Extra_Game_2021',
                'start_date': pd.Timestamp('2021-09-01'),
                'end_date': pd.Timestamp('2021-09-30'),
                'description': '17-game season introduction'
            }
        ]
    
    def cusum_detection(self,
                       returns: pd.Series,
                       threshold: float = None) -> Dict[str, Any]:
        """
        CUSUM (Cumulative Sum) implementation for strategy degradation detection.
        
        Detects when cumulative sum of standardized returns exceeds threshold.
        """
        threshold = threshold or self.config.cusum_threshold
        
        # Standardize returns
        mean_return = returns.mean()
        std_return = returns.std()
        standardized = (returns - mean_return) / std_return
        
        # Calculate CUSUM statistics
        cusum_pos = np.zeros(len(returns))
        cusum_neg = np.zeros(len(returns))
        
        for i in range(1, len(returns)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized.iloc[i])
        
        # Detect change points
        change_points = []
        for i in range(len(returns)):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                change_points.append(i)
        
        # Group consecutive change points
        regime_changes = []
        if change_points:
            current_regime_start = change_points[0]
            for i in range(1, len(change_points)):
                if change_points[i] - change_points[i-1] > 10:  # Gap of 10+ games
                    regime_changes.append({
                        'start_idx': current_regime_start,
                        'end_idx': change_points[i-1],
                        'severity': max(abs(cusum_pos[current_regime_start:change_points[i-1]].max()),
                                      abs(cusum_neg[current_regime_start:change_points[i-1]].min()))
                    })
                    current_regime_start = change_points[i]
        
        return {
            'cusum_positive': cusum_pos,
            'cusum_negative': cusum_neg,
            'change_points': change_points,
            'regime_changes': regime_changes,
            'n_regimes_detected': len(regime_changes),
            'max_cusum': max(cusum_pos.max(), abs(cusum_neg.min()))
        }
    
    def test_regime_performance(self,
                               data: pd.DataFrame,
                               model_func: Callable,
                               date_col: str = 'game_date') -> pd.DataFrame:
        """
        Test model performance across known regimes.
        """
        results = []
        
        for regime in self.known_regimes:
            # Filter data for regime period
            regime_data = data[
                (data[date_col] >= regime['start_date']) &
                (data[date_col] <= regime['end_date'])
            ].copy()
            
            if len(regime_data) < 10:  # Skip if too few games
                continue
            
            # Calculate performance metrics
            predictions = model_func(regime_data)
            actuals = regime_data['covered_spread'].values
            
            # Basic metrics
            accuracy = (predictions.round() == actuals).mean()
            auc = roc_auc_score(actuals, predictions) if len(np.unique(actuals)) > 1 else 0.5
            
            # Betting metrics
            betting_returns = self._calculate_betting_returns(predictions, actuals)
            
            results.append({
                'regime': regime['name'],
                'description': regime['description'],
                'n_games': len(regime_data),
                'accuracy': accuracy,
                'auc': auc,
                'roi': betting_returns['roi'],
                'sharpe': betting_returns['sharpe'],
                'max_drawdown': betting_returns['max_drawdown']
            })
        
        return pd.DataFrame(results)
    
    def track_feature_stability(self,
                               feature_importance: pd.DataFrame,
                               window_size: int = None) -> Dict[str, pd.DataFrame]:
        """
        Track feature importance stability over time.
        """
        window_size = window_size or self.config.lookback_window
        
        # Calculate rolling statistics
        stability_metrics = {}
        
        for feature in feature_importance.columns:
            if feature == 'date':
                continue
                
            # Rolling mean and std
            rolling_mean = feature_importance[feature].rolling(window=window_size).mean()
            rolling_std = feature_importance[feature].rolling(window=window_size).std()
            
            # Coefficient of variation
            cv = rolling_std / rolling_mean
            
            # Rank stability (how much rank changes)
            ranks = feature_importance[feature].rank()
            rank_changes = ranks.diff().abs()
            
            stability_metrics[feature] = pd.DataFrame({
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'coefficient_variation': cv,
                'rank_change': rank_changes,
                'is_stable': cv < 0.3  # CV < 30% considered stable
            })
        
        # Overall stability score
        overall_stability = pd.DataFrame({
            feature: {
                'mean_importance': metrics['rolling_mean'].mean(),
                'stability_score': 1 - metrics['coefficient_variation'].mean(),
                'avg_rank_change': metrics['rank_change'].mean()
            }
            for feature, metrics in stability_metrics.items()
        }).T
        
        return {
            'feature_metrics': stability_metrics,
            'overall_stability': overall_stability
        }
    
    def _calculate_betting_returns(self, 
                                  predictions: np.ndarray,
                                  actuals: np.ndarray) -> Dict[str, float]:
        """Helper to calculate betting returns."""
        # Simple unit betting
        bet_size = 1.0
        returns = []
        
        for pred, actual in zip(predictions, actuals):
            if pred > 0.5238:  # Bet threshold (breakeven)
                if actual == 1:
                    returns.append(bet_size * 0.91)  # Win at -110 odds
                else:
                    returns.append(-bet_size)  # Loss
        
        if not returns:
            return {'roi': 0, 'sharpe': 0, 'max_drawdown': 0}
        
        returns = np.array(returns)
        cumulative_returns = np.cumsum(returns)
        
        # ROI
        roi = returns.mean() if len(returns) > 0 else 0
        
        # Sharpe ratio (annualized for ~267 games per season)
        sharpe = (returns.mean() / returns.std() * np.sqrt(267)) if returns.std() > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        return {
            'roi': roi,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }


class BettingMetricsCalculator:
    """
    Calculate comprehensive betting-specific metrics.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize calculator with configuration."""
        self.config = config or ValidationConfig()
        
    def calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              periods_per_year: int = 267) -> float:
        """
        Calculate Sharpe ratio with proper annualization.
        
        Args:
            returns: Series of betting returns
            periods_per_year: NFL regular season games (267) for annualization
        """
        if len(returns) < 2:
            return 0.0
            
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe
        sharpe = mean_return / std_return * np.sqrt(periods_per_year)
        
        return sharpe
    
    def calculate_maximum_drawdown(self,
                                  returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and recovery metrics.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Find drawdown period
        end_idx = drawdown.idxmin()
        start_idx = cumulative[:end_idx].idxmax()
        
        # Recovery time (if recovered)
        recovery_idx = None
        if end_idx < len(cumulative) - 1:
            post_drawdown = cumulative[end_idx:]
            recovery_mask = post_drawdown >= cumulative[start_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
        
        recovery_time = (recovery_idx - end_idx).days if recovery_idx else None
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_start': start_idx,
            'max_drawdown_end': end_idx,
            'drawdown_duration_days': (end_idx - start_idx).days,
            'recovery_time_days': recovery_time,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    def calculate_clv(self,
                     predicted_probs: pd.Series,
                     closing_probs: pd.Series) -> Dict[str, float]:
        """
        Calculate Closing Line Value (CLV).
        
        CLV measures if you're beating the closing line consistently.
        """
        # CLV = predicted edge - closing edge
        predicted_edge = predicted_probs - 0.5
        closing_edge = closing_probs - 0.5
        
        clv = predicted_edge - closing_edge
        
        # Positive CLV percentage
        positive_clv_pct = (clv > 0).mean()
        
        # Average CLV
        avg_clv = clv.mean()
        
        # CLV correlation with profits
        if 'profit' in predicted_probs.index:
            clv_profit_corr = clv.corr(predicted_probs['profit'])
        else:
            clv_profit_corr = None
        
        return {
            'avg_clv': avg_clv,
            'positive_clv_rate': positive_clv_pct,
            'clv_std': clv.std(),
            'clv_profit_correlation': clv_profit_corr,
            'total_clv_captured': clv.sum()
        }
    
    def optimize_kelly_criterion(self,
                                win_prob: float,
                                odds: float = 1.91) -> Dict[str, float]:
        """
        Kelly Criterion optimization for bet sizing.
        
        f* = (p*b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = odds received on winning (decimal odds - 1)
        """
        b = odds - 1  # Convert to net odds
        p = win_prob
        q = 1 - p
        
        # Full Kelly
        kelly_full = (p * b - q) / b if b > 0 else 0
        kelly_full = max(0, kelly_full)  # Never negative
        
        # Fractional Kelly (more conservative)
        kelly_fraction = kelly_full * self.config.kelly_fraction
        
        # Capped Kelly (maximum bet size)
        kelly_capped = min(kelly_fraction, self.config.max_bet_size)
        
        # Expected growth rate
        if kelly_full > 0:
            growth_rate = p * np.log(1 + b * kelly_full) + q * np.log(1 - kelly_full)
        else:
            growth_rate = 0
        
        return {
            'kelly_full': kelly_full,
            'kelly_fraction': kelly_fraction,
            'kelly_capped': kelly_capped,
            'expected_growth_rate': growth_rate,
            'edge': p * b - q,
            'recommended_bet': kelly_capped
        }
    
    def calculate_risk_metrics(self,
                              returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        """
        # Value at Risk (VaR) - 95% confidence
        var_95 = returns.quantile(0.05)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (returns.mean() / downside_std * np.sqrt(267)) if downside_std > 0 else 0
        
        # Calmar Ratio (return / max drawdown)
        max_dd = self.calculate_maximum_drawdown(returns)['max_drawdown']
        calmar = returns.mean() / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else float('inf')
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'return_skewness': returns.skew(),
            'return_kurtosis': returns.kurtosis()
        }


class ValidationVisualizer:
    """
    Create comprehensive visualizations for model validation.
    """
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize visualizer."""
        self.config = config or ValidationConfig()
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_reliability_diagram(self,
                                predicted_probs: np.ndarray,
                                actual_outcomes: np.ndarray,
                                n_bins: int = 10,
                                save_path: str = None) -> plt.Figure:
        """
        Create reliability diagram for probability calibration.
        
        Shows how well predicted probabilities match actual frequencies.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate calibration curve
        fraction_positive, mean_predicted = calibration_curve(
            actual_outcomes, predicted_probs, n_bins=n_bins
        )
        
        # Plot perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot actual calibration
        ax1.plot(mean_predicted, fraction_positive, 'o-', 
                label='Model Calibration', markersize=8)
        
        # Add confidence bands using binomial confidence intervals
        for i, (mp, fp) in enumerate(zip(mean_predicted, fraction_positive)):
            n_samples_bin = len(predicted_probs[
                (predicted_probs >= i/n_bins) & 
                (predicted_probs < (i+1)/n_bins)
            ])
            if n_samples_bin > 0:
                # Wilson score interval
                z = 1.96  # 95% confidence
                p = fp
                n = n_samples_bin
                
                denominator = 1 + z**2/n
                centre = (p + z**2/(2*n)) / denominator
                offset = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
                
                ci_lower = centre - offset
                ci_upper = centre + offset
                
                ax1.plot([mp, mp], [ci_lower, ci_upper], 'gray', alpha=0.5, linewidth=2)
        
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Plot (Reliability Diagram)', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Histogram of predictions
        ax2.hist(predicted_probs, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predicted Probabilities', fontsize=14)
        ax2.axvline(x=0.5238, color='red', linestyle='--', 
                   label='Breakeven (52.38%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate calibration metrics
        ece = np.abs(fraction_positive - mean_predicted).mean()  # Expected Calibration Error
        mce = np.abs(fraction_positive - mean_predicted).max()   # Maximum Calibration Error
        
        fig.suptitle(f'Model Calibration Analysis\nECE: {ece:.4f}, MCE: {mce:.4f}', 
                    fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_performance(self,
                                dates: pd.Series,
                                returns: pd.Series,
                                window: int = None,
                                save_path: str = None) -> plt.Figure:
        """
        Plot rolling performance metrics over time.
        """
        window = window or self.config.lookback_window
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Calculate rolling metrics
        cumulative_returns = (1 + returns).cumprod()
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(267)
        
        # Calculate rolling win rate
        rolling_winrate = (returns > 0).rolling(window=window).mean()
        
        # Plot cumulative returns
        axes[0].plot(dates, cumulative_returns, linewidth=2, color='blue')
        axes[0].fill_between(dates, 1, cumulative_returns, 
                            where=(cumulative_returns >= 1), 
                            color='green', alpha=0.3, label='Profit')
        axes[0].fill_between(dates, 1, cumulative_returns, 
                            where=(cumulative_returns < 1), 
                            color='red', alpha=0.3, label='Loss')
        axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title('Cumulative Returns', fontsize=14)
        axes[0].set_ylabel('Cumulative Return', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot rolling Sharpe ratio
        axes[1].plot(dates, rolling_sharpe, linewidth=2, color='purple')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Sharpe = 1')
        axes[1].axhline(y=2, color='darkgreen', linestyle='--', alpha=0.3, label='Sharpe = 2')
        axes[1].fill_between(dates, 0, rolling_sharpe, 
                            where=(rolling_sharpe >= 0), 
                            color='green', alpha=0.2)
        axes[1].fill_between(dates, 0, rolling_sharpe, 
                            where=(rolling_sharpe < 0), 
                            color='red', alpha=0.2)
        axes[1].set_title(f'Rolling Sharpe Ratio ({window}-game window)', fontsize=14)
        axes[1].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot rolling win rate
        axes[2].plot(dates, rolling_winrate, linewidth=2, color='orange')
        axes[2].axhline(y=0.5238, color='red', linestyle='--', 
                       label='Breakeven (52.38%)')
        axes[2].axhline(y=0.55, color='green', linestyle='--', alpha=0.3,
                       label='55% Win Rate')
        axes[2].fill_between(dates, 0.5238, rolling_winrate, 
                            where=(rolling_winrate >= 0.5238), 
                            color='green', alpha=0.2)
        axes[2].fill_between(dates, 0.5238, rolling_winrate, 
                            where=(rolling_winrate < 0.5238), 
                            color='red', alpha=0.2)
        axes[2].set_title(f'Rolling Win Rate ({window}-game window)', fontsize=14)
        axes[2].set_ylabel('Win Rate', fontsize=12)
        axes[2].set_ylim([0.4, 0.65])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[3].fill_between(dates, 0, drawdown, color='red', alpha=0.5)
        axes[3].plot(dates, drawdown, linewidth=2, color='darkred')
        axes[3].set_title('Drawdown', fontsize=14)
        axes[3].set_ylabel('Drawdown (%)', fontsize=12)
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        fig.suptitle('Rolling Performance Metrics', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_stability(self,
                             feature_importance_df: pd.DataFrame,
                             top_n: int = 10,
                             save_path: str = None) -> plt.Figure:
        """
        Visualize feature importance stability over time.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get top N features by mean importance
        mean_importance = feature_importance_df.mean().sort_values(ascending=False)
        top_features = mean_importance.head(top_n).index.tolist()
        
        # Plot 1: Feature importance over time
        for feature in top_features:
            axes[0, 0].plot(feature_importance_df.index, 
                          feature_importance_df[feature],
                          label=feature, linewidth=2)
        
        axes[0, 0].set_title(f'Top {top_n} Feature Importance Over Time', fontsize=14)
        axes[0, 0].set_xlabel('Time Period', fontsize=12)
        axes[0, 0].set_ylabel('Importance Score', fontsize=12)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature rank stability
        ranks = feature_importance_df.rank(axis=1, ascending=False)
        for feature in top_features:
            axes[0, 1].plot(ranks.index, ranks[feature], 
                          label=feature, linewidth=2)
        
        axes[0, 1].set_title('Feature Rank Stability', fontsize=14)
        axes[0, 1].set_xlabel('Time Period', fontsize=12)
        axes[0, 1].set_ylabel('Rank', fontsize=12)
        axes[0, 1].invert_yaxis()  # Lower rank (1, 2, 3) at top
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Coefficient of variation heatmap
        cv_matrix = feature_importance_df[top_features].std() / feature_importance_df[top_features].mean()
        cv_data = cv_matrix.values.reshape(-1, 1)
        
        im = axes[1, 0].imshow(cv_data, cmap='RdYlGn_r', aspect='auto')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_title('Feature Stability (Coefficient of Variation)', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('CV (lower is more stable)', rotation=270, labelpad=15)
        
        # Add text annotations
        for i, feature in enumerate(top_features):
            axes[1, 0].text(0, i, f'{cv_matrix[feature]:.3f}', 
                          ha='center', va='center', color='white', fontweight='bold')
        
        # Plot 4: Correlation matrix of feature importances
        corr_matrix = feature_importance_df[top_features].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
        axes[1, 1].set_title('Feature Importance Correlations', fontsize=14)
        
        fig.suptitle('Feature Importance Stability Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_walk_forward_results(self,
                                 splits: List[Tuple],
                                 scores: List[float],
                                 dates: pd.Series,
                                 save_path: str = None) -> plt.Figure:
        """
        Visualize walk-forward validation results.
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot 1: Timeline visualization of splits
        for i, (train_idx, test_idx) in enumerate(splits):
            train_dates = dates.iloc[train_idx]
            test_dates = dates.iloc[test_idx]
            
            # Training period (blue)
            axes[0].add_patch(Rectangle(
                (train_dates.min(), i), 
                train_dates.max() - train_dates.min(),
                0.8,
                facecolor='blue', alpha=0.3, edgecolor='blue'
            ))
            
            # Test period (green)
            axes[0].add_patch(Rectangle(
                (test_dates.min(), i),
                test_dates.max() - test_dates.min(),
                0.8,
                facecolor='green', alpha=0.5, edgecolor='green'
            ))
        
        axes[0].set_ylim(-0.5, len(splits))
        axes[0].set_xlim(dates.min(), dates.max())
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Split Number', fontsize=12)
        axes[0].set_title('Walk-Forward Validation Splits Timeline', fontsize=14)
        axes[0].legend([Rectangle((0,0),1,1, facecolor='blue', alpha=0.3),
                       Rectangle((0,0),1,1, facecolor='green', alpha=0.5)],
                      ['Training Period', 'Test Period'])
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Performance across splits
        split_numbers = range(1, len(scores) + 1)
        axes[1].plot(split_numbers, scores, 'o-', linewidth=2, markersize=8)
        axes[1].axhline(y=0.5238, color='red', linestyle='--', 
                       label='Breakeven (52.38%)')
        axes[1].fill_between(split_numbers, 0.5238, scores,
                            where=[s > 0.5238 for s in scores],
                            color='green', alpha=0.3)
        axes[1].fill_between(split_numbers, 0.5238, scores,
                            where=[s <= 0.5238 for s in scores],
                            color='red', alpha=0.3)
        
        # Add confidence interval
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_upper = mean_score + 1.96 * std_score / np.sqrt(len(scores))
        ci_lower = mean_score - 1.96 * std_score / np.sqrt(len(scores))
        
        axes[1].axhline(y=mean_score, color='blue', linestyle='-', 
                       label=f'Mean: {mean_score:.4f}')
        axes[1].axhspan(ci_lower, ci_upper, alpha=0.2, color='blue',
                       label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        
        axes[1].set_xlabel('Split Number', fontsize=12)
        axes[1].set_ylabel('Test Score', fontsize=12)
        axes[1].set_title('Performance Across Walk-Forward Splits', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('Walk-Forward Validation Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def generate_sample_nfl_data(n_games: int = 5000,
                            start_date: str = '2015-09-01') -> pd.DataFrame:
    """
    Generate sample NFL betting data for demonstration.
    """
    np.random.seed(42)
    
    # Generate dates (NFL season runs Sep-Feb)
    dates = pd.date_range(start=start_date, periods=n_games, freq='D')
    
    # Generate features
    data = {
        'game_date': dates,
        'home_team': np.random.choice(['Team_' + str(i) for i in range(32)], n_games),
        'away_team': np.random.choice(['Team_' + str(i) for i in range(32)], n_games),
        
        # Some predictive features
        'feature_1': np.random.randn(n_games),
        'feature_2': np.random.randn(n_games),
        'feature_3': np.random.randn(n_games),
        
        # Target with some signal
        'covered_spread': np.random.binomial(1, 0.52, n_games),  # Slight edge
        
        # Model predictions (with some skill)
        'model_prediction': np.clip(np.random.beta(2, 2, n_games) * 0.3 + 0.4, 0, 1),
        
        # Closing line for CLV
        'closing_prob': np.clip(np.random.beta(2, 2, n_games) * 0.2 + 0.45, 0, 1)
    }
    
    df = pd.DataFrame(data)
    
    # Add correlation between prediction and outcome
    df.loc[df['model_prediction'] > 0.55, 'covered_spread'] = np.random.binomial(
        1, 0.58, (df['model_prediction'] > 0.55).sum()
    )
    
    return df


if __name__ == "__main__":
    print("NFL Betting Model Validation Framework")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating sample NFL data...")
    data = generate_sample_nfl_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['game_date'].min()} to {data['game_date'].max()}")
    
    # Initialize components
    config = ValidationConfig()
    ts_validator = NFLTimeSeriesValidator(config)
    stat_tester = StatisticalSignificanceTester(config)
    regime_detector = RegimeChangeDetector(config)
    metrics_calc = BettingMetricsCalculator(config)
    visualizer = ValidationVisualizer(config)
    
    # Run validation pipeline
    print("\n1. TIME-SERIES CROSS-VALIDATION")
    print("-" * 40)
    splits = ts_validator.walk_forward_split(data)
    print(f"Created {len(splits)} walk-forward splits")
    
    # Test first split
    if splits:
        train_idx, test_idx = splits[0]
        print(f"First split: Train={len(train_idx)} games, Test={len(test_idx)} games")
    
    print("\n2. STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 40)
    
    # Permutation test
    perm_results = stat_tester.permutation_test(
        data['model_prediction'].values,
        data['covered_spread'].values,
        lambda y, p: ((p > 0.5238) == y).mean(),  # Accuracy at threshold
        n_permutations=100  # Reduced for demo
    )
    print(f"Permutation test p-value: {perm_results['p_value']:.4f}")
    print(f"Effect size: {perm_results['effect_size']:.4f}")
    
    # Bootstrap confidence intervals
    boot_results = stat_tester.bootstrap_confidence_interval(
        data['model_prediction'].values,
        data['covered_spread'].values,
        lambda y, p: ((p > 0.5238) == y).mean(),
        n_bootstrap=1000  # Reduced for demo
    )
    print(f"Bootstrap CI: [{boot_results['ci_lower']:.4f}, {boot_results['ci_upper']:.4f}]")
    
    # Minimum sample size
    min_sample = stat_tester.calculate_minimum_sample_size(effect_size=0.03)
    print(f"Minimum sample size for 3% edge: {min_sample} games")
    
    print("\n3. REGIME CHANGE DETECTION")
    print("-" * 40)
    
    # Calculate returns for CUSUM
    returns = ((data['model_prediction'] > 0.5238) == data['covered_spread']).astype(float) - 0.5
    returns_series = pd.Series(returns.values, index=data['game_date'])
    
    cusum_results = regime_detector.cusum_detection(returns_series)
    print(f"Regime changes detected: {cusum_results['n_regimes_detected']}")
    print(f"Maximum CUSUM statistic: {cusum_results['max_cusum']:.4f}")
    
    print("\n4. BETTING METRICS")
    print("-" * 40)
    
    # Calculate betting returns
    betting_returns = []
    for pred, actual in zip(data['model_prediction'], data['covered_spread']):
        if pred > 0.5238:
            if actual == 1:
                betting_returns.append(0.91)  # Win at -110
            else:
                betting_returns.append(-1.0)  # Loss
    
    if betting_returns:
        returns_series = pd.Series(betting_returns, index=data['game_date'][:len(betting_returns)])
        
        # Sharpe ratio
        sharpe = metrics_calc.calculate_sharpe_ratio(returns_series)
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
        # Maximum drawdown
        dd_results = metrics_calc.calculate_maximum_drawdown(returns_series)
        print(f"Maximum Drawdown: {dd_results['max_drawdown']:.2%}")
        
        # CLV
        clv_results = metrics_calc.calculate_clv(
            data['model_prediction'],
            data['closing_prob']
        )
        print(f"Average CLV: {clv_results['avg_clv']:.4f}")
        print(f"Positive CLV rate: {clv_results['positive_clv_rate']:.2%}")
        
        # Kelly optimization
        avg_win_prob = (data['model_prediction'] > 0.5238).mean()
        kelly_results = metrics_calc.optimize_kelly_criterion(avg_win_prob)
        print(f"Kelly fraction: {kelly_results['kelly_fraction']:.2%}")
        print(f"Recommended bet size: {kelly_results['recommended_bet']:.2%}")
    
    print("\n5. VISUALIZATIONS")
    print("-" * 40)
    print("Creating validation visualizations...")
    
    # Reliability diagram
    fig1 = visualizer.plot_reliability_diagram(
        data['model_prediction'].values,
        data['covered_spread'].values
    )
    
    # Rolling performance (if we have returns)
    if betting_returns:
        fig2 = visualizer.plot_rolling_performance(
            data['game_date'][:len(betting_returns)],
            returns_series
        )
    
    print("\n" + "=" * 60)
    print("Validation framework demonstration complete!")
    print("=" * 60)
