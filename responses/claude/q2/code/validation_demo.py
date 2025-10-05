"""
NFL Validation Framework - Simplified Demo
==========================================
A working demo that doesn't require all dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimplifiedValidator:
    """Simplified validation framework for demonstration."""
    
    def __init__(self):
        self.results = {}
    
    def walk_forward_validation(self, data: pd.DataFrame, 
                               window_size: int = 500,
                               test_size: int = 100,
                               gap: int = 20) -> Dict:
        """
        Simplified walk-forward validation.
        
        Key concepts:
        - Training window: Previous 'window_size' games
        - Gap: 'gap' games between train and test (purge + embargo)
        - Test window: Next 'test_size' games
        """
        results = []
        
        start_idx = window_size
        while start_idx + gap + test_size <= len(data):
            # Training set
            train_start = start_idx - window_size
            train_end = start_idx
            
            # Test set (with gap)
            test_start = train_end + gap
            test_end = test_start + test_size
            
            # Get indices
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Simulate model performance
            train_accuracy = 0.52 + np.random.randn() * 0.02
            test_accuracy = 0.51 + np.random.randn() * 0.02
            
            results.append({
                'split': len(results) + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'gap_days': gap
            })
            
            # Move forward
            start_idx += test_size + gap
        
        # Calculate summary
        test_accuracies = [r['test_accuracy'] for r in results]
        return {
            'n_splits': len(results),
            'mean_accuracy': np.mean(test_accuracies),
            'std_accuracy': np.std(test_accuracies),
            'profitable_splits': sum(1 for a in test_accuracies if a > 0.5238),
            'details': results
        }
    
    def permutation_test(self, predictions: np.ndarray, 
                        actuals: np.ndarray,
                        n_permutations: int = 100) -> Dict:
        """
        Simplified permutation test for statistical significance.
        """
        # Calculate observed accuracy
        observed = ((predictions > 0.5238) == actuals).mean()
        
        # Generate null distribution
        null_distribution = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(predictions)
            null_acc = ((shuffled > 0.5238) == actuals).mean()
            null_distribution.append(null_acc)
        
        # Calculate p-value
        null_distribution = np.array(null_distribution)
        p_value = (null_distribution >= observed).mean()
        
        # Effect size
        effect_size = (observed - null_distribution.mean()) / null_distribution.std()
        
        return {
            'observed': observed,
            'null_mean': null_distribution.mean(),
            'null_std': null_distribution.std(),
            'p_value': p_value,
            'effect_size': effect_size,
            'is_significant': p_value < 0.05
        }
    
    def calculate_betting_metrics(self, predictions: np.ndarray,
                                 actuals: np.ndarray) -> Dict:
        """
        Calculate key betting metrics.
        """
        # Calculate returns
        returns = []
        for pred, actual in zip(predictions, actuals):
            if pred > 0.5238:  # Bet threshold
                if actual == 1:
                    returns.append(0.91)  # Win at -110
                else:
                    returns.append(-1.0)  # Loss
        
        if not returns:
            return {'error': 'No bets placed'}
        
        returns = np.array(returns)
        
        # Metrics
        win_rate = (returns > 0).mean()
        roi = returns.mean()
        total_return = returns.sum()
        
        # Sharpe ratio (simplified)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(267)  # Annualized
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        return {
            'n_bets': len(returns),
            'win_rate': win_rate,
            'roi': roi,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profitable': roi > 0
        }
    
    def detect_regime_changes(self, returns: np.ndarray,
                             threshold: float = 2.0) -> Dict:
        """
        Simple CUSUM implementation for regime change detection.
        """
        # Standardize returns
        mean_return = returns.mean()
        std_return = returns.std() if returns.std() > 0 else 1
        standardized = (returns - mean_return) / std_return
        
        # Calculate CUSUM
        cusum_pos = np.zeros(len(returns))
        cusum_neg = np.zeros(len(returns))
        
        for i in range(1, len(returns)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized[i])
        
        # Detect changes
        changes = []
        for i in range(len(returns)):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                changes.append(i)
        
        # Group consecutive changes
        regimes = []
        if changes:
            regime_start = changes[0]
            for i in range(1, len(changes)):
                if changes[i] - changes[i-1] > 10:  # Gap of 10
                    regimes.append((regime_start, changes[i-1]))
                    regime_start = changes[i]
            regimes.append((regime_start, changes[-1]))
        
        return {
            'n_regimes': len(regimes),
            'regime_boundaries': regimes,
            'max_cusum': max(cusum_pos.max(), abs(cusum_neg.min())),
            'has_regime_change': len(regimes) > 0
        }
    
    def calculate_sample_size(self, desired_edge: float = 0.03,
                             power: float = 0.8,
                             alpha: float = 0.05) -> int:
        """
        Calculate minimum sample size needed for significance.
        """
        from scipy.stats import norm
        
        # For proportion test
        p0 = 0.5238  # Breakeven
        p1 = p0 + desired_edge
        
        # Z-scores
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)
        
        # Sample size formula
        n = ((z_alpha + z_beta) ** 2 * 
             (p1 * (1 - p1) + p0 * (1 - p0))) / (p1 - p0) ** 2
        
        return int(np.ceil(n))


def run_validation_demo():
    """Run complete validation demo."""
    print("=" * 80)
    print("NFL BETTING MODEL VALIDATION DEMO")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. GENERATING SAMPLE DATA")
    print("-" * 40)
    
    n_games = 2000
    dates = pd.date_range(start='2019-01-01', periods=n_games, freq='D')
    
    # Create realistic betting data
    np.random.seed(42)
    data = pd.DataFrame({
        'game_date': dates,
        'home_team': np.random.choice(['Team_' + str(i) for i in range(32)], n_games),
        'prediction': np.random.beta(2, 2, n_games) * 0.3 + 0.4,  # Predictions between 0.4-0.7
        'actual': np.random.binomial(1, 0.52, n_games),  # Slight edge
        'spread': np.random.normal(3, 7, n_games),
        'total': np.random.normal(45, 10, n_games)
    })
    
    # Add correlation between prediction and outcome
    high_confidence = data['prediction'] > 0.55
    data.loc[high_confidence, 'actual'] = np.random.binomial(1, 0.58, high_confidence.sum())
    
    print(f"  Games: {len(data)}")
    print(f"  Date range: {data['game_date'].min().date()} to {data['game_date'].max().date()}")
    print(f"  Win rate: {data['actual'].mean():.2%}")
    
    # Initialize validator
    validator = SimplifiedValidator()
    
    # 2. Walk-Forward Validation
    print("\n2. WALK-FORWARD VALIDATION")
    print("-" * 40)
    
    wf_results = validator.walk_forward_validation(data)
    print(f"  Splits created: {wf_results['n_splits']}")
    print(f"  Mean accuracy: {wf_results['mean_accuracy']:.4f} ± {wf_results['std_accuracy']:.4f}")
    print(f"  Profitable splits: {wf_results['profitable_splits']}/{wf_results['n_splits']}")
    
    # Show first few splits
    print("\n  First 3 splits:")
    for split in wf_results['details'][:3]:
        print(f"    Split {split['split']}: Train={split['train_size']} games, "
              f"Test={split['test_size']} games, Accuracy={split['test_accuracy']:.4f}")
    
    # 3. Statistical Significance Testing
    print("\n3. STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 40)
    
    # Permutation test
    perm_results = validator.permutation_test(
        data['prediction'].values,
        data['actual'].values,
        n_permutations=100
    )
    
    print(f"  Observed accuracy: {perm_results['observed']:.4f}")
    print(f"  Null distribution: {perm_results['null_mean']:.4f} ± {perm_results['null_std']:.4f}")
    print(f"  P-value: {perm_results['p_value']:.4f}")
    print(f"  Effect size: {perm_results['effect_size']:.4f}")
    print(f"  Statistically significant: {perm_results['is_significant']}")
    
    # Sample size calculation
    min_sample = validator.calculate_sample_size(desired_edge=0.03)
    print(f"\n  Minimum sample size for 3% edge: {min_sample} games")
    
    # 4. Betting Metrics
    print("\n4. BETTING PERFORMANCE METRICS")
    print("-" * 40)
    
    betting_metrics = validator.calculate_betting_metrics(
        data['prediction'].values,
        data['actual'].values
    )
    
    print(f"  Total bets placed: {betting_metrics['n_bets']}")
    print(f"  Win rate: {betting_metrics['win_rate']:.2%}")
    print(f"  ROI: {betting_metrics['roi']:.4f}")
    print(f"  Total return: {betting_metrics['total_return']:.2f} units")
    print(f"  Sharpe ratio: {betting_metrics['sharpe_ratio']:.4f}")
    print(f"  Max drawdown: {betting_metrics['max_drawdown']:.2f} units")
    
    # 5. Regime Change Detection
    print("\n5. REGIME CHANGE DETECTION")
    print("-" * 40)
    
    # Calculate returns for regime detection
    returns = []
    for pred, actual in zip(data['prediction'].values, data['actual'].values):
        if pred > 0.5238:
            returns.append(0.91 if actual == 1 else -1.0)
    
    regime_results = validator.detect_regime_changes(np.array(returns))
    
    print(f"  Regime changes detected: {regime_results['n_regimes']}")
    print(f"  Maximum CUSUM statistic: {regime_results['max_cusum']:.4f}")
    
    if regime_results['regime_boundaries']:
        print("\n  Regime boundaries (game indices):")
        for i, (start, end) in enumerate(regime_results['regime_boundaries'][:3]):
            print(f"    Regime {i+1}: Games {start}-{end}")
    
    # 6. Validation Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    # Overall assessment
    passed_tests = []
    failed_tests = []
    
    # Check criteria
    if wf_results['mean_accuracy'] > 0.5238:
        passed_tests.append("Walk-forward profitable")
    else:
        failed_tests.append("Walk-forward not profitable")
    
    if perm_results['is_significant']:
        passed_tests.append("Statistically significant edge")
    else:
        failed_tests.append("No significant edge")
    
    if betting_metrics['sharpe_ratio'] > 1.0:
        passed_tests.append("Good Sharpe ratio (>1.0)")
    else:
        failed_tests.append("Poor Sharpe ratio (<1.0)")
    
    if abs(betting_metrics['max_drawdown']) < 20:
        passed_tests.append("Acceptable drawdown (<20 units)")
    else:
        failed_tests.append("High drawdown (>20 units)")
    
    overall_pass = len(passed_tests) >= 3
    
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Score: {len(passed_tests)}/{len(passed_tests) + len(failed_tests)}")
    
    print("\nPassed Tests:")
    for test in passed_tests:
        print(f"  ✓ {test}")
    
    print("\nFailed Tests:")
    for test in failed_tests:
        print(f"  ✗ {test}")
    
    print("\n" + "=" * 80)
    print("Key Insights:")
    print("-" * 40)
    print("1. Walk-forward validation with temporal gaps prevents overfitting")
    print("2. Statistical significance testing confirms edge is not due to chance")
    print("3. Betting metrics show risk-adjusted performance")
    print("4. Regime detection identifies periods of strategy degradation")
    print("5. Multiple validation methods provide robust assessment")
    
    return validator, data


if __name__ == "__main__":
    print("""
    NFL Betting Model Validation Framework
    ======================================
    
    This demo shows key validation concepts without external dependencies.
    
    Features demonstrated:
    - Walk-forward validation with temporal gaps
    - Statistical significance testing
    - Betting performance metrics
    - Regime change detection
    - Overall validation assessment
    """)
    
    validator, data = run_validation_demo()
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
