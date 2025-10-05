"""
NFL Validation Framework - Usage Examples
=========================================
Practical examples for using the validation framework with real betting models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from nfl_validation_framework import (
    ValidationConfig,
    NFLTimeSeriesValidator,
    StatisticalSignificanceTester,
    RegimeChangeDetector,
    BettingMetricsCalculator,
    ValidationVisualizer
)

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class NFLBettingModelValidator:
    """
    Complete validation pipeline for NFL betting models.
    """
    
    def __init__(self, model, features: list, target_col: str = 'covered_spread'):
        """
        Initialize validator with trained model.
        
        Args:
            model: Trained ML model with predict_proba method
            features: List of feature columns
            target_col: Target column name
        """
        self.model = model
        self.features = features
        self.target_col = target_col
        
        # Initialize validation components
        self.config = ValidationConfig()
        self.ts_validator = NFLTimeSeriesValidator(self.config)
        self.stat_tester = StatisticalSignificanceTester(self.config)
        self.regime_detector = RegimeChangeDetector(self.config)
        self.metrics_calc = BettingMetricsCalculator(self.config)
        self.visualizer = ValidationVisualizer(self.config)
        
        # Storage for results
        self.validation_results = {}
        self.performance_history = []
        
    def run_complete_validation(self, data: pd.DataFrame) -> dict:
        """
        Run complete validation pipeline on data.
        
        Returns comprehensive validation report.
        """
        print("=" * 80)
        print("NFL BETTING MODEL VALIDATION PIPELINE")
        print("=" * 80)
        
        results = {}
        
        # 1. Walk-Forward Validation
        print("\n1. WALK-FORWARD VALIDATION")
        print("-" * 40)
        results['walk_forward'] = self._run_walk_forward_validation(data)
        
        # 2. Statistical Significance Testing
        print("\n2. STATISTICAL SIGNIFICANCE TESTING")
        print("-" * 40)
        results['significance'] = self._run_significance_tests(data)
        
        # 3. Regime Analysis
        print("\n3. REGIME CHANGE ANALYSIS")
        print("-" * 40)
        results['regime'] = self._run_regime_analysis(data)
        
        # 4. Betting Metrics
        print("\n4. BETTING PERFORMANCE METRICS")
        print("-" * 40)
        results['betting'] = self._calculate_betting_metrics(data)
        
        # 5. Feature Stability
        print("\n5. FEATURE STABILITY ANALYSIS")
        print("-" * 40)
        results['features'] = self._analyze_feature_stability(data)
        
        # 6. Multiple Testing Correction
        print("\n6. MULTIPLE TESTING CORRECTION")
        print("-" * 40)
        results['multiple_testing'] = self._apply_multiple_testing_correction(results)
        
        # Store results
        self.validation_results = results
        
        # Generate report
        report = self._generate_validation_report(results)
        
        return report
    
    def _run_walk_forward_validation(self, data: pd.DataFrame) -> dict:
        """
        Perform walk-forward validation with proper temporal handling.
        """
        # Get splits
        splits = self.ts_validator.walk_forward_split(data)
        
        scores = {
            'accuracy': [],
            'auc': [],
            'sharpe': [],
            'max_drawdown': [],
            'profit': [],
            'n_bets': []
        }
        
        detailed_results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"  Processing split {i+1}/{len(splits)}...")
            
            # Get train/test data
            X_train = data.iloc[train_idx][self.features]
            y_train = data.iloc[train_idx][self.target_col]
            X_test = data.iloc[test_idx][self.features]
            y_test = data.iloc[test_idx][self.target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model (clone to avoid contamination)
            model_clone = self._clone_model()
            model_clone.fit(X_train_scaled, y_train)
            
            # Predictions
            predictions = model_clone.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = ((predictions > 0.5238) == y_test).mean()
            
            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_test, predictions)
            except:
                auc = 0.5
            
            # Betting returns
            returns = self._calculate_returns(predictions, y_test.values)
            
            if len(returns) > 0:
                sharpe = self.metrics_calc.calculate_sharpe_ratio(pd.Series(returns))
                dd_info = self.metrics_calc.calculate_maximum_drawdown(pd.Series(returns))
                max_dd = dd_info['max_drawdown']
                total_profit = sum(returns)
                n_bets = len(returns)
            else:
                sharpe = 0
                max_dd = 0
                total_profit = 0
                n_bets = 0
            
            # Store scores
            scores['accuracy'].append(accuracy)
            scores['auc'].append(auc)
            scores['sharpe'].append(sharpe)
            scores['max_drawdown'].append(max_dd)
            scores['profit'].append(total_profit)
            scores['n_bets'].append(n_bets)
            
            # Store detailed results
            detailed_results.append({
                'split': i + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_dates': (data.iloc[train_idx]['game_date'].min(), 
                              data.iloc[train_idx]['game_date'].max()),
                'test_dates': (data.iloc[test_idx]['game_date'].min(),
                             data.iloc[test_idx]['game_date'].max()),
                'accuracy': accuracy,
                'auc': auc,
                'sharpe': sharpe,
                'profit': total_profit,
                'n_bets': n_bets
            })
        
        # Calculate summary statistics
        summary = {
            'n_splits': len(splits),
            'mean_accuracy': np.mean(scores['accuracy']),
            'std_accuracy': np.std(scores['accuracy']),
            'mean_auc': np.mean(scores['auc']),
            'mean_sharpe': np.mean(scores['sharpe']),
            'total_profit': sum(scores['profit']),
            'total_bets': sum(scores['n_bets']),
            'profitable_splits': sum([p > 0 for p in scores['profit']]),
            'detailed_results': detailed_results
        }
        
        print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"  Mean Sharpe: {summary['mean_sharpe']:.4f}")
        print(f"  Profitable Splits: {summary['profitable_splits']}/{summary['n_splits']}")
        
        return summary
    
    def _run_significance_tests(self, data: pd.DataFrame) -> dict:
        """
        Run comprehensive statistical significance tests.
        """
        # Get predictions for full dataset
        X = data[self.features]
        y = data[self.target_col]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        results = {}
        
        # 1. Permutation test
        print("  Running permutation test (1000 iterations)...")
        perm_results = self.stat_tester.permutation_test(
            predictions,
            y.values,
            lambda y_true, y_pred: ((y_pred > 0.5238) == y_true).mean()
        )
        results['permutation'] = perm_results
        print(f"    P-value: {perm_results['p_value']:.4f}")
        print(f"    Effect size: {perm_results['effect_size']:.4f}")
        
        # 2. Bootstrap confidence intervals
        print("  Calculating bootstrap confidence intervals...")
        boot_results = self.stat_tester.bootstrap_confidence_interval(
            predictions,
            y.values,
            lambda y_true, y_pred: ((y_pred > 0.5238) == y_true).mean()
        )
        results['bootstrap'] = boot_results
        print(f"    95% CI: [{boot_results['ci_lower']:.4f}, {boot_results['ci_upper']:.4f}]")
        
        # 3. Minimum sample size calculation
        min_samples = {}
        for edge in [0.01, 0.02, 0.03, 0.05]:
            min_n = self.stat_tester.calculate_minimum_sample_size(effect_size=edge)
            min_samples[f'edge_{int(edge*100)}pct'] = min_n
        results['min_sample_sizes'] = min_samples
        print(f"    Min samples for 3% edge: {min_samples['edge_3pct']}")
        
        # 4. Test temporal independence
        errors = predictions - y.values
        errors_series = pd.Series(errors, index=data['game_date'])
        temporal_results = self.ts_validator.validate_temporal_independence(
            pd.Series(predictions),
            y,
            data['game_date']
        )
        results['temporal_independence'] = temporal_results
        print(f"    Is stationary: {temporal_results['is_stationary']}")
        print(f"    Has autocorrelation: {temporal_results['has_autocorrelation']}")
        
        return results
    
    def _run_regime_analysis(self, data: pd.DataFrame) -> dict:
        """
        Analyze performance across different regimes.
        """
        # Get predictions
        X = data[self.features]
        y = data[self.target_col]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate returns
        returns = self._calculate_returns(predictions, y.values)
        returns_series = pd.Series(returns, index=data['game_date'][:len(returns)])
        
        results = {}
        
        # 1. CUSUM detection
        print("  Running CUSUM change detection...")
        cusum_results = self.regime_detector.cusum_detection(returns_series)
        results['cusum'] = cusum_results
        print(f"    Regimes detected: {cusum_results['n_regimes_detected']}")
        
        # 2. Known regime performance
        print("  Testing known regime performance...")
        
        # Create a simple model function for regime testing
        def model_func(df):
            X = df[self.features]
            X_scaled = scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]
        
        regime_performance = self.regime_detector.test_regime_performance(
            data, model_func
        )
        results['known_regimes'] = regime_performance.to_dict('records') if not regime_performance.empty else []
        
        if not regime_performance.empty:
            print(f"    COVID-2020 ROI: {regime_performance[regime_performance['regime']=='COVID_2020']['roi'].values[0] if 'COVID_2020' in regime_performance['regime'].values else 'N/A'}")
        
        # 3. Feature stability tracking
        print("  Analyzing feature stability...")
        
        # Calculate feature importance over rolling windows
        window_size = 500  # games
        importance_over_time = []
        
        for i in range(window_size, len(data), 100):
            window_data = data.iloc[i-window_size:i]
            X_window = window_data[self.features]
            y_window = window_data[self.target_col]
            
            # Quick feature importance (using permutation importance)
            from sklearn.inspection import permutation_importance
            X_window_scaled = scaler.fit_transform(X_window)
            
            perm_imp = permutation_importance(
                self.model, X_window_scaled, y_window,
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            importance_dict = dict(zip(self.features, perm_imp.importances_mean))
            importance_dict['window_end'] = window_data['game_date'].max()
            importance_over_time.append(importance_dict)
        
        if importance_over_time:
            importance_df = pd.DataFrame(importance_over_time)
            importance_df.set_index('window_end', inplace=True)
            
            stability_results = self.regime_detector.track_feature_stability(importance_df)
            results['feature_stability'] = {
                'overall_scores': stability_results['overall_stability'].to_dict(),
                'n_stable_features': (stability_results['overall_stability']['stability_score'] > 0.7).sum()
            }
            print(f"    Stable features: {results['feature_stability']['n_stable_features']}/{len(self.features)}")
        
        return results
    
    def _calculate_betting_metrics(self, data: pd.DataFrame) -> dict:
        """
        Calculate comprehensive betting performance metrics.
        """
        # Get predictions
        X = data[self.features]
        y = data[self.target_col]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate returns
        returns = self._calculate_returns(predictions, y.values)
        returns_series = pd.Series(returns, index=data['game_date'][:len(returns)])
        
        results = {}
        
        # 1. Core metrics
        print("  Calculating core betting metrics...")
        
        # Sharpe ratio
        sharpe = self.metrics_calc.calculate_sharpe_ratio(returns_series)
        results['sharpe_ratio'] = sharpe
        print(f"    Sharpe Ratio: {sharpe:.4f}")
        
        # Maximum drawdown
        dd_results = self.metrics_calc.calculate_maximum_drawdown(returns_series)
        results['drawdown'] = dd_results
        print(f"    Max Drawdown: {dd_results['max_drawdown']:.2%}")
        print(f"    Recovery Time: {dd_results['recovery_time_days']} days")
        
        # 2. CLV (if closing line data available)
        if 'closing_prob' in data.columns:
            print("  Calculating CLV metrics...")
            clv_results = self.metrics_calc.calculate_clv(
                pd.Series(predictions),
                data['closing_prob']
            )
            results['clv'] = clv_results
            print(f"    Average CLV: {clv_results['avg_clv']:.4f}")
            print(f"    Positive CLV Rate: {clv_results['positive_clv_rate']:.2%}")
        
        # 3. Kelly optimization
        print("  Optimizing Kelly criterion...")
        win_rate = ((predictions > 0.5238) == y).mean()
        kelly_results = self.metrics_calc.optimize_kelly_criterion(win_rate)
        results['kelly'] = kelly_results
        print(f"    Optimal Kelly: {kelly_results['kelly_full']:.2%}")
        print(f"    Conservative Kelly: {kelly_results['kelly_fraction']:.2%}")
        
        # 4. Risk metrics
        print("  Calculating risk metrics...")
        risk_metrics = self.metrics_calc.calculate_risk_metrics(returns_series)
        results['risk'] = risk_metrics
        print(f"    Win Rate: {risk_metrics['win_rate']:.2%}")
        print(f"    Profit Factor: {risk_metrics['profit_factor']:.2f}")
        print(f"    Sortino Ratio: {risk_metrics['sortino_ratio']:.4f}")
        
        return results
    
    def _analyze_feature_stability(self, data: pd.DataFrame) -> dict:
        """
        Analyze feature importance stability across time periods.
        """
        results = {}
        
        # Split data into time periods
        n_periods = 5
        period_size = len(data) // n_periods
        
        feature_importance_by_period = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(data)
            
            period_data = data.iloc[start_idx:end_idx]
            X_period = period_data[self.features]
            y_period = period_data[self.target_col]
            
            # Calculate feature importance
            from sklearn.inspection import permutation_importance
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_period)
            
            perm_imp = permutation_importance(
                self.model, X_scaled, y_period,
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            importance_dict = dict(zip(self.features, perm_imp.importances_mean))
            importance_dict['period'] = i + 1
            feature_importance_by_period.append(importance_dict)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(feature_importance_by_period)
        importance_df.set_index('period', inplace=True)
        
        # Calculate stability metrics
        feature_stats = {}
        for feature in self.features:
            values = importance_df[feature]
            feature_stats[feature] = {
                'mean': values.mean(),
                'std': values.std(),
                'cv': values.std() / values.mean() if values.mean() != 0 else float('inf'),
                'range': values.max() - values.min(),
                'is_stable': values.std() / values.mean() < 0.3 if values.mean() != 0 else False
            }
        
        # Rank features by stability
        stable_features = sorted(
            [(f, stats['cv']) for f, stats in feature_stats.items() if stats['is_stable']],
            key=lambda x: x[1]
        )
        
        results['feature_stats'] = feature_stats
        results['stable_features'] = [f[0] for f in stable_features]
        results['n_stable'] = len(stable_features)
        results['stability_rate'] = len(stable_features) / len(self.features)
        
        print(f"  Stable features: {results['n_stable']}/{len(self.features)}")
        print(f"  Stability rate: {results['stability_rate']:.2%}")
        
        return results
    
    def _apply_multiple_testing_correction(self, results: dict) -> dict:
        """
        Apply multiple testing correction to p-values.
        """
        # Collect all p-values from various tests
        p_values = []
        p_value_sources = []
        
        # From significance tests
        if 'significance' in results:
            if 'permutation' in results['significance']:
                p_values.append(results['significance']['permutation']['p_value'])
                p_value_sources.append('permutation_test')
            
            if 'temporal_independence' in results['significance']:
                p_values.append(results['significance']['temporal_independence']['adf_pvalue'])
                p_value_sources.append('stationarity_test')
        
        # Add p-values for each feature (simulated for demo)
        for feature in self.features[:self.config.n_features_tested]:
            # In practice, these would come from feature-specific tests
            p_values.append(np.random.beta(1, 10))  # Simulated p-values
            p_value_sources.append(f'feature_{feature}')
        
        if not p_values:
            return {}
        
        # Apply Benjamini-Hochberg correction
        correction_results = self.stat_tester.multiple_testing_correction(p_values)
        
        # Map back to sources
        corrected_p_values = {}
        for source, p_orig, p_adj, rejected in zip(
            p_value_sources, 
            p_values,
            correction_results['adjusted_p_values'],
            correction_results['rejected_hypotheses']
        ):
            corrected_p_values[source] = {
                'original_p': p_orig,
                'adjusted_p': p_adj,
                'is_significant': rejected
            }
        
        print(f"  Tests performed: {len(p_values)}")
        print(f"  Significant after correction: {correction_results['n_significant']}")
        print(f"  FDR rate: {correction_results['fdr_rate']:.4f}")
        
        return {
            'corrected_p_values': corrected_p_values,
            'summary': correction_results
        }
    
    def _calculate_returns(self, predictions: np.ndarray, actuals: np.ndarray) -> list:
        """
        Calculate betting returns from predictions and actuals.
        """
        returns = []
        
        for pred, actual in zip(predictions, actuals):
            if pred > 0.5238:  # Betting threshold
                if actual == 1:
                    returns.append(0.91)  # Win at -110 odds
                else:
                    returns.append(-1.0)  # Loss
        
        return returns
    
    def _clone_model(self):
        """
        Clone the model for walk-forward validation.
        """
        from sklearn.base import clone
        return clone(self.model)
    
    def _generate_validation_report(self, results: dict) -> dict:
        """
        Generate comprehensive validation report.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': type(self.model).__name__,
            'n_features': len(self.features),
            'validation_summary': {
                'walk_forward_accuracy': results['walk_forward']['mean_accuracy'],
                'is_statistically_significant': results['significance']['permutation']['is_significant'],
                'sharpe_ratio': results['betting']['sharpe_ratio'],
                'max_drawdown': results['betting']['drawdown']['max_drawdown'],
                'n_regime_changes': results['regime']['cusum']['n_regimes_detected'],
                'feature_stability_rate': results['features']['stability_rate']
            },
            'detailed_results': results
        }
        
        # Add overall pass/fail assessment
        passed_tests = []
        failed_tests = []
        
        # Check key criteria
        if report['validation_summary']['walk_forward_accuracy'] > 0.5238:
            passed_tests.append('Profitable in walk-forward validation')
        else:
            failed_tests.append('Not profitable in walk-forward validation')
        
        if report['validation_summary']['is_statistically_significant']:
            passed_tests.append('Statistically significant edge')
        else:
            failed_tests.append('No statistically significant edge')
        
        if report['validation_summary']['sharpe_ratio'] > 1.0:
            passed_tests.append('Good risk-adjusted returns (Sharpe > 1)')
        else:
            failed_tests.append('Poor risk-adjusted returns (Sharpe < 1)')
        
        if abs(report['validation_summary']['max_drawdown']) < 0.20:
            passed_tests.append('Acceptable drawdown (<20%)')
        else:
            failed_tests.append('High drawdown risk (>20%)')
        
        report['validation_assessment'] = {
            'overall_status': 'PASS' if len(passed_tests) >= 3 else 'FAIL',
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'score': len(passed_tests) / (len(passed_tests) + len(failed_tests))
        }
        
        return report


def example_complete_validation():
    """
    Example of running complete validation on a model.
    """
    print("=" * 80)
    print("EXAMPLE: Complete Model Validation")
    print("=" * 80)
    
    # Generate sample data
    print("\nGenerating sample NFL data...")
    from nfl_validation_framework import generate_sample_nfl_data
    data = generate_sample_nfl_data(n_games=2000)
    
    # Add some features
    for i in range(10):
        data[f'feature_{i}'] = np.random.randn(len(data))
    
    features = [f'feature_{i}' for i in range(10)]
    
    # Train a simple model
    print("Training sample model...")
    X = data[features]
    y = data['covered_spread']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize validator
    validator = NFLBettingModelValidator(model, features)
    
    # Run validation
    print("\nRunning complete validation pipeline...")
    report = validator.run_complete_validation(data)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 80)
    
    summary = report['validation_summary']
    print(f"Model Type: {report['model_type']}")
    print(f"Features: {report['n_features']}")
    print(f"\nKey Metrics:")
    print(f"  Walk-Forward Accuracy: {summary['walk_forward_accuracy']:.4f}")
    print(f"  Statistically Significant: {summary['is_statistically_significant']}")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
    print(f"  Feature Stability: {summary['feature_stability_rate']:.2%}")
    
    print(f"\nValidation Assessment: {report['validation_assessment']['overall_status']}")
    print(f"Score: {report['validation_assessment']['score']:.2%}")
    
    print("\nPassed Tests:")
    for test in report['validation_assessment']['passed_tests']:
        print(f"  ✓ {test}")
    
    print("\nFailed Tests:")
    for test in report['validation_assessment']['failed_tests']:
        print(f"  ✗ {test}")
    
    return validator, report


def example_backtesting_with_regime_detection():
    """
    Example of backtesting with regime change detection.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Backtesting with Regime Detection")
    print("=" * 80)
    
    from nfl_validation_framework import generate_sample_nfl_data
    from nfl_validation_framework import RegimeChangeDetector, BettingMetricsCalculator
    
    # Generate data with regime changes
    data = generate_sample_nfl_data(n_games=2000)
    
    # Simulate regime change (e.g., COVID impact)
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2020-12-31')
    covid_mask = (data['game_date'] >= covid_start) & (data['game_date'] <= covid_end)
    
    # Reduce prediction accuracy during COVID
    data.loc[covid_mask, 'model_prediction'] *= 0.9
    
    # Calculate returns
    returns = []
    for pred, actual in zip(data['model_prediction'], data['covered_spread']):
        if pred > 0.5238:
            if actual == 1:
                returns.append(0.91)
            else:
                returns.append(-1.0)
    
    returns_series = pd.Series(returns[:len(data)], index=data['game_date'])
    
    # Detect regime changes
    detector = RegimeChangeDetector()
    cusum_results = detector.cusum_detection(returns_series)
    
    print(f"\nRegime changes detected: {cusum_results['n_regimes_detected']}")
    
    if cusum_results['regime_changes']:
        for regime in cusum_results['regime_changes']:
            start_date = data['game_date'].iloc[regime['start_idx']]
            end_date = data['game_date'].iloc[regime['end_idx']]
            print(f"  Regime: {start_date.date()} to {end_date.date()}")
            print(f"  Severity: {regime['severity']:.2f}")
    
    # Calculate metrics by regime
    metrics_calc = BettingMetricsCalculator()
    
    print("\nPerformance by period:")
    
    # Pre-COVID
    pre_covid = returns_series[returns_series.index < covid_start]
    if len(pre_covid) > 0:
        sharpe_pre = metrics_calc.calculate_sharpe_ratio(pre_covid)
        print(f"  Pre-COVID Sharpe: {sharpe_pre:.4f}")
    
    # During COVID
    covid_period = returns_series[
        (returns_series.index >= covid_start) & 
        (returns_series.index <= covid_end)
    ]
    if len(covid_period) > 0:
        sharpe_covid = metrics_calc.calculate_sharpe_ratio(covid_period)
        print(f"  COVID Sharpe: {sharpe_covid:.4f}")
    
    # Post-COVID
    post_covid = returns_series[returns_series.index > covid_end]
    if len(post_covid) > 0:
        sharpe_post = metrics_calc.calculate_sharpe_ratio(post_covid)
        print(f"  Post-COVID Sharpe: {sharpe_post:.4f}")
    
    return cusum_results


if __name__ == "__main__":
    print("""
    NFL Betting Model Validation - Usage Examples
    =============================================
    
    This script demonstrates practical usage of the validation framework
    for NFL betting models.
    """)
    
    # Run examples
    print("\n1. Complete Validation Example")
    validator, report = example_complete_validation()
    
    print("\n2. Regime Detection Example")
    regime_results = example_backtesting_with_regime_detection()
    
    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
