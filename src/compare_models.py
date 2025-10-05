#!/usr/bin/env python3
"""
Compare Random Forest vs XGBoost Models
Comprehensive comparison of model performance, calibration, and feature importance
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare Random Forest vs XGBoost models comprehensively"""

    def __init__(self):
        self.models_dir = Path('models/saved_models')
        self.data_dir = Path('ml_training_data/consolidated')
        self.output_dir = Path('model_comparison')
        self.output_dir.mkdir(exist_ok=True)

    def load_models(self):
        """Load all trained models"""
        models = {}
        
        # Load XGBoost models
        try:
            with open(self.models_dir / 'spread_model.pkl', 'rb') as f:
                models['xgboost_spread'] = pickle.load(f)
            with open(self.models_dir / 'spread_calibrator.pkl', 'rb') as f:
                models['xgboost_spread_cal'] = pickle.load(f)
            logger.info("✓ Loaded XGBoost spread model")
        except FileNotFoundError:
            logger.warning("XGBoost spread model not found")
            
        try:
            with open(self.models_dir / 'total_model.pkl', 'rb') as f:
                models['xgboost_total'] = pickle.load(f)
            with open(self.models_dir / 'total_calibrator.pkl', 'rb') as f:
                models['xgboost_total_cal'] = pickle.load(f)
            logger.info("✓ Loaded XGBoost total model")
        except FileNotFoundError:
            logger.warning("XGBoost total model not found")

        # Load Random Forest models
        try:
            with open(self.models_dir / 'random_forest_spread_model.pkl', 'rb') as f:
                models['random_forest_spread'] = pickle.load(f)
            with open(self.models_dir / 'random_forest_spread_calibrator.pkl', 'rb') as f:
                models['random_forest_spread_cal'] = pickle.load(f)
            logger.info("✓ Loaded Random Forest spread model")
        except FileNotFoundError:
            logger.warning("Random Forest spread model not found")
            
        try:
            with open(self.models_dir / 'random_forest_total_model.pkl', 'rb') as f:
                models['random_forest_total'] = pickle.load(f)
            logger.info("✓ Loaded Random Forest total model")
        except FileNotFoundError:
            logger.warning("Random Forest total model not found")

        return models

    def load_metrics(self):
        """Load model metrics from JSON files"""
        metrics = {}
        
        metric_files = [
            'spread_metrics.json',
            'total_metrics.json', 
            'random_forest_spread_metrics.json',
            'random_forest_total_metrics.json'
        ]
        
        for file in metric_files:
            try:
                with open(self.models_dir / file, 'r') as f:
                    metrics[file.replace('.json', '')] = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Metrics file {file} not found")
                
        return metrics

    def load_test_data(self):
        """Load test data for evaluation"""
        test = pd.read_csv(self.data_dir / 'test.csv')
        
        # Feature columns for XGBoost (original features in correct order)
        xgb_feature_cols = [
            'is_home', 'week_number', 'is_divisional', 'epa_differential',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor'
        ]
        
        # Feature columns for Random Forest (expanded features)
        rf_feature_cols = [
            'is_home', 'week_number', 'is_divisional',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'epa_differential', 'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor', 'is_playoff'
        ]
        
        X_test_xgb = test[xgb_feature_cols]
        X_test_rf = test[rf_feature_cols]
        y_spread = test['home_won']
        y_total = test['total_points']
        
        return X_test_xgb, X_test_rf, y_spread, y_total

    def compare_spread_models(self, models, X_test_xgb, X_test_rf, y_test):
        """Compare spread prediction models"""
        logger.info("\n" + "="*60)
        logger.info("SPREAD MODEL COMPARISON")
        logger.info("="*60)
        
        results = {}
        
        # XGBoost spread
        if 'xgboost_spread' in models and 'xgboost_spread_cal' in models:
            # Raw predictions
            xgb_probs = models['xgboost_spread'].predict_proba(X_test_xgb)[:, 1]
            xgb_preds = (xgb_probs > 0.5).astype(int)
            
            # Calibrated predictions
            xgb_cal_probs = models['xgboost_spread_cal'].transform(xgb_probs)
            xgb_cal_preds = (xgb_cal_probs > 0.5).astype(int)
            
            results['XGBoost (Raw)'] = {
                'accuracy': accuracy_score(y_test, xgb_preds),
                'log_loss': log_loss(y_test, xgb_probs),
                'auc': roc_auc_score(y_test, xgb_probs),
                'brier': brier_score_loss(y_test, xgb_probs)
            }
            
            results['XGBoost (Calibrated)'] = {
                'accuracy': accuracy_score(y_test, xgb_cal_preds),
                'log_loss': log_loss(y_test, xgb_cal_probs),
                'auc': roc_auc_score(y_test, xgb_cal_probs),
                'brier': brier_score_loss(y_test, xgb_cal_probs)
            }
            
            logger.info("XGBoost Spread Model:")
            logger.info(f"  Raw Accuracy: {results['XGBoost (Raw)']['accuracy']:.3f}")
            logger.info(f"  Calibrated Accuracy: {results['XGBoost (Calibrated)']['accuracy']:.3f}")
            logger.info(f"  Calibrated Log Loss: {results['XGBoost (Calibrated)']['log_loss']:.3f}")

        # Random Forest spread
        if 'random_forest_spread' in models and 'random_forest_spread_cal' in models:
            # Raw predictions
            rf_probs = models['random_forest_spread'].predict_proba(X_test_rf)[:, 1]
            rf_preds = (rf_probs > 0.5).astype(int)
            
            # Calibrated predictions
            rf_cal_probs = models['random_forest_spread_cal'].predict_proba(X_test_rf)[:, 1]
            rf_cal_preds = (rf_cal_probs > 0.5).astype(int)
            
            results['Random Forest (Raw)'] = {
                'accuracy': accuracy_score(y_test, rf_preds),
                'log_loss': log_loss(y_test, rf_probs),
                'auc': roc_auc_score(y_test, rf_probs),
                'brier': brier_score_loss(y_test, rf_probs)
            }
            
            results['Random Forest (Calibrated)'] = {
                'accuracy': accuracy_score(y_test, rf_cal_preds),
                'log_loss': log_loss(y_test, rf_cal_probs),
                'auc': roc_auc_score(y_test, rf_cal_probs),
                'brier': brier_score_loss(y_test, rf_cal_probs)
            }
            
            logger.info("Random Forest Spread Model:")
            logger.info(f"  Raw Accuracy: {results['Random Forest (Raw)']['accuracy']:.3f}")
            logger.info(f"  Calibrated Accuracy: {results['Random Forest (Calibrated)']['accuracy']:.3f}")
            logger.info(f"  Calibrated Log Loss: {results['Random Forest (Calibrated)']['log_loss']:.3f}")

        return results

    def compare_total_models(self, models, X_test_xgb, X_test_rf, y_test):
        """Compare total prediction models"""
        logger.info("\n" + "="*60)
        logger.info("TOTAL MODEL COMPARISON")
        logger.info("="*60)
        
        results = {}
        
        # XGBoost total
        if 'xgboost_total' in models and 'xgboost_total_cal' in models:
            # Convert to binary classification for comparison
            median_total = y_test.median()
            y_binary = (y_test > median_total).astype(int)
            
            # Raw predictions
            xgb_probs = models['xgboost_total'].predict_proba(X_test_xgb)[:, 1]
            xgb_preds = (xgb_probs > 0.5).astype(int)
            
            # Calibrated predictions
            xgb_cal_probs = models['xgboost_total_cal'].transform(xgb_probs)
            xgb_cal_preds = (xgb_cal_probs > 0.5).astype(int)
            
            results['XGBoost Total (Raw)'] = {
                'accuracy': accuracy_score(y_binary, xgb_preds),
                'log_loss': log_loss(y_binary, xgb_probs),
                'auc': roc_auc_score(y_binary, xgb_probs),
                'brier': brier_score_loss(y_binary, xgb_probs)
            }
            
            results['XGBoost Total (Calibrated)'] = {
                'accuracy': accuracy_score(y_binary, xgb_cal_preds),
                'log_loss': log_loss(y_binary, xgb_cal_probs),
                'auc': roc_auc_score(y_binary, xgb_cal_probs),
                'brier': brier_score_loss(y_binary, xgb_cal_probs)
            }
            
            logger.info("XGBoost Total Model:")
            logger.info(f"  Raw Accuracy: {results['XGBoost Total (Raw)']['accuracy']:.3f}")
            logger.info(f"  Calibrated Accuracy: {results['XGBoost Total (Calibrated)']['accuracy']:.3f}")

        # Random Forest total
        if 'random_forest_total' in models:
            # Regression predictions
            rf_preds = models['random_forest_total'].predict(X_test_rf)
            rf_binary_preds = (rf_preds > median_total).astype(int)
            
            results['Random Forest Total'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, rf_preds)),
                'r2': r2_score(y_test, rf_preds),
                'mae': np.mean(np.abs(y_test - rf_preds)),
                'accuracy': accuracy_score(y_binary, rf_binary_preds)
            }
            
            logger.info("Random Forest Total Model:")
            logger.info(f"  RMSE: {results['Random Forest Total']['rmse']:.2f}")
            logger.info(f"  R²: {results['Random Forest Total']['r2']:.3f}")
            logger.info(f"  Accuracy (vs median): {results['Random Forest Total']['accuracy']:.3f}")

        return results

    def compare_feature_importance(self, models):
        """Compare feature importance between models"""
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE COMPARISON")
        logger.info("="*60)
        
        # XGBoost feature columns (17 features)
        xgb_feature_cols = [
            'is_home', 'week_number', 'is_divisional', 'epa_differential',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor'
        ]
        
        # Random Forest feature columns (18 features)
        rf_feature_cols = [
            'is_home', 'week_number', 'is_divisional',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'epa_differential', 'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor', 'is_playoff'
        ]
        
        importance_data = []
        
        # XGBoost feature importance
        if 'xgboost_spread' in models:
            xgb_importance = models['xgboost_spread'].feature_importances_
            for i, feature in enumerate(xgb_feature_cols):
                importance_data.append({
                    'feature': feature,
                    'importance': xgb_importance[i],
                    'model': 'XGBoost'
                })
        
        # Random Forest feature importance
        if 'random_forest_spread' in models:
            rf_importance = models['random_forest_spread'].feature_importances_
            for i, feature in enumerate(rf_feature_cols):
                importance_data.append({
                    'feature': feature,
                    'importance': rf_importance[i],
                    'model': 'Random Forest'
                })
        
        if importance_data:
            df_importance = pd.DataFrame(importance_data)
            
            # Top 10 features comparison
            logger.info("Top 10 Features by Model:")
            for model in df_importance['model'].unique():
                model_df = df_importance[df_importance['model'] == model].sort_values('importance', ascending=False)
                logger.info(f"\n{model}:")
                for i, row in model_df.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_data

    def generate_comparison_report(self, spread_results, total_results, importance_data, metrics):
        """Generate comprehensive comparison report"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE MODEL COMPARISON REPORT")
        logger.info("="*60)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'spread_comparison': spread_results,
            'total_comparison': total_results,
            'feature_importance': importance_data,
            'metrics_summary': metrics
        }
        
        # Save report
        with open(self.output_dir / 'model_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary
        logger.info("\nSUMMARY:")
        logger.info("="*40)
        
        if spread_results:
            logger.info("SPREAD MODELS:")
            for model, results in spread_results.items():
                logger.info(f"  {model}:")
                logger.info(f"    Accuracy: {results['accuracy']:.3f}")
                logger.info(f"    Log Loss: {results['log_loss']:.3f}")
                logger.info(f"    AUC: {results['auc']:.3f}")
        
        if total_results:
            logger.info("\nTOTAL MODELS:")
            for model, results in total_results.items():
                logger.info(f"  {model}:")
                if 'accuracy' in results:
                    logger.info(f"    Accuracy: {results['accuracy']:.3f}")
                if 'rmse' in results:
                    logger.info(f"    RMSE: {results['rmse']:.2f}")
                if 'r2' in results:
                    logger.info(f"    R²: {results['r2']:.3f}")
        
        logger.info(f"\nReport saved to: {self.output_dir / 'model_comparison_report.json'}")
        
        return report

    def run_comparison(self):
        """Run complete model comparison"""
        logger.info("Starting model comparison...")
        
        # Load models and data
        models = self.load_models()
        metrics = self.load_metrics()
        X_test_xgb, X_test_rf, y_spread, y_total = self.load_test_data()
        
        # Compare models
        spread_results = self.compare_spread_models(models, X_test_xgb, X_test_rf, y_spread)
        total_results = self.compare_total_models(models, X_test_xgb, X_test_rf, y_total)
        importance_data = self.compare_feature_importance(models)
        
        # Generate report
        report = self.generate_comparison_report(spread_results, total_results, importance_data, metrics)
        
        return report


if __name__ == "__main__":
    comparator = ModelComparator()
    report = comparator.run_comparison()
