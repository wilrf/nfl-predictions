#!/usr/bin/env python3
"""
Comprehensive Validation Framework - Phase 4 Implementation
Fail-fast validation gates to ensure model reliability before production
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model combining multiple base models"""
    
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        predictions = []
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(self.weights.values(), predictions))
        
        # Convert to probability matrix
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

class ComprehensiveValidator:
    """Comprehensive validation framework with fail-fast gates"""
    
    def __init__(self):
        self.models_dir = Path('model_architecture/output')
        self.features_dir = Path('feature_engineering/output')
        self.output_dir = Path('validation_framework/output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation thresholds
        self.min_accuracy_threshold = 0.60
        self.min_auc_threshold = 0.65
        self.max_logloss_threshold = 0.80
        self.min_calibration_threshold = 0.80
        self.min_walk_forward_threshold = 0.50
        self.max_drift_threshold = 0.10
        
        logger.info("ComprehensiveValidator initialized")
    
    def run_comprehensive_validation_with_gates(self):
        """Run comprehensive validation with fail-fast gates"""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE VALIDATION FRAMEWORK WITH VALIDATION GATES")
        logger.info("=" * 60)
        
        # Load models and data
        logger.info("\n🔍 VALIDATION GATE 1: Model and Data Loading")
        models_data = self._load_models_and_data()
        if models_data is None:
            logger.error("❌ FAILED: Model and data loading validation failed")
            return None
        logger.info("✅ PASSED: Model and data loading validation")
        
        # Basic performance validation
        logger.info("\n🔍 VALIDATION GATE 2: Basic Performance Validation")
        basic_results = self._validate_basic_performance(models_data)
        if not basic_results['passed']:
            logger.error(f"❌ FAILED: Basic performance validation failed - {basic_results['reason']}")
            return None
        logger.info("✅ PASSED: Basic performance validation")
        
        # Walk-forward validation
        logger.info("\n🔍 VALIDATION GATE 3: Walk-Forward Validation")
        walk_forward_results = self._validate_walk_forward(models_data)
        if not walk_forward_results['passed']:
            logger.error(f"❌ FAILED: Walk-forward validation failed - {walk_forward_results['reason']}")
            return None
        logger.info("✅ PASSED: Walk-forward validation")
        
        # Calibration validation
        logger.info("\n🔍 VALIDATION GATE 4: Calibration Validation")
        calibration_results = self._validate_calibration(models_data)
        if not calibration_results['passed']:
            logger.error(f"❌ FAILED: Calibration validation failed - {calibration_results['reason']}")
            return None
        logger.info("✅ PASSED: Calibration validation")
        
        # Feature importance validation
        logger.info("\n🔍 VALIDATION GATE 5: Feature Importance Validation")
        feature_results = self._validate_feature_importance(models_data)
        if not feature_results['passed']:
            logger.error(f"❌ FAILED: Feature importance validation failed - {feature_results['reason']}")
            return None
        logger.info("✅ PASSED: Feature importance validation")
        
        # Model stability validation
        logger.info("\n🔍 VALIDATION GATE 6: Model Stability Validation")
        stability_results = self._validate_model_stability(models_data)
        if not stability_results['passed']:
            logger.error(f"❌ FAILED: Model stability validation failed - {stability_results['reason']}")
            return None
        logger.info("✅ PASSED: Model stability validation")
        
        # Generate comprehensive report
        logger.info("\n🔍 VALIDATION GATE 7: Comprehensive Report Generation")
        report = self._generate_comprehensive_report(
            basic_results, walk_forward_results, calibration_results,
            feature_results, stability_results, models_data
        )
        
        # Save validation results
        self._save_validation_results(report)
        
        logger.info("\n🎉 ALL VALIDATION GATES PASSED - PROCEEDING TO PHASE 5")
        return report
    
    def _load_models_and_data(self):
        """Load models and data for validation"""
        try:
            # Load ensemble model
            ensemble_file = self.models_dir / 'ensemble_model.pkl'
            if not ensemble_file.exists():
                logger.error("❌ Ensemble model file not found")
                return None
            
            with open(ensemble_file, 'rb') as f:
                ensemble_model = pickle.load(f)
            
            # Load individual models
            individual_models = {}
            for model_name in ['xgboost', 'lightgbm', 'random_forest', 'neural_network']:
                model_file = self.models_dir / f'{model_name}_model.pkl'
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        individual_models[model_name] = pickle.load(f)
            
            # Load model metadata
            metadata_file = self.models_dir / 'model_results.json'
            if not metadata_file.exists():
                logger.error("❌ Model metadata file not found")
                return None
            
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
            
            # Load features
            features_file = self.features_dir / 'comprehensive_features.csv'
            if not features_file.exists():
                logger.error("❌ Features file not found")
                return None
            
            features_df = pd.read_csv(features_file)
            
            logger.info(f"Loaded ensemble model and {len(individual_models)} individual models")
            logger.info(f"Loaded {len(features_df)} games with {len(features_df.columns)} features")
            
            return {
                'ensemble_model': ensemble_model,
                'individual_models': individual_models,
                'model_metadata': model_metadata,
                'features_df': features_df
            }
            
        except Exception as e:
            logger.error(f"❌ Model and data loading failed: {e}")
            return None
    
    def _validate_basic_performance(self, models_data):
        """Validate basic model performance"""
        try:
            ensemble_model = models_data['ensemble_model']
            features_df = models_data['features_df']
            model_metadata = models_data['model_metadata']
            
            # Prepare data splits
            metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_won']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            
            # Create target variable
            features_df['home_won'] = (features_df['home_score'] > features_df['away_score']).astype(int)
            
            # Split by season
            train_df = features_df[features_df['season'].isin([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])]
            val_df = features_df[features_df['season'] == 2024]
            
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df['home_won']
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df['home_won']
            
            # Get predictions
            y_pred_proba = ensemble_model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            logloss = log_loss(y_val, y_pred_proba)
            auc = roc_auc_score(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
            
            # Check thresholds
            passed = (
                accuracy >= self.min_accuracy_threshold and
                auc >= self.min_auc_threshold and
                logloss <= self.max_logloss_threshold
            )
            
            if not passed:
                return {'passed': False, 'reason': f'Performance below thresholds: acc={accuracy:.3f}, auc={auc:.3f}, logloss={logloss:.3f}'}
            
            logger.info(f"✅ Basic performance: acc={accuracy:.3f}, auc={auc:.3f}, logloss={logloss:.3f}")
            
            return {
                'passed': True,
                'accuracy': accuracy,
                'log_loss': logloss,
                'auc_roc': auc,
                'brier_score': brier,
                'validation_games': len(y_val)
            }
            
        except Exception as e:
            return {'passed': False, 'reason': f'Basic performance validation error: {e}'}
    
    def _validate_walk_forward(self, models_data):
        """Validate model using walk-forward analysis"""
        try:
            features_df = models_data['features_df']
            individual_models = models_data['individual_models']
            
            # Prepare data
            metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_won']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            features_df['home_won'] = (features_df['home_score'] > features_df['away_score']).astype(int)
            
            # Sort by season and week
            features_df = features_df.sort_values(['season', 'week'])
            
            # Walk-forward validation
            results = []
            min_train_size = 500
            
            for i in range(min_train_size, len(features_df), 50):  # Test every 50 games
                train_data = features_df.iloc[:i]
                test_data = features_df.iloc[i:i+1]
                
                if len(test_data) == 0:
                    continue
                
                X_train = train_data[feature_cols].fillna(0)
                y_train = train_data['home_won']
                X_test = test_data[feature_cols].fillna(0)
                y_test = test_data['home_won']
                
                # Use Random Forest for quick retraining
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)[0]
                y_actual = y_test.iloc[0]
                
                results.append({
                    'game_id': test_data['game_id'].iloc[0],
                    'season': test_data['season'].iloc[0],
                    'week': test_data['week'].iloc[0],
                    'predicted': y_pred,
                    'actual': y_actual,
                    'correct': y_pred == y_actual
                })
            
            # Calculate walk-forward accuracy
            if len(results) == 0:
                return {'passed': False, 'reason': 'No walk-forward results generated'}
            
            wf_accuracy = sum(r['correct'] for r in results) / len(results)
            
            # Check threshold
            passed = wf_accuracy >= self.min_walk_forward_threshold
            
            if not passed:
                return {'passed': False, 'reason': f'Walk-forward accuracy below threshold: {wf_accuracy:.3f} < {self.min_walk_forward_threshold}'}
            
            logger.info(f"✅ Walk-forward accuracy: {wf_accuracy:.3f} ({len(results)} games tested)")
            
            return {
                'passed': True,
                'accuracy': wf_accuracy,
                'games_tested': len(results),
                'results': results
            }
            
        except Exception as e:
            return {'passed': False, 'reason': f'Walk-forward validation error: {e}'}
    
    def _validate_calibration(self, models_data):
        """Validate model calibration"""
        try:
            ensemble_model = models_data['ensemble_model']
            features_df = models_data['features_df']
            
            # Prepare validation data
            metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_won']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            features_df['home_won'] = (features_df['home_score'] > features_df['away_score']).astype(int)
            
            val_df = features_df[features_df['season'] == 2024]
            X_val = val_df[feature_cols].fillna(0)
            y_val = val_df['home_won']
            
            # Get predictions
            y_pred_proba = ensemble_model.predict_proba(X_val)[:, 1]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_pred_proba, n_bins=10)
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Check threshold
            passed = calibration_error <= (1 - self.min_calibration_threshold)
            
            if not passed:
                return {'passed': False, 'reason': f'Calibration error above threshold: {calibration_error:.3f} > {1 - self.min_calibration_threshold}'}
            
            logger.info(f"✅ Calibration error: {calibration_error:.3f}")
            
            return {
                'passed': True,
                'calibration_error': calibration_error,
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
            
        except Exception as e:
            return {'passed': False, 'reason': f'Calibration validation error: {e}'}
    
    def _validate_feature_importance(self, models_data):
        """Validate feature importance consistency"""
        try:
            individual_models = models_data['individual_models']
            model_metadata = models_data['model_metadata']
            
            # Extract feature importance from models that support it
            feature_importance = {}
            feature_cols = model_metadata['model_metadata']['feature_count']
            
            for name, model in individual_models.items():
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_importance[name] = importance
            
            if not feature_importance:
                return {'passed': False, 'reason': 'No feature importance available'}
            
            # Calculate consistency across models
            importance_matrix = np.array(list(feature_importance.values()))
            correlation_matrix = np.corrcoef(importance_matrix)
            
            # Check if feature importance is consistent across models
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
            # Check threshold
            passed = avg_correlation >= 0.3
            
            if not passed:
                return {'passed': False, 'reason': f'Feature importance correlation below threshold: {avg_correlation:.3f} < 0.3'}
            
            logger.info(f"✅ Feature importance correlation: {avg_correlation:.3f}")
            
            return {
                'passed': True,
                'correlation': avg_correlation,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            return {'passed': False, 'reason': f'Feature importance validation error: {e}'}
    
    def _validate_model_stability(self, models_data):
        """Validate model stability across different data subsets"""
        try:
            ensemble_model = models_data['ensemble_model']
            features_df = models_data['features_df']
            
            # Prepare data
            metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_won']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            features_df['home_won'] = (features_df['home_score'] > features_df['away_score']).astype(int)
            
            # Test stability across different seasons
            season_results = {}
            for season in [2022, 2023, 2024]:
                season_df = features_df[features_df['season'] == season]
                if len(season_df) < 50:  # Need minimum games
                    continue
                
                X_season = season_df[feature_cols].fillna(0)
                y_season = season_df['home_won']
                
                y_pred_proba = ensemble_model.predict_proba(X_season)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                accuracy = accuracy_score(y_season, y_pred)
                season_results[season] = accuracy
            
            if len(season_results) < 2:
                return {'passed': False, 'reason': 'Insufficient seasons for stability testing'}
            
            # Calculate stability (low variance across seasons)
            accuracies = list(season_results.values())
            stability = 1 - np.std(accuracies)  # Higher is more stable
            
            # Check threshold
            passed = stability >= self.min_calibration_threshold
            
            if not passed:
                return {'passed': False, 'reason': f'Model stability below threshold: {stability:.3f} < {self.min_calibration_threshold}'}
            
            logger.info(f"✅ Model stability: {stability:.3f} (seasons: {list(season_results.keys())})")
            
            return {
                'passed': True,
                'stability': stability,
                'season_results': season_results
            }
            
        except Exception as e:
            return {'passed': False, 'reason': f'Model stability validation error: {e}'}
    
    def _generate_comprehensive_report(self, basic_results, walk_forward_results, calibration_results, feature_results, stability_results, models_data):
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'all_gates_passed': True,
                'total_gates': 7,
                'passed_gates': 7
            },
            'basic_performance': basic_results,
            'walk_forward_validation': walk_forward_results,
            'calibration_validation': calibration_results,
            'feature_importance_validation': feature_results,
            'model_stability_validation': stability_results,
            'model_metadata': models_data['model_metadata'],
            'overall_assessment': {
                'model_quality': 'Excellent' if basic_results['accuracy'] > 0.70 else 'Good' if basic_results['accuracy'] > 0.65 else 'Acceptable',
                'production_ready': True,
                'recommendations': self._generate_recommendations(basic_results, walk_forward_results, calibration_results, feature_results, stability_results)
            }
        }
        
        return report
    
    def _generate_recommendations(self, basic_results, walk_forward_results, calibration_results, feature_results, stability_results):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if basic_results['accuracy'] < 0.70:
            recommendations.append("Consider additional feature engineering to improve accuracy")
        
        if walk_forward_results['accuracy'] < 0.60:
            recommendations.append("Model may need more frequent retraining")
        
        if calibration_results['calibration_error'] > 0.1:
            recommendations.append("Consider calibration techniques to improve probability estimates")
        
        if feature_results['correlation'] < 0.5:
            recommendations.append("Feature importance inconsistency suggests model instability")
        
        if stability_results['stability'] < 0.8:
            recommendations.append("Model stability across seasons needs improvement")
        
        if not recommendations:
            recommendations.append("Model is performing well across all validation metrics")
        
        return recommendations
    
    def _save_validation_results(self, report):
        """Save validation results"""
        # Save summary only (avoid circular references)
        summary = {
            'timestamp': report['timestamp'],
            'all_gates_passed': report['validation_summary']['all_gates_passed'],
            'basic_accuracy': float(report['basic_performance']['accuracy']),
            'walk_forward_accuracy': float(report['walk_forward_validation']['accuracy']),
            'calibration_error': float(report['calibration_validation']['calibration_error']),
            'model_stability': float(report['model_stability_validation']['stability']),
            'overall_assessment': report['overall_assessment']
        }
        
        with open(self.output_dir / 'validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Validation results saved to {self.output_dir}")

if __name__ == "__main__":
    validator = ComprehensiveValidator()
    report = validator.run_comprehensive_validation_with_gates()
    
    if report is not None:
        logger.info("🎉 Phase 4 Validation Framework: SUCCESS")
        logger.info("Ready to proceed to Phase 5: Production System")
    else:
        logger.error("❌ Phase 4 Validation Framework: FAILED")
        logger.error("Fix validation issues before proceeding")
