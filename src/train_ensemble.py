#!/usr/bin/env python3
"""
Train Ensemble Model (Random Forest + XGBoost)
Combines Random Forest and XGBoost predictions with weighted voting
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ensemble_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Train ensemble model combining Random Forest and XGBoost"""

    def __init__(self):
        self.data_dir = Path('ml_training_data/consolidated')
        self.models_dir = Path('models/saved_models')
        self.output_dir = Path('models/saved_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature columns for each model
        self.xgb_feature_cols = [
            'is_home', 'week_number', 'is_divisional', 'epa_differential',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor'
        ]

        self.rf_feature_cols = [
            'is_home', 'week_number', 'is_divisional',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'epa_differential', 'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor', 'is_playoff'
        ]

    def load_models(self):
        """Load pre-trained Random Forest and XGBoost models"""
        models = {}
        
        # Load XGBoost models
        try:
            with open(self.models_dir / 'spread_model.pkl', 'rb') as f:
                models['xgboost_spread'] = pickle.load(f)
            with open(self.models_dir / 'spread_calibrator.pkl', 'rb') as f:
                models['xgboost_spread_cal'] = pickle.load(f)
            logger.info("✓ Loaded XGBoost spread model")
        except FileNotFoundError:
            logger.error("XGBoost spread model not found")
            return None
            
        try:
            with open(self.models_dir / 'total_model.pkl', 'rb') as f:
                models['xgboost_total'] = pickle.load(f)
            with open(self.models_dir / 'total_calibrator.pkl', 'rb') as f:
                models['xgboost_total_cal'] = pickle.load(f)
            logger.info("✓ Loaded XGBoost total model")
        except FileNotFoundError:
            logger.error("XGBoost total model not found")
            return None

        # Load Random Forest models
        try:
            with open(self.models_dir / 'random_forest_spread_model.pkl', 'rb') as f:
                models['random_forest_spread'] = pickle.load(f)
            with open(self.models_dir / 'random_forest_spread_calibrator.pkl', 'rb') as f:
                models['random_forest_spread_cal'] = pickle.load(f)
            logger.info("✓ Loaded Random Forest spread model")
        except FileNotFoundError:
            logger.error("Random Forest spread model not found")
            return None
            
        try:
            with open(self.models_dir / 'random_forest_total_model.pkl', 'rb') as f:
                models['random_forest_total'] = pickle.load(f)
            logger.info("✓ Loaded Random Forest total model")
        except FileNotFoundError:
            logger.error("Random Forest total model not found")
            return None

        return models

    def load_data(self):
        """Load train/val/test datasets"""
        train = pd.read_csv(self.data_dir / 'train.csv')
        val = pd.read_csv(self.data_dir / 'validation.csv')
        test = pd.read_csv(self.data_dir / 'test.csv')

        return train, val, test

    def optimize_ensemble_weights(self, models, X_val_xgb, X_val_rf, y_val):
        """Optimize ensemble weights using validation data"""
        logger.info("\nOptimizing ensemble weights...")
        
        # Get predictions from both models
        xgb_probs = models['xgboost_spread_cal'].transform(
            models['xgboost_spread'].predict_proba(X_val_xgb)[:, 1]
        )
        rf_probs = models['random_forest_spread_cal'].predict_proba(X_val_rf)[:, 1]
        
        # Test different weight combinations
        best_weight = 0.5
        best_score = float('inf')
        
        for weight in np.arange(0.1, 0.9, 0.1):
            ensemble_probs = weight * xgb_probs + (1 - weight) * rf_probs
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            # Use log loss as optimization metric
            score = log_loss(y_val, ensemble_probs)
            
            if score < best_score:
                best_score = score
                best_weight = weight
        
        logger.info(f"Optimal XGBoost weight: {best_weight:.2f}")
        logger.info(f"Optimal Random Forest weight: {1-best_weight:.2f}")
        logger.info(f"Best validation log loss: {best_score:.4f}")
        
        return best_weight, 1 - best_weight

    def train_spread_ensemble(self, models, train, val, test):
        """Train ensemble spread model"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING ENSEMBLE SPREAD MODEL")
        logger.info("="*60)

        # Prepare data
        X_train_xgb = train[self.xgb_feature_cols]
        X_train_rf = train[self.rf_feature_cols]
        X_val_xgb = val[self.xgb_feature_cols]
        X_val_rf = val[self.rf_feature_cols]
        X_test_xgb = test[self.xgb_feature_cols]
        X_test_rf = test[self.rf_feature_cols]

        y_train = train['home_won']
        y_val = val['home_won']
        y_test = test['home_won']

        logger.info(f"  Train samples: {len(train):,}")
        logger.info(f"  Validation samples: {len(val):,}")
        logger.info(f"  Test samples: {len(test):,}")

        # Optimize weights
        xgb_weight, rf_weight = self.optimize_ensemble_weights(models, X_val_xgb, X_val_rf, y_val)

        # Create ensemble predictions
        logger.info("\nGenerating ensemble predictions...")
        
        # Validation predictions
        xgb_val_probs = models['xgboost_spread_cal'].transform(
            models['xgboost_spread'].predict_proba(X_val_xgb)[:, 1]
        )
        rf_val_probs = models['random_forest_spread_cal'].predict_proba(X_val_rf)[:, 1]
        ensemble_val_probs = xgb_weight * xgb_val_probs + rf_weight * rf_val_probs
        ensemble_val_preds = (ensemble_val_probs > 0.5).astype(int)

        # Test predictions
        xgb_test_probs = models['xgboost_spread_cal'].transform(
            models['xgboost_spread'].predict_proba(X_test_xgb)[:, 1]
        )
        rf_test_probs = models['random_forest_spread_cal'].predict_proba(X_test_rf)[:, 1]
        ensemble_test_probs = xgb_weight * xgb_test_probs + rf_weight * rf_test_probs
        ensemble_test_preds = (ensemble_test_probs > 0.5).astype(int)

        # Calculate metrics
        logger.info("\nEnsemble Spread Model Performance:")
        
        val_acc = accuracy_score(y_val, ensemble_val_preds)
        val_loss = log_loss(y_val, ensemble_val_probs)
        val_auc = roc_auc_score(y_val, ensemble_val_probs)
        val_brier = brier_score_loss(y_val, ensemble_val_probs)

        test_acc = accuracy_score(y_test, ensemble_test_preds)
        test_loss = log_loss(y_test, ensemble_test_probs)
        test_auc = roc_auc_score(y_test, ensemble_test_probs)

        logger.info(f"  Validation Accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
        logger.info(f"  Validation Log Loss: {val_loss:.3f}")
        logger.info(f"  Validation AUC: {val_auc:.3f}")
        logger.info(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        logger.info(f"  Test Log Loss: {test_loss:.3f}")
        logger.info(f"  Test AUC: {test_auc:.3f}")

        # Save ensemble configuration
        ensemble_config = {
            'model_type': 'ensemble_spread',
            'timestamp': pd.Timestamp.now().isoformat(),
            'weights': {
                'xgboost': float(xgb_weight),
                'random_forest': float(rf_weight)
            },
            'feature_sets': {
                'xgboost': self.xgb_feature_cols,
                'random_forest': self.rf_feature_cols
            },
            'validation_metrics': {
                'accuracy': float(val_acc),
                'log_loss': float(val_loss),
                'auc': float(val_auc),
                'brier_score': float(val_brier)
            },
            'test_metrics': {
                'accuracy': float(test_acc),
                'log_loss': float(test_loss),
                'auc': float(test_auc)
            }
        }

        with open(self.output_dir / 'ensemble_spread_config.json', 'w') as f:
            json.dump(ensemble_config, f, indent=2)

        logger.info("✓ Saved ensemble_spread_config.json")

        return ensemble_config

    def train_total_ensemble(self, models, train, val, test):
        """Train ensemble total model"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING ENSEMBLE TOTAL MODEL")
        logger.info("="*60)

        # Prepare data
        X_train_xgb = train[self.xgb_feature_cols]
        X_train_rf = train[self.rf_feature_cols]
        X_val_xgb = val[self.xgb_feature_cols]
        X_val_rf = val[self.rf_feature_cols]
        X_test_xgb = test[self.xgb_feature_cols]
        X_test_rf = test[self.rf_feature_cols]

        y_train = train['total_points']
        y_val = val['total_points']
        y_test = test['total_points']

        # Convert to binary classification for XGBoost comparison
        median_total = y_train.median()
        y_train_binary = (y_train > median_total).astype(int)
        y_val_binary = (y_val > median_total).astype(int)
        y_test_binary = (y_test > median_total).astype(int)

        logger.info(f"  Median total: {median_total:.1f}")
        logger.info(f"  Train samples: {len(train):,}")
        logger.info(f"  Validation samples: {len(val):,}")
        logger.info(f"  Test samples: {len(test):,}")

        # Get XGBoost predictions (binary classification)
        xgb_val_probs = models['xgboost_total_cal'].transform(
            models['xgboost_total'].predict_proba(X_val_xgb)[:, 1]
        )
        xgb_test_probs = models['xgboost_total_cal'].transform(
            models['xgboost_total'].predict_proba(X_test_xgb)[:, 1]
        )

        # Get Random Forest predictions (regression, convert to probabilities)
        rf_val_preds = models['random_forest_total'].predict(X_val_rf)
        rf_test_preds = models['random_forest_total'].predict(X_test_rf)

        # Convert RF regression predictions to probabilities
        # Use normal distribution assumption
        rf_val_std = np.std(rf_val_preds - y_val)
        rf_test_std = np.std(rf_test_preds - y_test)

        from scipy.stats import norm
        rf_val_probs = 1 - norm.cdf(median_total, loc=rf_val_preds, scale=rf_val_std)
        rf_test_probs = 1 - norm.cdf(median_total, loc=rf_test_preds, scale=rf_test_std)

        # Optimize weights for total model
        logger.info("\nOptimizing total ensemble weights...")
        best_weight = 0.5
        best_score = float('inf')
        
        for weight in np.arange(0.1, 0.9, 0.1):
            ensemble_probs = weight * xgb_val_probs + (1 - weight) * rf_val_probs
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            score = log_loss(y_val_binary, ensemble_probs)
            
            if score < best_score:
                best_score = score
                best_weight = weight

        xgb_weight_total = best_weight
        rf_weight_total = 1 - best_weight

        logger.info(f"Optimal XGBoost weight: {xgb_weight_total:.2f}")
        logger.info(f"Optimal Random Forest weight: {rf_weight_total:.2f}")

        # Create ensemble predictions
        ensemble_val_probs = xgb_weight_total * xgb_val_probs + rf_weight_total * rf_val_probs
        ensemble_test_probs = xgb_weight_total * xgb_test_probs + rf_weight_total * rf_test_probs

        ensemble_val_preds = (ensemble_val_probs > 0.5).astype(int)
        ensemble_test_preds = (ensemble_test_probs > 0.5).astype(int)

        # Calculate metrics
        logger.info("\nEnsemble Total Model Performance:")
        
        val_acc = accuracy_score(y_val_binary, ensemble_val_preds)
        val_loss = log_loss(y_val_binary, ensemble_val_probs)
        val_auc = roc_auc_score(y_val_binary, ensemble_val_probs)

        test_acc = accuracy_score(y_test_binary, ensemble_test_preds)
        test_loss = log_loss(y_test_binary, ensemble_test_probs)
        test_auc = roc_auc_score(y_test_binary, ensemble_test_probs)

        logger.info(f"  Validation Accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
        logger.info(f"  Validation Log Loss: {val_loss:.3f}")
        logger.info(f"  Validation AUC: {val_auc:.3f}")
        logger.info(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        logger.info(f"  Test Log Loss: {test_loss:.3f}")
        logger.info(f"  Test AUC: {test_auc:.3f}")

        # Save ensemble configuration
        ensemble_config = {
            'model_type': 'ensemble_total',
            'timestamp': pd.Timestamp.now().isoformat(),
            'weights': {
                'xgboost': float(xgb_weight_total),
                'random_forest': float(rf_weight_total)
            },
            'feature_sets': {
                'xgboost': self.xgb_feature_cols,
                'random_forest': self.rf_feature_cols
            },
            'median_total': float(median_total),
            'validation_metrics': {
                'accuracy': float(val_acc),
                'log_loss': float(val_loss),
                'auc': float(val_auc)
            },
            'test_metrics': {
                'accuracy': float(test_acc),
                'log_loss': float(test_loss),
                'auc': float(test_auc)
            }
        }

        with open(self.output_dir / 'ensemble_total_config.json', 'w') as f:
            json.dump(ensemble_config, f, indent=2)

        logger.info("✓ Saved ensemble_total_config.json")

        return ensemble_config

    def train_all_ensembles(self):
        """Train both spread and total ensemble models"""
        logger.info("=" * 60)
        logger.info("NFL ENSEMBLE MODEL TRAINING")
        logger.info("=" * 60)

        # Load models
        models = self.load_models()
        if not models:
            logger.error("Failed to load required models")
            return

        # Load data
        logger.info("\nLoading data...")
        train, val, test = self.load_data()
        logger.info(f"  ✓ Train: {len(train):,} games")
        logger.info(f"  ✓ Validation: {len(val):,} games")
        logger.info(f"  ✓ Test: {len(test):,} games")

        # Train ensembles
        spread_config = self.train_spread_ensemble(models, train, val, test)
        total_config = self.train_total_ensemble(models, train, val, test)

        logger.info("\n" + "=" * 60)
        logger.info("✅ ENSEMBLE TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Ensemble configurations saved:")
        logger.info("  - ensemble_spread_config.json")
        logger.info("  - ensemble_total_config.json")
        logger.info("=" * 60)

        return spread_config, total_config


if __name__ == "__main__":
    trainer = EnsembleTrainer()
    trainer.train_all_ensembles()
