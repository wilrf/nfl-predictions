#!/usr/bin/env python3
"""
Train Random Forest Models for NFL Betting
Implements Random Forest classifiers and regressors with proper validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, mean_squared_error, r2_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import pickle
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/random_forest_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Train Random Forest models with proper validation and calibration"""

    def __init__(self):
        self.data_dir = Path('ml_training_data/consolidated')
        self.output_dir = Path('models/saved_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature columns (expanded from current 8 to 20 available)
        self.feature_cols = [
            'is_home', 'week_number', 'is_divisional',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'epa_differential', 'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played', 'is_outdoor', 'is_playoff'
        ]

    def train_all_models(self):
        """Train both spread and total Random Forest models"""
        logger.info("=" * 60)
        logger.info("NFL RANDOM FOREST MODEL TRAINING")
        logger.info("=" * 60)

        # Load data
        logger.info("\nLoading data...")
        train, val, test = self.load_data()
        logger.info(f"  ✓ Train: {len(train):,} games (2015-2023)")
        logger.info(f"  ✓ Validation: {len(val):,} games (2024)")
        logger.info(f"  ✓ Test: {len(test):,} games (2025)")

        # Train spread model
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RANDOM FOREST SPREAD MODEL")
        logger.info("=" * 60)
        spread_model, spread_cal = self.train_spread_model(train, val, test)

        # Train total model
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RANDOM FOREST TOTAL MODEL")
        logger.info("=" * 60)
        total_model, total_cal = self.train_total_model(train, val, test)

        logger.info("\n" + "=" * 60)
        logger.info("✅ RANDOM FOREST TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Models saved:")
        logger.info("  - random_forest_spread_model.pkl")
        logger.info("  - random_forest_spread_calibrator.pkl")
        logger.info("  - random_forest_total_model.pkl")
        logger.info("  - random_forest_total_calibrator.pkl")
        logger.info("=" * 60)

    def load_data(self) -> tuple:
        """Load train/val/test datasets"""
        train = pd.read_csv(self.data_dir / 'train.csv')
        val = pd.read_csv(self.data_dir / 'validation.csv')
        test = pd.read_csv(self.data_dir / 'test.csv')

        return train, val, test

    def train_spread_model(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Train Random Forest spread prediction model"""
        logger.info("\n1. Preparing spread data...")

        # Features
        X_train = train[self.feature_cols]
        X_val = val[self.feature_cols]
        X_test = test[self.feature_cols]

        # Target: home team wins (binary classification)
        y_train = train['home_won']
        y_val = val['home_won']
        y_test = test['home_won']

        logger.info(f"  Features: {len(self.feature_cols)} columns")
        logger.info(f"  Train target distribution: {y_train.mean():.3f} (home win rate)")
        logger.info(f"  Val target distribution: {y_val.mean():.3f} (home win rate)")

        # Train Random Forest
        logger.info("\n2. Training Random Forest classifier...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)

        logger.info(f"  Trees trained: {model.n_estimators}")
        logger.info(f"  Max depth: {model.max_depth}")

        # Validate on validation set
        logger.info("\n3. Validating on 2024 data...")
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)

        val_acc = accuracy_score(y_val, val_preds)
        val_loss = log_loss(y_val, val_probs)
        val_auc = roc_auc_score(y_val, val_probs)
        val_brier = brier_score_loss(y_val, val_probs)

        logger.info(f"  Accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
        logger.info(f"  Log Loss: {val_loss:.3f}")
        logger.info(f"  AUC-ROC: {val_auc:.3f}")
        logger.info(f"  Brier Score: {val_brier:.3f}")

        # Calibrate probabilities
        logger.info("\n4. Calibrating probabilities...")
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrator.fit(X_val, y_val)

        cal_probs = calibrator.predict_proba(X_val)[:, 1]
        cal_loss = log_loss(y_val, cal_probs)
        cal_brier = brier_score_loss(y_val, cal_probs)

        logger.info(f"  Calibrated Log Loss: {cal_loss:.3f} (uncal: {val_loss:.3f})")
        logger.info(f"  Calibrated Brier: {cal_brier:.3f} (uncal: {val_brier:.3f})")

        # Test on 2025 data
        logger.info("\n5. Testing on 2025 data...")
        test_probs = calibrator.predict_proba(X_test)[:, 1]
        test_preds = (test_probs > 0.5).astype(int)

        test_acc = accuracy_score(y_test, test_preds)
        test_loss = log_loss(y_test, test_probs)

        logger.info(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        logger.info(f"  Test Log Loss: {test_loss:.3f}")

        # Feature importance analysis
        logger.info("\n6. Feature importance analysis...")
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("  Top 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

        # Save models
        logger.info("\n7. Saving Random Forest spread model...")
        with open(self.output_dir / 'random_forest_spread_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(self.output_dir / 'random_forest_spread_calibrator.pkl', 'wb') as f:
            pickle.dump(calibrator, f)

        logger.info("  ✓ Saved random_forest_spread_model.pkl")
        logger.info("  ✓ Saved random_forest_spread_calibrator.pkl")

        # Save metrics
        metrics = {
            'model': 'random_forest_spread',
            'timestamp': pd.Timestamp.now().isoformat(),
            'training': {
                'samples': len(train),
                'features': len(self.feature_cols),
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth
            },
            'validation': {
                'accuracy': float(val_acc),
                'log_loss': float(val_loss),
                'auc_roc': float(val_auc),
                'brier_score': float(val_brier),
                'calibrated_log_loss': float(cal_loss)
            },
            'test': {
                'accuracy': float(test_acc),
                'log_loss': float(test_loss)
            },
            'feature_importance': feature_importance.to_dict('records')
        }

        with open(self.output_dir / 'random_forest_spread_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("  ✓ Saved random_forest_spread_metrics.json")

        return model, calibrator

    def train_total_model(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Train Random Forest total (over/under) prediction model"""
        logger.info("\n1. Preparing total data...")

        # Features
        X_train = train[self.feature_cols]
        X_val = val[self.feature_cols]
        X_test = test[self.feature_cols]

        # Target: total points (regression)
        y_train = train['total_points']
        y_val = val['total_points']
        y_test = test['total_points']

        logger.info(f"  Train target range: {y_train.min():.1f} - {y_train.max():.1f}")
        logger.info(f"  Train target mean: {y_train.mean():.1f}")

        # Train Random Forest Regressor
        logger.info("\n2. Training Random Forest regressor...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        logger.info(f"  Trees trained: {model.n_estimators}")
        logger.info(f"  Max depth: {model.max_depth}")

        # Validate
        logger.info("\n3. Validating on 2024 data...")
        val_preds = model.predict(X_val)

        val_mse = mean_squared_error(y_val, val_preds)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, val_preds)
        val_mae = np.mean(np.abs(y_val - val_preds))

        logger.info(f"  RMSE: {val_rmse:.2f}")
        logger.info(f"  R²: {val_r2:.3f}")
        logger.info(f"  MAE: {val_mae:.2f}")

        # Test
        logger.info("\n4. Testing on 2025 data...")
        test_preds = model.predict(X_test)

        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, test_preds)
        test_mae = np.mean(np.abs(y_test - test_preds))

        logger.info(f"  Test RMSE: {test_rmse:.2f}")
        logger.info(f"  Test R²: {test_r2:.3f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")

        # Feature importance analysis
        logger.info("\n5. Feature importance analysis...")
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("  Top 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

        # Save models
        logger.info("\n6. Saving Random Forest total model...")
        with open(self.output_dir / 'random_forest_total_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        logger.info("  ✓ Saved random_forest_total_model.pkl")

        # Save metrics
        metrics = {
            'model': 'random_forest_total',
            'timestamp': pd.Timestamp.now().isoformat(),
            'training': {
                'samples': len(train),
                'features': len(self.feature_cols),
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth
            },
            'validation': {
                'rmse': float(val_rmse),
                'r2': float(val_r2),
                'mae': float(val_mae),
                'mse': float(val_mse)
            },
            'test': {
                'rmse': float(test_rmse),
                'r2': float(test_r2),
                'mae': float(test_mae),
                'mse': float(test_mse)
            },
            'feature_importance': feature_importance.to_dict('records')
        }

        with open(self.output_dir / 'random_forest_total_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("  ✓ Saved random_forest_total_metrics.json")

        return model, None  # No calibrator for regression


if __name__ == "__main__":
    trainer = RandomForestTrainer()
    trainer.train_all_models()
