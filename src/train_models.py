#!/usr/bin/env python3
"""
Train Spread and Total Models
Uses XGBoost with isotonic calibration
Realistic validation targets: 52-55% accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NFLModelTrainer:
    """Train spread and total models with proper validation"""

    def __init__(self):
        self.data_dir = Path('ml_training_data/consolidated')
        self.output_dir = Path('models/saved_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature columns (from feature_reference.json)
        self.feature_cols = [
            'is_home', 'week_number', 'is_divisional',
            'epa_differential',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played',
            'is_outdoor'
        ]

    def train_all_models(self):
        """Train both spread and total models"""
        logger.info("=" * 60)
        logger.info("NFL MODEL TRAINING")
        logger.info("=" * 60)

        # Load data
        logger.info("\nLoading data...")
        train, val, test = self.load_data()
        logger.info(f"  ✓ Train: {len(train):,} games (2015-2023)")
        logger.info(f"  ✓ Validation: {len(val):,} games (2024)")
        logger.info(f"  ✓ Test: {len(test):,} games (2025)")

        # Train spread model
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SPREAD MODEL")
        logger.info("=" * 60)
        spread_model, spread_cal = self.train_spread_model(train, val, test)

        # Train total model
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING TOTAL MODEL")
        logger.info("=" * 60)
        total_model, total_cal = self.train_total_model(train, val, test)

        logger.info("\n" + "=" * 60)
        logger.info("✅ MODEL TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Models saved:")
        logger.info("  - spread_model.pkl")
        logger.info("  - spread_calibrator.pkl")
        logger.info("  - total_model.pkl")
        logger.info("  - total_calibrator.pkl")
        logger.info("=" * 60)

    def load_data(self) -> tuple:
        """Load train/val/test datasets"""
        train = pd.read_csv(self.data_dir / 'train.csv')
        val = pd.read_csv(self.data_dir / 'validation.csv')
        test = pd.read_csv(self.data_dir / 'test.csv')

        return train, val, test

    def train_spread_model(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Train spread prediction model"""
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

        # Train XGBoost
        logger.info("\n2. Training XGBoost classifier...")

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        logger.info(f"  Best iteration: {model.best_iteration}")

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

        if val_acc < 0.50:
            logger.warning("  ⚠️  Accuracy below 50% - model may need tuning")
        elif val_acc > 0.56:
            logger.warning("  ⚠️  Accuracy above 56% - possible overfitting")
        else:
            logger.info("  ✓ Accuracy in realistic range (50-56%)")

        # Calibrate probabilities
        logger.info("\n4. Calibrating probabilities...")
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_probs, y_val)

        cal_probs = calibrator.transform(val_probs)
        cal_loss = log_loss(y_val, cal_probs)
        cal_brier = brier_score_loss(y_val, cal_probs)

        logger.info(f"  Calibrated Log Loss: {cal_loss:.3f} (uncal: {val_loss:.3f})")
        logger.info(f"  Calibrated Brier: {cal_brier:.3f} (uncal: {val_brier:.3f})")

        # Test on 2025 data
        logger.info("\n5. Testing on 2025 data...")
        test_probs = model.predict_proba(X_test)[:, 1]
        test_cal_probs = calibrator.transform(test_probs)
        test_preds = (test_cal_probs > 0.5).astype(int)

        test_acc = accuracy_score(y_test, test_preds)
        test_loss = log_loss(y_test, test_cal_probs)

        logger.info(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        logger.info(f"  Test Log Loss: {test_loss:.3f}")

        # Save models
        logger.info("\n6. Saving spread model...")
        with open(self.output_dir / 'spread_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(self.output_dir / 'spread_calibrator.pkl', 'wb') as f:
            pickle.dump(calibrator, f)

        logger.info("  ✓ Saved spread_model.pkl")
        logger.info("  ✓ Saved spread_calibrator.pkl")

        # Save metrics
        metrics = {
            'model': 'spread',
            'timestamp': pd.Timestamp.now().isoformat(),
            'training': {
                'samples': len(train),
                'features': len(self.feature_cols),
                'best_iteration': int(model.best_iteration)
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
            }
        }

        with open(self.output_dir / 'spread_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("  ✓ Saved spread_metrics.json")

        return model, calibrator

    def train_total_model(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Train total (over/under) prediction model"""
        logger.info("\n1. Preparing total data...")

        # Features
        X_train = train[self.feature_cols]
        X_val = val[self.feature_cols]
        X_test = test[self.feature_cols]

        # Target: game goes over average total (binary classification)
        # Calculate median total as threshold
        median_total = train['total_points'].median()
        logger.info(f"  Median total points: {median_total:.1f}")

        y_train = (train['total_points'] > median_total).astype(int)
        y_val = (val['total_points'] > median_total).astype(int)
        y_test = (test['total_points'] > median_total).astype(int)

        logger.info(f"  Train over rate: {y_train.mean():.3f}")
        logger.info(f"  Val over rate: {y_val.mean():.3f}")

        # Train XGBoost
        logger.info("\n2. Training XGBoost classifier...")

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        logger.info(f"  Best iteration: {model.best_iteration}")

        # Validate
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

        # Calibrate
        logger.info("\n4. Calibrating probabilities...")
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_probs, y_val)

        cal_probs = calibrator.transform(val_probs)
        cal_loss = log_loss(y_val, cal_probs)

        logger.info(f"  Calibrated Log Loss: {cal_loss:.3f} (uncal: {val_loss:.3f})")

        # Test
        logger.info("\n5. Testing on 2025 data...")
        test_probs = model.predict_proba(X_test)[:, 1]
        test_cal_probs = calibrator.transform(test_probs)
        test_preds = (test_cal_probs > 0.5).astype(int)

        test_acc = accuracy_score(y_test, test_preds)
        test_loss = log_loss(y_test, test_cal_probs)

        logger.info(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        logger.info(f"  Test Log Loss: {test_loss:.3f}")

        # Save models
        logger.info("\n6. Saving total model...")
        with open(self.output_dir / 'total_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(self.output_dir / 'total_calibrator.pkl', 'wb') as f:
            pickle.dump(calibrator, f)

        logger.info("  ✓ Saved total_model.pkl")
        logger.info("  ✓ Saved total_calibrator.pkl")

        # Save metrics
        metrics = {
            'model': 'total',
            'timestamp': pd.Timestamp.now().isoformat(),
            'median_total': float(median_total),
            'training': {
                'samples': len(train),
                'features': len(self.feature_cols),
                'best_iteration': int(model.best_iteration)
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
            }
        }

        with open(self.output_dir / 'total_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("  ✓ Saved total_metrics.json")

        return model, calibrator


if __name__ == "__main__":
    trainer = NFLModelTrainer()
    trainer.train_all_models()
