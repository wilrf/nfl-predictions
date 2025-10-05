#!/usr/bin/env python3
"""
Retrain Models with 2025 Weeks 1-4 Added to Training Set
This gives us MORE data to learn from for future predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("RETRAINING MODELS WITH EXPANDED DATASET")
print("=" * 70)
print()

# Load data
print("1. Loading datasets...")
train_old = pd.read_csv('ml_training_data/consolidated/train.csv')
val_old = pd.read_csv('ml_training_data/consolidated/validation.csv')
test_2025 = pd.read_csv('ml_training_data/consolidated/test.csv')

print(f"   Old training: {len(train_old):,} games (2015-2023)")
print(f"   Old validation: {len(val_old):,} games (2024)")
print(f"   2025 weeks 1-4: {len(test_2025):,} games")
print()

# Combine all completed games into new training set
print("2. Creating expanded training set...")
new_train = pd.concat([train_old, val_old, test_2025], ignore_index=True)
print(f"   ✓ New training set: {len(new_train):,} games (2015-2025 week 4)")
print()

# Use a subset of latest data as validation
print("3. Creating new validation split...")
# Use last 200 games as validation (mix of 2024-2025)
new_val = new_train.tail(200).copy()
new_train = new_train.head(len(new_train) - 200).copy()

print(f"   ✓ Training: {len(new_train):,} games")
print(f"   ✓ Validation: {len(new_val):,} games")
print()

# Prepare features
feature_cols = [
    'is_home', 'week_number', 'is_divisional',
    'epa_differential',
    'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
    'home_off_success_rate', 'away_off_success_rate',
    'home_redzone_td_pct', 'away_redzone_td_pct',
    'home_third_down_pct', 'away_third_down_pct',
    'home_games_played', 'away_games_played',
    'is_outdoor'
]

X_train = new_train[feature_cols]
X_val = new_val[feature_cols]

# Train spread model
print("4. Training SPREAD model...")
y_train_spread = new_train['home_won']
y_val_spread = new_val['home_won']

spread_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50
)

spread_model.fit(
    X_train, y_train_spread,
    eval_set=[(X_val, y_val_spread)],
    verbose=False
)

val_probs = spread_model.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val_spread, (val_probs > 0.5).astype(int))
print(f"   ✓ Validation accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")

# Calibrate
spread_cal = IsotonicRegression(out_of_bounds='clip')
spread_cal.fit(val_probs, y_val_spread)
print(f"   ✓ Calibrated")
print()

# Train total model
print("5. Training TOTAL model...")
median_total = new_train['total_points'].median()
y_train_total = (new_train['total_points'] > median_total).astype(int)
y_val_total = (new_val['total_points'] > median_total).astype(int)

total_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50
)

total_model.fit(
    X_train, y_train_total,
    eval_set=[(X_val, y_val_total)],
    verbose=False
)

val_probs_total = total_model.predict_proba(X_val)[:, 1]
val_acc_total = accuracy_score(y_val_total, (val_probs_total > 0.5).astype(int))
print(f"   ✓ Validation accuracy: {val_acc_total:.3f} ({val_acc_total*100:.1f}%)")

# Calibrate
total_cal = IsotonicRegression(out_of_bounds='clip')
total_cal.fit(val_probs_total, y_val_total)
print(f"   ✓ Calibrated")
print()

# Save models
print("6. Saving retrained models...")
output_dir = Path('models/saved_models')

with open(output_dir / 'spread_model.pkl', 'wb') as f:
    pickle.dump(spread_model, f)
with open(output_dir / 'spread_calibrator.pkl', 'wb') as f:
    pickle.dump(spread_cal, f)
with open(output_dir / 'total_model.pkl', 'wb') as f:
    pickle.dump(total_model, f)
with open(output_dir / 'total_calibrator.pkl', 'wb') as f:
    pickle.dump(total_cal, f)

print(f"   ✓ Saved to {output_dir}")
print()

# Save updated datasets
print("7. Saving updated dataset splits...")
new_train.to_csv('ml_training_data/consolidated/train.csv', index=False)
new_val.to_csv('ml_training_data/consolidated/validation.csv', index=False)

print(f"   ✓ Saved updated train.csv ({len(new_train):,} games)")
print(f"   ✓ Saved updated validation.csv ({len(new_val):,} games)")
print()

# Summary
print("=" * 70)
print("✅ RETRAINING COMPLETE!")
print("=" * 70)
print()
print(f"📊 New Model Stats:")
print(f"   • Trained on: {len(new_train):,} games (2015-2025)")
print(f"   • Spread accuracy: {val_acc*100:.1f}%")
print(f"   • Total accuracy: {val_acc_total*100:.1f}%")
print()
print(f"🎯 Models now ready to predict 2025 Week 5+ games!")
print("=" * 70)

# Save metadata
metadata = {
    'retrain_date': pd.Timestamp.now().isoformat(),
    'training_games': len(new_train),
    'validation_games': len(new_val),
    'training_range': '2015-2025 week 4',
    'spread_accuracy': float(val_acc),
    'total_accuracy': float(val_acc_total),
    'median_total': float(median_total)
}

with open('models/saved_models/retrain_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved")
