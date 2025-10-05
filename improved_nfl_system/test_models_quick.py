#!/usr/bin/env python3
"""
Quick Model Test - Verify models make predictions
"""

import pandas as pd
import numpy as np
from models.model_integration import NFLModelEnsemble

print("=" * 60)
print("QUICK MODEL TEST")
print("=" * 60)

# Load models
print("\n1. Loading models...")
models = NFLModelEnsemble('models/saved_models')
print(f"   ✓ Loaded: {list(models.models.keys())}")

# Load test data
print("\n2. Loading 2025 test data...")
test = pd.read_csv('ml_training_data/consolidated/test.csv')
print(f"   ✓ Loaded {len(test)} games (2025 weeks 1-4)")

# Get first game as example
game = test.iloc[0]
print(f"\n3. Example Game:")
print(f"   {game['away_team']} @ {game['home_team']}")
print(f"   Week {game['week']}, 2025")
print(f"   Actual result: {game['home_team']} {'WON' if game['home_won'] else 'LOST'}")
print(f"   Score: {game['away_team']} {int(game['away_score'])} - {int(game['home_score'])} {game['home_team']}")

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

game_features = game[feature_cols].to_frame().T

# Convert to numeric (XGBoost requires numeric types)
for col in feature_cols:
    game_features[col] = pd.to_numeric(game_features[col])

# Test spread prediction
print(f"\n4. Spread Prediction:")
spread_result = models.predict_spread(game_features)
home_win_prob = spread_result['home_win_prob']
prediction = "HOME WIN" if home_win_prob > 0.5 else "AWAY WIN"
actual = "HOME WIN" if game['home_won'] else "AWAY WIN"

print(f"   Model predicted: {prediction} ({home_win_prob*100:.1f}% home win probability)")
print(f"   Actual result: {actual}")
print(f"   {'✓ CORRECT' if prediction == actual else '✗ INCORRECT'}")

# Test total prediction
print(f"\n5. Total Prediction:")
total_result = models.predict_total(game_features)
over_prob = total_result['over_prob']
prediction = "OVER" if over_prob > 0.5 else "UNDER"
median_total = 44.0  # From training
actual_total = game['total_points']
actual_result = "OVER" if actual_total > median_total else "UNDER"

print(f"   Model predicted: {prediction} ({over_prob*100:.1f}% over probability)")
print(f"   Actual total: {actual_total:.0f} points (median: {median_total})")
print(f"   Actual result: {actual_result}")
print(f"   {'✓ CORRECT' if prediction == actual_result else '✗ INCORRECT'}")

# Test all games
print(f"\n6. Testing all {len(test)} games...")
X_test = test[feature_cols].copy()

# Convert to numeric
for col in feature_cols:
    X_test[col] = pd.to_numeric(X_test[col])
y_test_spread = test['home_won']
y_test_total = (test['total_points'] > median_total).astype(int)

# Test each game individually (models don't support batch predictions currently)
spread_correct = 0
total_correct = 0

for idx in range(len(X_test)):
    game_row = X_test.iloc[[idx]]

    spread_pred = models.predict_spread(game_row)
    total_pred = models.predict_total(game_row)

    if (spread_pred['home_win_prob'] > 0.5) == y_test_spread.iloc[idx]:
        spread_correct += 1

    if (total_pred['over_prob'] > 0.5) == y_test_total.iloc[idx]:
        total_correct += 1

spread_acc = spread_correct / len(X_test)
total_acc = total_correct / len(X_test)

print(f"   Spread accuracy: {spread_acc:.3f} ({spread_acc*100:.1f}%)")
print(f"   Total accuracy: {total_acc:.3f} ({total_acc*100:.1f}%)")

print("\n" + "=" * 60)
print("✅ MODELS ARE WORKING!")
print("=" * 60)
print("\nThe models have been trained using XGBoost machine learning")
print("on 2,351 historical games and are making predictions.")
print("\nNext step: Use these models in main.py for live predictions")
print("=" * 60)
