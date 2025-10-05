#!/usr/bin/env python3
"""
Interactive Model Prediction Demo
Shows real predictions on real 2025 NFL games
"""

import pandas as pd
import numpy as np
from models.model_integration import NFLModelEnsemble
from datetime import datetime

def load_models():
    """Load trained models"""
    print("Loading trained ML models...")
    models = NFLModelEnsemble('models/saved_models')
    print(f"✓ Loaded: {list(models.models.keys())}\n")
    return models

def prepare_features(game_row):
    """Prepare features for prediction"""
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

    features = game_row[feature_cols].to_frame().T.copy()

    # Convert to numeric (required by XGBoost)
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col])

    return features

def predict_game(models, game_row, show_details=True):
    """Make predictions for a single game"""
    features = prepare_features(game_row)

    # Get predictions
    spread_pred = models.predict_spread(features)
    total_pred = models.predict_total(features)

    if show_details:
        print("=" * 70)
        print(f"GAME: {game_row['away_team']} @ {game_row['home_team']}")
        print(f"Week {int(game_row['week'])}, 2025")
        print("=" * 70)

        # Spread prediction
        print(f"\n📊 SPREAD PREDICTION:")
        home_win_prob = spread_pred['home_win_prob']
        away_win_prob = spread_pred['away_win_prob']

        if home_win_prob > 0.5:
            print(f"   Predicted Winner: {game_row['home_team']} (Home)")
            print(f"   Confidence: {home_win_prob*100:.1f}%")
        else:
            print(f"   Predicted Winner: {game_row['away_team']} (Away)")
            print(f"   Confidence: {away_win_prob*100:.1f}%")

        print(f"   Model Confidence: {spread_pred['model_confidence']*100:.1f}%")

        # Total prediction
        print(f"\n📈 TOTAL PREDICTION:")
        over_prob = total_pred['over_prob']
        under_prob = total_pred['under_prob']

        if over_prob > 0.5:
            print(f"   Predicted: OVER")
            print(f"   Confidence: {over_prob*100:.1f}%")
        else:
            print(f"   Predicted: UNDER")
            print(f"   Confidence: {under_prob*100:.1f}%")

        print(f"   Model Confidence: {total_pred['model_confidence']*100:.1f}%")

        # Actual result (if game is completed)
        if pd.notna(game_row['home_score']):
            print(f"\n✅ ACTUAL RESULT:")
            print(f"   Final Score: {game_row['away_team']} {int(game_row['away_score'])} - {int(game_row['home_score'])} {game_row['home_team']}")

            actual_winner = game_row['home_team'] if game_row['home_won'] else game_row['away_team']
            predicted_winner = game_row['home_team'] if home_win_prob > 0.5 else game_row['away_team']

            spread_correct = "✓ CORRECT" if actual_winner == predicted_winner else "✗ WRONG"
            print(f"   Winner: {actual_winner} - Prediction: {spread_correct}")

            actual_total = game_row['total_points']
            median_total = 44.0
            actual_over = "OVER" if actual_total > median_total else "UNDER"
            predicted_over = "OVER" if over_prob > 0.5 else "UNDER"

            total_correct = "✓ CORRECT" if actual_over == predicted_over else "✗ WRONG"
            print(f"   Total: {int(actual_total)} points ({actual_over}) - Prediction: {total_correct}")
        else:
            print(f"\n⏳ Game not yet played")

        print()

    return spread_pred, total_pred

def demo_single_games(models, test_data, num_games=5):
    """Show predictions for individual games"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: INDIVIDUAL GAME PREDICTIONS")
    print("=" * 70)
    print(f"Showing predictions for {num_games} games from 2025 season\n")

    for i in range(min(num_games, len(test_data))):
        predict_game(models, test_data.iloc[i])
        if i < num_games - 1:
            input("Press Enter to see next game...")
            print("\n")

def demo_accuracy_test(models, test_data):
    """Test accuracy across all games"""
    print("\n" + "=" * 70)
    print("ACCURACY TEST: ALL 2025 GAMES")
    print("=" * 70)

    spread_correct = 0
    total_correct = 0
    median_total = 44.0

    predictions = []

    for idx in range(len(test_data)):
        game = test_data.iloc[idx]
        features = prepare_features(game)

        spread_pred = models.predict_spread(features)
        total_pred = models.predict_total(features)

        # Check spread accuracy
        predicted_home_win = spread_pred['home_win_prob'] > 0.5
        actual_home_win = game['home_won']

        if predicted_home_win == actual_home_win:
            spread_correct += 1

        # Check total accuracy
        predicted_over = total_pred['over_prob'] > 0.5
        actual_over = game['total_points'] > median_total

        if predicted_over == actual_over:
            total_correct += 1

        # Store prediction
        predictions.append({
            'game': f"{game['away_team']} @ {game['home_team']}",
            'week': game['week'],
            'spread_correct': predicted_home_win == actual_home_win,
            'total_correct': predicted_over == actual_over,
            'home_win_prob': spread_pred['home_win_prob'],
            'over_prob': total_pred['over_prob']
        })

    spread_acc = spread_correct / len(test_data)
    total_acc = total_correct / len(test_data)

    print(f"\n📊 RESULTS:")
    print(f"   Total games tested: {len(test_data)}")
    print(f"   Spread accuracy: {spread_correct}/{len(test_data)} ({spread_acc*100:.1f}%)")
    print(f"   Total accuracy: {total_correct}/{len(test_data)} ({total_acc*100:.1f}%)")

    # Show by week
    print(f"\n📅 BREAKDOWN BY WEEK:")
    df = pd.DataFrame(predictions)
    for week in sorted(df['week'].unique()):
        week_df = df[df['week'] == week]
        week_spread_acc = week_df['spread_correct'].mean()
        week_total_acc = week_df['total_correct'].mean()
        print(f"   Week {int(week)}: Spread {week_spread_acc*100:.1f}%, Total {week_total_acc*100:.1f}%")

    # Show confidence analysis
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    high_conf = df[df['home_win_prob'].apply(lambda x: abs(x - 0.5) > 0.15)]
    if len(high_conf) > 0:
        high_conf_acc = high_conf['spread_correct'].mean()
        print(f"   High confidence picks (>65% or <35%): {len(high_conf)} games")
        print(f"   High confidence accuracy: {high_conf_acc*100:.1f}%")

    return predictions

def demo_best_predictions(models, test_data):
    """Show the model's best predictions"""
    print("\n" + "=" * 70)
    print("BEST PREDICTIONS: Highest Confidence Games")
    print("=" * 70)

    all_preds = []

    for idx in range(len(test_data)):
        game = test_data.iloc[idx]
        features = prepare_features(game)

        spread_pred = models.predict_spread(features)
        confidence = abs(spread_pred['home_win_prob'] - 0.5) * 2  # Convert to 0-1 scale

        all_preds.append({
            'game': game,
            'confidence': confidence,
            'home_win_prob': spread_pred['home_win_prob'],
            'predicted_winner': game['home_team'] if spread_pred['home_win_prob'] > 0.5 else game['away_team'],
            'actual_winner': game['home_team'] if game['home_won'] else game['away_team']
        })

    # Sort by confidence
    all_preds.sort(key=lambda x: x['confidence'], reverse=True)

    print(f"\nTop 5 Most Confident Predictions:\n")

    for i, pred in enumerate(all_preds[:5], 1):
        game = pred['game']
        correct = "✓" if pred['predicted_winner'] == pred['actual_winner'] else "✗"

        print(f"{i}. {game['away_team']} @ {game['home_team']} (Week {int(game['week'])})")
        print(f"   Predicted: {pred['predicted_winner']} ({pred['home_win_prob']*100:.1f}%)")
        print(f"   Actual: {pred['actual_winner']} {correct}")
        print(f"   Confidence: {pred['confidence']*100:.1f}%")
        print()

def main():
    """Main demo"""
    print("=" * 70)
    print("NFL ML MODEL PREDICTION DEMO")
    print("=" * 70)
    print("\nThis demonstrates the trained XGBoost models making predictions")
    print("on real 2025 NFL games (Weeks 1-4)")
    print()

    # Load models
    models = load_models()

    # Load test data
    print("Loading 2025 test data...")
    test_data = pd.read_csv('ml_training_data/consolidated/test.csv')
    print(f"✓ Loaded {len(test_data)} completed games\n")

    # Menu
    while True:
        print("=" * 70)
        print("CHOOSE A DEMO:")
        print("=" * 70)
        print("1. Show individual game predictions (with details)")
        print("2. Test accuracy across all games")
        print("3. Show best/highest confidence predictions")
        print("4. Predict a specific game")
        print("5. Exit")
        print()

        choice = input("Enter choice (1-5): ").strip()

        if choice == '1':
            num = input("How many games to show? (1-10): ").strip()
            try:
                num = int(num)
                demo_single_games(models, test_data, num)
            except:
                print("Invalid number, showing 5 games")
                demo_single_games(models, test_data, 5)

        elif choice == '2':
            demo_accuracy_test(models, test_data)

        elif choice == '3':
            demo_best_predictions(models, test_data)

        elif choice == '4':
            print("\nAvailable games:")
            for i, row in test_data.iterrows():
                print(f"{i}: Week {int(row['week'])} - {row['away_team']} @ {row['home_team']}")

            try:
                idx = int(input("\nEnter game index: "))
                predict_game(models, test_data.iloc[idx])
            except:
                print("Invalid index")

        elif choice == '5':
            print("\nThanks for testing the models!")
            break

        else:
            print("Invalid choice")

        print()

if __name__ == "__main__":
    main()
