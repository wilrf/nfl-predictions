#!/usr/bin/env python3
"""
NFL ML Prediction Web Application
FastAPI backend with visualization endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_integration import NFLModelEnsemble

app = FastAPI(title="NFL ML Predictions", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = None
test_data = None

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global models, test_data

    models_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    models = NFLModelEnsemble(str(models_dir))

    data_path = Path(__file__).parent.parent / 'ml_training_data' / 'consolidated' / 'test.csv'
    test_data = pd.read_csv(data_path)

    print("✓ Models loaded")
    print(f"✓ Test data loaded: {len(test_data)} games")

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
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col])

    return features

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve main page"""
    template_path = Path(__file__).parent / 'templates' / 'index.html'
    return FileResponse(template_path)

@app.get("/api/stats")
async def get_stats():
    """Get overall model statistics"""

    spread_correct = 0
    total_correct = 0
    median_total = 44.0

    high_conf_games = []

    for idx in range(len(test_data)):
        game = test_data.iloc[idx]
        features = prepare_features(game)

        spread_pred = models.predict_spread(features)
        total_pred = models.predict_total(features)

        # Spread accuracy
        if (spread_pred['home_win_prob'] > 0.5) == game['home_won']:
            spread_correct += 1

        # Total accuracy
        if (total_pred['over_prob'] > 0.5) == (game['total_points'] > median_total):
            total_correct += 1

        # High confidence games
        confidence = abs(spread_pred['home_win_prob'] - 0.5) * 2
        if confidence > 0.30:
            high_conf_games.append({
                'confidence': confidence,
                'correct': (spread_pred['home_win_prob'] > 0.5) == game['home_won']
            })

    high_conf_acc = sum(1 for g in high_conf_games if g['correct']) / len(high_conf_games) if high_conf_games else 0

    return {
        'total_games': len(test_data),
        'spread_accuracy': spread_correct / len(test_data),
        'total_accuracy': total_correct / len(test_data),
        'spread_correct': spread_correct,
        'total_correct': total_correct,
        'high_confidence_count': len(high_conf_games),
        'high_confidence_accuracy': high_conf_acc
    }

@app.get("/api/games")
async def get_games():
    """Get all games with predictions"""

    games = []
    median_total = 44.0

    for idx in range(len(test_data)):
        game = test_data.iloc[idx]
        features = prepare_features(game)

        spread_pred = models.predict_spread(features)
        total_pred = models.predict_total(features)

        predicted_winner = game['home_team'] if spread_pred['home_win_prob'] > 0.5 else game['away_team']
        actual_winner = game['home_team'] if game['home_won'] else game['away_team']

        predicted_total = "OVER" if total_pred['over_prob'] > 0.5 else "UNDER"
        actual_total = "OVER" if game['total_points'] > median_total else "UNDER"

        games.append({
            'game_id': game['game_id'],
            'week': int(game['week']),
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'away_score': int(game['away_score']) if pd.notna(game['away_score']) else None,
            'home_score': int(game['home_score']) if pd.notna(game['home_score']) else None,
            'spread_prediction': {
                'predicted_winner': predicted_winner,
                'home_win_prob': float(spread_pred['home_win_prob']),
                'away_win_prob': float(spread_pred['away_win_prob']),
                'confidence': float(spread_pred['model_confidence']),
                'correct': predicted_winner == actual_winner
            },
            'total_prediction': {
                'predicted': predicted_total,
                'over_prob': float(total_pred['over_prob']),
                'under_prob': float(total_pred['under_prob']),
                'confidence': float(total_pred['model_confidence']),
                'correct': predicted_total == actual_total
            },
            'actual_winner': actual_winner,
            'total_points': int(game['total_points']) if pd.notna(game['total_points']) else None
        })

    return games

@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    """Get detailed prediction for a specific game"""

    game = test_data[test_data['game_id'] == game_id]
    if game.empty:
        raise HTTPException(status_code=404, detail="Game not found")

    game = game.iloc[0]
    features = prepare_features(game)

    spread_pred = models.predict_spread(features)
    total_pred = models.predict_total(features)

    return {
        'game_info': {
            'game_id': game['game_id'],
            'week': int(game['week']),
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'away_score': int(game['away_score']) if pd.notna(game['away_score']) else None,
            'home_score': int(game['home_score']) if pd.notna(game['home_score']) else None,
        },
        'spread_prediction': spread_pred,
        'total_prediction': total_pred,
        'team_stats': {
            'home_off_epa': float(game['home_off_epa']),
            'home_def_epa': float(game['home_def_epa']),
            'away_off_epa': float(game['away_off_epa']),
            'away_def_epa': float(game['away_def_epa']),
            'epa_differential': float(game['epa_differential'])
        }
    }

@app.get("/api/weekly_performance")
async def get_weekly_performance():
    """Get week-by-week performance breakdown"""

    weekly_stats = []
    median_total = 44.0

    for week in sorted(test_data['week'].unique()):
        week_games = test_data[test_data['week'] == week]

        spread_correct = 0
        total_correct = 0

        for idx, game in week_games.iterrows():
            features = prepare_features(game)
            spread_pred = models.predict_spread(features)
            total_pred = models.predict_total(features)

            if (spread_pred['home_win_prob'] > 0.5) == game['home_won']:
                spread_correct += 1

            if (total_pred['over_prob'] > 0.5) == (game['total_points'] > median_total):
                total_correct += 1

        weekly_stats.append({
            'week': int(week),
            'games': len(week_games),
            'spread_accuracy': spread_correct / len(week_games),
            'total_accuracy': total_correct / len(week_games),
            'spread_correct': spread_correct,
            'total_correct': total_correct
        })

    return weekly_stats

@app.get("/api/confidence_analysis")
async def get_confidence_analysis():
    """Analyze predictions by confidence level"""

    confidence_buckets = {
        'very_high': {'min': 0.75, 'predictions': []},
        'high': {'min': 0.65, 'predictions': []},
        'medium': {'min': 0.55, 'predictions': []},
        'low': {'min': 0.50, 'predictions': []}
    }

    for idx in range(len(test_data)):
        game = test_data.iloc[idx]
        features = prepare_features(game)
        spread_pred = models.predict_spread(features)

        prob = max(spread_pred['home_win_prob'], spread_pred['away_win_prob'])
        correct = (spread_pred['home_win_prob'] > 0.5) == game['home_won']

        for bucket_name, bucket in confidence_buckets.items():
            if prob >= bucket['min']:
                bucket['predictions'].append(correct)
                break

    results = []
    for bucket_name, bucket in confidence_buckets.items():
        if bucket['predictions']:
            accuracy = sum(bucket['predictions']) / len(bucket['predictions'])
            results.append({
                'bucket': bucket_name,
                'min_confidence': bucket['min'],
                'count': len(bucket['predictions']),
                'accuracy': accuracy
            })

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
