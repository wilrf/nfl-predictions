#!/usr/bin/env python3
"""
Optimized NFL ML Prediction Web Application
FastAPI backend with caching and performance improvements
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import time
from functools import lru_cache
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_integration import NFLModelEnsemble

app = FastAPI(title="NFL ML Predictions", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global variables
models = None
test_data = None
cache = {}
cache_timestamps = {}
CACHE_DURATION = 300  # 5 minutes in seconds
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global models, test_data

    start_time = time.time()

    models_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    models = NFLModelEnsemble(str(models_dir))

    data_path = Path(__file__).parent.parent / 'ml_training_data' / 'consolidated' / 'test.csv'
    test_data = pd.read_csv(data_path)

    # Pre-compute common statistics
    await precompute_stats()

    load_time = time.time() - start_time
    print(f"✓ Models loaded in {load_time:.2f}s")
    print(f"✓ Test data loaded: {len(test_data)} games")

async def precompute_stats():
    """Pre-compute statistics for caching"""
    global cache, cache_timestamps

    # Pre-compute all stats
    cache['stats'] = await compute_stats()
    cache['games'] = await compute_games()
    cache['weekly_performance'] = await compute_weekly_performance()
    cache['confidence_analysis'] = await compute_confidence_analysis()

    # Set cache timestamps
    current_time = time.time()
    for key in cache.keys():
        cache_timestamps[key] = current_time

def get_cached_or_compute(key: str, compute_func):
    """Get cached result or compute if cache is stale"""
    current_time = time.time()

    if key in cache and key in cache_timestamps:
        if current_time - cache_timestamps[key] < CACHE_DURATION:
            return cache[key]

    # Compute and cache
    result = asyncio.run(compute_func())
    cache[key] = result
    cache_timestamps[key] = current_time
    return result

@lru_cache(maxsize=128)
def prepare_features(game_idx: int):
    """Prepare features for prediction with caching"""
    game_row = test_data.iloc[game_idx]

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

async def compute_stats() -> Dict[str, Any]:
    """Compute overall model statistics"""
    spread_correct = 0
    total_correct = 0
    median_total = 44.0
    high_conf_games = []

    # Process in batches for better performance
    batch_size = 50
    for batch_start in range(0, len(test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(test_data))

        for idx in range(batch_start, batch_end):
            game = test_data.iloc[idx]
            features = prepare_features(idx)

            spread_pred = models.predict_spread(features)
            total_pred = models.predict_total(features)

            # Spread accuracy
            if (spread_pred['home_win_prob'] > 0.5) == game['home_won']:
                spread_correct += 1

            # Total accuracy
            if ((total_pred['predicted_total'] > median_total) ==
                (game['total_points'] > median_total)):
                total_correct += 1

            # Track high confidence games
            confidence = max(spread_pred['model_confidence'], total_pred['model_confidence']) * 100
            if confidence >= 75:
                high_conf_games.append({
                    'confidence': confidence,
                    'spread_correct': (spread_pred['home_win_prob'] > 0.5) == game['home_won'],
                    'total_correct': ((total_pred['predicted_total'] > median_total) ==
                                     (game['total_points'] > median_total))
                })

    # Calculate high confidence accuracy
    high_conf_spread_correct = sum(1 for g in high_conf_games if g['spread_correct'])
    high_conf_total = len(high_conf_games)

    return {
        'total_games': len(test_data),
        'spread_accuracy': spread_correct / len(test_data),
        'total_accuracy': total_correct / len(test_data),
        'spread_correct': spread_correct,
        'total_correct': total_correct,
        'high_confidence_count': high_conf_total,
        'high_confidence_accuracy': high_conf_spread_correct / high_conf_total if high_conf_total > 0 else 0
    }

async def compute_games() -> List[Dict[str, Any]]:
    """Compute game predictions and results"""
    games = []
    median_total = 44.0

    # Process games in parallel batches
    batch_size = 20
    for batch_start in range(0, min(len(test_data), 200), batch_size):  # Limit to 200 games for performance
        batch_end = min(batch_start + batch_size, len(test_data))

        batch_games = []
        for idx in range(batch_start, batch_end):
            game = test_data.iloc[idx]
            features = prepare_features(idx)

            spread_pred = models.predict_spread(features)
            total_pred = models.predict_total(features)

            game_data = {
                'game_id': f"2024_W{int(game['week_number'])}_{game['away_team']}_{game['home_team']}",
                'week': int(game['week_number']),
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'away_score': int(game['away_score']) if pd.notna(game['away_score']) else None,
                'home_score': int(game['home_score']) if pd.notna(game['home_score']) else None,
                'spread_prediction': {
                    'predicted_winner': game['home_team'] if spread_pred['home_win_prob'] > 0.5 else game['away_team'],
                    'home_win_prob': round(spread_pred['home_win_prob'], 3),
                    'away_win_prob': round(spread_pred['away_win_prob'], 3),
                    'confidence': round(spread_pred['model_confidence'] * 100, 1),
                    'correct': (spread_pred['home_win_prob'] > 0.5) == game['home_won']
                },
                'total_prediction': {
                    'predicted': 'OVER' if total_pred['predicted_total'] > median_total else 'UNDER',
                    'over_prob': round(total_pred['over_prob'], 3),
                    'under_prob': round(total_pred['under_prob'], 3),
                    'confidence': round(total_pred['model_confidence'] * 100, 1),
                    'correct': ((total_pred['predicted_total'] > median_total) ==
                              (game['total_points'] > median_total))
                },
                'actual_winner': game['home_team'] if game['home_won'] else game['away_team'],
                'total_points': int(game['total_points']) if pd.notna(game['total_points']) else None
            }
            batch_games.append(game_data)

        games.extend(batch_games)

    return games

async def compute_weekly_performance() -> List[Dict[str, Any]]:
    """Compute weekly performance metrics"""
    weekly_data = []
    median_total = 44.0

    for week in sorted(test_data['week_number'].unique()):
        week_games = test_data[test_data['week_number'] == week]
        spread_correct = 0
        total_correct = 0

        for idx in week_games.index:
            game = test_data.loc[idx]
            features = prepare_features(idx)

            spread_pred = models.predict_spread(features)
            total_pred = models.predict_total(features)

            if (spread_pred['home_win_prob'] > 0.5) == game['home_won']:
                spread_correct += 1
            if ((total_pred['predicted_total'] > median_total) ==
                (game['total_points'] > median_total)):
                total_correct += 1

        weekly_data.append({
            'week': int(week),
            'games': len(week_games),
            'spread_accuracy': round(spread_correct / len(week_games), 3) if len(week_games) > 0 else 0,
            'total_accuracy': round(total_correct / len(week_games), 3) if len(week_games) > 0 else 0,
            'spread_correct': spread_correct,
            'total_correct': total_correct
        })

    return weekly_data

async def compute_confidence_analysis() -> List[Dict[str, Any]]:
    """Compute confidence bucket analysis"""
    confidence_buckets = {
        '50-60': {'min': 50, 'max': 60, 'games': []},
        '60-70': {'min': 60, 'max': 70, 'games': []},
        '70-80': {'min': 70, 'max': 80, 'games': []},
        '80-90': {'min': 80, 'max': 90, 'games': []},
        '90+': {'min': 90, 'max': 100, 'games': []}
    }

    for idx in range(len(test_data)):
        game = test_data.iloc[idx]
        features = prepare_features(idx)

        spread_pred = models.predict_spread(features)
        confidence = spread_pred['model_confidence'] * 100

        for bucket_name, bucket_data in confidence_buckets.items():
            if bucket_data['min'] <= confidence < bucket_data['max'] or \
               (bucket_name == '90+' and confidence >= 90):
                bucket_data['games'].append({
                    'correct': (spread_pred['home_win_prob'] > 0.5) == game['home_won']
                })
                break

    results = []
    for bucket_name, bucket_data in confidence_buckets.items():
        if bucket_data['games']:
            correct = sum(1 for g in bucket_data['games'] if g['correct'])
            results.append({
                'bucket': bucket_name,
                'min_confidence': bucket_data['min'],
                'count': len(bucket_data['games']),
                'accuracy': round(correct / len(bucket_data['games']), 3)
            })

    return results

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
    """Get overall model statistics with caching"""
    return JSONResponse(
        content=get_cached_or_compute('stats', compute_stats),
        headers={'Cache-Control': 'public, max-age=60'}
    )

@app.get("/api/games")
async def get_games():
    """Get all game predictions with caching"""
    return JSONResponse(
        content=get_cached_or_compute('games', compute_games),
        headers={'Cache-Control': 'public, max-age=60'}
    )

@app.get("/api/weekly_performance")
async def get_weekly_performance():
    """Get weekly performance metrics with caching"""
    return JSONResponse(
        content=get_cached_or_compute('weekly_performance', compute_weekly_performance),
        headers={'Cache-Control': 'public, max-age=60'}
    )

@app.get("/api/confidence_analysis")
async def get_confidence_analysis():
    """Get confidence bucket analysis with caching"""
    return JSONResponse(
        content=get_cached_or_compute('confidence_analysis', compute_confidence_analysis),
        headers={'Cache-Control': 'public, max-age=60'}
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'models_loaded': models is not None,
        'data_loaded': test_data is not None,
        'cache_entries': len(cache),
        'timestamp': time.time()
    }

@app.get("/api/cache/clear")
async def clear_cache():
    """Clear the cache (admin endpoint)"""
    global cache, cache_timestamps
    cache.clear()
    cache_timestamps.clear()
    # Re-compute after clearing
    await precompute_stats()
    return {'status': 'cache_cleared', 'timestamp': time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)