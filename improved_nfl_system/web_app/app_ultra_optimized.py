#!/usr/bin/env python3
"""
Ultra-Optimized NFL ML Prediction Web Application
With lazy loading, advanced caching, and streaming responses
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import time
import asyncio
from functools import lru_cache
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import gzip
import hashlib
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# SINGLETON MODEL MANAGER WITH LAZY LOADING
# ============================================================================

class ModelManager:
    """Singleton pattern for lazy model loading"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.models = None
            self.test_data = None
            self.cache = {}
            self.cache_timestamps = {}
            self.loading = False
            self.load_complete = threading.Event()
            self._initialized = True
            self.CACHE_DURATION = 3600  # 1 hour cache

    async def ensure_loaded(self):
        """Ensure models are loaded before use"""
        if self.models is not None:
            return True

        if not self.loading:
            self.loading = True
            # Start loading in background
            loop = asyncio.get_event_loop()
            loop.run_in_executor(executor, self._load_models_sync)

        # Wait for loading to complete
        await asyncio.get_event_loop().run_in_executor(executor, self.load_complete.wait)
        return True

    def _load_models_sync(self):
        """Synchronous model loading"""
        try:
            start_time = time.time()

            # Import here to avoid loading at startup
            from models.model_integration import NFLModelEnsemble

            models_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
            self.models = NFLModelEnsemble(str(models_dir))

            # Load test data
            data_path = Path(__file__).parent.parent / 'ml_training_data' / 'consolidated' / 'test.csv'
            self.test_data = pd.read_csv(data_path)

            # Pre-compute initial cache
            self._precompute_basic_stats()

            load_time = time.time() - start_time
            print(f"✓ Models loaded lazily in {load_time:.2f}s")

        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            self.models = None
            self.test_data = None
        finally:
            self.load_complete.set()

    def _precompute_basic_stats(self):
        """Pre-compute basic statistics"""
        if self.test_data is None:
            return

        # Cache basic stats that don't require models
        self.cache['data_info'] = {
            'total_games': len(self.test_data),
            'weeks': sorted(self.test_data['week_number'].unique().tolist()),
            'teams': sorted(pd.concat([
                self.test_data['home_team'],
                self.test_data['away_team']
            ]).unique().tolist())
        }
        self.cache_timestamps['data_info'] = time.time()

    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached result with TTL check"""
        if key in self.cache and key in self.cache_timestamps:
            if time.time() - self.cache_timestamps[key] < self.CACHE_DURATION:
                return self.cache[key]
        return None

    def set_cache(self, key: str, value: Any):
        """Set cache with timestamp"""
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

# ============================================================================
# OPTIMIZED FASTAPI APP WITH LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI"""
    # Startup
    print("Starting NFL ML API (lazy loading enabled)...")

    # Start background model loading
    manager = ModelManager()
    asyncio.create_task(manager.ensure_loaded())

    yield

    # Shutdown
    print("Shutting down NFL ML API...")

app = FastAPI(
    title="NFL ML Predictions Ultra",
    version="3.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=500)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# OPTIMIZED HELPER FUNCTIONS
# ============================================================================

@lru_cache(maxsize=512)
def prepare_features_cached(game_idx: int, data_hash: str):
    """Cached feature preparation"""
    manager = ModelManager()
    if manager.test_data is None:
        return None

    game_row = manager.test_data.iloc[game_idx]

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

def get_data_hash():
    """Get hash of test data for cache invalidation"""
    manager = ModelManager()
    if manager.test_data is None:
        return "empty"
    return hashlib.md5(pd.util.hash_pandas_object(manager.test_data).values).hexdigest()[:8]

# ============================================================================
# STREAMING RESPONSE GENERATORS
# ============================================================================

async def stream_json_response(data_generator):
    """Stream JSON response in chunks"""
    yield b'{'
    first = True

    async for key, value in data_generator:
        if not first:
            yield b','
        first = False

        chunk = f'"{key}":{json.dumps(value)}'.encode()
        yield chunk

    yield b'}'

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve main page"""
    template_path = Path(__file__).parent / 'templates' / 'index.html'
    if not template_path.exists():
        return JSONResponse({"message": "NFL ML Predictions API", "version": "3.0.0"})
    return FileResponse(template_path)

@app.get("/api/health")
async def health_check():
    """Fast health check endpoint"""
    manager = ModelManager()
    return JSONResponse({
        'status': 'healthy',
        'models_loaded': manager.models is not None,
        'data_loaded': manager.test_data is not None,
        'cache_entries': len(manager.cache),
        'loading': manager.loading,
        'timestamp': time.time()
    }, headers={'Cache-Control': 'no-cache'})

@app.get("/api/stats")
async def get_stats():
    """Get overall statistics with aggressive caching"""
    manager = ModelManager()

    # Check cache first
    cached = manager.get_cached('stats')
    if cached:
        return JSONResponse(cached, headers={'Cache-Control': 'public, max-age=300'})

    # Ensure models loaded
    await manager.ensure_loaded()

    if manager.models is None or manager.test_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Compute stats
    stats = await compute_stats_optimized(manager)

    # Cache result
    manager.set_cache('stats', stats)

    return JSONResponse(stats, headers={'Cache-Control': 'public, max-age=300'})

@app.get("/api/games")
async def get_games(week: Optional[int] = None, limit: int = 20, offset: int = 0):
    """Get games with pagination"""
    manager = ModelManager()

    # Cache key with parameters
    cache_key = f'games_{week}_{limit}_{offset}'
    cached = manager.get_cached(cache_key)
    if cached:
        return JSONResponse(cached, headers={'Cache-Control': 'public, max-age=60'})

    await manager.ensure_loaded()

    if manager.models is None or manager.test_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Filter by week if specified
    data = manager.test_data
    if week:
        data = data[data['week_number'] == week]

    # Apply pagination
    total_games = len(data)
    data = data.iloc[offset:offset + limit]

    games = await compute_games_batch(manager, data, offset)

    result = {
        'games': games,
        'total': total_games,
        'limit': limit,
        'offset': offset,
        'has_more': offset + limit < total_games
    }

    manager.set_cache(cache_key, result)

    return JSONResponse(result, headers={'Cache-Control': 'public, max-age=60'})

@app.get("/api/weekly_performance")
async def get_weekly_performance():
    """Get weekly performance with caching"""
    manager = ModelManager()

    cached = manager.get_cached('weekly_performance')
    if cached:
        return JSONResponse(cached, headers={'Cache-Control': 'public, max-age=300'})

    await manager.ensure_loaded()

    if manager.models is None or manager.test_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    result = await compute_weekly_performance_optimized(manager)
    manager.set_cache('weekly_performance', result)

    return JSONResponse(result, headers={'Cache-Control': 'public, max-age=300'})

@app.get("/api/confidence_analysis")
async def get_confidence_analysis():
    """Get confidence analysis with caching"""
    manager = ModelManager()

    cached = manager.get_cached('confidence_analysis')
    if cached:
        return JSONResponse(cached, headers={'Cache-Control': 'public, max-age=300'})

    await manager.ensure_loaded()

    if manager.models is None or manager.test_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    result = await compute_confidence_optimized(manager)
    manager.set_cache('confidence_analysis', result)

    return JSONResponse(result, headers={'Cache-Control': 'public, max-age=300'})

@app.get("/api/cache/status")
async def cache_status():
    """Get cache status"""
    manager = ModelManager()

    cache_info = {}
    current_time = time.time()

    for key, timestamp in manager.cache_timestamps.items():
        age = current_time - timestamp
        ttl_remaining = max(0, manager.CACHE_DURATION - age)
        cache_info[key] = {
            'age_seconds': round(age, 1),
            'ttl_remaining': round(ttl_remaining, 1),
            'expired': age > manager.CACHE_DURATION
        }

    return JSONResponse({
        'cache_entries': len(manager.cache),
        'cache_duration': manager.CACHE_DURATION,
        'details': cache_info
    })

@app.post("/api/cache/clear")
async def clear_cache():
    """Clear cache and reload"""
    manager = ModelManager()
    manager.cache.clear()
    manager.cache_timestamps.clear()

    # Trigger background reload
    asyncio.create_task(manager.ensure_loaded())

    return JSONResponse({
        'status': 'cache_cleared',
        'timestamp': time.time()
    })

# ============================================================================
# OPTIMIZED COMPUTATION FUNCTIONS
# ============================================================================

async def compute_stats_optimized(manager: ModelManager) -> Dict:
    """Optimized stats computation with batching"""
    data_hash = get_data_hash()
    spread_correct = 0
    total_correct = 0
    median_total = 44.0
    high_conf_games = []

    # Process in parallel batches
    batch_size = 50
    tasks = []

    for batch_start in range(0, len(manager.test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(manager.test_data))
        tasks.append(process_batch_stats(manager, batch_start, batch_end, data_hash))

    # Wait for all batches
    results = await asyncio.gather(*tasks)

    # Aggregate results
    for result in results:
        spread_correct += result['spread_correct']
        total_correct += result['total_correct']
        high_conf_games.extend(result['high_conf_games'])

    # Calculate final stats
    total_games = len(manager.test_data)
    high_conf_correct = sum(1 for g in high_conf_games if g['spread_correct'])

    return {
        'total_games': total_games,
        'spread_accuracy': round(spread_correct / total_games, 3) if total_games > 0 else 0,
        'total_accuracy': round(total_correct / total_games, 3) if total_games > 0 else 0,
        'spread_correct': spread_correct,
        'total_correct': total_correct,
        'high_confidence_count': len(high_conf_games),
        'high_confidence_accuracy': round(high_conf_correct / len(high_conf_games), 3) if high_conf_games else 0
    }

async def process_batch_stats(manager, start: int, end: int, data_hash: str) -> Dict:
    """Process a batch of games for stats"""
    spread_correct = 0
    total_correct = 0
    median_total = 44.0
    high_conf_games = []

    for idx in range(start, end):
        game = manager.test_data.iloc[idx]
        features = prepare_features_cached(idx, data_hash)

        if features is None:
            continue

        spread_pred = manager.models.predict_spread(features)
        total_pred = manager.models.predict_total(features)

        # Spread accuracy
        if (spread_pred['home_win_prob'] > 0.5) == game['home_won']:
            spread_correct += 1

        # Total accuracy
        if ((total_pred['predicted_total'] > median_total) ==
            (game['total_points'] > median_total)):
            total_correct += 1

        # Track high confidence
        confidence = max(spread_pred['model_confidence'], total_pred['model_confidence']) * 100
        if confidence >= 75:
            high_conf_games.append({
                'confidence': confidence,
                'spread_correct': (spread_pred['home_win_prob'] > 0.5) == game['home_won'],
                'total_correct': ((total_pred['predicted_total'] > median_total) ==
                                (game['total_points'] > median_total))
            })

    return {
        'spread_correct': spread_correct,
        'total_correct': total_correct,
        'high_conf_games': high_conf_games
    }

async def compute_games_batch(manager, data: pd.DataFrame, offset: int) -> List[Dict]:
    """Compute games predictions in batch"""
    games = []
    median_total = 44.0
    data_hash = get_data_hash()

    for relative_idx, (abs_idx, game) in enumerate(data.iterrows()):
        features = prepare_features_cached(abs_idx, data_hash)

        if features is None:
            continue

        spread_pred = manager.models.predict_spread(features)
        total_pred = manager.models.predict_total(features)

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
        games.append(game_data)

    return games

async def compute_weekly_performance_optimized(manager) -> List[Dict]:
    """Optimized weekly performance computation"""
    weekly_data = []
    median_total = 44.0
    data_hash = get_data_hash()

    for week in sorted(manager.test_data['week_number'].unique()):
        week_games = manager.test_data[manager.test_data['week_number'] == week]
        spread_correct = 0
        total_correct = 0

        for idx in week_games.index:
            game = manager.test_data.loc[idx]
            features = prepare_features_cached(idx, data_hash)

            if features is None:
                continue

            spread_pred = manager.models.predict_spread(features)
            total_pred = manager.models.predict_total(features)

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

async def compute_confidence_optimized(manager) -> List[Dict]:
    """Optimized confidence analysis"""
    confidence_buckets = {
        '50-60': {'min': 50, 'max': 60, 'games': []},
        '60-70': {'min': 60, 'max': 70, 'games': []},
        '70-80': {'min': 70, 'max': 80, 'games': []},
        '80-90': {'min': 80, 'max': 90, 'games': []},
        '90+': {'min': 90, 'max': 100, 'games': []}
    }

    data_hash = get_data_hash()

    for idx in range(len(manager.test_data)):
        game = manager.test_data.iloc[idx]
        features = prepare_features_cached(idx, data_hash)

        if features is None:
            continue

        spread_pred = manager.models.predict_spread(features)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")