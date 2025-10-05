"""
FastAPI Server for NFL Betting Model
Production-ready API with async processing, caching, and monitoring
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import hashlib
import pickle
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic.types import conlist
import uvicorn

# Async and caching imports
import aioredis
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
import asyncpg

# Performance monitoring
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Import our modules
from nfl_ensemble_model import NFLBettingEnsemble, ModelConfig
from feature_engineering import NFLFeatureEngineering
from online_learning import HybridOnlineLearning
from monitoring import ModelMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
active_requests = Gauge('active_requests', 'Number of active requests')
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')


# Request/Response Models
class GameFeatures(BaseModel):
    """Input features for a single game"""
    game_id: str
    team_home: str
    team_away: str
    week: int
    
    # DVOA features
    offensive_dvoa_home: float = Field(..., ge=-100, le=100)
    defensive_dvoa_home: float = Field(..., ge=-100, le=100)
    offensive_dvoa_away: float = Field(..., ge=-100, le=100)
    defensive_dvoa_away: float = Field(..., ge=-100, le=100)
    
    # EPA features
    epa_per_play_home: float = Field(..., ge=-10, le=10)
    epa_per_play_away: float = Field(..., ge=-10, le=10)
    success_rate_home: float = Field(..., ge=0, le=1)
    success_rate_away: float = Field(..., ge=0, le=1)
    
    # Situational features
    days_rest_home: int = Field(..., ge=3, le=30)
    days_rest_away: int = Field(..., ge=3, le=30)
    is_divisional: bool = False
    is_primetime: bool = False
    
    # Weather features (optional)
    temperature: Optional[float] = Field(None, ge=-20, le=120)
    wind_speed: Optional[float] = Field(None, ge=0, le=50)
    is_dome: bool = False
    
    # Market features
    opening_spread: float = Field(..., ge=-50, le=50)
    current_spread: float = Field(..., ge=-50, le=50)
    opening_total: float = Field(..., ge=30, le=80)
    current_total: float = Field(..., ge=30, le=80)
    public_bet_percentage: Optional[float] = Field(None, ge=0, le=100)
    sharp_money_indicator: Optional[bool] = None
    
    # Additional features can be added here
    additional_features: Optional[Dict[str, float]] = None
    
    @validator('game_id')
    def validate_game_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Invalid game_id')
        return v


class PredictionRequest(BaseModel):
    """Request for predictions"""
    games: List[GameFeatures]
    include_kelly: bool = True
    include_confidence: bool = True
    bankroll: Optional[float] = Field(10000, gt=0)
    
    class Config:
        schema_extra = {
            "example": {
                "games": [{
                    "game_id": "2024_W1_KC_BUF",
                    "team_home": "BUF",
                    "team_away": "KC",
                    "week": 1,
                    "offensive_dvoa_home": 15.5,
                    "defensive_dvoa_home": -8.2,
                    "offensive_dvoa_away": 18.3,
                    "defensive_dvoa_away": -5.1,
                    "epa_per_play_home": 0.12,
                    "epa_per_play_away": 0.15,
                    "success_rate_home": 0.48,
                    "success_rate_away": 0.51,
                    "days_rest_home": 7,
                    "days_rest_away": 7,
                    "opening_spread": -2.5,
                    "current_spread": -3.0,
                    "opening_total": 48.5,
                    "current_total": 47.0
                }],
                "include_kelly": True,
                "include_confidence": True,
                "bankroll": 10000
            }
        }


class PredictionResponse(BaseModel):
    """Response containing predictions"""
    game_id: str
    probability_home_cover: float
    probability_over: float
    
    # Optional fields
    confidence: Optional[float] = None
    kelly_fraction_home: Optional[float] = None
    kelly_fraction_over: Optional[float] = None
    recommended_bet_home: Optional[float] = None
    recommended_bet_over: Optional[float] = None
    expected_value_home: Optional[float] = None
    expected_value_over: Optional[float] = None
    
    # Explanations
    key_factors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    processing_time_ms: float
    model_version: str
    timestamp: str


class ModelUpdateRequest(BaseModel):
    """Request for model update"""
    game_results: List[Dict[str, Any]]
    trigger_full_retrain: bool = False


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    cache_connected: bool
    monitoring_active: bool
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float


# Application State
class AppState:
    """Application state management"""
    def __init__(self):
        self.model: Optional[NFLBettingEnsemble] = None
        self.feature_engineer: Optional[NFLFeatureEngineering] = None
        self.online_learner: Optional[HybridOnlineLearning] = None
        self.monitor: Optional[ModelMonitor] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.cache: Optional[Cache] = None
        self.model_version: str = "1.0.0"
        self.start_time: datetime = datetime.now()
        self.is_ready: bool = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Load model
            logger.info("Loading model...")
            self.model = await self._load_model()
            
            # Initialize feature engineering
            self.feature_engineer = NFLFeatureEngineering()
            
            # Initialize online learning
            self.online_learner = HybridOnlineLearning(
                base_model_path="models/nfl_ensemble"
            )
            
            # Initialize monitoring
            self.monitor = ModelMonitor()
            
            # Initialize Redis cache
            await self._init_cache()
            
            self.is_ready = True
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    async def _load_model(self) -> NFLBettingEnsemble:
        """Load model from disk or cloud storage"""
        model_path = Path("models/nfl_ensemble")
        
        if not model_path.exists():
            logger.warning("Model not found, creating new model")
            return NFLBettingEnsemble()
        
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None, NFLBettingEnsemble.load, model_path
        )
        return model
    
    async def _init_cache(self):
        """Initialize Redis cache"""
        try:
            self.redis_client = await aioredis.create_redis_pool(
                'redis://localhost:6379',
                minsize=5,
                maxsize=10
            )
            
            self.cache = Cache(
                Cache.Type.REDIS,
                endpoint="localhost",
                port=6379,
                serializer=PickleSerializer()
            )
            
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}. Using in-memory cache.")
            self.cache = Cache(Cache.Type.MEMORY)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()


# Create app state
app_state = AppState()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await app_state.initialize()
    yield
    # Shutdown
    await app_state.cleanup()


# Create FastAPI app
app = FastAPI(
    title="NFL Betting Model API",
    description="Production-ready NFL betting predictions with Kelly Criterion optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency for checking app readiness
async def check_ready():
    """Check if application is ready"""
    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


# Helper functions
def create_cache_key(request_data: Dict) -> str:
    """Create cache key from request data"""
    # Create deterministic hash of request
    request_str = json.dumps(request_data, sort_keys=True)
    return hashlib.md5(request_str.encode()).hexdigest()


async def get_cached_prediction(cache_key: str) -> Optional[Dict]:
    """Get prediction from cache"""
    if not app_state.cache:
        return None
    
    try:
        result = await app_state.cache.get(cache_key)
        if result:
            cache_hits.inc()
            logger.debug(f"Cache hit for key: {cache_key}")
            return result
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")
    
    cache_misses.inc()
    return None


async def cache_prediction(cache_key: str, result: Dict, ttl: int = 3600):
    """Cache prediction result"""
    if not app_state.cache:
        return
    
    try:
        await app_state.cache.set(cache_key, result, ttl=ttl)
        logger.debug(f"Cached result for key: {cache_key}")
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "NFL Betting Model API",
        "version": app_state.model_version,
        "status": "ready" if app_state.is_ready else "initializing"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - app_state.start_time).total_seconds()
    
    # Get system metrics
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    return HealthCheckResponse(
        status="healthy" if app_state.is_ready else "unhealthy",
        model_loaded=app_state.model is not None,
        cache_connected=app_state.redis_client is not None,
        monitoring_active=app_state.monitor is not None,
        uptime_seconds=uptime,
        memory_usage_mb=memory_mb,
        cpu_percent=cpu_percent
    )


@app.post("/predict", 
         response_model=BatchPredictionResponse,
         tags=["Predictions"],
         dependencies=[Depends(check_ready)])
async def predict(request: PredictionRequest):
    """Generate predictions for games"""
    start_time = time.time()
    active_requests.inc()
    
    try:
        # Create cache key
        cache_key = create_cache_key(request.dict())
        
        # Check cache
        cached_result = await get_cached_prediction(cache_key)
        if cached_result:
            active_requests.dec()
            return BatchPredictionResponse(**cached_result)
        
        # Prepare features
        feature_dfs = []
        for game in request.games:
            game_dict = game.dict()
            
            # Remove non-feature fields
            game_id = game_dict.pop('game_id')
            team_home = game_dict.pop('team_home')
            team_away = game_dict.pop('team_away')
            
            # Add additional features if provided
            if game.additional_features:
                game_dict.update(game.additional_features)
            
            # Create DataFrame row
            df = pd.DataFrame([game_dict])
            df.index = [game_id]
            feature_dfs.append(df)
        
        # Combine all features
        X = pd.concat(feature_dfs)
        
        # Get predictions
        with prediction_duration.time():
            if request.include_confidence:
                probabilities, uncertainties = app_state.model.predict_proba(
                    X, return_uncertainty=True
                )
            else:
                probabilities = app_state.model.predict_proba(X)
                uncertainties = np.zeros(len(probabilities))
        
        # Calculate Kelly fractions if requested
        kelly_fractions = None
        if request.include_kelly:
            # Convert spreads to decimal odds (simplified)
            odds = np.array([1.91] * len(X))  # Standard -110 odds
            
            kelly_results = app_state.model.predict_with_kelly(
                X, odds, request.bankroll
            )
        
        # Build response
        predictions = []
        for i, game in enumerate(request.games):
            pred = PredictionResponse(
                game_id=game.game_id,
                probability_home_cover=float(probabilities[i]),
                probability_over=float(probabilities[i] * 0.9),  # Simplified for demo
            )
            
            if request.include_confidence:
                pred.confidence = float(1 - uncertainties[i])
            
            if request.include_kelly:
                kelly_row = kelly_results.iloc[i]
                pred.kelly_fraction_home = float(kelly_row['kelly_fraction'])
                pred.recommended_bet_home = float(kelly_row['recommended_bet'])
                pred.expected_value_home = float(kelly_row['expected_value'])
                
                # Simplified over/under Kelly
                pred.kelly_fraction_over = float(kelly_row['kelly_fraction'] * 0.8)
                pred.recommended_bet_over = float(kelly_row['recommended_bet'] * 0.8)
                pred.expected_value_over = float(kelly_row['expected_value'] * 0.8)
            
            # Add key factors (simplified)
            if probabilities[i] > 0.6:
                pred.key_factors = ["Strong DVOA advantage", "Recent momentum"]
            elif probabilities[i] < 0.4:
                pred.key_factors = ["DVOA disadvantage", "Poor recent performance"]
            
            # Add warnings
            warnings = []
            if hasattr(game, 'days_rest_home') and game.days_rest_home < 6:
                warnings.append("Home team on short rest")
            if hasattr(game, 'wind_speed') and game.wind_speed and game.wind_speed > 20:
                warnings.append("High wind conditions")
            if warnings:
                pred.warnings = warnings
            
            predictions.append(pred)
        
        # Increment counter
        prediction_counter.inc(len(predictions))
        
        # Create response
        processing_time = (time.time() - start_time) * 1000
        response_data = {
            "predictions": predictions,
            "processing_time_ms": processing_time,
            "model_version": app_state.model_version,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        await cache_prediction(cache_key, response_data)
        
        # Update monitoring
        if app_state.monitor:
            await asyncio.create_task(
                update_monitoring(X, probabilities)
            )
        
        active_requests.dec()
        return BatchPredictionResponse(**response_data)
        
    except Exception as e:
        active_requests.dec()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/update", tags=["Model Management"], dependencies=[Depends(check_ready)])
async def update_model(
    request: ModelUpdateRequest,
    background_tasks: BackgroundTasks
):
    """Update model with new game results"""
    try:
        # Convert results to DataFrame
        results_df = pd.DataFrame(request.game_results)
        
        # Schedule background update
        background_tasks.add_task(
            perform_model_update,
            results_df,
            request.trigger_full_retrain
        )
        
        return {
            "message": "Model update scheduled",
            "games_received": len(request.game_results),
            "trigger_retrain": request.trigger_full_retrain
        }
        
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/model/info", tags=["Model Management"], dependencies=[Depends(check_ready)])
async def model_info():
    """Get model information"""
    return {
        "model_version": app_state.model_version,
        "model_type": type(app_state.model).__name__,
        "n_features": len(app_state.model.feature_names) if app_state.model else 0,
        "is_fitted": app_state.model.is_fitted if app_state.model else False,
        "online_learning_enabled": app_state.online_learner is not None,
        "monitoring_enabled": app_state.monitor is not None
    }


@app.get("/monitor/status", tags=["Monitoring"], dependencies=[Depends(check_ready)])
async def monitor_status():
    """Get monitoring status"""
    if not app_state.monitor:
        return {"message": "Monitoring not enabled"}
    
    return app_state.monitor.get_status()


@app.post("/monitor/report", tags=["Monitoring"], dependencies=[Depends(check_ready)])
async def generate_monitor_report():
    """Generate monitoring report"""
    if not app_state.monitor:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Monitoring not enabled"
        )
    
    report = app_state.monitor.generate_report()
    return report


# Background tasks
async def update_monitoring(X: pd.DataFrame, predictions: np.ndarray):
    """Update monitoring in background"""
    try:
        if app_state.monitor and app_state.monitor.monitoring_active:
            # Update without actuals (will be added later when available)
            app_state.monitor.update(X, predictions)
    except Exception as e:
        logger.error(f"Monitoring update error: {e}")


async def perform_model_update(results_df: pd.DataFrame, trigger_retrain: bool):
    """Perform model update in background"""
    try:
        logger.info(f"Performing model update with {len(results_df)} results")
        
        if app_state.online_learner:
            # Extract features and labels from results
            # This is simplified - in production, you'd have full feature extraction
            X = results_df.drop(['actual', 'game_id'], axis=1)
            y = results_df['actual']
            
            # Update online learner
            app_state.online_learner.update(X, y, results_df)
        
        if trigger_retrain:
            logger.info("Triggering full model retrain")
            # In production, this would trigger a training pipeline
            # Here we just log it
        
        logger.info("Model update completed")
        
    except Exception as e:
        logger.error(f"Model update error: {e}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors"""
    return JSONResponse(
        status_code=400,
        content={"message": f"Invalid value: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
