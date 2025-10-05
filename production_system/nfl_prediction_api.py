#!/usr/bin/env python3
"""
NFL Prediction API - Phase 5 Implementation
Production-ready FastAPI service with comprehensive monitoring
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import aiohttp
from contextlib import asynccontextmanager
import redis
import psutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and data
models = {}
feature_columns = []
model_metadata = {}
redis_client = None

class EnsembleModel:
    """Ensemble model combining multiple base models"""
    
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        predictions = []
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(self.weights.values(), predictions))
        
        # Convert to probability matrix
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

# Pydantic models for API
class GamePredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team abbreviation")
    away_team: str = Field(..., description="Away team abbreviation")
    season: int = Field(..., description="NFL season year")
    week: int = Field(..., description="Week number")
    spread_line: Optional[float] = Field(None, description="Spread line")
    total_line: Optional[float] = Field(None, description="Total line")
    temperature: Optional[float] = Field(None, description="Temperature in Fahrenheit")
    wind_speed: Optional[float] = Field(None, description="Wind speed in mph")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    precipitation: Optional[float] = Field(None, description="Precipitation amount")

class GamePredictionResponse(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int
    home_win_probability: float = Field(..., description="Probability of home team winning")
    away_win_probability: float = Field(..., description="Probability of away team winning")
    confidence: float = Field(..., description="Model confidence (0-1)")
    prediction: str = Field(..., description="Predicted winner")
    model_metadata: Dict[str, Any] = Field(..., description="Model information")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    redis_connected: bool
    memory_usage: float
    cpu_usage: float
    uptime: float

class SystemMetrics(BaseModel):
    total_predictions: int
    average_response_time: float
    error_rate: float
    last_prediction_time: Optional[str]
    model_accuracy: Optional[float]

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting NFL Prediction API...")
    await load_models()
    await setup_redis()
    logger.info("NFL Prediction API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NFL Prediction API...")
    if redis_client:
        await redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="NFL Prediction API",
    description="Production-ready NFL game prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metrics
start_time = time.time()
prediction_count = 0
total_response_time = 0
error_count = 0
last_prediction_time = None

async def load_models():
    """Load models and metadata"""
    global models, feature_columns, model_metadata
    
    try:
        models_dir = Path('model_architecture/output')
        
        # Load ensemble model
        ensemble_file = models_dir / 'ensemble_model.pkl'
        if not ensemble_file.exists():
            raise FileNotFoundError("Ensemble model not found")
        
        with open(ensemble_file, 'rb') as f:
            models['ensemble'] = pickle.load(f)
        
        # Load individual models
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'neural_network']:
            model_file = models_dir / f'{model_name}_model.pkl'
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    models[model_name] = pickle.load(f)
        
        # Load model metadata
        metadata_file = models_dir / 'model_results.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
        
        # Load feature columns from validation
        validation_dir = Path('validation_framework/output')
        summary_file = validation_dir / 'validation_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                validation_summary = json.load(f)
                # Extract feature count from model metadata
                feature_count = model_metadata.get('model_metadata', {}).get('feature_count', 46)
                feature_columns = [f'feature_{i}' for i in range(feature_count)]
        
        logger.info(f"Loaded {len(models)} models successfully")
        logger.info(f"Feature columns: {len(feature_columns)}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

async def setup_redis():
    """Setup Redis connection for caching and metrics"""
    global redis_client
    
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        # Test connection
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None

def create_feature_vector(request: GamePredictionRequest) -> np.ndarray:
    """Create feature vector from request"""
    # Initialize feature vector with zeros
    features = np.zeros(len(feature_columns))
    
    # Map request fields to feature indices (simplified mapping)
    feature_mapping = {
        'spread_line': 0,
        'total_line': 1,
        'week_number': 3,
        'temperature': 16,
        'wind_speed': 17,
        'humidity': 18,
        'precipitation': 19
    }
    
    # Set known features
    if request.spread_line is not None:
        features[feature_mapping['spread_line']] = request.spread_line
    if request.total_line is not None:
        features[feature_mapping['total_line']] = request.total_line
    if request.week is not None:
        features[feature_mapping['week_number']] = request.week
    if request.temperature is not None:
        features[feature_mapping['temperature']] = request.temperature
    if request.wind_speed is not None:
        features[feature_mapping['wind_speed']] = request.wind_speed
    if request.humidity is not None:
        features[feature_mapping['humidity']] = request.humidity
    if request.precipitation is not None:
        features[feature_mapping['precipitation']] = request.precipitation
    
    # Set default values for other features
    features[2] = 1  # is_home
    features[4] = 0  # is_divisional
    features[5] = 0  # is_playoff
    features[6] = 1  # is_outdoor
    features[7] = 7  # rest_days_home
    features[8] = 7  # rest_days_away
    features[9] = 0  # rest_advantage
    features[10] = request.week / 18.0  # season_progress
    
    return features.reshape(1, -1)

async def log_prediction(request: GamePredictionRequest, response: GamePredictionResponse, response_time: float):
    """Log prediction for monitoring"""
    global prediction_count, total_response_time, last_prediction_time
    
    prediction_count += 1
    total_response_time += response_time
    last_prediction_time = datetime.now().isoformat()
    
    # Log to Redis if available
    if redis_client:
        try:
            prediction_log = {
                'timestamp': datetime.now().isoformat(),
                'home_team': request.home_team,
                'away_team': request.away_team,
                'prediction': response.prediction,
                'confidence': response.confidence,
                'response_time': response_time
            }
            redis_client.lpush('prediction_logs', json.dumps(prediction_log))
            redis_client.ltrim('prediction_logs', 0, 999)  # Keep last 1000 predictions
        except Exception as e:
            logger.warning(f"Failed to log to Redis: {e}")

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "NFL Prediction API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=len(models) > 0,
        redis_connected=redis_connected,
        memory_usage=memory_usage,
        cpu_usage=cpu_usage,
        uptime=uptime
    )

@app.post("/predict", response_model=GamePredictionResponse)
async def predict_game(request: GamePredictionRequest, background_tasks: BackgroundTasks):
    """Predict NFL game outcome"""
    global error_count
    
    start_time_pred = time.time()
    
    try:
        # Validate request
        if not models:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Create feature vector
        features = create_feature_vector(request)
        
        # Get prediction from ensemble model
        ensemble_model = models['ensemble']
        probabilities = ensemble_model.predict_proba(features)
        
        home_prob = float(probabilities[0][1])
        away_prob = float(probabilities[0][0])
        
        # Determine prediction and confidence
        if home_prob > away_prob:
            prediction = request.home_team
            confidence = abs(home_prob - 0.5) * 2
        else:
            prediction = request.away_team
            confidence = abs(away_prob - 0.5) * 2
        
        # Create response
        response = GamePredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            season=request.season,
            week=request.week,
            home_win_probability=home_prob,
            away_win_probability=away_prob,
            confidence=confidence,
            prediction=prediction,
            model_metadata={
                "model_version": "1.0.0",
                "feature_count": len(feature_columns),
                "ensemble_weights": getattr(ensemble_model, 'weights', {}),
                "prediction_timestamp": datetime.now().isoformat()
            }
        )
        
        # Log prediction
        response_time = time.time() - start_time_pred
        background_tasks.add_task(log_prediction, request, response, response_time)
        
        return response
        
    except Exception as e:
        error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system metrics"""
    avg_response_time = total_response_time / prediction_count if prediction_count > 0 else 0
    error_rate = error_count / prediction_count if prediction_count > 0 else 0
    
    # Get model accuracy from validation summary
    model_accuracy = None
    validation_dir = Path('validation_framework/output')
    summary_file = validation_dir / 'validation_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            validation_summary = json.load(f)
            model_accuracy = validation_summary.get('basic_accuracy')
    
    return SystemMetrics(
        total_predictions=prediction_count,
        average_response_time=avg_response_time,
        error_rate=error_rate,
        last_prediction_time=last_prediction_time,
        model_accuracy=model_accuracy
    )

@app.get("/models/info")
async def get_model_info():
    """Get model information"""
    if not model_metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    
    return {
        "models_loaded": list(models.keys()),
        "feature_count": len(feature_columns),
        "model_metadata": model_metadata,
        "validation_summary": "See /metrics endpoint for validation results"
    }

@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions from Redis"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        predictions = redis_client.lrange('prediction_logs', 0, limit - 1)
        return [json.loads(pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent predictions: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "nfl_prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
