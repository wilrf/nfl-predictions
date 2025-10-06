#!/usr/bin/env python3
"""
Simple Flask API server for local development
"""
from flask import Flask, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "NFL Edge Analytics",
        "version": "1.0.0"
    })

@app.route('/api/stats')
def stats():
    """Stats endpoint"""
    return jsonify({
        "total_games": 156,
        "spread_accuracy": 0.582,
        "total_accuracy": 0.547,
        "avg_confidence": 0.73,
        "profitable_games": 89,
        "total_profit": 1247.50
    })

@app.route('/api/games')
def games():
    """Games endpoint"""
    return jsonify([
        {
            "game_id": "1",
            "week": 1,
            "away_team": "Bills",
            "home_team": "Chiefs",
            "away_score": 24,
            "home_score": 31,
            "spread_prediction": {
                "predicted_winner": "Chiefs",
                "home_win_prob": 0.78,
                "away_win_prob": 0.22,
                "confidence": 0.78,
                "correct": True
            },
            "total_prediction": {
                "predicted": "Over",
                "over_prob": 0.65,
                "under_prob": 0.35,
                "confidence": 0.65,
                "correct": True
            },
            "actual_winner": "Chiefs",
            "total_points": 55
        },
        {
            "game_id": "2",
            "week": 1,
            "away_team": "Eagles",
            "home_team": "Cowboys",
            "away_score": 28,
            "home_score": 24,
            "spread_prediction": {
                "predicted_winner": "Eagles",
                "home_win_prob": 0.35,
                "away_win_prob": 0.65,
                "confidence": 0.65,
                "correct": True
            },
            "total_prediction": {
                "predicted": "Over",
                "over_prob": 0.72,
                "under_prob": 0.28,
                "confidence": 0.72,
                "correct": True
            },
            "actual_winner": "Eagles",
            "total_points": 52
        },
        {
            "game_id": "3",
            "week": 1,
            "away_team": "Ravens",
            "home_team": "Steelers",
            "away_score": 21,
            "home_score": 17,
            "spread_prediction": {
                "predicted_winner": "Ravens",
                "home_win_prob": 0.42,
                "away_win_prob": 0.58,
                "confidence": 0.58,
                "correct": True
            },
            "total_prediction": {
                "predicted": "Under",
                "over_prob": 0.38,
                "under_prob": 0.62,
                "confidence": 0.62,
                "correct": True
            },
            "actual_winner": "Ravens",
            "total_points": 38
        }
    ])

@app.route('/api/weekly_performance')
def weekly_performance():
    """Weekly performance endpoint"""
    return jsonify([
        {"week": 1, "accuracy": 0.65, "profit": 150.00},
        {"week": 2, "accuracy": 0.72, "profit": 230.00},
        {"week": 3, "accuracy": 0.58, "profit": -45.00},
        {"week": 4, "accuracy": 0.69, "profit": 180.00}
    ])

@app.route('/api/confidence_analysis')
def confidence_analysis():
    """Confidence analysis endpoint"""
    return jsonify({
        "high_confidence": {"count": 45, "accuracy": 0.78},
        "medium_confidence": {"count": 67, "accuracy": 0.62},
        "low_confidence": {"count": 44, "accuracy": 0.45}
    })

@app.route('/api/predictions')
def predictions():
    """Predictions endpoint"""
    return jsonify([
        {
            "game_id": 1,
            "prediction": "Home",
            "confidence": 0.78,
            "expected_value": 0.12
        },
        {
            "game_id": 2,
            "prediction": "Away",
            "confidence": 0.65,
            "expected_value": 0.08
        }
    ])

@app.route('/api')
def api_root():
    """API root endpoint"""
    return jsonify({
        "message": "NFL Edge Analytics API",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/stats",
            "/api/games",
            "/api/weekly_performance",
            "/api/confidence_analysis",
            "/api/predictions"
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting NFL Edge Analytics API Server...")
    print("📊 Available endpoints:")
    print("   • http://localhost:8000/api/health")
    print("   • http://localhost:8000/api/stats")
    print("   • http://localhost:8000/api/games")
    print("   • http://localhost:8000/api/weekly_performance")
    print("   • http://localhost:8000/api/confidence_analysis")
    print("   • http://localhost:8000/api/predictions")
    print("   • http://localhost:8000/api")
    print(f"\n🔄 Frontend: http://localhost:3000")
    print(f"🛑 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
