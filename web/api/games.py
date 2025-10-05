"""
Games endpoint for NFL Edge Analytics API
"""
from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime, timedelta

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Get all games with predictions"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Mock game data matching frontend interface
        games = [
            {
                "game_id": "2025_05_BUF_HOU",
                "week": 5,
                "away_team": "BUF",
                "home_team": "HOU",
                "away_score": None,
                "home_score": None,
                "spread_prediction": {
                    "predicted_winner": "BUF",
                    "home_win_prob": 0.28,
                    "away_win_prob": 0.72,
                    "confidence": 0.72,
                    "correct": None
                },
                "total_prediction": {
                    "predicted": "OVER",
                    "over_prob": 0.68,
                    "under_prob": 0.32,
                    "confidence": 0.68,
                    "correct": None
                },
                "actual_winner": None,
                "total_points": None
            },
            {
                "game_id": "2025_05_GB_LAR",
                "week": 5,
                "away_team": "GB",
                "home_team": "LAR",
                "away_score": None,
                "home_score": None,
                "spread_prediction": {
                    "predicted_winner": "GB",
                    "home_win_prob": 0.19,
                    "away_win_prob": 0.81,
                    "confidence": 0.81,
                    "correct": None
                },
                "total_prediction": {
                    "predicted": "UNDER",
                    "over_prob": 0.35,
                    "under_prob": 0.65,
                    "confidence": 0.65,
                    "correct": None
                },
                "actual_winner": None,
                "total_points": None
            },
            {
                "game_id": "2025_05_KC_NO",
                "week": 5,
                "away_team": "KC",
                "home_team": "NO",
                "away_score": None,
                "home_score": None,
                "spread_prediction": {
                    "predicted_winner": "KC",
                    "home_win_prob": 0.25,
                    "away_win_prob": 0.75,
                    "confidence": 0.75,
                    "correct": None
                },
                "total_prediction": {
                    "predicted": "OVER",
                    "over_prob": 0.70,
                    "under_prob": 0.30,
                    "confidence": 0.70,
                    "correct": None
                },
                "actual_winner": None,
                "total_points": None
            }
        ]

        self.wfile.write(json.dumps(games).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
