"""
Predictions endpoint for NFL Edge Analytics API
"""
from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from datetime import datetime

# Add lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Get current predictions"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Mock predictions for now - will connect to actual system
        predictions = {
            "week": 5,
            "season": 2025,
            "generated_at": datetime.utcnow().isoformat(),
            "games": [
                {
                    "game_id": "2025_05_BUF_HOU",
                    "home_team": "HOU",
                    "away_team": "BUF",
                    "prediction": {
                        "spread": {
                            "pick": "BUF -2.5",
                            "confidence": 72,
                            "expected_margin": 3.1
                        },
                        "total": {
                            "pick": "OVER 47.5",
                            "confidence": 68,
                            "expected_total": 49.2
                        }
                    },
                    "kickoff": "2025-10-06T17:00:00Z"
                },
                {
                    "game_id": "2025_05_GB_LAR",
                    "home_team": "LAR",
                    "away_team": "GB",
                    "prediction": {
                        "spread": {
                            "pick": "GB -3.5",
                            "confidence": 81,
                            "expected_margin": 5.2
                        },
                        "total": {
                            "pick": "UNDER 48.5",
                            "confidence": 65,
                            "expected_total": 46.8
                        }
                    },
                    "kickoff": "2025-10-06T20:25:00Z"
                }
            ],
            "summary": {
                "total_games": 2,
                "premium_picks": 1,
                "standard_picks": 1,
                "average_confidence": 71.5
            }
        }

        self.wfile.write(json.dumps(predictions).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return