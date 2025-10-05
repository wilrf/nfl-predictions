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

        # Mock game data - will connect to actual system later
        now = datetime.utcnow()
        games = [
            {
                "id": "2025_05_BUF_HOU",
                "home_team": "HOU",
                "away_team": "BUF",
                "week": 5,
                "season": 2025,
                "kickoff": (now + timedelta(hours=2)).isoformat() + "Z",
                "spread_pick": "BUF -2.5",
                "spread_confidence": 72,
                "total_pick": "OVER 47.5",
                "total_confidence": 68,
                "tier": "standard",
                "expected_margin": 3.1
            },
            {
                "id": "2025_05_GB_LAR",
                "home_team": "LAR",
                "away_team": "GB",
                "week": 5,
                "season": 2025,
                "kickoff": (now + timedelta(hours=5)).isoformat() + "Z",
                "spread_pick": "GB -3.5",
                "spread_confidence": 81,
                "total_pick": "UNDER 48.5",
                "total_confidence": 65,
                "tier": "premium",
                "expected_margin": 5.2
            },
            {
                "id": "2025_05_KC_NO",
                "home_team": "NO",
                "away_team": "KC",
                "week": 5,
                "season": 2025,
                "kickoff": (now + timedelta(hours=8)).isoformat() + "Z",
                "spread_pick": "KC -7.5",
                "spread_confidence": 75,
                "total_pick": "OVER 45.5",
                "total_confidence": 70,
                "tier": "standard",
                "expected_margin": 8.4
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
