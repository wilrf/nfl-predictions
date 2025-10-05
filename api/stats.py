"""
Stats endpoint for NFL Edge Analytics API
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Get overall statistics"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Mock stats - will connect to actual system later
        stats = {
            "total_predictions": 156,
            "win_rate": 0.582,
            "average_confidence": 67.3,
            "premium_win_rate": 0.689,
            "standard_win_rate": 0.547,
            "total_roi": 8.7,
            "current_week": 5,
            "current_season": 2025
        }

        self.wfile.write(json.dumps(stats).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
