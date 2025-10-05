"""
Weekly Performance endpoint for NFL Edge Analytics API
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Get weekly performance data"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Mock weekly performance data - will connect to actual system later
        performance = {
            "weeks": [
                {"week": 1, "win_rate": 0.625, "roi": 5.2, "picks": 8},
                {"week": 2, "win_rate": 0.571, "roi": 3.8, "picks": 7},
                {"week": 3, "win_rate": 0.667, "roi": 8.4, "picks": 9},
                {"week": 4, "win_rate": 0.500, "roi": -1.2, "picks": 10},
                {"week": 5, "win_rate": 0.600, "roi": 6.5, "picks": 5}
            ],
            "season_summary": {
                "total_picks": 39,
                "overall_win_rate": 0.590,
                "overall_roi": 5.8,
                "best_week": 3,
                "worst_week": 4
            }
        }

        self.wfile.write(json.dumps(performance).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
