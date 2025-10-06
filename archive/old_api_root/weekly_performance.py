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

        # Mock weekly performance data matching frontend interface
        weekly_data = [
            {
                "week": 1,
                "games": 8,
                "spread_accuracy": 0.625,
                "total_accuracy": 0.500,
                "spread_correct": 5,
                "total_correct": 4
            },
            {
                "week": 2,
                "games": 7,
                "spread_accuracy": 0.571,
                "total_accuracy": 0.429,
                "spread_correct": 4,
                "total_correct": 3
            },
            {
                "week": 3,
                "games": 9,
                "spread_accuracy": 0.667,
                "total_accuracy": 0.556,
                "spread_correct": 6,
                "total_correct": 5
            },
            {
                "week": 4,
                "games": 10,
                "spread_accuracy": 0.500,
                "total_accuracy": 0.600,
                "spread_correct": 5,
                "total_correct": 6
            },
            {
                "week": 5,
                "games": 5,
                "spread_accuracy": 0.600,
                "total_accuracy": 0.400,
                "spread_correct": 3,
                "total_correct": 2
            }
        ]

        self.wfile.write(json.dumps(weekly_data).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
