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

        # Mock stats matching frontend interface
        stats = {
            "total_games": 156,
            "spread_accuracy": 0.582,
            "total_accuracy": 0.547,
            "spread_correct": 91,
            "total_correct": 85,
            "high_confidence_count": 46,
            "high_confidence_accuracy": 0.689
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
