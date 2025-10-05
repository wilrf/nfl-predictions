"""
Confidence Analysis endpoint for NFL Edge Analytics API
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Get confidence bucket analysis"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Mock confidence analysis matching frontend interface
        confidence_data = [
            {
                "bucket": "80-90%",
                "min_confidence": 0.80,
                "count": 12,
                "accuracy": 0.689
            },
            {
                "bucket": "70-79%",
                "min_confidence": 0.70,
                "count": 34,
                "accuracy": 0.618
            },
            {
                "bucket": "60-69%",
                "min_confidence": 0.60,
                "count": 45,
                "accuracy": 0.547
            },
            {
                "bucket": "50-59%",
                "min_confidence": 0.50,
                "count": 65,
                "accuracy": 0.523
            }
        ]

        self.wfile.write(json.dumps(confidence_data).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
