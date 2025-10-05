"""
NFL Edge Analytics API
Main serverless function handler for Vercel
"""
from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from urllib.parse import parse_qs, urlparse

# Add lib to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path

        if path == '/api' or path == '/api/':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                "service": "NFL Edge Analytics API",
                "version": "1.0.0",
                "status": "operational",
                "endpoints": [
                    "/api/health",
                    "/api/predictions",
                    "/api/games",
                    "/api/stats"
                ]
            }
            self.wfile.write(json.dumps(response).encode())
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not Found')
        return

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return