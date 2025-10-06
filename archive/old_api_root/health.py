"""
Health check endpoint for NFL Edge Analytics API
"""
from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Check environment variables
        env_status = {
            "SUPABASE_URL": bool(os.environ.get('SUPABASE_URL')),
            "SUPABASE_KEY": bool(os.environ.get('SUPABASE_KEY')),
            "ODDS_API_KEY": bool(os.environ.get('ODDS_API_KEY'))
        }

        response = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "NFL Edge Analytics",
            "environment": {
                "configured": all(env_status.values()),
                "variables": env_status
            },
            "version": "1.0.0"
        }

        self.wfile.write(json.dumps(response).encode())
        return

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return