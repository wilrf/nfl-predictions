#!/bin/bash
# Launch NFL ML Predictions Web Application

echo "=================================================="
echo "   NFL ML PREDICTIONS - WEB APPLICATION"
echo "=================================================="
echo ""
echo "Starting web server..."
echo ""

cd "$(dirname "$0")/web_app"

# Start the server
python3 app.py

# Server will be available at http://localhost:8000
