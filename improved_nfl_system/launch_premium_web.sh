#!/bin/bash

# Launch Premium NFL Betting Dashboard
# This script starts both the FastAPI backend and Next.js frontend

echo "🚀 Launching Premium NFL Betting Dashboard..."
echo ""

# Check if we're in the right directory
if [ ! -d "web_app" ] || [ ! -d "web_frontend" ]; then
    echo "❌ Error: Must run from improved_nfl_system directory"
    exit 1
fi

# Start FastAPI backend in background
echo "📡 Starting FastAPI backend on port 8000..."
cd web_app
python3 app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Start Next.js frontend
echo "🎨 Starting Next.js frontend on port 3000..."
cd web_frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers are running!"
echo ""
echo "📊 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
