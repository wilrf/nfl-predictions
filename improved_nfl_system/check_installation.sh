#!/bin/bash

echo "🔍 Checking Premium Dashboard Installation..."
echo ""

# Check if we're in the right directory
if [ ! -d "web_frontend" ]; then
    echo "❌ Error: Run this from improved_nfl_system directory"
    exit 1
fi

echo "✓ Directory structure OK"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found - install from https://nodejs.org"
    exit 1
fi
echo "✓ Node.js $(node --version) installed"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found"
    exit 1
fi
echo "✓ npm $(npm --version) installed"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✓ Python $(python3 --version) installed"

# Check if node_modules exists
if [ ! -d "web_frontend/node_modules" ]; then
    echo "⚠️  Frontend dependencies not installed"
    echo "   Run: cd web_frontend && npm install"
else
    echo "✓ Frontend dependencies installed"
fi

# Check if .env.local exists
if [ ! -f "web_frontend/.env.local" ]; then
    echo "⚠️  .env.local not found - creating..."
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > web_frontend/.env.local
    echo "✓ Created .env.local"
else
    echo "✓ .env.local configured"
fi

# Check Python packages
if python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "✓ Backend dependencies (FastAPI, uvicorn) installed"
else
    echo "⚠️  Backend dependencies missing"
    echo "   Run: pip3 install fastapi uvicorn"
fi

# Check for test data
if [ -f "ml_training_data/consolidated/test.csv" ]; then
    echo "✓ Test data found"
else
    echo "⚠️  Test data not found at ml_training_data/consolidated/test.csv"
fi

# Check for models
if [ -d "models/saved_models" ]; then
    echo "✓ Models directory exists"
else
    echo "⚠️  Models directory not found"
fi

echo ""
echo "────────────────────────────────────"
echo ""

# Summary
echo "📋 Installation Summary:"
echo ""
echo "Required:"
echo "  ✓ Node.js & npm"
echo "  ✓ Python 3"
echo "  ✓ Directory structure"
echo ""
echo "To Launch:"
echo "  ./launch_premium_web.sh"
echo ""
echo "Or manually:"
echo "  Terminal 1: cd web_app && python3 app.py"
echo "  Terminal 2: cd web_frontend && npm run dev"
echo ""
echo "────────────────────────────────────"
