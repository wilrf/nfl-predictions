#!/bin/bash
# Quick Deploy Script - NFL Betting System
# Usage: ./quick_deploy.sh [preview|production]

set -e  # Exit on error

MODE=${1:-preview}
PROJECT_DIR="/Users/wilfowler/Sports Model"

echo "🚀 NFL Betting System - Quick Deploy"
echo "======================================="
echo "Mode: $MODE"
echo ""

cd "$PROJECT_DIR"

# Check if web server is running (optional pre-deploy test)
if command -v lsof &> /dev/null; then
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Warning: Web server already running on port 8000"
        echo "   This is fine, but may cause issues during deployment"
    fi
fi

# Run quick health check
echo "🏥 Running health check..."
if python3 test_setup.py; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed! Fix issues before deploying."
    exit 1
fi

# Git status
echo ""
echo "📊 Git Status:"
git status --short

# Ask for confirmation
echo ""
if [ "$MODE" = "production" ]; then
    echo "⚠️  WARNING: Deploying to PRODUCTION!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "❌ Deployment cancelled"
        exit 0
    fi
fi

# Deploy
echo ""
echo "🚢 Deploying to Vercel..."
if [ "$MODE" = "production" ]; then
    vercel --prod
else
    vercel
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Test the deployment URL"
echo "2. Check Vercel logs: vercel logs"
echo "3. If issues, rollback: vercel rollback"
