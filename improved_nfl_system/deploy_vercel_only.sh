#!/bin/bash

echo "🚀 Deploying NFL Dashboard to Vercel (All-in-One)"
echo "================================================="
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

echo "📁 Navigating to frontend directory..."
cd improved_nfl_system/web_frontend

echo "🔐 Logging into Vercel..."
vercel login

echo ""
echo "🚀 Deploying to Vercel..."
echo "This will deploy both:"
echo "  ✅ Next.js Frontend"
echo "  ✅ Python FastAPI Backend"
echo ""

vercel --prod

echo ""
echo "================================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================="
echo ""
echo "Your app is now live on Vercel!"
echo ""
echo "📊 Check your deployment:"
echo "   Dashboard: https://vercel.com/dashboard"
echo ""
echo "Both frontend and backend are on the same domain!"
echo "No environment variables needed!"
echo ""
