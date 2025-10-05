#!/bin/bash

# ===================================
# NFL Edge Analytics - Deployment Script
# Deploy to Vercel via GitHub
# ===================================

set -e

echo "🚀 NFL Edge Analytics - Deployment Pipeline"
echo "==========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed"
    exit 1
fi

if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI is not installed"
    echo "Install with: npm i -g vercel"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites satisfied${NC}"
echo ""

# 2. Create production build
echo -e "${BLUE}🏗️  Preparing production build...${NC}"

# Create requirements.txt for Vercel
cat > improved_nfl_system/requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
numpy==1.24.3
xgboost==2.0.0
psycopg2-binary==2.9.7
python-dotenv==1.0.0
pydantic==2.4.2
nfl-data-py==0.3.1
EOF

echo -e "${GREEN}✅ Requirements file created${NC}"

# 3. Git operations
echo ""
echo -e "${BLUE}📦 Preparing Git commit...${NC}"

# Check for changes
if [[ -n $(git status -s) ]]; then
    echo "Found changes to commit:"
    git status -s

    # Add all changes
    git add .

    # Commit with message
    read -p "Enter commit message: " commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="Deploy: Update NFL Edge Analytics dashboard"
    fi

    git commit -m "$commit_msg"
    echo -e "${GREEN}✅ Changes committed${NC}"
else
    echo "No changes to commit"
fi

# 4. Push to GitHub
echo ""
echo -e "${BLUE}🔄 Pushing to GitHub...${NC}"
git push origin main 2>/dev/null || git push origin test
echo -e "${GREEN}✅ Pushed to GitHub${NC}"

# 5. Deploy to Vercel
echo ""
echo -e "${BLUE}🚀 Deploying to Vercel...${NC}"

# Check if project is linked
if [ ! -f ".vercel/project.json" ]; then
    echo -e "${YELLOW}⚠️  Project not linked to Vercel${NC}"
    echo "Initializing Vercel project..."
    vercel link
fi

# Deploy based on branch
current_branch=$(git branch --show-current)

if [ "$current_branch" = "main" ] || [ "$current_branch" = "master" ]; then
    echo "Deploying to production..."
    vercel --prod
else
    echo "Deploying preview for branch: $current_branch"
    vercel
fi

echo ""
echo -e "${GREEN}✨ Deployment Complete!${NC}"
echo ""
echo "Your NFL Edge Analytics dashboard is now live!"
echo ""
echo "Next steps:"
echo "1. Check your deployment at: https://nfl-edge-analytics.vercel.app"
echo "2. Set environment variables in Vercel dashboard:"
echo "   - SUPABASE_URL"
echo "   - SUPABASE_KEY"
echo "   - ODDS_API_KEY"
echo "3. Monitor logs: vercel logs"
echo ""