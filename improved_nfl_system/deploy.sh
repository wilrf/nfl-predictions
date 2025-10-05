#!/bin/bash
# Auto-deploy script for NFL Predictions Dashboard
# This script pushes to GitHub main branch, triggering Vercel auto-deployment

echo "🚀 NFL Predictions Dashboard - Auto Deploy to Main"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure we're in the project root
if [ ! -d "web_frontend" ] || [ ! -d "web_app" ]; then
    echo -e "${RED}❌ Error: Not in project root directory${NC}"
    echo "Please run from improved_nfl_system/"
    exit 1
fi

# Get current branch
BRANCH=$(git branch --show-current)
echo -e "${YELLOW}📍 Current branch:${NC} $BRANCH"

# If not on main, offer to switch
if [ "$BRANCH" != "main" ]; then
    echo -e "${YELLOW}⚠️  Not on main branch${NC}"
    read -p "Switch to main branch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout main
        git pull origin main
        BRANCH="main"
    else
        echo -e "${YELLOW}Warning: Auto-deploy only works from main branch${NC}"
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}📝 Uncommitted changes detected:${NC}"
    git status --short
    echo ""

    read -p "Commit these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter commit message: " COMMIT_MSG
        git add .
        git commit -m "$COMMIT_MSG"
        echo -e "${GREEN}✅ Changes committed${NC}"
    else
        echo -e "${RED}❌ Cannot deploy with uncommitted changes${NC}"
        exit 1
    fi
fi

# Push to GitHub main branch (triggers Vercel auto-deploy)
echo -e "${YELLOW}📤 Pushing to GitHub main branch...${NC}"
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Successfully pushed to main branch!${NC}"
    echo ""
    echo "=================================================="
    echo -e "${GREEN}🎉 AUTO-DEPLOYMENT TRIGGERED!${NC}"
    echo "=================================================="
    echo ""
    echo "📊 Monitor deployment:"
    echo "   • Dashboard: https://vercel.com/wilrfs-projects/web_frontend"
    echo "   • Live URL: https://webfrontend-oddx9ba95-wilrfs-projects.vercel.app"
    echo ""
    echo "⏱️  Deployment usually takes 1-2 minutes"
    echo ""

    # Optional: Check deployment status
    read -p "Check deployment logs? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}📜 Fetching deployment logs...${NC}"
        cd web_frontend
        vercel logs --follow --scope=wilrfs-projects
    fi
else
    echo -e "${RED}❌ Failed to push to GitHub${NC}"
    echo "Check your internet connection and GitHub credentials"
    exit 1
fi

echo ""
echo -e "${GREEN}✨ Done! Your changes are being deployed automatically.${NC}"
