# Deployment Guide

## Changes Made

### 1. Fixed `web/vercel.json`
- Updated to point to `../api/*.py` instead of `../web_app/app.py`
- Added explicit routes for all API endpoints
- Removed monorepo complexity

### 2. Simplified `web/next.config.ts`
- Removed localhost rewrites
- Vercel now handles API routing automatically

### 3. Removed Turbopack
- Changed build script from `next build --turbopack` to `next build`
- More stable builds on Vercel

### 4. Deleted root `vercel.json`
- Avoids conflicts with `web/vercel.json`

## How to Deploy

### Option 1: Push to GitHub (Auto-deploy)
```bash
git add -A
git commit -m "Fix deployment configuration for Vercel"
git push origin main
```
Vercel will auto-deploy from the `web/` directory.

### Option 2: Deploy via Vercel CLI
```bash
cd web
vercel --prod
```

## What This Fixes

- ✅ Module resolution errors
- ✅ Turbopack build failures
- ✅ API routing issues
- ✅ Path alias conflicts

## Expected Result

- Build succeeds in ~1-2 minutes
- Dashboard loads without 404 errors
- API endpoints return correct data
- All imports resolve correctly

