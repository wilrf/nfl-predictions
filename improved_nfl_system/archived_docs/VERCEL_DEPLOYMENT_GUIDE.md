# 🚀 Vercel + Railway Deployment Guide

## Architecture

```
User Browser
     ↓
Vercel (Next.js Frontend)
     ↓ API calls
Railway (FastAPI + ML Models)
```

**Frontend**: Hosted on Vercel (Free tier)
**Backend**: Hosted on Railway (Free $5 credit/month)

---

## 📋 Prerequisites

1. **GitHub Account** - To connect repos
2. **Vercel Account** - Sign up at https://vercel.com (free)
3. **Railway Account** - Sign up at https://railway.app (free)

---

## 🎯 Step-by-Step Deployment

### Part 1: Push to GitHub (If Not Already Done)

```bash
cd /Users/wilfowler/Sports\ Model

# Initialize git if needed
git init
git add .
git commit -m "Premium NFL Dashboard - Ready for deployment"

# Create GitHub repo and push
gh repo create nfl-predictions-dashboard --public --source=. --remote=origin --push
```

---

### Part 2: Deploy Backend to Railway

#### Option A: Using Railway CLI (Recommended)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Navigate to backend
cd improved_nfl_system/web_app

# Initialize Railway project
railway init

# Deploy
railway up

# Get your backend URL
railway status
# Note the URL (e.g., https://your-app.railway.app)
```

#### Option B: Using Railway Dashboard

1. Go to https://railway.app/dashboard
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repo
5. Railway will auto-detect Python
6. Set **Root Directory**: `improved_nfl_system/web_app`
7. Add environment variables (if any)
8. Click **"Deploy"**
9. Wait for deployment (2-3 minutes)
10. Copy your Railway URL (e.g., `https://nfl-api.railway.app`)

---

### Part 3: Deploy Frontend to Vercel

#### Option A: Using Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend
cd improved_nfl_system/web_frontend

# Deploy
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Scope: Your account
# - Link to existing project? No
# - Project name: nfl-predictions-dashboard
# - Directory: ./
# - Override settings? No

# Set environment variable
vercel env add NEXT_PUBLIC_API_URL
# When prompted, enter your Railway URL: https://your-app.railway.app

# Deploy to production
vercel --prod
```

#### Option B: Using Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Click **"Add New Project"**
3. Import your GitHub repo
4. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `improved_nfl_system/web_frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
5. Add Environment Variable:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your Railway URL (e.g., `https://nfl-api.railway.app`)
6. Click **"Deploy"**
7. Wait 2-3 minutes
8. Your app is live! 🎉

---

## 🔧 Configuration Files Explained

### `web_frontend/vercel.json`
```json
{
  "framework": "nextjs",
  "env": {
    "NEXT_PUBLIC_API_URL": "@api_url"
  }
}
```
- Tells Vercel this is a Next.js project
- Sets environment variable for API URL

### `web_app/railway.toml`
```toml
[deploy]
startCommand = "python app.py"
healthcheckPath = "/api/stats"
```
- Tells Railway how to start the app
- Health check ensures API is responding

### `web_app/requirements.txt`
- Lists all Python dependencies
- Railway auto-installs these

---

## 🌐 Your Live URLs

After deployment:

- **Frontend**: `https://nfl-predictions-dashboard.vercel.app`
- **Backend**: `https://nfl-api.railway.app`
- **API Docs**: `https://nfl-api.railway.app/docs`

---

## 🔒 Environment Variables

### Vercel (Frontend)
```
NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app
```

Set via:
```bash
vercel env add NEXT_PUBLIC_API_URL production
```

Or in Vercel Dashboard:
- Go to Project Settings
- Click "Environment Variables"
- Add variable

### Railway (Backend)
```
PORT=8000
PYTHON_VERSION=3.9.6
```

Railway auto-sets these, but you can override in Railway Dashboard.

---

## 🚨 Troubleshooting

### Frontend Deploy Fails
```bash
# Check build locally
cd web_frontend
npm run build

# If successful, redeploy
vercel --prod
```

### Backend Deploy Fails
```bash
# Check Railway logs
railway logs

# Common issues:
# 1. Missing dependencies - check requirements.txt
# 2. Port binding - Railway sets PORT automatically
# 3. Model files too large - use Railway Pro or compress
```

### API Connection Issues

**Check CORS in `web_app/app.py`:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Check Environment Variable:**
```bash
# Verify in Vercel
vercel env ls

# Should show NEXT_PUBLIC_API_URL
```

### Models Not Loading

If XGBoost models are too large:

**Option 1: Use Git LFS**
```bash
git lfs track "*.pkl"
git add .gitattributes
git add models/saved_models/*.pkl
git commit -m "Add models with LFS"
git push
```

**Option 2: Upload to Railway Storage**
```bash
# In Railway dashboard
# Go to your project → Data → Add Volume
# Upload model files there
```

---

## 💰 Cost Breakdown

### Free Tier (Perfect for Testing)

**Vercel:**
- ✅ 100GB bandwidth/month
- ✅ Unlimited deployments
- ✅ Automatic HTTPS
- ✅ Global CDN
- **Cost**: FREE

**Railway:**
- ✅ $5 free credits/month
- ✅ ~100-500 hours runtime (depending on resources)
- ✅ 1GB RAM, 1 vCPU
- **Cost**: FREE (with $5 credit)

**Total**: $0/month for hobby projects! 🎉

### Paid Tier (For Production)

**Vercel Pro**: $20/month
- 1TB bandwidth
- Analytics
- Custom domains
- Team collaboration

**Railway**: ~$5-20/month (usage-based)
- More RAM/CPU
- Better uptime
- Faster deployments

---

## 🎨 Custom Domain (Optional)

### On Vercel:
1. Go to Project Settings → Domains
2. Add your domain (e.g., `nfl-predictions.com`)
3. Update DNS records as shown
4. Automatic HTTPS!

### On Railway:
1. Go to Project Settings → Networking
2. Add custom domain
3. Update DNS records
4. Automatic HTTPS!

---

## 📊 Monitoring

### Vercel Analytics
- Free basic analytics
- Page views, performance
- Real-time visitor data

### Railway Metrics
- CPU/RAM usage
- Request logs
- Deployment history

---

## 🔄 Auto-Deploy on Git Push

Both platforms support automatic deployment:

**Vercel:**
- Push to `main` branch → Auto-deploy to production
- Push to other branch → Preview deployment

**Railway:**
- Push to `main` branch → Auto-deploy
- Can configure deployment branch in settings

```bash
# To deploy
git add .
git commit -m "Update predictions"
git push

# Both platforms deploy automatically! 🚀
```

---

## 🚀 Quick Deploy Commands

```bash
# Full deployment from scratch

# 1. Backend (Railway)
cd improved_nfl_system/web_app
railway login
railway init
railway up

# 2. Frontend (Vercel)
cd ../web_frontend
vercel
vercel env add NEXT_PUBLIC_API_URL
vercel --prod

# Done! 🎉
```

---

## 📝 Next Steps After Deployment

1. ✅ Test all features on live URL
2. ✅ Set up custom domain (optional)
3. ✅ Enable analytics
4. ✅ Monitor performance
5. ✅ Share with users!

---

## 🆘 Need Help?

- **Vercel Docs**: https://vercel.com/docs
- **Railway Docs**: https://docs.railway.app
- **Next.js Docs**: https://nextjs.org/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

**🎉 Your premium NFL predictions dashboard is ready for the world!**
