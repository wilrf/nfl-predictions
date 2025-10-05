# 🚀 Deploy to Vercel - Quick Start

## 3-Minute Deployment

### Prerequisites
- GitHub account
- Vercel account (free): https://vercel.com
- Railway account (free): https://railway.app

---

## Option 1: Automated Script (Easiest)

```bash
cd improved_nfl_system
./deploy.sh
```

Follow the prompts, and you're done! 🎉

---

## Option 2: Manual (Step-by-Step)

### Step 1: Install CLIs
```bash
npm install -g vercel @railway/cli
```

### Step 2: Deploy Backend (Railway)
```bash
cd improved_nfl_system/web_app
railway login
railway init
railway up
```

**Copy your Railway URL** (e.g., `https://nfl-api.railway.app`)

### Step 3: Deploy Frontend (Vercel)
```bash
cd ../web_frontend
vercel login
vercel
```

When prompted for environment variable:
- **Key**: `NEXT_PUBLIC_API_URL`
- **Value**: Your Railway URL from Step 2

```bash
vercel --prod
```

**Done!** Your app is live! 🎊

---

## Option 3: Using Dashboards (No CLI)

### Railway Dashboard
1. Go to https://railway.app/new
2. Connect GitHub repo
3. Set root directory: `improved_nfl_system/web_app`
4. Deploy
5. Copy URL

### Vercel Dashboard
1. Go to https://vercel.com/new
2. Import GitHub repo
3. Set root directory: `improved_nfl_system/web_frontend`
4. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = Your Railway URL
5. Deploy

---

## 🔗 Your Live URLs

After deployment:
- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-app.railway.app`

---

## 💰 Cost

**100% FREE** for hobby projects!
- Vercel: Free tier (100GB/month)
- Railway: $5 free credits/month (~100-500 hours)

---

## 🆘 Troubleshooting

**Build fails?**
```bash
cd web_frontend
npm run build  # Test locally first
```

**API not connecting?**
- Check environment variable in Vercel dashboard
- Verify Railway URL is correct
- Check CORS settings in `web_app/app.py`

**Need help?**
- Read: [VERCEL_DEPLOYMENT_GUIDE.md](./VERCEL_DEPLOYMENT_GUIDE.md)
- Vercel docs: https://vercel.com/docs
- Railway docs: https://docs.railway.app

---

## 🔄 Auto-Deploy

Once set up, every `git push` auto-deploys! 🚀

```bash
git add .
git commit -m "Update"
git push
```

Both Vercel and Railway will automatically redeploy.

---

**That's it! Your premium NFL dashboard is live! 🏆**
