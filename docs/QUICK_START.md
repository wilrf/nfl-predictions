# 🚀 Quick Start - Premium NFL Dashboard

## One-Command Launch

```bash
cd improved_nfl_system
./launch_premium_web.sh
```

Then open: **http://localhost:3000**

---

## Manual Launch

### Terminal 1 - Backend
```bash
cd improved_nfl_system/web_app
python3 app.py
```

### Terminal 2 - Frontend
```bash
cd improved_nfl_system/web_frontend
npm run dev
```

---

## URLs

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## What You'll See

✨ **Beautiful black & white design**
📊 **4 stat cards** with animations
📈 **2 interactive charts**
🏈 **Game predictions** with win probabilities
🎯 **Weekly filtering** tabs

---

## First Time Setup

If this is your first time:

```bash
# Install frontend dependencies
cd improved_nfl_system/web_frontend
npm install

# Make launch script executable
cd ..
chmod +x launch_premium_web.sh
```

---

## Troubleshooting

**Frontend won't start?**
```bash
cd web_frontend
npm install
npm run dev
```

**Backend error?**
```bash
cd web_app
pip3 install -r requirements.txt
python3 app.py
```

**Port already in use?**
- Frontend (3000): Kill process or change port in package.json
- Backend (8000): Kill process or change port in web_app/app.py

---

## 🎨 Features

- Premium black & white design
- Glass morphism effects
- Framer Motion animations (60fps)
- Recharts visualizations
- Fully responsive
- TypeScript + Next.js 14

---

**Built with Next.js, TypeScript, Tailwind CSS, and Framer Motion**
