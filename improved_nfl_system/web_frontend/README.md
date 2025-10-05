# Premium NFL ML Predictions Dashboard

A beautiful, sleek, black & white modern web interface for NFL betting predictions powered by XGBoost machine learning.

## 🎨 Design Features

- **Premium Black & White Theme** - Minimalist, high-end design
- **Glass Morphism Effects** - Modern, transparent card designs
- **Framer Motion Animations** - Smooth, buttery 60fps transitions
- **Advanced Visualizations** - Recharts with custom styling
- **Fully Responsive** - Mobile-first design
- **Performance Optimized** - Next.js 14 with App Router

## 🚀 Quick Start

### Option 1: Use the Launch Script (Recommended)

```bash
cd improved_nfl_system
./launch_premium_web.sh
```

This starts both the FastAPI backend and Next.js frontend automatically.

### Option 2: Manual Launch

**Terminal 1 - Backend:**
```bash
cd improved_nfl_system/web_app
python3 app.py
```

**Terminal 2 - Frontend:**
```bash
cd improved_nfl_system/web_frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## 📦 Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Advanced animations
- **Recharts** - Data visualizations
- **Lucide React** - Beautiful icons

### Backend
- **FastAPI** - Python API framework
- **XGBoost** - ML predictions
- **nfl_data_py** - NFL statistics

## 🎯 Key Features

### Dashboard Components

1. **Hero Header** - Large typography, live status, smooth animations
2. **Stat Cards** - Glass morphism, hover effects, trend indicators
3. **Interactive Charts** - Weekly performance & confidence analysis
4. **Game Cards** - Animated probability bars, team matchups
5. **Week Tabs** - Interactive filtering with smooth transitions

## 🎨 Design System

### Color Palette
```
Black:        #000000  (background)
Charcoal:     #1a1a1a  (surfaces)
Dark Gray:    #2d2d2d  (elevated surfaces)
Medium Gray:  #6b6b6b  (muted text)
Light Gray:   #e5e5e5  (accents)
White:        #ffffff  (primary text)
```

### Typography
- **Headings**: Geist Sans (bold, tracking-tight)
- **Body**: Geist Sans (regular)
- **Monospace**: Geist Mono

## 📱 Responsive Design

- Mobile: 1 column layout
- Tablet: 2 column layout
- Desktop: 4 column layout
- Ultra-wide: Max 7xl container

## 🔧 Development

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

### Build for Production
```bash
npm run build
npm start
```

### Environment Variables
Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📊 API Integration

Connects to FastAPI backend at `http://localhost:8000`:

- `GET /api/stats` - Overall statistics
- `GET /api/games` - All game predictions
- `GET /api/weekly_performance` - Week-by-week breakdown
- `GET /api/confidence_analysis` - Confidence bucket analysis

## 🏆 Performance

- Lighthouse Score: 95+
- First Contentful Paint: <1s
- Time to Interactive: <2s
- 60fps animations throughout
