# 🏈 NFL ML Predictions Web Application

## 🚀 Quick Start

### Launch the Web App

```bash
cd "/Users/wilfowler/Sports Model/improved_nfl_system"
./launch_web_app.sh
```

**Or manually:**

```bash
cd "/Users/wilfowler/Sports Model/improved_nfl_system/web_app"
python3 app.py
```

Then open your browser to: **http://localhost:8000**

---

## ✨ Features

### 📊 Live Dashboard
- **Real-time statistics** showing overall model performance
- **64.1% spread accuracy** on 2025 games
- **High confidence tracking** (100% accuracy on 6 high-confidence picks)

### 📈 Interactive Visualizations
1. **Weekly Performance Chart**
   - Line chart showing accuracy week-by-week
   - Tracks both spread and total predictions
   - Identify trends and improvements

2. **Confidence vs Accuracy Chart**
   - Bar chart analyzing predictions by confidence level
   - Shows higher confidence = higher accuracy
   - Validates model calibration

### 🎮 Game Predictions
- **Week-by-week tabs** to filter games
- **Individual game cards** with:
  - Team matchup and final scores
  - Predicted winner with probability
  - Visual probability bar
  - ✓/✗ showing if prediction was correct
  - Total over/under predictions

### 🎯 Real Data
- All predictions on **actual 2025 NFL games**
- Weeks 1-4 completed games (64 total)
- Real scores and outcomes

---

## 🏗️ Architecture

### Backend (FastAPI)
**File:** `web_app/app.py`

**API Endpoints:**
- `GET /` - Serve main dashboard
- `GET /api/stats` - Overall statistics
- `GET /api/games` - All game predictions
- `GET /api/game/{game_id}` - Specific game details
- `GET /api/weekly_performance` - Week-by-week breakdown
- `GET /api/confidence_analysis` - Confidence bucket analysis

### Frontend (HTML/CSS/JS)
**File:** `web_app/templates/index.html`

**Technologies:**
- Pure JavaScript (no frameworks)
- Chart.js for visualizations
- CSS Grid for responsive layout
- Gradient purple theme
- Glassmorphism design

### ML Models
**Location:** `models/saved_models/`

**Models Used:**
- `spread_model.pkl` - XGBoost classifier (500 trees)
- `spread_calibrator.pkl` - Isotonic regression
- `total_model.pkl` - XGBoost classifier
- `total_calibrator.pkl` - Isotonic regression

---

## 📊 What You'll See

### Stats Cards (Top Section)
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Total Games │ Spread Acc  │ Total Acc   │ High Conf   │
│     64      │   64.1%     │   45.3%     │   100.0%    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Charts
- **Weekly Performance:** Line chart tracking accuracy across weeks 1-4
- **Confidence Analysis:** Bar chart showing accuracy by confidence bucket

### Game Cards
```
┌──────────────────────────────────────────────────┐
│  DAL (20)  │  Predicted: PHI (64.3%)  │  PHI (24) │
│            │  ████████████░░░░        │           │
│            │  Result: ✓ Correct       │           │
└──────────────────────────────────────────────────┘
```

---

## 🎨 Design Features

### Visual Elements
- **Gradient Background:** Purple to violet gradient
- **Glassmorphism:** Frosted glass effect on cards
- **Smooth Animations:** Hover effects and transitions
- **Color Coding:**
  - Green (✓) for correct predictions
  - Red (✗) for incorrect predictions
  - Gold for game scores

### Responsive Design
- Works on desktop, tablet, and mobile
- Grid layout adapts to screen size
- Charts resize automatically

---

## 🔧 How It Works

### Data Flow
1. **Startup:** Web server loads ML models and test data
2. **User visits page:** Browser requests `/`
3. **API calls:** JavaScript fetches data from endpoints
4. **Predictions:** Models run on each game's features
5. **Visualization:** Charts render with Chart.js
6. **Display:** Results shown in beautiful UI

### Prediction Process (per game)
```python
features = prepare_features(game)  # Extract 17 features
spread_pred = models.predict_spread(features)  # XGBoost prediction
# Returns: {home_win_prob, away_win_prob, confidence}
```

---

## 📱 Screenshots

### Main Dashboard
Shows 4 stat cards, 2 interactive charts, and week tabs

### Game Cards
Individual predictions with probability bars and results

---

## 🚀 Next Steps

### Want to enhance?
1. **Add live games:** Fetch upcoming games from API
2. **More charts:** Add team performance breakdown
3. **Export data:** Download predictions as CSV
4. **Dark mode:** Toggle between themes
5. **Filters:** Filter by team, confidence level
6. **Live updates:** WebSocket for real-time updates

---

## 🐛 Troubleshooting

**Port already in use?**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill
```

**Models not loading?**
Check that models exist:
```bash
ls models/saved_models/*.pkl
```

**Browser errors?**
Check browser console (F12) for JavaScript errors

---

## 📖 Technical Details

### Dependencies
- Python 3.9+
- FastAPI 0.117+
- Uvicorn (ASGI server)
- pandas, numpy
- Chart.js (CDN, no install needed)

### Performance
- **Model load time:** ~1 second
- **API response:** <100ms per endpoint
- **Page load:** <2 seconds

### Data Size
- Test dataset: 64 games
- API payload: ~50KB (all games)
- Models: 1.7MB total

---

## 🎉 Enjoy Your ML Dashboard!

The web app provides a beautiful, interactive way to explore your trained machine learning models and see real predictions on real NFL games.

**Have fun exploring the data!** 🏈
