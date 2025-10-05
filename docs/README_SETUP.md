# NFL Betting Suggestion System - Complete Setup Guide

## 📁 Complete File Directory Structure

```
/Users/wilfowler/Sports Model/improved_nfl_system/
│
├── 📄 main.py                          # ← RUN THIS to start the system
├── 📄 requirements.txt                  # Python packages to install
├── 📄 .env.example                      # Environment variables template
├── 📄 .env                              # ← CREATE THIS from .env.example (add your API key)
│
├── 📂 database/
│   ├── 📄 schema.sql                   # Database table definitions
│   ├── 📄 db_manager.py                # Database operations (FAIL FAST)
│   └── 📄 nfl_suggestions.db           # ← CREATED AUTOMATICALLY when you run main.py
│
├── 📂 data/
│   ├── 📄 nfl_data_fetcher.py          # Gets REAL NFL data (nfl_data_py)
│   └── 📄 odds_client.py               # Gets odds from The Odds API
│
├── 📂 calculators/
│   ├── 📄 confidence.py                # Calculates 50-90 confidence score
│   ├── 📄 margin.py                    # Calculates 0-30 margin score
│   └── 📄 correlation.py               # Detects correlated bets (warnings only)
│
├── 📂 models/
│   ├── 📄 model_integration.py         # Integrates your XGBoost models
│   └── 📂 saved_models/                # ← CREATE THIS FOLDER
│       ├── 📄 spread_model.pkl         # ← ADD YOUR spread model here
│       └── 📄 total_model.pkl          # ← ADD YOUR total model here
│
└── 📂 logs/
    └── 📄 nfl_system.log               # ← CREATED AUTOMATICALLY for logging
```

## 🚀 Quick Start Instructions

### Step 1: Install Python Packages
```bash
cd /Users/wilfowler/Sports Model/improved_nfl_system
pip install -r requirements.txt
```

### Step 2: Get Your API Key
1. Go to https://the-odds-api.com/
2. Sign up for FREE account (500 requests/month)
3. Get your API key

### Step 3: Set Up Environment File
```bash
# Copy the template
cp .env.example .env

# Edit .env file and add your API key
# Change this line: ODDS_API_KEY=your_api_key_here
# To this: ODDS_API_KEY=baa3a174dc025d9865dcf65c5e8a4609
```

### Step 4: Add Your XGBoost Models
```bash
# Create models directory
mkdir -p models/saved_models

# Copy your existing models here:
# - spread_model.pkl
# - total_model.pkl
```

### Step 5: Run the System
```bash
python main.py
```

## 📋 What Each File Does

### Core Files:
- **main.py** - The main program that runs everything
- **requirements.txt** - List of Python packages needed
- **.env** - Your configuration (API keys, settings)

### Database Files:
- **database/schema.sql** - Defines all database tables
- **database/db_manager.py** - Handles all database operations
- **database/nfl_suggestions.db** - SQLite database (auto-created)

### Data Collection:
- **data/nfl_data_fetcher.py** - Gets REAL NFL stats (free, unlimited)
- **data/odds_client.py** - Gets betting odds (limited by API tier)

### Calculations:
- **calculators/confidence.py** - Scores bets 50-90 (50=minimum, 90=exceptional)
- **calculators/margin.py** - Scores value 0-30 (0=low, 30=high)
- **calculators/correlation.py** - Warns about correlated bets

### Machine Learning:
- **models/model_integration.py** - Runs your XGBoost models
- **models/saved_models/** - Where your trained models go

## 🎯 How It Works

1. **Fetches Real Data**
   - NFL stats from `nfl_data_py` (free, unlimited)
   - Betting odds from The Odds API (500/month free)

2. **Makes Predictions**
   - Uses your XGBoost models
   - Calculates win probabilities

3. **Calculates Suggestions**
   - Confidence: 50-90 scale
   - Margin: 0-30 scale
   - Only shows bets with 2%+ edge

4. **Shows Warnings**
   - 🔴 High correlation (70%+)
   - 🟡 Moderate correlation (40-70%)
   - 🟢 Low correlation (<40%)

5. **Displays Results**
   - Premium picks (80+ confidence)
   - Standard picks (65-79 confidence)
   - Reference picks (50-64 confidence)

## ⏰ When to Run (Free Tier Schedule)

| Day | Time | Purpose | API Calls |
|-----|------|---------|-----------|
| Tuesday | 6-8 AM | Get opening lines | 1 |
| Thursday | 5-8 PM | Check line movement | 1 |
| Saturday | 10 PM+ | Pre-game update | 1 |
| Sunday | 8-11 AM | Get closing lines (CLV) | 1 |

**Total: 4 API calls per week = 16 per month (well under 500 limit)**

## ❗ Important Rules

1. **FAIL FAST** - Any error stops the system completely (no fallbacks)
2. **REAL DATA ONLY** - Never uses fake/synthetic data
3. **SUGGESTIONS ONLY** - System suggests, YOU decide to bet
4. **CORRELATION WARNINGS** - Shows warnings but doesn't remove bets

## 🔧 Troubleshooting

### Error: "ODDS_API_KEY not set"
→ Edit `.env` file and add your API key

### Error: "Required model not found"
→ Add your XGBoost models to `models/saved_models/`

### Error: "No API credits remaining"
→ You've used your 500 free requests this month

### Error: "No games found"
→ Check if it's NFL season (Sep-Jan)

## 📊 Example Output

```
==============================================================
NFL BETTING SUGGESTIONS - 2024-01-14 09:00
==============================================================

🟢 PREMIUM PICKS (80+ Confidence)
----------------------------------------

Buffalo Bills SPREAD -3.5 @ -110
  Confidence: 82.3 | Margin: 24.5
  Edge: 5.2% | Kelly: 2.1%

🟡 STANDARD PICKS (65-79 Confidence)
----------------------------------------

Chiefs vs Bills TOTAL 47.5 OVER @ -105
  Confidence: 71.2 | Margin: 18.3
  Edge: 3.8% | Kelly: 1.5%
  ⚠️  Same game bets (45% correlation): Bills spread and total

==============================================================
TOTAL SUGGESTIONS: 2
Average Confidence: 76.8
Average Margin: 21.4

⚠️  CORRELATION WARNINGS: 1
Review correlated bets carefully before wagering
==============================================================
```

## ✅ System Ready Checklist

- [ ] Python packages installed (`pip install -r requirements.txt`)
- [ ] API key from The Odds API
- [ ] `.env` file created with API key
- [ ] XGBoost models in `models/saved_models/`
- [ ] Run `python main.py` successfully

Once all items are checked, your system is ready to use!