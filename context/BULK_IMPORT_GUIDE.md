# Bulk Historical Data Import Guide

## Overview

The `bulk_import_historical_data.py` script imports massive NFL historical data for ML training. It fetches games, calculates EPA metrics, and generates ML-ready features.

## What It Does

1. **Fetches game schedules** from nfl_data_py (2015-2024 available)
2. **Calculates EPA metrics** from 1.3M+ play-by-play records
3. **Generates game features** for ML training (15+ predictive features)
4. **Saves to CSV** for inspection and backup
5. **Generates SQL** for Supabase loading via MCP

## Quick Start

### Test on Single Season (Recommended First)
```bash
python3 bulk_import_historical_data.py --season 2024
```

**Output:**
- `ml_training_data/season_2024/games.csv` - 272 games
- `ml_training_data/season_2024/team_epa_stats.csv` - 704 team-week records
- `ml_training_data/season_2024/game_features.csv` - 272 feature vectors
- `ml_training_data/season_2024_import.sql` - SQL for Supabase

### Import Full Dataset (2015-2024)
```bash
python3 bulk_import_historical_data.py --start 2015 --end 2024
```

**Expected:**
- ~2,700 games (270 per season × 10 seasons)
- ~1.3M plays
- ~15,000 team-week EPA records
- Duration: 40-60 minutes

### Import Custom Range
```bash
# Last 5 seasons only
python3 bulk_import_historical_data.py --start 2020 --end 2024

# Maximum data (1999-2024)
python3 bulk_import_historical_data.py --start 1999 --end 2024
```

## Resume Capability

The script tracks progress in `bulk_import_progress.json`. If interrupted:
- Re-run the same command
- Already-imported seasons are skipped automatically
- Picks up where it left off

## Output Structure

```
ml_training_data/
├── season_2015/
│   ├── games.csv              # Game schedules and results
│   ├── team_epa_stats.csv     # Team EPA by week
│   ├── game_features.csv      # ML training features
│   └── season_2015_import.sql # SQL for loading
├── season_2016/
│   └── ...
├── ...
├── import_stats.json          # Overall statistics
└── bulk_import_progress.json  # Resume tracking
```

## Feature Set

Each game has 29 features calculated:

### Target Variables (Outcomes)
- `home_score`, `away_score`
- `point_differential` (home - away)
- `total_points` (home + away)
- `home_won` (binary)

### Core Predictors (Tier 1)
- `epa_differential` - **Strongest predictor** (~0.22 correlation)
- `home_off_epa`, `home_def_epa`, `away_off_epa`, `away_def_epa`
- `home_off_success_rate`, `away_off_success_rate`
- `is_home` - Home field advantage
- `is_divisional` - Familiarity factor

### Scoring Efficiency (Tier 2)
- `home_redzone_td_pct`, `away_redzone_td_pct`
- `home_third_down_pct`, `away_third_down_pct`

### Context
- `week_number` - Early vs late season
- `home_games_played`, `away_games_played` - Sample size
- `stadium`, `is_outdoor` - Venue factors

## Data Quality

### Temporal Validation
- **Week 1 games:** All EPA features = 0.0 (no prior data - correct!)
- **Week 2+:** Real EPA values from previous weeks
- **No data leakage:** Features only use data available *before* the game

### Example (Week 2, 2024):
```csv
game_id,home_off_epa,away_off_epa,epa_differential
2024_02_BUF_MIA,-0.0063,0.1914,-0.0147
```
✅ Uses only Week 1 data to predict Week 2

## Loading to Supabase

### Via MCP (Recommended)
```bash
# Use MCP commands to execute SQL
mcp__supabase__execute_sql < ml_training_data/season_2024_import.sql
```

### Via Python (Alternative)
```python
from supabase import create_client
import pandas as pd

# Load CSV
games = pd.read_csv('ml_training_data/season_2024/game_features.csv')

# Upload via Supabase client
supabase.table('ml_training_games').insert(games.to_dict('records')).execute()
```

## Troubleshooting

### "No play-by-play data for season X"
- **Cause:** Season hasn't been played yet
- **Solution:** Only import completed seasons

### "Missing team EPA for Week 1"
- **Expected behavior** - Week 1 has no prior data
- Features will be 0.0 or use previous season

### Script crashes mid-import
- **Solution:** Just re-run the same command
- Progress is saved after each season
- Already-imported seasons are skipped

### Want more historical data?
```bash
# Maximum available (1999-2024)
python3 bulk_import_historical_data.py --start 1999 --end 2024
```

## Next Steps After Import

### 1. Validate Data Quality
```bash
# Check row counts
wc -l ml_training_data/*/game_features.csv

# Inspect first few rows
head ml_training_data/season_2024/game_features.csv
```

### 2. Train ML Models
```python
import pandas as pd
from xgboost import XGBClassifier

# Load all seasons
features = []
for season in range(2015, 2025):
    df = pd.read_csv(f'ml_training_data/season_{season}/game_features.csv')
    features.append(df)

all_data = pd.concat(features, ignore_index=True)

# Remove incomplete games
all_data = all_data[all_data['home_score'].notna()]

# Split features and targets
X = all_data[[
    'epa_differential', 'home_off_epa', 'home_def_epa',
    'away_off_epa', 'away_def_epa', 'is_home', 'week_number',
    'home_redzone_td_pct', 'away_redzone_td_pct',
    'home_third_down_pct', 'away_third_down_pct'
]]
y = all_data['home_won']

# Train model
model = XGBClassifier(n_estimators=100, max_depth=3)
model.fit(X, y)
```

### 3. Feature Validation
See `FEATURE_VALIDATION_STRATEGY.md` for systematic validation process.

## Performance Expectations

### Import Speed
- **Single season:** ~10-15 seconds
- **10 seasons (2015-2024):** ~40-60 minutes
- **26 seasons (1999-2024):** ~2-3 hours

### Dataset Sizes
- **2015-2024:** 2,700 games, ~120MB
- **2010-2024:** 4,000 games, ~180MB
- **1999-2024:** 6,942 games, ~300MB

### ML Training Expectations
With 2,700 games (2015-2024):
- **Baseline accuracy:** 50-51% (better than coin flip)
- **With tuning:** 52-53% (profitable)
- **With ensemble:** 53-54% (sharp-level)

## Logs

All activity logged to:
- Console (real-time)
- `logs/bulk_import.log` (persistent)

Check logs for detailed progress and any errors.

## Strategy: Incremental Approach

### Recommended: Start Small, Expand as Needed

**Week 1: Modern Era (2015-2024)**
- Import 2,700 games
- Train initial models
- If accuracy ≥52%: ✅ Done!
- If accuracy 50-52%: Proceed to Week 2

**Week 2: Extended Era (2010-2024)**
- Add 1,300 more games
- Retrain models
- If accuracy ≥52%: ✅ Done!
- If accuracy <52%: Proceed to Week 3

**Week 3: Maximum Data (1999-2024)**
- Add final 2,942 games (total 6,942)
- This is the limit
- If still <52%: Need better features, not more data

## Academic Validation

This approach aligns with sharp betting best practices:
- **2,700 games** = 207 samples per feature (well above 100 minimum)
- **Modern era** = Consistent rules and parity
- **Temporal validation** = No data leakage
- **Feature quality** > Data quantity

Sources:
- "The Logic of Sports Betting" (Kovalchik, 2016)
- "Assessing Market Efficiency" (Woodland & Woodland, 2003)
- nfl_data_py documentation

## Support

Questions or issues? Check:
1. `logs/bulk_import.log` for detailed errors
2. `ML_SUCCESS_ROADMAP.md` for strategy context
3. `FEATURE_VALIDATION_STRATEGY.md` for feature selection
4. `MASSIVE_DATA_IMPORT_STRATEGY.md` for import planning
