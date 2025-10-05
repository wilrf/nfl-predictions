# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NFL Betting Suggestions System** - Professional-grade NFL betting analysis system that generates data-driven suggestions using machine learning models. The system provides suggestions only and never places bets.

### Core Principle: FAIL FAST
When coding, if something fails, don't try to get around it - report the error and stop immediately. No fallbacks, no synthetic data, no workarounds.

## Development Commands

### Running the System
```bash
# Main entry point - run NFL betting analysis
python src/main.py

# Install dependencies
pip install -r requirements.txt

# Web interface
python web_app/launch.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_system.py -v

# Run single test
pytest tests/test_system.py::test_specific_function -v

# Web integration tests
pytest web_app/tests/test_integration.py -v

# Playwright end-to-end tests
pytest web_app/tests/test_playwright.py -v
```

### Model Training
```bash
# Train XGBoost models
python src/train_models.py

# Train ensemble models
python src/train_ensemble.py

# Train Random Forest
python src/train_random_forest.py

# Compare model performance
python src/compare_models.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
pylint src/ database/
```

## Architecture

### Directory Structure
```
Sports Model/
├── src/                    # Main system code
│   ├── main.py            # Entry point orchestrator
│   ├── operations_runbook.py  # Weekly operations checklist
│   ├── calculators/       # Confidence, margin, correlation
│   ├── data/             # NFL data fetcher, odds client
│   └── data_sources/     # Enhanced data sources (ESPN, weather)
├── database/              # Database layer
│   ├── db_manager.py     # SQLite operations
│   ├── schema.sql        # Database schema
│   └── nfl_suggestions.db  # SQLite database
├── saved_models/          # Machine learning models
│   ├── model_integration.py  # Model loading & prediction
│   └── saved_models/     # Pickled XGBoost models
├── web_app/              # FastAPI web interface
│   ├── app.py           # Web application
│   ├── launch.py        # Server launcher
│   ├── bridge/          # Core system integration
│   └── tests/           # Web-specific tests
├── tests/                # Core system tests
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── logs/                 # System logs
```

### Key Principles

**FAIL FAST Philosophy:**
- Any error stops execution completely
- No fallback mechanisms
- No synthetic/fake data - real data only
- Comprehensive logging to `logs/nfl_system.log`
- Detailed error messages for troubleshooting

**Data Sources:**
- NFL stats: `nfl_data_py` (free, unlimited)
- Betting odds: The Odds API (500 requests/month free tier)
- Database: SQLite (local) + PostgreSQL via Supabase MCP (remote)

**Professional Features:**
- CLV (Closing Line Value) tracking
- Model versioning and experiment tracking
- Leakage-free backtesting with temporal validation
- Kelly criterion optimization
- Correlation analysis between bets

### Database

**SQLite (Local):**
- Location: `database/nfl_suggestions.db`
- Tables: games, odds_snapshots, suggestions, clv_tracking, correlation_warnings

**Supabase (Remote via MCP):**
- PostgreSQL database
- 1,343 games loaded (2016-2024)
- Accessed via MCP tools in Claude Code

### Models

**Location:** `saved_models/saved_models/`

**Models:**
- `spread_model.pkl` - XGBoost for spread predictions
- `total_model.pkl` - XGBoost for totals predictions
- `random_forest_spread_model.pkl` - Random Forest for spreads
- `*_calibrator.pkl` - Calibration models
- `*_metrics.json` - Model performance metrics
- `ensemble_*_config.json` - Ensemble configurations

**Feature Engineering:**
- Team-specific stats (EPA, success rate, yards per play)
- Situational efficiency (red zone, third down)
- Game context (outdoor, time of day)
- Weather and stadium factors
- Recent form (uses temporal decay)

### API Rate Limiting

The Odds API free tier: 500 requests/month

**Recommended Schedule:**
- Tuesday 6-8 AM: Opening lines (1 call)
- Thursday 5-8 PM: Line movement (1 call)
- Saturday 10 PM+: Pre-game update (1 call)
- Sunday 8-11 AM: Closing lines for CLV (1 call)

Total: ~4 calls per week = 16 per month

## Key Workflows

### Weekly Betting Suggestions

1. **Tuesday Morning:** Fetch opening lines via Odds API
2. **Wednesday-Friday:** Monitor injuries, weather updates
3. **Saturday:** Generate betting suggestions
4. **Sunday Pre-Game:** Capture closing lines
5. **Monday:** Calculate CLV, update performance metrics

### Pre-Betting Checklist (48 hours before games)

From `src/operations_runbook.py`:
1. Verify injury data freshness (< 48 hours old)
2. Confirm opening lines captured
3. Check model calibration drift
4. Validate data completeness
5. Run leakage-free backtest

### Suggestion Tiers

- **Premium Picks** (80-90 confidence): Exceptional opportunities
- **Standard Picks** (65-79 confidence): Solid opportunities
- **Reference Picks** (50-64 confidence): Marginal opportunities

## Common Operations

### Week 1 Predictions
Week 1 uses prior season stats (with warnings) due to no current season data. Expect multiple warnings about reduced accuracy.

### CLV Tracking
- Opening lines must be captured early in week
- Closing lines captured Sunday morning
- CLV calculated post-game
- Health check: `system.check_clv_health(season, week)`

### Model Validation
- Features validated before predictions
- Temporal validation ensures no data leakage
- Correlation warnings applied to confidence scores

### Error Handling
All errors follow FAIL FAST:
```python
raise SystemError("Clear error message")  # System-level
raise DatabaseError("Database issue")     # Database
raise ModelError("Model problem")         # Model
raise NFLDataError("Data fetch failed")   # NFL data
raise OddsAPIError("API failed")          # Odds API
```

## Environment Setup

### Required Files
1. `.env` file with:
   - `ODDS_API_KEY` - From The Odds API
   - `SUPABASE_URL` - Supabase project URL
   - `SUPABASE_KEY` - Supabase API key

2. Models in `saved_models/saved_models/`:
   - XGBoost spread and total models
   - Calibrator models
   - Metrics files

### Python Version
- Requires Python 3.8+
- Using `nfl_data_py` (deprecated, requires Python 3.10+ for `nflreadpy` migration)

### Key Dependencies
- `xgboost>=2.0.0` - Machine learning models
- `nfl-data-py>=0.3.1` - NFL statistics
- `pandas>=2.1.0`, `numpy>=1.24.0` - Data processing
- `fastapi>=0.104.0`, `uvicorn>=0.24.0` - Web interface
- `pytest>=7.4.0` - Testing
- `scikit-learn>=1.3.0` - Model calibration

## Important Guidelines

### Feature Engineering
- All models expect specific 22-feature format
- Check `_create_features()` in `src/main.py:259-303` for feature list
- Team-specific features are used (home/away EPA, success rates, etc.)
- Features validated before predictions via `validate_features()`

### Correlation Warnings
Threshold-based system:
- **High** (≥70%): 30% confidence penalty
- **Moderate** (40-70%): 15% confidence penalty
- **Low** (<40%): 5% confidence penalty

Common correlations:
- Same game favorite + over: 73%
- Same game underdog + under: 68%
- Same team spread + moneyline: 85%

### Database Operations
- Validation happens before DB inserts (not in DB layer)
- All writes use transactions
- Foreign keys enforced
- Indexes on season/week, confidence, timestamps

### Web Interface
- Runs on port 8000 by default
- FastAPI with Jinja2 templates
- Real-time health monitoring
- Bridge connects to core system

## Common Issues & Solutions

**Missing odds data:**
- Check `ODDS_API_KEY` in `.env`
- Verify API rate limits (500/month)
- Check `api_usage` table for remaining credits

**Model loading errors:**
- Ensure `.pkl` files exist in `saved_models/saved_models/`
- Check Python version compatibility
- Verify model metrics files present

**Web interface not starting:**
- Check port 8000 availability: `lsof -i :8000`
- Verify FastAPI/uvicorn installed
- Check logs in `logs/`

**Database locked:**
- Close other SQLite connections
- Check for zombie processes: `ps aux | grep python`
- Restart if necessary

**Week 1 predictions:**
- Warnings expected (uses prior season data)
- Accuracy significantly reduced
- Consider skipping or reducing bet sizes

**CLV tracking failures:**
- Ensure opening lines captured Tuesday
- Confirm closing lines captured Sunday morning
- Run post-game CLV update if needed
- Check `clv_tracking` table coverage

## Testing Strategy

### Test Categories
1. **Unit Tests** - Individual components
2. **Integration Tests** - System workflows
3. **Validation Tests** - Temporal validation, no data leakage
4. **Web Tests** - FastAPI endpoints
5. **E2E Tests** - Playwright browser automation

### Temporal Validation
Critical for preventing data leakage:
- Uses "as-of" dates for historical predictions
- Week N predictions use only data through Week N-1
- Strictly enforced in backtesting

## Documentation

**Core Docs:**
- `README.md` - Project overview
- `TECHNICAL_IMPLEMENTATION_PLAN.md` - Complete architecture
- `docs/README_SETUP.md` - Detailed setup guide
- `docs/WEB_INTERFACE_GUIDE.md` - Web interface usage

**Reference:**
- `docs/IMPLEMENTATION_SUMMARY.md` - Technical details
- `docs/ENHANCED_DATA_SOURCES.md` - Data source integration
- `docs/NFL Algo_Model ML Dataset Training Best Practices.txt` - ML best practices

## Supabase Integration

**MCP Tools Available:**
- `mcp__supabase__list_tables` - List database tables
- `mcp__supabase__execute_sql` - Run SQL queries
- `mcp__supabase__apply_migration` - Apply migrations
- `mcp__supabase__get_logs` - Fetch service logs
- `mcp__supabase__get_advisors` - Security/performance checks

**Data in Supabase:**
- 1,343 games (2016-2024)
- Comprehensive schemas in `database/comprehensive_schema.sql`
- Team stats, EPA metrics, betting outcomes

## Housecleaning Protocol

When the user requests "commence with housecleaning" or "housecleaning scan", perform a comprehensive audit of the entire repository to ensure cleanliness and organization:

### Housecleaning Checklist

1. **Scan for Misplaced Code Files**
   - Python files (`.py`) should be in: `src/`, `scripts/`, `tests/`, `database/`, `validation/`, `api/`
   - TypeScript/JavaScript files (`.ts`, `.tsx`, `.js`) should be in: `web/`, `api/`, `scripts/`
   - Report any code files outside expected directories

2. **Scan for Misplaced Database Files**
   - SQL files (`.sql`) should be in: `database/`, `ml_training_data/` (for imports only)
   - Database files (`.db`, `.sqlite`) should be in: `database/` or `database/backups/`
   - Report any database files in unexpected locations

3. **Check for Duplicate Files**
   - Use MD5 hashing to find duplicate `.py` files across directories
   - Use MD5 hashing to find duplicate `.sql` files (excluding training data imports)
   - Report duplicates with file paths and recommend which to keep/delete

4. **Verify Large Files**
   - Find files >1MB and verify they belong
   - Check directory sizes to identify unexpected bloat
   - Expected large directories:
     - `web/web_frontend/node_modules/` (~506 MB)
     - `saved_models/` (~9 MB)
     - `ml_training_data/` (~5 MB)
     - `database/` (~4 MB including backups)

5. **Check File Organization**
   - Verify documentation (`.md`) files are in `docs/` or root for important files
   - Verify test files are in `tests/` or `web/tests/`
   - Check for temp files, `.DS_Store`, cache files that should be removed

### Housecleaning Commands

```bash
# Find misplaced Python files
find . -name "*.py" -not -path "*/node_modules/*" -not -path "*/.next/*" -not -path "*/archive/*" -not -path "*/__pycache__/*" -not -path "*/.git/*"

# Find misplaced SQL/database files
find . -type f \( -name "*.sql" -o -name "*.db" -o -name "*.sqlite" \) -not -path "*/node_modules/*" -not -path "*/archive/*" -not -path "*/.git/*"

# Find duplicate Python files (by MD5)
find . -name "*.py" -not -path "*/node_modules/*" -not -path "*/archive/*" -not -path "*/__pycache__/*" -exec md5 {} \; | sort | uniq -d

# Find large files >1MB
find . -type f -not -path "*/node_modules/*" -not -path "*/.next/*" -not -path "*/archive/*" -not -path "*/.git/*" -size +1M -ls

# Check directory sizes
du -h -d 1 . | sort -rn | head -20
```

### Housecleaning Report Format

After scanning, provide:
1. ✅ **No Issues** sections (what's clean)
2. ⚠️ **Issues Found** sections (what needs fixing)
3. **Recommended Actions** with specific commands to execute
4. **Wait for user approval** before executing any moves/deletes

### Important Rules

- **NEVER delete files without user approval**
- **MOVE unique data, DELETE only confirmed duplicates**
- **Verify duplicates with MD5 before recommending deletion**
- **Follow FAIL FAST**: If unsure about a file's purpose, ask the user

## Performance Tips

- Run data collection during off-peak hours
- Cache API responses when possible
- Use batch processing for multiple games
- Monitor log file sizes in `logs/`
- Check API credits before large operations
- Run health checks before generating suggestions
