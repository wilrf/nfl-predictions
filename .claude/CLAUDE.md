# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NFL Betting Suggestions System** - A professional-grade NFL betting analysis system that generates data-driven suggestions using machine learning models, with strict adherence to responsible gambling principles.

## Development Commands

### Core System
```bash
# Main entry point - run NFL betting analysis
cd improved_nfl_system
python main.py

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
pytest web/tests/

# Web interface
cd web
python launch.py
```

### Testing
```bash
# System tests (includes leakage-free backtesting)
pytest tests/test_system.py -v

# Web integration tests
pytest web/tests/test_integration.py -v

# Playwright end-to-end tests
pytest web/tests/test_playwright.py -v
```

### Code Quality
```bash
# Format code
black .

# Lint code
pylint *.py
```

## Architecture Overview

### Core System (`improved_nfl_system/`)
This is an NFL betting suggestions system with strict "FAIL FAST" philosophy - no fallbacks or synthetic data.

**Main Components:**
- `main.py` - Entry point orchestrator with error handling
- `nfl_betting_system.py` - Core betting logic with professional-grade features (CLV tracking, model versioning, etc.)
- `data_pipeline.py` - Data collection with temporal integrity and freshness monitoring
- `operations_runbook.py` - Weekly operations checklist and health monitoring

**Key Principles:**
- FAIL FAST: Any error stops execution completely
- REAL DATA ONLY: Never uses synthetic/fake data
- SUGGESTIONS ONLY: System suggests, never places bets
- Professional-grade: CLV tracking, model versioning, risk management

### Data Layer
- `database/db_manager.py` - SQLite database operations
- `data/nfl_data_fetcher.py` - NFL stats from nfl_data_py (free, unlimited)
- `data/odds_client.py` - Betting odds from The Odds API (rate limited)

### Enhanced Data Sources (`data_sources/`)
- `nfl_official_client.py` - Official NFL data integration
- `espn_client.py` - ESPN stats and metrics
- `weather_client.py` - Weather data for game conditions

### Models & Calculations
- `models/model_integration.py` - XGBoost model integration
- `calculators/confidence.py` - Confidence scoring (50-90 scale)
- `calculators/margin.py` - Expected margin calculations (0-30 range)
- `calculators/correlation.py` - Bet correlation analysis

### Web Interface (`web/`)
- `app.py` - FastAPI web application
- `bridge/nfl_bridge.py` - Bridge between web and core system
- `launch.py` - Web server launcher with health checks

## Environment Setup

### Required Files
1. `.env` file with `ODDS_API_KEY` (from The Odds API)
2. XGBoost models in `models/saved_models/`:
   - `spread_model.pkl`
   - `total_model.pkl`

### Python Dependencies
Install all dependencies:
```bash
cd improved_nfl_system
pip install -r requirements.txt
```

Key packages:
- `xgboost` - Machine learning models
- `nfl-data-py` - NFL statistics
- `pandas`, `numpy` - Data processing
- `fastapi`, `uvicorn` - Web interface
- `pytest`, `playwright` - Testing

### API Rate Limiting
Free tier allows 500 requests/month. Recommended schedule:
- Tuesday 6-8 AM: Opening lines (1 call)
- Thursday 5-8 PM: Line movement (1 call)
- Saturday 10 PM+: Pre-game update (1 call)
- Sunday 8-11 AM: Closing lines for CLV (1 call)

## Key Features

### Professional-Grade Components
- **CLV Tracking**: Systematic closing line value monitoring
- **Model Versioning**: Complete experiment tracking and governance
- **Leakage-Free Backtesting**: Strict temporal validation with as-of data
- **Risk Management**: Kelly optimization with correlation analysis
- **Data Quality**: Freshness monitoring and health checks

### Operational Workflow
- **Pre-betting Checklist**: 48-hour validation before betting (`operations_runbook.py`)
- **Daily Health Checks**: 2-minute morning system validation
- **Weekly Reports**: Complete CLV analysis and performance review

### Output Tiers
- **Premium Picks**: 80+ confidence (exceptional opportunities)
- **Standard Picks**: 65-79 confidence (solid opportunities)
- **Reference Picks**: 50-64 confidence (marginal opportunities)

## Database Schema

Main tables:
- `games` - NFL game data with stadium/weather info
- `odds` - Betting lines with timestamp tracking
- `suggestions` - Generated betting suggestions with metadata
- `clv_tracking` - Closing line value analysis

## Error Handling

System uses "FAIL FAST" approach:
- Any missing data stops execution
- No fallback mechanisms
- Comprehensive logging to `logs/nfl_system.log`
- Detailed error messages for troubleshooting

## Documentation Files

### Core Documentation
- `README.md` - Main project documentation
- `README_SETUP.md` - Detailed setup instructions
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `WEB_INTERFACE_GUIDE.md` - Web interface usage guide
- `ENHANCED_DATA_SOURCES.md` - Enhanced data source integration guide

### ML & Architecture Guides
- `Theoretical Foundations for an Optimal NFL Spread Betting Model Architecture.pdf` - Model architecture theory
- `NFL Algo_Model ML Dataset Training Best Practices.txt` - ML training best practices

## Test Files
- `test_enhanced_sources.py` - Test enhanced data sources
- `tests/test_system.py` - Core system tests
- `web/tests/test_integration.py` - Web integration tests
- `web/tests/test_playwright.py` - E2E browser tests

## Development Notes

### Important Guidelines
- All models expect specific feature formats - check `_create_features()` in `main.py`
- Correlation warnings use threshold-based system (70%+ high, 40-70% moderate, <40% low)
- Web interface includes real-time system health monitoring
- Test suite includes temporal validation to prevent data leakage
- Kitchen directory (`kitchen/ingredients/`) contains data processing components
- Always verify data freshness before generating suggestions
- Never modify production database without backups
- Test all changes in isolated environment first

### Common Issues & Solutions
- **Missing odds data**: Check API key and rate limits
- **Model loading errors**: Ensure pickle files are in `models/saved_models/`
- **Web interface not starting**: Check port 8000 availability
- **Database locked**: Close other connections to SQLite

### Performance Tips
- Run data collection during off-peak hours
- Cache API responses when possible
- Use batch processing for multiple games
- Monitor log file size in `logs/`