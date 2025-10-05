# NFL Sports Model

Professional-grade NFL betting analysis system with machine learning predictions and comprehensive data pipelines.

## ✅ System Status
- **Database**: Connected via MCP to Supabase
- **Data**: 1,343 games loaded (2016-2024)
- **Models**: XGBoost ready for predictions
- **MCP Tools**: Fully configured and working

## Quick Start

```bash
cd improved_nfl_system
python3 main.py
```

## Project Structure

```
Sports Model/
├── improved_nfl_system/        # Main system code
│   ├── main.py                 # Entry point
│   ├── nfl_betting_system.py   # Core betting logic
│   ├── data_pipeline.py        # Data collection
│   ├── models/                 # ML models
│   ├── web/                    # Web interface
│   └── database/               # SQLite & schemas
├── docs/                       # Documentation
├── context/                    # Historical context
└── CLAUDE.md                   # AI assistant guide
```

## Key Features

- **ML Predictions**: XGBoost models for spread and totals
- **Data Pipeline**: NFL stats, odds, weather integration
- **Web Interface**: FastAPI dashboard
- **MCP Integration**: Direct database access via Supabase
- **Risk Management**: Kelly optimization, CLV tracking

## Database Tables

| Table | Records | Purpose |
|-------|---------|---------|
| games | 1,343 | Game data with scores |
| teams | 32 | NFL teams |
| historical_games | 523 | Historical reference |
| game_features | 0 | Ready for features |
| epa_metrics | 0 | EPA statistics |
| betting_outcomes | 0 | Betting results |

## MCP Configuration

MCP servers are configured via Claude CLI:
```bash
claude mcp list  # Shows connected servers
```

- ✅ Supabase: Database operations
- ✅ GitHub: Repository management

## Documentation

- [Setup Guide](improved_nfl_system/README_SETUP.md)
- [Quick Start](improved_nfl_system/QUICK_START.md)
- [Web Interface](improved_nfl_system/WEB_INTERFACE_GUIDE.md)
- [Implementation](improved_nfl_system/IMPLEMENTATION_SUMMARY.md)
- [Data Sources](improved_nfl_system/ENHANCED_DATA_SOURCES.md)

## Requirements

- Python 3.8+
- PostgreSQL (via Supabase)
- API Keys: The Odds API
- XGBoost models in `models/saved_models/`

## Recent Cleanup

Project cleaned on 2024-10-04:
- Removed 100+ migration artifacts
- Archived old documentation
- Consolidated duplicate files
- Freed ~12-15 MB space

## Getting Help

See [CLAUDE.md](CLAUDE.md) for AI assistant capabilities and commands.

---

*NFL Betting Suggestions System - Suggestions Only, Never Places Bets*
