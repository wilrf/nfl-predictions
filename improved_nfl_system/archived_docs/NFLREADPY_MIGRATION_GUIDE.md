# nflreadpy Migration Guide

**Date Created**: October 2, 2025
**Current Status**: Using deprecated `nfl_data_py` v0.3.2
**Target**: Migrate to `nflreadpy` when Python upgraded to 3.10+

---

## Background

`nfl_data_py` has been officially deprecated in favor of `nflreadpy`. All future development will occur in `nflreadpy`.

**Deprecation Notice**: https://github.com/cooperdff/nfl_data_py

**New Library**: https://github.com/nflverse/nflreadpy

---

## Current Blocker

**nflreadpy requires Python 3.10+**

**Current system**:
- Python version: 3.9.6 (macOS system Python)
- Location: `/usr/bin/python3`
- Cannot install nflreadpy without Python upgrade

```bash
$ python3 -m pip install nflreadpy
ERROR: Requires-Python >=3.10
```

---

## Migration Path

### Step 1: Upgrade Python to 3.10+ (Choose One Method)

**Option A: Homebrew (Recommended)**
```bash
# Install Homebrew Python 3.11 or 3.12
brew install python@3.11

# Update PATH to use Homebrew Python
export PATH="/opt/homebrew/bin:$PATH"  # M1/M2 Macs
# OR
export PATH="/usr/local/bin:$PATH"     # Intel Macs

# Verify
python3 --version  # Should show 3.11.x
```

**Option B: pyenv (More Control)**
```bash
# Install pyenv
brew install pyenv

# Install Python 3.11
pyenv install 3.11.9

# Set as global or local version
pyenv global 3.11.9
# OR
cd /Users/wilfowler/Sports\ Model && pyenv local 3.11.9

# Verify
python3 --version  # Should show 3.11.9
```

**Option C: python.org Installer**
- Download from https://www.python.org/downloads/
- Install Python 3.11 or 3.12
- Update PATH if needed

### Step 2: Install nflreadpy

```bash
cd "/Users/wilfowler/Sports Model"
python3 -m pip install nflreadpy
```

### Step 3: Update Requirements

**File**: `improved_nfl_system/requirements.txt`

```diff
# Data fetching
- nfl_data_py>=0.3.1
+ nflreadpy>=0.1.0
+ polars>=0.20.0  # Required by nflreadpy
requests>=2.31.0
aiohttp>=3.9.0
```

### Step 4: Migrate Code

**Key API Changes**:

| nfl_data_py | nflreadpy | Notes |
|-------------|-----------|-------|
| `import nfl_data_py as nfl` | `import nflreadpy as nfl` | Same import pattern |
| Returns pandas DataFrames | Returns Polars DataFrames | Need `.to_pandas()` |
| `import_schedules([2024])` | `load_schedules([2024])` | Function name change |
| `import_pbp_data([2024])` | `load_pbp([2024])` | Function name change |
| `import_weekly_data([2024])` | `load_player_stats([2024])` | Function name change |

**Example Migration**:

```python
# OLD (nfl_data_py)
import nfl_data_py as nfl
schedule = nfl.import_schedules([2024])
pbp = nfl.import_pbp_data([2024])
# Already pandas DataFrames

# NEW (nflreadpy)
import nflreadpy as nfl
schedule = nfl.load_schedules([2024])
pbp = nfl.load_pbp([2024])
# Convert Polars to pandas
schedule = schedule.to_pandas()
pbp = pbp.to_pandas()
```

---

## Files Requiring Updates (22 files)

### Priority 1: Core Data Files
1. [data/nfl_data_fetcher.py](data/nfl_data_fetcher.py) - Main data fetcher (344 lines)
2. [data_pipeline.py](data_pipeline.py) - Data pipeline integration
3. [bulk_import_all_data.py](bulk_import_all_data.py) - Bulk data import script

### Priority 2: Import Scripts
4. [import_2024_season.py](import_2024_season.py)
5. [import_2025_partial.py](import_2025_partial.py)
6. [predict_week5.py](predict_week5.py)
7. [add_playoff_games_safe.py](add_playoff_games_safe.py)

### Priority 3: Testing & Utilities
8. [test_nflreadpy_migration.py](test_nflreadpy_migration.py)
9. [data_collection/historical_data_builder.py](data_collection/historical_data_builder.py)
10. [data_sources/comprehensive_loader.py](data_sources/comprehensive_loader.py)

### Priority 4: Documentation & Backups
11-22. Various documentation and backup files

---

## Migration Template

**File**: `data/nfl_data_fetcher.py` (Primary Example)

```python
"""
NFL Data Fetcher using nflreadpy
Migrated from nfl_data_py (deprecated)
"""

# OLD
# import nfl_data_py as nfl

# NEW
import nflreadpy as nfl
import pandas as pd

class NFLDataFetcher:
    def __init__(self):
        # OLD
        # test = nfl.import_team_desc()

        # NEW
        test = nfl.load_teams()
        if isinstance(test, pl.DataFrame):
            test = test.to_pandas()

        if test.empty:
            raise NFLDataError("nflreadpy returned empty team data")
        self.teams = test

    def fetch_week_games(self, season: int, week: int) -> pd.DataFrame:
        # OLD
        # schedule = nfl.import_schedules([season])

        # NEW
        schedule = nfl.load_schedules([season])
        if isinstance(schedule, pl.DataFrame):
            schedule = schedule.to_pandas()

        # Rest of code stays the same
        week_games = schedule[schedule['week'] == week].copy()
        return week_games

    def fetch_pbp_data(self, season: int, week: int) -> pd.DataFrame:
        # OLD
        # pbp = nfl.import_pbp_data([season])

        # NEW
        pbp = nfl.load_pbp([season])
        if isinstance(pbp, pl.DataFrame):
            pbp = pbp.to_pandas()

        # Rest of code stays the same
        pbp = pbp[pbp['week'] == week].copy()
        return pbp
```

---

## Testing After Migration

```bash
cd improved_nfl_system

# Test data fetching
python3 -c "
import nflreadpy as nfl
schedule = nfl.load_schedules([2024]).to_pandas()
print(f'✓ Loaded {len(schedule)} games')
"

# Test full system
python3 main.py

# Test web interface
cd web_app && python3 app.py
```

---

## Benefits of Migration

1. **Active Development**: nflreadpy is actively maintained
2. **Better Performance**: Polars is faster than pandas for large datasets
3. **More Features**: Access to new data sources and functions
4. **Better Caching**: Built-in intelligent caching
5. **Future-Proof**: Will receive updates and bug fixes

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Keep nfl_data_py installed temporarily, test nflreadpy in parallel

### Risk 2: Data Format Differences
**Mitigation**: Convert all Polars DataFrames to pandas immediately after loading

### Risk 3: Missing Data Sources
**Mitigation**: Check nflreadpy docs for equivalent functions before migration

---

## Timeline Estimate

**Total Time**: 2-3 hours

- Python upgrade: 30 min
- Install nflreadpy: 5 min
- Update requirements: 5 min
- Migrate nfl_data_fetcher.py: 45 min (most complex)
- Migrate other files: 30 min
- Testing: 30 min
- Fix issues: 15 min buffer

---

## Current Workaround

**While using Python 3.9.6**, continue using `nfl_data_py`:

```python
# This still works fine
import nfl_data_py as nfl
schedule = nfl.import_schedules([2024])  # Returns pandas directly
pbp = nfl.import_pbp_data([2024])        # Returns pandas directly
```

**Known limitations**:
- No new features will be added
- Security updates unlikely
- May break with future pandas/numpy versions
- Community support declining

**When to migrate**: Before making major new features or encountering compatibility issues

---

## Compatibility Matrix

| Component | nfl_data_py (Current) | nflreadpy (Target) |
|-----------|----------------------|-------------------|
| Python Version | 3.6+ | **3.10+** |
| Main Dependency | pandas | polars + pandas |
| Install Size | ~5 MB | ~50 MB (includes polars) |
| Performance | Good | Excellent |
| Maintenance | ❌ Deprecated | ✅ Active |
| Data Freshness | 2024 | 2025+ |

---

## Resources

- **nflreadpy Docs**: https://nflreadpy.readthedocs.io/
- **nflreadpy GitHub**: https://github.com/nflverse/nflreadpy
- **Migration Announcement**: https://github.com/cooperdff/nfl_data_py#deprecation-notice
- **Polars Docs**: https://pola-rs.github.io/polars/
- **Polars to Pandas**: https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.to_pandas.html

---

## Decision: Stay with nfl_data_py for Now ✅

**Rationale**:
1. System Python 3.9.6 cannot install nflreadpy (requires 3.10+)
2. nfl_data_py v0.3.2 still works perfectly
3. Upgrading macOS system Python can cause issues
4. Migration can happen later with Python upgrade
5. Current data needs are met

**Action Items**:
- ✅ Document the deprecation (this file)
- ✅ Add to technical debt backlog
- ✅ Plan Python upgrade when convenient
- ✅ Test migration on separate environment first

**Next Review**: When encountering compatibility issues or planning Python 3.10+ upgrade

---

*Last Updated: October 2, 2025*
*Status: Documented - Migration Deferred*
