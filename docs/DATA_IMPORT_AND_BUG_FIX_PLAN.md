# Complete Data Import & Bug Fix Execution Plan
**Created:** 2025-10-01
**Goal:** Fix all bugs + Import 5 seasons (2020-2024) to Supabase + Enable weekly updates

---

## Data Source Analysis ✅

### Available via nfl_data_py (FREE, UNLIMITED)
- **Games:** 1,408 games (2020-2024)
- **Play-by-Play:** ~50,000 plays/season with EPA metrics
- **Team Stats:** Comprehensive offensive/defensive stats
- **Player Stats:** Weekly player performance
- **Schedule:** Complete game schedule with dates/times
- **Injuries:** Limited availability (sparse)

### Available Columns per Game:
```python
['game_id', 'season', 'game_type', 'week', 'gameday', 'weekday', 'gametime',
 'away_team', 'away_score', 'home_team', 'home_score', 'location', 'result',
 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff']
```

### EPA Data Available:
- `epa` (Expected Points Added per play)
- `success` (Success rate)
- Situational EPA (red zone, third down, etc.)
- Team offensive/defensive EPA
- Trend analysis capabilities

### Current Data Status:
| Location | Games | EPA Stats | Features | Status |
|----------|-------|-----------|----------|--------|
| validation_data.db | 1,087 | 2,816 | 1,343 | ✅ Complete |
| Supabase | 1,343 | 0 | 0 | ⚠️ Partial |
| nfl_data_py | 1,408 | Unlimited | Unlimited | ✅ Available |

---

## Bug Inventory (9 Total)

### CRITICAL BUGS (Fix First - Day 1)

1. **Database DateTime Serialization** - `db_manager.py:83`
   - **Impact:** System crashes on game insertion
   - **Fix Time:** 30 mins
   - **Priority:** #1 (BLOCKS EVERYTHING)

2. **Kelly Math Error** - `main.py:356-360` + `main.py:414-419`
   - **Impact:** Wrong bet sizing for away bets
   - **Fix Time:** 30 mins
   - **Priority:** #2

3. **nfl_data_fetcher API Bug** - `nfl_data_fetcher.py:86, 116`
   - **Impact:** Cannot fetch play-by-play data
   - **Fix Time:** 20 mins
   - **Priority:** #3 (BLOCKS DATA IMPORT)
   - **Issue:** Using `weeks=[week]` parameter that doesn't exist

### HIGH PRIORITY BUGS (Day 1-2)

4. **Missing JSON Import** - `operations_runbook.py:247`
   - **Impact:** NameError on CLV tracking
   - **Fix Time:** 5 mins
   - **Priority:** #4

5. **Missing get_closing_line() Method** - `db_manager.py`
   - **Impact:** CLV tracking fails
   - **Fix Time:** 45 mins
   - **Priority:** #5

6. **Random CLV Generation** - `operations_runbook.py:250`
   - **Impact:** Fake data violation
   - **Fix Time:** 30 mins
   - **Priority:** #6

### MEDIUM PRIORITY BUGS (Day 2-3)

7. **Redis Hard Dependency** - `nfl_betting_system.py:54-56`
   - **Impact:** Crashes if Redis down
   - **Fix Time:** 2 hours
   - **Priority:** #7

8. **66 Fake Data Stubs** - Multiple files
   - **Impact:** FAIL FAST violations
   - **Fix Time:** 12 hours
   - **Priority:** #8

9. **Supabase Migration Incomplete** - 5 empty tables
   - **Impact:** Analytics unavailable
   - **Fix Time:** 4 hours
   - **Priority:** #9

---

## Execution Strategy

### Phase 1: Critical Bug Fixes (4 hours)
**Goal:** Unblock data import and core functionality

```bash
# Order of operations:
1. Fix database datetime bug       (30 min)
2. Fix nfl_data_fetcher API bug   (20 min)
3. Test data insertion             (10 min)
4. Fix Kelly math bug              (30 min)
5. Add missing import              (5 min)
6. Add get_closing_line()          (45 min)
7. Fix random CLV                  (30 min)
8. Test all fixes                  (1 hour)
```

**Validation After Phase 1:**
- [ ] Can insert games into database
- [ ] Can fetch PBP data from nfl_data_py
- [ ] Kelly calculations correct
- [ ] CLV tracking functional

---

### Phase 2: Data Import Setup (2 hours)
**Goal:** Create robust bulk import script

**Script Requirements:**
```python
# bulk_import_to_supabase.py

Features needed:
- Fetch 5 seasons (2020-2024) from nfl_data_py
- Calculate EPA metrics for each game
- Generate betting features
- Transform to Supabase schema
- Handle team code normalization (WSH→WAS, etc.)
- Insert via MCP with error handling
- Progress tracking
- Resume capability (if interrupted)
```

**Tables to Populate:**
1. `games` - Game schedule/results (1,408 rows)
2. `team_epa_stats` - Weekly EPA stats (~5,000 rows)
3. `game_features` - Calculated features (1,408 rows)
4. `epa_metrics` - EPA differentials (1,408 rows)
5. `team_features` - Team-specific features (~2,800 rows)
6. `betting_outcomes` - Results (historical games only, ~1,200 rows)

**Total Rows to Import:** ~12,000

---

### Phase 3: Bulk Data Import (4-6 hours runtime)
**Goal:** Load all 5 seasons to Supabase

**Execution:**
```bash
# Run import script
python bulk_import_to_supabase.py --seasons 2020,2021,2022,2023,2024

# Monitor progress
tail -f logs/data_import.log

# Verify completion
python verify_supabase_data.py
```

**Expected Output:**
```
Season 2020: 267 games → 267 games + 534 EPA stats + 267 features
Season 2021: 285 games → 285 games + 570 EPA stats + 285 features
Season 2022: 284 games → 284 games + 568 EPA stats + 284 features
Season 2023: 285 games → 285 games + 570 EPA stats + 285 features
Season 2024: 287 games → 287 games + 574 EPA stats + 287 features
---
Total: 1,408 games + ~12,000 related rows
```

---

### Phase 4: Weekly Auto-Update Setup (2 hours)
**Goal:** Keep Supabase updated with current season

**Create:** `weekly_data_updater.py`

**Features:**
- Detect current NFL season/week
- Fetch new games since last update
- Calculate features for new games
- Update Supabase incrementally
- Log all changes
- Send notifications on errors

**Schedule:**
```bash
# Run every Tuesday 6 AM (after Monday Night Football)
0 6 * * 2 cd /path/to/improved_nfl_system && python3 weekly_data_updater.py

# Or manually:
python weekly_data_updater.py --season 2024 --week 10
```

---

### Phase 5: Remaining Bugs (14 hours)
**Goal:** Clean up all fake data and optional features

**Day 3-4:**
- Redis graceful fallback (2 hours)
- Remove 66 pd.DataFrame() stubs (12 hours)
  - Test each file after modification
  - Replace with proper error handling
  - Ensure FAIL FAST behavior

---

### Phase 6: Comprehensive Testing (4 hours)
**Goal:** Verify everything works end-to-end

**Test Suite:**
```bash
# 1. Data integrity tests
pytest tests/test_data_integrity.py -v

# 2. Bug fix validation
pytest tests/test_bug_fixes.py -v

# 3. System integration test
python main.py --season 2024 --week 5

# 4. Supabase query tests
python test_supabase_queries.py

# 5. Weekly update test
python weekly_data_updater.py --dry-run
```

---

## Detailed Bug Fixes

### Bug #1: Database DateTime Serialization

**File:** `database/db_manager.py:76-87`

**Current Code:**
```python
self.conn.execute(sql, (
    game_data['game_id'],
    game_data['season'],
    game_data['week'],
    game_data['game_type'],
    game_data['home_team'],
    game_data['away_team'],
    game_data['game_time'],  # ← BUG: datetime object
    game_data['stadium'],
    game_data.get('is_outdoor', False)
))
```

**Fixed Code:**
```python
# Serialize datetime before insertion
game_time = game_data['game_time']
if isinstance(game_time, (datetime, pd.Timestamp)):
    game_time = game_time.isoformat()
elif pd.isna(game_time):
    game_time = None

self.conn.execute(sql, (
    game_data['game_id'],
    game_data['season'],
    game_data['week'],
    game_data['game_type'],
    game_data['home_team'],
    game_data['away_team'],
    game_time,  # ← Now a string
    game_data['stadium'],
    game_data.get('is_outdoor', False)
))
```

---

### Bug #3: nfl_data_fetcher API Incorrect Usage

**File:** `data/nfl_data_fetcher.py:86, 116`

**Issue:** nfl_data_py doesn't accept `weeks` parameter

**Current Code:**
```python
# Line 86
pbp = nfl.import_pbp_data([season], weeks=[week])

# Line 116
week_pbp = nfl.import_pbp_data([season], weeks=[week])
```

**Fixed Code:**
```python
# Fetch full season, then filter
pbp = nfl.import_pbp_data([season])
if pbp.empty:
    raise NFLDataError(f"No PBP data for season {season}")

# Filter to specific week
pbp = pbp[pbp['week'] == week].copy()
if pbp.empty:
    raise NFLDataError(f"No PBP data for season {season} week {week}")
```

**Note:** Fetching full season is fine - nfl_data_py caches it locally after first fetch

---

## Data Import Script Structure

```python
# bulk_import_to_supabase.py

import nfl_data_py as nfl
import pandas as pd
from typing import Dict, List
import logging

class BulkDataImporter:
    """Import 5 seasons of NFL data to Supabase"""

    def __init__(self):
        self.seasons = [2020, 2021, 2022, 2023, 2024]
        self.team_mapping = {
            'WSH': 'WAS',
            'LAR': 'LA',
            'JAC': 'JAX'
        }

    def import_all_seasons(self):
        """Main import orchestrator"""
        for season in self.seasons:
            print(f"\n{'='*60}")
            print(f"Importing Season {season}")
            print(f"{'='*60}")

            # 1. Import games
            games = self._import_games(season)

            # 2. Import EPA stats
            epa_stats = self._import_epa_stats(season)

            # 3. Calculate features
            features = self._calculate_features(games, epa_stats)

            # 4. Load to Supabase
            self._load_to_supabase(games, epa_stats, features)

            print(f"✅ Season {season} complete!")

    def _import_games(self, season: int) -> pd.DataFrame:
        """Import game schedule"""
        schedules = nfl.import_schedules([season])
        # Process and normalize
        return schedules

    def _import_epa_stats(self, season: int) -> pd.DataFrame:
        """Import and calculate EPA stats"""
        pbp = nfl.import_pbp_data([season])
        # Calculate team EPA metrics
        return self._calculate_team_epa(pbp)

    def _calculate_features(self, games, epa_stats) -> Dict:
        """Calculate all features for ML"""
        return {
            'game_features': self._calc_game_features(games, epa_stats),
            'team_features': self._calc_team_features(games, epa_stats),
            'epa_metrics': self._calc_epa_metrics(epa_stats)
        }

    def _load_to_supabase(self, games, epa_stats, features):
        """Load all data via MCP"""
        # Insert in order respecting foreign keys:
        # 1. games
        # 2. team_epa_stats
        # 3. game_features, team_features, epa_metrics
        pass

if __name__ == "__main__":
    importer = BulkDataImporter()
    importer.import_all_seasons()
```

---

## Timeline Summary

| Phase | Task | Time | Days |
|-------|------|------|------|
| 1 | Critical bug fixes | 4 hours | Day 1 |
| 2 | Data import script | 2 hours | Day 1 |
| 3 | Bulk import (runtime) | 4-6 hours | Day 1-2 |
| 4 | Weekly updater | 2 hours | Day 2 |
| 5 | Remaining bugs | 14 hours | Day 3-4 |
| 6 | Testing | 4 hours | Day 4 |
| **Total** | **30 hours** | **4 days** |

---

## Data Quality Checks

After import, verify:
```sql
-- Check row counts
SELECT 'games' as table_name, COUNT(*) FROM games
UNION ALL SELECT 'team_epa_stats', COUNT(*) FROM team_epa_stats
UNION ALL SELECT 'game_features', COUNT(*) FROM game_features
UNION ALL SELECT 'epa_metrics', COUNT(*) FROM epa_metrics
UNION ALL SELECT 'team_features', COUNT(*) FROM team_features;

-- Check for nulls in critical fields
SELECT
    COUNT(*) FILTER (WHERE home_team IS NULL) as null_home_team,
    COUNT(*) FILTER (WHERE away_team IS NULL) as null_away_team,
    COUNT(*) FILTER (WHERE gameday IS NULL) as null_gameday
FROM games;

-- Verify EPA calculations
SELECT
    season,
    COUNT(*) as games,
    AVG(home_score) as avg_home_score,
    AVG(away_score) as avg_away_score
FROM games
WHERE home_score IS NOT NULL
GROUP BY season
ORDER BY season;
```

---

## Next Steps

**Immediate Actions:**
1. ✅ Review this plan
2. ✅ Approve execution order
3. 🔄 Start Phase 1: Fix critical bugs
4. ⏳ Create bulk import script
5. ⏳ Run data import
6. ⏳ Set up weekly updates

**Questions Before We Start:**
- Approve the 4-day timeline?
- Any specific seasons you want (vs 2020-2024)?
- Should I include playoff games?
- Any additional features you want calculated?

Ready to start fixing bugs?
