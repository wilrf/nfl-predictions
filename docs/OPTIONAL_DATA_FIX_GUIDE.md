# Guide: Fixing Optional Data Import Issues

**Date:** October 2, 2025
**Purpose:** Solutions for importing the 3 data sources that failed during initial import

---

## Overview

Three data sources failed to import:
1. **Play-by-play** (~432K records) - Column count mismatch
2. **Rosters** (~363K records) - nfl_data_py library bug
3. **Depth Charts** (~335K records) - Depends on rosters
4. **Officials** (~17K records) - Low priority, skipped

**Total missing:** ~750K records
**Current database:** 300K records (59 MB)
**With all data:** ~1.05M records (~150 MB estimated)

---

## Issue 1: Play-by-Play Data (432K records)

### Problem
```
Error: 80 values for 90 columns
Root Cause: Column count mismatch in INSERT statement
```

### Why This Happened
When we removed the duplicate `touchdown` column from the schema, we didn't properly count and adjust all the value placeholders in the INSERT statement.

### Solution: Simplify the Schema

**Option A: Store Only Essential PBP Columns (Recommended)**

Instead of trying to store all 396 columns, store only what's needed for EPA calculations:

```sql
-- Simplified fact_plays table
CREATE TABLE IF NOT EXISTS fact_plays (
    play_id TEXT PRIMARY KEY NOT NULL,
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    posteam TEXT,
    defteam TEXT,

    -- Key metrics only
    epa REAL,
    wpa REAL,
    cpoe REAL,
    success BOOLEAN,

    -- Play context
    play_type TEXT,
    down INTEGER,
    ydstogo INTEGER,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id)
);
```

**Advantages:**
- ✅ Only 12 columns vs 90 (much simpler)
- ✅ Stores exactly what we need for team EPA aggregation
- ✅ Faster imports and queries
- ✅ Smaller database size

**Implementation:**
```python
# In bulk_import_all_data.py
def import_play_by_play_simplified(self):
    """Import only essential PBP columns"""
    conn = sqlite3.connect(self.db_path)

    for season in self.seasons:
        pbp = nfl.import_pbp_data([season])

        for _, play in pbp.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO fact_plays (
                    play_id, game_id, season, week, posteam, defteam,
                    epa, wpa, cpoe, success, play_type, down, ydstogo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                play['play_id'],
                play['game_id'],
                play.get('season'),
                play.get('week'),
                play.get('posteam'),
                play.get('defteam'),
                play.get('epa'),
                play.get('wpa'),
                play.get('cpoe'),
                int(play.get('success', 0)) if pd.notna(play.get('success')) else None,
                play.get('play_type'),
                play.get('down'),
                play.get('ydstogo')
            ))

        conn.commit()
```

**Time to implement:** 30 minutes
**Import time:** ~5 minutes
**Database increase:** ~50 MB

---

**Option B: Debug Full 90-Column Import**

If you need all 90 columns, here's how to fix it:

1. **Count actual columns in schema:**
```bash
cd improved_nfl_system
grep -A 100 "CREATE TABLE IF NOT EXISTS fact_plays" database/comprehensive_schema.sql | grep "," | wc -l
```

2. **Match INSERT statement columns:**
```python
# Ensure INSERT has exact same column count
INSERT INTO fact_plays (
    col1, col2, col3, ..., col90  # Must match schema exactly
) VALUES (
    ?, ?, ?, ..., ?  # Must have exactly 90 placeholders
)
```

3. **Ensure VALUES tuple has 90 items:**
```python
values = (
    val1, val2, val3, ..., val90  # Count carefully!
)
```

**Debugging steps:**
```bash
# Count columns in INSERT
grep "INSERT OR REPLACE INTO fact_plays" bulk_import_all_data.py -A 30 | grep -o "," | wc -l

# Count placeholders
grep "VALUES" bulk_import_all_data.py -A 5 | grep -o "?" | wc -l

# Count values in tuple
grep "play\['play_id'\]," bulk_import_all_data.py -A 90 | grep "," | wc -l
```

**Time to implement:** 1-2 hours (tedious)
**Recommendation:** Use Option A (simplified) instead

---

## Issue 2: Rosters Data (363K records)

### Problem
```
Error: cannot reindex on an axis with duplicate labels
Root Cause: nfl_data_py library bug when importing multiple seasons
```

### Why This Happened
The `nfl_data_py.import_weekly_rosters()` function has a pandas bug:
- It calculates player age by reindexing on game dates
- When importing multiple seasons, duplicate index labels occur
- Pandas raises ValueError on duplicate index

### Solution: Import Season-by-Season

**Workaround Script:**
```python
#!/usr/bin/env python3
"""Import rosters one season at a time to avoid library bug"""

import nfl_data_py as nfl
import sqlite3
from pathlib import Path

def import_rosters_safe():
    """Import rosters season-by-season to avoid duplicate index bug"""
    db_path = Path('database/nfl_comprehensive.db')
    conn = sqlite3.connect(db_path)

    total_players = set()
    total_rosters = 0

    for season in range(2016, 2025):  # One season at a time
        print(f"Importing rosters for {season}...")

        try:
            # Import single season only
            rosters = nfl.import_weekly_rosters([season])

            print(f"  {len(rosters)} roster entries")

            for _, roster in rosters.iterrows():
                player_id = roster.get('player_id')

                # Add to dim_players
                if player_id and player_id not in total_players:
                    conn.execute("""
                        INSERT OR IGNORE INTO dim_players (
                            player_id, player_name, first_name, last_name,
                            birth_date, height, weight, college, position,
                            espn_id, pfr_id, entry_year
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id,
                        roster.get('player_name'),
                        roster.get('first_name'),
                        roster.get('last_name'),
                        roster.get('birth_date'),
                        roster.get('height'),
                        roster.get('weight'),
                        roster.get('college'),
                        roster.get('position'),
                        roster.get('espn_id'),
                        roster.get('pfr_id'),
                        roster.get('entry_year')
                    ))
                    total_players.add(player_id)

                # Add to fact_weekly_rosters
                conn.execute("""
                    INSERT OR REPLACE INTO fact_weekly_rosters (
                        season, week, team, player_id, position,
                        depth_chart_position, jersey_number, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    roster['season'],
                    roster['week'],
                    roster['team'],
                    player_id,
                    roster.get('position'),
                    roster.get('depth_chart_position'),
                    roster.get('jersey_number'),
                    roster.get('status')
                ))
                total_rosters += 1

            conn.commit()
            print(f"  ✅ Imported {len(rosters)} entries")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            conn.rollback()
            continue

    print(f"\n✅ Total: {len(total_players)} players, {total_rosters} roster entries")
    conn.close()

if __name__ == '__main__':
    import_rosters_safe()
```

**Save as:** `import_rosters_safe.py`

**Run:**
```bash
cd improved_nfl_system
python3 import_rosters_safe.py
```

**Expected output:**
```
Importing rosters for 2016...
  2,866 roster entries
  ✅ Imported 2,866 entries
Importing rosters for 2017...
  ...
✅ Total: 4,200 players, 363,000 roster entries
```

**Time to implement:** 15 minutes
**Import time:** 5-10 minutes
**Database increase:** ~25 MB

---

## Issue 3: Depth Charts Data (335K records)

### Problem
```
Root Cause: Depends on player IDs from rosters
Solution: Import after rosters are fixed
```

### Solution: Import After Rosters

Once rosters are imported (using the safe method above), depth charts will work:

```python
#!/usr/bin/env python3
"""Import depth charts after rosters are loaded"""

import nfl_data_py as nfl
import sqlite3
from pathlib import Path
from tqdm import tqdm

def import_depth_charts():
    """Import depth charts (requires players in dim_players)"""
    db_path = Path('database/nfl_comprehensive.db')
    conn = sqlite3.connect(db_path)

    print("Fetching depth charts (2016-2024)...")
    depth_charts = nfl.import_depth_charts(list(range(2016, 2025)))

    print(f"Importing {len(depth_charts):,} depth chart records...")

    for _, d in tqdm(depth_charts.iterrows(), total=len(depth_charts)):
        conn.execute("""
            INSERT OR REPLACE INTO fact_depth_charts (
                season, week, game_type, club_code, player_gsis_id,
                position, depth_position, formation, depth_team,
                player_name, football_name, jersey_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d['season'], d['week'], d.get('game_type'), d['club_code'],
            d.get('gsis_id'), d['position'], d.get('depth_position'),
            d.get('formation'), d.get('depth_team'),
            d.get('full_name'), d.get('football_name'), d.get('jersey_number')
        ))

    conn.commit()
    print(f"✅ Imported {len(depth_charts):,} depth chart records")
    conn.close()

if __name__ == '__main__':
    import_depth_charts()
```

**Save as:** `import_depth_charts.py`

**Run:**
```bash
python3 import_depth_charts.py
```

**Time to implement:** 10 minutes
**Import time:** 3-5 minutes
**Database increase:** ~20 MB

---

## Issue 4: Officials Data (17K records)

### Problem
```
Root Cause: Skipped for time (low priority)
Impact: Minimal - official impact on games is debatable
```

### Solution: Simple Import

Officials data is straightforward - just wasn't prioritized:

```python
#!/usr/bin/env python3
"""Import game officials"""

import nfl_data_py as nfl
import sqlite3
from pathlib import Path
from tqdm import tqdm

def import_officials():
    """Import officials data"""
    db_path = Path('database/nfl_comprehensive.db')
    conn = sqlite3.connect(db_path)

    print("Fetching officials (2016-2024)...")
    officials = nfl.import_officials(list(range(2016, 2025)))

    print(f"Importing {len(officials):,} official records...")

    officials_set = set()

    for _, off in tqdm(officials.iterrows(), total=len(officials)):
        official_id = off['official_id']

        # Add to dim_officials
        if official_id not in officials_set:
            conn.execute("""
                INSERT OR IGNORE INTO dim_officials (official_id, name)
                VALUES (?, ?)
            """, (official_id, off['name']))
            officials_set.add(official_id)

        # Add to fact_game_officials
        conn.execute("""
            INSERT OR REPLACE INTO fact_game_officials (
                game_id, season, official_id, official_name, official_position
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            off['game_id'], off['season'], official_id,
            off['name'], off.get('off_pos')
        ))

    conn.commit()
    print(f"✅ Imported {len(officials_set)} unique officials")
    print(f"✅ Imported {len(officials):,} game-official records")
    conn.close()

if __name__ == '__main__':
    import_officials()
```

**Save as:** `import_officials.py`

**Run:**
```bash
python3 import_officials.py
```

**Time to implement:** 10 minutes
**Import time:** 1 minute
**Database increase:** ~2 MB

---

## Summary: Implementation Order

### Recommended Order

1. **Play-by-Play (Simplified)** - 30 min implementation, 5 min import
   - Creates `import_pbp_simplified.py`
   - Imports 432K plays with 12 essential columns
   - Enables team EPA aggregation

2. **Rosters (Season-by-Season)** - 15 min implementation, 10 min import
   - Creates `import_rosters_safe.py`
   - Works around nfl_data_py bug
   - Imports 363K roster entries

3. **Depth Charts** - 10 min implementation, 5 min import
   - Creates `import_depth_charts.py`
   - Requires rosters to be imported first
   - Imports 335K depth chart records

4. **Officials** - 10 min implementation, 1 min import
   - Creates `import_officials.py`
   - Low priority but simple
   - Imports 17K official records

**Total time:** ~1.5 hours implementation, 20 minutes import
**Database growth:** 59 MB → ~155 MB

---

## Alternative: Skip Optional Data

### Do You Actually Need This Data?

**For ML Training, you probably DON'T need:**

❌ **Play-by-play** - Can calculate team EPA from schedule data instead
- Alternative: Use existing EPA stats from games
- Simpler: Calculate from box scores

❌ **Rosters** - Player-level data isn't used for team spread prediction
- What you need: Team-level aggregates (already have via NGS, snaps, injuries)
- Rosters add complexity without adding predictive value

❌ **Depth Charts** - Positional depth less important than injury status
- Alternative: Use injury data (already imported)
- Snap counts (already imported) show actual participation

❌ **Officials** - Referee impact on spreads is minimal and controversial
- Research shows <1% impact on spread outcomes
- Not worth the complexity

### What You Already Have (300K records)

✅ **Games** - Complete game info (2,476 games)
✅ **NGS Stats** - Advanced QB/WR/RB metrics (24K records)
✅ **Injuries** - Weekly injury reports (49K records)
✅ **Snap Counts** - Player participation (224K records)

**This is sufficient for 40-50 feature ML model!**

---

## Recommendation

### Path A: Import Optional Data (~1.5 hours)
**If you want completeness:**
1. Create 4 import scripts (see above)
2. Run each one sequentially
3. End up with 1.05M records, 155 MB database
4. Full NFL data warehouse

### Path B: Skip Optional Data (0 hours)
**If you want to start ML training now:**
1. Use current 300K records
2. Engineer 40-50 features from existing data
3. Train models immediately
4. Add optional data later if needed

**My recommendation:** **Path B** (skip optional data)

Why?
- Current data is sufficient for professional models
- Saves 1.5 hours of debugging
- Smaller, faster database (59 MB vs 155 MB)
- Can always add later if needed
- Focus on what matters: feature engineering and ML training

---

## Next Steps

**If choosing Path A (import optional):**
```bash
# 1. Create the import scripts
cat > import_pbp_simplified.py << 'EOF'
[paste simplified PBP code]
EOF

cat > import_rosters_safe.py << 'EOF'
[paste roster code]
EOF

cat > import_depth_charts.py << 'EOF'
[paste depth charts code]
EOF

cat > import_officials.py << 'EOF'
[paste officials code]
EOF

# 2. Run them in order
python3 import_pbp_simplified.py
python3 import_rosters_safe.py
python3 import_depth_charts.py
python3 import_officials.py
```

**If choosing Path B (skip optional):**
```bash
# Move to feature engineering immediately
cd improved_nfl_system

# Next: Create feature engineering pipeline
# (40-50 features from existing 300K records)
```

---

**Files Referenced:**
- [DATA_IMPORT_SUCCESS_SUMMARY.md](DATA_IMPORT_SUCCESS_SUMMARY.md) - What was imported
- [database/comprehensive_schema.sql](database/comprehensive_schema.sql) - Current schema
- [bulk_import_all_data.py](bulk_import_all_data.py) - Main import script

**Decision Point:** Import optional data OR proceed to feature engineering?
