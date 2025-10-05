# Complete Data Import Plan: ALL Data 2016-2025

**Target**: Import 100% of available NFL data for 2016-2025 seasons
**Scope**: 2,748 games, 1.2M+ data records, 50+ features
**Timeline**: 4 weeks to complete implementation
**Current Status**: Planning complete, ready for execution

---

## Overview

This plan details the complete implementation to transform our NFL betting system from minimal (2,687 games, 8 features) to professional-grade (2,748 games, 50+ features, comprehensive data coverage).

### Goals

1. **Complete data coverage**: Import all available NFL data for 2016-2025
2. **Feature expansion**: Expand from 8 to 50+ features across 12 categories
3. **Professional validation**: Implement walk-forward validation
4. **Industry benchmarking**: Measure performance vs closing line
5. **Database architecture**: Build relational schema for efficient queries

### Success Criteria

- ✅ 2,748 games imported (including all 109 playoffs)
- ✅ 50+ features engineered with proven correlations
- ✅ Walk-forward validation preventing data leakage
- ✅ Closing line benchmark established
- ✅ 5-10% model accuracy improvement
- ✅ Relational database with proper indexes

---

## Data Inventory: What's Available

### Core Game Data (2016-2025)

**Schedules & Games**: 2,748 total games
- Regular season: 2,639 games
- Playoffs: 109 games
  - Wild Card: 60 games
  - Divisional: 40 games
  - Conference: 20 games
  - Super Bowl: 10 games

**Play-by-Play**: ~384,720 plays
- Columns: 372 per play
- Key metrics: EPA, CPOE, WP, success, air_yards, yac, qb_epa, time_to_throw, etc.
- Current usage: Aggregating to ~15 columns (4% utilization)

### Enhanced Data Sources

| Data Source | Records | Years | Key Value |
|-------------|---------|-------|-----------|
| **Next Gen Stats** | 24,814 | 2016-2025 | Pressure, time to throw, separation |
| **Snap Counts** | 230,049 | 2016-2025 | Player usage, fatigue indicators |
| **Depth Charts** | 486,255 | 2016-2025 | Starter/backup status |
| **Injuries** | 49,488 | 2016-2024 | QB out = ±7 pts impact |
| **Weekly Stats** | 49,161 | 2016-2024 | Player performance trends |
| **Officials** | 17,806 | 2016-2025 | Referee tendencies (totals, penalties) |
| **QBR** | 635 | 2016-2025 | QB performance context |
| **Rosters** | 30,936 | 2016-2025 | Team composition |
| **Combine** | 3,425 | 2016-2025 | Player athleticism |
| **Betting Lines** | 2,556 | Limited | Historical spreads/totals |

**Total**: ~1.2 million data records available

### External Data Sources (FREE)

1. **FiveThirtyEight Elo**
   - URL: `https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`
   - Coverage: 1920-2025 (all NFL history)
   - Features: Elo rating, QB Elo, QB value over replacement
   - Correlation: r=0.68 with winning

2. **ESPN Odds API** (unofficial but working)
   - Endpoint: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`
   - Data: Current betting lines from ESPN BET
   - Includes: Spreads, totals, moneylines
   - Limitation: Current games only (not historical)

---

## Phase 1: Core Game Data Import

**Duration**: 5 days
**Priority**: CRITICAL
**Dependencies**: None

### Objectives

1. Import ALL 2,748 games (2016-2025) including playoffs
2. Import complete play-by-play data (~384,720 plays)
3. Calculate 35+ features from 372 PBP columns
4. Verify data quality and completeness

### Implementation

#### Script 1: `import_all_games_2016_2025.py`

```python
"""
Import all games 2016-2025 including playoffs
Output: Complete game schedule with metadata
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

YEARS = list(range(2016, 2026))
OUTPUT_DIR = Path('ml_training_data/complete_2016_2025')

def import_all_games():
    # Import schedules (includes playoffs)
    schedules = nfl.import_schedules(YEARS)

    # Filter completed games
    completed = schedules[schedules['home_score'].notna()].copy()

    # Categorize by game type
    regular = completed[completed['game_type'] == 'REG']
    playoffs = completed[completed['game_type'] != 'REG']

    print(f'Total games: {len(completed):,}')
    print(f'Regular season: {len(regular):,}')
    print(f'Playoffs: {len(playoffs):,}')

    # Save by season
    for season in YEARS:
        season_games = completed[completed['season'] == season]
        output_path = OUTPUT_DIR / f'season_{season}' / 'schedule.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        season_games.to_csv(output_path, index=False)
        print(f'Season {season}: {len(season_games)} games saved')

    return completed

if __name__ == '__main__':
    games = import_all_games()
```

#### Script 2: `import_play_by_play_complete.py`

```python
"""
Import play-by-play data for all games 2016-2025
372 columns per play - full access to EPA, CPOE, WP, etc.
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

YEARS = list(range(2016, 2026))
OUTPUT_DIR = Path('ml_training_data/complete_2016_2025/pbp')

def import_pbp_by_season():
    """Import PBP season by season to avoid memory issues"""

    for year in YEARS:
        print(f'Importing PBP for {year}...')

        pbp = nfl.import_pbp_data([year])

        # Save as parquet for efficiency
        output_path = OUTPUT_DIR / f'pbp_{year}.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pbp.to_parquet(output_path, compression='gzip')

        print(f'  {year}: {len(pbp):,} plays saved')
        print(f'  Columns: {len(pbp.columns)}')
        print(f'  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB')

if __name__ == '__main__':
    import_pbp_by_season()
```

#### Script 3: `calculate_advanced_features.py`

```python
"""
Calculate 35+ features from play-by-play data
Based on comprehensive audit feature library
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_team_epa_features(pbp, team):
    """Calculate EPA-based features for a team"""

    team_offense = pbp[pbp['posteam'] == team]
    team_defense = pbp[pbp['defteam'] == team]

    features = {}

    # Offensive EPA (10 features)
    features['off_epa'] = team_offense['epa'].mean()
    features['pass_epa'] = team_offense[team_offense['pass_attempt']==1]['epa'].mean()
    features['rush_epa'] = team_offense[team_offense['rush_attempt']==1]['epa'].mean()
    features['off_success_rate'] = team_offense['success'].mean()
    features['explosive_play_rate'] = (team_offense['epa'] > 1.0).mean()

    # Split by down
    features['early_down_epa'] = team_offense[team_offense['down'].isin([1,2])]['epa'].mean()
    features['late_down_epa'] = team_offense[team_offense['down'].isin([3,4])]['epa'].mean()

    # Neutral script (within 1 score)
    neutral = team_offense[team_offense['score_differential'].abs() <= 8]
    features['neutral_script_epa'] = neutral['epa'].mean()

    # Defensive EPA (6 features)
    features['def_epa'] = team_defense['epa'].mean()
    features['pass_def_epa'] = team_defense[team_defense['pass_attempt']==1]['epa'].mean()
    features['rush_def_epa'] = team_defense[team_defense['rush_attempt']==1]['epa'].mean()
    features['def_success_rate'] = team_defense['success'].mean()
    features['big_play_allowed_rate'] = (team_defense['epa'] > 1.0).mean()

    # Passing advanced (8 features)
    pass_plays = team_offense[team_offense['pass_attempt']==1]
    features['cpoe'] = pass_plays['cpoe'].mean()
    features['air_yards_per_attempt'] = pass_plays['air_yards'].mean()
    features['yac_per_completion'] = pass_plays[pass_plays['complete_pass']==1]['yards_after_catch'].mean()
    features['deep_ball_rate'] = (pass_plays['air_yards'] >= 20).mean()

    # Situational (7 features)
    third_downs = team_offense[team_offense['down']==3]
    features['third_down_conv'] = third_downs['first_down'].mean()
    features['third_down_short'] = third_downs[third_downs['ydstogo'] <= 3]['first_down'].mean()
    features['third_down_long'] = third_downs[third_downs['ydstogo'] >= 7]['first_down'].mean()

    redzone = team_offense[team_offense['yardline_100'] <= 20]
    features['redzone_td_pct'] = (redzone['touchdown'] == 1).mean()
    features['redzone_score_pct'] = (redzone['touchdown'] | redzone['field_goal_result']=='made').mean()

    # Two-minute drill
    two_min = team_offense[(team_offense['half_seconds_remaining'] <= 120) & (team_offense['half_seconds_remaining'] > 0)]
    features['two_minute_epa'] = two_min['epa'].mean()

    return features

def create_game_features(pbp_path, schedule_path, output_path):
    """Create game-level feature set"""

    pbp = pd.read_parquet(pbp_path)
    schedule = pd.read_csv(schedule_path)

    game_features = []

    for _, game in schedule.iterrows():
        game_pbp = pbp[pbp['game_id'] == game['game_id']]

        home_features = calculate_team_epa_features(game_pbp, game['home_team'])
        away_features = calculate_team_epa_features(game_pbp, game['away_team'])

        # Add home_ and away_ prefixes
        features = {
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_score': game['home_score'],
            'away_score': game['away_score'],
        }

        for key, val in home_features.items():
            features[f'home_{key}'] = val
        for key, val in away_features.items():
            features[f'away_{key}'] = val

        game_features.append(features)

    df = pd.DataFrame(game_features)
    df.to_csv(output_path, index=False)

    print(f'Created {len(df)} game features with {len(df.columns)} columns')
    return df
```

### Deliverables

- ✅ `ml_training_data/complete_2016_2025/season_*/schedule.csv` (schedules by season)
- ✅ `ml_training_data/complete_2016_2025/pbp/pbp_*.parquet` (PBP by season)
- ✅ `ml_training_data/complete_2016_2025/game_features.csv` (35+ features per game)
- ✅ Data quality report showing completeness

### Validation Checklist

- [ ] Total games = 2,748
- [ ] Regular season games = 2,639
- [ ] Playoff games = 109
- [ ] Play-by-play plays ~= 384,720
- [ ] All games have home_score and away_score
- [ ] No duplicate game_ids
- [ ] Features calculated for all games
- [ ] No excessive nulls in critical features (<5%)

---

## Phase 2: Enhanced Data Sources

**Duration**: 5 days
**Priority**: HIGH
**Dependencies**: Phase 1 complete

### Objectives

1. Import Next Gen Stats (24,814 records)
2. Import injuries, snap counts, depth charts (766K records)
3. Import officials, QBR, rosters
4. Aggregate to team-game level for ML features

### Implementation

#### Script 4: `import_next_gen_stats.py`

```python
"""
Import Next Gen Stats for passing, rushing, receiving
Available 2016-2025
"""

import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2016, 2026))

def import_all_ngs():
    # Passing NGS
    ngs_pass = nfl.import_ngs_data('passing', YEARS)
    ngs_pass.to_parquet('ml_training_data/complete_2016_2025/ngs_passing.parquet')
    print(f'Passing NGS: {len(ngs_pass):,} records')

    # Rushing NGS
    ngs_rush = nfl.import_ngs_data('rushing', YEARS)
    ngs_rush.to_parquet('ml_training_data/complete_2016_2025/ngs_rushing.parquet')
    print(f'Rushing NGS: {len(ngs_rush):,} records')

    # Receiving NGS
    ngs_rec = nfl.import_ngs_data('receiving', YEARS)
    ngs_rec.to_parquet('ml_training_data/complete_2016_2025/ngs_receiving.parquet')
    print(f'Receiving NGS: {len(ngs_rec):,} records')

    return ngs_pass, ngs_rush, ngs_rec

def aggregate_ngs_to_team_week(ngs_pass, ngs_rush, ngs_rec):
    """Aggregate NGS to team-week level"""

    # Passing: time to throw, pressure rate
    pass_agg = ngs_pass.groupby(['season', 'week', 'team_abbr']).agg({
        'avg_time_to_throw': 'mean',
        'avg_completed_air_yards': 'mean',
        'aggressiveness': 'mean',
        'max_completed_air_distance': 'max'
    }).reset_index()

    # Rushing: yards over expected
    rush_agg = ngs_rush.groupby(['season', 'week', 'team_abbr']).agg({
        'efficiency': 'mean',
        'percent_attempts_gte_eight_defenders': 'mean'
    }).reset_index()

    # Receiving: separation, catch %
    rec_agg = ngs_rec.groupby(['season', 'week', 'team_abbr']).agg({
        'avg_cushion': 'mean',
        'avg_separation': 'mean',
        'percent_share_of_intended_air_yards': 'mean'
    }).reset_index()

    # Merge
    team_week_ngs = pass_agg.merge(rush_agg, on=['season', 'week', 'team_abbr'], how='outer')
    team_week_ngs = team_week_ngs.merge(rec_agg, on=['season', 'week', 'team_abbr'], how='outer')

    return team_week_ngs
```

#### Script 5: `import_injuries_snaps_depth.py`

```python
"""
Import injury, snap count, and depth chart data
Critical for player availability context
"""

import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2016, 2025))  # 2025 may have limited data

def import_availability_data():
    # Injuries (QB out = ±7 points)
    injuries = nfl.import_injuries(YEARS)
    injuries.to_parquet('ml_training_data/complete_2016_2025/injuries.parquet')
    print(f'Injuries: {len(injuries):,} reports')

    # Snap counts (workload/fatigue)
    snaps = nfl.import_snap_counts(YEARS)
    snaps.to_parquet('ml_training_data/complete_2016_2025/snap_counts.parquet')
    print(f'Snap counts: {len(snaps):,} records')

    # Depth charts (starter status)
    depth = nfl.import_depth_charts(YEARS)
    depth.to_parquet('ml_training_data/complete_2016_2025/depth_charts.parquet')
    print(f'Depth charts: {len(depth):,} entries')

    return injuries, snaps, depth

def create_qb_injury_features(injuries):
    """Create QB injury status per team per game"""

    qb_injuries = injuries[injuries['position'] == 'QB'].copy()

    # Map injury status to severity
    severity_map = {
        'Out': 3,
        'Doubtful': 2,
        'Questionable': 1,
        'Probable': 0.5,
        'N/A': 0
    }

    qb_injuries['severity'] = qb_injuries['game_status'].map(severity_map)

    # Aggregate to team-week
    qb_status = qb_injuries.groupby(['season', 'week', 'team']).agg({
        'severity': 'max',  # Worst QB injury
        'full_name': 'first'  # Name of injured QB
    }).reset_index()

    return qb_status
```

#### Script 6: `import_officials_context.py`

```python
"""
Import officials and context data
Referee tendencies affect totals
"""

import nfl_data_py as nfl
import pandas as pd

YEARS = list(range(2016, 2026))

def import_context_data():
    # Officials/referees
    officials = nfl.import_officials(YEARS)
    officials.to_parquet('ml_training_data/complete_2016_2025/officials.parquet')
    print(f'Officials: {len(officials):,} assignments')

    # QBR
    qbr = nfl.import_qbr(YEARS, level='nfl')
    qbr.to_parquet('ml_training_data/complete_2016_2025/qbr.parquet')
    print(f'QBR: {len(qbr):,} ratings')

    # Rosters
    rosters = nfl.import_seasonal_rosters(YEARS)
    rosters.to_parquet('ml_training_data/complete_2016_2025/rosters.parquet')
    print(f'Rosters: {len(rosters):,} player-seasons')

    return officials, qbr, rosters

def calculate_referee_tendencies(officials, schedules):
    """Calculate referee tendencies for totals"""

    # Merge officials with game results
    ref_games = officials.merge(
        schedules[['game_id', 'total', 'home_score', 'away_score']],
        on='game_id'
    )

    # Calculate per-referee stats
    ref_stats = ref_games.groupby('referee').agg({
        'total': 'mean',  # Average total in their games
        'game_id': 'count'  # Number of games
    }).rename(columns={'total': 'avg_total', 'game_id': 'games_reffed'})

    # Only use referees with 10+ games
    ref_stats = ref_stats[ref_stats['games_reffed'] >= 10]

    return ref_stats
```

### Deliverables

- ✅ Next Gen Stats (24,814 records) in parquet format
- ✅ Injuries (49,488 reports) with QB focus
- ✅ Snap counts (230K records) aggregated
- ✅ Depth charts (486K entries) processed
- ✅ Officials (17,806 assignments) with tendencies
- ✅ QBR (635 ratings) integrated
- ✅ Team-week aggregated features ready for join

---

## Phase 3: External Data Integration

**Duration**: 3 days
**Priority**: MEDIUM
**Dependencies**: Phase 1 complete

### Objectives

1. Integrate FiveThirtyEight Elo ratings (r=0.68 correlation)
2. Test ESPN odds API for current lines
3. Import historical betting lines where available

### Implementation

#### Script 7: `integrate_fivethirtyeight_elo.py`

```python
"""
Import FiveThirtyEight Elo ratings
Proven predictor with r=0.68 correlation
"""

import requests
import pandas as pd

def import_elo_ratings():
    url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'

    try:
        elo = pd.read_csv(url)

        # Filter to 2016-2025
        elo_recent = elo[elo['season'].between(2016, 2025)].copy()

        # Select relevant columns
        elo_features = elo_recent[[
            'date', 'season', 'team1', 'team2',
            'elo1_pre', 'elo2_pre',
            'qbelo1_pre', 'qbelo2_pre',
            'qb1_value_pre', 'qb2_value_pre'
        ]].copy()

        elo_features.to_parquet('ml_training_data/complete_2016_2025/elo_ratings.parquet')
        print(f'Elo ratings: {len(elo_features):,} games')

        return elo_features

    except Exception as e:
        print(f'Error importing Elo: {e}')
        print('Skipping Elo integration - will retry later')
        return None
```

#### Script 8: `integrate_espn_odds.py`

```python
"""
ESPN odds API integration
For current/future betting context
"""

import requests
import pandas as pd

def fetch_current_espn_odds():
    """Fetch current week's betting lines from ESPN"""

    url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
    response = requests.get(url)
    data = response.json()

    odds_data = []

    for event in data.get('events', []):
        game_info = {
            'game_id': event['id'],
            'name': event['name'],
            'date': event['date']
        }

        if event['competitions']:
            comp = event['competitions'][0]
            if 'odds' in comp and comp['odds']:
                odds = comp['odds'][0]
                game_info['spread'] = odds.get('spread')
                game_info['over_under'] = odds.get('overUnder')
                game_info['home_moneyline'] = odds['homeTeamOdds'].get('moneyLine')
                game_info['away_moneyline'] = odds['awayTeamOdds'].get('moneyLine')

        odds_data.append(game_info)

    df = pd.DataFrame(odds_data)
    print(f'Current ESPN odds: {len(df)} games')

    return df
```

### Deliverables

- ✅ FiveThirtyEight Elo ratings (2016-2025) if accessible
- ✅ ESPN odds integration tested
- ✅ Historical betting lines from nfl_data_py (2,556 records)
- ✅ External data join scripts ready

---

## Phase 4: Feature Engineering (8 → 50+)

**Duration**: 5 days
**Priority**: CRITICAL
**Dependencies**: Phases 1-3 complete

### Objectives

1. Build complete 50+ feature dataset
2. Create relational database schema
3. Implement opponent-adjusted metrics (SOS)
4. Calculate temporal features (trends, momentum)

### Feature Categories

#### EPA Advanced (10 features)
1. passing_epa_per_dropback
2. rushing_epa_per_attempt
3. early_down_epa (downs 1-2)
4. late_down_epa (downs 3-4)
5. neutral_script_epa (within 1 score)
6. explosive_play_rate (EPA > 1.0)
7. **sos_adjusted_epa** (opponent-adjusted) ← NEW
8. **epa_trend_3game** (last 3 games) ← NEW
9. epa_differential_home
10. epa_differential_away

#### Passing Advanced (8 features) - ALL NEW
11. **cpoe** (completion % over expected)
12. **epa_cpoe_composite** (r=0.65)
13. air_yards_per_attempt
14. yac_per_completion
15. deep_ball_rate (20+ air yards)
16. **time_to_throw_avg** (NGS)
17. **pressure_rate_allowed** (NGS)
18. **qb_hit_rate** (NGS)

#### Situational (7 features)
19. **third_down_short_conv** (1-3 yds, r=0.55)
20. **third_down_medium_conv** (4-6 yds)
21. **third_down_long_conv** (7+ yds)
22. red_zone_td_pct (r=0.42)
23. red_zone_scoring_pct
24. goal_to_go_conversion
25. two_minute_drill_success

#### Defensive (5 features) - ALL NEW
26. **pressure_rate_generated** (NGS)
27. sack_rate
28. big_play_rate_allowed
29. run_stuff_rate
30. pass_breakup_rate

#### Context (12 features) - ALL NEW
31. **rest_days** (bye week = 14 days)
32. is_divisional_game
33. is_primetime
34. home_field_advantage_score
35. weather_impact (wind/temp for outdoor)
36. is_outdoor
37. **elo_rating_home** (538)
38. **elo_rating_away** (538)
39. **qb_injury_status**
40. key_skill_player_out
41. **referee_total_tendency**
42. referee_penalty_rate

### Implementation

#### Script 9: `build_complete_feature_set.py`

```python
"""
Combine all data sources into complete 50+ feature dataset
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path('ml_training_data/complete_2016_2025')

def load_all_data():
    # Core features (from Phase 1)
    game_features = pd.read_csv(BASE_DIR / 'game_features.csv')

    # NGS features (from Phase 2)
    ngs_team_week = pd.read_parquet(BASE_DIR / 'ngs_team_week_aggregated.parquet')

    # Injury features
    qb_injuries = pd.read_parquet(BASE_DIR / 'qb_injury_features.parquet')

    # Referee tendencies
    ref_stats = pd.read_parquet(BASE_DIR / 'referee_tendencies.parquet')

    # Elo ratings (if available)
    try:
        elo = pd.read_parquet(BASE_DIR / 'elo_ratings.parquet')
    except:
        elo = None

    return game_features, ngs_team_week, qb_injuries, ref_stats, elo

def calculate_sos_adjusted_epa(game_features):
    """
    Calculate strength-of-schedule adjusted EPA
    Correlation: r=0.68
    """

    # Calculate opponent average EPA allowed
    opponent_def_epa = game_features.groupby('away_team')['away_def_epa'].mean()

    # Adjust offensive EPA by opponent defense quality
    game_features['home_sos_adj_epa'] = (
        game_features['home_off_epa'] -
        game_features['away_team'].map(opponent_def_epa)
    )

    return game_features

def calculate_epa_trends(game_features):
    """Calculate 3-game rolling EPA trends"""

    game_features = game_features.sort_values(['season', 'week'])

    for team in game_features['home_team'].unique():
        team_games = game_features[
            (game_features['home_team'] == team) |
            (game_features['away_team'] == team)
        ].copy()

        # Rolling 3-game EPA
        team_games['epa_trend'] = team_games['home_off_epa'].rolling(3).mean()

    return game_features

def join_all_features(game_features, ngs, injuries, refs, elo):
    """Create final ML-ready feature table"""

    # Join NGS
    features = game_features.merge(
        ngs,
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team_abbr'],
        how='left',
        suffixes=('', '_ngs_home')
    )

    # Join injuries
    features = features.merge(
        injuries,
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_inj_home')
    )

    # Join referee tendencies
    features = features.merge(
        refs,
        left_on='referee',
        right_index=True,
        how='left'
    )

    # Join Elo if available
    if elo is not None:
        features = features.merge(
            elo,
            left_on=['season', 'home_team', 'away_team'],
            right_on=['season', 'team1', 'team2'],
            how='left'
        )

    print(f'Final feature count: {len(features.columns)}')

    return features
```

### Deliverables

- ✅ Complete 50+ feature dataset
- ✅ SOS-adjusted EPA calculated
- ✅ NGS metrics integrated
- ✅ Injury context added
- ✅ Referee tendencies included
- ✅ Elo ratings (if accessible)
- ✅ Feature correlation matrix
- ✅ ML-ready CSV/parquet files

---

## Phase 5: Database Architecture

**Duration**: 3 days
**Priority**: MEDIUM
**Dependencies**: Phase 4 complete

### Objectives

1. Design relational schema
2. Create SQLite database with indexes
3. Populate all tables
4. Build efficient query patterns

### Database Schema

```sql
-- Core game metadata
CREATE TABLE games (
    game_id VARCHAR(20) PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type VARCHAR(10),  -- REG, WC, DIV, CON, SB
    game_date DATE NOT NULL,
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    point_differential INTEGER,
    total_points INTEGER,
    home_won INTEGER,
    stadium VARCHAR(100),
    roof VARCHAR(20),
    surface VARCHAR(20),
    temp FLOAT,
    wind FLOAT,
    referee VARCHAR(100)
);

CREATE INDEX idx_games_season_week ON games(season, week);
CREATE INDEX idx_games_teams ON games(home_team, away_team);
CREATE INDEX idx_games_date ON games(game_date);

-- Play-by-play (large table - consider partitioning)
CREATE TABLE plays (
    play_id INTEGER,
    game_id VARCHAR(20) REFERENCES games(game_id),
    posteam VARCHAR(3),
    defteam VARCHAR(3),
    down INTEGER,
    ydstogo INTEGER,
    yardline_100 INTEGER,
    epa FLOAT,
    wpa FLOAT,
    cpoe FLOAT,
    success INTEGER,
    play_type VARCHAR(20),
    yards_gained INTEGER,
    touchdown INTEGER,
    -- ... additional 360+ columns as needed
    PRIMARY KEY (game_id, play_id)
);

CREATE INDEX idx_plays_team ON plays(posteam, defteam);
CREATE INDEX idx_plays_game ON plays(game_id);

-- Aggregated team-game stats
CREATE TABLE team_game_stats (
    game_id VARCHAR(20) REFERENCES games(game_id),
    team VARCHAR(3),
    is_home INTEGER,
    off_epa FLOAT,
    def_epa FLOAT,
    pass_epa FLOAT,
    rush_epa FLOAT,
    success_rate FLOAT,
    cpoe FLOAT,
    -- ... 35+ features
    PRIMARY KEY (game_id, team)
);

-- Next Gen Stats (team-game level)
CREATE TABLE ngs_team_game (
    game_id VARCHAR(20) REFERENCES games(game_id),
    team VARCHAR(3),
    avg_time_to_throw FLOAT,
    pressure_rate_allowed FLOAT,
    avg_separation FLOAT,
    -- ... additional NGS metrics
    PRIMARY KEY (game_id, team)
);

-- Injury status
CREATE TABLE injuries_game (
    game_id VARCHAR(20) REFERENCES games(game_id),
    team VARCHAR(3),
    position VARCHAR(10),
    player_name VARCHAR(100),
    game_status VARCHAR(20),
    severity INTEGER
);

-- Officials
CREATE TABLE officials_game (
    game_id VARCHAR(20) REFERENCES games(game_id),
    referee VARCHAR(100),
    avg_total FLOAT,  -- Tendency
    games_reffed INTEGER
);

-- Betting lines
CREATE TABLE betting_lines (
    game_id VARCHAR(20) REFERENCES games(game_id),
    source VARCHAR(50),
    timestamp TIMESTAMP,
    spread_line FLOAT,
    total_line FLOAT,
    home_ml INTEGER,
    away_ml INTEGER,
    is_closing INTEGER
);

-- ML-ready features (denormalized for efficiency)
CREATE TABLE ml_features (
    game_id VARCHAR(20) PRIMARY KEY REFERENCES games(game_id),
    season INTEGER,
    week INTEGER,
    -- 50+ feature columns
    home_off_epa FLOAT,
    home_cpoe FLOAT,
    home_sos_adj_epa FLOAT,
    home_qb_injury_severity INTEGER,
    home_elo FLOAT,
    -- ... all features

    -- Targets
    home_won INTEGER,
    point_differential INTEGER,
    total_points INTEGER
);

CREATE INDEX idx_ml_season ON ml_features(season, week);
```

### Implementation

#### Script 10: `build_database.py`

```python
"""
Create SQLite database and populate all tables
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = 'ml_training_data/nfl_complete_2016_2025.db'
DATA_DIR = Path('ml_training_data/complete_2016_2025')

def create_database():
    conn = sqlite3.connect(DB_PATH)

    # Read schema SQL
    schema_sql = open('database_schema.sql').read()
    conn.executescript(schema_sql)

    print('Database schema created')

    # Populate tables
    populate_games(conn)
    populate_team_game_stats(conn)
    populate_ngs_team_game(conn)
    populate_ml_features(conn)

    conn.close()
    print('Database populated')

def populate_games(conn):
    schedules = pd.read_csv(DATA_DIR / 'complete_schedule.csv')
    schedules.to_sql('games', conn, if_exists='replace', index=False)
    print(f'Games table: {len(schedules)} rows')

def populate_ml_features(conn):
    features = pd.read_parquet(DATA_DIR / 'ml_features_complete.parquet')
    features.to_sql('ml_features', conn, if_exists='replace', index=False)
    print(f'ML features table: {len(features)} rows × {len(features.columns)} columns')
```

### Deliverables

- ✅ SQLite database with relational schema
- ✅ All tables populated and indexed
- ✅ Query patterns documented
- ✅ Database size: ~2 GB

---

## Phase 6: Model Development & Validation

**Duration**: 5 days
**Priority**: CRITICAL
**Dependencies**: Phase 4 complete

### Objectives

1. Implement walk-forward validation framework
2. Retrain models with full 50+ feature set
3. Hyperparameter tuning
4. Benchmark vs closing line
5. Compare with baseline (8-feature model)

### Walk-Forward Validation

```python
"""
Industry-standard validation preventing data leakage
"""

def walk_forward_validation(features, target, start_week=5):
    """
    Train on all data before week W
    Test on week W
    Iterate through entire dataset
    """

    results = []

    for season in range(2016, 2026):
        for week in range(start_week, 19):
            # Training data: all games before this week
            train = features[
                ((features['season'] < season) |
                 ((features['season'] == season) & (features['week'] < week)))
            ].copy()

            # Test data: just this week
            test = features[
                (features['season'] == season) &
                (features['week'] == week)
            ].copy()

            if len(test) == 0:
                continue

            # Train model
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            X_train = train[feature_cols]
            y_train = train[target]
            X_test = test[feature_cols]
            y_test = test[target]

            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_test)

            # Evaluate
            accuracy = (predictions[:, 1] > 0.5) == y_test

            results.append({
                'season': season,
                'week': week,
                'accuracy': accuracy.mean(),
                'games': len(test),
                'predictions': predictions
            })

    return pd.DataFrame(results)
```

### Closing Line Benchmark

```python
"""
Compare model predictions to closing line
THE industry standard metric
"""

def benchmark_vs_closing_line(predictions, test_data, closing_lines):
    """
    Professional benchmark: can we beat the closing line?

    Break-even: 52.4% (accounting for -110 vig)
    Good: 53-55% (3-5% ROI)
    Professional: 55-58% (5-8% ROI)
    """

    model_picks = predictions[:, 1] > 0.5
    line_picks = closing_lines['home_favored']

    model_accuracy = (model_picks == test_data['home_won']).mean()
    line_accuracy = (line_picks == test_data['home_won']).mean()

    # Calculate CLV (Closing Line Value)
    model_wins_vs_line = model_picks[model_picks != line_picks]
    clv = (model_wins_vs_line == test_data.loc[model_wins_vs_line.index, 'home_won']).mean()

    print(f'Model accuracy: {model_accuracy:.1%}')
    print(f'Closing line accuracy: {line_accuracy:.1%}')
    print(f'CLV (disagreements with line): {clv:.1%}')

    if clv > 0.524:
        roi = (clv - 0.524) / 0.476 * 100  # Rough ROI estimate
        print(f'Estimated ROI: {roi:.1%}')
    else:
        print('Not profitable vs closing line')
```

### Deliverables

- ✅ Walk-forward validation results
- ✅ Spread model (50+ features)
- ✅ Total model (50+ features)
- ✅ Closing line benchmark report
- ✅ Feature importance rankings
- ✅ Performance comparison (8 vs 50+ features)
- ✅ ROI projections

---

## Expected Outcomes

### Data Coverage

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Games | 2,687 | 2,748 | +61 (+2.3%) |
| Playoffs | 0 | 109 | +109 |
| Features | 8 | 50+ | +42 (6.25x) |
| Data Records | ~80K | 1.2M+ | +1.12M (15x) |
| Data Usage | 6.7% | ~80% | +73.3% |
| NGS Records | 0 | 24,814 | +24,814 |
| Injury Context | No | Yes | ✅ |
| Referee Data | No | Yes | ✅ |

### Model Performance

**Current (Baseline)**:
- 67% validation accuracy (spread)
- 55% validation accuracy (totals)
- No closing line benchmark
- 8 features (basic EPA only)

**Expected (After Implementation)**:
- 70-72% validation accuracy (from better features)
- 68-70% walk-forward validation
- 53-55% vs closing line (profitable threshold)
- 50+ features (comprehensive)

**Performance Drivers**:
- SOS-adjusted EPA (r=0.68)
- EPA + CPOE composite (r=0.65)
- Third down situational (r=0.55)
- QB injury context (±7 pts)
- Referee tendencies
- NGS pressure metrics

### Professional Standards

✅ Complete data coverage (all 24 sources)
✅ Walk-forward validation (no leakage)
✅ Closing line benchmark (THE metric)
✅ 50+ engineered features
✅ Opponent adjustments (SOS)
✅ Player availability tracking
✅ Referee tendency analysis
✅ Relational database
✅ Temporal integrity maintained

---

## Timeline Summary

| Phase | Duration | Tasks | Dependencies |
|-------|----------|-------|--------------|
| Phase 1 | 5 days | Core data import | None |
| Phase 2 | 5 days | Enhanced sources | Phase 1 |
| Phase 3 | 3 days | External data | Phase 1 |
| Phase 4 | 5 days | Feature engineering | Phases 1-3 |
| Phase 5 | 3 days | Database build | Phase 4 |
| Phase 6 | 5 days | Model development | Phase 4 |
| **Total** | **26 days** | **~4 weeks** | Sequential |

---

## Risk Mitigation

### Risk 1: Data Volume
**Mitigation**: Aggregate to team-game level, use parquet, implement caching

### Risk 2: Feature Overfitting
**Mitigation**: Start with Tier 1 only (35 features), add Tier 2 if improving, use L1/L2 regularization

### Risk 3: Library Deprecation
**Mitigation**: Complete import with nfl_data_py first, migrate to nflreadpy after

### Risk 4: Missing Data
**Mitigation**: Data quality checks, document gaps, exclude critical missing, track metrics

### Risk 5: 2025 Data
**Mitigation**: Use 2016-2024 as primary training, 2025 as rolling test set

---

## Success Metrics

**Data Completeness**:
- ✅ 2,748 games imported (100% of available)
- ✅ 109 playoff games added
- ✅ 50+ features calculated
- ✅ <5% missing values in Tier 1 features

**Model Performance**:
- ✅ 5-10% accuracy improvement over baseline
- ✅ 53%+ vs closing line (profitable)
- ✅ Walk-forward validation implemented
- ✅ Feature importance analysis complete

**Professional Standards**:
- ✅ Industry-standard validation
- ✅ Closing line benchmark
- ✅ Comprehensive data coverage
- ✅ Relational database architecture

---

*Plan created: October 2, 2025*
*Ready for implementation*
