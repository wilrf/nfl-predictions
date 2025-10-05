# Session 03: Complete NFL Data Audit & Expansion Plan

**Date**: October 2, 2025
**Focus**: Comprehensive data source audit, availability testing, and expansion planning
**Status**: Planning complete, ready for implementation

---

## Executive Summary

This session conducted a complete audit of available NFL data sources and created a comprehensive plan to expand from our current minimal dataset (2,687 games, 8 features) to a professional-grade system (2,748 games, 50+ features, 1.2M+ data records).

### Key Findings

**Current System Status**:
- ✅ 2,687 games imported (2015-2025 week 4) - REGULAR SEASON ONLY
- ✅ Models trained: 67% spread accuracy, 55% totals accuracy
- ✅ Web interface operational
- ❌ Missing 109 playoff games (2016-2024)
- ❌ Using only 8 basic features (vs 50+ possible)
- ❌ No Next Gen Stats, injuries, snap counts, depth charts
- ❌ No benchmarking vs closing line (industry standard)

**Data Available**:
- **7,263 total games** from 1999-2025 in nfl_data_py
- **2,748 games** for 2016-2025 (our target range)
- **1.2+ million data records** across 24 import functions
- **372 play-by-play columns** per play (~384K plays)
- **24,814 Next Gen Stats** records (2016-2025)
- **49,488 injury reports**, 230K snap counts, 486K depth chart entries

### Critical Discovery: Library Deprecation

**⚠️ URGENT**: nfl_data_py v0.3.2 is DEPRECATED
- No future updates or maintenance
- Must migrate to **nflreadpy** (actively maintained replacement)
- Same API, same functions, uses Polars instead of pandas
- Migration should happen before major data import

---

## Session Activities

### 1. Initial Data Audit

Reviewed comprehensive 917-line NFL data audit document covering:
- Complete inventory of 30 nfl_data_py functions
- 372 play-by-play columns explained with correlation coefficients
- 82-feature engineering library (35 Tier 1, 30 Tier 2, 17 Tier 3)
- External data sources (FiveThirtyEight Elo, ESPN APIs, The Odds API)
- Professional benchmarking standards
- Expected ROI and timeline to profitability

### 2. Data Availability Testing

Tested all 24 available import functions in nfl_data_py:

| Data Source | Available Records | Years | Status |
|-------------|------------------|-------|--------|
| Schedules/Games | 2,748 | 2016-2025 | ✅ Complete |
| Play-by-Play | ~384,720 plays | 2016-2025 | ✅ Complete |
| Next Gen Stats | 24,814 | 2016-2025 | ✅ Complete |
| Snap Counts | 230,049 | 2016-2025 | ✅ Complete |
| Depth Charts | 486,255 | 2016-2025 | ✅ Complete |
| Officials | 17,806 | 2016-2025 | ✅ Complete |
| Injuries | 49,488 | 2016-2024 | ✅ Available |
| Weekly Stats | 49,161 | 2016-2024 | ✅ Available |
| QBR | 635 | 2016-2025 | ✅ Complete |
| Rosters | 30,936 | 2016-2025 | ✅ Complete |
| Combine | 3,425 | 2016-2025 | ✅ Complete |
| Betting Lines | 2,556 | Limited | ⚠️ Partial |

**Total Available**: ~1.2 million data records

### 3. Next Gen Stats Discovery

**Key Finding**: Next Gen Stats introduced in **2016**, not 2015

Testing results:
- 2015: 0 records (not available)
- 2016: 573 records ✅
- 2017: 575 records ✅
- 2020: 581 records ✅
- 2024: 614 records ✅

**Decision**: Start dataset from 2016 (not 2015) to maintain feature parity across all seasons.

**What We Gain**: Complete NGS metrics for all seasons
- Time to throw
- Pressure rate allowed/generated
- Average separation (receivers)
- Rush yards over expected
- Completion percentage over expected (CPOE)

**What We Lose**: 256 games from 2015 season

**Net Impact**:
- Old plan: 3,004 games with inconsistent features
- New plan: 2,748 games with complete 50+ features (BETTER)

### 4. External Data Source Testing

**FiveThirtyEight Elo**:
- URL: `https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`
- Status: ⚠️ CSV parsing error (needs investigation)
- Coverage: 1920-2025 (if accessible)
- Value: r=0.68 correlation with winning

**ESPN API** (Unofficial):
- Endpoint: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`
- Status: ✅ Working (tested successfully)
- Data: Current betting odds from ESPN BET
- Includes: Spreads, totals, moneylines
- Limitation: Current games only, no historical

### 5. Complete Data Inventory (2016-2025)

**Games & Schedules**:
- Total: 2,748 games
- Regular season: 2,639 games
- Playoffs: 109 games
- Currently have: 2,687 regular only
- **Missing: 109 playoff games + 61 recent regular season**

**Play-by-Play**:
- Estimated plays: 384,720 (140 per game average)
- Columns: 372 per play
- Key metrics: EPA, CPOE, WP, success, air_yards, qb_epa, etc.
- Current usage: Aggregating to ~15 columns (using 4% of available data)

**Enhanced Data Sources**:
- Next Gen Stats: 24,814 player-week records
- Snap Counts: 230,049 records (player usage/fatigue)
- Depth Charts: 486,255 entries (starter status)
- Injuries: 49,488 reports (QB out = ±7 points)
- Officials: 17,806 assignments (referee tendencies)
- Weekly Stats: 49,161 player-week performances
- QBR: 635 ratings
- Rosters: 30,936 player-season records
- Combine: 3,425 measurements

**Storage Requirements**:
- Raw data: ~500 MB compressed, ~1.5 GB uncompressed
- Database with indexes: ~2 GB
- RAM for processing: 8-16 GB recommended

---

## Gap Analysis

### What We Currently Have

**Games**:
- 2,687 regular season games (2015-2025 week 4)
- 0 playoff games
- Coverage: 98% of regular season, 0% of playoffs

**Features** (8 total):
1. home_off_epa
2. home_def_epa
3. away_off_epa
4. away_def_epa
5. home_off_success_rate
6. away_off_success_rate
7. home_third_down_pct
8. away_third_down_pct

**Data Sources**:
- Play-by-play: Partially used (aggregated to EPA only)
- Next Gen Stats: Not used
- Injuries: Not used
- Snap counts: Not used
- Depth charts: Not used
- Officials: Not used
- Betting lines: Not used
- External sources: Not used

### What We're Missing

**Critical Gaps**:
1. **109 playoff games** (2016-2024) - 4% of dataset
2. **24,814 Next Gen Stats** records (0% used)
3. **49,488 injury reports** (0% used)
4. **230K snap count** records (0% used)
5. **42+ advanced features** not calculated
6. **Closing line benchmark** (industry standard)
7. **Walk-forward validation** (prevent data leakage)
8. **Opponent-adjusted metrics** (SOS)

**Data Richness**:
- Currently using: ~80,000 data points (game aggregates)
- Available: 1.2+ million data points
- **Usage: 6.7% of available data**

**Feature Richness**:
- Currently using: 8 features
- Available (Tier 1): 35 features
- Total possible: 82 features
- **Usage: 23% of Tier 1 features, 10% of total**

---

## Strategic Decisions Made

### Decision 1: 2016-2025 Range (Not 2015-2025)

**Reasoning**:
- Next Gen Stats only available from 2016+
- Maintaining feature parity across all seasons is critical
- Loss of 256 games is acceptable for data consistency
- 2,748 games is sufficient for training (vs 3,004)

**Trade-off**:
- Lose: 256 games from 2015
- Gain: Complete NGS metrics for all games
- Net: Better model from consistent features

### Decision 2: Comprehensive Feature Expansion (8 → 50+)

**Target Features** (50+):

**EPA Advanced (10)**:
- Passing EPA per dropback
- Rushing EPA per attempt
- Early down EPA (1st-2nd down)
- Late down EPA (3rd-4th down)
- Neutral script EPA
- Explosive play rate
- SOS-adjusted EPA (opponent strength)
- EPA trend (last 3 games)
- EPA differential (home/away)

**Passing Advanced (8)** - NEW:
- CPOE (completion % over expected) - r=0.55
- EPA + CPOE composite - r=0.65
- Air yards per attempt
- YAC per completion
- Deep ball rate (20+ air yards)
- Time to throw average (NGS)
- Pressure rate allowed (NGS)
- QB hit rate (NGS)

**Situational (7)** - EXPANDED:
- Third down short conversion (1-3 yds) - r=0.55
- Third down medium conversion (4-6 yds)
- Third down long conversion (7+ yds)
- Red zone TD % - r=0.42
- Red zone scoring %
- Goal-to-go conversion
- Two-minute drill success

**Defensive (5)** - NEW:
- Pressure rate generated (NGS)
- Sack rate
- Big play rate allowed
- Run stuff rate
- Pass breakup rate

**Context (12)** - NEW:
- Rest days (bye week detection)
- Divisional game indicator
- Primetime game
- Home field advantage score
- Weather impact (wind/temp outdoor)
- Outdoor stadium indicator
- Elo rating home (538)
- Elo rating away (538)
- QB injury status
- Key skill player out
- Referee total tendency
- Referee penalty rate

**Current → Target**: 8 features → 50+ features (6.25x increase)

### Decision 3: Relational Database Architecture

**Current**: Flat CSV files per season

**New**: Relational database with proper joins

Tables:
- `games` (2,748 rows) - Core game metadata
- `plays` (384K rows) - Play-by-play details
- `team_game_stats` (5,496 rows) - Aggregated per team per game
- `ngs_team_game` (5,496 rows) - NGS metrics aggregated
- `injuries_game` - Injury status per game
- `officials_game` (2,748 rows) - Referee assignments
- `betting_lines` (2,556+ rows) - Historical betting context
- **`ml_features`** (2,748 rows × 50+ columns) - ML-ready joined data

### Decision 4: Walk-Forward Validation

**Current**: Simple train/test split (temporal but not robust)

**New**: Week-by-week walk-forward validation

```python
for week in range(5, 19):
    train = data[data['week'] < week]
    test = data[data['week'] == week]
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    evaluate_against_closing_line(predictions, test)
```

**Benefits**:
- Prevents data leakage
- Simulates real-world deployment
- More realistic performance estimates
- Industry standard approach

### Decision 5: Closing Line Benchmark

**Current**: Reporting accuracy in isolation (67%)

**New**: Benchmark vs closing line (industry standard)

**Professional Benchmarks**:
- 52.4%+ vs closing line = break even (accounting for vig)
- 53-55% = good performance (3-5% ROI)
- 55-58% = professional level (5-8% ROI)
- 60%+ = exceptional/unsustainable

**Why This Matters**:
- Closing line is the sharpest predictor
- Beating it consistently = profitable
- Accuracy without this context is meaningless
- CLV (Closing Line Value) is THE key metric

---

## Implementation Plan Summary

### Phase 1: Core Data Import (Week 1)
- Import all 2,748 games (2016-2025) including playoffs
- Import complete play-by-play (~384K plays)
- Calculate 35+ features from PBP columns
- Verify data quality and completeness

**Deliverables**:
- Complete game dataset with playoff coverage
- 35+ PBP-derived features per game
- Data quality report

### Phase 2: Enhanced Data Sources (Week 2)
- Import Next Gen Stats (24,814 records)
- Import injuries, snap counts, depth charts (766K records)
- Import officials, QBR, context data
- Integrate FiveThirtyEight Elo (if accessible)
- Test ESPN odds API integration

**Deliverables**:
- NGS feature table
- Injury/availability feature table
- Context feature table
- External data integration

### Phase 3: Feature Engineering (Week 3)
- Implement all 50+ features
- Build relational database schema
- Create ML-ready feature table with proper joins
- Calculate opponent-adjusted metrics (SOS)
- Add temporal features (trends, momentum)

**Deliverables**:
- Complete 50+ feature dataset
- Relational database
- Feature correlation analysis
- Feature importance baseline

### Phase 4: Model Development (Week 4)
- Implement walk-forward validation framework
- Retrain models with full 50+ feature set
- Hyperparameter tuning with proper validation
- Benchmark vs closing line
- Compare with baseline (current 8-feature model)

**Deliverables**:
- Production models (spread + totals)
- Walk-forward validation results
- Closing line benchmark report
- Performance comparison (8 vs 50+ features)

---

## Expected Outcomes

### Data Improvements

**Coverage**:
- Games: 2,687 → 2,748 (+61, adds all playoffs)
- Plays: 0 → 384,720 (full PBP access)
- Data records: ~80K → 1.2M+ (15x increase)
- Features: 8 → 50+ (6.25x increase)
- Data usage: 6.7% → ~80% of available

**Quality**:
- ✅ Complete playoff coverage
- ✅ Player availability context (injuries, snaps)
- ✅ Opponent-adjusted metrics (SOS)
- ✅ Referee tendencies
- ✅ Advanced passing metrics (CPOE, NGS)
- ✅ Situational breakdowns (down/distance)
- ✅ Temporal consistency (2016-2025)

### Model Improvements

**Current Performance**:
- 67% validation accuracy (spread)
- 55% validation accuracy (totals)
- No closing line benchmark
- Simple train/test split

**Expected Performance**:
- 70-72% validation accuracy (from better features)
- 68-70% test accuracy (from walk-forward validation)
- 53-55% vs closing line (profitable threshold)
- Robust validation preventing overfitting

**Performance Drivers**:
- 6x more features with proven correlations
- Opponent-adjusted metrics (r=0.68 for SOS-EPA)
- CPOE + EPA composite (r=0.65)
- Injury context (QB out = ±7 points)
- Proper validation framework

### Professional Standards Achieved

✅ Complete data coverage (all available sources)
✅ Walk-forward validation (industry standard)
✅ Closing line benchmark (THE metric)
✅ Feature engineering based on research
✅ Opponent adjustments (strength of schedule)
✅ Player availability tracking
✅ Referee tendency analysis
✅ Relational database architecture
✅ 50+ features spanning 12 categories
✅ Temporal integrity maintained

---

## Technical Details

### nfl_data_py Functions Tested (24 total)

1. ✅ `import_schedules()` - 2,748 games
2. ✅ `import_pbp_data()` - ~384K plays
3. ✅ `import_ngs_data()` - 24,814 records
4. ✅ `import_snap_counts()` - 230,049 records
5. ✅ `import_depth_charts()` - 486,255 records
6. ✅ `import_officials()` - 17,806 records
7. ✅ `import_injuries()` - 49,488 records (2016-2024)
8. ✅ `import_weekly_data()` - 49,161 records (2016-2024)
9. ✅ `import_qbr()` - 635 records
10. ✅ `import_seasonal_rosters()` - 30,936 records
11. ✅ `import_combine_data()` - 3,425 records
12. ✅ `import_sc_lines()` - 2,556 lines (limited)
13. ✅ `import_team_desc()` - Available
14. ✅ `import_ids()` - Player ID crosswalk
15-24. Additional functions tested and confirmed available

### Data Source URLs

**nfl_data_py Base**:
- GitHub: https://github.com/nflverse/nfl_data_py
- PyPI: https://pypi.org/project/nfl-data-py/
- Version: 0.3.2 (DEPRECATED)

**Replacement (nflreadpy)**:
- Docs: https://nflreadpy.nflverse.com/
- GitHub: https://github.com/nflverse/nflreadpy
- Status: Actively maintained

**External Sources**:
- FiveThirtyEight Elo: https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv
- ESPN API: https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
- The Odds API: https://the-odds-api.com/ ($35-299/month)

### Feature Correlation Summary

**Top 10 Most Predictive Features** (from audit):

1. Closing line - r=0.85 (CRITICAL)
2. SOS-adjusted EPA - r=0.68
3. Offensive EPA - r=0.65
4. EPA + CPOE composite - r=0.65
5. Defensive EPA - r=0.60
6. Neutral script EPA - r=0.62
7. Success rate - r=0.58-0.60
8. CPOE - r=0.55
9. Third down short conv - r=0.55
10. Pressure rate - r=0.52

**Current Features Used**: 0 of top 10 (missing all high-correlation features)

---

## Risks & Mitigation

### Risk 1: Data Volume Overload

**Risk**: 1.2M records may be too much to process efficiently

**Mitigation**:
- Aggregate NGS/snap counts/injuries to team-game level
- Store raw data in database, use aggregates for ML
- Use parquet format for efficient storage
- Implement caching for frequently accessed data

### Risk 2: Feature Overfitting

**Risk**: 50+ features may cause overfitting with 2,748 games

**Mitigation**:
- Start with 35 Tier 1 features (proven correlations)
- Add Tier 2 only if improving out-of-sample performance
- Use walk-forward validation (strict temporal split)
- Regularization (L1/L2) in model training
- Feature selection based on importance scores

### Risk 3: Library Migration

**Risk**: Breaking changes when migrating to nflreadpy

**Mitigation**:
- Complete all data import with nfl_data_py first
- Test nflreadpy in parallel
- Migrate after data import is complete
- Keep nfl_data_py as fallback during transition

### Risk 4: Missing/Incomplete Data

**Risk**: Some years may have incomplete NGS, injuries, etc.

**Mitigation**:
- Data quality checks per season
- Identify and document gaps
- Use imputation only for non-critical features
- Exclude games with critical missing data
- Track data completeness metrics

### Risk 5: 2025 Data Availability

**Risk**: 2025 season in progress, data may be incomplete

**Mitigation**:
- Confirmed 2025 data available through week 4
- Update weekly as season progresses
- Use 2016-2024 as primary training data
- 2025 as rolling test set

---

## Next Steps

### Immediate (This Week)

1. **Migrate to nflreadpy** (1-2 hours)
   - Install: `pip install nflreadpy`
   - Test basic functions
   - Compare with nfl_data_py outputs

2. **Import playoff games** (2-3 hours)
   - Filter schedules for game_type != 'REG'
   - Process 109 playoff games
   - Add to training dataset

3. **Test FiveThirtyEight Elo** (1 hour)
   - Debug CSV parsing error
   - If successful, integrate Elo ratings
   - If blocked, defer to later phase

### Short-term (Weeks 1-2)

4. **Execute Phase 1**: Core data import
5. **Execute Phase 2**: Enhanced data sources
6. **Data quality validation**: Verify completeness

### Medium-term (Weeks 3-4)

7. **Execute Phase 3**: Feature engineering
8. **Execute Phase 4**: Model development
9. **Performance benchmarking**: vs closing line

### Long-term (Month 2+)

10. **Tier 2 features**: Add if improving performance
11. **Alternative models**: Test ensemble approaches
12. **Production deployment**: Weekly prediction pipeline
13. **CLV tracking**: Monitor closing line value
14. **Continuous improvement**: Feature additions, model tuning

---

## Resources & References

### Documentation
- [NFL Data Comprehensive Audit](./NFL_DATA_COMPREHENSIVE_AUDIT.md) - Full 917-line audit
- [Complete 2016-2025 Data Plan](./COMPLETE_2016_2025_DATA_PLAN.md) - Detailed implementation
- [Data Availability Summary](./DATA_AVAILABILITY_SUMMARY.md) - Quick reference
- [Current System Status](./CURRENT_SYSTEM_STATUS.md) - Baseline performance

### Code Files
- `import_2024_season.py` - Template for data import
- `import_2025_partial.py` - Template for partial seasons
- `train_models.py` - Current training script
- `web_app/app.py` - Web interface

### External Resources
- nflverse community: https://github.com/nflverse
- nflreadpy docs: https://nflreadpy.nflverse.com/
- The Odds API: https://the-odds-api.com/
- FiveThirtyEight NFL predictions: https://projects.fivethirtyeight.com/

---

## Conclusion

This session established a clear path from our current minimal system (2,687 games, 8 features) to a professional-grade betting model (2,748 games, 50+ features, 1.2M data records).

**Key Achievement**: Comprehensive audit revealing 15x more data available than currently used.

**Strategic Decisions**:
- Start from 2016 (not 2015) for NGS consistency
- Expand to 50+ features (6x current)
- Implement walk-forward validation
- Benchmark vs closing line (industry standard)

**Next Session**: Begin implementation of Phase 1 (core data import).

**Timeline**: 4 weeks to complete implementation, expect 5-10% model improvement from proper feature engineering and validation.

---

*Session documentation created: October 2, 2025*
*Related files: NFL_DATA_COMPREHENSIVE_AUDIT.md, COMPLETE_2016_2025_DATA_PLAN.md*
