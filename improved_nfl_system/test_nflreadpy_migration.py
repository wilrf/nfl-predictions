#!/usr/bin/env python3
"""
Safe Migration Test: nfl_data_py → nflreadpy
Tests new library before modifying production code
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_nflreadpy_installation():
    """Test if nflreadpy can be installed and imported"""
    logger.info("="*60)
    logger.info("STEP 1: Testing nflreadpy Installation")
    logger.info("="*60)

    try:
        import nflreadpy as nfl
        logger.info("✅ nflreadpy imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ nflreadpy not installed: {e}")
        logger.info("\nTo install, run:")
        logger.info("  pip3 install nflreadpy")
        return False

def test_basic_data_fetch():
    """Test basic data fetching with nflreadpy"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Testing Basic Data Fetch")
    logger.info("="*60)

    try:
        import nflreadpy as nfl

        # Test 1: Load schedule data
        logger.info("\nTest 1: Loading 2023 schedule...")
        schedule = nfl.load_schedules([2023])
        logger.info(f"✅ Loaded {len(schedule)} games")
        logger.info(f"   Columns: {len(schedule.columns)}")
        logger.info(f"   Sample columns: {list(schedule.columns[:10])}")

        # Test 2: Check game types available
        if 'season_type' in schedule.columns:
            game_types = schedule['season_type'].value_counts()
            logger.info(f"\nGame types found:")
            for game_type, count in game_types.items():
                logger.info(f"   {game_type}: {count} games")

        # Test 3: Load play-by-play (small sample)
        logger.info("\nTest 2: Loading play-by-play data (2023, limited)...")
        pbp = nfl.load_pbp([2023])
        logger.info(f"✅ Loaded {len(pbp)} plays")
        logger.info(f"   Columns: {len(pbp.columns)}")
        logger.info(f"   Memory: {pbp.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ Data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_old_library():
    """Compare nflreadpy output with nfl_data_py"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Comparing Libraries (Same Data Test)")
    logger.info("="*60)

    try:
        # Import both libraries
        import nfl_data_py as nfl_old
        import nflreadpy as nfl_new

        # Test same season with both libraries
        test_season = 2023
        logger.info(f"\nFetching {test_season} schedules with BOTH libraries...")

        # OLD library
        old_schedule = nfl_old.import_schedules([test_season])
        logger.info(f"\nnfl_data_py (OLD):")
        logger.info(f"   Games: {len(old_schedule)}")
        logger.info(f"   Columns: {len(old_schedule.columns)}")

        # NEW library
        new_schedule = nfl_new.load_schedules([test_season])
        logger.info(f"\nnflreadpy (NEW):")
        logger.info(f"   Games: {len(new_schedule)}")
        logger.info(f"   Columns: {len(new_schedule.columns)}")

        # Compare counts
        if len(old_schedule) == len(new_schedule):
            logger.info(f"\n✅ Game counts MATCH ({len(old_schedule)} games)")
        else:
            logger.warning(f"\n⚠️  Game counts DIFFER: old={len(old_schedule)}, new={len(new_schedule)}")

        # Check if new library has playoff games
        if 'season_type' in new_schedule.columns:
            playoff_games = new_schedule[new_schedule['season_type'] == 'POST']
            logger.info(f"\n✅ NEW library has playoff support: {len(playoff_games)} playoff games found")

        return True

    except ImportError as e:
        logger.error(f"❌ Could not import libraries for comparison: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_playoff_games():
    """Verify playoff games are accessible"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Testing Playoff Game Access")
    logger.info("="*60)

    try:
        import nflreadpy as nfl

        # Test multiple seasons for playoff games
        test_seasons = [2022, 2023]
        total_playoffs = 0

        for season in test_seasons:
            schedule = nfl.load_schedules([season])

            if 'season_type' in schedule.columns:
                playoffs = schedule[schedule['season_type'] == 'POST']
                total_playoffs += len(playoffs)
                logger.info(f"\nSeason {season}:")
                logger.info(f"   Regular season: {len(schedule[schedule['season_type'] == 'REG'])} games")
                logger.info(f"   Playoffs: {len(playoffs)} games")

                if len(playoffs) > 0:
                    game_types = playoffs['game'].value_counts() if 'game' in playoffs.columns else "N/A"
                    logger.info(f"   Playoff breakdown: {dict(game_types) if game_types != 'N/A' else 'N/A'}")

        if total_playoffs > 0:
            logger.info(f"\n✅ SUCCESS: Found {total_playoffs} playoff games across {len(test_seasons)} seasons")
            return True
        else:
            logger.warning("\n⚠️  No playoff games found - check data availability")
            return False

    except Exception as e:
        logger.error(f"❌ Playoff test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_column_coverage():
    """Check if new library has more columns (features)"""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Analyzing Feature Coverage (PBP Columns)")
    logger.info("="*60)

    try:
        import nflreadpy as nfl

        # Load small sample
        pbp = nfl.load_pbp([2023])

        logger.info(f"\nPlay-by-play dataset:")
        logger.info(f"   Total columns: {len(pbp.columns)}")
        logger.info(f"   Total plays: {len(pbp):,}")

        # Check for high-value columns
        valuable_columns = [
            'epa', 'wp', 'wpa', 'cpoe', 'air_yards', 'yards_after_catch',
            'qb_hit', 'qb_hurry', 'was_pressure', 'success',
            'interception', 'fumble_lost', 'touchdown'
        ]

        found_columns = [col for col in valuable_columns if col in pbp.columns]
        missing_columns = [col for col in valuable_columns if col not in pbp.columns]

        logger.info(f"\nHigh-value columns found: {len(found_columns)}/{len(valuable_columns)}")
        if found_columns:
            logger.info(f"   ✅ Available: {', '.join(found_columns[:10])}...")
        if missing_columns:
            logger.info(f"   ⚠️  Missing: {', '.join(missing_columns)}")

        # Show sample of all columns
        logger.info(f"\nAll columns (first 30):")
        for i, col in enumerate(list(pbp.columns[:30]), 1):
            logger.info(f"   {i:2d}. {col}")

        if len(pbp.columns) > 30:
            logger.info(f"   ... and {len(pbp.columns) - 30} more columns")

        return True

    except Exception as e:
        logger.error(f"❌ Column coverage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all migration tests"""
    logger.info("\n" + "="*60)
    logger.info("NFLREADPY MIGRATION SAFETY TEST")
    logger.info("="*60)
    logger.info("\nThis script tests the new nflreadpy library before")
    logger.info("making any changes to production code.\n")

    results = {}

    # Run all tests
    results['installation'] = test_nflreadpy_installation()

    if results['installation']:
        results['basic_fetch'] = test_basic_data_fetch()
        results['comparison'] = compare_with_old_library()
        results['playoff_games'] = test_playoff_games()
        results['column_coverage'] = test_column_coverage()
    else:
        logger.error("\n❌ Cannot proceed - nflreadpy not installed")
        logger.info("\nTo install: pip3 install nflreadpy")
        return False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:10s} - {test_name}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n" + "="*60)
        logger.info("✅ ALL TESTS PASSED - SAFE TO MIGRATE")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Update requirements.txt")
        logger.info("2. Update import statements in production code")
        logger.info("3. Add playoff game import logic")
        logger.info("4. Run validation tests")
    else:
        logger.warning("\n" + "="*60)
        logger.warning("⚠️  SOME TESTS FAILED - DO NOT MIGRATE YET")
        logger.warning("="*60)

    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
