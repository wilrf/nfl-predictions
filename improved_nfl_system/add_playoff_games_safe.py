#!/usr/bin/env python3
"""
SAFE Playoff Games Import Script
Adds 130 playoff games (2015-2024) to existing dataset
Uses EXISTING nfl_data_py library (no migration needed)
"""

import nfl_data_py as nfl
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlayoffImporter:
    """Safely import playoff games without affecting existing data"""

    def __init__(self, base_dir='ml_training_data'):
        self.base_dir = Path(base_dir)
        self.seasons = range(2015, 2025)  # 2015-2024
        self.stats = {
            'total_playoffs': 0,
            'by_season': {},
            'game_types': {}
        }

    def test_playoff_availability(self):
        """Test if playoff data is actually available"""
        logger.info("="*60)
        logger.info("STEP 1: Testing Playoff Data Availability")
        logger.info("="*60)

        try:
            # Test with one recent season
            test_season = 2023
            logger.info(f"\nTesting season {test_season}...")

            schedule = nfl.import_schedules([test_season])
            logger.info(f"Total games in schedule: {len(schedule)}")

            # Check for game_type column
            if 'game_type' in schedule.columns:
                game_types = schedule['game_type'].value_counts()
                logger.info(f"\nGame types found:")
                for game_type, count in game_types.items():
                    logger.info(f"  {game_type}: {count} games")

                # Check if playoffs exist
                playoff_games = schedule[schedule['game_type'].isin(['WC', 'DIV', 'CON', 'SB'])]
                if len(playoff_games) > 0:
                    logger.info(f"\n✅ Playoff games ARE available: {len(playoff_games)} playoff games found")
                    return True
                else:
                    logger.warning(f"\n⚠️  No playoff games found in {test_season}")
                    return False
            else:
                logger.error("❌ No 'game_type' column in schedule data")
                return False

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def import_season_playoffs(self, season):
        """Import playoff games for a specific season"""
        logger.info(f"\nProcessing season {season}...")

        try:
            # Import full schedule
            schedule = nfl.import_schedules([season])

            # Filter to playoff games only
            playoff_games = schedule[schedule['game_type'].isin(['WC', 'DIV', 'CON', 'SB'])].copy()

            if len(playoff_games) == 0:
                logger.warning(f"  No playoff games found for {season}")
                return None

            logger.info(f"  Found {len(playoff_games)} playoff games")

            # Import play-by-play for EPA calculations
            logger.info(f"  Loading play-by-play data...")
            pbp = nfl.import_pbp_data([season])

            # Filter to playoff games
            playoff_game_ids = playoff_games['game_id'].unique()
            playoff_pbp = pbp[pbp['game_id'].isin(playoff_game_ids)].copy()

            logger.info(f"  Loaded {len(playoff_pbp)} plays from playoff games")

            # Calculate team EPA stats for playoffs
            team_epa_stats = self.calculate_team_epa_stats(playoff_pbp, season)

            # Store stats
            self.stats['by_season'][season] = len(playoff_games)
            self.stats['total_playoffs'] += len(playoff_games)

            # Count game types
            for game_type in playoff_games['game_type'].unique():
                if game_type not in self.stats['game_types']:
                    self.stats['game_types'][game_type] = 0
                self.stats['game_types'][game_type] += len(playoff_games[playoff_games['game_type'] == game_type])

            return {
                'games': playoff_games,
                'pbp': playoff_pbp,
                'team_epa': team_epa_stats,
                'season': season
            }

        except Exception as e:
            logger.error(f"  ❌ Failed to import {season} playoffs: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_team_epa_stats(self, pbp_df, season):
        """Calculate EPA stats from play-by-play data"""

        # Filter to relevant plays (pass or rush only)
        valid_plays = pbp_df[
            ((pbp_df['pass'] == 1) | (pbp_df['rush'] == 1)) &
            (pbp_df['epa'].notna())
        ].copy()

        logger.info(f"  Calculating EPA from {len(valid_plays)} valid plays...")

        if len(valid_plays) == 0:
            logger.warning(f"  ⚠️  No valid plays with EPA data")
            return pd.DataFrame()

        # Calculate offensive EPA per team
        off_epa = valid_plays.groupby('posteam').agg({
            'epa': ['mean', 'sum', 'count'],
            'success': 'mean'
        }).round(4)

        off_epa.columns = ['off_epa_mean', 'off_epa_sum', 'plays', 'success_rate']

        # Calculate defensive EPA (allowed) per team
        def_epa = valid_plays.groupby('defteam').agg({
            'epa': ['mean', 'sum'],
            'success': 'mean'
        }).round(4)

        def_epa.columns = ['def_epa_allowed_mean', 'def_epa_allowed_sum', 'def_success_allowed']

        # Combine
        team_stats = off_epa.join(def_epa, how='outer')
        team_stats['season'] = season
        team_stats['is_playoff'] = True

        return team_stats.reset_index().rename(columns={'index': 'team'})

    def save_playoff_data(self, playoff_data, season):
        """Save playoff data to season directory"""

        season_dir = self.base_dir / f'season_{season}'
        season_dir.mkdir(parents=True, exist_ok=True)

        # Save playoff games
        playoff_file = season_dir / 'playoff_games.csv'
        playoff_data['games'].to_csv(playoff_file, index=False)
        logger.info(f"  ✅ Saved to {playoff_file}")

        # Save playoff EPA stats
        epa_file = season_dir / 'playoff_team_epa.csv'
        playoff_data['team_epa'].to_csv(epa_file, index=False)
        logger.info(f"  ✅ Saved EPA stats to {epa_file}")

        return True

    def run_import(self, dry_run=True):
        """Run the full playoff import process"""
        logger.info("="*60)
        logger.info("PLAYOFF GAMES IMPORT")
        logger.info("="*60)
        logger.info(f"\nMode: {'DRY RUN (no files written)' if dry_run else 'LIVE (will write files)'}")
        logger.info(f"Seasons: {self.seasons.start}-{self.seasons.stop-1}")
        logger.info("")

        # Step 1: Test availability
        if not self.test_playoff_availability():
            logger.error("\n❌ Playoff data not available - aborting")
            return False

        # Step 2: Import each season
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Importing Playoff Games by Season")
        logger.info("="*60)

        all_playoff_data = []

        for season in self.seasons:
            playoff_data = self.import_season_playoffs(season)

            if playoff_data:
                all_playoff_data.append(playoff_data)

                # Save if not dry run
                if not dry_run:
                    self.save_playoff_data(playoff_data, season)

        # Step 3: Summary
        logger.info("\n" + "="*60)
        logger.info("IMPORT SUMMARY")
        logger.info("="*60)

        logger.info(f"\nTotal playoff games imported: {self.stats['total_playoffs']}")
        logger.info(f"\nBy season:")
        for season, count in sorted(self.stats['by_season'].items()):
            logger.info(f"  {season}: {count} games")

        logger.info(f"\nBy game type:")
        for game_type, count in sorted(self.stats['game_types'].items()):
            game_type_names = {
                'WC': 'Wild Card',
                'DIV': 'Divisional',
                'CON': 'Conference Championship',
                'SB': 'Super Bowl'
            }
            logger.info(f"  {game_type_names.get(game_type, game_type)}: {count} games")

        # Expected vs actual
        expected_playoffs = len(self.seasons) * 13  # 13 playoff games per season (typically)
        logger.info(f"\nExpected: ~{expected_playoffs} games")
        logger.info(f"Actual: {self.stats['total_playoffs']} games")

        if self.stats['total_playoffs'] >= expected_playoffs - 10:  # Allow some variance
            logger.info("\n✅ SUCCESS: Playoff import complete")
        else:
            logger.warning(f"\n⚠️  Found fewer games than expected (expected ~{expected_playoffs})")

        if dry_run:
            logger.info("\n" + "="*60)
            logger.info("DRY RUN COMPLETE - No files were written")
            logger.info("="*60)
            logger.info("\nTo actually import the data, run:")
            logger.info("  python add_playoff_games_safe.py --live")
        else:
            logger.info("\n" + "="*60)
            logger.info("LIVE IMPORT COMPLETE")
            logger.info("="*60)
            logger.info(f"\nPlayoff data saved to: {self.base_dir}/season_YYYY/")
            logger.info("\nNext steps:")
            logger.info("1. Run consolidation script to merge with regular season data")
            logger.info("2. Validate combined dataset has 2,754 games")
            logger.info("3. Regenerate train/val/test splits")

        return True

def main():
    """Main entry point"""
    import sys

    # Check for --live flag
    dry_run = '--live' not in sys.argv

    importer = PlayoffImporter()
    success = importer.run_import(dry_run=dry_run)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
