#!/usr/bin/env python3
"""
DAILY NFL AUTO-UPDATE SCRIPT
============================
Automatically checks for new NFL games and updates the database incrementally.

Features:
- Checks for new completed games daily
- Updates only changed/new data (incremental)
- Supports both SQLite and Supabase
- Logs all updates with timestamps
- Can be run via cron or scheduler

Usage:
    python3 daily_nfl_update.py [--supabase] [--dry-run] [--verbose]

Author: NFL Betting System
Date: 2025-10-04
"""

import sys
import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import our modules
from complete_data_import_nflreadpy import DatabaseConnection, SUPABASE_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NFLDailyUpdater:
    """Daily auto-update for NFL data"""

    def __init__(self, use_supabase: bool = False):
        self.use_supabase = use_supabase
        self.db = DatabaseConnection(use_supabase=use_supabase)
        self.today = datetime.now().date()

    def check_for_new_games(self) -> dict:
        """Check for new completed games since last update"""
        logger.info("🔍 Checking for new completed games...")

        try:
            import nflreadpy as nfl

            # Get current season and week
            current_season, current_week = nfl.get_current_season(), nfl.get_current_week()
            logger.info(f"Current NFL season: {current_season}, week: {current_week}")

            # Get all games for current season
            schedules = nfl.load_schedules([current_season])
            if hasattr(schedules, 'to_pandas'):
                schedules = schedules.to_pandas()

            # Filter to completed games (have scores)
            completed_games = schedules[schedules['home_score'].notna()].copy()
            logger.info(f"Found {len(completed_games)} completed games in season {current_season}")

            # Check which games are already in our database
            existing_games = self._get_existing_games(current_season)

            # Find new games (not in database)
            new_games = []
            for _, game in completed_games.iterrows():
                if game['game_id'] not in existing_games:
                    new_games.append(game)
                    logger.info(f"🆕 New game found: {game['game_id']} - {game['away_team']} @ {game['home_team']}")

            logger.info(f"📊 New games to import: {len(new_games)}")
            return {
                'new_games': new_games,
                'current_season': current_season,
                'current_week': current_week
            }

        except Exception as e:
            logger.error(f"❌ Error checking for new games: {e}")
            return {'new_games': [], 'error': str(e)}

    def _get_existing_games(self, season: int) -> set:
        """Get set of existing game IDs for a season"""
        try:
            if self.use_supabase:
                cursor = self.db.execute("SELECT game_id FROM fact_games WHERE season = %s", (season,))
                return {row[0] for row in cursor.fetchall()}
            else:
                cursor = self.db.execute("SELECT game_id FROM fact_games WHERE season = ?", (season,))
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.warning(f"Error getting existing games: {e}")
            return set()

    def update_games_data(self, games_data: dict) -> dict:
        """Update database with new games data"""
        logger.info("🔄 Updating database with new games...")

        new_games = games_data['new_games']
        if not new_games:
            logger.info("✅ No new games to update")
            return {'updated': 0, 'message': 'No new games'}

        try:
            games_imported = 0

            for game in new_games:
                # Insert game data
                self.db.execute("""
                    INSERT OR REPLACE INTO fact_games (
                        game_id, season, week, game_type, gameday, weekday, gametime,
                        home_team, away_team, home_score, away_score, completed,
                        location, result, total, overtime, attendance,
                        old_game_id, gsis, nfl_detail_id, pfr, pff, espn, ftn,
                        home_rest, away_rest, spread_line, home_spread_odds, away_spread_odds,
                        total_line, over_odds, under_odds, home_moneyline, away_moneyline,
                        div_game, roof, surface, temp, wind, humidity,
                        home_qb_id, away_qb_id, home_qb_name, away_qb_name,
                        home_coach, away_coach, referee, stadium_id, stadium
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    game['game_id'], game['season'], game['week'], game['game_type'],
                    game['gameday'], game.get('weekday'), game.get('gametime'),
                    game['home_team'], game['away_team'], game.get('home_score'),
                    game.get('away_score'), 1 if pd.notna(game.get('home_score')) else 0,
                    game.get('location'), game.get('result'), game.get('total'),
                    game.get('overtime', 0), game.get('attendance'),
                    game.get('old_game_id'), game.get('gsis'), game.get('nfl_detail_id'),
                    game.get('pfr'), game.get('pff'), game.get('espn'), game.get('ftn'),
                    game.get('home_rest'), game.get('away_rest'), game.get('spread_line'),
                    game.get('home_spread_odds'), game.get('away_spread_odds'),
                    game.get('total_line'), game.get('over_odds'), game.get('under_odds'),
                    game.get('home_moneyline'), game.get('away_moneyline'),
                    game.get('div_game', 0), game.get('roof'), game.get('surface'),
                    game.get('temp'), game.get('wind'), game.get('humidity'),
                    game.get('home_qb_id'), game.get('away_qb_id'),
                    game.get('home_qb_name'), game.get('away_qb_name'),
                    game.get('home_coach'), game.get('away_coach'),
                    game.get('referee'), game.get('stadium_id'), game.get('stadium')
                ))

                games_imported += 1

            self.db.commit()

            logger.info(f"✅ Successfully updated {games_imported} new games")

            return {
                'updated': games_imported,
                'season': games_data['current_season'],
                'week': games_data['current_week']
            }

        except Exception as e:
            logger.error(f"❌ Error updating games: {e}")
            self.db.rollback()
            return {'updated': 0, 'error': str(e)}

    def update_play_by_play(self, games_data: dict) -> dict:
        """Update play-by-play data for new games"""
        logger.info("🏈 Updating play-by-play data...")

        new_games = games_data['new_games']
        if not new_games:
            return {'updated': 0, 'message': 'No new games'}

        try:
            import nflreadpy as nfl

            total_plays = 0

            for game in new_games:
                # Get PBP data for this game
                pbp = nfl.load_pbp([game['season']])
                if hasattr(pbp, 'to_pandas'):
                    pbp = pbp.to_pandas()

                # Filter to this specific game
                game_pbp = pbp[pbp['game_id'] == game['game_id']]

                if len(game_pbp) == 0:
                    logger.warning(f"No PBP data found for game {game['game_id']}")
                    continue

                # Import plays for this game
                for _, play in game_pbp.iterrows():
                    # Get table columns and only use those that exist in nflreadpy data
                    table_columns = [col[1] for col in self.db.execute("PRAGMA table_info(fact_plays)").fetchall()]
                    pbp_columns = [col for col in play.index if col in table_columns]

                    # Build INSERT statement with only matching columns
                    columns_str = ','.join(pbp_columns)
                    placeholders = ','.join(['?' for _ in pbp_columns])
                    values = [play.get(col) for col in pbp_columns]

                    # Execute INSERT
                    self.db.execute(f"INSERT OR REPLACE INTO fact_plays ({columns_str}) VALUES ({placeholders})", values)

                total_plays += len(game_pbp)
                logger.info(f"✅ Imported {len(game_pbp)} plays for {game['game_id']}")

            self.db.commit()
            logger.info(f"✅ Total plays updated: {total_plays}")

            return {'updated': total_plays}

        except Exception as e:
            logger.error(f"❌ Error updating PBP data: {e}")
            self.db.rollback()
            return {'updated': 0, 'error': str(e)}

    def log_update(self, update_data: dict):
        """Log the update operation"""
        try:
            # Create update log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'daily_update',
                'games_updated': update_data.get('updated', 0),
                'season': update_data.get('season'),
                'week': update_data.get('week'),
                'success': update_data.get('error') is None,
                'error': update_data.get('error', ''),
                'message': update_data.get('message', '')
            }

            # Log to file
            with open('logs/daily_updates.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            logger.info(f"📝 Update logged: {log_entry['games_updated']} games updated")

        except Exception as e:
            logger.error(f"❌ Error logging update: {e}")

    def run_daily_update(self, dry_run: bool = False) -> dict:
        """Run complete daily update process"""
        logger.info("="*60)
        logger.info("🚀 NFL DAILY AUTO-UPDATE")
        logger.info("="*60)

        if dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")

        start_time = datetime.now()

        try:
            # 1. Check for new games
            games_data = self.check_for_new_games()

            if 'error' in games_data:
                return {'success': False, 'error': games_data['error']}

            # 2. Update games (dry run check)
            if not dry_run:
                games_result = self.update_games_data(games_data)
                pbp_result = self.update_play_by_play(games_data)

                # Log the update
                self.log_update({**games_result, **pbp_result})

                total_updated = games_result.get('updated', 0) + pbp_result.get('updated', 0)

                duration = datetime.now() - start_time
                logger.info(f"\n⏱️  Update completed in {duration}")
                logger.info(f"📊 Games updated: {games_result.get('updated', 0)}")
                logger.info(f"🏈 Plays updated: {pbp_result.get('updated', 0)}")

                return {
                    'success': True,
                    'games_updated': games_result.get('updated', 0),
                    'plays_updated': pbp_result.get('updated', 0),
                    'duration_seconds': duration.total_seconds()
                }
            else:
                # Dry run - just report what would be updated
                return {
                    'success': True,
                    'dry_run': True,
                    'new_games_found': len(games_data['new_games']),
                    'current_season': games_data['current_season'],
                    'current_week': games_data['current_week']
                }

        except Exception as e:
            logger.error(f"❌ Daily update failed: {e}")
            return {'success': False, 'error': str(e)}

    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Daily NFL data auto-update')
    parser.add_argument('--supabase', action='store_true', help='Use Supabase instead of SQLite')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        updater = NFLDailyUpdater(use_supabase=args.supabase)
        result = updater.run_daily_update(dry_run=args.dry_run)

        if result['success']:
            if args.dry_run:
                logger.info("🔍 DRY RUN RESULTS:")
                logger.info(f"   New games found: {result.get('new_games_found', 0)}")
                logger.info(f"   Current season: {result.get('current_season', 'N/A')}")
                logger.info(f"   Current week: {result.get('current_week', 'N/A')}")
            else:
                logger.info("✅ DAILY UPDATE COMPLETED SUCCESSFULLY")
                logger.info(f"   Games updated: {result.get('games_updated', 0)}")
                logger.info(f"   Plays updated: {result.get('plays_updated', 0)}")

            sys.exit(0)
        else:
            logger.error(f"❌ DAILY UPDATE FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️  Update cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if 'updater' in locals():
            updater.close()


if __name__ == '__main__':
    main()
