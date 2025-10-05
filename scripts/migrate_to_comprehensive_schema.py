#!/usr/bin/env python3
"""
Migrate to Comprehensive Database Schema
=========================================
Purpose: Migrate from simple betting schema to comprehensive NFL data warehouse
Actions:
1. Backup current database
2. Create new comprehensive database
3. Preserve existing odds/suggestions/CLV data
4. Set up new schema for bulk import

Author: NFL Betting System
Date: 2025-10-02
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaMigrator:
    def __init__(self, old_db_path: str, new_db_path: str):
        self.old_db_path = Path(old_db_path)
        self.new_db_path = Path(new_db_path)
        self.backup_path = None
        self.schema_path = Path(__file__).parent / 'database' / 'comprehensive_schema.sql'

    def backup_current_database(self) -> Path:
        """Create timestamped backup of current database"""
        if not self.old_db_path.exists():
            logger.warning(f"Old database not found: {self.old_db_path}")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.old_db_path.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)

        self.backup_path = backup_dir / f"nfl_betting_backup_{timestamp}.db"
        shutil.copy2(self.old_db_path, self.backup_path)

        logger.info(f"✅ Backup created: {self.backup_path}")
        logger.info(f"   Size: {self.backup_path.stat().st_size / 1024:.2f} KB")

        return self.backup_path

    def create_new_database(self):
        """Create new database with comprehensive schema"""
        if self.new_db_path.exists():
            logger.warning(f"New database already exists: {self.new_db_path}")
            response = input("Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Migration cancelled")
                return False

        # Read schema file
        if not self.schema_path.exists():
            logger.error(f"Schema file not found: {self.schema_path}")
            return False

        with open(self.schema_path, 'r') as f:
            schema_sql = f.read()

        # Create new database
        conn = sqlite3.connect(self.new_db_path)
        cursor = conn.cursor()

        try:
            # Execute schema (split by ; and execute each statement)
            statements = schema_sql.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)

            conn.commit()
            logger.info(f"✅ New database created: {self.new_db_path}")

            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            logger.info(f"   Tables created: {table_count}")

            return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error creating database: {e}")
            conn.rollback()
            return False

        finally:
            conn.close()

    def migrate_existing_data(self):
        """Migrate existing odds/suggestions/CLV data from old to new database"""
        if not self.old_db_path.exists():
            logger.info("No existing database to migrate from")
            return True

        old_conn = sqlite3.connect(self.old_db_path)
        new_conn = sqlite3.connect(self.new_db_path)

        try:
            # 1. Migrate games table (simple games -> fact_games)
            logger.info("Migrating games table...")
            old_cursor = old_conn.cursor()
            old_cursor.execute("SELECT * FROM games")
            games = old_cursor.fetchall()

            if games:
                # Get column names
                old_cursor.execute("PRAGMA table_info(games)")
                old_columns = [col[1] for col in old_cursor.fetchall()]

                # Map to new schema
                new_cursor = new_conn.cursor()
                for game in games:
                    game_dict = dict(zip(old_columns, game))

                    # Insert into fact_games with only overlapping columns
                    new_cursor.execute("""
                        INSERT OR IGNORE INTO fact_games (
                            game_id, season, week, game_type,
                            home_team, away_team, game_time, stadium, is_outdoor,
                            home_score, away_score, completed, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_dict.get('game_id'),
                        game_dict.get('season'),
                        game_dict.get('week'),
                        game_dict.get('game_type', 'REG'),
                        game_dict.get('home_team'),
                        game_dict.get('away_team'),
                        game_dict.get('game_time'),
                        game_dict.get('stadium'),
                        game_dict.get('is_outdoor', 0),
                        game_dict.get('home_score'),
                        game_dict.get('away_score'),
                        game_dict.get('completed', 0),
                        game_dict.get('created_at')
                    ))

                logger.info(f"   Migrated {len(games)} games")

            # 2. Migrate odds_snapshots -> fact_odds_snapshots
            logger.info("Migrating odds snapshots...")
            old_cursor.execute("SELECT * FROM odds_snapshots")
            odds = old_cursor.fetchall()

            if odds:
                old_cursor.execute("PRAGMA table_info(odds_snapshots)")
                odds_columns = [col[1] for col in old_cursor.fetchall()]

                for odd in odds:
                    odd_dict = dict(zip(odds_columns, odd))
                    new_cursor.execute("""
                        INSERT OR IGNORE INTO fact_odds_snapshots (
                            snapshot_id, game_id, timestamp, snapshot_type, book,
                            spread_home, spread_away, total_over, total_under,
                            ml_home, ml_away, spread_odds_home, spread_odds_away,
                            total_odds_over, total_odds_under, api_credits_used
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        odd_dict.get('snapshot_id'),
                        odd_dict.get('game_id'),
                        odd_dict.get('timestamp'),
                        odd_dict.get('snapshot_type'),
                        odd_dict.get('book'),
                        odd_dict.get('spread_home'),
                        odd_dict.get('spread_away'),
                        odd_dict.get('total_over'),
                        odd_dict.get('total_under'),
                        odd_dict.get('ml_home'),
                        odd_dict.get('ml_away'),
                        odd_dict.get('spread_odds_home', -110),
                        odd_dict.get('spread_odds_away', -110),
                        odd_dict.get('total_odds_over', -110),
                        odd_dict.get('total_odds_under', -110),
                        odd_dict.get('api_credits_used', 1)
                    ))

                logger.info(f"   Migrated {len(odds)} odds snapshots")

            # 3. Migrate suggestions -> fact_suggestions
            logger.info("Migrating suggestions...")
            old_cursor.execute("SELECT * FROM suggestions")
            suggestions = old_cursor.fetchall()

            if suggestions:
                old_cursor.execute("PRAGMA table_info(suggestions)")
                sugg_columns = [col[1] for col in old_cursor.fetchall()]

                for sugg in suggestions:
                    sugg_dict = dict(zip(sugg_columns, sugg))
                    new_cursor.execute("""
                        INSERT OR IGNORE INTO fact_suggestions (
                            suggestion_id, game_id, bet_type, selection, line, odds,
                            confidence, margin, edge, kelly_fraction,
                            model_probability, market_probability,
                            suggested_at, outcome, actual_result, pnl
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sugg_dict.get('suggestion_id'),
                        sugg_dict.get('game_id'),
                        sugg_dict.get('bet_type'),
                        sugg_dict.get('selection'),
                        sugg_dict.get('line'),
                        sugg_dict.get('odds'),
                        sugg_dict.get('confidence'),
                        sugg_dict.get('margin'),
                        sugg_dict.get('edge'),
                        sugg_dict.get('kelly_fraction'),
                        sugg_dict.get('model_probability'),
                        sugg_dict.get('market_probability'),
                        sugg_dict.get('suggested_at'),
                        sugg_dict.get('outcome'),
                        sugg_dict.get('actual_result'),
                        sugg_dict.get('pnl')
                    ))

                logger.info(f"   Migrated {len(suggestions)} suggestions")

            # 4. Migrate CLV tracking -> fact_clv_tracking
            logger.info("Migrating CLV tracking...")
            old_cursor.execute("SELECT * FROM clv_tracking")
            clv = old_cursor.fetchall()

            if clv:
                old_cursor.execute("PRAGMA table_info(clv_tracking)")
                clv_columns = [col[1] for col in old_cursor.fetchall()]

                for clv_record in clv:
                    clv_dict = dict(zip(clv_columns, clv_record))
                    new_cursor.execute("""
                        INSERT OR IGNORE INTO fact_clv_tracking (
                            clv_id, suggestion_id, game_id, bet_type,
                            opening_line, closing_line, our_line,
                            clv_points, clv_percentage, beat_closing, tracked_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        clv_dict.get('clv_id'),
                        clv_dict.get('suggestion_id'),
                        clv_dict.get('game_id'),
                        clv_dict.get('bet_type'),
                        clv_dict.get('opening_line'),
                        clv_dict.get('closing_line'),
                        clv_dict.get('our_line'),
                        clv_dict.get('clv_points'),
                        clv_dict.get('clv_percentage'),
                        clv_dict.get('beat_closing'),
                        clv_dict.get('tracked_at')
                    ))

                logger.info(f"   Migrated {len(clv)} CLV records")

            new_conn.commit()
            logger.info("✅ Data migration complete")

            return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error migrating data: {e}")
            new_conn.rollback()
            return False

        finally:
            old_conn.close()
            new_conn.close()

    def populate_team_dimension(self):
        """Populate dim_teams from nfl_data_py"""
        logger.info("Populating team dimension table...")

        try:
            import nfl_data_py as nfl
            teams = nfl.import_team_desc()

            conn = sqlite3.connect(self.new_db_path)
            cursor = conn.cursor()

            for _, team in teams.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO dim_teams (
                        team_abbr, team_name, team_id, team_nick,
                        team_conf, team_division,
                        team_color, team_color2, team_color3, team_color4,
                        team_logo_wikipedia, team_logo_espn, team_wordmark,
                        team_conference_logo, team_league_logo, team_logo_squared
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    team['team_abbr'],
                    team['team_name'],
                    team.get('team_id'),
                    team.get('team_nick'),
                    team.get('team_conf'),
                    team.get('team_division'),
                    team.get('team_color'),
                    team.get('team_color2'),
                    team.get('team_color3'),
                    team.get('team_color4'),
                    team.get('team_logo_wikipedia'),
                    team.get('team_logo_espn'),
                    team.get('team_wordmark'),
                    team.get('team_conference_logo'),
                    team.get('team_league_logo'),
                    team.get('team_logo_squared')
                ))

            conn.commit()
            logger.info(f"✅ Populated {len(teams)} teams")

            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Error populating teams: {e}")
            return False

    def validate_new_database(self):
        """Run validation checks on new database"""
        logger.info("\nValidating new database...")

        conn = sqlite3.connect(self.new_db_path)
        cursor = conn.cursor()

        # Check table counts
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        logger.info(f"✅ Tables: {table_count}")

        # Check team dimension
        cursor.execute("SELECT COUNT(*) FROM dim_teams")
        team_count = cursor.fetchone()[0]
        logger.info(f"✅ Teams in dim_teams: {team_count}")

        # Check migrated games
        cursor.execute("SELECT COUNT(*) FROM fact_games")
        game_count = cursor.fetchone()[0]
        logger.info(f"✅ Games in fact_games: {game_count}")

        # Check migrated suggestions
        cursor.execute("SELECT COUNT(*) FROM fact_suggestions")
        sugg_count = cursor.fetchone()[0]
        logger.info(f"✅ Suggestions in fact_suggestions: {sugg_count}")

        # Check database size
        db_size = self.new_db_path.stat().st_size
        logger.info(f"✅ Database size: {db_size / 1024:.2f} KB")

        conn.close()

        logger.info("\n✅ Validation complete!")
        return True

    def run_migration(self, skip_backup=False):
        """Run full migration process"""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE SCHEMA MIGRATION")
        logger.info("=" * 60)

        # Step 1: Backup
        if not skip_backup:
            backup_path = self.backup_current_database()
            if backup_path:
                logger.info(f"Backup saved to: {backup_path}")

        # Step 2: Create new database
        if not self.create_new_database():
            logger.error("Failed to create new database")
            return False

        # Step 3: Populate team dimension
        if not self.populate_team_dimension():
            logger.error("Failed to populate team dimension")
            return False

        # Step 4: Migrate existing data
        if not self.migrate_existing_data():
            logger.error("Failed to migrate existing data")
            return False

        # Step 5: Validate
        if not self.validate_new_database():
            logger.error("Validation failed")
            return False

        logger.info("\n" + "=" * 60)
        logger.info("✅ MIGRATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"\nNew database: {self.new_db_path}")
        logger.info(f"Backup: {self.backup_path}")
        logger.info("\nNext steps:")
        logger.info("1. Run bulk_import_all_data.py to import NFL data")
        logger.info("2. Validate data integrity")
        logger.info("3. Update nfl_betting_system.py to use new schema")

        return True


def main():
    """Main migration execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate to comprehensive database schema')
    parser.add_argument('--old-db', default='database/nfl_betting.db', help='Path to old database')
    parser.add_argument('--new-db', default='database/nfl_comprehensive.db', help='Path to new database')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backup step')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info(f"Old database: {args.old_db}")
        logger.info(f"New database: {args.new_db}")
        logger.info(f"Skip backup: {args.skip_backup}")
        return

    migrator = SchemaMigrator(args.old_db, args.new_db)
    success = migrator.run_migration(skip_backup=args.skip_backup)

    if success:
        logger.info("\n✅ Migration successful!")
        exit(0)
    else:
        logger.error("\n❌ Migration failed!")
        exit(1)


if __name__ == '__main__':
    main()
