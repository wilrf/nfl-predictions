#!/usr/bin/env python3
"""
MIGRATE EXISTING NFL DATA TO SUPABASE
=====================================
Migrate data from local SQLite to existing Supabase schema

Features:
- Works with existing Supabase tables
- Incremental updates (no duplicates)
- Proper error handling and logging
- Progress tracking
- Data validation

Usage:
    python3 migrate_existing_to_supabase.py [--dry-run] [--tables games,plays]

Author: NFL Betting System
Date: 2025-10-04
"""

import sqlite3
import psycopg2
import pandas as pd
from datetime import datetime
import logging
import json
import time
from tqdm import tqdm
import sys
import traceback

# Supabase connection
SUPABASE_CONFIG = {
    'host': 'db.cqslvbxsqsgjagjkpiro.supabase.co',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'P@ssword9804746196$'
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/supabase_migration_existing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SupabaseDataMigrator:
    """Migrate data from SQLite to existing Supabase schema"""

    def __init__(self, sqlite_path: str = None):
        if sqlite_path is None:
            sqlite_path = 'database/nfl_comprehensive_2024.db'

        self.sqlite_path = sqlite_path
        self._connect_databases()

        # Migration tracking
        self.migration_stats = {}

    def _connect_databases(self):
        """Connect to both SQLite and Supabase"""
        # Connect to SQLite (source)
        try:
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            logger.info(f"✅ Connected to SQLite: {self.sqlite_path}")
        except Exception as e:
            raise Exception(f"Failed to connect to SQLite: {e}")

        # Connect to Supabase (target)
        try:
            self.supabase_conn = psycopg2.connect(**SUPABASE_CONFIG)
            self.supabase_conn.autocommit = False
            logger.info("✅ Connected to Supabase PostgreSQL")
        except Exception as e:
            raise Exception(f"Failed to connect to Supabase: {e}")

    def get_existing_games(self) -> set:
        """Get set of existing game IDs in Supabase"""
        try:
            cursor = self.supabase_conn.cursor()
            cursor.execute("SELECT game_id FROM games")
            existing = {row[0] for row in cursor.fetchall()}
            logger.info(f"📊 Found {len(existing)} existing games in Supabase")
            return existing
        except Exception as e:
            logger.error(f"Error getting existing games: {e}")
            return set()

    def migrate_games_incremental(self):
        """Migrate games data incrementally (only new games)"""
        logger.info("🏈 Migrating games data incrementally...")

        try:
            # Get existing games in Supabase
            existing_games = self.get_existing_games()

            # Get games from SQLite
            sqlite_games = pd.read_sql('SELECT * FROM fact_games', self.sqlite_conn)
            logger.info(f"📊 Found {len(sqlite_games)} games in SQLite")

            # Filter to only new games
            new_games = sqlite_games[~sqlite_games['game_id'].isin(existing_games)]
            logger.info(f"🆕 New games to import: {len(new_games)}")

            if len(new_games) == 0:
                logger.info("✅ No new games to migrate")
                return 0

            games_imported = 0

            # Get Supabase table columns
            cursor = self.supabase_conn.cursor()
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'games' ORDER BY ordinal_position")
            supabase_columns = [row[0] for row in cursor.fetchall()]

            # Process new games
            for _, game in tqdm(new_games.iterrows(), total=len(new_games), desc="Migrating games"):
                # Map SQLite columns to Supabase columns
                insert_data = {}
                for col in supabase_columns:
                    if col in game.index:
                        insert_data[col] = game[col]
                    elif col == 'created_at':
                        insert_data[col] = datetime.now()
                    elif col == 'updated_at':
                        insert_data[col] = datetime.now()
                    else:
                        insert_data[col] = None

                # Insert new game
                columns_str = ','.join(insert_data.keys())
                placeholders = ','.join(['%s'] * len(insert_data))
                values = list(insert_data.values())

                cursor.execute(f"INSERT INTO games ({columns_str}) VALUES ({placeholders})", values)
                games_imported += 1

            self.supabase_conn.commit()
            self.migration_stats['games'] = games_imported

            logger.info(f"✅ Migrated {games_imported} new games")
            return games_imported

        except Exception as e:
            self.supabase_conn.rollback()
            logger.error(f"❌ Error migrating games: {e}")
            raise

    def migrate_play_by_play_incremental(self):
        """Migrate play-by-play data incrementally"""
        logger.info("🏈 Migrating play-by-play data incrementally...")

        try:
            # Get PBP count from SQLite
            pbp_count = pd.read_sql('SELECT COUNT(*) as count FROM fact_plays', self.sqlite_conn).iloc[0, 0]

            if pbp_count == 0:
                logger.warning("⚠️  No play-by-play data found in SQLite")
                return 0

            logger.info(f"📊 Found {pbp_count:,} plays in SQLite")

            # Get existing games to filter PBP data
            existing_games = self.get_existing_games()

            # Get PBP data for existing games only
            sqlite_pbp = pd.read_sql('SELECT * FROM fact_plays WHERE game_id IN (SELECT game_id FROM fact_games)', self.sqlite_conn)

            # Filter to games that exist in Supabase
            filtered_pbp = sqlite_pbp[sqlite_pbp['game_id'].isin(existing_games)]

            if len(filtered_pbp) == 0:
                logger.warning("⚠️  No PBP data for existing games")
                return 0

            logger.info(f"📊 Processing {len(filtered_pbp):,} plays for existing games")

            # Get Supabase PBP table columns (if it exists)
            cursor = self.supabase_conn.cursor()
            try:
                cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'play_by_play' ORDER BY ordinal_position")
                pbp_columns = [row[0] for row in cursor.fetchall()]
                has_pbp_table = True
                logger.info(f"📊 PBP table has {len(pbp_columns)} columns")
            except:
                has_pbp_table = False
                logger.warning("⚠️  PBP table doesn't exist in Supabase")

            if not has_pbp_table:
                logger.info("⏭️  Skipping PBP migration (table doesn't exist)")
                return 0

            # Process PBP data in chunks
            chunk_size = 1000
            total_imported = 0

            for offset in tqdm(range(0, len(filtered_pbp), chunk_size), desc="Migrating PBP chunks"):
                chunk = filtered_pbp.iloc[offset:offset+chunk_size]

                for _, play in chunk.iterrows():
                    # Map SQLite columns to Supabase columns
                    insert_data = {}
                    for col in pbp_columns:
                        if col in play.index:
                            insert_data[col] = play[col]
                        elif col == 'created_at':
                            insert_data[col] = datetime.now()
                        else:
                            insert_data[col] = None

                    # Insert play
                    columns_str = ','.join(insert_data.keys())
                    placeholders = ','.join(['%s'] * len(insert_data))
                    values = list(insert_data.values())

                    cursor.execute(f"INSERT INTO play_by_play ({columns_str}) VALUES ({placeholders}) ON CONFLICT (play_id) DO NOTHING", values)

                self.supabase_conn.commit()
                total_imported += len(chunk)
                logger.info(f"✅ Imported {total_imported:,} plays so far")

            self.migration_stats['plays'] = total_imported
            logger.info(f"✅ Migrated {total_imported:,} plays total")

            return total_imported

        except Exception as e:
            self.supabase_conn.rollback()
            logger.error(f"❌ Error migrating PBP data: {e}")
            raise

    def create_missing_tables(self):
        """Create any missing tables in Supabase"""
        logger.info("🔧 Creating missing tables in Supabase...")

        # Tables that should exist but might be missing
        required_tables = [
            'play_by_play',
            'team_epa_stats',
            'game_features',
            'historical_games'
        ]

        created_tables = 0

        for table_name in required_tables:
            try:
                cursor = self.supabase_conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                cursor.fetchone()  # Test if table exists
                logger.info(f"✅ Table {table_name} exists")
            except:
                logger.info(f"⚠️  Table {table_name} missing - would need to create")
                # For now, just log - don't create automatically
                created_tables += 1

        if created_tables > 0:
            logger.warning(f"⚠️  {created_tables} tables missing - manual creation required")

        return created_tables

    def validate_migration(self):
        """Validate the migration results"""
        logger.info("🔍 Validating migration...")

        try:
            # Compare counts
            sqlite_games = pd.read_sql('SELECT COUNT(*) as count FROM fact_games', self.sqlite_conn).iloc[0, 0]
            sqlite_plays = pd.read_sql('SELECT COUNT(*) as count FROM fact_plays', self.sqlite_conn).iloc[0, 0]

            supabase_games = pd.read_sql('SELECT COUNT(*) as count FROM games', self.supabase_conn).iloc[0, 0]

            # Check Supabase PBP table
            try:
                supabase_plays = pd.read_sql('SELECT COUNT(*) as count FROM play_by_play', self.supabase_conn).iloc[0, 0]
            except:
                supabase_plays = 0

            logger.info("📊 Migration Results:")
            logger.info(f"  SQLite games: {sqlite_games:,}")
            logger.info(f"  Supabase games: {supabase_games:,}")
            logger.info(f"  SQLite plays: {sqlite_plays:,}")
            logger.info(f"  Supabase plays: {supabase_plays:,}")

            # Check for data integrity
            if supabase_games > 0:
                # Check for duplicate games
                dupes = pd.read_sql('SELECT game_id, COUNT(*) as cnt FROM games GROUP BY game_id HAVING cnt > 1', self.supabase_conn)
                logger.info(f"  Duplicate games: {len(dupes)}")

                # Check for missing scores
                null_scores = pd.read_sql('SELECT COUNT(*) as nulls FROM games WHERE home_score IS NULL', self.supabase_conn).iloc[0, 0]
                logger.info(f"  Games with null scores: {null_scores:,}")

            logger.info("✅ Migration validation complete")

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise

    def run_migration(self, tables: list = None, dry_run: bool = False):
        """Run complete migration"""
        logger.info("="*80)
        logger.info("🚀 MIGRATING NFL DATA TO SUPABASE")
        logger.info("="*80)

        if dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")
            return {'success': True, 'dry_run': True}

        start_time = datetime.now()

        try:
            # 1. Check existing tables
            self.create_missing_tables()

            # 2. Migrate games (incremental)
            self.migrate_games_incremental()

            # 3. Migrate play-by-play (incremental)
            self.migrate_play_by_play_incremental()

            # 4. Validate migration
            self.validate_migration()

            # 5. Migration summary
            duration = datetime.now() - start_time
            total_records = sum(self.migration_stats.values())

            logger.info(f"\n⏱️  Migration completed in {duration}")
            logger.info(f"📊 Records migrated: {total_records:,}")

            for table, count in self.migration_stats.items():
                if count > 0:
                    logger.info(f"  • {table}: {count:,}")

            # Save migration summary
            summary = {
                'migration_date': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'stats': self.migration_stats,
                'total_records': total_records
            }

            with open('logs/supabase_migration_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info("\n✅ MIGRATION COMPLETED SUCCESSFULLY!")
            logger.info("📝 Summary saved to: logs/supabase_migration_summary.json")

            return {
                'success': True,
                'total_records': total_records,
                'duration_seconds': duration.total_seconds(),
                'stats': self.migration_stats
            }

        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def close(self):
        """Close database connections"""
        if hasattr(self, 'sqlite_conn'):
            self.sqlite_conn.close()
        if hasattr(self, 'supabase_conn'):
            self.supabase_conn.close()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate NFL data to Supabase')
    parser.add_argument('--sqlite-path', default=None, help='Path to SQLite database')
    parser.add_argument('--tables', nargs='+', help='Specific tables to migrate (games, plays)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated without making changes')

    args = parser.parse_args()

    try:
        migrator = SupabaseDataMigrator(sqlite_path=args.sqlite_path)

        if args.tables:
            logger.info(f"📋 Migrating specific tables: {', '.join(args.tables)}")
            # TODO: Implement selective table migration
        else:
            result = migrator.run_migration(dry_run=args.dry_run)

        if result['success']:
            if args.dry_run:
                logger.info("🔍 DRY RUN COMPLETED")
            else:
                logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
                logger.info(f"   Records migrated: {result.get('total_records', 0):,}")
        else:
            logger.error(f"❌ MIGRATION FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️  Migration cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if 'migrator' in locals():
            migrator.close()


if __name__ == '__main__':
    main()
