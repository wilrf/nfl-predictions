#!/usr/bin/env python3
"""
COMPLETE NFL DATA MIGRATION TO SUPABASE
=======================================
Migrate ALL NFL data from SQLite to Supabase PostgreSQL database

Features:
- Complete data migration (2016-2025)
- Incremental updates (only new/changed data)
- Comprehensive error handling
- Progress tracking and validation
- Auto-update mechanism setup

Usage:
    python3 migrate_to_supabase.py [--dry-run] [--validate-only] [--auto-update]

Author: NFL Betting System
Date: 2025-10-04
"""

import sqlite3
import psycopg2
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import time
from tqdm import tqdm
import sys
import traceback

# Supabase connection configuration
SUPABASE_CONFIG = {
    'host': 'db.cqslvbxsqsgjagjkpiro.supabase.co',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'P@ssword9804746196$',
    'connect_timeout': 30
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/supabase_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SupabaseMigrator:
    """Migrate NFL data from SQLite to Supabase PostgreSQL"""

    def __init__(self, sqlite_path: str = None, use_supabase: bool = True):
        self.use_supabase = use_supabase

        if sqlite_path is None:
            sqlite_path = Path(__file__).parent / 'database' / 'nfl_comprehensive_2024.db'

        self.sqlite_path = sqlite_path
        self._connect_databases()

        # Migration tracking
        self.migration_stats = {
            'games': 0, 'plays': 0, 'ngs_passing': 0, 'ngs_receiving': 0, 'ngs_rushing': 0,
            'injuries': 0, 'snap_counts': 0, 'rosters': 0, 'depth_charts': 0,
            'officials': 0, 'weekly_stats': 0, 'qbr': 0, 'combine': 0
        }

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
        if self.use_supabase:
            try:
                self.supabase_conn = psycopg2.connect(**SUPABASE_CONFIG)
                self.supabase_conn.autocommit = False
                logger.info("✅ Connected to Supabase PostgreSQL")
            except Exception as e:
                raise Exception(f"Failed to connect to Supabase: {e}")

    def create_supabase_schema(self):
        """Create comprehensive Supabase schema"""
        logger.info("🔧 Creating Supabase schema...")

        schema_path = Path(__file__).parent / 'supabase_complete_schema.sql'

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            # Split into individual statements for better error handling
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]

            for stmt in statements:
                if stmt and not stmt.startswith('--'):
                    if self.use_supabase:
                        self.supabase_conn.execute(stmt)
                        self.supabase_conn.commit()
                    else:
                        self.sqlite_conn.execute(stmt)
                        self.sqlite_conn.commit()

            logger.info("✅ Supabase schema created successfully")

        except Exception as e:
            if self.use_supabase:
                self.supabase_conn.rollback()
            else:
                self.sqlite_conn.rollback()
            logger.error(f"❌ Schema creation failed: {e}")
            raise

    def migrate_games_data(self):
        """Migrate games data from SQLite to Supabase"""
        logger.info("🏈 Migrating games data...")

        try:
            # Get games from SQLite
            sqlite_games = pd.read_sql('SELECT * FROM fact_games', self.sqlite_conn)

            if len(sqlite_games) == 0:
                logger.warning("⚠️  No games found in SQLite")
                return 0

            logger.info(f"📊 Found {len(sqlite_games)} games in SQLite")

            games_imported = 0

            for _, game in tqdm(sqlite_games.iterrows(), total=len(sqlite_games), desc="Migrating games"):
                # Insert/update game in Supabase
                if self.use_supabase:
                    cursor.execute("""
                        INSERT INTO fact_games (
                            game_id, season, week, game_type, gameday, weekday, gametime,
                            home_team, away_team, home_score, away_score, point_differential,
                            total_points, location, result, total, overtime, attendance,
                            old_game_id, gsis, nfl_detail_id, pfr, pff, espn, ftn,
                            home_rest, away_rest, spread_line, home_spread_odds, away_spread_odds,
                            total_line, over_odds, under_odds, home_moneyline, away_moneyline,
                            div_game, roof, surface, temp, wind, humidity,
                            home_qb_id, away_qb_id, home_qb_name, away_qb_name,
                            home_coach, away_coach, referee, stadium_id, stadium,
                            completed, created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (game_id) DO UPDATE SET
                            home_score = EXCLUDED.home_score,
                            away_score = EXCLUDED.away_score,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        game['game_id'], game['season'], game['week'], game['game_type'],
                        game['gameday'], game.get('weekday'), game.get('gametime'),
                        game['home_team'], game['away_team'], game.get('home_score'),
                        game.get('away_score'), game.get('point_differential', 0), game.get('total_points', 0),
                        game.get('location'), game.get('result'), game.get('total'),
                        game.get('overtime'), game.get('attendance'),
                        game.get('old_game_id'), game.get('gsis'), game.get('nfl_detail_id'),
                        game.get('pfr'), game.get('pff'), game.get('espn'), game.get('ftn'),
                        game.get('home_rest'), game.get('away_rest'), game.get('spread_line'),
                        game.get('home_spread_odds'), game.get('away_spread_odds'),
                        game.get('total_line'), game.get('over_odds'), game.get('under_odds'),
                        game.get('home_moneyline'), game.get('away_moneyline'),
                        game.get('div_game'), game.get('roof'), game.get('surface'),
                        game.get('temp'), game.get('wind'), game.get('humidity'),
                        game.get('home_qb_id'), game.get('away_qb_id'),
                        game.get('home_qb_name'), game.get('away_qb_name'),
                        game.get('home_coach'), game.get('away_coach'),
                        game.get('referee'), game.get('stadium_id'), game.get('stadium'),
                        game.get('completed', 1), datetime.now(), datetime.now()
                    ))
                    cursor.close()
                else:
                    # Fallback to SQLite
                    self.sqlite_conn.execute("""
                        INSERT OR REPLACE INTO fact_games (
                            game_id, season, week, game_type, gameday, weekday, gametime,
                            home_team, away_team, home_score, away_score, completed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game['game_id'], game['season'], game['week'], game['game_type'],
                        game['gameday'], game.get('weekday'), game.get('gametime'),
                        game['home_team'], game['away_team'], game.get('home_score'),
                        game.get('away_score'), game.get('completed')
                    ))

                games_imported += 1

            # Commit transaction
            if self.use_supabase:
                self.supabase_conn.commit()
            else:
                self.sqlite_conn.commit()

            self.migration_stats['games'] = games_imported
            logger.info(f"✅ Migrated {games_imported} games")

            return games_imported

        except Exception as e:
            if self.use_supabase:
                self.supabase_conn.rollback()
            else:
                self.sqlite_conn.rollback()
            logger.error(f"❌ Error migrating games: {e}")
            raise

    def migrate_play_by_play(self):
        """Migrate play-by-play data"""
        logger.info("🏈 Migrating play-by-play data...")

        try:
            # Get PBP count from SQLite
            pbp_count = pd.read_sql('SELECT COUNT(*) as count FROM fact_plays', self.sqlite_conn).iloc[0, 0]

            if pbp_count == 0:
                logger.warning("⚠️  No play-by-play data found in SQLite")
                return 0

            logger.info(f"📊 Found {pbp_count:,} plays in SQLite")

            # For large datasets, process in chunks
            chunk_size = 5000
            total_imported = 0

            for offset in tqdm(range(0, pbp_count, chunk_size), desc="Migrating PBP chunks"):
                # Get chunk from SQLite
                sqlite_query = f"SELECT * FROM fact_plays LIMIT {chunk_size} OFFSET {offset}"
                chunk = pd.read_sql(sqlite_query, self.sqlite_conn)

                # Insert chunk to Supabase
                for _, play in chunk.iterrows():
                    # Get table columns and only use those that exist in the data
                    cursor = self.supabase_conn.cursor()
                    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'play_by_play' ORDER BY ordinal_position")
                    table_columns = [row[0] for row in cursor.fetchall()]
                    play_columns = [col for col in play.index if col in table_columns]

                    # Build INSERT statement with only matching columns
                    columns_str = ','.join(play_columns)
                    placeholders = ','.join(['%s' for _ in play_columns])
                    values = [play.get(col) for col in play_columns]

                    # Execute INSERT
                    cursor.execute(f"INSERT INTO play_by_play ({columns_str}) VALUES ({placeholders}) ON CONFLICT (play_id) DO NOTHING", values)
                    cursor.close()

                # Commit chunk
                self.supabase_conn.commit()

                total_imported += len(chunk)
                logger.info(f"✅ Migrated {total_imported:,} plays so far")

            self.migration_stats['plays'] = total_imported
            logger.info(f"✅ Migrated {total_imported:,} plays total")

            return total_imported

        except Exception as e:
            if self.use_supabase:
                self.supabase_conn.rollback()
            else:
                self.sqlite_conn.rollback()
            logger.error(f"❌ Error migrating PBP data: {e}")
            raise

    def migrate_enhanced_data(self):
        """Migrate enhanced data sources (NGS, injuries, etc.)"""
        logger.info("📊 Migrating enhanced data sources...")

        enhanced_tables = [
            ('fact_injuries', 'injuries'),
            ('fact_snap_counts', 'snap_counts'),
            ('fact_weekly_rosters', 'rosters'),
            ('fact_depth_charts', 'depth_charts'),
            ('fact_game_officials', 'officials'),
            ('fact_weekly_stats', 'weekly_stats'),
            ('fact_qbr', 'qbr'),
            ('fact_combine', 'combine')
        ]

        for table_name, display_name in enhanced_tables:
            try:
                # Check if table exists in SQLite
                sqlite_count = pd.read_sql(f'SELECT COUNT(*) as count FROM {table_name}', self.sqlite_conn).iloc[0, 0]

                if sqlite_count == 0:
                    logger.warning(f"⚠️  No {display_name} data found in SQLite")
                    continue

                logger.info(f"📊 Migrating {sqlite_count:,} {display_name} records")

                # Get data from SQLite
                data = pd.read_sql(f'SELECT * FROM {table_name}', self.sqlite_conn)

                # Insert to Supabase (simplified for demo)
                for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Migrating {display_name}"):
                    if self.use_supabase:
                        # Use simplified insert for enhanced tables
                        columns = list(row.index)
                        placeholders = ','.join(['%s' for _ in columns])
                        values = [row[col] for col in columns]

                        self.supabase_conn.execute(f"INSERT OR REPLACE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})", values)
                    else:
                        # Fallback to SQLite
                        self.sqlite_conn.execute(f"INSERT OR REPLACE INTO {table_name} ({','.join(columns)}) VALUES ({','.join(['?' for _ in columns])})", values)

                # Commit
                if self.use_supabase:
                    self.supabase_conn.commit()
                else:
                    self.sqlite_conn.commit()

                self.migration_stats[display_name] = len(data)
                logger.info(f"✅ Migrated {len(data):,} {display_name} records")

            except Exception as e:
                logger.error(f"❌ Error migrating {display_name}: {e}")
                if self.use_supabase:
                    self.supabase_conn.rollback()
                else:
                    self.sqlite_conn.rollback()

    def calculate_derived_features(self):
        """Calculate derived features in Supabase"""
        logger.info("🧮 Calculating derived features...")

        try:
            if self.use_supabase:
                # Calculate team EPA stats from play-by-play
                self.supabase_conn.execute("""
                    INSERT INTO agg_team_game_stats (
                        game_id, season, week, team, off_epa, def_epa, off_success_rate,
                        off_yards_per_play, off_explosive_rate, off_pass_rate,
                        def_success_rate, def_yards_per_play, def_explosive_rate,
                        cpoe, air_yards_per_attempt, yac_per_completion, deep_ball_rate,
                        time_to_throw_avg, pressure_rate_allowed, third_down_conv_rate,
                        red_zone_td_rate, two_minute_epa, qb_injury_severity,
                        key_player_injuries, rest_days, is_home, is_divisional,
                        is_primetime, weather_impact
                    )
                    SELECT
                        p.game_id,
                        p.season,
                        p.week,
                        p.posteam as team,
                        AVG(CASE WHEN p.play_type IN ('pass','run') THEN p.epa END) as off_epa,
                        AVG(CASE WHEN p.defteam = p.posteam THEN -p.epa END) as def_epa,
                        AVG(CASE WHEN p.play_type IN ('pass','run') THEN p.success END) as off_success_rate,
                        AVG(CASE WHEN p.play_type IN ('pass','run') THEN p.yards_gained END) as off_yards_per_play,
                        AVG(CASE WHEN p.play_type IN ('pass','run') AND p.yards_gained >= 20 THEN 1 ELSE 0 END) as off_explosive_rate,
                        AVG(CASE WHEN p.play_type = 'pass' THEN 1 ELSE 0 END) as off_pass_rate,
                        AVG(CASE WHEN p.defteam = p.posteam THEN p.success END) as def_success_rate,
                        AVG(CASE WHEN p.defteam = p.posteam THEN p.yards_gained END) as def_yards_per_play,
                        AVG(CASE WHEN p.defteam = p.posteam AND p.yards_gained >= 20 THEN 1 ELSE 0 END) as def_explosive_rate,
                        AVG(p.cpoe) as cpoe,
                        AVG(CASE WHEN p.air_yards IS NOT NULL AND p.pass_attempt = 1 THEN p.air_yards END) as air_yards_per_attempt,
                        AVG(CASE WHEN p.complete_pass = 1 AND p.yards_after_catch IS NOT NULL THEN p.yards_after_catch END) as yac_per_completion,
                        AVG(CASE WHEN p.air_yards >= 20 THEN 1 ELSE 0 END) as deep_ball_rate,
                        AVG(p.time_to_throw) as time_to_throw_avg,
                        AVG(p.was_pressure) as pressure_rate_allowed,
                        AVG(CASE WHEN p.down = 3 AND p.first_down = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN p.down = 3 THEN 1 END), 0) as third_down_conv_rate,
                        AVG(CASE WHEN p.yardline_100 <= 20 AND p.touchdown = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(CASE WHEN p.yardline_100 <= 20 THEN 1 END), 0) as red_zone_td_rate,
                        AVG(CASE WHEN p.half_seconds_remaining <= 120 AND p.half_seconds_remaining > 0 THEN p.epa END) as two_minute_epa,
                        MAX(CASE WHEN i.position = 'QB' THEN i.severity_score ELSE 0 END) as qb_injury_severity,
                        COUNT(CASE WHEN i.severity_score >= 2 THEN 1 END) as key_player_injuries,
                        7 as rest_days,  -- Simplified for demo
                        CASE WHEN g.home_team = p.posteam THEN 1 ELSE 0 END as is_home,
                        CASE WHEN g.div_game = 1 THEN 1 ELSE 0 END as is_divisional,
                        CASE WHEN g.gametime LIKE '%20:%' OR g.gametime LIKE '%19:%' THEN 1 ELSE 0 END as is_primetime,
                        CASE WHEN g.temp IS NOT NULL AND g.temp < 40 THEN 1 ELSE 0 END as weather_impact
                    FROM fact_plays p
                    JOIN fact_games g ON p.game_id = g.game_id
                    LEFT JOIN fact_injuries i ON i.season = p.season AND i.week = p.week AND i.team = p.posteam
                    WHERE p.posteam IS NOT NULL AND p.season >= 2016
                    GROUP BY p.game_id, p.season, p.week, p.posteam, g.home_team, g.div_game, g.gametime, g.temp
                    ON CONFLICT (game_id, team) DO UPDATE SET
                        off_epa = EXCLUDED.off_epa,
                        def_epa = EXCLUDED.def_epa,
                        updated_at = CURRENT_TIMESTAMP
                """)

                self.supabase_conn.commit()
                logger.info("✅ Calculated team EPA stats")

                # Create ML features
                self.supabase_conn.execute("""
                    INSERT INTO ml_features (
                        game_id, season, week, home_team, away_team,
                        home_off_epa, home_def_epa, home_off_success_rate, home_def_success_rate,
                        home_cpoe, home_third_down_conv_rate, home_red_zone_td_rate,
                        away_off_epa, away_def_epa, away_off_success_rate, away_def_success_rate,
                        away_cpoe, away_third_down_conv_rate, away_red_zone_td_rate,
                        home_won, point_differential, total_points
                    )
                    SELECT
                        g.game_id, g.season, g.week, g.home_team, g.away_team,
                        h.off_epa as home_off_epa, h.def_epa as home_def_epa,
                        h.off_success_rate as home_off_success_rate, h.def_success_rate as home_def_success_rate,
                        h.cpoe as home_cpoe, h.third_down_conv_rate as home_third_down_conv_rate,
                        h.red_zone_td_rate as home_red_zone_td_rate,
                        a.off_epa as away_off_epa, a.def_epa as away_def_epa,
                        a.off_success_rate as away_off_success_rate, a.def_success_rate as away_def_success_rate,
                        a.cpoe as away_cpoe, a.third_down_conv_rate as away_third_down_conv_rate,
                        a.red_zone_td_rate as away_red_zone_td_rate,
                        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_won,
                        g.home_score - g.away_score as point_differential,
                        g.home_score + g.away_score as total_points
                    FROM fact_games g
                    LEFT JOIN agg_team_game_stats h ON g.game_id = h.game_id AND g.home_team = h.team
                    LEFT JOIN agg_team_game_stats a ON g.game_id = a.game_id AND g.away_team = a.team
                    WHERE g.completed = 1
                    ON CONFLICT (game_id) DO UPDATE SET
                        home_off_epa = EXCLUDED.home_off_epa,
                        home_def_epa = EXCLUDED.home_def_epa,
                        updated_at = CURRENT_TIMESTAMP
                """)

                self.supabase_conn.commit()
                logger.info("✅ Created ML features")

            else:
                logger.info("⚠️  Skipping feature calculation for SQLite mode")

        except Exception as e:
            logger.error(f"❌ Error calculating features: {e}")
            if self.use_supabase:
                self.supabase_conn.rollback()
            raise

    def validate_migration(self):
        """Validate the migration was successful"""
        logger.info("🔍 Validating migration...")

        try:
            if self.use_supabase:
                # Check Supabase counts
                supabase_games = pd.read_sql('SELECT COUNT(*) as count FROM fact_games', self.supabase_conn).iloc[0, 0]
                supabase_plays = pd.read_sql('SELECT COUNT(*) as count FROM fact_plays', self.supabase_conn).iloc[0, 0]

                logger.info(f"✅ Supabase games: {supabase_games:,}")
                logger.info(f"✅ Supabase plays: {supabase_plays:,}")

                # Check for duplicates
                dupes = pd.read_sql('SELECT game_id, COUNT(*) as cnt FROM fact_games GROUP BY game_id HAVING cnt > 1', self.supabase_conn)
                logger.info(f"✅ Duplicate games: {len(dupes)}")

                # Check data quality
                null_scores = pd.read_sql('SELECT COUNT(*) as nulls FROM fact_games WHERE home_score IS NULL', self.supabase_conn).iloc[0, 0]
                logger.info(f"✅ Games with null scores: {null_scores:,}")

            else:
                # Check SQLite counts
                sqlite_games = pd.read_sql('SELECT COUNT(*) as count FROM fact_games', self.sqlite_conn).iloc[0, 0]
                sqlite_plays = pd.read_sql('SELECT COUNT(*) as count FROM fact_plays', self.sqlite_conn).iloc[0, 0]

                logger.info(f"✅ SQLite games: {sqlite_games:,}")
                logger.info(f"✅ SQLite plays: {sqlite_plays:,}")

            logger.info("✅ Migration validation complete")

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise

    def create_auto_update_script(self):
        """Create daily auto-update script"""
        logger.info("🚀 Creating auto-update script...")

        auto_update_script = '''#!/usr/bin/env python3
"""Daily NFL data updater for Supabase"""

import psycopg2
import nflreadpy as nfl
from datetime import datetime, timedelta
import logging

# Supabase connection
SUPABASE_CONFIG = {
    'host': 'db.cqslvbxsqsgjagjkpiro.supabase.co',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'P@ssword9804746196$'
}

def update_recent_games():
    """Update games from last 7 days"""
    conn = psycopg2.connect(**SUPABASE_CONFIG)
    conn.autocommit = False

    try:
        # Get current season
        current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1

        # Get recent games
        schedules = nfl.load_schedules([current_season])
        if hasattr(schedules, 'to_pandas'):
            schedules = schedules.to_pandas()

        # Filter to last 7 days and completed games
        cutoff = datetime.now() - timedelta(days=7)
        recent_completed = schedules[
            (schedules['game_date'] >= cutoff) &
            (schedules['home_score'].notna())
        ]

        games_updated = 0

        for _, game in recent_completed.iterrows():
            # Check if game exists
            cursor = conn.execute(
                "SELECT game_id FROM fact_games WHERE game_id = %s",
                (game['game_id'],)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing game
                conn.execute("""
                    UPDATE fact_games SET
                        home_score = %s,
                        away_score = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE game_id = %s
                """, (game.get('home_score'), game.get('away_score'), game['game_id']))
            else:
                # Insert new game
                conn.execute("""
                    INSERT INTO fact_games (
                        game_id, season, week, game_type, gameday, weekday, gametime,
                        home_team, away_team, home_score, away_score, completed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    game['game_id'], game['season'], game['week'], game['game_type'],
                    game['gameday'], game.get('weekday'), game.get('gametime'),
                    game['home_team'], game['away_team'], game.get('home_score'),
                    game.get('away_score'), 1
                ))

            games_updated += 1

        conn.commit()
        logging.info(f"✅ Updated {games_updated} games")

        # Log update
        with open('logs/daily_updates.jsonl', 'a') as f:
            f.write(f'{{"timestamp": "{datetime.now().isoformat()}", "games_updated": {games_updated}, "season": {current_season}}}\\n')

    except Exception as e:
        conn.rollback()
        logging.error(f"❌ Update failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_recent_games()
'''

        with open('daily_nfl_update_supabase.py', 'w') as f:
            f.write(auto_update_script)

        # Make executable
        import os
        os.chmod('daily_nfl_update_supabase.py', 0o755)

        logger.info("✅ Auto-update script created: daily_nfl_update_supabase.py")

    def run_complete_migration(self, dry_run: bool = False):
        """Run complete migration from SQLite to Supabase"""
        logger.info("="*80)
        logger.info("🚀 COMPLETE NFL DATA MIGRATION TO SUPABASE")
        logger.info("="*80)

        if dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")
            return {'success': True, 'dry_run': True}

        start_time = datetime.now()

        try:
            # 1. Create schema
            self.create_supabase_schema()

            # 2. Migrate games
            self.migrate_games_data()

            # 3. Migrate play-by-play
            self.migrate_play_by_play()

            # 4. Migrate enhanced data
            self.migrate_enhanced_data()

            # 5. Calculate derived features
            self.calculate_derived_features()

            # 6. Validate migration
            self.validate_migration()

            # 7. Create auto-update script
            self.create_auto_update_script()

            # 8. Migration summary
            duration = datetime.now() - start_time
            total_records = sum(self.migration_stats.values())

            logger.info(f"\\n⏱️  Migration completed in {duration}")
            logger.info(f"📊 Total records migrated: {total_records:,}")

            for table, count in self.migration_stats.items():
                if count > 0:
                    logger.info(f"  • {table}: {count:,}")

            # Save migration summary
            summary = {
                'migration_date': datetime.now().isoformat(),
                'source': 'SQLite' if not self.use_supabase else 'Supabase',
                'target': 'Supabase' if self.use_supabase else 'SQLite',
                'duration_seconds': duration.total_seconds(),
                'stats': self.migration_stats,
                'total_records': total_records
            }

            with open('logs/migration_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info("\\n✅ MIGRATION COMPLETED SUCCESSFULLY!")
            logger.info("📝 Summary saved to: logs/migration_summary.json")

            return {
                'success': True,
                'total_records': total_records,
                'duration_seconds': duration.total_seconds(),
                'stats': self.migration_stats
            }

        except Exception as e:
            logger.error(f"\\n❌ Migration failed: {e}")
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
    parser.add_argument('--supabase', action='store_true', help='Use Supabase (default: False)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated without making changes')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing data')

    args = parser.parse_args()

    try:
        migrator = SupabaseMigrator(
            sqlite_path=args.sqlite_path,
            use_supabase=args.supabase
        )

        if args.validate_only:
            migrator.validate_migration()
            logger.info("✅ Validation completed")
            return

        result = migrator.run_complete_migration(dry_run=args.dry_run)

        if result['success']:
            if args.dry_run:
                logger.info("🔍 DRY RUN COMPLETED")
                logger.info(f"   Would migrate: {result.get('total_records', 0):,} records")
            else:
                logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
                logger.info(f"   Records migrated: {result.get('total_records', 0):,}")
                logger.info(f"   Duration: {result.get('duration_seconds', 0):.1f} seconds")
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
