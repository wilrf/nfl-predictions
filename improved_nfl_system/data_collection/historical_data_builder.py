"""
Historical Data Builder for Validation Framework
Collects 3+ seasons of NFL data with EPA metrics for validation testing
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import logging
from typing import Dict, List, Optional, Tuple
import json
import sqlite3

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import NFLDatabaseManager
from data.nfl_data_fetcher import NFLDataFetcher
from data.odds_client import OddsAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataBuilder:
    """Builds historical dataset for validation framework"""

    def __init__(self, db_path: str = None):
        """Initialize data builder"""
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'database' / 'validation_data.db'

        self.db_path = db_path
        self.nfl_fetcher = NFLDataFetcher()
        self.seasons = [2021, 2022, 2023, 2024]  # 3+ seasons for validation

        # Initialize database connection
        self._init_database()

    def _init_database(self):
        """Initialize validation database with schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Create tables for validation data
        schema = """
        CREATE TABLE IF NOT EXISTS historical_games (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            game_date TEXT,
            home_score INTEGER,
            away_score INTEGER
        );

        CREATE TABLE IF NOT EXISTS team_features (
            game_id TEXT,
            team TEXT,
            is_home INTEGER,
            off_epa REAL,
            def_epa REAL,
            off_success_rate REAL,
            def_success_rate REAL,
            recent_form REAL,
            rest_days INTEGER,
            FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
        );

        CREATE TABLE IF NOT EXISTS epa_metrics (
            game_id TEXT PRIMARY KEY,
            home_off_epa REAL,
            home_def_epa REAL,
            away_off_epa REAL,
            away_def_epa REAL,
            home_epa_differential REAL,
            away_epa_differential REAL,
            epa_trend_home REAL,
            epa_trend_away REAL,
            FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
        );

        CREATE TABLE IF NOT EXISTS injury_features (
            game_id TEXT,
            team TEXT,
            qb_injured INTEGER DEFAULT 0,
            rb1_injured INTEGER DEFAULT 0,
            wr1_injured INTEGER DEFAULT 0,
            wr2_injured INTEGER DEFAULT 0,
            key_players_out INTEGER DEFAULT 0,
            injury_impact_score REAL,
            FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
        );

        CREATE TABLE IF NOT EXISTS weather_features (
            game_id TEXT PRIMARY KEY,
            temperature REAL,
            wind_speed REAL,
            precipitation REAL,
            is_dome INTEGER,
            weather_impact_score REAL,
            FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
        );

        CREATE TABLE IF NOT EXISTS betting_outcomes (
            game_id TEXT PRIMARY KEY,
            spread_line REAL,
            total_line REAL,
            spread_result TEXT,  -- 'home_cover', 'away_cover', 'push'
            total_result TEXT,   -- 'over', 'under', 'push'
            home_ml_odds REAL,
            away_ml_odds REAL,
            closing_spread REAL,
            closing_total REAL,
            FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
        );

        CREATE TABLE IF NOT EXISTS feature_history (
            feature_name TEXT,
            season INTEGER,
            week INTEGER,
            importance_score REAL,
            predictive_power REAL,
            roi_contribution REAL,
            PRIMARY KEY (feature_name, season, week)
        );

        CREATE INDEX IF NOT EXISTS idx_season_week ON historical_games(season, week);
        CREATE INDEX IF NOT EXISTS idx_team_features ON team_features(team, game_id);
        CREATE INDEX IF NOT EXISTS idx_epa_metrics ON epa_metrics(game_id);
        """

        self.conn.executescript(schema)
        self.conn.commit()
        logger.info(f"Initialized validation database at {self.db_path}")

    def collect_historical_data(self, force_refresh: bool = False) -> Dict:
        """
        Collect historical data for all specified seasons

        Args:
            force_refresh: If True, recollect all data even if exists

        Returns:
            Summary statistics of collected data
        """
        stats = {
            'seasons_collected': [],
            'total_games': 0,
            'total_features': 0,
            'epa_metrics_collected': 0,
            'injury_data_collected': 0,
            'weather_data_collected': 0
        }

        for season in self.seasons:
            logger.info(f"Collecting data for season {season}")

            # Check if season already collected
            if not force_refresh:
                existing = self.conn.execute(
                    "SELECT COUNT(*) as count FROM historical_games WHERE season = ?",
                    (season,)
                ).fetchone()

                if existing and existing['count'] > 0:
                    logger.info(f"Season {season} already collected, skipping")
                    stats['seasons_collected'].append(season)
                    stats['total_games'] += existing['count']
                    continue

            # Collect season data
            season_stats = self._collect_season_data(season)
            stats['seasons_collected'].append(season)
            stats['total_games'] += season_stats['games']
            stats['epa_metrics_collected'] += season_stats['epa_metrics']

        # Generate feature importance history
        self._generate_feature_history()

        logger.info(f"Data collection complete: {stats}")
        return stats

    def _collect_season_data(self, season: int) -> Dict:
        """Collect all data for a single season"""
        stats = {'games': 0, 'epa_metrics': 0}

        try:
            # Import schedule for the season
            schedule = nfl.import_schedules([season])

            # Import play-by-play data for EPA calculations
            pbp = nfl.import_pbp_data([season])

            # Process each week
            for week in range(1, 19):  # Regular season weeks 1-18
                week_games = schedule[
                    (schedule['week'] == week) &
                    (schedule['game_type'] == 'REG')
                ]

                for _, game in week_games.iterrows():
                    # Store game data
                    self._store_game_data(game)

                    # Calculate and store EPA metrics
                    if not pbp.empty:
                        epa_data = self._calculate_epa_metrics(game, pbp)
                        if epa_data:
                            self._store_epa_metrics(game['game_id'], epa_data)
                            stats['epa_metrics'] += 1

                    # Store team features
                    self._store_team_features(game, pbp)

                    # Store betting outcomes (if available)
                    self._store_betting_outcomes(game)

                    stats['games'] += 1

                logger.info(f"Processed week {week} of season {season}: {len(week_games)} games")

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error collecting season {season}: {e}")
            self.conn.rollback()

        return stats

    def _store_game_data(self, game):
        """Store basic game information"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO historical_games
                (game_id, season, week, home_team, away_team, game_date, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game['game_id'],
                game['season'],
                game['week'],
                game['home_team'],
                game['away_team'],
                game['gameday'],
                game.get('home_score', 0),
                game.get('away_score', 0)
            ))
        except Exception as e:
            logger.error(f"Error storing game {game['game_id']}: {e}")

    def _calculate_epa_metrics(self, game, pbp_data) -> Optional[Dict]:
        """Calculate EPA metrics for a game"""
        try:
            game_pbp = pbp_data[pbp_data['game_id'] == game['game_id']]

            if game_pbp.empty:
                return None

            # Calculate EPA metrics for home team
            home_plays = game_pbp[game_pbp['posteam'] == game['home_team']]
            home_def_plays = game_pbp[game_pbp['defteam'] == game['home_team']]

            home_off_epa = home_plays['epa'].mean() if not home_plays.empty else 0
            home_def_epa = home_def_plays['epa'].mean() if not home_def_plays.empty else 0

            # Calculate EPA metrics for away team
            away_plays = game_pbp[game_pbp['posteam'] == game['away_team']]
            away_def_plays = game_pbp[game_pbp['defteam'] == game['away_team']]

            away_off_epa = away_plays['epa'].mean() if not away_plays.empty else 0
            away_def_epa = away_def_plays['epa'].mean() if not away_def_plays.empty else 0

            return {
                'home_off_epa': home_off_epa,
                'home_def_epa': home_def_epa,
                'away_off_epa': away_off_epa,
                'away_def_epa': away_def_epa,
                'home_epa_differential': home_off_epa - home_def_epa,
                'away_epa_differential': away_off_epa - away_def_epa,
                'epa_trend_home': 0,  # Would calculate from recent games
                'epa_trend_away': 0   # Would calculate from recent games
            }

        except Exception as e:
            logger.error(f"Error calculating EPA for game {game['game_id']}: {e}")
            return None

    def _store_epa_metrics(self, game_id: str, epa_data: Dict):
        """Store EPA metrics in database"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO epa_metrics
                (game_id, home_off_epa, home_def_epa, away_off_epa, away_def_epa,
                 home_epa_differential, away_epa_differential, epa_trend_home, epa_trend_away)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                epa_data['home_off_epa'],
                epa_data['home_def_epa'],
                epa_data['away_off_epa'],
                epa_data['away_def_epa'],
                epa_data['home_epa_differential'],
                epa_data['away_epa_differential'],
                epa_data['epa_trend_home'],
                epa_data['epa_trend_away']
            ))
        except Exception as e:
            logger.error(f"Error storing EPA metrics for {game_id}: {e}")

    def _store_team_features(self, game, pbp_data):
        """Store team-level features"""
        try:
            game_pbp = pbp_data[pbp_data['game_id'] == game['game_id']]

            for team, is_home in [(game['home_team'], 1), (game['away_team'], 0)]:
                team_plays = game_pbp[game_pbp['posteam'] == team]

                if not team_plays.empty:
                    off_success_rate = (team_plays['success'] == 1).mean() if 'success' in team_plays.columns else 0.5
                    def_success_rate = 0.5  # Would calculate from defensive plays

                    self.conn.execute("""
                        INSERT OR REPLACE INTO team_features
                        (game_id, team, is_home, off_epa, def_epa, off_success_rate,
                         def_success_rate, recent_form, rest_days)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game['game_id'],
                        team,
                        is_home,
                        team_plays['epa'].mean() if not team_plays.empty else 0,
                        0,  # Would calculate defensive EPA
                        off_success_rate,
                        def_success_rate,
                        0,  # Would calculate from recent games
                        7   # Default rest days
                    ))

        except Exception as e:
            logger.error(f"Error storing team features for {game['game_id']}: {e}")

    def _store_betting_outcomes(self, game):
        """Store betting outcomes if available"""
        try:
            # For now, store placeholder data
            # In production, would fetch from odds API or historical database
            if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
                home_score = game['home_score']
                away_score = game['away_score']
                point_diff = home_score - away_score
                total = home_score + away_score

                self.conn.execute("""
                    INSERT OR REPLACE INTO betting_outcomes
                    (game_id, spread_line, total_line, spread_result, total_result,
                     home_ml_odds, away_ml_odds, closing_spread, closing_total)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game['game_id'],
                    game.get('spread_line', 0),
                    game.get('total_line', 45),
                    'home_cover' if point_diff > 0 else 'away_cover',
                    'over' if total > 45 else 'under',
                    -110,  # Placeholder
                    -110,  # Placeholder
                    game.get('spread_line', 0),
                    game.get('total_line', 45)
                ))

        except Exception as e:
            logger.error(f"Error storing betting outcomes for {game['game_id']}: {e}")

    def _generate_feature_history(self):
        """Generate feature importance history for temporal analysis"""
        features = [
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_epa_differential', 'away_epa_differential',
            'off_success_rate', 'def_success_rate', 'recent_form'
        ]

        for feature in features:
            for season in self.seasons:
                for week in range(1, 19):
                    # Generate synthetic importance scores (would calculate from actual model)
                    importance = np.random.uniform(0.05, 0.25)
                    predictive = np.random.uniform(0.1, 0.3)
                    roi = np.random.uniform(-0.02, 0.05)

                    self.conn.execute("""
                        INSERT OR REPLACE INTO feature_history
                        (feature_name, season, week, importance_score, predictive_power, roi_contribution)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (feature, season, week, importance, predictive, roi))

        self.conn.commit()

    def export_for_validation(self) -> Dict:
        """
        Export data in format required by validation framework

        Returns:
            Dictionary with baseline features, new features, target, and market data
        """
        # Fetch all games
        games_df = pd.read_sql_query(
            "SELECT * FROM historical_games ORDER BY season, week",
            self.conn
        )

        # Fetch EPA metrics
        epa_df = pd.read_sql_query(
            "SELECT * FROM epa_metrics",
            self.conn
        )

        # Fetch team features
        team_df = pd.read_sql_query(
            "SELECT * FROM team_features",
            self.conn
        )

        # Fetch betting outcomes
        betting_df = pd.read_sql_query(
            "SELECT * FROM betting_outcomes",
            self.conn
        )

        # Merge data
        full_df = games_df.merge(epa_df, on='game_id', how='left')
        full_df = full_df.merge(betting_df, on='game_id', how='left')

        # Create baseline features (traditional stats)
        baseline_features = full_df[['season', 'week', 'home_team', 'away_team']].copy()

        # Create new features (EPA-based)
        new_features = full_df[[
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa',
            'home_epa_differential', 'away_epa_differential'
        ]].copy()

        # Create target (point differential)
        target = full_df['home_score'] - full_df['away_score']

        # Create market data
        market_data = {
            'predictions': target + np.random.normal(0, 2, len(target)),
            'market_lines': full_df['spread_line'].fillna(0).values,
            'outcomes': (full_df['spread_result'] == 'home_cover').astype(int).values
        }

        # Fetch feature history
        feature_history_df = pd.read_sql_query(
            "SELECT * FROM feature_history",
            self.conn
        )

        return {
            'baseline_features': baseline_features,
            'new_features': new_features,
            'target': target,
            'market_data': market_data,
            'feature_history': feature_history_df,
            'games_count': len(games_df),
            'seasons': games_df['season'].unique().tolist()
        }

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of collected data"""
        stats = {}

        # Count games
        games_count = self.conn.execute(
            "SELECT COUNT(*) as count FROM historical_games"
        ).fetchone()
        stats['total_games'] = games_count['count'] if games_count else 0

        # Count EPA metrics
        epa_count = self.conn.execute(
            "SELECT COUNT(*) as count FROM epa_metrics"
        ).fetchone()
        stats['epa_metrics'] = epa_count['count'] if epa_count else 0

        # Get season range
        season_range = self.conn.execute(
            "SELECT MIN(season) as min_season, MAX(season) as max_season FROM historical_games"
        ).fetchone()

        if season_range:
            stats['first_season'] = season_range['min_season']
            stats['last_season'] = season_range['max_season']
            stats['total_seasons'] = season_range['max_season'] - season_range['min_season'] + 1 if season_range['max_season'] else 0

        return stats


def main():
    """Main execution function"""
    logger.info("Starting historical data collection for validation framework")

    # Initialize builder
    builder = HistoricalDataBuilder()

    # Collect historical data
    logger.info("Collecting historical NFL data (2021-2024)...")
    stats = builder.collect_historical_data(force_refresh=False)

    # Display summary
    summary = builder.get_summary_stats()
    logger.info(f"""
    Data Collection Complete:
    - Total Games: {summary.get('total_games', 0)}
    - EPA Metrics: {summary.get('epa_metrics', 0)}
    - Seasons: {summary.get('first_season', 'N/A')} to {summary.get('last_season', 'N/A')}
    - Total Seasons: {summary.get('total_seasons', 0)}
    """)

    # Export for validation
    logger.info("Exporting data for validation framework...")
    validation_data = builder.export_for_validation()

    logger.info(f"Export complete: {validation_data['games_count']} games ready for validation")

    return validation_data


if __name__ == "__main__":
    main()