#!/usr/bin/env python3
"""
Import 2025 Season Weeks 1-4 (Completed Games) to Training Data
Adds 64 completed games for test set validation
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/import_2025_partial.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Season2025PartialImporter:
    """Import 2025 completed games (weeks 1-4) for test set"""

    def __init__(self):
        self.season = 2025
        self.max_week = 4  # Only import completed games
        self.output_dir = Path('ml_training_data/season_2025')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def import_season(self):
        """Main import workflow"""
        logger.info(f"Starting 2025 partial season import (weeks 1-{self.max_week})...")
        logger.info("=" * 60)

        # Step 1: Fetch schedule (filter completed only)
        logger.info("Step 1: Fetching 2025 schedule...")
        schedule = self.fetch_schedule()
        logger.info(f"  ✓ Fetched {len(schedule)} completed games")

        # Step 2: Fetch play-by-play data
        logger.info("Step 2: Fetching play-by-play data...")
        pbp = self.fetch_pbp()
        logger.info(f"  ✓ Fetched {len(pbp):,} plays")

        # Step 3: Calculate team EPA stats
        logger.info("Step 3: Calculating team EPA statistics...")
        team_stats = self.calculate_team_epa_stats(pbp)
        logger.info(f"  ✓ Calculated stats for {len(team_stats)} team-weeks")

        # Step 4: Generate game features
        logger.info("Step 4: Generating game features...")
        game_features = self.generate_game_features(schedule, team_stats)
        logger.info(f"  ✓ Generated features for {len(game_features)} games")

        # Step 5: Save outputs
        logger.info("Step 5: Saving outputs...")
        self.save_outputs(schedule, team_stats, game_features)
        logger.info(f"  ✓ Saved to {self.output_dir}")

        # Step 6: Validation
        logger.info("Step 6: Validating outputs...")
        self.validate_output(game_features)

        logger.info("=" * 60)
        logger.info(f"✅ 2025 Partial Season Import Complete!")
        logger.info(f"   Games: {len(game_features)}")
        logger.info(f"   Weeks: 1-{self.max_week}")
        logger.info(f"   Output: {self.output_dir}")

        return game_features

    def fetch_schedule(self) -> pd.DataFrame:
        """Fetch 2025 schedule and filter to completed games"""
        schedule = nfl.import_schedules([self.season])

        # Filter to regular season
        schedule = schedule[schedule['game_type'] == 'REG'].copy()

        # Filter to weeks 1-4
        schedule = schedule[schedule['week'] <= self.max_week].copy()

        # Filter to completed games only (have scores)
        schedule = schedule[schedule['home_score'].notna()].copy()

        logger.info(f"  Weeks 1-{self.max_week}: {len(schedule)} completed games")

        # Add required fields
        schedule['game_time'] = pd.to_datetime(schedule['gameday'])
        schedule['season'] = self.season
        schedule['stadium'] = schedule['home_team'] + ' Stadium'
        schedule['is_outdoor'] = 1  # Default to outdoor

        return schedule

    def fetch_pbp(self) -> pd.DataFrame:
        """Fetch 2025 play-by-play data for completed weeks"""
        try:
            pbp = nfl.import_pbp_data([self.season])

            # Filter to regular season
            pbp = pbp[pbp['season_type'] == 'REG'].copy()

            # Filter to weeks 1-4
            pbp = pbp[pbp['week'] <= self.max_week].copy()

            return pbp
        except Exception as e:
            logger.warning(f"  Could not fetch 2025 PBP data: {e}")
            logger.warning(f"  Creating empty DataFrame - features will use 2024 carryover")
            return pd.DataFrame()

    def calculate_team_epa_stats(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling team EPA stats by week"""
        if pbp.empty:
            logger.warning("  No PBP data - will use 2024 end-of-season stats as baseline")
            return self.load_2024_baseline_stats()

        all_stats = []

        # Get unique teams
        teams = pbp['posteam'].dropna().unique()

        for team in teams:
            # Process each week
            for week in range(1, self.max_week + 1):
                # Get plays through this week (cumulative)
                week_pbp = pbp[pbp['week'] <= week]

                # Offensive plays (team as offense)
                team_off = week_pbp[week_pbp['posteam'] == team]

                # Defensive plays (team as defense)
                team_def = week_pbp[week_pbp['defteam'] == team]

                if team_off.empty and team_def.empty:
                    continue

                # Calculate EPA stats
                stats = {
                    'team': team,
                    'season': self.season,
                    'week': week,
                    'off_epa_play': float(team_off['epa'].mean()) if not team_off.empty else 0.0,
                    'def_epa_play': float(team_def['epa'].mean()) if not team_def.empty else 0.0,
                    'off_success_rate': float(team_off['success'].mean()) if not team_off.empty else 0.0,
                    'def_success_rate': float(team_def['success'].mean()) if not team_def.empty else 0.0,
                    'games_played': int(team_off['game_id'].nunique()) if not team_off.empty else 0
                }

                # Red zone stats
                team_rz_off = team_off[team_off['yardline_100'] <= 20]
                if not team_rz_off.empty:
                    rz_tds = team_rz_off[team_rz_off['touchdown'] == 1].groupby('game_id').size()
                    rz_drives = team_rz_off.groupby('game_id').size()
                    stats['redzone_td_pct'] = float(len(rz_tds) / len(rz_drives)) if len(rz_drives) > 0 else 0.0
                else:
                    stats['redzone_td_pct'] = 0.0

                # Third down stats
                team_3rd = team_off[team_off['down'] == 3]
                if not team_3rd.empty:
                    third_conversions = team_3rd[team_3rd['first_down'] == 1]
                    stats['third_down_pct'] = float(len(third_conversions) / len(team_3rd))
                else:
                    stats['third_down_pct'] = 0.0

                all_stats.append(stats)

        return pd.DataFrame(all_stats)

    def load_2024_baseline_stats(self) -> pd.DataFrame:
        """Load 2024 end-of-season stats as baseline for 2025 week 1"""
        stats_path = Path('ml_training_data/season_2024/team_epa_stats.csv')

        if not stats_path.exists():
            logger.error(f"  Cannot find 2024 stats at {stats_path}")
            return pd.DataFrame()

        stats_2024 = pd.read_csv(stats_path)

        # Get week 18 stats (end of season)
        baseline = stats_2024[stats_2024['week'] == 18].copy()

        # Convert to 2025 week 0 (for use as prior stats in week 1)
        baseline['season'] = 2025
        baseline['week'] = 0

        logger.info(f"  Loaded 2024 baseline stats for {len(baseline)} teams")
        return baseline

    def generate_game_features(self, schedule: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Generate game-level features matching existing format"""
        game_features = []

        # Load 2024 baseline for week 1 games
        if team_stats.empty or team_stats['week'].min() > 0:
            baseline_stats = self.load_2024_baseline_stats()
            team_stats = pd.concat([baseline_stats, team_stats], ignore_index=True)

        for _, game in schedule.iterrows():
            game_id = game['game_id']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']

            # Get team stats leading into this game (prior week)
            prior_week = week - 1

            home_stats = team_stats[
                (team_stats['team'] == home_team) &
                (team_stats['week'] == prior_week)
            ]

            away_stats = team_stats[
                (team_stats['team'] == away_team) &
                (team_stats['week'] == prior_week)
            ]

            # Fallback to default if no stats
            if home_stats.empty:
                home_stats = pd.DataFrame([{
                    'off_epa_play': 0.0, 'def_epa_play': 0.0,
                    'off_success_rate': 0.0, 'def_success_rate': 0.0,
                    'redzone_td_pct': 0.0, 'third_down_pct': 0.0,
                    'games_played': 0
                }])

            if away_stats.empty:
                away_stats = pd.DataFrame([{
                    'off_epa_play': 0.0, 'def_epa_play': 0.0,
                    'off_success_rate': 0.0, 'def_success_rate': 0.0,
                    'redzone_td_pct': 0.0, 'third_down_pct': 0.0,
                    'games_played': 0
                }])

            home_stats = home_stats.iloc[0]
            away_stats = away_stats.iloc[0]

            # Calculate EPA differential
            epa_diff = (home_stats['off_epa_play'] - home_stats['def_epa_play']) - \
                       (away_stats['off_epa_play'] - away_stats['def_epa_play'])

            # Create feature row
            features = {
                # Metadata
                'game_id': game_id,
                'season': self.season,
                'week': week,
                'home_team': home_team,
                'away_team': away_team,
                'game_time': game['game_time'],
                'home_score': game.get('home_score'),
                'away_score': game.get('away_score'),

                # Derived targets
                'point_differential': game['home_score'] - game['away_score'] if pd.notna(game['home_score']) else None,
                'total_points': game['home_score'] + game['away_score'] if pd.notna(game['home_score']) else None,
                'home_won': 1 if game['home_score'] > game['away_score'] else 0,

                # Features
                'is_home': 1,
                'week_number': week,
                'is_divisional': 1 if game.get('div_game', False) else 0,
                'home_off_epa': home_stats['off_epa_play'],
                'home_def_epa': home_stats['def_epa_play'],
                'away_off_epa': away_stats['off_epa_play'],
                'away_def_epa': away_stats['def_epa_play'],
                'epa_differential': epa_diff,
                'home_off_success_rate': home_stats['off_success_rate'],
                'away_off_success_rate': away_stats['off_success_rate'],
                'home_redzone_td_pct': home_stats['redzone_td_pct'],
                'away_redzone_td_pct': away_stats['redzone_td_pct'],
                'home_third_down_pct': home_stats['third_down_pct'],
                'away_third_down_pct': away_stats['third_down_pct'],
                'home_games_played': home_stats['games_played'],
                'away_games_played': away_stats['games_played'],
                'stadium': game['stadium'],
                'is_outdoor': game['is_outdoor']
            }

            game_features.append(features)

        df = pd.DataFrame(game_features)

        # Ensure column order matches existing data
        column_order = [
            'game_id', 'season', 'week', 'home_team', 'away_team', 'game_time',
            'home_score', 'away_score', 'point_differential', 'total_points', 'home_won',
            'is_home', 'week_number', 'is_divisional',
            'home_off_epa', 'home_def_epa', 'away_off_epa', 'away_def_epa', 'epa_differential',
            'home_off_success_rate', 'away_off_success_rate',
            'home_redzone_td_pct', 'away_redzone_td_pct',
            'home_third_down_pct', 'away_third_down_pct',
            'home_games_played', 'away_games_played',
            'stadium', 'is_outdoor'
        ]

        return df[column_order]

    def save_outputs(self, schedule: pd.DataFrame, team_stats: pd.DataFrame, game_features: pd.DataFrame):
        """Save all outputs to disk"""
        schedule.to_csv(self.output_dir / 'games.csv', index=False)
        team_stats.to_csv(self.output_dir / 'team_epa_stats.csv', index=False)
        game_features.to_csv(self.output_dir / 'game_features.csv', index=False)

        logger.info(f"  Saved games.csv: {len(schedule)} games")
        logger.info(f"  Saved team_epa_stats.csv: {len(team_stats)} records")
        logger.info(f"  Saved game_features.csv: {len(game_features)} games")

    def validate_output(self, game_features: pd.DataFrame):
        """Validate output matches expected format"""
        if len(game_features.columns) != 29:
            logger.warning(f"⚠️  Column count: {len(game_features.columns)} (expected 29)")
        else:
            logger.info(f"  ✓ Column count: 29")

        required_cols = ['game_id', 'epa_differential', 'home_won']
        missing = [c for c in required_cols if c not in game_features.columns]
        if missing:
            logger.error(f"  ✗ Missing columns: {missing}")
        else:
            logger.info(f"  ✓ All required columns present")

        completed = game_features['home_score'].notna().sum()
        logger.info(f"  ✓ Completed games: {completed}/{len(game_features)}")
        logger.info(f"  ✓ Validation complete")


if __name__ == "__main__":
    importer = Season2025PartialImporter()
    game_features = importer.import_season()

    print("\n" + "=" * 60)
    print("2025 Partial Season Import Summary:")
    print("=" * 60)
    print(f"Total games: {len(game_features)}")
    print(f"Completed games: {game_features['home_score'].notna().sum()}")
    print(f"Weeks imported: 1-{importer.max_week}")
    print(f"Output directory: {importer.output_dir}")
    print("=" * 60)
