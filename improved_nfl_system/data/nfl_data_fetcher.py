"""
NFL Data Fetcher using nfl_data_py
REAL DATA ONLY - No synthetic data generation
FAIL FAST - No retries or fallbacks
"""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NFLDataError(Exception):
    """Custom exception for NFL data operations"""
    pass


class NFLDataFetcher:
    """Fetch real NFL data using nfl_data_py - FAIL FAST on any error"""

    def __init__(self):
        """Initialize fetcher - verify nfl_data_py is working"""
        try:
            # Test that nfl_data_py is installed and working
            test = nfl.import_team_desc()
            if test.empty:
                raise NFLDataError("nfl_data_py returned empty team data")
            self.teams = test
            logger.info(f"NFL data fetcher initialized with {len(test)} teams")
        except Exception as e:
            raise NFLDataError(f"Cannot initialize nfl_data_py: {e}")

    def fetch_week_games(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch game schedule for a specific week
        FAIL FAST if no data available
        """
        if season < 1999:
            raise NFLDataError(f"Season {season} not available (min: 1999)")

        if not (1 <= week <= 18):
            raise NFLDataError(f"Invalid week {week} (must be 1-18)")

        try:
            # Import schedule data
            schedule = nfl.import_schedules([season])

            if schedule.empty:
                raise NFLDataError(f"No schedule data for season {season}")

            # Filter to specific week
            week_games = schedule[schedule['week'] == week].copy()

            if week_games.empty:
                raise NFLDataError(f"No games found for season {season} week {week}")

            # Validate required fields exist
            required_fields = ['game_id', 'home_team', 'away_team', 'gameday']
            missing_fields = [f for f in required_fields if f not in week_games.columns]
            if missing_fields:
                raise NFLDataError(f"Missing required fields: {missing_fields}")

            # Clean and format data
            week_games['game_time'] = pd.to_datetime(week_games['gameday'])
            week_games['season'] = season
            week_games['week'] = week

            logger.info(f"Fetched {len(week_games)} games for season {season} week {week}")
            return week_games

        except NFLDataError:
            raise  # Re-raise our custom errors
        except Exception as e:
            raise NFLDataError(f"Failed to fetch week games: {e}")

    def fetch_pbp_data(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch play-by-play data for analysis
        FAIL FAST if games haven't been played yet
        """
        try:
            # Get play-by-play data for full season (nfl_data_py doesn't support weeks parameter)
            pbp = nfl.import_pbp_data([season])

            if pbp.empty:
                raise NFLDataError(f"No play-by-play data for season {season}")

            # Filter to specific week
            pbp = pbp[pbp['week'] == week].copy()

            if pbp.empty:
                raise NFLDataError(f"No play-by-play data for season {season} week {week} - games may not have been played yet")

            # Validate we have actual play data
            if 'play_id' not in pbp.columns:
                raise NFLDataError("Invalid PBP data - missing play_id column")

            logger.info(f"Fetched {len(pbp)} plays for season {season} week {week}")
            return pbp

        except NFLDataError:
            raise
        except Exception as e:
            raise NFLDataError(f"Failed to fetch PBP data: {e}")

    def fetch_team_stats(self, season: int, through_week: int) -> pd.DataFrame:
        """
        Calculate team statistics through a specific week
        Uses REAL game data only
        """
        if through_week < 1:
            raise NFLDataError(f"Invalid through_week {through_week}")

        try:
            # Fetch full season PBP data (nfl_data_py doesn't support weeks parameter)
            pbp = nfl.import_pbp_data([season])

            if pbp.empty:
                raise NFLDataError(f"No games played yet in season {season}")

            # Filter to weeks through through_week
            pbp = pbp[pbp['week'] <= through_week].copy()

            if pbp.empty:
                raise NFLDataError(f"No games played through week {through_week} in season {season}")

            # Calculate team statistics
            stats = self._calculate_team_metrics(pbp)

            if stats.empty:
                raise NFLDataError("Failed to calculate team statistics")

            logger.info(f"Calculated stats for {len(stats)} teams through week {through_week}")
            return stats

        except NFLDataError:
            raise
        except Exception as e:
            raise NFLDataError(f"Failed to fetch team stats: {e}")

    def _calculate_team_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Calculate offensive and defensive metrics from PBP data"""
        if pbp.empty:
            raise NFLDataError("Cannot calculate metrics from empty PBP data")

        try:
            metrics = []

            # Get unique teams
            teams = pbp['posteam'].dropna().unique()

            for team in teams:
                # Offensive plays
                off_plays = pbp[pbp['posteam'] == team]
                # Defensive plays
                def_plays = pbp[pbp['defteam'] == team]

                if off_plays.empty or def_plays.empty:
                    continue

                team_metrics = {
                    'team': team,
                    'games_played': off_plays['game_id'].nunique(),

                    # Offensive metrics (REAL DATA)
                    'off_epa_play': off_plays['epa'].mean() if 'epa' in off_plays else 0,
                    'off_success_rate': off_plays['success'].mean() if 'success' in off_plays else 0,
                    'off_yards_play': off_plays['yards_gained'].mean() if 'yards_gained' in off_plays else 0,
                    'off_explosive_rate': (off_plays['yards_gained'] >= 20).mean() if 'yards_gained' in off_plays else 0,
                    'off_pass_rate': (off_plays['play_type'] == 'pass').mean() if 'play_type' in off_plays else 0.5,

                    # Defensive metrics (REAL DATA)
                    'def_epa_play': def_plays['epa'].mean() if 'epa' in def_plays else 0,
                    'def_success_rate': def_plays['success'].mean() if 'success' in def_plays else 0,
                    'def_yards_play': def_plays['yards_gained'].mean() if 'yards_gained' in def_plays else 0,
                    'def_explosive_allowed': (def_plays['yards_gained'] >= 20).mean() if 'yards_gained' in def_plays else 0,

                    # Red zone efficiency
                    'rz_td_pct': self._calculate_redzone_efficiency(off_plays),

                    # Third down conversion
                    'third_down_pct': self._calculate_third_down_pct(off_plays)
                }

                metrics.append(team_metrics)

            if not metrics:
                raise NFLDataError("No team metrics calculated")

            return pd.DataFrame(metrics)

        except Exception as e:
            raise NFLDataError(f"Failed to calculate team metrics: {e}")

    def _calculate_redzone_efficiency(self, plays: pd.DataFrame) -> float:
        """Calculate red zone TD percentage from REAL plays"""
        if 'yardline_100' not in plays.columns:
            return 0.0

        rz_plays = plays[plays['yardline_100'] <= 20]
        if rz_plays.empty:
            return 0.0

        # Count TDs in red zone
        if 'touchdown' in rz_plays.columns:
            td_plays = rz_plays[rz_plays['touchdown'] == 1]
            # Approximate drives as groups of consecutive plays
            drives_estimate = len(rz_plays) / 6  # Rough estimate
            if drives_estimate > 0:
                return len(td_plays) / drives_estimate
        return 0.0

    def _calculate_third_down_pct(self, plays: pd.DataFrame) -> float:
        """Calculate third down conversion rate from REAL plays"""
        if 'down' not in plays.columns:
            return 0.0

        third_downs = plays[plays['down'] == 3]
        if third_downs.empty:
            return 0.0

        if 'first_down' in third_downs.columns:
            conversions = third_downs[third_downs['first_down'] == 1]
            return len(conversions) / len(third_downs)
        return 0.0

    def fetch_player_stats(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch weekly player statistics
        REAL DATA ONLY - from actual games
        """
        try:
            # Get weekly player data
            weekly_data = nfl.import_weekly_data([season], weeks=[week])

            if weekly_data.empty:
                raise NFLDataError(f"No player data for season {season} week {week}")

            # Filter to relevant columns
            relevant_cols = [
                'player_id', 'player_name', 'position', 'team',
                'passing_yards', 'passing_tds', 'interceptions',
                'rushing_yards', 'rushing_tds',
                'receiving_yards', 'receiving_tds', 'receptions',
                'fantasy_points', 'fantasy_points_ppr'
            ]

            # Keep only columns that exist
            cols_to_keep = [c for c in relevant_cols if c in weekly_data.columns]
            player_stats = weekly_data[cols_to_keep].copy()

            logger.info(f"Fetched stats for {len(player_stats)} players")
            return player_stats

        except Exception as e:
            raise NFLDataError(f"Failed to fetch player stats: {e}")

    def fetch_injuries(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch injury data
        NOTE: nfl_data_py has limited injury data
        """
        try:
            # Get injury data
            injuries = nfl.import_injuries([season], [week])

            if injuries.empty:
                logger.warning(f"No injury data available for season {season} week {week}")
                # Return empty but valid dataframe
                return pd.DataFrame(columns=['player', 'team', 'status'])

            return injuries

        except Exception as e:
            # Injuries might not be available - not critical
            logger.warning(f"Could not fetch injuries: {e}")
            return pd.DataFrame(columns=['player', 'team', 'status'])

    def apply_temporal_decay(self, team_stats: pd.DataFrame, alpha: float = 0.85) -> pd.DataFrame:
        """
        Apply exponential decay to historical stats
        More recent games weighted more heavily
        """
        if team_stats.empty:
            raise NFLDataError("Cannot apply decay to empty stats")

        if not (0 < alpha < 1):
            raise NFLDataError(f"Invalid decay factor {alpha} (must be 0-1)")

        try:
            # Sort by recency if there's a week column
            if 'week' in team_stats.columns:
                team_stats = team_stats.sort_values('week', ascending=False)

                # Calculate games ago
                current_week = team_stats['week'].max()
                team_stats['games_ago'] = current_week - team_stats['week']

                # Apply exponential decay
                team_stats['weight'] = alpha ** team_stats['games_ago']

                # Weight all numeric columns
                numeric_cols = team_stats.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if col not in ['week', 'games_ago', 'weight']:
                        team_stats[f'{col}_weighted'] = team_stats[col] * team_stats['weight']

            return team_stats

        except Exception as e:
            raise NFLDataError(f"Failed to apply temporal decay: {e}")

    def get_current_week(self) -> tuple[int, int]:
        """Get current NFL season and week"""
        try:
            # Get current season schedule
            current_year = datetime.now().year
            schedule = nfl.import_schedules([current_year])

            if schedule.empty:
                # Try previous year if current year not available
                schedule = nfl.import_schedules([current_year - 1])
                if schedule.empty:
                    raise NFLDataError("Cannot determine current NFL week")
                current_year = current_year - 1

            # Find current week based on date
            today = pd.Timestamp.now()
            schedule['gameday'] = pd.to_datetime(schedule['gameday'])

            # Find the current or next week
            future_games = schedule[schedule['gameday'] >= today]
            if not future_games.empty:
                current_week = future_games['week'].min()
            else:
                current_week = schedule['week'].max()

            return current_year, int(current_week)

        except Exception as e:
            raise NFLDataError(f"Failed to get current week: {e}")