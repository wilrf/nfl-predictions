"""
Comprehensive NFL Data Loader
Loads ALL nfl_data_py data efficiently for betting system
"""

import nfl_data_py as nfl
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ComprehensiveNFLDataLoader:
    """Load all NFL data intelligently - aggregate where needed, store raw where valuable"""

    def __init__(self, db_path: str = 'database/nfl_betting.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def load_all_data(self, seasons: List[int] = [2020, 2021, 2022, 2023, 2024]):
        """Master loading function - call this to get everything"""

        logger.info(f"Starting comprehensive data load for seasons: {seasons}")

        # Priority 1: Core betting data
        self.load_games_and_schedules(seasons)
        self.load_stadiums()
        self.load_weather_data(seasons)
        self.load_officials_data(seasons)

        # Priority 2: Performance data (aggregated)
        self.load_play_by_play_aggregated(seasons)
        self.load_team_stats(seasons)

        # Priority 3: Player data (key players only)
        self.load_players_and_rosters(seasons)
        self.load_injuries(seasons)
        self.load_ngs_stats(seasons)

        # Priority 4: Betting-specific
        self.calculate_referee_tendencies()
        self.calculate_team_tendencies(seasons)
        self.build_matchup_history()

        logger.info("Comprehensive data load complete!")

    # ============================================
    # PRIORITY 1: CORE BETTING DATA
    # ============================================

    def load_games_and_schedules(self, seasons: List[int]):
        """Load game schedules with results"""
        logger.info("Loading games and schedules...")

        # Get schedules for all seasons
        schedules = nfl.import_schedules(seasons)

        # Clean and prepare data
        schedules['game_id'] = schedules['game_id'].astype(str)
        schedules['completed'] = schedules['home_score'].notna().astype(int)

        # Map to our schema
        games_df = schedules[[
            'game_id', 'season', 'week', 'game_type',
            'gameday', 'gametime', 'away_team', 'home_team',
            'away_score', 'home_score', 'location', 'result', 'total',
            'overtime', 'div_game', 'roof', 'surface', 'temp', 'wind',
            'away_rest', 'home_rest', 'away_coach', 'home_coach',
            'referee', 'stadium_id', 'stadium'
        ]].copy()

        # Save to database
        games_df.to_sql('games', self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded {len(games_df)} games")

    def load_stadiums(self):
        """Load stadium information"""
        logger.info("Loading stadium data...")

        # Get team descriptions which include stadium info
        teams = nfl.import_team_desc()

        stadiums_df = teams[[
            'team_abbr', 'team_name', 'team_conf', 'team_division'
        ]].copy()

        # Add stadium metadata (you might need to enhance this from other sources)
        stadiums_df['stadium_type'] = 'outdoor'  # Default, update based on known domes

        # Known domes/retractable roofs (2024)
        dome_teams = ['ARI', 'ATL', 'DAL', 'DET', 'HOU', 'IND', 'LA', 'LV', 'MIN', 'NO']
        stadiums_df.loc[stadiums_df['team_abbr'].isin(dome_teams), 'stadium_type'] = 'dome'

        stadiums_df.to_sql('teams_info', self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded {len(stadiums_df)} team/stadium records")

    def load_weather_data(self, seasons: List[int]):
        """Extract weather from play-by-play data"""
        logger.info("Loading weather data...")

        weather_data = []

        for season in seasons:
            # Load PBP data (has weather strings)
            pbp = nfl.import_pbp_data([season], columns=['game_id', 'weather', 'temp', 'wind'])

            # Get unique games with weather
            game_weather = pbp.groupby('game_id').first().reset_index()

            # Parse weather strings
            for _, row in game_weather.iterrows():
                weather_dict = self._parse_weather_string(row.get('weather', ''))
                weather_dict['game_id'] = row['game_id']
                weather_dict['temperature'] = row.get('temp')
                weather_dict['wind_speed'] = row.get('wind')
                weather_data.append(weather_dict)

        weather_df = pd.DataFrame(weather_data)
        weather_df.to_sql('game_weather', self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded weather for {len(weather_df)} games")

    def _parse_weather_string(self, weather_str: str) -> Dict:
        """Parse weather string into structured data"""
        result = {
            'weather_condition': 'Unknown',
            'weather_detail': weather_str,
            'humidity': None,
            'precipitation': None
        }

        if not weather_str:
            return result

        weather_lower = weather_str.lower()

        # Determine condition
        if 'dome' in weather_lower or 'closed' in weather_lower:
            result['weather_condition'] = 'Dome'
        elif 'rain' in weather_lower:
            result['weather_condition'] = 'Rain'
        elif 'snow' in weather_lower:
            result['weather_condition'] = 'Snow'
        elif 'clear' in weather_lower or 'sunny' in weather_lower:
            result['weather_condition'] = 'Clear'
        elif 'cloud' in weather_lower:
            result['weather_condition'] = 'Cloudy'
        elif 'fog' in weather_lower:
            result['weather_condition'] = 'Fog'

        # Extract humidity
        humidity_match = re.search(r'(\d+)%', weather_str)
        if humidity_match:
            result['humidity'] = int(humidity_match.group(1))

        return result

    def load_officials_data(self, seasons: List[int]):
        """Load officials and game assignments"""
        logger.info("Loading officials data...")

        officials_data = []
        game_officials_data = []

        for season in seasons:
            # Officials data is in the schedules
            schedules = nfl.import_schedules([season])

            for _, game in schedules.iterrows():
                if pd.notna(game.get('referee')):
                    # Add to officials table
                    officials_data.append({
                        'official_name': game['referee'],
                        'position': 'R',  # Referee
                        'seasons_experience': None  # Would need additional source
                    })

                    # Add to game assignments
                    game_officials_data.append({
                        'game_id': game['game_id'],
                        'official_id': game['referee'].replace(' ', '_').lower(),
                        'position': 'R',
                        'is_head_referee': 1
                    })

        # Remove duplicates and save
        officials_df = pd.DataFrame(officials_data).drop_duplicates(subset=['official_name'])
        officials_df['official_id'] = officials_df['official_name'].str.replace(' ', '_').str.lower()

        officials_df.to_sql('officials', self.conn, if_exists='replace', index=False)

        game_officials_df = pd.DataFrame(game_officials_data)
        game_officials_df.to_sql('game_officials', self.conn, if_exists='replace', index=False)

        logger.info(f"Loaded {len(officials_df)} officials with {len(game_officials_df)} assignments")

    # ============================================
    # PRIORITY 2: AGGREGATED PERFORMANCE DATA
    # ============================================

    def load_play_by_play_aggregated(self, seasons: List[int]):
        """Load play-by-play but AGGREGATE to game level"""
        logger.info("Loading and aggregating play-by-play data...")

        game_stats = []

        for season in seasons:
            logger.info(f"Processing season {season}...")

            # Load PBP in chunks to manage memory
            pbp = nfl.import_pbp_data([season], columns=[
                'game_id', 'posteam', 'defteam', 'play_type',
                'epa', 'success', 'yards_gained', 'air_yards',
                'qtr', 'down', 'ydstogo', 'yardline_100',
                'score_differential', 'wp', 'cpoe'
            ])

            # Aggregate by game and team
            for game_id in pbp['game_id'].unique():
                game_plays = pbp[pbp['game_id'] == game_id]

                # Get teams
                home_team = game_plays['home_team'].iloc[0] if 'home_team' in game_plays else None
                away_team = game_plays['away_team'].iloc[0] if 'away_team' in game_plays else None

                if not home_team or not away_team:
                    continue

                # Calculate aggregated stats
                home_plays = game_plays[game_plays['posteam'] == home_team]
                away_plays = game_plays[game_plays['posteam'] == away_team]

                stats = {
                    'game_id': game_id,
                    'home_epa_per_play': home_plays['epa'].mean() if len(home_plays) > 0 else None,
                    'away_epa_per_play': away_plays['epa'].mean() if len(away_plays) > 0 else None,
                    'home_success_rate': home_plays['success'].mean() if len(home_plays) > 0 else None,
                    'away_success_rate': away_plays['success'].mean() if len(away_plays) > 0 else None,
                    'total_plays': len(game_plays),
                    # Add more aggregated metrics as needed
                }

                game_stats.append(stats)

        # Save aggregated stats
        stats_df = pd.DataFrame(game_stats)
        stats_df.to_sql('game_stats_advanced', self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded aggregated stats for {len(stats_df)} games")

    def load_team_stats(self, seasons: List[int]):
        """Load seasonal team stats"""
        logger.info("Loading team stats...")

        # Use nfl_data_py's seasonal data
        seasonal = nfl.import_seasonal_data(seasons)

        # Filter to team-level stats
        team_stats = seasonal.groupby(['season', 'recent_team']).agg({
            'completions': 'sum',
            'attempts': 'sum',
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'receptions': 'sum',
            'targets': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum'
        }).reset_index()

        team_stats.to_sql('team_season_stats', self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded stats for {len(team_stats)} team-seasons")

    # ============================================
    # PRIORITY 3: KEY PLAYER DATA
    # ============================================

    def load_players_and_rosters(self, seasons: List[int]):
        """Load player data - focus on skill positions"""
        logger.info("Loading player and roster data...")

        # Get rosters
        rosters = nfl.import_rosters(seasons)

        # Filter to key positions for betting
        key_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        key_players = rosters[rosters['position'].isin(key_positions)]

        # Create simplified player table
        players_df = key_players[[
            'player_id', 'player_name', 'position', 'team',
            'jersey_number', 'height', 'weight', 'years_exp', 'status'
        ]].drop_duplicates(subset=['player_id'])

        # Convert height to inches
        players_df['height_inches'] = players_df['height'].apply(self._height_to_inches)

        players_df.to_sql('players', self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded {len(players_df)} key players")

    def _height_to_inches(self, height_str):
        """Convert height string (6-2) to inches"""
        if pd.isna(height_str) or '-' not in height_str:
            return None
        try:
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return None

    def load_injuries(self, seasons: List[int]):
        """Load injury data"""
        logger.info("Loading injury data...")

        all_injuries = []

        for season in seasons:
            # Get injuries for each week
            for week in range(1, 19):  # Regular season
                try:
                    injuries = nfl.import_injuries([season], [week])
                    injuries['season'] = season
                    injuries['week'] = week
                    all_injuries.append(injuries)
                except:
                    continue

        if all_injuries:
            injuries_df = pd.concat(all_injuries, ignore_index=True)

            # Map to our schema
            injuries_clean = injuries_df[[
                'player_id', 'season', 'week', 'team',
                'report_status', 'report_primary_injury'
            ]].copy()

            injuries_clean.columns = [
                'player_id', 'season', 'week', 'team_code',
                'injury_status', 'injury_area'
            ]

            injuries_clean.to_sql('injuries', self.conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(injuries_clean)} injury records")

    def load_ngs_stats(self, seasons: List[int]):
        """Load Next Gen Stats - aggregated"""
        logger.info("Loading NGS data...")

        # Load NGS passing
        ngs_passing = nfl.import_ngs_data('passing', seasons)

        # Aggregate to game level for key metrics
        ngs_game = ngs_passing.groupby(['player_gsis_id', 'week', 'season']).agg({
            'avg_time_to_throw': 'mean',
            'aggressiveness': 'mean',
            'completion_percentage_above_expectation': 'mean'
        }).reset_index()

        # Note: Would need to add game_id mapping here

        logger.info(f"Loaded NGS data for {len(ngs_game)} player-games")

    # ============================================
    # PRIORITY 4: BETTING CALCULATIONS
    # ============================================

    def calculate_referee_tendencies(self):
        """Calculate referee over/under tendencies"""
        logger.info("Calculating referee tendencies...")

        query = """
        WITH referee_games AS (
            SELECT
                go.official_id,
                g.game_id,
                g.total,
                g.total_line,
                CASE WHEN g.total > g.total_line THEN 1 ELSE 0 END as went_over
            FROM game_officials go
            JOIN games g ON go.game_id = g.game_id
            WHERE go.is_head_referee = 1
                AND g.completed = 1
                AND g.total IS NOT NULL
                AND g.total_line IS NOT NULL
        )
        INSERT OR REPLACE INTO referee_tendencies
        SELECT
            official_id,
            COUNT(*) as games_count,
            AVG(total) as avg_total_points,
            NULL as avg_penalties,
            NULL as avg_penalty_yards,
            NULL as home_win_pct,
            AVG(went_over) as over_pct,
            DATE('now') as last_updated
        FROM referee_games
        GROUP BY official_id
        HAVING COUNT(*) >= 10  -- Minimum games for significance
        """

        self.conn.execute(query)
        self.conn.commit()
        logger.info("Calculated referee tendencies")

    def calculate_team_tendencies(self, seasons: List[int]):
        """Calculate team betting tendencies"""
        logger.info("Calculating team tendencies...")

        for season in seasons:
            query = f"""
            INSERT OR REPLACE INTO team_tendencies
            SELECT
                home_team as team_code,
                {season} as season,
                COUNT(*) as games_played,
                AVG(home_score) as avg_points_scored,
                AVG(away_score) as avg_points_allowed,
                NULL as avg_plays_per_game,
                NULL as avg_seconds_per_play,
                AVG(CASE WHEN qtr <= 2 THEN home_score ELSE 0 END) as avg_first_half_points,
                AVG(CASE WHEN qtr > 2 THEN home_score ELSE 0 END) as avg_second_half_points,
                NULL as redzone_td_pct,
                NULL as ats_record_home,
                NULL as ats_record_away,
                NULL as over_under_record
            FROM games
            WHERE season = {season} AND completed = 1
            GROUP BY home_team
            """

            self.conn.execute(query)

        self.conn.commit()
        logger.info("Calculated team tendencies")

    def build_matchup_history(self):
        """Build historical matchup data"""
        logger.info("Building matchup history...")

        query = """
        INSERT OR REPLACE INTO matchup_history
        SELECT
            CASE WHEN home_team < away_team THEN home_team ELSE away_team END as team1,
            CASE WHEN home_team < away_team THEN away_team ELSE home_team END as team2,
            COUNT(*) as games_played,
            SUM(CASE WHEN home_score > away_score AND home_team = team1 THEN 1
                     WHEN away_score > home_score AND away_team = team1 THEN 1
                     ELSE 0 END) as team1_wins,
            SUM(CASE WHEN home_score > away_score AND home_team = team2 THEN 1
                     WHEN away_score > home_score AND away_team = team2 THEN 1
                     ELSE 0 END) as team2_wins,
            AVG(total) as avg_total_points,
            AVG(ABS(home_score - away_score)) as avg_point_differential,
            MAX(gameday) as last_meeting_date
        FROM games
        WHERE completed = 1
        GROUP BY team1, team2
        HAVING COUNT(*) >= 3  -- Minimum games for relevance
        """

        self.conn.execute(query)
        self.conn.commit()
        logger.info("Built matchup history")


if __name__ == "__main__":
    # Example usage
    loader = ComprehensiveNFLDataLoader()

    # Load everything
    loader.load_all_data([2022, 2023, 2024])

    print("Data loading complete!")