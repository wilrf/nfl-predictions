"""
NFL.com Official Data Client
Access to NFL's public JSON endpoints for official statistics and live data
No authentication required - these are public feeds
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
import time
import json
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class NFLOfficialClient:
    """Client for NFL.com's official public data feeds"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 2  # Be respectful to NFL.com

    def get_live_scores(self, season: int = None, week: int = None, season_type: str = 'REG') -> pd.DataFrame:
        """Get live scores from NFL.com scorestrip"""
        current_season = season or datetime.now().year
        current_week = week or self._get_current_week()

        # NFL.com scorestrip XML endpoint
        url = "http://www.nfl.com/ajax/scorestrip"
        params = {
            'season': current_season,
            'seasonType': season_type,  # REG, PRE, POST
            'week': current_week
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            games = []

            for game in root.findall('.//g'):
                game_data = {
                    'nfl_game_id': game.get('eid'),
                    'game_date': game.get('t'),
                    'game_time': game.get('q'),  # Quarter info
                    'game_clock': game.get('k', ''),  # Game clock
                    'away_team': game.get('v'),
                    'home_team': game.get('h'),
                    'away_score': int(game.get('vs', 0)),
                    'home_score': int(game.get('hs', 0)),
                    'game_status': self._parse_game_status(game.get('q')),
                    'quarter': game.get('q'),
                    'redzone': game.get('rz', '0') == '1',
                    'possession': game.get('p', ''),
                    'down': game.get('d', ''),
                    'yards_to_go': game.get('togo', ''),
                    'yard_line': game.get('yl', ''),
                    'season': current_season,
                    'week': current_week,
                    'season_type': season_type
                }

                games.append(game_data)

            return pd.DataFrame(games)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch NFL scores: {e}")
            return pd.DataFrame()
        except ET.ParseError as e:
            logger.error(f"Failed to parse NFL scorestrip XML: {e}")
            return pd.DataFrame()

    def get_team_stats(self, season: int = None, season_type: str = 'REG') -> pd.DataFrame:
        """Get team statistics from NFL.com"""
        current_season = season or datetime.now().year

        # NFL.com team stats endpoint
        url = f"http://www.nfl.com/stats/team"
        params = {
            'season': current_season,
            'seasontype': 2 if season_type == 'REG' else 1,  # 2=regular, 1=preseason
            'conference': 'ALL'
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            # This would parse the HTML/JSON response
            # NFL.com structure can vary, so this is a template
            return self._parse_team_stats_response(response.text, current_season)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch NFL team stats: {e}")
            return pd.DataFrame()

    def get_player_stats(self, position: str = 'QB', season: int = None, season_type: str = 'REG') -> pd.DataFrame:
        """Get player statistics by position"""
        current_season = season or datetime.now().year

        url = "http://www.nfl.com/stats/player"
        params = {
            'season': current_season,
            'seasontype': 2 if season_type == 'REG' else 1,
            'position': position.upper()
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            return self._parse_player_stats_response(response.text, position, current_season)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch NFL player stats: {e}")
            return pd.DataFrame()

    def get_standings(self, season: int = None) -> pd.DataFrame:
        """Get current NFL standings"""
        current_season = season or datetime.now().year

        url = "http://www.nfl.com/standings"
        params = {'season': current_season}

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            return self._parse_standings_response(response.text, current_season)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch NFL standings: {e}")
            return pd.DataFrame()

    def get_injury_report(self, week: int = None, season: int = None) -> pd.DataFrame:
        """Get official injury reports"""
        current_season = season or datetime.now().year
        current_week = week or self._get_current_week()

        # NFL injury report endpoint (if available)
        url = "http://www.nfl.com/injuries"
        params = {
            'season': current_season,
            'week': current_week
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            return self._parse_injury_response(response.text, current_season, current_week)

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch NFL injury report: {e}")
            return pd.DataFrame()

    def get_schedule(self, season: int = None, season_type: str = 'REG') -> pd.DataFrame:
        """Get complete NFL schedule"""
        current_season = season or datetime.now().year

        url = "http://www.nfl.com/schedules"
        params = {
            'season': current_season,
            'seasonType': season_type
        }

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            return self._parse_schedule_response(response.text, current_season, season_type)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch NFL schedule: {e}")
            return pd.DataFrame()

    def _parse_game_status(self, quarter_info: str) -> str:
        """Parse NFL quarter info to game status"""
        if not quarter_info:
            return 'Scheduled'

        if quarter_info == 'P':
            return 'Pregame'
        elif quarter_info == 'F':
            return 'Final'
        elif quarter_info == 'FO':
            return 'Final Overtime'
        elif quarter_info in ['1', '2', '3', '4']:
            return f'Q{quarter_info}'
        elif quarter_info == 'H':
            return 'Halftime'
        elif quarter_info == 'OT':
            return 'Overtime'
        else:
            return 'In Progress'

    def _parse_team_stats_response(self, response_text: str, season: int) -> pd.DataFrame:
        """Parse team statistics from NFL.com response"""
        # This would implement actual HTML/JSON parsing
        # NFL.com format can vary, so this is a structured placeholder

        # Example structure for what we'd extract:
        team_stats_template = {
            'team': [],
            'season': season,
            'games_played': [],
            'total_yards': [],
            'passing_yards': [],
            'rushing_yards': [],
            'points_scored': [],
            'points_allowed': [],
            'turnovers': [],
            'penalties': [],
            'time_of_possession': []
        }

        logger.info(f"Would parse team stats from NFL.com for season {season}")
        return pd.DataFrame(team_stats_template)

    def _parse_player_stats_response(self, response_text: str, position: str, season: int) -> pd.DataFrame:
        """Parse player statistics from NFL.com response"""
        # Template for player stats structure
        player_stats_template = {
            'player_name': [],
            'team': [],
            'position': position,
            'season': season,
            'games_played': [],
            'games_started': []
        }

        # Position-specific stats would be added here
        if position == 'QB':
            player_stats_template.update({
                'pass_attempts': [],
                'pass_completions': [],
                'passing_yards': [],
                'passing_tds': [],
                'interceptions': [],
                'qb_rating': []
            })
        elif position == 'RB':
            player_stats_template.update({
                'rushing_attempts': [],
                'rushing_yards': [],
                'rushing_tds': [],
                'receptions': [],
                'receiving_yards': [],
                'receiving_tds': []
            })

        logger.info(f"Would parse {position} stats from NFL.com for season {season}")
        return pd.DataFrame(player_stats_template)

    def _parse_standings_response(self, response_text: str, season: int) -> pd.DataFrame:
        """Parse standings from NFL.com response"""
        standings_template = {
            'team': [],
            'conference': [],
            'division': [],
            'wins': [],
            'losses': [],
            'ties': [],
            'win_percentage': [],
            'division_record': [],
            'conference_record': [],
            'points_for': [],
            'points_against': [],
            'point_differential': [],
            'season': season
        }

        logger.info(f"Would parse standings from NFL.com for season {season}")
        return pd.DataFrame(standings_template)

    def _parse_injury_response(self, response_text: str, season: int, week: int) -> pd.DataFrame:
        """Parse injury report from NFL.com response"""
        injury_template = {
            'player_name': [],
            'team': [],
            'position': [],
            'injury_status': [],  # Out, Doubtful, Questionable, Probable
            'injury_type': [],
            'body_part': [],
            'week': week,
            'season': season,
            'report_date': []
        }

        logger.info(f"Would parse injury report from NFL.com for season {season} week {week}")
        return pd.DataFrame(injury_template)

    def _parse_schedule_response(self, response_text: str, season: int, season_type: str) -> pd.DataFrame:
        """Parse schedule from NFL.com response"""
        schedule_template = {
            'game_id': [],
            'season': season,
            'season_type': season_type,
            'week': [],
            'game_date': [],
            'game_time': [],
            'home_team': [],
            'away_team': [],
            'tv_network': [],
            'stadium': []
        }

        logger.info(f"Would parse schedule from NFL.com for season {season} {season_type}")
        return pd.DataFrame(schedule_template)

    def _get_current_week(self) -> int:
        """Estimate current NFL week"""
        now = datetime.now()

        # NFL season typically starts first Thursday of September
        season_start = datetime(now.year, 9, 1)

        # Find first Thursday
        while season_start.weekday() != 3:  # Thursday = 3
            season_start += timedelta(days=1)

        # Calculate weeks since season start
        weeks_elapsed = (now - season_start).days // 7

        return max(1, min(weeks_elapsed + 1, 18))  # Weeks 1-18


# Alternative API endpoints that might be available
class NFLAPIAlternatives:
    """Alternative NFL data endpoints"""

    @staticmethod
    def get_nfl_feed_endpoints() -> Dict[str, str]:
        """Known NFL public data endpoints"""
        return {
            'scorestrip': 'http://www.nfl.com/ajax/scorestrip',
            'live_update': 'http://www.nfl.com/liveupdate/scores/scores.json',
            'game_center': 'http://www.nfl.com/liveupdate/game-center/{game_id}/{game_id}_gtd.json',
            'drive_chart': 'http://www.nfl.com/liveupdate/game-center/{game_id}/{game_id}_gtd.json',
            'play_by_play': 'http://www.nfl.com/liveupdate/game-center/{game_id}/{game_id}_gtd.json'
        }

    @staticmethod
    def get_mobile_endpoints() -> Dict[str, str]:
        """NFL mobile app endpoints (sometimes less restricted)"""
        return {
            'mobile_scores': 'http://static.nfl.com/liveupdate/scores/scores.json',
            'mobile_schedule': 'http://static.nfl.com/static/content/public/static/schedule',
            'mobile_standings': 'http://static.nfl.com/static/content/public/static/standings'
        }


# Integration function
def integrate_nfl_official_data(pipeline_instance):
    """Add NFL.com official data to existing pipeline"""
    nfl_client = NFLOfficialClient()

    def _get_nfl_live_scores(season: int, week: int) -> pd.DataFrame:
        """Official NFL live scores"""
        return nfl_client.get_live_scores(season, week)

    def _get_nfl_team_stats(season: int) -> pd.DataFrame:
        """Official NFL team statistics"""
        return nfl_client.get_team_stats(season)

    def _get_nfl_standings(season: int) -> pd.DataFrame:
        """Official NFL standings"""
        return nfl_client.get_standings(season)

    def _get_nfl_injury_report(season: int, week: int) -> pd.DataFrame:
        """Official NFL injury reports"""
        return nfl_client.get_injury_report(week, season)

    # Add to pipeline
    pipeline_instance._get_nfl_live_scores = _get_nfl_live_scores
    pipeline_instance._get_nfl_team_stats = _get_nfl_team_stats
    pipeline_instance._get_nfl_standings = _get_nfl_standings
    pipeline_instance._get_nfl_injury_report = _get_nfl_injury_report
    pipeline_instance.nfl_client = nfl_client

    logger.info("NFL.com official data integration added to pipeline")
    return pipeline_instance