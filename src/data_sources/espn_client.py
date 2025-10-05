"""
ESPN API Client
Free data source for NFL stats, scores, and team information
No authentication required for public endpoints
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class ESPNClient:
    """Client for ESPN's public NFL API endpoints"""

    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 1  # 1 second between requests to be respectful

    def get_live_scores(self, season: int = None, week: int = None) -> pd.DataFrame:
        """Get live scores and game status"""
        url = f"{self.base_url}/scoreboard"

        params = {}
        if season and week:
            # ESPN uses different date format
            params['dates'] = self._get_week_dates(season, week)

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            games = []

            for event in data.get('events', []):
                competition = event['competitions'][0]
                game_info = {
                    'espn_game_id': event['id'],
                    'game_date': event['date'],
                    'game_status': event['status']['type']['name'],
                    'game_status_detail': event['status']['type']['detail'],
                    'home_team': competition['competitors'][0]['team']['abbreviation'],
                    'away_team': competition['competitors'][1]['team']['abbreviation'],
                    'home_score': int(competition['competitors'][0].get('score', 0)),
                    'away_score': int(competition['competitors'][1].get('score', 0)),
                    'is_neutral_site': competition.get('neutralSite', False),
                    'venue_name': competition['venue']['fullName'],
                    'venue_city': competition['venue']['address']['city'],
                    'venue_state': competition['venue']['address']['state']
                }

                # Add odds if available
                if 'odds' in competition:
                    odds = competition['odds'][0] if competition['odds'] else {}
                    game_info.update({
                        'spread': odds.get('spread'),
                        'over_under': odds.get('overUnder'),
                        'home_moneyline': odds.get('homeTeamOdds', {}).get('moneyLine'),
                        'away_moneyline': odds.get('awayTeamOdds', {}).get('moneyLine')
                    })

                games.append(game_info)

            return pd.DataFrame(games)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ESPN scores: {e}")
            return pd.DataFrame()

    def get_team_stats(self, season: int = None) -> pd.DataFrame:
        """Get comprehensive team statistics"""
        url = f"{self.base_url}/teams"

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            teams_data = []

            for team in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                team_info = {
                    'espn_team_id': team['team']['id'],
                    'team_code': team['team']['abbreviation'],
                    'team_name': team['team']['displayName'],
                    'team_location': team['team']['location'],
                    'conference': team['team'].get('conferenceId'),
                    'division': team['team'].get('divisionId'),
                    'color': team['team'].get('color'),
                    'logo_url': team['team'].get('logos', [{}])[0].get('href'),
                    'wins': 0,  # Will be populated by separate stats call
                    'losses': 0,
                    'ties': 0
                }

                # Get detailed stats for this team
                team_stats = self._get_team_detailed_stats(team['team']['id'], season)
                team_info.update(team_stats)

                teams_data.append(team_info)

            return pd.DataFrame(teams_data)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ESPN team stats: {e}")
            return pd.DataFrame()

    def _get_team_detailed_stats(self, team_id: str, season: int = None) -> Dict:
        """Get detailed statistics for a specific team"""
        season_param = season or datetime.now().year
        url = f"{self.base_url}/teams/{team_id}/statistics"

        params = {'season': season_param}

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            stats = {}

            # Parse team statistics
            for category in data.get('statistics', []):
                category_name = category.get('name', '').lower().replace(' ', '_')

                for stat in category.get('stats', []):
                    stat_name = f"{category_name}_{stat.get('name', '').lower().replace(' ', '_')}"
                    stat_value = stat.get('value', 0)

                    # Convert to numeric if possible
                    try:
                        stat_value = float(stat_value)
                    except (ValueError, TypeError):
                        pass

                    stats[stat_name] = stat_value

            return stats

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch detailed stats for team {team_id}: {e}")
            return {}

    def get_player_stats(self, position: str = None, season: int = None) -> pd.DataFrame:
        """Get player statistics by position"""
        url = f"{self.base_url}/athletes"

        params = {}
        if season:
            params['season'] = season
        if position:
            params['position'] = position

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            players = []

            for athlete in data.get('athletes', []):
                player_info = {
                    'espn_player_id': athlete['id'],
                    'player_name': athlete['displayName'],
                    'position': athlete.get('position', {}).get('abbreviation'),
                    'team': athlete.get('team', {}).get('abbreviation'),
                    'jersey_number': athlete.get('jersey'),
                    'age': athlete.get('age'),
                    'height': athlete.get('displayHeight'),
                    'weight': athlete.get('displayWeight'),
                    'experience': athlete.get('experience'),
                    'headshot_url': athlete.get('headshot', {}).get('href')
                }

                # Get player statistics
                player_stats = self._get_player_detailed_stats(athlete['id'], season)
                player_info.update(player_stats)

                players.append(player_info)

            return pd.DataFrame(players)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ESPN player stats: {e}")
            return pd.DataFrame()

    def _get_player_detailed_stats(self, player_id: str, season: int = None) -> Dict:
        """Get detailed statistics for a specific player"""
        season_param = season or datetime.now().year
        url = f"{self.base_url}/athletes/{player_id}/statistics"

        params = {'season': season_param}

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            stats = {}

            # Parse player statistics
            for split in data.get('splits', []):
                for category in split.get('categories', []):
                    category_name = category.get('name', '').lower().replace(' ', '_')

                    for stat in category.get('stats', []):
                        stat_name = f"{category_name}_{stat.get('name', '').lower().replace(' ', '_')}"
                        stat_value = stat.get('value', 0)

                        # Convert to numeric if possible
                        try:
                            stat_value = float(stat_value)
                        except (ValueError, TypeError):
                            pass

                        stats[stat_name] = stat_value

            return stats

        except requests.RequestException as e:
            logger.debug(f"No detailed stats available for player {player_id}")
            return {}

    def get_standings(self, season: int = None) -> pd.DataFrame:
        """Get current NFL standings"""
        url = f"{self.base_url}/standings"

        params = {}
        if season:
            params['season'] = season

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            standings = []

            for group in data.get('standings', []):
                for entry in group.get('entries', []):
                    team_data = {
                        'team_code': entry['team']['abbreviation'],
                        'team_name': entry['team']['displayName'],
                        'wins': entry['stats'][0]['value'],  # Wins usually first
                        'losses': entry['stats'][1]['value'],  # Losses usually second
                        'ties': entry['stats'][2]['value'] if len(entry['stats']) > 2 else 0,
                        'win_percentage': entry['stats'][3]['value'] if len(entry['stats']) > 3 else 0,
                        'conference_rank': entry.get('rank', 0),
                        'division': group.get('name', ''),
                        'conference': group.get('parent', {}).get('name', '')
                    }

                    # Add additional stats if available
                    for i, stat in enumerate(entry.get('stats', [])):
                        if i > 3:  # Beyond basic win/loss/tie/pct
                            stat_name = stat.get('name', f'stat_{i}').lower().replace(' ', '_')
                            team_data[stat_name] = stat.get('value', 0)

                    standings.append(team_data)

            return pd.DataFrame(standings)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ESPN standings: {e}")
            return pd.DataFrame()

    def _get_week_dates(self, season: int, week: int) -> str:
        """Convert season/week to ESPN date format"""
        # This is a simplified conversion - ESPN uses specific date ranges
        # Would need more sophisticated logic for exact week mapping
        start_date = datetime(season, 9, 1) + timedelta(weeks=week-1)
        return start_date.strftime('%Y%m%d')


# Integration with existing data pipeline
def integrate_espn_data(pipeline_instance):
    """Add ESPN data to existing pipeline"""
    espn_client = ESPNClient()

    # Add ESPN methods to pipeline
    def _get_espn_scores(season: int, week: int) -> pd.DataFrame:
        """Enhanced scores with ESPN data"""
        return espn_client.get_live_scores(season, week)

    def _get_espn_team_stats(season: int) -> pd.DataFrame:
        """Enhanced team stats with ESPN data"""
        return espn_client.get_team_stats(season)

    def _get_espn_standings(season: int) -> pd.DataFrame:
        """Current standings from ESPN"""
        return espn_client.get_standings(season)

    # Monkey patch the methods (or better: inherit and extend)
    pipeline_instance._get_espn_scores = _get_espn_scores
    pipeline_instance._get_espn_team_stats = _get_espn_team_stats
    pipeline_instance._get_espn_standings = _get_espn_standings

    logger.info("ESPN integration added to data pipeline")
    return pipeline_instance