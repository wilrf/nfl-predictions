"""
The Odds API Client - Free Tier (500 requests/month)
FAIL FAST - No retries, no fallbacks
Batch all requests to minimize API usage
"""

import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OddsAPIError(Exception):
    """Custom exception for Odds API operations"""
    pass


class OddsAPIClient:
    """
    The Odds API client optimized for free tier (500 req/month)
    Strategy: 4 requests per week = 16 per month
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "americanfootball_nfl"

    # Free tier limits
    FREE_TIER_MONTHLY = 500
    FREE_TIER_DAILY = 16  # Self-imposed to spread usage

    def __init__(self, api_key: str, db_manager):
        """Initialize client with API key and database"""
        if not api_key:
            raise OddsAPIError("API key required")

        self.api_key = api_key
        self.db = db_manager

        # Check remaining credits
        self.remaining_credits = self._get_remaining_credits()
        logger.info(f"Odds API initialized with {self.remaining_credits} credits remaining")

    def _get_remaining_credits(self) -> int:
        """Get remaining API credits from database"""
        try:
            return self.db.get_remaining_api_credits()
        except Exception as e:
            raise OddsAPIError(f"Cannot get API credits: {e}")

    def fetch_week_odds(self, week_type: str = "current") -> Dict:
        """
        Fetch odds for all games in ONE request
        week_type: 'opening', 'midweek', 'closing', 'current'
        """
        # Validate we have credits
        if self.remaining_credits <= 0:
            raise OddsAPIError(f"No API credits remaining (0/{self.FREE_TIER_MONTHLY})")

        # Validate request timing based on type
        self._validate_request_timing(week_type)

        # Prepare request
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
            'bookmakers': 'pinnacle,draftkings,fanduel,betmgm,caesars'
        }

        try:
            # Make request - FAIL FAST on any error
            response = requests.get(url, params=params, timeout=30)

            # Check response
            if response.status_code != 200:
                error_msg = response.text if response.text else f"Status {response.status_code}"
                raise OddsAPIError(f"API request failed: {error_msg}")

            # Get remaining credits from header
            remaining = int(response.headers.get('x-requests-remaining', 0))
            used = int(response.headers.get('x-requests-used', 1))

            # Track usage in database
            games_data = response.json()
            self.db.track_api_usage(
                endpoint='/sports/nfl/odds',
                credits_used=used,
                remaining=remaining,
                response_code=200,
                games_fetched=len(games_data)
            )

            # Update local count
            self.remaining_credits = remaining

            # Process and store odds
            processed = self._process_odds_response(games_data, week_type)

            logger.info(f"Fetched odds for {len(games_data)} games. Credits: {remaining}/{self.FREE_TIER_MONTHLY}")

            return processed

        except requests.RequestException as e:
            # Track failed request
            self.db.track_api_usage(
                endpoint='/sports/nfl/odds',
                credits_used=0,
                remaining=self.remaining_credits,
                response_code=0,
                games_fetched=0,
                error=str(e)
            )
            raise OddsAPIError(f"Request failed: {e}")

    def _validate_request_timing(self, week_type: str):
        """
        Validate timing for different snapshot types
        FREE TIER STRATEGY:
        - Tuesday 6am: Opening lines
        - Thursday 6pm: Midweek check
        - Saturday 11pm: Pre-Sunday
        - Sunday 9am: Closing lines
        """
        now = datetime.now()
        day = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour

        if week_type == "opening":
            # Tuesday morning for opening lines (recommended)
            if day != 1 or hour < 5 or hour > 8:
                logger.warning(f"Opening lines recommended Tuesday 6-8am (currently {now.strftime('%A %I%p')})")

        elif week_type == "midweek":
            # Thursday evening (recommended)
            if day != 3 or hour < 17 or hour > 20:
                logger.warning(f"Midweek update recommended Thursday 5-8pm (currently {now.strftime('%A %I%p')})")

        elif week_type == "closing":
            # Sunday morning before games (recommended)
            if day != 6 or hour < 8 or hour > 11:
                logger.warning(f"Closing lines recommended Sunday 8-11am (currently {now.strftime('%A %I%p')})")

        elif week_type == "current":
            # Saturday night check (recommended)
            if day != 5 or hour < 22:
                logger.warning(f"Current update recommended Saturday after 10pm (currently {now.strftime('%A %I%p')})")

    def _process_odds_response(self, games: List[Dict], snapshot_type: str) -> Dict:
        """Process and store odds data in database"""
        timestamp = datetime.now()
        processed_games = []

        for game in games:
            game_id = game.get('id')
            if not game_id:
                continue

            # Extract game info
            game_info = {
                'game_id': game_id,
                'home_team': game.get('home_team'),
                'away_team': game.get('away_team'),
                'commence_time': game.get('commence_time')
            }

            # Process each bookmaker's odds
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker.get('key')

                # Initialize odds data
                odds_data = {
                    'game_id': game_id,
                    'timestamp': timestamp,
                    'snapshot_type': snapshot_type,
                    'book': book_name,
                    'spread_home': None,
                    'spread_away': None,
                    'total_over': None,
                    'total_under': None,
                    'ml_home': None,
                    'ml_away': None
                }

                # Extract markets
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')

                    if market_key == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == game_info['home_team']:
                                odds_data['spread_home'] = outcome.get('point', 0)
                                odds_data['spread_odds_home'] = outcome.get('price', -110)
                            elif outcome['name'] == game_info['away_team']:
                                odds_data['spread_away'] = outcome.get('point', 0)
                                odds_data['spread_odds_away'] = outcome.get('price', -110)

                    elif market_key == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == 'Over':
                                odds_data['total_over'] = outcome.get('point', 0)
                                odds_data['total_odds_over'] = outcome.get('price', -110)
                            elif outcome['name'] == 'Under':
                                odds_data['total_under'] = outcome.get('point', 0)
                                odds_data['total_odds_under'] = outcome.get('price', -110)

                    elif market_key == 'h2h':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == game_info['home_team']:
                                odds_data['ml_home'] = outcome.get('price', 0)
                            elif outcome['name'] == game_info['away_team']:
                                odds_data['ml_away'] = outcome.get('price', 0)

                # Store odds if we have valid data
                if odds_data['spread_home'] is not None:
                    try:
                        self.db.insert_odds_snapshot(odds_data)
                    except Exception as e:
                        logger.error(f"Failed to store odds for {game_id}/{book_name}: {e}")

            processed_games.append(game_info)

        return {
            'games': processed_games,
            'timestamp': timestamp,
            'snapshot_type': snapshot_type,
            'games_count': len(processed_games)
        }

    def get_sharp_consensus(self, game_id: str) -> Dict:
        """
        Get consensus from sharp books (Pinnacle, Circa)
        Uses cached data only - no API call
        """
        try:
            # Query database for sharp book odds
            sharp_books = ['pinnacle', 'circa', 'bookmaker']

            sql = """
                SELECT book, spread_home, total_over, ml_home, ml_away
                FROM odds_snapshots
                WHERE game_id = ? AND book IN ({})
                ORDER BY timestamp DESC
            """.format(','.join(['?' for _ in sharp_books]))

            cursor = self.db.conn.execute(sql, [game_id] + sharp_books)
            odds = cursor.fetchall()

            if not odds:
                raise OddsAPIError(f"No sharp book odds for game {game_id}")

            # Calculate consensus
            spreads = [o[1] for o in odds if o[1] is not None]
            totals = [o[2] for o in odds if o[2] is not None]

            consensus = {
                'spread': sum(spreads) / len(spreads) if spreads else None,
                'total': sum(totals) / len(totals) if totals else None,
                'books_included': len(odds)
            }

            return consensus

        except Exception as e:
            raise OddsAPIError(f"Failed to get sharp consensus: {e}")

    def calculate_no_vig_probability(self, american_odds: int) -> float:
        """Convert American odds to no-vig probability"""
        if american_odds == 0:
            raise OddsAPIError("Invalid odds: 0")

        # Convert to decimal odds
        if american_odds > 0:
            decimal_odds = (american_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(american_odds)) + 1

        # Convert to implied probability
        implied_prob = 1 / decimal_odds

        return implied_prob

    def remove_vig(self, odds1: int, odds2: int) -> tuple[float, float]:
        """
        Remove vig from two-way market
        Returns no-vig probabilities for both sides
        """
        # Get implied probabilities
        prob1 = self.calculate_no_vig_probability(odds1)
        prob2 = self.calculate_no_vig_probability(odds2)

        # Calculate total (includes vig)
        total = prob1 + prob2

        if total == 0:
            raise OddsAPIError("Invalid odds - total probability is 0")

        # Remove vig
        no_vig_prob1 = prob1 / total
        no_vig_prob2 = prob2 / total

        return no_vig_prob1, no_vig_prob2

    def detect_line_movement(self, game_id: str) -> Dict:
        """
        Detect significant line movement
        Uses cached data only
        """
        try:
            sql = """
                SELECT snapshot_type, timestamp, spread_home, total_over
                FROM odds_snapshots
                WHERE game_id = ? AND book = 'pinnacle'
                ORDER BY timestamp
            """

            cursor = self.db.conn.execute(sql, (game_id,))
            movements = cursor.fetchall()

            if len(movements) < 2:
                return {'spread_move': 0, 'total_move': 0}

            # Compare first and last
            first = movements[0]
            last = movements[-1]

            spread_move = last[2] - first[2] if first[2] and last[2] else 0
            total_move = last[3] - first[3] if first[3] and last[3] else 0

            return {
                'spread_move': spread_move,
                'total_move': total_move,
                'is_steam': abs(spread_move) >= 1.0 or abs(total_move) >= 1.5
            }

        except Exception as e:
            raise OddsAPIError(f"Failed to detect line movement: {e}")

    def emergency_fetch(self, game_id: str) -> Dict:
        """
        Emergency single game fetch - uses precious credits!
        Only for critical situations
        """
        if self.remaining_credits <= 5:
            raise OddsAPIError(f"Cannot use emergency fetch with only {self.remaining_credits} credits")

        logger.warning(f"EMERGENCY FETCH for {game_id} - using valuable API credits!")

        # This would fetch just one game but The Odds API doesn't support single game fetch
        # So we fetch all and filter
        all_odds = self.fetch_week_odds("current")

        # Find the specific game
        for game in all_odds.get('games', []):
            if game['game_id'] == game_id:
                return game

        raise OddsAPIError(f"Game {game_id} not found in emergency fetch")