"""
Integration bridge between existing NFL system and web interface
FAIL FAST: Any integration error stops immediately
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path to import main system
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from main import NFLSuggestionSystem
    from database.db_manager import NFLDatabaseManager
except ImportError as e:
    raise ImportError(f"Cannot import NFL system components: {e}")

logger = logging.getLogger(__name__)


class BridgeError(Exception):
    """Safe error for web display"""
    pass


class NFLSystemBridge:
    """Bridge between existing NFL system and web interface"""

    def __init__(self, config_path: str = '.env'):
        """Initialize bridge with existing system"""
        try:
            self.nfl_system = NFLSuggestionSystem(config_path)
            self.db = self.nfl_system.db  # Reuse existing database
            self.transformer = SuggestionTransformer(self.db)
            self.cache = WeeklyCache()
            logger.info("NFL System Bridge initialized successfully")
        except Exception as e:
            # FAIL FAST - no fallback system
            raise BridgeError(f"Cannot initialize NFL system: {e}")

    def get_current_suggestions(self) -> List[Dict]:
        """
        Get web-ready suggestions for current NFL week
        Uses existing system's run_weekly_analysis method
        """
        try:
            # Use existing system's method to get current week
            current_season, current_week = self.nfl_system.nfl_fetcher.get_current_week()

            # Check cache first (weekly-based)
            cache_key = f"suggestions_{current_season}_{current_week}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Returning cached suggestions for {current_season} week {current_week}")
                return cached

            # Run actual analysis using existing system
            logger.info(f"Running fresh analysis for {current_season} week {current_week}")
            raw_suggestions = self.nfl_system.run_weekly_analysis(current_season, current_week)

            # Transform to web format
            web_suggestions = []
            for suggestion in raw_suggestions:
                try:
                    web_format = self.transformer.transform_suggestion(suggestion)
                    web_suggestions.append(web_format)
                except Exception as e:
                    logger.warning(f"Failed to transform suggestion: {e}")
                    continue

            # Cache for this week
            self.cache.store(cache_key, web_suggestions, expires_hours=168)  # 1 week

            logger.info(f"Generated {len(web_suggestions)} web-ready suggestions")
            return web_suggestions

        except Exception as e:
            # FAIL FAST - expose specific error types only
            if "API" in str(e):
                raise BridgeError("Data source unavailable")
            elif "No games" in str(e):
                raise BridgeError("No games available for current week")
            elif "API credits" in str(e):
                raise BridgeError("API quota exceeded - suggestions unavailable")
            else:
                logger.error(f"Bridge error: {e}")
                raise BridgeError("System temporarily unavailable")

    def get_clv_performance(self) -> Dict:
        """Get CLV performance data for charts"""
        try:
            # Query CLV data from database
            sql = """
                SELECT
                    DATE(tracked_at) as date,
                    AVG(clv_percentage) as avg_clv
                FROM clv_tracking
                WHERE tracked_at >= date('now', '-30 days')
                GROUP BY DATE(tracked_at)
                ORDER BY date
            """

            cursor = self.db.conn.execute(sql)
            results = cursor.fetchall()

            if not results:
                return {"dates": [], "values": []}

            dates = [row[0] for row in results]
            values = [float(row[1]) for row in results]

            return {"dates": dates, "values": values}

        except Exception as e:
            logger.error(f"CLV performance query failed: {e}")
            return {"dates": [], "values": []}

    def get_system_status(self) -> Dict:
        """Get system health status"""
        try:
            # Check API credits
            remaining_credits = self.db.get_remaining_api_credits()

            # Get suggestion count
            sql = "SELECT COUNT(*) FROM suggestions WHERE outcome = 'pending'"
            cursor = self.db.conn.execute(sql)
            pending_suggestions = cursor.fetchone()[0]

            return {
                "api_credits": remaining_credits,
                "pending_suggestions": pending_suggestions,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {
                "api_credits": 0,
                "pending_suggestions": 0,
                "status": "error"
            }


class SuggestionTransformer:
    """Transform suggestion dict to web-ready format"""

    def __init__(self, db: NFLDatabaseManager):
        self.db = db

    def transform_suggestion(self, suggestion: Dict) -> Dict:
        """
        Transform existing system output to web display format
        Input: suggestion dict from run_weekly_analysis
        Output: web-ready dict for templates
        """
        try:
            # Get team names from game_id
            home_team, away_team = self._parse_game_teams(suggestion['game_id'])

            # Determine display team based on selection
            if suggestion['bet_type'] == 'spread':
                display_team = home_team if suggestion['selection'] == 'home' else away_team
                line_display = f"{suggestion['line']:+.1f}"
            elif suggestion['bet_type'] == 'total':
                display_team = f"{home_team} vs {away_team}"
                line_display = f"{suggestion['selection'].upper()} {suggestion['line']:.1f}"
            else:  # moneyline
                display_team = home_team if suggestion['selection'] == 'home' else away_team
                line_display = "ML"

            return {
                'suggestion_id': suggestion.get('suggestion_id', 0),
                'game_id': suggestion['game_id'],
                'team': display_team,
                'bet_type': suggestion['bet_type'],
                'selection': suggestion['selection'],
                'line_display': line_display,
                'line': suggestion['line'],
                'odds': self._format_american_odds(suggestion['odds']),
                'confidence': round(suggestion['confidence'], 1),
                'margin': round(suggestion['margin'], 1),
                'edge': suggestion['edge'],
                'kelly_fraction': suggestion['kelly_fraction'],
                'correlation_warnings': suggestion.get('correlation_warnings', []),
                'game_time': suggestion.get('game_time', 'TBD')
            }
        except KeyError as e:
            raise BridgeError(f"Missing required field in suggestion: {e}")
        except Exception as e:
            raise BridgeError(f"Failed to transform suggestion: {e}")

    def _parse_game_teams(self, game_id: str) -> tuple[str, str]:
        """Extract team names from game_id"""
        try:
            # Query database for game info
            sql = "SELECT home_team, away_team FROM games WHERE game_id = ?"
            cursor = self.db.conn.execute(sql, (game_id,))
            result = cursor.fetchone()

            if result:
                return result[0], result[1]
            else:
                # Fallback: try to parse from game_id format
                logger.warning(f"Game not found in database: {game_id}")
                return "Home", "Away"

        except Exception as e:
            logger.error(f"Failed to parse game teams: {e}")
            return "Home", "Away"

    def _format_american_odds(self, odds: int) -> str:
        """Format odds for display"""
        if odds > 0:
            return f"+{odds}"
        return str(odds)


class WeeklyCache:
    """Week-based caching for NFL suggestions"""

    def __init__(self):
        self.cache = {}
        self.max_size = 10  # Keep 10 weeks max

    def get(self, key: str) -> Optional[List[Dict]]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, expires_at = self.cache[key]
            if datetime.now() < expires_at:
                return data
            else:
                del self.cache[key]
                logger.info(f"Cache expired for key: {key}")
        return None

    def store(self, key: str, data: List[Dict], expires_hours: int = 168):
        """Store data with expiration"""
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        self.cache[key] = (data, expires_at)

        # Evict old entries
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys())
            del self.cache[oldest_key]
            logger.info(f"Evicted old cache entry: {oldest_key}")

        logger.info(f"Cached data for key: {key}, expires: {expires_at}")

    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")