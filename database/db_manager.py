"""
SQLite Database Manager with FAIL FAST principle
No fallbacks, no retries - any error causes immediate failure
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


class NFLDatabaseManager:
    """Database manager with strict FAIL FAST behavior"""

    def __init__(self, db_path: str = None):
        """Initialize database connection - FAIL if cannot connect"""
        if db_path is None:
            db_path = Path(__file__).parent / 'nfl_suggestions.db'

        self.db_path = db_path

        # Try to connect - FAIL FAST if unable
        try:
            self.conn = sqlite3.connect(str(db_path))
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.conn.execute("PRAGMA foreign_keys = ON")  # Enforce foreign keys
            self.conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        except sqlite3.Error as e:
            raise DatabaseError(f"Cannot connect to database at {db_path}: {e}")

        # Initialize schema - FAIL if cannot create tables
        self._initialize_schema()

    def _initialize_schema(self):
        """Create tables from schema.sql - FAIL if error"""
        schema_path = Path(__file__).parent / 'schema.sql'

        if not schema_path.exists():
            raise DatabaseError(f"Schema file not found at {schema_path}")

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            self.conn.executescript(schema_sql)
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize schema: {e}")
        except Exception as e:
            raise DatabaseError(f"Unexpected error initializing schema: {e}")

    def insert_game(self, game_data: Dict) -> str:
        """Insert game data - FAIL if any required field missing"""
        required_fields = ['game_id', 'season', 'week', 'game_type',
                          'home_team', 'away_team', 'game_time', 'stadium']

        for field in required_fields:
            if field not in game_data:
                raise DatabaseError(f"Missing required field: {field}")

        sql = """
            INSERT INTO games (game_id, season, week, game_type, home_team,
                             away_team, game_time, stadium, is_outdoor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            # Serialize datetime to ISO format string for SQLite
            game_time = game_data['game_time']
            if isinstance(game_time, (datetime, pd.Timestamp)):
                game_time = game_time.isoformat()
            elif pd.isna(game_time):
                game_time = None

            self.conn.execute(sql, (
                game_data['game_id'],
                game_data['season'],
                game_data['week'],
                game_data['game_type'],
                game_data['home_team'],
                game_data['away_team'],
                game_time,  # Now properly serialized
                game_data['stadium'],
                game_data.get('is_outdoor', False)
            ))
            self.conn.commit()
            return game_data['game_id']
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed' in str(e):
                raise DatabaseError(f"Game {game_data['game_id']} already exists")
            raise DatabaseError(f"Integrity error inserting game: {e}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert game: {e}")

    def insert_odds_snapshot(self, odds_data: Dict) -> int:
        """Insert odds snapshot - FAIL if invalid data"""
        required = ['game_id', 'timestamp', 'snapshot_type', 'book',
                   'spread_home', 'spread_away', 'total_over', 'total_under',
                   'ml_home', 'ml_away']

        for field in required:
            if field not in odds_data:
                raise DatabaseError(f"Missing required odds field: {field}")

        # Validate snapshot type
        valid_types = ['opening', 'midweek', 'closing', 'current']
        if odds_data['snapshot_type'] not in valid_types:
            raise DatabaseError(f"Invalid snapshot_type: {odds_data['snapshot_type']}")

        sql = """
            INSERT INTO odds_snapshots
            (game_id, timestamp, snapshot_type, book, spread_home, spread_away,
             total_over, total_under, ml_home, ml_away, spread_odds_home,
             spread_odds_away, total_odds_over, total_odds_under, api_credits_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            cursor = self.conn.execute(sql, (
                odds_data['game_id'],
                odds_data['timestamp'],
                odds_data['snapshot_type'],
                odds_data['book'],
                odds_data['spread_home'],
                odds_data['spread_away'],
                odds_data['total_over'],
                odds_data['total_under'],
                odds_data['ml_home'],
                odds_data['ml_away'],
                odds_data.get('spread_odds_home', -110),
                odds_data.get('spread_odds_away', -110),
                odds_data.get('total_odds_over', -110),
                odds_data.get('total_odds_under', -110),
                odds_data.get('api_credits_used', 1)
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert odds snapshot: {e}")

    def insert_suggestion(self, suggestion: Dict) -> int:
        """Insert betting suggestion - assumes validation done upstream"""
        sql = """
            INSERT INTO suggestions
            (game_id, bet_type, selection, line, odds, confidence, margin,
             edge, kelly_fraction, model_probability, market_probability, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """

        try:
            cursor = self.conn.execute(sql, (
                suggestion['game_id'],
                suggestion['bet_type'],
                suggestion['selection'],
                suggestion['line'],
                suggestion['odds'],
                suggestion['confidence'],
                suggestion['margin'],
                suggestion['edge'],
                suggestion['kelly_fraction'],
                suggestion['model_probability'],
                suggestion['market_probability']
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert suggestion: {e}")

    def add_correlation_warning(self, sugg_id1: int, sugg_id2: int,
                               correlation_type: str, correlation_value: float):
        """Add correlation warning between suggestions"""
        if not (0 <= correlation_value <= 1):
            raise DatabaseError(f"Invalid correlation value: {correlation_value}")

        # Determine warning level
        if correlation_value > 0.7:
            warning_level = 'high'
            warning_message = f"HIGH CORRELATION ({correlation_value:.0%}): {correlation_type}"
        elif correlation_value > 0.4:
            warning_level = 'moderate'
            warning_message = f"Moderate correlation ({correlation_value:.0%}): {correlation_type}"
        else:
            warning_level = 'low'
            warning_message = f"Low correlation ({correlation_value:.0%}): {correlation_type}"

        # Ensure id1 < id2 for uniqueness
        if sugg_id1 > sugg_id2:
            sugg_id1, sugg_id2 = sugg_id2, sugg_id1

        sql = """
            INSERT INTO correlation_warnings
            (suggestion_id1, suggestion_id2, correlation_type, correlation_value,
             warning_level, warning_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        try:
            self.conn.execute(sql, (
                sugg_id1, sugg_id2, correlation_type, correlation_value,
                warning_level, warning_message
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add correlation warning: {e}")

    def track_api_usage(self, endpoint: str, credits_used: int,
                       remaining: int, response_code: int,
                       games_fetched: int = 0, error: str = None):
        """Track API usage for free tier management"""
        sql = """
            INSERT INTO api_usage
            (endpoint, credits_used, remaining_credits, response_code,
             games_fetched, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        try:
            self.conn.execute(sql, (
                endpoint, credits_used, remaining, response_code,
                games_fetched, error
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to track API usage: {e}")

    def get_remaining_api_credits(self) -> int:
        """Get remaining API credits - FAIL if cannot determine"""
        sql = """
            SELECT remaining_credits
            FROM api_usage
            ORDER BY request_time DESC
            LIMIT 1
        """

        try:
            cursor = self.conn.execute(sql)
            row = cursor.fetchone()

            if row is None:
                # First time - assume full 500 for free tier
                return 500

            return row[0]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get API credits: {e}")

    def get_week_games(self, season: int, week: int) -> List[Dict]:
        """Get all games for a week"""
        sql = """
            SELECT * FROM games
            WHERE season = ? AND week = ?
            ORDER BY game_time
        """

        try:
            cursor = self.conn.execute(sql, (season, week))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get week games: {e}")

    def get_latest_odds(self, game_id: str) -> Dict:
        """Get most recent odds for a game"""
        sql = """
            SELECT * FROM odds_snapshots
            WHERE game_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            cursor = self.conn.execute(sql, (game_id,))
            row = cursor.fetchone()

            if row is None:
                raise DatabaseError(f"No odds found for game {game_id}")

            return dict(row)
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get latest odds: {e}")

    def get_opening_line(self, game_id: str, bet_type: str) -> Optional[float]:
        """Get opening line for CLV calculation - uses earliest available line"""
        sql = """
            SELECT spread_home, total_over
            FROM odds_snapshots
            WHERE game_id = ?
            ORDER BY timestamp ASC
            LIMIT 1
        """

        try:
            cursor = self.conn.execute(sql, (game_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            if bet_type == 'spread':
                return row[0]
            elif bet_type == 'total':
                return row[1]
            else:
                raise DatabaseError(f"Invalid bet type for line: {bet_type}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get opening line: {e}")

    def get_closing_line(self, game_id: str, bet_type: str) -> Optional[Dict]:
        """Get closing line for CLV calculation - uses latest available line"""
        sql = """
            SELECT spread_home, spread_away, total_over, total_under,
                   spread_odds_home, spread_odds_away, total_odds_over, total_odds_under
            FROM odds_snapshots
            WHERE game_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            cursor = self.conn.execute(sql, (game_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            if bet_type == 'spread':
                return {
                    'line': row[0],  # spread_home
                    'odds': row[4]   # spread_odds_home
                }
            elif bet_type == 'total':
                return {
                    'line': row[2],  # total_over
                    'odds': row[6]   # total_odds_over
                }
            else:
                raise DatabaseError(f"Invalid bet type: {bet_type}")

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get closing line: {e}")

    def record_clv(self, suggestion_id: int, opening: float, closing: float):
        """Record CLV for a suggestion"""
        # Get suggestion details
        sql = """
            SELECT game_id, bet_type, line
            FROM suggestions
            WHERE suggestion_id = ?
        """

        try:
            cursor = self.conn.execute(sql, (suggestion_id,))
            sugg = cursor.fetchone()

            if sugg is None:
                raise DatabaseError(f"Suggestion {suggestion_id} not found")

            clv_points = closing - opening
            clv_pct = abs(clv_points / opening) if opening != 0 else 0
            beat_closing = abs(sugg[2] - opening) < abs(closing - opening)

            insert_sql = """
                INSERT INTO clv_tracking
                (suggestion_id, game_id, bet_type, opening_line, closing_line,
                 our_line, clv_points, clv_percentage, beat_closing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            self.conn.execute(insert_sql, (
                suggestion_id, sugg[0], sugg[1], opening, closing,
                sugg[2], clv_points, clv_pct, beat_closing
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to record CLV: {e}")

    def calculate_postgame_clv(self, game_id: str) -> Dict:
        """Calculate CLV for all suggestions for a completed game"""
        try:
            # Get all suggestions for this game
            suggestions_sql = """
                SELECT suggestion_id, game_id, bet_type, line
                FROM suggestions
                WHERE game_id = ? AND outcome != 'pending'
            """
            cursor = self.conn.execute(suggestions_sql, (game_id,))
            suggestions = cursor.fetchall()
            
            if not suggestions:
                return {'processed': 0, 'errors': 0, 'success': 0}
            
            processed = 0
            errors = 0
            success = 0
            
            for sugg in suggestions:
                suggestion_id, game_id, bet_type, our_line = sugg
                
                try:
                    # Get opening and closing lines
                    opening_line = self.get_opening_line(game_id, bet_type)
                    closing_data = self.get_closing_line(game_id, bet_type)
                    
                    if opening_line is None or closing_data is None:
                        errors += 1
                        continue
                    
                    # Record CLV
                    self.record_clv(suggestion_id, opening_line, closing_data['line'])
                    success += 1
                    
                except DatabaseError:
                    errors += 1
                
                processed += 1
            
            return {
                'processed': processed,
                'errors': errors,
                'success': success,
                'coverage': success / processed if processed > 0 else 0
            }
            
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to calculate post-game CLV: {e}")

    def get_clv_health_report(self) -> Dict:
        """Get CLV tracking health report"""
        try:
            # Count total suggestions
            total_sql = "SELECT COUNT(*) FROM suggestions"
            cursor = self.conn.execute(total_sql)
            total_suggestions = cursor.fetchone()[0]
            
            # Count suggestions with CLV tracked
            clv_sql = """
                SELECT COUNT(DISTINCT suggestion_id) 
                FROM clv_tracking
            """
            cursor = self.conn.execute(clv_sql)
            clv_tracked = cursor.fetchone()[0]
            
            # Count games with odds data
            odds_sql = "SELECT COUNT(DISTINCT game_id) FROM odds_snapshots"
            cursor = self.conn.execute(odds_sql)
            games_with_odds = cursor.fetchone()[0]
            
            # Count total games
            games_sql = "SELECT COUNT(*) FROM games"
            cursor = self.conn.execute(games_sql)
            total_games = cursor.fetchone()[0]
            
            # Calculate coverage
            clv_coverage = clv_tracked / total_suggestions if total_suggestions > 0 else 0
            odds_coverage = games_with_odds / total_games if total_games > 0 else 0
            
            return {
                'total_suggestions': total_suggestions,
                'clv_tracked': clv_tracked,
                'clv_coverage': clv_coverage,
                'total_games': total_games,
                'games_with_odds': games_with_odds,
                'odds_coverage': odds_coverage,
                'status': 'healthy' if clv_coverage > 0.8 else 'needs_attention'
            }
            
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get CLV health report: {e}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()