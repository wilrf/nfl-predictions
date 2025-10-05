"""
NFL Betting Suggestion System - Main Orchestrator
FAIL FAST - No fallbacks, no synthetic data
SUGGESTIONS ONLY - System suggests, never places bets
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Import our modules (updated paths for new structure)
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import NFLDatabaseManager, DatabaseError
from src.data.nfl_data_fetcher import NFLDataFetcher, NFLDataError
from src.data.odds_client import OddsAPIClient, OddsAPIError
from src.calculators.confidence import ConfidenceCalculator, ConfidenceError
from src.calculators.margin import MarginCalculator, MarginError
from src.calculators.correlation import CorrelationEngine, Bet, CorrelationError
from saved_models.model_integration import NFLModelEnsemble, ModelError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nfl_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemError(Exception):
    """Main system error - FAIL FAST"""
    pass


class NFLSuggestionSystem:
    """
    Main orchestrator for NFL betting suggestions
    FAIL FAST on any error - no fallbacks
    """

    def __init__(self, env_file: str = '.env'):
        """
        Initialize the suggestion system

        Args:
            env_file: Path to environment file
        """
        # Load environment variables
        if not Path(env_file).exists():
            raise SystemError(f"Environment file not found: {env_file}")

        load_dotenv(env_file)

        # Validate required environment variables
        self.odds_api_key = os.getenv('ODDS_API_KEY')
        if not self.odds_api_key:
            raise SystemError("ODDS_API_KEY not set in environment")

        # Initialize components - FAIL if any component fails
        try:
            logger.info("Initializing NFL Suggestion System...")

            # Database
            db_path = os.getenv('DATABASE_PATH', 'database/nfl_suggestions.db')
            self.db = NFLDatabaseManager(db_path)

            # Data fetchers
            self.nfl_fetcher = NFLDataFetcher()
            self.odds_client = OddsAPIClient(self.odds_api_key, self.db)

            # Calculators
            bankroll = float(os.getenv('BANKROLL', 10000))
            self.confidence_calc = ConfidenceCalculator()
            self.margin_calc = MarginCalculator(bankroll)
            self.correlation_engine = CorrelationEngine()

            # Models (optional - system can run without them)
            models_dir = os.getenv('MODELS_DIR', 'models/saved_models')
            try:
                self.models = NFLModelEnsemble(models_dir)
                logger.info("XGBoost models loaded successfully")
            except ModelError as e:
                logger.warning(f"Models not available: {e}")
                logger.warning("System will run without predictions - fetching data only")
                self.models = None

            # Configuration
            self.min_edge = float(os.getenv('MIN_EDGE', 0.02))
            self.max_kelly = float(os.getenv('MAX_KELLY', 0.25))
            self.max_weekly_bets = int(os.getenv('MAX_WEEKLY_BETS', 20))

            logger.info("System initialized successfully")

        except Exception as e:
            raise SystemError(f"Failed to initialize system: {e}")

    def run_weekly_analysis(self, season: int = None, week: int = None) -> List[Dict]:
        """
        Run complete weekly analysis and generate suggestions

        Args:
            season: NFL season (defaults to current)
            week: NFL week (defaults to current)

        Returns:
            List of betting suggestions
        """
        try:
            # Get current season/week if not provided
            if season is None or week is None:
                current_season, current_week = self.nfl_fetcher.get_current_week()
                season = season or current_season
                week = week or current_week

            logger.info(f"Running analysis for Season {season} Week {week}")

            # Step 1: Fetch NFL game data (REAL DATA ONLY)
            games = self._fetch_and_store_games(season, week)
            if not games:
                raise SystemError(f"No games found for Season {season} Week {week}")

            # Step 2: Fetch odds (if within API limits)
            self._fetch_and_store_odds()

            # Step 3: Generate predictions (if models available)
            predictions = self._generate_predictions(games)

            # Step 4: Calculate suggestions
            suggestions = self._calculate_suggestions(games, predictions)

            # Step 5: Check correlations and add warnings
            if suggestions:
                self._add_correlation_warnings(suggestions)

            # Step 6: Store suggestions in database
            self._store_suggestions(suggestions)

            logger.info(f"Generated {len(suggestions)} suggestions")
            return suggestions

        except Exception as e:
            raise SystemError(f"Weekly analysis failed: {e}")

    def _fetch_and_store_games(self, season: int, week: int) -> List[Dict]:
        """Fetch and store NFL game data"""
        try:
            # Fetch games from nfl_data_py
            games_df = self.nfl_fetcher.fetch_week_games(season, week)

            games_list = []
            for _, game in games_df.iterrows():
                game_data = {
                    'game_id': game['game_id'],
                    'season': season,
                    'week': week,
                    'game_type': game.get('game_type', 'REG'),
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_time': game['game_time'],
                    'stadium': game.get('stadium', 'Unknown'),
                    'is_outdoor': self._is_outdoor_stadium(game.get('stadium', ''))
                }

                # Store in database
                try:
                    self.db.insert_game(game_data)
                except DatabaseError as e:
                    if "already exists" in str(e):
                        logger.debug(f"Game {game_data['game_id']} already in database")
                    else:
                        raise

                games_list.append(game_data)

            return games_list

        except NFLDataError as e:
            raise SystemError(f"Failed to fetch games: {e}")

    def _fetch_and_store_odds(self):
        """Fetch odds if within API limits"""
        try:
            # Check if we should fetch based on day/time
            now = datetime.now()
            day = now.weekday()
            hour = now.hour

            # Determine snapshot type based on timing
            if day == 1 and 5 <= hour <= 8:  # Tuesday morning
                snapshot_type = "opening"
            elif day == 3 and 17 <= hour <= 20:  # Thursday evening
                snapshot_type = "midweek"
            elif day == 5 and hour >= 22:  # Saturday night
                snapshot_type = "current"
            elif day == 6 and 8 <= hour <= 11:  # Sunday morning
                snapshot_type = "closing"
            else:
                logger.info(f"Not scheduled odds fetch time (Day {day}, Hour {hour})")
                return

            # Check API credits
            remaining = self.odds_client.remaining_credits
            if remaining <= 0:
                logger.warning("No API credits remaining")
                return

            logger.info(f"Fetching {snapshot_type} odds (Credits: {remaining})")
            self.odds_client.fetch_week_odds(snapshot_type)

        except OddsAPIError as e:
            logger.error(f"Failed to fetch odds: {e}")
            # Don't fail the entire system for odds issues

    def _generate_predictions(self, games: List[Dict]) -> Dict:
        """Generate model predictions if available"""
        predictions = {}

        if not self.models:
            logger.warning("No models loaded - skipping predictions")
            return predictions

        try:
            # Get team stats for features
            season = games[0]['season']
            week = games[0]['week']

            if week > 1:
                team_stats = self.nfl_fetcher.fetch_team_stats(season, week - 1)
            else:
                logger.info("Week 1 - using prior season stats for predictions")
                # Use prior season stats for Week 1 predictions
                prior_season = season - 1
                try:
                    team_stats = self.nfl_fetcher.fetch_team_stats(prior_season, 18)  # Full season stats
                    logger.info(f"Using {prior_season} season stats for Week 1 predictions")
                except Exception as e:
                    logger.error(f"Failed to fetch prior season stats: {e}")
                    return predictions

            # Generate predictions for each game
            for game in games:
                features = self._create_features(game, team_stats)
                if features is not None:
                    game_id = game['game_id']

                    # Validate features before prediction
                    try:
                        self.models.validate_features(features, 'spread')
                        self.models.validate_features(features, 'total')
                    except ModelError as e:
                        logger.error(f"Feature validation failed for {game_id}: {e}")
                        continue

                    # Predict spread
                    spread_pred = self.models.predict_spread(features)
                    predictions[f"{game_id}_spread"] = spread_pred

                    # Predict total
                    total_pred = self.models.predict_total(features)
                    predictions[f"{game_id}_total"] = total_pred

        except ModelError as e:
            logger.error(f"Model prediction failed: {e}")

        return predictions

    def _create_features(self, game: Dict, team_stats: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create features for model prediction"""
        try:
            home_team = game['home_team']
            away_team = game['away_team']

            # Get team statistics
            home_stats = team_stats[team_stats['team'] == home_team]
            away_stats = team_stats[team_stats['team'] == away_team]

            if home_stats.empty or away_stats.empty:
                logger.warning(f"Missing stats for {home_team} or {away_team}")
                return None

            # Create comprehensive feature vector
            game_time = pd.to_datetime(game['game_time'])
            features = pd.DataFrame([{
                # Core EPA metrics
                'home_off_epa': home_stats.iloc[0]['off_epa_play'],
                'home_def_epa': home_stats.iloc[0]['def_epa_play'],
                'away_off_epa': away_stats.iloc[0]['off_epa_play'],
                'away_def_epa': away_stats.iloc[0]['def_epa_play'],
                
                # Success rates
                'home_off_success': home_stats.iloc[0]['off_success_rate'],
                'away_off_success': away_stats.iloc[0]['off_success_rate'],
                'home_def_success': home_stats.iloc[0]['def_success_rate'],
                'away_def_success': away_stats.iloc[0]['def_success_rate'],
                
                # Yards per play
                'home_off_yards': home_stats.iloc[0]['off_yards_play'],
                'away_off_yards': away_stats.iloc[0]['off_yards_play'],
                'home_def_yards': home_stats.iloc[0]['def_yards_play'],
                'away_def_yards': away_stats.iloc[0]['def_yards_play'],
                
                # Explosive play rates
                'home_off_explosive': home_stats.iloc[0]['off_explosive_rate'],
                'away_off_explosive': away_stats.iloc[0]['off_explosive_rate'],
                'home_def_explosive': home_stats.iloc[0]['def_explosive_allowed'],
                'away_def_explosive': away_stats.iloc[0]['def_explosive_allowed'],
                
                # Situational efficiency
                'home_rz_td_pct': home_stats.iloc[0]['rz_td_pct'],
                'away_rz_td_pct': away_stats.iloc[0]['rz_td_pct'],
                'home_third_down_pct': home_stats.iloc[0]['third_down_pct'],
                'away_third_down_pct': away_stats.iloc[0]['third_down_pct'],
                
                # Game context
                'is_outdoor': game['is_outdoor'],
                'hour_of_day': game_time.hour,
                'day_of_week': game_time.weekday(),
                'is_primetime': 1 if game_time.hour >= 20 else 0,
                'week': game['week'],
                'season': game['season']
            }])

            return features

        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            return None

    def _calculate_suggestions(self, games: List[Dict], predictions: Dict) -> List[Dict]:
        """Calculate betting suggestions"""
        suggestions = []

        for game in games:
            game_id = game['game_id']

            # Get latest odds
            try:
                odds = self.db.get_latest_odds(game_id)
            except DatabaseError:
                logger.warning(f"No odds for game {game_id}")
                continue

            # Check spread bet
            spread_key = f"{game_id}_spread"
            if spread_key in predictions:
                spread_sugg = self._evaluate_spread_bet(game, odds, predictions[spread_key])
                if spread_sugg:
                    suggestions.append(spread_sugg)

            # Check total bet
            total_key = f"{game_id}_total"
            if total_key in predictions:
                total_sugg = self._evaluate_total_bet(game, odds, predictions[total_key])
                if total_sugg:
                    suggestions.append(total_sugg)

        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        # Limit to max weekly bets
        if len(suggestions) > self.max_weekly_bets:
            logger.info(f"Limiting suggestions from {len(suggestions)} to {self.max_weekly_bets}")
            suggestions = suggestions[:self.max_weekly_bets]

        return suggestions

    def _evaluate_spread_bet(self, game: Dict, odds: Dict, prediction: Dict) -> Optional[Dict]:
        """Evaluate potential spread bet"""
        try:
            # Get model probability
            model_prob = prediction['home_win_prob']

            # Get market probability (remove vig)
            market_prob, _ = self.odds_client.remove_vig(
                odds['spread_odds_home'],
                odds['spread_odds_away']
            )

            # Calculate edge
            edge = model_prob - market_prob

            # Check minimum edge
            if abs(edge) < self.min_edge:
                return None

            # Determine which side to bet
            if edge > 0:
                selection = 'home'
                bet_prob = model_prob
                bet_market_prob = market_prob
                bet_odds = odds['spread_odds_home']
                line = odds['spread_home']
            else:
                selection = 'away'
                bet_prob = 1 - model_prob
                bet_market_prob = 1 - market_prob  # FIX: Invert market probability for away bets
                bet_odds = odds['spread_odds_away']
                line = odds['spread_away']
                edge = abs(edge)

            # Calculate confidence
            confidence = self.confidence_calc.calculate(
                edge=edge,
                model_probability=bet_prob,
                market_probability=bet_market_prob,  # Use inverted probability
                model_certainty=prediction.get('model_confidence', 0.5)
            )

            # Calculate Kelly and margin
            kelly = self.margin_calc.calculate_kelly_fraction(edge, bet_prob, self.max_kelly)
            margin = self.margin_calc.calculate(edge, kelly, bet_odds, confidence)

            return {
                'game_id': game['game_id'],
                'bet_type': 'spread',
                'selection': selection,
                'team': game[f'{selection}_team'],
                'line': line,
                'odds': bet_odds,
                'confidence': confidence,
                'margin': margin,
                'edge': edge,
                'kelly_fraction': kelly,
                'model_probability': bet_prob,
                'market_probability': market_prob,
                'is_favorite': line < 0 if selection == 'home' else line > 0
            }

        except (ConfidenceError, MarginError) as e:
            logger.debug(f"Spread bet rejected: {e}")
            return None

    def _evaluate_total_bet(self, game: Dict, odds: Dict, prediction: Dict) -> Optional[Dict]:
        """Evaluate potential total bet"""
        try:
            # Similar logic for totals
            model_prob = prediction['over_prob']

            market_prob, _ = self.odds_client.remove_vig(
                odds['total_odds_over'],
                odds['total_odds_under']
            )

            edge = model_prob - market_prob

            if abs(edge) < self.min_edge:
                return None

            if edge > 0:
                selection = 'over'
                bet_prob = model_prob
                bet_market_prob = market_prob
                bet_odds = odds['total_odds_over']
                line = odds['total_over']
            else:
                selection = 'under'
                bet_prob = 1 - model_prob
                bet_market_prob = 1 - market_prob  # FIX: Invert market probability for under bets
                bet_odds = odds['total_odds_under']
                line = odds['total_under']
                edge = abs(edge)

            confidence = self.confidence_calc.calculate(
                edge=edge,
                model_probability=bet_prob,
                market_probability=bet_market_prob,  # Use inverted probability
                model_certainty=prediction.get('model_confidence', 0.5)
            )

            kelly = self.margin_calc.calculate_kelly_fraction(edge, bet_prob, self.max_kelly)
            margin = self.margin_calc.calculate(edge, kelly, bet_odds, confidence)

            return {
                'game_id': game['game_id'],
                'bet_type': 'total',
                'selection': selection,
                'team': f"{game['home_team']} vs {game['away_team']}",
                'line': line,
                'odds': bet_odds,
                'confidence': confidence,
                'margin': margin,
                'edge': edge,
                'kelly_fraction': kelly,
                'model_probability': bet_prob,
                'market_probability': market_prob,
                'is_favorite': False  # N/A for totals
            }

        except (ConfidenceError, MarginError) as e:
            logger.debug(f"Total bet rejected: {e}")
            return None

    def _add_correlation_warnings(self, suggestions: List[Dict]):
        """Add correlation warnings to suggestions"""
        # Convert to Bet objects for correlation engine
        bets = []
        for sugg in suggestions:
            bet = Bet(
                game_id=sugg['game_id'],
                bet_type=sugg['bet_type'],
                selection=sugg['selection'],
                team=sugg['team'],
                line=sugg['line'],
                is_favorite=sugg.get('is_favorite', False),
                confidence=sugg['confidence'],
                margin=sugg['margin']
            )
            bets.append(bet)

        # Check correlations
        warnings = self.correlation_engine.check_correlations(bets)

        # Add warnings and apply penalties to suggestions
        for warning in warnings:
            # Find matching suggestions
            for sugg in suggestions:
                if (sugg['game_id'] == warning.bet1.game_id and
                    sugg['bet_type'] == warning.bet1.bet_type):
                    if 'correlation_warnings' not in sugg:
                        sugg['correlation_warnings'] = []
                    sugg['correlation_warnings'].append(warning.message)
                    
                    # Apply correlation penalty to confidence
                    penalty_multiplier = self._get_correlation_penalty(warning.correlation_value)
                    original_confidence = sugg['confidence']
                    sugg['confidence'] = original_confidence * penalty_multiplier
                    sugg['original_confidence'] = original_confidence
                    sugg['correlation_penalty'] = penalty_multiplier
                    
                    logger.info(f"Applied correlation penalty {penalty_multiplier:.2f} to {sugg['game_id']} {sugg['bet_type']}")

    def _get_correlation_penalty(self, correlation_value: float) -> float:
        """Calculate confidence penalty based on correlation strength"""
        if correlation_value >= 0.8:
            return 0.7  # 30% penalty for high correlation
        elif correlation_value >= 0.6:
            return 0.85  # 15% penalty for moderate correlation
        elif correlation_value >= 0.4:
            return 0.95  # 5% penalty for low correlation
        else:
            return 1.0  # No penalty for weak correlation

    def _store_suggestions(self, suggestions: List[Dict]):
        """Store suggestions in database and track CLV"""
        for sugg in suggestions:
            try:
                # Validate suggestion before storing
                if not self._validate_suggestion(sugg):
                    logger.warning(f"Suggestion failed validation: {sugg.get('game_id', 'unknown')}")
                    continue
                    
                sugg_id = self.db.insert_suggestion(sugg)
                sugg['suggestion_id'] = sugg_id
                
                # Track CLV if opening and closing lines are available
                self._track_clv_for_suggestion(sugg_id, sugg)
                
            except DatabaseError as e:
                logger.error(f"Failed to store suggestion: {e}")

    def _validate_suggestion(self, suggestion: Dict) -> bool:
        """Validate suggestion meets requirements before database storage"""
        try:
            # Validate confidence range
            if not (50 <= suggestion['confidence'] <= 90):
                logger.warning(f"Confidence {suggestion['confidence']} not in 50-90 range")
                return False

            # Validate margin range
            if not (0 <= suggestion['margin'] <= 30):
                logger.warning(f"Margin {suggestion['margin']} not in 0-30 range")
                return False

            # Validate minimum edge
            if suggestion['edge'] < 0.02:
                logger.warning(f"Edge {suggestion['edge']} below 2% minimum")
                return False

            # Validate kelly fraction
            if not (0 < suggestion['kelly_fraction'] <= 0.25):
                logger.warning(f"Invalid Kelly fraction: {suggestion['kelly_fraction']}")
                return False

            return True
            
        except KeyError as e:
            logger.error(f"Missing required field in suggestion: {e}")
            return False

    def _track_clv_for_suggestion(self, suggestion_id: int, suggestion: Dict):
        """Track CLV for a suggestion if opening/closing lines available"""
        try:
            game_id = suggestion['game_id']
            bet_type = suggestion['bet_type']
            
            # Get opening line
            opening_line = self.db.get_opening_line(game_id, bet_type)
            if opening_line is None:
                logger.debug(f"No opening line available for {game_id} {bet_type}")
                return
                
            # Get closing line
            closing_data = self.db.get_closing_line(game_id, bet_type)
            if closing_data is None:
                logger.debug(f"No closing line available for {game_id} {bet_type}")
                return
                
            # Record CLV
            self.db.record_clv(suggestion_id, opening_line, closing_data['line'])
            logger.info(f"Recorded CLV for suggestion {suggestion_id}: {opening_line} -> {closing_data['line']}")
            
        except DatabaseError as e:
            logger.warning(f"Failed to track CLV for suggestion {suggestion_id}: {e}")

    def _is_outdoor_stadium(self, stadium: str) -> bool:
        """Check if stadium is outdoor"""
        outdoor_stadiums = [
            'Lambeau', 'Soldier', 'Heinz', 'FirstEnergy', 'Bills',
            'Gillette', 'MetLife', 'M&T Bank', 'Paul Brown', 'FedEx'
        ]
        return any(outdoor in stadium for outdoor in outdoor_stadiums)

    def display_suggestions(self, suggestions: List[Dict]):
        """Display suggestions in tiered format"""
        if not suggestions:
            print("\n" + "=" * 60)
            print("NO SUGGESTIONS FOR THIS WEEK")
            print("No games meet minimum edge requirement (2%)")
            print("=" * 60)
            return

        # Group by confidence tier
        premium = [s for s in suggestions if s['confidence'] >= 80]
        standard = [s for s in suggestions if 65 <= s['confidence'] < 80]
        reference = [s for s in suggestions if 50 <= s['confidence'] < 65]

        print("\n" + "=" * 60)
        print(f"NFL BETTING SUGGESTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)

        # Premium tier
        if premium:
            print("\n🟢 PREMIUM PICKS (80+ Confidence)")
            print("-" * 40)
            for sugg in premium:
                self._print_suggestion(sugg)

        # Standard tier
        if standard:
            print("\n🟡 STANDARD PICKS (65-79 Confidence)")
            print("-" * 40)
            for sugg in standard:
                self._print_suggestion(sugg)

        # Reference tier
        if reference:
            print("\n⚪ REFERENCE PICKS (50-64 Confidence)")
            print("-" * 40)
            for sugg in reference:
                self._print_suggestion(sugg)

        # Summary
        print("\n" + "=" * 60)
        print(f"TOTAL SUGGESTIONS: {len(suggestions)}")
        print(f"Average Confidence: {sum(s['confidence'] for s in suggestions)/len(suggestions):.1f}")
        print(f"Average Margin: {sum(s['margin'] for s in suggestions)/len(suggestions):.1f}")

        # Correlation warnings summary
        total_warnings = sum(len(s.get('correlation_warnings', [])) for s in suggestions)
        if total_warnings > 0:
            print(f"\n⚠️  CORRELATION WARNINGS: {total_warnings}")
            print("Review correlated bets carefully before wagering")

        print("=" * 60)

    def _print_suggestion(self, sugg: Dict):
        """Print individual suggestion"""
        print(f"\n{sugg['team']} {sugg['bet_type'].upper()} {sugg['line']:+.1f} @ {sugg['odds']:+d}")
        print(f"  Confidence: {sugg['confidence']:.1f} | Margin: {sugg['margin']:.1f}")
        print(f"  Edge: {sugg['edge']:.1%} | Kelly: {sugg['kelly_fraction']:.1%}")

        # Show correlation warnings
        if 'correlation_warnings' in sugg:
            for warning in sugg['correlation_warnings']:
                print(f"  ⚠️  {warning}")


def main():
    """Main entry point"""
    try:
        # Initialize system
        system = NFLSuggestionSystem()

        # Run analysis for current week
        suggestions = system.run_weekly_analysis()

        # Display results
        system.display_suggestions(suggestions)

    except SystemError as e:
        logger.error(f"System error: {e}")
        print(f"\n❌ SYSTEM ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()