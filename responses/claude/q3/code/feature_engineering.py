"""
NFL Feature Engineering Module
Comprehensive feature extraction for NFL betting predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
from scipy import stats
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Time windows for rolling features
    short_window: int = 3  # Last 3 games
    medium_window: int = 6  # Last 6 games  
    long_window: int = 10  # Last 10 games
    
    # DVOA parameters
    dvoa_baseline_year: int = 2020
    dvoa_adjustment_factor: float = 0.8
    
    # EPA parameters
    epa_decay_rate: float = 0.9  # Exponential decay for weighted EPA
    
    # Weather thresholds
    wind_impact_threshold: float = 20.0  # mph
    temp_impact_threshold: float = 32.0  # fahrenheit
    precipitation_impact_threshold: float = 0.1  # inches
    
    # Market parameters
    sharp_money_threshold: float = 0.2  # 20% divergence
    reverse_line_threshold: float = 1.0  # points


class DVOACalculator:
    """Defense-adjusted Value Over Average (DVOA) calculator"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.league_baselines = {}
        
    def calculate_dvoa(self, team_stats: pd.DataFrame, 
                      opponent_stats: pd.DataFrame,
                      league_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DVOA for offense and defense
        
        Args:
            team_stats: Team offensive/defensive statistics
            opponent_stats: Opponent statistics
            league_stats: League average statistics
        
        Returns:
            DataFrame with DVOA metrics
        """
        dvoa_features = pd.DataFrame(index=team_stats.index)
        
        # Offensive DVOA
        dvoa_features['offensive_dvoa'] = self._calculate_offensive_dvoa(
            team_stats, league_stats
        )
        
        # Defensive DVOA (negative is better)
        dvoa_features['defensive_dvoa'] = self._calculate_defensive_dvoa(
            team_stats, league_stats
        )
        
        # Special Teams DVOA
        dvoa_features['special_teams_dvoa'] = self._calculate_special_teams_dvoa(
            team_stats, league_stats
        )
        
        # Weighted DVOA (recent games weighted more)
        dvoa_features['weighted_dvoa'] = self._calculate_weighted_dvoa(
            dvoa_features, team_stats
        )
        
        # Opponent-adjusted DVOA
        if opponent_stats is not None:
            dvoa_features['opponent_adjusted_dvoa'] = self._adjust_for_opponent(
                dvoa_features, opponent_stats, league_stats
            )
        
        return dvoa_features
    
    def _calculate_offensive_dvoa(self, team_stats: pd.DataFrame, 
                                 league_stats: pd.DataFrame) -> pd.Series:
        """Calculate offensive DVOA"""
        # Success rate on plays
        success_rate = team_stats['successful_plays'] / team_stats['total_plays']
        league_success_rate = league_stats['successful_plays'].mean() / league_stats['total_plays'].mean()
        
        # Yards per play adjusted
        ypp = team_stats['total_yards'] / team_stats['total_plays']
        league_ypp = league_stats['total_yards'].mean() / league_stats['total_plays'].mean()
        
        # Points per drive
        ppd = team_stats['points_scored'] / team_stats['offensive_drives']
        league_ppd = league_stats['points_scored'].mean() / league_stats['offensive_drives'].mean()
        
        # Calculate DVOA
        dvoa = (
            0.4 * (success_rate - league_success_rate) / league_success_rate +
            0.3 * (ypp - league_ypp) / league_ypp +
            0.3 * (ppd - league_ppd) / league_ppd
        ) * 100
        
        return dvoa
    
    def _calculate_defensive_dvoa(self, team_stats: pd.DataFrame, 
                                 league_stats: pd.DataFrame) -> pd.Series:
        """Calculate defensive DVOA (negative is better)"""
        # Success rate allowed
        success_rate_allowed = team_stats['successful_plays_allowed'] / team_stats['defensive_plays']
        league_success_rate = league_stats['successful_plays'].mean() / league_stats['total_plays'].mean()
        
        # Yards per play allowed
        ypp_allowed = team_stats['yards_allowed'] / team_stats['defensive_plays']
        league_ypp = league_stats['total_yards'].mean() / league_stats['total_plays'].mean()
        
        # Points per drive allowed
        ppd_allowed = team_stats['points_allowed'] / team_stats['defensive_drives']
        league_ppd = league_stats['points_scored'].mean() / league_stats['offensive_drives'].mean()
        
        # Calculate DVOA (lower is better for defense)
        dvoa = (
            0.4 * (success_rate_allowed - league_success_rate) / league_success_rate +
            0.3 * (ypp_allowed - league_ypp) / league_ypp +
            0.3 * (ppd_allowed - league_ppd) / league_ppd
        ) * 100
        
        return dvoa
    
    def _calculate_special_teams_dvoa(self, team_stats: pd.DataFrame, 
                                     league_stats: pd.DataFrame) -> pd.Series:
        """Calculate special teams DVOA"""
        # Field goal success
        fg_success = team_stats['field_goals_made'] / (team_stats['field_goals_attempted'] + 1e-10)
        league_fg_success = league_stats['field_goals_made'].mean() / (league_stats['field_goals_attempted'].mean() + 1e-10)
        
        # Average starting field position
        field_position = team_stats['avg_starting_field_position']
        league_field_position = league_stats['avg_starting_field_position'].mean()
        
        # Return yards
        return_yards = team_stats['total_return_yards'] / (team_stats['total_returns'] + 1e-10)
        league_return_yards = league_stats['total_return_yards'].mean() / (league_stats['total_returns'].mean() + 1e-10)
        
        # Calculate DVOA
        dvoa = (
            0.3 * (fg_success - league_fg_success) / (league_fg_success + 1e-10) +
            0.4 * (field_position - league_field_position) / (league_field_position + 1e-10) +
            0.3 * (return_yards - league_return_yards) / (league_return_yards + 1e-10)
        ) * 100
        
        return dvoa
    
    def _calculate_weighted_dvoa(self, dvoa_features: pd.DataFrame, 
                                team_stats: pd.DataFrame) -> pd.Series:
        """Calculate weighted DVOA with recent games weighted more"""
        weights = np.array([0.5, 0.3, 0.2])  # Weights for offense, defense, special teams
        
        weighted_dvoa = (
            dvoa_features['offensive_dvoa'] * weights[0] -
            dvoa_features['defensive_dvoa'] * weights[1] +  # Negative because lower is better
            dvoa_features['special_teams_dvoa'] * weights[2]
        )
        
        # Apply recency weighting if game number available
        if 'game_number' in team_stats.columns:
            recency_weight = 1 + (team_stats['game_number'] - 1) * 0.05
            weighted_dvoa *= recency_weight
        
        return weighted_dvoa
    
    def _adjust_for_opponent(self, dvoa_features: pd.DataFrame,
                            opponent_stats: pd.DataFrame,
                            league_stats: pd.DataFrame) -> pd.Series:
        """Adjust DVOA for opponent strength"""
        # Calculate opponent DVOA
        opp_offensive_dvoa = self._calculate_offensive_dvoa(opponent_stats, league_stats)
        opp_defensive_dvoa = self._calculate_defensive_dvoa(opponent_stats, league_stats)
        
        # Adjust team DVOA based on opponent strength
        adjusted_dvoa = (
            dvoa_features['offensive_dvoa'] * (1 + opp_defensive_dvoa / 100) +
            dvoa_features['defensive_dvoa'] * (1 + opp_offensive_dvoa / 100)
        ) / 2
        
        return adjusted_dvoa


class EPACalculator:
    """Expected Points Added (EPA) calculator"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.ep_model = self._initialize_ep_model()
    
    def _initialize_ep_model(self) -> Dict[str, float]:
        """Initialize expected points model based on field position"""
        # Simplified EP model - in production, use actual NFL data
        ep_by_yard_line = {}
        for yard in range(1, 100):
            if yard <= 20:  # Red zone
                ep_by_yard_line[yard] = 3.0 + (yard / 20) * 2
            elif yard <= 50:  # Own half
                ep_by_yard_line[yard] = -0.5 + (yard / 50) * 2
            else:  # Opponent half
                ep_by_yard_line[yard] = 1.5 - ((yard - 50) / 50) * 3
        return ep_by_yard_line
    
    def calculate_epa(self, play_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EPA metrics from play-by-play data
        
        Args:
            play_data: Play-by-play data with field position and outcomes
        
        Returns:
            DataFrame with EPA metrics
        """
        epa_features = pd.DataFrame(index=play_data.index)
        
        # Basic EPA
        epa_features['epa_per_play'] = self._calculate_epa_per_play(play_data)
        
        # Weighted EPA (recent plays weighted more)
        epa_features['weighted_epa'] = self._calculate_weighted_epa(play_data)
        
        # Situational EPA
        epa_features['early_down_epa'] = self._calculate_situational_epa(
            play_data, downs=[1, 2]
        )
        epa_features['third_down_epa'] = self._calculate_situational_epa(
            play_data, downs=[3]
        )
        epa_features['redzone_epa'] = self._calculate_redzone_epa(play_data)
        
        # EPA variance (consistency metric)
        epa_features['epa_variance'] = self._calculate_epa_variance(play_data)
        
        # Success rate
        epa_features['success_rate'] = self._calculate_success_rate(play_data)
        
        # Explosive play rate
        epa_features['explosive_rate'] = self._calculate_explosive_rate(play_data)
        
        return epa_features
    
    def _calculate_epa_per_play(self, play_data: pd.DataFrame) -> pd.Series:
        """Calculate basic EPA per play"""
        if 'epa' in play_data.columns:
            return play_data.groupby('game_id')['epa'].mean()
        
        # Calculate from scratch if EPA not provided
        epa_values = []
        for idx, row in play_data.iterrows():
            start_ep = self.ep_model.get(row.get('yard_line', 50), 0)
            end_ep = self.ep_model.get(row.get('end_yard_line', 50), 0)
            
            # Add points for scoring plays
            if row.get('touchdown', 0):
                end_ep += 7
            elif row.get('field_goal', 0):
                end_ep += 3
            
            epa = end_ep - start_ep
            epa_values.append(epa)
        
        play_data['calculated_epa'] = epa_values
        return play_data.groupby('game_id')['calculated_epa'].mean()
    
    def _calculate_weighted_epa(self, play_data: pd.DataFrame) -> pd.Series:
        """Calculate weighted EPA with exponential decay"""
        grouped = play_data.groupby('game_id')
        
        weighted_epas = []
        for game_id, group in grouped:
            if 'epa' in group.columns:
                epa_values = group['epa'].values
            else:
                epa_values = group.get('calculated_epa', pd.Series(0)).values
            
            # Apply exponential weighting (recent plays weighted more)
            n_plays = len(epa_values)
            weights = self.config.epa_decay_rate ** np.arange(n_plays - 1, -1, -1)
            weights = weights / weights.sum()
            
            weighted_epa = np.dot(epa_values, weights)
            weighted_epas.append(weighted_epa)
        
        return pd.Series(weighted_epas, index=grouped.groups.keys())
    
    def _calculate_situational_epa(self, play_data: pd.DataFrame, 
                                  downs: List[int]) -> pd.Series:
        """Calculate EPA for specific down situations"""
        filtered_data = play_data[play_data['down'].isin(downs)]
        return self._calculate_epa_per_play(filtered_data)
    
    def _calculate_redzone_epa(self, play_data: pd.DataFrame) -> pd.Series:
        """Calculate EPA in the red zone (within 20 yards)"""
        redzone_data = play_data[play_data['yard_line'] <= 20]
        return self._calculate_epa_per_play(redzone_data)
    
    def _calculate_epa_variance(self, play_data: pd.DataFrame) -> pd.Series:
        """Calculate EPA variance (consistency metric)"""
        if 'epa' in play_data.columns:
            return play_data.groupby('game_id')['epa'].std()
        elif 'calculated_epa' in play_data.columns:
            return play_data.groupby('game_id')['calculated_epa'].std()
        else:
            return pd.Series(0, index=play_data['game_id'].unique())
    
    def _calculate_success_rate(self, play_data: pd.DataFrame) -> pd.Series:
        """Calculate success rate (positive EPA plays)"""
        if 'epa' in play_data.columns:
            epa_col = 'epa'
        elif 'calculated_epa' in play_data.columns:
            epa_col = 'calculated_epa'
        else:
            return pd.Series(0.5, index=play_data['game_id'].unique())
        
        success = play_data.groupby('game_id').apply(
            lambda x: (x[epa_col] > 0).mean()
        )
        return success
    
    def _calculate_explosive_rate(self, play_data: pd.DataFrame) -> pd.Series:
        """Calculate explosive play rate (gains > 20 yards)"""
        explosive = play_data.groupby('game_id').apply(
            lambda x: (x.get('yards_gained', 0) > 20).mean()
        )
        return explosive


class SituationalFeatures:
    """Extract situational and contextual features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_features(self, game_data: pd.DataFrame,
                        schedule_data: pd.DataFrame,
                        weather_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract all situational features
        
        Args:
            game_data: Game information
            schedule_data: Schedule and timing information
            weather_data: Weather conditions
        
        Returns:
            DataFrame with situational features
        """
        features = pd.DataFrame(index=game_data.index)
        
        # Rest and schedule features
        features = pd.concat([features, self._extract_rest_features(schedule_data)], axis=1)
        
        # Game context features
        features = pd.concat([features, self._extract_game_context(game_data)], axis=1)
        
        # Weather features
        if weather_data is not None:
            features = pd.concat([features, self._extract_weather_features(weather_data)], axis=1)
        
        # Travel features
        features = pd.concat([features, self._extract_travel_features(game_data)], axis=1)
        
        # Time zone features
        features = pd.concat([features, self._extract_timezone_features(game_data)], axis=1)
        
        return features
    
    def _extract_rest_features(self, schedule_data: pd.DataFrame) -> pd.DataFrame:
        """Extract rest and schedule-related features"""
        features = pd.DataFrame(index=schedule_data.index)
        
        # Days of rest
        features['days_rest'] = (
            schedule_data['game_date'] - schedule_data['prev_game_date']
        ).dt.days
        
        # Short rest (Thursday game after Sunday)
        features['short_rest'] = features['days_rest'] < 6
        
        # Long rest (after bye or Monday night)
        features['long_rest'] = features['days_rest'] > 10
        
        # Back-to-back road games
        features['consecutive_road'] = (
            (schedule_data['is_home'] == False) & 
            (schedule_data['prev_is_home'] == False)
        ).astype(int)
        
        # Games in X days windows
        features['games_last_14_days'] = schedule_data['games_last_14_days']
        features['games_last_21_days'] = schedule_data['games_last_21_days']
        
        return features
    
    def _extract_game_context(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """Extract game context features"""
        features = pd.DataFrame(index=game_data.index)
        
        # Divisional game
        features['is_divisional'] = game_data['is_divisional'].astype(int)
        
        # Conference game
        features['is_conference'] = game_data['is_conference'].astype(int)
        
        # Primetime game
        features['is_primetime'] = game_data['is_primetime'].astype(int)
        
        # Week of season
        features['week_of_season'] = game_data['week']
        
        # Late season (playoff implications)
        features['late_season'] = (game_data['week'] >= 14).astype(int)
        
        # Playoff clinched/eliminated
        features['playoff_clinched'] = game_data.get('playoff_clinched', 0)
        features['playoff_eliminated'] = game_data.get('playoff_eliminated', 0)
        
        # Revenge game (lost to opponent last meeting)
        features['revenge_game'] = game_data.get('lost_last_meeting', 0)
        
        # Win/loss streaks
        features['win_streak'] = game_data.get('win_streak', 0)
        features['loss_streak'] = game_data.get('loss_streak', 0)
        
        return features
    
    def _extract_weather_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Extract weather-related features"""
        features = pd.DataFrame(index=weather_data.index)
        
        # Temperature
        features['temperature'] = weather_data['temperature']
        features['extreme_cold'] = (weather_data['temperature'] < self.config.temp_impact_threshold).astype(int)
        features['extreme_heat'] = (weather_data['temperature'] > 85).astype(int)
        
        # Wind
        features['wind_speed'] = weather_data['wind_speed']
        features['high_wind'] = (weather_data['wind_speed'] > self.config.wind_impact_threshold).astype(int)
        
        # Precipitation
        features['precipitation'] = weather_data.get('precipitation', 0)
        features['is_precipitation'] = (weather_data.get('precipitation', 0) > self.config.precipitation_impact_threshold).astype(int)
        
        # Indoor/outdoor
        features['is_dome'] = weather_data.get('is_dome', 0)
        features['is_retractable'] = weather_data.get('is_retractable', 0)
        
        # Weather impact score (composite)
        features['weather_impact'] = (
            features['extreme_cold'] * 0.3 +
            features['extreme_heat'] * 0.2 +
            features['high_wind'] * 0.4 +
            features['is_precipitation'] * 0.1
        )
        
        return features
    
    def _extract_travel_features(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """Extract travel-related features"""
        features = pd.DataFrame(index=game_data.index)
        
        # Travel distance (if coordinates available)
        if 'travel_distance' in game_data.columns:
            features['travel_distance'] = game_data['travel_distance']
            features['long_travel'] = (game_data['travel_distance'] > 1500).astype(int)
            features['cross_country'] = (game_data['travel_distance'] > 2000).astype(int)
        
        # Surface change
        if 'home_surface' in game_data.columns and 'away_surface' in game_data.columns:
            features['surface_change'] = (game_data['home_surface'] != game_data['away_surface']).astype(int)
        
        # Altitude change (for Denver games)
        if 'altitude_difference' in game_data.columns:
            features['altitude_change'] = game_data['altitude_difference']
            features['high_altitude'] = (game_data['altitude_difference'] > 4000).astype(int)
        
        return features
    
    def _extract_timezone_features(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """Extract timezone-related features"""
        features = pd.DataFrame(index=game_data.index)
        
        if 'timezone_difference' in game_data.columns:
            features['timezone_difference'] = game_data['timezone_difference']
            features['west_to_east'] = ((game_data['timezone_difference'] < 0) & game_data['is_home']).astype(int)
            features['east_to_west'] = ((game_data['timezone_difference'] > 0) & game_data['is_home']).astype(int)
            
            # Early game for west coast team
            features['early_west_coast'] = (
                (game_data['game_time'].dt.hour < 13) & 
                (game_data['timezone_difference'] >= 3)
            ).astype(int)
        
        return features


class MarketIntelligence:
    """Extract market and betting line movement features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_features(self, betting_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract market intelligence features
        
        Args:
            betting_data: Betting lines, volumes, and movements
        
        Returns:
            DataFrame with market features
        """
        features = pd.DataFrame(index=betting_data.index)
        
        # Line movement features
        features = pd.concat([features, self._extract_line_movement(betting_data)], axis=1)
        
        # Sharp money indicators
        features = pd.concat([features, self._extract_sharp_indicators(betting_data)], axis=1)
        
        # Public betting patterns
        features = pd.concat([features, self._extract_public_patterns(betting_data)], axis=1)
        
        # Value indicators
        features = pd.concat([features, self._extract_value_indicators(betting_data)], axis=1)
        
        return features
    
    def _extract_line_movement(self, betting_data: pd.DataFrame) -> pd.DataFrame:
        """Extract line movement features"""
        features = pd.DataFrame(index=betting_data.index)
        
        # Spread movement
        features['spread_movement'] = betting_data['current_spread'] - betting_data['opening_spread']
        features['spread_movement_abs'] = np.abs(features['spread_movement'])
        
        # Total movement
        features['total_movement'] = betting_data['current_total'] - betting_data['opening_total']
        features['total_movement_abs'] = np.abs(features['total_movement'])
        
        # Reverse line movement
        features['reverse_line_movement'] = (
            (features['spread_movement'] * betting_data['public_bet_percentage'] < 0) &
            (features['spread_movement_abs'] >= self.config.reverse_line_threshold)
        ).astype(int)
        
        # Steam move (sharp action in same direction)
        features['steam_move'] = (
            (features['spread_movement_abs'] >= 2.5) &
            (betting_data['sharp_action'] == np.sign(features['spread_movement']))
        ).astype(int)
        
        # Line freeze (no movement despite heavy action)
        features['line_freeze'] = (
            (features['spread_movement_abs'] < 0.5) &
            (betting_data['bet_volume'] > betting_data['avg_bet_volume'] * 1.5)
        ).astype(int)
        
        return features
    
    def _extract_sharp_indicators(self, betting_data: pd.DataFrame) -> pd.DataFrame:
        """Extract sharp money indicators"""
        features = pd.DataFrame(index=betting_data.index)
        
        # Money vs bet percentage divergence
        features['money_bet_divergence'] = (
            betting_data['money_percentage'] - betting_data['bet_percentage']
        )
        
        # Sharp money indicator
        features['sharp_money'] = (
            np.abs(features['money_bet_divergence']) > self.config.sharp_money_threshold
        ).astype(int)
        
        # Sharp money side
        features['sharp_on_favorite'] = (
            (features['sharp_money'] == 1) &
            (features['money_bet_divergence'] > 0) &
            (betting_data['is_favorite'] == 1)
        ).astype(int)
        
        features['sharp_on_dog'] = (
            (features['sharp_money'] == 1) &
            (features['money_bet_divergence'] > 0) &
            (betting_data['is_favorite'] == 0)
        ).astype(int)
        
        # Professional betting percentage (estimated)
        if 'pro_bet_percentage' in betting_data.columns:
            features['pro_bet_percentage'] = betting_data['pro_bet_percentage']
            features['pro_vs_public'] = (
                betting_data['pro_bet_percentage'] - betting_data['public_bet_percentage']
            )
        
        # Ticket vs money ratio
        features['avg_bet_size_ratio'] = (
            betting_data['money_percentage'] / (betting_data['bet_percentage'] + 1e-10)
        )
        
        return features
    
    def _extract_public_patterns(self, betting_data: pd.DataFrame) -> pd.DataFrame:
        """Extract public betting patterns"""
        features = pd.DataFrame(index=betting_data.index)
        
        # Public betting percentage
        features['public_on_favorite'] = betting_data.get('public_bet_percentage', 0)
        features['public_fade'] = features['public_on_favorite'] > 70
        
        # Contrarian indicators
        features['contrarian_spot'] = (
            (features['public_on_favorite'] > 75) &
            (betting_data['spread'] >= 7)
        ).astype(int)
        
        # Public dog (rare)
        features['public_dog'] = (
            (features['public_on_favorite'] < 35) &
            (betting_data['is_favorite'] == 0)
        ).astype(int)
        
        # Recency bias (public on team that covered last game)
        if 'covered_last' in betting_data.columns:
            features['recency_bias'] = (
                (features['public_on_favorite'] > 60) &
                betting_data['covered_last']
            ).astype(int)
        
        return features
    
    def _extract_value_indicators(self, betting_data: pd.DataFrame) -> pd.DataFrame:
        """Extract value and edge indicators"""
        features = pd.DataFrame(index=betting_data.index)
        
        # Closing line value
        if 'closing_spread' in betting_data.columns:
            features['closing_line_value'] = (
                betting_data['bet_spread'] - betting_data['closing_spread']
            )
            features['positive_clv'] = (features['closing_line_value'] > 0).astype(int)
        
        # Key numbers
        key_numbers = [3, 7, 10, 14]
        features['on_key_number'] = betting_data['spread'].isin(key_numbers).astype(int)
        features['near_key_number'] = betting_data['spread'].apply(
            lambda x: min([abs(x - k) for k in key_numbers]) <= 0.5
        ).astype(int)
        
        # Line shopping value
        if 'best_available_spread' in betting_data.columns:
            features['line_value'] = (
                betting_data['best_available_spread'] - betting_data['current_spread']
            )
        
        # Teaser value
        features['wong_teaser'] = (
            (betting_data['spread'].between(-8.5, -5.5)) |
            (betting_data['spread'].between(1.5, 2.5))
        ).astype(int)
        
        return features


class NFLFeatureEngineering:
    """Main feature engineering pipeline for NFL betting"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.dvoa_calculator = DVOACalculator(self.config)
        self.epa_calculator = EPACalculator(self.config)
        self.situational_extractor = SituationalFeatures(self.config)
        self.market_extractor = MarketIntelligence(self.config)
        
    def transform(self, 
                 game_data: pd.DataFrame,
                 team_stats: pd.DataFrame,
                 play_data: Optional[pd.DataFrame] = None,
                 betting_data: Optional[pd.DataFrame] = None,
                 weather_data: Optional[pd.DataFrame] = None,
                 schedule_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform raw data into features for model training
        
        Args:
            game_data: Basic game information
            team_stats: Team statistics
            play_data: Play-by-play data (optional)
            betting_data: Betting market data (optional)
            weather_data: Weather conditions (optional)
            schedule_data: Schedule information (optional)
        
        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=game_data.index)
        
        # DVOA features
        if not team_stats.empty:
            league_stats = team_stats.groupby('season').mean()
            dvoa_features = self.dvoa_calculator.calculate_dvoa(
                team_stats, 
                team_stats,  # Use same for opponent in this example
                league_stats
            )
            features = pd.concat([features, dvoa_features], axis=1)
        
        # EPA features
        if play_data is not None and not play_data.empty:
            epa_features = self.epa_calculator.calculate_epa(play_data)
            features = pd.concat([features, epa_features], axis=1)
        
        # Situational features
        if schedule_data is not None:
            situational_features = self.situational_extractor.extract_features(
                game_data, schedule_data, weather_data
            )
            features = pd.concat([features, situational_features], axis=1)
        
        # Market intelligence features
        if betting_data is not None and not betting_data.empty:
            market_features = self.market_extractor.extract_features(betting_data)
            features = pd.concat([features, market_features], axis=1)
        
        # Add rolling averages
        features = self._add_rolling_features(features, team_stats)
        
        # Add interaction features
        features = self._add_interaction_features(features)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        return features
    
    def _add_rolling_features(self, features: pd.DataFrame, 
                             team_stats: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features"""
        rolling_features = pd.DataFrame(index=features.index)
        
        # Define columns for rolling calculations
        stat_columns = ['points_scored', 'points_allowed', 'total_yards', 
                       'yards_allowed', 'turnovers', 'turnovers_forced']
        
        for col in stat_columns:
            if col in team_stats.columns:
                for window in [self.config.short_window, 
                              self.config.medium_window, 
                              self.config.long_window]:
                    rolling_features[f'{col}_last_{window}'] = (
                        team_stats.groupby('team')[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
        
        return pd.concat([features, rolling_features], axis=1)
    
    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different feature groups"""
        interaction_features = pd.DataFrame(index=features.index)
        
        # DVOA and situational interactions
        if 'offensive_dvoa' in features.columns and 'is_divisional' in features.columns:
            interaction_features['dvoa_divisional'] = (
                features['offensive_dvoa'] * features['is_divisional']
            )
        
        # EPA and weather interactions
        if 'epa_per_play' in features.columns and 'weather_impact' in features.columns:
            interaction_features['epa_weather'] = (
                features['epa_per_play'] * (1 - features.get('weather_impact', 0))
            )
        
        # Market and performance interactions
        if 'sharp_money' in features.columns and 'weighted_dvoa' in features.columns:
            interaction_features['sharp_dvoa_alignment'] = (
                features['sharp_money'] * features['weighted_dvoa']
            )
        
        # Rest and performance
        if 'days_rest' in features.columns and 'defensive_dvoa' in features.columns:
            interaction_features['rest_defense'] = (
                features['days_rest'] * features['defensive_dvoa']
            )
        
        return pd.concat([features, interaction_features], axis=1)
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # For numeric features, use median
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if features[col].isnull().any():
                features[col].fillna(features[col].median(), inplace=True)
        
        # For binary features, use mode
        binary_columns = [col for col in features.columns 
                         if features[col].dropna().isin([0, 1]).all()]
        for col in binary_columns:
            if features[col].isnull().any():
                features[col].fillna(features[col].mode().iloc[0] if not features[col].mode().empty else 0, 
                                    inplace=True)
        
        return features
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for importance analysis"""
        return {
            'dvoa': ['offensive_dvoa', 'defensive_dvoa', 'special_teams_dvoa', 
                    'weighted_dvoa', 'opponent_adjusted_dvoa'],
            'epa': ['epa_per_play', 'weighted_epa', 'early_down_epa', 
                   'third_down_epa', 'redzone_epa', 'success_rate'],
            'situational': ['days_rest', 'is_divisional', 'is_primetime', 
                          'weather_impact', 'travel_distance'],
            'market': ['sharp_money', 'reverse_line_movement', 'spread_movement',
                      'money_bet_divergence', 'closing_line_value']
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data
    n_games = 100
    
    # Game data
    game_data = pd.DataFrame({
        'game_id': range(n_games),
        'week': np.random.randint(1, 18, n_games),
        'is_divisional': np.random.choice([0, 1], n_games),
        'is_conference': np.random.choice([0, 1], n_games),
        'is_primetime': np.random.choice([0, 1], n_games, p=[0.8, 0.2])
    })
    
    # Team stats
    team_stats = pd.DataFrame({
        'game_id': range(n_games),
        'team': ['Team_' + str(i % 32) for i in range(n_games)],
        'season': 2024,
        'successful_plays': np.random.randint(20, 40, n_games),
        'total_plays': np.random.randint(60, 80, n_games),
        'total_yards': np.random.randint(250, 450, n_games),
        'points_scored': np.random.randint(10, 40, n_games),
        'offensive_drives': np.random.randint(10, 15, n_games),
        'successful_plays_allowed': np.random.randint(20, 40, n_games),
        'defensive_plays': np.random.randint(60, 80, n_games),
        'yards_allowed': np.random.randint(250, 450, n_games),
        'points_allowed': np.random.randint(10, 40, n_games),
        'defensive_drives': np.random.randint(10, 15, n_games),
        'field_goals_made': np.random.randint(0, 4, n_games),
        'field_goals_attempted': np.random.randint(0, 5, n_games),
        'avg_starting_field_position': np.random.randint(20, 35, n_games),
        'total_return_yards': np.random.randint(20, 150, n_games),
        'total_returns': np.random.randint(3, 8, n_games)
    })
    
    # Betting data
    betting_data = pd.DataFrame({
        'game_id': range(n_games),
        'opening_spread': np.random.uniform(-14, 14, n_games),
        'current_spread': np.random.uniform(-14, 14, n_games),
        'opening_total': np.random.uniform(38, 54, n_games),
        'current_total': np.random.uniform(38, 54, n_games),
        'bet_percentage': np.random.uniform(30, 70, n_games),
        'money_percentage': np.random.uniform(30, 70, n_games),
        'public_bet_percentage': np.random.uniform(40, 80, n_games),
        'is_favorite': np.random.choice([0, 1], n_games),
        'spread': np.random.uniform(-14, 14, n_games),
        'bet_volume': np.random.uniform(1000, 10000, n_games),
        'avg_bet_volume': 5000,
        'sharp_action': np.random.choice([-1, 0, 1], n_games)
    })
    
    # Weather data  
    weather_data = pd.DataFrame({
        'game_id': range(n_games),
        'temperature': np.random.uniform(20, 90, n_games),
        'wind_speed': np.random.uniform(0, 30, n_games),
        'precipitation': np.random.uniform(0, 1, n_games),
        'is_dome': np.random.choice([0, 1], n_games, p=[0.7, 0.3])
    })
    
    # Schedule data
    schedule_data = pd.DataFrame({
        'game_id': range(n_games),
        'game_date': pd.date_range('2024-09-01', periods=n_games, freq='D'),
        'prev_game_date': pd.date_range('2024-08-25', periods=n_games, freq='D'),
        'is_home': np.random.choice([True, False], n_games),
        'prev_is_home': np.random.choice([True, False], n_games),
        'games_last_14_days': np.random.randint(1, 3, n_games),
        'games_last_21_days': np.random.randint(2, 4, n_games)
    })
    
    # Initialize feature engineering
    feature_engineer = NFLFeatureEngineering()
    
    # Transform data
    features = feature_engineer.transform(
        game_data=game_data,
        team_stats=team_stats,
        betting_data=betting_data,
        weather_data=weather_data,
        schedule_data=schedule_data
    )
    
    print("Feature Engineering Complete!")
    print(f"Number of features: {len(features.columns)}")
    print(f"Feature groups: {feature_engineer.get_feature_importance_groups().keys()}")
    print("\nSample features:")
    print(features.head())
    print("\nFeature statistics:")
    print(features.describe())
