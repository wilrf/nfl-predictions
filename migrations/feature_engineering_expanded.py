#!/usr/bin/env python3
"""
Expanded Feature Engineering Script
Creates 25+ features including NGS data, injury status, and advanced metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from supabase import create_client, Client
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedFeatureEngineer:
    """Engineer expanded features for NFL prediction system"""
    
    def __init__(self):
        """Initialize feature engineer with Supabase connection"""
        self.supabase_url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1MDQyMDUsImV4cCI6MjA3NDA4MDIwNX0.8rBzK19aRnuJb7gLdLDR3aZPg-rzqW0usiXb354N0t0"
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        self.feature_stats = {
            'games_processed': 0,
            'features_created': 0,
            'errors': 0
        }
    
    def create_expanded_features(self, start_season: int = 2023, end_season: int = 2024) -> pd.DataFrame:
        """
        Create expanded feature set for games
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            
        Returns:
            DataFrame with expanded features
        """
        logger.info(f"Creating expanded features for seasons {start_season}-{end_season}")
        
        try:
            # Get base game data
            games_df = self._get_base_game_data(start_season, end_season)
            logger.info(f"Found {len(games_df)} games")
            
            # Create expanded features
            expanded_features = []
            
            for _, game in games_df.iterrows():
                try:
                    features = self._create_game_features(game)
                    expanded_features.append(features)
                    self.feature_stats['games_processed'] += 1
                    
                    if self.feature_stats['games_processed'] % 100 == 0:
                        logger.info(f"Processed {self.feature_stats['games_processed']} games")
                        
                except Exception as e:
                    logger.error(f"Error processing game {game['game_id']}: {str(e)}")
                    self.feature_stats['errors'] += 1
            
            # Convert to DataFrame
            features_df = pd.DataFrame(expanded_features)
            self.feature_stats['features_created'] = len(features_df)
            
            logger.info(f"Created {len(features_df)} feature records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating expanded features: {str(e)}")
            return pd.DataFrame()
    
    def _get_base_game_data(self, start_season: int, end_season: int) -> pd.DataFrame:
        """Get base game data from Supabase"""
        try:
            result = self.supabase.table('fact_games').select(
                'game_id,season,week,home_team,away_team,home_score,away_score,'
                'spread_line,total_line,home_rest,away_rest,div_game,roof,surface,'
                'temp,wind,humidity,home_qb_id,away_qb_id,home_qb_name,away_qb_name'
            ).gte('season', start_season).lte('season', end_season).execute()
            
            return pd.DataFrame(result.data)
            
        except Exception as e:
            logger.error(f"Error getting base game data: {str(e)}")
            return pd.DataFrame()
    
    def _create_game_features(self, game: pd.Series) -> Dict:
        """Create expanded features for a single game"""
        features = {
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_score': game['home_score'],
            'away_score': game['away_score'],
            'spread_line': game['spread_line'],
            'total_line': game['total_line']
        }
        
        # Basic features (existing)
        features.update(self._create_basic_features(game))
        
        # NGS features (new)
        features.update(self._create_ngs_features(game))
        
        # Injury features (new)
        features.update(self._create_injury_features(game))
        
        # Advanced features (new)
        features.update(self._create_advanced_features(game))
        
        return features
    
    def _create_basic_features(self, game: pd.Series) -> Dict:
        """Create basic features"""
        features = {}
        
        # Home field advantage
        features['is_home'] = 1  # Always 1 for home team perspective
        
        # Week number
        features['week_number'] = game['week']
        
        # Divisional game
        features['is_divisional'] = 1 if game['div_game'] == 1 else 0
        
        # Stadium type
        features['is_outdoor'] = 1 if game['roof'] == 'outdoors' else 0
        
        # Weather features
        features['temperature'] = game['temp'] if pd.notna(game['temp']) else 70
        features['wind_speed'] = game['wind'] if pd.notna(game['wind']) else 0
        features['humidity'] = game['humidity'] if pd.notna(game['humidity']) else 50
        
        # Rest days
        features['home_rest_days'] = game['home_rest'] if pd.notna(game['home_rest']) else 7
        features['away_rest_days'] = game['away_rest'] if pd.notna(game['away_rest']) else 7
        
        return features
    
    def _create_ngs_features(self, game: pd.Series) -> Dict:
        """Create Next Gen Stats features"""
        features = {}
        
        try:
            # Get NGS data for the game
            passing_result = self.supabase.table('fact_ngs_passing').select('*').eq('game_id', game['game_id']).execute()
            rushing_result = self.supabase.table('fact_ngs_rushing').select('*').eq('game_id', game['game_id']).execute()
            receiving_result = self.supabase.table('fact_ngs_receiving').select('*').eq('game_id', game['game_id']).execute()
            
            # Home team NGS features
            home_passing = [r for r in passing_result.data if r['team'] == game['home_team']]
            home_rushing = [r for r in rushing_result.data if r['team'] == game['home_team']]
            home_receiving = [r for r in receiving_result.data if r['team'] == game['home_team']]
            
            # Away team NGS features
            away_passing = [r for r in passing_result.data if r['team'] == game['away_team']]
            away_rushing = [r for r in rushing_result.data if r['team'] == game['away_team']]
            away_receiving = [r for r in receiving_result.data if r['team'] == game['away_team']]
            
            # Calculate team averages
            features.update(self._calculate_team_ngs_averages(home_passing, home_rushing, home_receiving, 'home'))
            features.update(self._calculate_team_ngs_averages(away_passing, away_rushing, away_receiving, 'away'))
            
        except Exception as e:
            logger.error(f"Error creating NGS features for {game['game_id']}: {str(e)}")
            # Set default values
            features.update(self._get_default_ngs_features())
        
        return features
    
    def _calculate_team_ngs_averages(self, passing: List, rushing: List, receiving: List, team_prefix: str) -> Dict:
        """Calculate team NGS averages"""
        features = {}
        
        # Passing features
        if passing:
            features[f'{team_prefix}_cpoe'] = np.mean([p['completion_percentage_above_expectation'] for p in passing if p['completion_percentage_above_expectation'] is not None])
            features[f'{team_prefix}_avg_time_to_throw'] = np.mean([p['avg_time_to_throw'] for p in passing if p['avg_time_to_throw'] is not None])
            features[f'{team_prefix}_aggressiveness'] = np.mean([p['aggressiveness'] for p in passing if p['aggressiveness'] is not None])
        else:
            features[f'{team_prefix}_cpoe'] = 0
            features[f'{team_prefix}_avg_time_to_throw'] = 2.5
            features[f'{team_prefix}_aggressiveness'] = 0
        
        # Rushing features
        if rushing:
            features[f'{team_prefix}_rush_efficiency'] = np.mean([r['efficiency'] for r in rushing if r['efficiency'] is not None])
            features[f'{team_prefix}_rush_yards_over_expected'] = np.mean([r['rush_yards_over_expected'] for r in rushing if r['rush_yards_over_expected'] is not None])
        else:
            features[f'{team_prefix}_rush_efficiency'] = 0
            features[f'{team_prefix}_rush_yards_over_expected'] = 0
        
        # Receiving features
        if receiving:
            features[f'{team_prefix}_avg_separation'] = np.mean([r['avg_separation'] for r in receiving if r['avg_separation'] is not None])
            features[f'{team_prefix}_avg_cushion'] = np.mean([r['avg_cushion'] for r in receiving if r['avg_cushion'] is not None])
        else:
            features[f'{team_prefix}_avg_separation'] = 2.5
            features[f'{team_prefix}_avg_cushion'] = 7.0
        
        return features
    
    def _get_default_ngs_features(self) -> Dict:
        """Get default NGS feature values"""
        return {
            'home_cpoe': 0, 'away_cpoe': 0,
            'home_avg_time_to_throw': 2.5, 'away_avg_time_to_throw': 2.5,
            'home_aggressiveness': 0, 'away_aggressiveness': 0,
            'home_rush_efficiency': 0, 'away_rush_efficiency': 0,
            'home_rush_yards_over_expected': 0, 'away_rush_yards_over_expected': 0,
            'home_avg_separation': 2.5, 'away_avg_separation': 2.5,
            'home_avg_cushion': 7.0, 'away_avg_cushion': 7.0
        }
    
    def _create_injury_features(self, game: pd.Series) -> Dict:
        """Create injury-based features"""
        features = {}
        
        try:
            # Get injury data for the week
            injury_result = self.supabase.table('fact_injuries').select('*').eq('season', game['season']).eq('week', game['week']).execute()
            
            # Home team injuries
            home_injuries = [i for i in injury_result.data if i['team'] == game['home_team']]
            away_injuries = [i for i in injury_result.data if i['team'] == game['away_team']]
            
            # Calculate injury impact
            features.update(self._calculate_injury_impact(home_injuries, 'home'))
            features.update(self._calculate_injury_impact(away_injuries, 'away'))
            
        except Exception as e:
            logger.error(f"Error creating injury features for {game['game_id']}: {str(e)}")
            # Set default values
            features.update(self._get_default_injury_features())
        
        return features
    
    def _calculate_injury_impact(self, injuries: List, team_prefix: str) -> Dict:
        """Calculate injury impact features"""
        features = {}
        
        if injuries:
            # QB status (most important)
            qb_injuries = [i for i in injuries if i['position'] == 'QB']
            if qb_injuries:
                qb_status = qb_injuries[0]['report_status']
                if qb_status == 'Out':
                    features[f'{team_prefix}_qb_status'] = 0  # Out
                elif qb_status == 'Doubtful':
                    features[f'{team_prefix}_qb_status'] = 0.25
                elif qb_status == 'Questionable':
                    features[f'{team_prefix}_qb_status'] = 0.5
                elif qb_status == 'Probable':
                    features[f'{team_prefix}_qb_status'] = 0.75
                else:
                    features[f'{team_prefix}_qb_status'] = 1.0  # Healthy
            else:
                features[f'{team_prefix}_qb_status'] = 1.0  # No QB injuries
            
            # Key position injuries (RB, WR, TE, OL)
            key_positions = ['RB', 'WR', 'TE', 'OL', 'T', 'G', 'C']
            key_injuries = [i for i in injuries if i['position'] in key_positions and i['severity_score'] >= 2]
            features[f'{team_prefix}_key_injuries'] = len(key_injuries)
            
            # Total injury count
            features[f'{team_prefix}_total_injuries'] = len(injuries)
            
            # Average severity
            features[f'{team_prefix}_avg_injury_severity'] = np.mean([i['severity_score'] for i in injuries if i['severity_score'] is not None])
        else:
            features[f'{team_prefix}_qb_status'] = 1.0
            features[f'{team_prefix}_key_injuries'] = 0
            features[f'{team_prefix}_total_injuries'] = 0
            features[f'{team_prefix}_avg_injury_severity'] = 0
        
        return features
    
    def _get_default_injury_features(self) -> Dict:
        """Get default injury feature values"""
        return {
            'home_qb_status': 1.0, 'away_qb_status': 1.0,
            'home_key_injuries': 0, 'away_key_injuries': 0,
            'home_total_injuries': 0, 'away_total_injuries': 0,
            'home_avg_injury_severity': 0, 'away_avg_injury_severity': 0
        }
    
    def _create_advanced_features(self, game: pd.Series) -> Dict:
        """Create advanced features"""
        features = {}
        
        # Rest advantage
        home_rest = game['home_rest'] if pd.notna(game['home_rest']) else 7
        away_rest = game['away_rest'] if pd.notna(game['away_rest']) else 7
        features['rest_advantage'] = home_rest - away_rest
        
        # Weather impact score
        temp = game['temp'] if pd.notna(game['temp']) else 70
        wind = game['wind'] if pd.notna(game['wind']) else 0
        is_outdoor = 1 if game['roof'] == 'outdoors' else 0
        
        # Weather impact (higher is worse for offense)
        weather_impact = 0
        if is_outdoor:
            if temp < 32:  # Freezing
                weather_impact += 2
            elif temp > 85:  # Hot
                weather_impact += 1
            if wind > 15:  # High wind
                weather_impact += 2
            elif wind > 10:  # Moderate wind
                weather_impact += 1
        
        features['weather_impact_score'] = weather_impact
        
        # Pressure rate (placeholder - would need more NGS data)
        features['home_pressure_rate'] = 0.3  # Default
        features['away_pressure_rate'] = 0.3  # Default
        
        # SOS-adjusted EPA (placeholder - would need historical data)
        features['sos_adjusted_epa'] = 0  # Default
        
        # Neutral script EPA (placeholder)
        features['neutral_script_epa'] = 0  # Default
        
        # Explosive play rate (placeholder)
        features['explosive_play_rate'] = 0.15  # Default
        
        # Red zone efficiency (placeholder)
        features['red_zone_efficiency'] = 0.6  # Default
        
        return features
    
    def save_features(self, features_df: pd.DataFrame, table_name: str = 'expanded_game_features'):
        """Save features to Supabase"""
        try:
            if features_df.empty:
                logger.warning("No features to save")
                return
            
            # Convert to records
            records = features_df.to_dict('records')
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                result = self.supabase.table(table_name).upsert(
                    batch,
                    on_conflict='game_id'
                ).execute()
                
                logger.info(f"Saved batch {i//batch_size + 1} ({len(batch)} records)")
            
            logger.info(f"Successfully saved {len(records)} feature records")
            
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
    
    def get_feature_summary(self) -> Dict:
        """Get summary of feature engineering process"""
        return self.feature_stats

def main():
    """Main function to run feature engineering"""
    try:
        # Create feature engineer
        engineer = ExpandedFeatureEngineer()
        
        # Create expanded features
        features_df = engineer.create_expanded_features(start_season=2023, end_season=2024)
        
        if not features_df.empty:
            # Save features
            engineer.save_features(features_df)
            
            # Print summary
            stats = engineer.get_feature_summary()
            print(f"\nFeature Engineering Summary:")
            print(f"Games processed: {stats['games_processed']}")
            print(f"Features created: {stats['features_created']}")
            print(f"Errors: {stats['errors']}")
            print(f"Feature columns: {len(features_df.columns)}")
            
            # Show sample features
            print(f"\nSample features:")
            print(features_df.head(2).to_dict('records'))
            
        else:
            print("No features created")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        return None

if __name__ == "__main__":
    main()
