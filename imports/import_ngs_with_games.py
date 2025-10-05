#!/usr/bin/env python3
"""
NGS Import with proper game_id mapping
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_ngs_with_games():
    """Import NGS data with proper game_id mapping"""
    
    # Supabase connection
    supabase_url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1MDQyMDUsImV4cCI6MjA3NDA4MDIwNX0.8rBzK19aRnuJb7gLdLDR3aZPg-rzqW0usiXb354N0t0"
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Import nfl_data_py
    import nfl_data_py as nfl
    
    stats = {'passing': 0, 'rushing': 0, 'receiving': 0}
    
    # Get existing games to map to
    games_result = supabase.table('fact_games').select('game_id,season,week,home_team,away_team').execute()
    games_df = pd.DataFrame(games_result.data)
    
    # Import data for recent seasons
    for season in [2023, 2024]:
        logger.info(f"Importing NGS data for season {season}")
        
        # Get games for this season
        season_games = games_df[games_df['season'] == season].copy()
        
        # Passing data
        try:
            passing_df = nfl.import_ngs_data('passing', [season])
            if not passing_df.empty:
                # Clean and prepare passing data
                passing_clean = passing_df.copy()
                passing_clean['player_id'] = passing_clean['player_gsis_id']
                passing_clean['team'] = passing_clean['team_abbr']
                
                # Map to actual game_ids
                passing_with_games = []
                for _, row in passing_clean.iterrows():
                    # Find matching game
                    matching_games = season_games[
                        (season_games['week'] == row['week']) & 
                        ((season_games['home_team'] == row['team']) | (season_games['away_team'] == row['team']))
                    ]
                    
                    if not matching_games.empty:
                        game_id = matching_games.iloc[0]['game_id']
                        
                        # Select only columns that exist in our schema
                        passing_record = {
                            'player_id': str(row['player_id']),
                            'game_id': game_id,
                            'season': int(row['season']),
                            'week': int(row['week']),
                            'team': str(row['team']),
                            'avg_time_to_throw': float(row['avg_time_to_throw']) if pd.notna(row['avg_time_to_throw']) else None,
                            'completion_percentage_above_expectation': float(row['completion_percentage_above_expectation']) if pd.notna(row['completion_percentage_above_expectation']) else None,
                            'aggressiveness': float(row['aggressiveness']) if pd.notna(row['aggressiveness']) else None,
                            'avg_air_yards_to_sticks': float(row['avg_air_yards_to_sticks']) if pd.notna(row['avg_air_yards_to_sticks']) else None,
                            'avg_completed_air_yards': float(row['avg_completed_air_yards']) if pd.notna(row['avg_completed_air_yards']) else None,
                            'avg_intended_air_yards': float(row['avg_intended_air_yards']) if pd.notna(row['avg_intended_air_yards']) else None,
                            'completion_percentage': float(row['completion_percentage']) if pd.notna(row['completion_percentage']) else None,
                            'expected_completion_percentage': float(row['expected_completion_percentage']) if pd.notna(row['expected_completion_percentage']) else None
                        }
                        
                        # Filter out None values and infinite values
                        passing_record = {k: v for k, v in passing_record.items() 
                                        if v is not None and not (isinstance(v, float) and (np.isinf(v) or np.isnan(v)))}
                        
                        passing_with_games.append(passing_record)
                
                if passing_with_games:
                    # Insert into Supabase
                    result = supabase.table('fact_ngs_passing').upsert(passing_with_games, on_conflict='player_id,game_id').execute()
                    stats['passing'] += len(passing_with_games)
                    logger.info(f"Inserted {len(passing_with_games)} passing records for {season}")
                
        except Exception as e:
            logger.error(f"Error with passing data for {season}: {str(e)}")
        
        # Rushing data
        try:
            rushing_df = nfl.import_ngs_data('rushing', [season])
            if not rushing_df.empty:
                # Clean and prepare rushing data
                rushing_clean = rushing_df.copy()
                rushing_clean['player_id'] = rushing_clean['player_gsis_id']
                rushing_clean['team'] = rushing_clean['team_abbr']
                
                # Map to actual game_ids
                rushing_with_games = []
                for _, row in rushing_clean.iterrows():
                    # Find matching game
                    matching_games = season_games[
                        (season_games['week'] == row['week']) & 
                        ((season_games['home_team'] == row['team']) | (season_games['away_team'] == row['team']))
                    ]
                    
                    if not matching_games.empty:
                        game_id = matching_games.iloc[0]['game_id']
                        
                        # Select only columns that exist in our schema
                        rushing_record = {
                            'player_id': str(row['player_id']),
                            'game_id': game_id,
                            'season': int(row['season']),
                            'week': int(row['week']),
                            'team': str(row['team']),
                            'efficiency': float(row['efficiency']) if pd.notna(row['efficiency']) else None,
                            'rush_yards_over_expected': float(row['rush_yards_over_expected']) if pd.notna(row['rush_yards_over_expected']) else None,
                            'percent_attempts_gte_eight_defenders': float(row['percent_attempts_gte_eight_defenders']) if pd.notna(row['percent_attempts_gte_eight_defenders']) else None
                        }
                        
                        # Filter out None values and infinite values
                        rushing_record = {k: v for k, v in rushing_record.items() 
                                        if v is not None and not (isinstance(v, float) and (np.isinf(v) or np.isnan(v)))}
                        
                        rushing_with_games.append(rushing_record)
                
                if rushing_with_games:
                    # Insert into Supabase
                    result = supabase.table('fact_ngs_rushing').upsert(rushing_with_games, on_conflict='player_id,game_id').execute()
                    stats['rushing'] += len(rushing_with_games)
                    logger.info(f"Inserted {len(rushing_with_games)} rushing records for {season}")
                
        except Exception as e:
            logger.error(f"Error with rushing data for {season}: {str(e)}")
        
        # Receiving data
        try:
            receiving_df = nfl.import_ngs_data('receiving', [season])
            if not receiving_df.empty:
                # Clean and prepare receiving data
                receiving_clean = receiving_df.copy()
                receiving_clean['player_id'] = receiving_clean['player_gsis_id']
                receiving_clean['team'] = receiving_clean['team_abbr']
                
                # Map to actual game_ids
                receiving_with_games = []
                for _, row in receiving_clean.iterrows():
                    # Find matching game
                    matching_games = season_games[
                        (season_games['week'] == row['week']) & 
                        ((season_games['home_team'] == row['team']) | (season_games['away_team'] == row['team']))
                    ]
                    
                    if not matching_games.empty:
                        game_id = matching_games.iloc[0]['game_id']
                        
                        # Select only columns that exist in our schema
                        receiving_record = {
                            'player_id': str(row['player_id']),
                            'game_id': game_id,
                            'season': int(row['season']),
                            'week': int(row['week']),
                            'team': str(row['team']),
                            'avg_separation': float(row['avg_separation']) if pd.notna(row['avg_separation']) else None,
                            'avg_cushion': float(row['avg_cushion']) if pd.notna(row['avg_cushion']) else None,
                            'avg_yac_above_expectation': float(row['avg_yac_above_expectation']) if pd.notna(row['avg_yac_above_expectation']) else None
                        }
                        
                        # Filter out None values and infinite values
                        receiving_record = {k: v for k, v in receiving_record.items() 
                                          if v is not None and not (isinstance(v, float) and (np.isinf(v) or np.isnan(v)))}
                        
                        receiving_with_games.append(receiving_record)
                
                if receiving_with_games:
                    # Insert into Supabase
                    result = supabase.table('fact_ngs_receiving').upsert(receiving_with_games, on_conflict='player_id,game_id').execute()
                    stats['receiving'] += len(receiving_with_games)
                    logger.info(f"Inserted {len(receiving_with_games)} receiving records for {season}")
                
        except Exception as e:
            logger.error(f"Error with receiving data for {season}: {str(e)}")
    
    # Print summary
    total = stats['passing'] + stats['rushing'] + stats['receiving']
    print(f"\nNGS Import Summary:")
    print(f"Passing records: {stats['passing']}")
    print(f"Rushing records: {stats['rushing']}")
    print(f"Receiving records: {stats['receiving']}")
    print(f"Total records: {total}")
    
    return stats

if __name__ == "__main__":
    import_ngs_with_games()
