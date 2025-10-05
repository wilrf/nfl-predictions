#!/usr/bin/env python3
"""
Simple Next Gen Stats Data Import Script
Imports NGS data into Supabase for NFL prediction system
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_ngs_data():
    """Import NGS data for recent seasons"""
    
    # Supabase connection
    supabase_url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1MDQyMDUsImV4cCI6MjA3NDA4MDIwNX0.8rBzK19aRnuJb7gLdLDR3aZPg-rzqW0usiXb354N0t0"
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Import nfl_data_py
    import nfl_data_py as nfl
    
    stats = {'passing': 0, 'rushing': 0, 'receiving': 0}
    
    # Import data for recent seasons
    for season in [2023, 2024]:
        logger.info(f"Importing NGS data for season {season}")
        
        # Passing data
        try:
            passing_df = nfl.import_ngs_data('passing', [season])
            if not passing_df.empty:
                # Clean and prepare passing data
                passing_clean = passing_df.copy()
                passing_clean['player_id'] = passing_clean['player_gsis_id']
                passing_clean['team'] = passing_clean['team_abbr']
                passing_clean['game_id'] = passing_clean['season'].astype(str) + '_' + passing_clean['week'].astype(str) + '_' + passing_clean['team']
                
                # Select only columns that exist in our schema
                passing_cols = ['player_id', 'game_id', 'season', 'week', 'team',
                               'avg_time_to_throw', 'completion_percentage_above_expectation',
                               'aggressiveness', 'avg_air_yards_to_sticks', 'avg_completed_air_yards',
                               'avg_intended_air_yards', 'completion_percentage',
                               'expected_completion_percentage']
                
                passing_final = passing_clean[passing_cols].copy()
                
                # Insert into Supabase
                records = passing_final.to_dict('records')
                result = supabase.table('fact_ngs_passing').upsert(records, on_conflict='player_id,game_id').execute()
                stats['passing'] += len(records)
                logger.info(f"Inserted {len(records)} passing records for {season}")
                
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
                rushing_clean['game_id'] = rushing_clean['season'].astype(str) + '_' + rushing_clean['week'].astype(str) + '_' + rushing_clean['team']
                
                # Select only columns that exist in our schema
                rushing_cols = ['player_id', 'game_id', 'season', 'week', 'team',
                               'efficiency', 'rush_yards_over_expected',
                               'percent_attempts_gte_eight_defenders']
                
                rushing_final = rushing_clean[rushing_cols].copy()
                
                # Insert into Supabase
                records = rushing_final.to_dict('records')
                result = supabase.table('fact_ngs_rushing').upsert(records, on_conflict='player_id,game_id').execute()
                stats['rushing'] += len(records)
                logger.info(f"Inserted {len(records)} rushing records for {season}")
                
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
                receiving_clean['game_id'] = receiving_clean['season'].astype(str) + '_' + receiving_clean['week'].astype(str) + '_' + receiving_clean['team']
                
                # Select only columns that exist in our schema
                receiving_cols = ['player_id', 'game_id', 'season', 'week', 'team',
                                 'avg_separation', 'avg_cushion', 'avg_yac_above_expectation']
                
                receiving_final = receiving_clean[receiving_cols].copy()
                
                # Insert into Supabase
                records = receiving_final.to_dict('records')
                result = supabase.table('fact_ngs_receiving').upsert(records, on_conflict='player_id,game_id').execute()
                stats['receiving'] += len(records)
                logger.info(f"Inserted {len(records)} receiving records for {season}")
                
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
    import_ngs_data()
