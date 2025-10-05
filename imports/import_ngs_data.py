#!/usr/bin/env python3
"""
Next Gen Stats Data Import Script
Imports 24,814 NGS records into Supabase for NFL prediction system
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from supabase import create_client, Client
from dotenv import load_dotenv

# Add the improved_nfl_system to path
sys.path.append('/Users/wilfowler/Sports Model/improved_nfl_system')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/wilfowler/Sports Model/logs/ngs_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NGSDataImporter:
    """Import Next Gen Stats data into Supabase"""
    
    def __init__(self):
        """Initialize the importer with Supabase connection"""
        # Use hardcoded Supabase credentials for now
        self.supabase_url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1MDQyMDUsImV4cCI6MjA3NDA4MDIwNX0.8rBzK19aRnuJb7gLdLDR3aZPg-rzqW0usiXb354N0t0"
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.batch_size = 1000
        self.stats = {
            'passing_records': 0,
            'rushing_records': 0,
            'receiving_records': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def import_all_ngs_data(self, start_season: int = 2016, end_season: int = 2025) -> Dict:
        """
        Import all Next Gen Stats data for specified seasons
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Starting NGS data import for seasons {start_season}-{end_season}")
        
        try:
            # Import each season's data
            for season in range(start_season, end_season + 1):
                logger.info(f"Importing NGS data for season {season}")
                self._import_season_ngs_data(season)
            
            # Calculate final statistics
            duration = datetime.now() - self.stats['start_time']
            self.stats['duration'] = str(duration)
            self.stats['total_records'] = (
                self.stats['passing_records'] + 
                self.stats['rushing_records'] + 
                self.stats['receiving_records']
            )
            
            logger.info(f"NGS import completed successfully!")
            logger.info(f"Total records imported: {self.stats['total_records']}")
            logger.info(f"Duration: {duration}")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error during NGS import: {str(e)}")
            self.stats['errors'] += 1
            raise
    
    def _import_season_ngs_data(self, season: int):
        """Import NGS data for a specific season"""
        try:
            # Import passing data
            passing_data = self._fetch_ngs_passing_data(season)
            if not passing_data.empty:
                self._insert_ngs_passing_data(passing_data)
                self.stats['passing_records'] += len(passing_data)
                logger.info(f"Imported {len(passing_data)} passing records for {season}")
            
            # Import rushing data
            rushing_data = self._fetch_ngs_rushing_data(season)
            if not rushing_data.empty:
                self._insert_ngs_rushing_data(rushing_data)
                self.stats['rushing_records'] += len(rushing_data)
                logger.info(f"Imported {len(rushing_data)} rushing records for {season}")
            
            # Import receiving data
            receiving_data = self._fetch_ngs_receiving_data(season)
            if not receiving_data.empty:
                self._insert_ngs_receiving_data(receiving_data)
                self.stats['receiving_records'] += len(receiving_data)
                logger.info(f"Imported {len(receiving_data)} receiving records for {season}")
                
        except Exception as e:
            logger.error(f"Error importing NGS data for season {season}: {str(e)}")
            self.stats['errors'] += 1
    
    def _fetch_ngs_passing_data(self, season: int) -> pd.DataFrame:
        """Fetch NGS passing data for a season"""
        try:
            # Import nfl_data_py for data fetching
            import nfl_data_py as nfl
            
            # Fetch NGS passing stats
            passing_stats = nfl.import_ngs_data(
                stat_type='passing',
                years=[season]
            )
            
            if passing_stats.empty:
                logger.warning(f"No passing NGS data found for season {season}")
                return pd.DataFrame()
            
            # Clean and prepare data
            passing_data = self._clean_ngs_passing_data(passing_stats, season)
            return passing_data
            
        except Exception as e:
            logger.error(f"Error fetching passing NGS data for season {season}: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_ngs_rushing_data(self, season: int) -> pd.DataFrame:
        """Fetch NGS rushing data for a season"""
        try:
            import nfl_data_py as nfl
            
            # Fetch NGS rushing stats
            rushing_stats = nfl.import_ngs_data(
                stat_type='rushing',
                years=[season]
            )
            
            if rushing_stats.empty:
                logger.warning(f"No rushing NGS data found for season {season}")
                return pd.DataFrame()
            
            # Clean and prepare data
            rushing_data = self._clean_ngs_rushing_data(rushing_stats, season)
            return rushing_data
            
        except Exception as e:
            logger.error(f"Error fetching rushing NGS data for season {season}: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_ngs_receiving_data(self, season: int) -> pd.DataFrame:
        """Fetch NGS receiving data for a season"""
        try:
            import nfl_data_py as nfl
            
            # Fetch NGS receiving stats
            receiving_stats = nfl.import_ngs_data(
                stat_type='receiving',
                years=[season]
            )
            
            if receiving_stats.empty:
                logger.warning(f"No receiving NGS data found for season {season}")
                return pd.DataFrame()
            
            # Clean and prepare data
            receiving_data = self._clean_ngs_receiving_data(receiving_stats, season)
            return receiving_data
            
        except Exception as e:
            logger.error(f"Error fetching receiving NGS data for season {season}: {str(e)}")
            return pd.DataFrame()
    
    def _clean_ngs_passing_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Clean and prepare NGS passing data for insertion"""
        try:
            # Select relevant columns and rename to match schema
            columns_mapping = {
                'player_gsis_id': 'player_id',
                'season': 'season',
                'week': 'week',
                'team_abbr': 'team',
                'avg_time_to_throw': 'avg_time_to_throw',
                'completion_percentage_above_expectation': 'completion_percentage_above_expectation',
                'aggressiveness': 'aggressiveness',
                'avg_air_yards_to_sticks': 'avg_air_yards_to_sticks',
                'avg_completed_air_yards': 'avg_completed_air_yards',
                'avg_intended_air_yards': 'avg_intended_air_yards',
                'avg_air_yards_differential': 'avg_air_yards_differential',
                'completion_percentage': 'completion_percentage',
                'expected_completion_percentage': 'expected_completion_percentage',
                'avg_air_time': 'avg_air_time',
                'max_air_time': 'max_air_time',
                'avg_sack_time': 'avg_sack_time',
                'max_sack_time': 'max_sack_time',
                'blitz_pct': 'blitz_pct',
                'hurry_pct': 'hurry_pct',
                'knockdown_pct': 'knockdown_pct',
                'throw_away_pct': 'throw_away_pct',
                'batted_pass_pct': 'batted_pass_pct',
                'on_target_throw_pct': 'on_target_throw_pct',
                'off_target_throw_pct': 'off_target_throw_pct',
                'dropped_pct': 'dropped_pct',
                'pickable_pct': 'pickable_pct',
                'pocket_time': 'pocket_time',
                'time_to_pressure': 'time_to_pressure'
            }
            
            # Filter to only include columns that exist in the dataframe
            available_columns = {k: v for k, v in columns_mapping.items() if k in df.columns}
            
            # Select and rename columns
            cleaned_df = df[list(available_columns.keys())].copy()
            cleaned_df = cleaned_df.rename(columns=available_columns)
            
            # Add season if not present
            if 'season' not in cleaned_df.columns:
                cleaned_df['season'] = season
            
            # Handle missing values
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
            
            # Convert data types
            cleaned_df['player_id'] = cleaned_df['player_id'].astype(str)
            cleaned_df['game_id'] = cleaned_df['game_id'].astype(str)
            cleaned_df['team'] = cleaned_df['team'].astype(str)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning passing NGS data: {str(e)}")
            return pd.DataFrame()
    
    def _clean_ngs_rushing_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Clean and prepare NGS rushing data for insertion"""
        try:
            # Select relevant columns and rename to match schema
            columns_mapping = {
                'player_id': 'player_id',
                'game_id': 'game_id',
                'season': 'season',
                'week': 'week',
                'team': 'team',
                'efficiency': 'efficiency',
                'rush_yards_over_expected': 'rush_yards_over_expected',
                'percent_attempts_gte_eight_defenders': 'percent_attempts_gte_eight_defenders',
                'avg_time_to_los': 'avg_time_to_los',
                'avg_defenders_in_box': 'avg_defenders_in_box',
                'rush_attempts': 'rush_attempts',
                'rush_yards': 'rush_yards',
                'rush_touchdowns': 'rush_touchdowns',
                'rush_first_downs': 'rush_first_downs',
                'rush_broken_tackles': 'rush_broken_tackles',
                'rush_yards_before_contact': 'rush_yards_before_contact',
                'rush_yards_after_contact': 'rush_yards_after_contact'
            }
            
            # Filter to only include columns that exist in the dataframe
            available_columns = {k: v for k, v in columns_mapping.items() if k in df.columns}
            
            # Select and rename columns
            cleaned_df = df[list(available_columns.keys())].copy()
            cleaned_df = cleaned_df.rename(columns=available_columns)
            
            # Add season if not present
            if 'season' not in cleaned_df.columns:
                cleaned_df['season'] = season
            
            # Handle missing values
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
            
            # Convert data types
            cleaned_df['player_id'] = cleaned_df['player_id'].astype(str)
            cleaned_df['game_id'] = cleaned_df['game_id'].astype(str)
            cleaned_df['team'] = cleaned_df['team'].astype(str)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning rushing NGS data: {str(e)}")
            return pd.DataFrame()
    
    def _clean_ngs_receiving_data(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        """Clean and prepare NGS receiving data for insertion"""
        try:
            # Select relevant columns and rename to match schema
            columns_mapping = {
                'player_id': 'player_id',
                'game_id': 'game_id',
                'season': 'season',
                'week': 'week',
                'team': 'team',
                'avg_separation': 'avg_separation',
                'avg_cushion': 'avg_cushion',
                'avg_yac_above_expectation': 'avg_yac_above_expectation',
                'avg_target_separation': 'avg_target_separation',
                'avg_caught_separation': 'avg_caught_separation',
                'avg_intended_air_yards': 'avg_intended_air_yards',
                'avg_completed_air_yards': 'avg_completed_air_yards',
                'avg_yac': 'avg_yac',
                'avg_yac_over_expected': 'avg_yac_over_expected',
                'catch_percentage': 'catch_percentage',
                'drop_rate': 'drop_rate',
                'avg_target_depth': 'avg_target_depth',
                'avg_reception_depth': 'avg_reception_depth'
            }
            
            # Filter to only include columns that exist in the dataframe
            available_columns = {k: v for k, v in columns_mapping.items() if k in df.columns}
            
            # Select and rename columns
            cleaned_df = df[list(available_columns.keys())].copy()
            cleaned_df = cleaned_df.rename(columns=available_columns)
            
            # Add season if not present
            if 'season' not in cleaned_df.columns:
                cleaned_df['season'] = season
            
            # Handle missing values
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
            
            # Convert data types
            cleaned_df['player_id'] = cleaned_df['player_id'].astype(str)
            cleaned_df['game_id'] = cleaned_df['game_id'].astype(str)
            cleaned_df['team'] = cleaned_df['team'].astype(str)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning receiving NGS data: {str(e)}")
            return pd.DataFrame()
    
    def _insert_ngs_passing_data(self, df: pd.DataFrame):
        """Insert NGS passing data into Supabase"""
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Insert in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                # Use upsert to handle duplicates
                result = self.supabase.table('fact_ngs_passing').upsert(
                    batch,
                    on_conflict='player_id,game_id'
                ).execute()
                
                logger.info(f"Inserted batch {i//self.batch_size + 1} of passing data ({len(batch)} records)")
                
        except Exception as e:
            logger.error(f"Error inserting passing NGS data: {str(e)}")
            self.stats['errors'] += 1
    
    def _insert_ngs_rushing_data(self, df: pd.DataFrame):
        """Insert NGS rushing data into Supabase"""
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Insert in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                # Use upsert to handle duplicates
                result = self.supabase.table('fact_ngs_rushing').upsert(
                    batch,
                    on_conflict='player_id,game_id'
                ).execute()
                
                logger.info(f"Inserted batch {i//self.batch_size + 1} of rushing data ({len(batch)} records)")
                
        except Exception as e:
            logger.error(f"Error inserting rushing NGS data: {str(e)}")
            self.stats['errors'] += 1
    
    def _insert_ngs_receiving_data(self, df: pd.DataFrame):
        """Insert NGS receiving data into Supabase"""
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Insert in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                # Use upsert to handle duplicates
                result = self.supabase.table('fact_ngs_receiving').upsert(
                    batch,
                    on_conflict='player_id,game_id'
                ).execute()
                
                logger.info(f"Inserted batch {i//self.batch_size + 1} of receiving data ({len(batch)} records)")
                
        except Exception as e:
            logger.error(f"Error inserting receiving NGS data: {str(e)}")
            self.stats['errors'] += 1
    
    def validate_import(self) -> Dict:
        """Validate the imported NGS data"""
        try:
            validation_results = {}
            
            # Check passing data
            passing_count = self.supabase.table('fact_ngs_passing').select('*', count='exact').execute()
            validation_results['passing_records'] = passing_count.count
            
            # Check rushing data
            rushing_count = self.supabase.table('fact_ngs_rushing').select('*', count='exact').execute()
            validation_results['rushing_records'] = rushing_count.count
            
            # Check receiving data
            receiving_count = self.supabase.table('fact_ngs_receiving').select('*', count='exact').execute()
            validation_results['receiving_records'] = receiving_count.count
            
            validation_results['total_records'] = (
                validation_results['passing_records'] + 
                validation_results['rushing_records'] + 
                validation_results['receiving_records']
            )
            
            logger.info(f"Validation results: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating import: {str(e)}")
            return {}

def main():
    """Main function to run the NGS import"""
    try:
        # Create importer instance
        importer = NGSDataImporter()
        
        # Import all NGS data
        results = importer.import_all_ngs_data(start_season=2016, end_season=2025)
        
        # Validate import
        validation = importer.validate_import()
        
        # Print summary
        print("\n" + "="*50)
        print("NGS DATA IMPORT SUMMARY")
        print("="*50)
        print(f"Total records imported: {results['total_records']}")
        print(f"Passing records: {results['passing_records']}")
        print(f"Rushing records: {results['rushing_records']}")
        print(f"Receiving records: {results['receiving_records']}")
        print(f"Errors: {results['errors']}")
        print(f"Duration: {results['duration']}")
        print("="*50)
        
        # Save results to file
        import json
        with open('/Users/wilfowler/Sports Model/logs/ngs_import_results.json', 'w') as f:
            json.dump({
                'import_results': results,
                'validation_results': validation,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        return None

if __name__ == "__main__":
    main()
