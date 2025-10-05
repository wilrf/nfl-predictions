#!/usr/bin/env python3
"""
Injury Data Import Script
Imports injury reports into Supabase for NFL prediction system
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_injury_data():
    """Import injury data for recent seasons"""
    
    # Supabase connection
    supabase_url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1MDQyMDUsImV4cCI6MjA3NDA4MDIwNX0.8rBzK19aRnuJb7gLdLDR3aZPg-rzqW0usiXb354N0t0"
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Import nfl_data_py
    import nfl_data_py as nfl
    
    stats = {'total_records': 0, 'errors': 0}
    
    # Import data for recent seasons
    for season in [2023, 2024]:
        logger.info(f"Importing injury data for season {season}")
        
        try:
            injury_df = nfl.import_injuries([season])
            if not injury_df.empty:
                logger.info(f"Found {len(injury_df)} injury records for {season}")
                
                # Clean and prepare injury data
                injury_clean = injury_df.copy()
                
                # Map columns to our schema
                injury_records = []
                for _, row in injury_clean.iterrows():
                    # Calculate severity score based on status
                    severity_score = 0
                    if pd.notna(row.get('report_status')):
                        status = str(row['report_status']).lower()
                        if 'out' in status:
                            severity_score = 3
                        elif 'doubtful' in status:
                            severity_score = 2
                        elif 'questionable' in status:
                            severity_score = 1
                        elif 'probable' in status:
                            severity_score = 0
                    
                    # Create record
                    injury_record = {
                        'season': int(row['season']) if pd.notna(row['season']) else season,
                        'week': int(row['week']) if pd.notna(row['week']) else None,
                        'game_type': str(row.get('game_type', 'REG')) if pd.notna(row.get('game_type')) else 'REG',
                        'team': str(row['team']) if pd.notna(row['team']) else None,
                        'gsis_id': str(row.get('gsis_id', '')) if pd.notna(row.get('gsis_id')) else None,
                        'player_name': str(row['full_name']) if pd.notna(row['full_name']) else None,
                        'position': str(row.get('position', '')) if pd.notna(row.get('position')) else None,
                        'report_primary_injury': str(row.get('report_primary_injury', '')) if pd.notna(row.get('report_primary_injury')) else None,
                        'report_secondary_injury': str(row.get('report_secondary_injury', '')) if pd.notna(row.get('report_secondary_injury')) else None,
                        'report_status': str(row.get('report_status', '')) if pd.notna(row.get('report_status')) else None,
                        'practice_primary_injury': str(row.get('practice_primary_injury', '')) if pd.notna(row.get('practice_primary_injury')) else None,
                        'practice_secondary_injury': str(row.get('practice_secondary_injury', '')) if pd.notna(row.get('practice_secondary_injury')) else None,
                        'practice_status': str(row.get('practice_status', '')) if pd.notna(row.get('practice_status')) else None,
                        'date_modified': str(row.get('date_modified', '')) if pd.notna(row.get('date_modified')) else None,
                        'severity_score': severity_score,
                        'games_missed': 0  # Default value
                    }
                    
                    # Filter out None values for required fields
                    if injury_record['season'] and injury_record['team'] and injury_record['player_name']:
                        injury_records.append(injury_record)
                
                if injury_records:
                    # Insert in batches
                    batch_size = 1000
                    for i in range(0, len(injury_records), batch_size):
                        batch = injury_records[i:i + batch_size]
                        
                        try:
                            result = supabase.table('fact_injuries').upsert(
                                batch, 
                                on_conflict='season,week,team,player_name'
                            ).execute()
                            stats['total_records'] += len(batch)
                            logger.info(f"Inserted batch {i//batch_size + 1} ({len(batch)} records) for {season}")
                        except Exception as e:
                            logger.error(f"Error inserting batch {i//batch_size + 1} for {season}: {str(e)}")
                            stats['errors'] += 1
                
        except Exception as e:
            logger.error(f"Error importing injury data for {season}: {str(e)}")
            stats['errors'] += 1
    
    # Print summary
    print(f"\nInjury Import Summary:")
    print(f"Total records: {stats['total_records']}")
    print(f"Errors: {stats['errors']}")
    
    return stats

if __name__ == "__main__":
    import_injury_data()
