"""
Test Supabase integration using MCP server
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data_from_sqlite():
    """Load sample data from SQLite database"""
    conn = sqlite3.connect('database/validation_data.db')
    
    # Load a few games
    games_df = pd.read_sql("""
        SELECT * FROM game_features 
        WHERE home_score IS NOT NULL 
        ORDER BY season DESC, week DESC 
        LIMIT 10
    """, conn)
    
    # Load team EPA stats
    epa_df = pd.read_sql("""
        SELECT * FROM team_epa_stats 
        ORDER BY season DESC, week DESC 
        LIMIT 20
    """, conn)
    
    conn.close()
    
    return games_df, epa_df

def test_mcp_connection():
    """Test MCP Supabase connection"""
    print("="*80)
    print("TESTING SUPABASE MCP CONNECTION")
    print("="*80)
    
    # Note: We'll use the MCP tools directly from Claude
    # This script is for preparing the data
    
    games_df, epa_df = load_sample_data_from_sqlite()
    
    print(f"\nLoaded {len(games_df)} games from SQLite")
    print(f"Loaded {len(epa_df)} EPA records from SQLite")
    
    # Prepare sample game data for insertion
    if not games_df.empty:
        sample_game = games_df.iloc[0]
        
        print("\nSample game for testing:")
        print(f"  Season: {sample_game['season']}")
        print(f"  Week: {sample_game['week']}")
        print(f"  {sample_game['home_team']} vs {sample_game['away_team']}")
        print(f"  Score: {sample_game['home_score']}-{sample_game['away_score']}")
        print(f"  EPA Differential: {sample_game['epa_differential']:.3f}")
        
        # Create SQL for inserting test data
        insert_sql = f"""
        -- Test game insertion
        INSERT INTO games (game_uuid, season, week, game_date, home_team_id, away_team_id, home_score, away_score)
        VALUES (
            'TEST_{sample_game['season']}_{sample_game['week']}_{sample_game['home_team']}_{sample_game['away_team']}',
            {sample_game['season']},
            {sample_game['week']},
            CURRENT_TIMESTAMP,
            (SELECT team_id FROM teams WHERE team_code = '{sample_game['home_team']}'),
            (SELECT team_id FROM teams WHERE team_code = '{sample_game['away_team']}'),
            {sample_game['home_score']},
            {sample_game['away_score']}
        )
        ON CONFLICT (game_uuid) DO NOTHING;
        """
        
        print("\nSQL prepared for testing:")
        print(insert_sql[:200] + "...")
        
        return True, insert_sql
    
    return False, None

if __name__ == "__main__":
    success, sql = test_mcp_connection()
    if success:
        print("\n✓ Test data prepared successfully")
        print("\nNext steps:")
        print("1. Use mcp__supabase__execute_sql to insert test data")
        print("2. Use mcp__supabase__list_tables to verify data")
        print("3. Run full validation test with Supabase data")
    else:
        print("\n✗ Failed to prepare test data")