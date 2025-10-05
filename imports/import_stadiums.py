#!/usr/bin/env python3
"""
Import NFL Stadiums Data into Supabase
This script populates the dim_stadiums table with current NFL stadium information.
"""

import psycopg2
import pandas as pd
from datetime import datetime

def get_stadiums_data():
    """Get current NFL stadiums data"""
    stadiums = [
        {
            'stadium_id': 'ATL_01',
            'stadium_name': 'Mercedes-Benz Stadium',
            'location': 'Atlanta, GA',
            'roof': 'retractable',
            'surface': 'Artificial',
            'capacity': 71000,
            'team_abbr': 'ATL',
            'latitude': 33.755,
            'longitude': -84.401,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'BUF_01',
            'stadium_name': 'Highmark Stadium',
            'location': 'Orchard Park, NY',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 71608,
            'team_abbr': 'BUF',
            'latitude': 42.7738,
            'longitude': -78.7869,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'CAR_01',
            'stadium_name': 'Bank of America Stadium',
            'location': 'Charlotte, NC',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 74867,
            'team_abbr': 'CAR',
            'latitude': 35.2258,
            'longitude': -80.8528,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'CHI_01',
            'stadium_name': 'Soldier Field',
            'location': 'Chicago, IL',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 61500,
            'team_abbr': 'CHI',
            'latitude': 41.8625,
            'longitude': -87.6167,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'CIN_01',
            'stadium_name': 'Paycor Stadium',
            'location': 'Cincinnati, OH',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 65515,
            'team_abbr': 'CIN',
            'latitude': 39.0950,
            'longitude': -84.5160,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'CLE_01',
            'stadium_name': 'Cleveland Browns Stadium',
            'location': 'Cleveland, OH',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 67895,
            'team_abbr': 'CLE',
            'latitude': 41.5061,
            'longitude': -81.7966,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'DAL_01',
            'stadium_name': 'AT&T Stadium',
            'location': 'Arlington, TX',
            'roof': 'retractable',
            'surface': 'Artificial',
            'capacity': 80000,
            'team_abbr': 'DAL',
            'latitude': 32.7473,
            'longitude': -97.0945,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'DEN_01',
            'stadium_name': 'Empower Field at Mile High',
            'location': 'Denver, CO',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 76125,
            'team_abbr': 'DEN',
            'latitude': 39.7439,
            'longitude': -105.0200,
            'timezone': 'America/Denver'
        },
        {
            'stadium_id': 'DET_01',
            'stadium_name': 'Ford Field',
            'location': 'Detroit, MI',
            'roof': 'dome',
            'surface': 'Artificial',
            'capacity': 65000,
            'team_abbr': 'DET',
            'latitude': 42.3400,
            'longitude': -83.0456,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'GB_01',
            'stadium_name': 'Lambeau Field',
            'location': 'Green Bay, WI',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 81441,
            'team_abbr': 'GB',
            'latitude': 44.5013,
            'longitude': -88.0622,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'HOU_01',
            'stadium_name': 'NRG Stadium',
            'location': 'Houston, TX',
            'roof': 'retractable',
            'surface': 'Artificial',
            'capacity': 72220,
            'team_abbr': 'HOU',
            'latitude': 29.6847,
            'longitude': -95.4107,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'IND_01',
            'stadium_name': 'Lucas Oil Stadium',
            'location': 'Indianapolis, IN',
            'roof': 'retractable',
            'surface': 'Artificial',
            'capacity': 67000,
            'team_abbr': 'IND',
            'latitude': 39.7601,
            'longitude': -86.1639,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'JAX_01',
            'stadium_name': 'EverBank Stadium',
            'location': 'Jacksonville, FL',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 67814,
            'team_abbr': 'JAX',
            'latitude': 30.3239,
            'longitude': -81.6372,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'KC_01',
            'stadium_name': 'Arrowhead Stadium',
            'location': 'Kansas City, MO',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 76416,
            'team_abbr': 'KC',
            'latitude': 39.0489,
            'longitude': -94.4839,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'LAC_01',
            'stadium_name': 'SoFi Stadium',
            'location': 'Inglewood, CA',
            'roof': 'open',
            'surface': 'Artificial',
            'capacity': 70240,
            'team_abbr': 'LAC',
            'latitude': 33.9533,
            'longitude': -118.3389,
            'timezone': 'America/Los_Angeles'
        },
        {
            'stadium_id': 'LV_01',
            'stadium_name': 'Allegiant Stadium',
            'location': 'Las Vegas, NV',
            'roof': 'dome',
            'surface': 'Artificial',
            'capacity': 65000,
            'team_abbr': 'LV',
            'latitude': 36.0908,
            'longitude': -115.1836,
            'timezone': 'America/Los_Angeles'
        },
        {
            'stadium_id': 'MIA_01',
            'stadium_name': 'Hard Rock Stadium',
            'location': 'Miami Gardens, FL',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 65326,
            'team_abbr': 'MIA',
            'latitude': 25.9581,
            'longitude': -80.2389,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'MIN_01',
            'stadium_name': 'U.S. Bank Stadium',
            'location': 'Minneapolis, MN',
            'roof': 'dome',
            'surface': 'Artificial',
            'capacity': 66655,
            'team_abbr': 'MIN',
            'latitude': 44.9739,
            'longitude': -93.2581,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'NE_01',
            'stadium_name': 'Gillette Stadium',
            'location': 'Foxborough, MA',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 65878,
            'team_abbr': 'NE',
            'latitude': 42.0909,
            'longitude': -71.2643,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'NO_01',
            'stadium_name': 'Caesars Superdome',
            'location': 'New Orleans, LA',
            'roof': 'dome',
            'surface': 'Artificial',
            'capacity': 73208,
            'team_abbr': 'NO',
            'latitude': 29.9508,
            'longitude': -90.0811,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'NYG_01',
            'stadium_name': 'MetLife Stadium',
            'location': 'East Rutherford, NJ',
            'roof': 'open',
            'surface': 'Artificial',
            'capacity': 82500,
            'team_abbr': 'NYG',
            'latitude': 40.8136,
            'longitude': -74.0744,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'NYJ_01',
            'stadium_name': 'MetLife Stadium',
            'location': 'East Rutherford, NJ',
            'roof': 'open',
            'surface': 'Artificial',
            'capacity': 82500,
            'team_abbr': 'NYJ',
            'latitude': 40.8136,
            'longitude': -74.0744,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'PHI_01',
            'stadium_name': 'Lincoln Financial Field',
            'location': 'Philadelphia, PA',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 69596,
            'team_abbr': 'PHI',
            'latitude': 39.9008,
            'longitude': -75.1675,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'PIT_01',
            'stadium_name': 'Acrisure Stadium',
            'location': 'Pittsburgh, PA',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 68400,
            'team_abbr': 'PIT',
            'latitude': 40.4468,
            'longitude': -80.0158,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'SF_01',
            'stadium_name': 'Levi\'s Stadium',
            'location': 'Santa Clara, CA',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 68500,
            'team_abbr': 'SF',
            'latitude': 37.4030,
            'longitude': -121.9700,
            'timezone': 'America/Los_Angeles'
        },
        {
            'stadium_id': 'SEA_01',
            'stadium_name': 'Lumen Field',
            'location': 'Seattle, WA',
            'roof': 'open',
            'surface': 'Artificial',
            'capacity': 68000,
            'team_abbr': 'SEA',
            'latitude': 47.5952,
            'longitude': -122.3316,
            'timezone': 'America/Los_Angeles'
        },
        {
            'stadium_id': 'TB_01',
            'stadium_name': 'Raymond James Stadium',
            'location': 'Tampa, FL',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 65890,
            'team_abbr': 'TB',
            'latitude': 27.9759,
            'longitude': -82.5033,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'TEN_01',
            'stadium_name': 'Nissan Stadium',
            'location': 'Nashville, TN',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 69143,
            'team_abbr': 'TEN',
            'latitude': 36.1664,
            'longitude': -86.7714,
            'timezone': 'America/Chicago'
        },
        {
            'stadium_id': 'WAS_01',
            'stadium_name': 'FedExField',
            'location': 'Landover, MD',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 82000,
            'team_abbr': 'WAS',
            'latitude': 38.9078,
            'longitude': -76.8644,
            'timezone': 'America/New_York'
        },
        {
            'stadium_id': 'ARI_01',
            'stadium_name': 'State Farm Stadium',
            'location': 'Glendale, AZ',
            'roof': 'retractable',
            'surface': 'Artificial',
            'capacity': 63400,
            'team_abbr': 'ARI',
            'latitude': 33.5275,
            'longitude': -112.2625,
            'timezone': 'America/Phoenix'
        },
        {
            'stadium_id': 'LAR_01',
            'stadium_name': 'SoFi Stadium',
            'location': 'Inglewood, CA',
            'roof': 'open',
            'surface': 'Artificial',
            'capacity': 70240,
            'team_abbr': 'LAR',
            'latitude': 33.9533,
            'longitude': -118.3389,
            'timezone': 'America/Los_Angeles'
        },
        {
            'stadium_id': 'BAL_01',
            'stadium_name': 'M&T Bank Stadium',
            'location': 'Baltimore, MD',
            'roof': 'open',
            'surface': 'Natural',
            'capacity': 71008,
            'team_abbr': 'BAL',
            'latitude': 39.2781,
            'longitude': -76.6228,
            'timezone': 'America/New_York'
        }
    ]
    
    return stadiums

def import_stadiums_to_supabase():
    """Import stadiums data to Supabase"""
    print("🏟️ Importing NFL Stadiums Data to Supabase")
    print("=" * 50)
    
    # Get stadiums data
    stadiums = get_stadiums_data()
    print(f"Found {len(stadiums)} stadiums to import")
    
    # Connect to Supabase
    try:
        conn = psycopg2.connect(
            host='db.cqslvbxsqsgjagjkpiro.supabase.co',
            port=5432,
            database='postgres',
            user='postgres',
            password='P@ssword9804746196$'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("✅ Connected to Supabase")
        
        # Clear existing data
        cursor.execute("DELETE FROM dim_stadiums")
        print("🗑️ Cleared existing stadiums data")
        
        # Insert stadiums data
        successful_inserts = 0
        failed_inserts = 0
        
        for stadium in stadiums:
            try:
                insert_sql = """
                INSERT INTO dim_stadiums (
                    stadium_id, stadium_name, location, roof, surface, 
                    capacity, team_abbr, latitude, longitude, timezone
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                cursor.execute(insert_sql, (
                    stadium['stadium_id'],
                    stadium['stadium_name'],
                    stadium['location'],
                    stadium['roof'],
                    stadium['surface'],
                    stadium['capacity'],
                    stadium['team_abbr'],
                    stadium['latitude'],
                    stadium['longitude'],
                    stadium['timezone']
                ))
                
                successful_inserts += 1
                print(f"  ✅ {stadium['team_abbr']}: {stadium['stadium_name']}")
                
            except Exception as e:
                failed_inserts += 1
                print(f"  ❌ {stadium['team_abbr']}: Error - {e}")
        
        # Verify import
        cursor.execute("SELECT COUNT(*) FROM dim_stadiums")
        count = cursor.fetchone()[0]
        
        print(f"\n📊 Import Summary:")
        print(f"  Successful: {successful_inserts}")
        print(f"  Failed: {failed_inserts}")
        print(f"  Total in database: {count}")
        
        if count == len(stadiums):
            print("🎉 All stadiums imported successfully!")
        else:
            print(f"⚠️ Import incomplete: {count}/{len(stadiums)} stadiums")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")

def create_stadiums_csv():
    """Create CSV file for manual import"""
    stadiums = get_stadiums_data()
    df = pd.DataFrame(stadiums)
    csv_file = 'stadiums_data.csv'
    df.to_csv(csv_file, index=False)
    print(f"📄 Created {csv_file} with {len(stadiums)} stadiums")
    return csv_file

if __name__ == "__main__":
    print("🏟️ NFL Stadiums Import Script")
    print("Running direct import to Supabase...")
    import_stadiums_to_supabase()
