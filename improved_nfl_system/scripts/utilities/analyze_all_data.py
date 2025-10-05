#!/usr/bin/env python3
"""
Analyze all SQLite databases to prepare for Supabase migration
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path

def analyze_database(db_path):
    """Analyze a single database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    db_info = {
        'path': str(db_path),
        'tables': {}
    }
    
    for table_name in tables:
        table_name = table_name[0]
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get sample data
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        
        db_info['tables'][table_name] = {
            'row_count': row_count,
            'columns': [{
                'name': col[1],
                'type': col[2],
                'nullable': not col[3],
                'primary_key': bool(col[5])
            } for col in columns],
            'sample_data': df.to_dict('records') if not df.empty else []
        }
    
    conn.close()
    return db_info

def main():
    databases = [
        'database/nfl_suggestions.db',
        'database/validation_data.db',
        'database/test_validation.db'
    ]
    
    all_data = {}
    total_rows = 0
    
    print("="*80)
    print("DATABASE ANALYSIS FOR SUPABASE MIGRATION")
    print("="*80)
    
    for db_path in databases:
        if Path(db_path).exists():
            print(f"\nAnalyzing {db_path}...")
            db_info = analyze_database(db_path)
            all_data[db_path] = db_info
            
            print(f"  Tables: {len(db_info['tables'])}")
            for table_name, table_info in db_info['tables'].items():
                rows = table_info['row_count']
                cols = len(table_info['columns'])
                total_rows += rows
                print(f"    - {table_name}: {rows:,} rows, {cols} columns")
    
    print(f"\nTOTAL: {total_rows:,} rows across all databases")
    
    # Save analysis
    with open('database_analysis.json', 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    
    print("\nAnalysis saved to database_analysis.json")
    
    # Identify unique tables and overlaps
    all_tables = set()
    table_locations = {}
    
    for db_path, db_info in all_data.items():
        for table_name in db_info['tables'].keys():
            all_tables.add(table_name)
            if table_name not in table_locations:
                table_locations[table_name] = []
            table_locations[table_name].append(db_path)
    
    print("\n" + "="*80)
    print("TABLE DISTRIBUTION")
    print("="*80)
    
    print("\nUnique tables:")
    for table in sorted(all_tables):
        locations = table_locations[table]
        if len(locations) == 1:
            print(f"  - {table}: {locations[0]}")
    
    print("\nTables in multiple databases:")
    for table in sorted(all_tables):
        locations = table_locations[table]
        if len(locations) > 1:
            print(f"  - {table}: {', '.join([Path(p).name for p in locations])}")
    
    return all_data

if __name__ == "__main__":
    main()