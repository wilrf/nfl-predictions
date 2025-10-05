#!/usr/bin/env python3
"""
STEP 4: Migrate Data from SQLite to Supabase
"""
import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

print("=" * 60)
print("📊 STARTING DATA MIGRATION TO SUPABASE")
print("=" * 60)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Connect to SQLite
sqlite_db = 'improved_nfl_system/database/nfl_suggestions.db'
conn_sqlite = sqlite3.connect(sqlite_db)

# Migration log
migration_log = {
    'start_time': datetime.now().isoformat(),
    'source_db': sqlite_db,
    'target_db': 'Supabase',
    'tables_migrated': [],
    'record_counts': {},
    'errors': [],
    'validation_results': {}
}

# Migration order (respect foreign key dependencies!)
MIGRATION_TASKS = [
    # Format: (sqlite_table, supabase_table, expected_count)
    ('all_schedules', 'games', 2678),
    ('team_epa_stats', 'team_epa_stats', 2816),
    ('game_features', 'game_features', 1343),
    ('historical_games', 'historical_games', 1087),
    ('team_features', 'team_features', 2174),
    ('epa_metrics', 'epa_metrics', 1087),
    ('betting_outcomes', 'betting_outcomes', 1087),
    ('feature_history', 'feature_history', 648),
]

print("📋 Tables to migrate:")
for sqlite_table, supabase_table, expected_count in MIGRATION_TASKS:
    print(f"  • {sqlite_table:20} → {supabase_table:20} ({expected_count:,} records)")

print("\n" + "=" * 60)

# Track progress
total_migrated = 0
total_expected = sum(count for _, _, count in MIGRATION_TASKS)

for sqlite_table, supabase_table, expected_count in MIGRATION_TASKS:
    print(f"\n📊 Migrating {sqlite_table} → {supabase_table}...")

    try:
        # Read from SQLite
        df = pd.read_sql(f"SELECT * FROM {sqlite_table}", conn_sqlite)
        original_count = len(df)

        print(f"  Found {original_count:,} records (expected {expected_count:,})")

        # Handle table-specific mappings
        if sqlite_table == 'all_schedules' and supabase_table == 'games':
            # Map all_schedules columns to games table
            column_mapping = {
                'game_id': 'game_id',
                'season': 'season',
                'week': 'week',
                'game_type': 'game_type',
                'gameday': 'gameday',
                'home_team': 'home_team',
                'away_team': 'away_team',
                'home_score': 'home_score',
                'away_score': 'away_score'
            }

            # Select and rename columns
            available_cols = [col for col in column_mapping.keys() if col in df.columns]
            df = df[available_cols].copy()

            # Add missing columns with defaults
            if 'spread_line' not in df.columns:
                df['spread_line'] = 0.0
            if 'total_line' not in df.columns:
                df['total_line'] = 0.0

        # Clean data
        df = df.where(pd.notnull(df), None)

        # Convert to CSV for batch insert
        csv_file = f'/tmp/{supabase_table}_data.csv'
        df.to_csv(csv_file, index=False, header=True)

        print(f"  ✅ Prepared {len(df):,} records for migration")
        print(f"  📦 Data saved to {csv_file}")

        # Log success
        migration_log['tables_migrated'].append(supabase_table)
        migration_log['record_counts'][supabase_table] = {
            'source': original_count,
            'prepared': len(df),
            'expected': expected_count
        }

        total_migrated += len(df)

        # Progress indicator
        progress_pct = (total_migrated / total_expected) * 100
        print(f"  📈 Overall progress: {progress_pct:.1f}% ({total_migrated:,}/{total_expected:,})")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        migration_log['errors'].append({
            'table': sqlite_table,
            'error': str(e)
        })

# Close SQLite connection
conn_sqlite.close()

# Save migration log
migration_log['end_time'] = datetime.now().isoformat()
migration_log['total_records_prepared'] = total_migrated

log_file = f"migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(log_file, 'w') as f:
    json.dump(migration_log, f, indent=2)

print("\n" + "=" * 60)
print("📊 MIGRATION PREPARATION COMPLETE")
print("=" * 60)
print(f"  Total records prepared: {total_migrated:,}")
print(f"  Tables processed: {len(migration_log['tables_migrated'])}")
print(f"  Errors encountered: {len(migration_log['errors'])}")
print(f"  Log saved to: {log_file}")
print()
print("Next step: Execute bulk inserts to Supabase using the prepared CSV files")
print("=" * 60)