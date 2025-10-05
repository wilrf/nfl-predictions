#!/usr/bin/env python3
"""
Final bulk migration script - Insert all data into Supabase
"""
import pandas as pd
from datetime import datetime
import sys

print("=" * 60)
print("🚀 BULK MIGRATION TO SUPABASE")
print("=" * 60)
print(f"Start time: {datetime.now().isoformat()}")
print()

# Define migration tasks
MIGRATION_TASKS = [
    {
        'name': 'historical_games',
        'file': '/tmp/historical_games_data.csv',
        'total': 1087,
        'batch_size': 100,
        'skip_first': 10  # Already inserted 10
    },
    {
        'name': 'games',
        'file': '/tmp/games_data.csv',
        'total': 2678,
        'batch_size': 100,
        'skip_first': 0
    },
    {
        'name': 'team_epa_stats',
        'file': '/tmp/team_epa_stats_data.csv',
        'total': 2816,
        'batch_size': 100,
        'skip_first': 0
    },
    {
        'name': 'game_features',
        'file': '/tmp/game_features_data.csv',
        'total': 1343,
        'batch_size': 100,
        'skip_first': 0
    },
    {
        'name': 'team_features',
        'file': '/tmp/team_features_data.csv',
        'total': 2174,
        'batch_size': 100,
        'skip_first': 0
    },
    {
        'name': 'epa_metrics',
        'file': '/tmp/epa_metrics_data.csv',
        'total': 1087,
        'batch_size': 100,
        'skip_first': 0
    },
    {
        'name': 'betting_outcomes',
        'file': '/tmp/betting_outcomes_data.csv',
        'total': 1087,
        'batch_size': 100,
        'skip_first': 0
    },
    {
        'name': 'feature_history',
        'file': '/tmp/feature_history_data.csv',
        'total': 648,
        'batch_size': 100,
        'skip_first': 0
    }
]

def escape_value(val, col_name=None):
    """Escape value for SQL insertion"""
    if pd.isna(val) or val is None or val == '':
        return 'NULL'

    # Check if it's a string column
    if isinstance(val, str):
        # Escape single quotes
        escaped = val.replace("'", "''")
        return f"'{escaped}'"

    # Boolean
    if isinstance(val, bool):
        return 'true' if val else 'false'

    # Numeric
    return str(val)

def generate_insert_sql(table_name, df_batch):
    """Generate INSERT SQL for a batch of data"""
    if df_batch.empty:
        return None

    columns = df_batch.columns.tolist()

    # Build values
    values_list = []
    for _, row in df_batch.iterrows():
        values = []
        for col in columns:
            val = row[col]

            # Special handling for certain columns
            if col in ['game_id', 'team', 'home_team', 'away_team', 'feature_name']:
                # String columns
                values.append(escape_value(val))
            elif col in ['is_home']:
                # Boolean columns
                if pd.isna(val):
                    values.append('NULL')
                else:
                    values.append('true' if val else 'false')
            elif col in ['game_date', 'gameday', 'created_at']:
                # Date columns
                if pd.isna(val) or val == '':
                    values.append('NULL')
                else:
                    values.append(f"'{val}'")
            else:
                # Numeric or other columns
                values.append(escape_value(val))

        values_list.append(f"({','.join(values)})")

    # Build INSERT statement
    sql = f"""INSERT INTO {table_name} ({','.join(columns)})
VALUES
{','.join(values_list)}
ON CONFLICT DO NOTHING;"""

    return sql

# Process each table
total_migrated = 0

for task in MIGRATION_TASKS:
    print(f"\n📊 Processing {task['name']}...")
    print(f"  Source: {task['file']}")
    print(f"  Expected: {task['total']:,} records")

    try:
        # Load CSV
        df = pd.read_csv(task['file'])

        # Skip already inserted records
        if task['skip_first'] > 0:
            print(f"  Skipping first {task['skip_first']} records (already inserted)")
            df = df.iloc[task['skip_first']:]

        # Process in batches
        batch_num = 0
        inserted = 0

        for i in range(0, len(df), task['batch_size']):
            batch = df.iloc[i:min(i + task['batch_size'], len(df))]
            batch_num += 1

            # Generate SQL
            sql = generate_insert_sql(task['name'], batch)

            if sql:
                # Save SQL to file (for manual execution if needed)
                sql_file = f"/tmp/{task['name']}_batch_{batch_num}.sql"
                with open(sql_file, 'w') as f:
                    f.write(sql)

                inserted += len(batch)
                print(f"  Batch {batch_num}: {len(batch)} records prepared")

        print(f"  ✅ Total prepared: {inserted:,} records")
        total_migrated += inserted

    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

print("\n" + "=" * 60)
print("📊 MIGRATION SUMMARY")
print("=" * 60)
print(f"Total records prepared: {total_migrated:,}")
print(f"End time: {datetime.now().isoformat()}")
print()
print("SQL files saved to /tmp/ for execution")
print("Use MCP tools to execute the SQL files")
print("=" * 60)