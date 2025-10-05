#!/usr/bin/env python3
"""
Direct insertion of data into Supabase using batched SQL
"""
import pandas as pd
from datetime import datetime

print("=" * 60)
print("📤 INSERTING DATA INTO SUPABASE")
print("=" * 60)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Start with games table (all_schedules)
print("📊 Processing games table...")
df = pd.read_csv('/tmp/games_data.csv')
print(f"  Loaded {len(df):,} records")

# Clean the data
df = df.where(pd.notnull(df), None)

# Take first 100 records as a test batch
batch = df.head(100)

# Prepare values for SQL
values_list = []
for _, row in batch.iterrows():
    values = []
    for col in batch.columns:
        val = row[col]
        if pd.isna(val) or val is None:
            values.append('NULL')
        elif col in ['game_id', 'game_type', 'gameday', 'home_team', 'away_team']:
            # String columns
            values.append(f"'{str(val).replace('\'', '\'\'')}'")
        else:
            # Numeric columns
            values.append(str(val))
    values_list.append(f"({','.join(values)})")

# Build INSERT SQL
columns = ','.join(batch.columns)
values_sql = ',\n'.join(values_list[:10])  # First 10 records

insert_sql = f"""
INSERT INTO games ({columns})
VALUES
{values_sql}
ON CONFLICT (game_id) DO NOTHING;
"""

print("📝 Sample SQL (first 10 records):")
print(insert_sql[:500] + "...")
print(f"\n✅ Ready to insert {len(batch)} records in first batch")