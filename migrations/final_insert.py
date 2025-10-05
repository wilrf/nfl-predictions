#!/usr/bin/env python3
"""
Final insertion script - directly insert data into Supabase
"""
import pandas as pd

# Load the games data (smallest table first for testing)
df = pd.read_csv('/tmp/historical_games_data.csv')
print(f"Loading {len(df)} historical games...")

# Clean data
df = df.fillna('')

# Take first 10 records as test
test_batch = df.head(10)

# Generate SQL
values = []
for _, row in test_batch.iterrows():
    game_id = str(row['game_id']).replace("'", "''") if row['game_id'] else ''
    season = int(row['season']) if row['season'] else 0
    week = int(row['week']) if row['week'] else 0
    home_team = str(row['home_team']).replace("'", "''") if row['home_team'] else ''
    away_team = str(row['away_team']).replace("'", "''") if row['away_team'] else ''
    game_date = str(row['game_date']) if row['game_date'] else 'NULL'
    home_score = int(row['home_score']) if row['home_score'] else 0
    away_score = int(row['away_score']) if row['away_score'] else 0

    if game_date != 'NULL':
        game_date = f"'{game_date}'"

    values.append(f"('{game_id}', {season}, {week}, '{home_team}', '{away_team}', {game_date}, {home_score}, {away_score})")

sql = f"""
INSERT INTO historical_games (game_id, season, week, home_team, away_team, game_date, home_score, away_score)
VALUES
{','.join(values)}
ON CONFLICT (game_id) DO NOTHING;
"""

print(sql)