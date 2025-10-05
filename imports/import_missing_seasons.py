#!/usr/bin/env python3
"""
Import missing NFL seasons (2016-2019) into production database
"""
import sqlite3
import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import os

def import_missing_seasons():
    """Import seasons 2016-2019 into the production database"""

    production_db = "improved_nfl_system/database/nfl_suggestions.db"
    missing_seasons = [2016, 2017, 2018, 2019]

    print("🏈 Starting import of missing NFL seasons...")
    print(f"📅 Seasons to import: {missing_seasons}")

    # Connect to database
    conn = sqlite3.connect(production_db)
    cursor = conn.cursor()

    try:
        for season in missing_seasons:
            print(f"\n📊 Importing season {season}...")

            # Import schedules
            schedules = nfl.import_schedules([season])
            if not schedules.empty:
                # Clean and prepare data
                schedules['season'] = season
                schedules['game_date'] = pd.to_datetime(schedules['gameday'], errors='coerce')

                # Select relevant columns for all_schedules table
                schedule_cols = ['game_id', 'season', 'game_type', 'week', 'gameday',
                               'away_team', 'home_team', 'away_score', 'home_score',
                               'stadium', 'surface', 'temp', 'wind', 'away_qb_id',
                               'home_qb_id', 'away_coach', 'home_coach', 'referee',
                               'stadium_id', 'old_game_id']

                # Keep only columns that exist
                existing_cols = [col for col in schedule_cols if col in schedules.columns]
                schedule_data = schedules[existing_cols].copy()

                # Fill missing values
                schedule_data = schedule_data.fillna('')

                # Insert into database
                schedule_data.to_sql('all_schedules', conn, if_exists='append', index=False)
                print(f"  ✅ Imported {len(schedule_data)} games for season {season}")

                # Import team stats (skip for now - will use play-by-play data)
                # team_stats = nfl.import_seasonal_data([season])
                # Skip team stats import for now due to column issues
                        team_data = team_stats[team_stats['team'] == team].iloc[0]

                        epa_record = {
                            'season': season,
                            'team': team,
                            'games_played': 16,  # Standard season
                            'offensive_epa': team_data.get('epa_per_play_offense', 0),
                            'defensive_epa': team_data.get('epa_per_play_defense', 0),
                            'passing_epa': team_data.get('passing_epa', 0),
                            'rushing_epa': team_data.get('rushing_epa', 0),
                            'success_rate': team_data.get('success_rate_offense', 0),
                            'explosive_play_rate': 0,  # Will calculate if possible
                            'turnover_epa': 0,  # Will calculate if possible
                            'field_position_avg': 0,
                            'red_zone_success': 0,
                            'third_down_success': 0,
                            'pressure_rate': 0,
                            'created_at': datetime.now().isoformat()
                        }

                        # Insert into database
                        cursor.execute("""
                            INSERT OR REPLACE INTO team_epa_stats
                            (season, team, games_played, offensive_epa, defensive_epa,
                             passing_epa, rushing_epa, success_rate, explosive_play_rate,
                             turnover_epa, field_position_avg, red_zone_success,
                             third_down_success, pressure_rate, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, tuple(epa_record.values()))

                    print(f"  ✅ Imported team EPA stats for {len(team_stats['team'].unique())} teams")

            # Import play-by-play data for game features
            print(f"  📊 Importing play-by-play data for detailed metrics...")
            pbp = nfl.import_pbp_data([season], columns=['game_id', 'home_team', 'away_team',
                                                         'home_score', 'away_score', 'total_line',
                                                         'spread_line', 'home_wp', 'away_wp'])

            if not pbp.empty:
                # Group by game to get game-level features
                game_features = pbp.groupby('game_id').agg({
                    'home_team': 'first',
                    'away_team': 'first',
                    'home_score': 'max',
                    'away_score': 'max',
                    'total_line': 'first',
                    'spread_line': 'first'
                }).reset_index()

                game_features['season'] = season
                game_features['created_at'] = datetime.now().isoformat()

                # Add to game_features table
                for _, game in game_features.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO game_features
                        (game_id, season, home_team, away_team, home_score, away_score,
                         total_line, spread_line, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (game['game_id'], season, game['home_team'], game['away_team'],
                          game['home_score'], game['away_score'], game['total_line'],
                          game['spread_line'], game['created_at']))

                print(f"  ✅ Imported game features for {len(game_features)} games")

        # Commit all changes
        conn.commit()

        # Verify import
        cursor.execute("SELECT COUNT(*) FROM all_schedules")
        total_games = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(season), MAX(season) FROM all_schedules")
        season_range = cursor.fetchone()

        print(f"\n✅ Import complete!")
        print(f"📊 Database now contains {total_games} games")
        print(f"📅 Season range: {season_range[0]} to {season_range[1]}")

    except Exception as e:
        print(f"❌ Error during import: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    import_missing_seasons()