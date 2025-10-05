#!/usr/bin/env python3
"""
Import missing NFL seasons (2016-2019) into production database - Fixed version
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

    try:
        for season in missing_seasons:
            print(f"\n📊 Importing season {season}...")

            # Import schedules
            try:
                schedules = nfl.import_schedules([season])
                if not schedules.empty:
                    # Clean and prepare data
                    schedules['season'] = season

                    # Select relevant columns for all_schedules table
                    schedule_cols = ['game_id', 'season', 'game_type', 'week', 'gameday',
                                   'away_team', 'home_team', 'away_score', 'home_score']

                    # Keep only columns that exist
                    existing_cols = [col for col in schedule_cols if col in schedules.columns]
                    schedule_data = schedules[existing_cols].copy()

                    # Fill missing values
                    schedule_data = schedule_data.fillna('')

                    # Insert into database
                    schedule_data.to_sql('all_schedules', conn, if_exists='append', index=False)
                    print(f"  ✅ Imported {len(schedule_data)} games for season {season}")
            except Exception as e:
                print(f"  ⚠️  Error importing schedules for {season}: {e}")

            # Import play-by-play data for EPA stats
            try:
                print(f"  📊 Calculating team EPA stats from play-by-play data...")
                pbp = nfl.import_pbp_data([season], columns=['game_id', 'posteam', 'epa',
                                                             'success', 'pass', 'rush'])

                if not pbp.empty:
                    # Calculate team EPA stats
                    team_stats = []
                    teams = pd.concat([pbp['posteam'].dropna()]).unique()

                    for team in teams:
                        if pd.isna(team) or team == '':
                            continue

                        team_plays = pbp[pbp['posteam'] == team]
                        opp_plays = pbp[pbp['defteam'] == team] if 'defteam' in pbp.columns else pd.DataFrame()

                        if len(team_plays) > 0:
                            stats = {
                                'season': season,
                                'team': team,
                                'games_played': len(team_plays['game_id'].unique()),
                                'offensive_epa': team_plays['epa'].mean() if 'epa' in team_plays.columns else 0,
                                'defensive_epa': opp_plays['epa'].mean() if len(opp_plays) > 0 and 'epa' in opp_plays.columns else 0,
                                'passing_epa': team_plays[team_plays['pass'] == 1]['epa'].mean() if 'pass' in team_plays.columns else 0,
                                'rushing_epa': team_plays[team_plays['rush'] == 1]['epa'].mean() if 'rush' in team_plays.columns else 0,
                                'success_rate': team_plays['success'].mean() if 'success' in team_plays.columns else 0,
                                'explosive_play_rate': 0,
                                'turnover_epa': 0,
                                'field_position_avg': 0,
                                'red_zone_success': 0,
                                'third_down_success': 0,
                                'pressure_rate': 0,
                                'created_at': datetime.now().isoformat()
                            }

                            # Handle NaN values
                            for key, value in stats.items():
                                if pd.isna(value):
                                    stats[key] = 0

                            team_stats.append(stats)

                    # Insert team stats into database
                    if team_stats:
                        team_stats_df = pd.DataFrame(team_stats)
                        team_stats_df.to_sql('team_epa_stats', conn, if_exists='append', index=False)
                        print(f"  ✅ Calculated and imported EPA stats for {len(team_stats)} teams")
            except Exception as e:
                print(f"  ⚠️  Error calculating EPA stats for {season}: {e}")

            # Import game features
            try:
                print(f"  📊 Importing game features...")
                pbp_games = nfl.import_pbp_data([season], columns=['game_id', 'home_team', 'away_team',
                                                                   'home_score', 'away_score'])

                if not pbp_games.empty:
                    # Group by game to get final scores
                    game_features = pbp_games.groupby('game_id').agg({
                        'home_team': 'first',
                        'away_team': 'first',
                        'home_score': 'max',
                        'away_score': 'max'
                    }).reset_index()

                    game_features['season'] = season
                    game_features['created_at'] = datetime.now().isoformat()

                    # Add default values for missing columns
                    game_features['total_line'] = 0
                    game_features['spread_line'] = 0

                    # Insert into database
                    game_features.to_sql('game_features', conn, if_exists='append', index=False)
                    print(f"  ✅ Imported game features for {len(game_features)} games")
            except Exception as e:
                print(f"  ⚠️  Error importing game features for {season}: {e}")

        # Commit all changes
        conn.commit()

        # Verify import
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM all_schedules")
        total_games = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT season) FROM all_schedules")
        total_seasons = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(season), MAX(season) FROM all_schedules")
        season_range = cursor.fetchone()

        print(f"\n✅ Import complete!")
        print(f"📊 Database now contains {total_games} games")
        print(f"📅 {total_seasons} seasons total: {season_range[0]} to {season_range[1]}")

    except Exception as e:
        print(f"❌ Error during import: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    import_missing_seasons()