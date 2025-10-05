#!/usr/bin/env python3
"""
Comprehensive NFL Data Bulk Import
===================================
Purpose: Import ALL available NFL data from nfl_data_py (2016-2024)
Data sources:
  - Play-by-play: ~432K plays
  - Schedules: ~2,400 games
  - NGS (passing/receiving/rushing): ~25K records
  - Injuries: ~54K records
  - Snap counts: ~234K records
  - Rosters: ~362K records
  - Depth charts: ~335K records
  - Officials: ~17K records

Total: ~1.13M records

Author: NFL Betting System
Date: 2025-10-02
"""

import sqlite3
import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import json
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bulk_import_comprehensive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveDataImporter:
    """Import ALL NFL data into comprehensive schema"""

    def __init__(self, db_path: str, start_year: int = 2016, end_year: int = 2024):
        self.db_path = Path(db_path)
        self.start_year = start_year
        self.end_year = end_year
        self.seasons = list(range(start_year, end_year + 1))

        self.stats = {
            'games': 0,
            'plays': 0,
            'ngs_passing': 0,
            'ngs_receiving': 0,
            'ngs_rushing': 0,
            'injuries': 0,
            'snap_counts': 0,
            'rosters': 0,
            'depth_charts': 0,
            'officials': 0,
            'players': 0
        }

    def import_schedules_and_games(self):
        """Import all game schedules -> fact_games"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING SCHEDULES ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            # Import all seasons at once
            logger.info(f"Fetching schedules for {len(self.seasons)} seasons...")
            schedules = nfl.import_schedules(self.seasons)

            logger.info(f"Processing {len(schedules)} games...")

            for _, game in tqdm(schedules.iterrows(), total=len(schedules), desc="Importing games"):
                # Insert into fact_games
                conn.execute("""
                    INSERT OR REPLACE INTO fact_games (
                        game_id, season, week, game_type, gameday, weekday, gametime,
                        home_team, away_team, home_score, away_score,
                        location, result, total, overtime,
                        old_game_id, gsis, nfl_detail_id, pfr, pff, espn, ftn,
                        home_rest, away_rest,
                        spread_line, home_spread_odds, away_spread_odds,
                        total_line, over_odds, under_odds,
                        home_moneyline, away_moneyline,
                        div_game, roof, surface, temp, wind,
                        home_qb_id, away_qb_id, home_qb_name, away_qb_name,
                        home_coach, away_coach, referee, stadium_id, stadium,
                        completed
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?
                    )
                """, (
                    game['game_id'],
                    game['season'],
                    game['week'],
                    game['game_type'],
                    game['gameday'],
                    game.get('weekday'),
                    game.get('gametime'),
                    game['home_team'],
                    game['away_team'],
                    game.get('home_score'),
                    game.get('away_score'),
                    game.get('location'),
                    game.get('result'),
                    game.get('total'),
                    int(game.get('overtime', 0)) if pd.notna(game.get('overtime')) else 0,
                    game.get('old_game_id'),
                    game.get('gsis'),
                    game.get('nfl_detail_id'),
                    game.get('pfr'),
                    game.get('pff'),
                    game.get('espn'),
                    game.get('ftn'),
                    game.get('home_rest'),
                    game.get('away_rest'),
                    game.get('spread_line'),
                    game.get('home_spread_odds'),
                    game.get('away_spread_odds'),
                    game.get('total_line'),
                    game.get('under_odds'),
                    game.get('over_odds'),
                    game.get('away_moneyline'),
                    game.get('home_moneyline'),
                    int(game.get('div_game', 0)) if pd.notna(game.get('div_game')) else 0,
                    game.get('roof'),
                    game.get('surface'),
                    game.get('temp'),
                    game.get('wind'),
                    game.get('away_qb_id'),
                    game.get('home_qb_id'),
                    game.get('away_qb_name'),
                    game.get('home_qb_name'),
                    game.get('home_coach'),
                    game.get('away_coach'),
                    game.get('referee'),
                    game.get('stadium_id'),
                    game.get('stadium'),
                    1 if pd.notna(game.get('home_score')) else 0  # completed if scores exist
                ))

            conn.commit()
            self.stats['games'] = len(schedules)
            logger.info(f"✅ Imported {self.stats['games']:,} games")

        except Exception as e:
            logger.error(f"❌ Error importing schedules: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_play_by_play(self):
        """Import ALL play-by-play data -> fact_plays"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING PLAY-BY-PLAY ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")
        logger.info("⚠️  WARNING: This will import ~432,000 plays and may take 10-15 minutes")

        conn = sqlite3.connect(self.db_path)
        total_plays = 0

        try:
            # Import season by season to manage memory
            for season in self.seasons:
                logger.info(f"\nProcessing season {season}...")
                pbp = nfl.import_pbp_data([season])

                logger.info(f"Importing {len(pbp):,} plays from {season}...")

                # Batch insert for performance
                batch_size = 1000
                for i in tqdm(range(0, len(pbp), batch_size), desc=f"Season {season}"):
                    batch = pbp.iloc[i:i+batch_size]

                    for _, play in batch.iterrows():
                        conn.execute("""
                            INSERT OR REPLACE INTO fact_plays (
                                play_id, game_id, home_team, away_team, season_type, week,
                                drive, qtr, down, ydstogo, yardline_100, goal_to_go,
                                game_date, quarter_seconds_remaining, half_seconds_remaining,
                                game_seconds_remaining, game_half,
                                posteam, defteam, posteam_type, side_of_field,
                                play_type, desc, yards_gained,
                                epa, wpa, success,
                                air_epa, yac_epa, comp_air_epa, comp_yac_epa, qb_epa,
                                pass_oe, cpoe,
                                passer_player_id, passer_player_name, passing_yards,
                                air_yards, yards_after_catch, xyac_epa,
                                complete_pass, incomplete_pass, interception, pass_touchdown,
                                qb_hit, sack, was_pressure, time_to_throw, time_to_pressure,
                                rusher_player_id, rusher_player_name, rushing_yards, rush_touchdown,
                                receiver_player_id, receiver_player_name, receiving_yards,
                                field_goal_result, kick_distance, punt_blocked,
                                penalty, penalty_team, penalty_yards, fumble, fumble_lost,
                                third_down_converted, third_down_failed, fourth_down_converted, fourth_down_failed,
                                td_team, td_player_id, td_player_name,
                                extra_point_result, two_point_conv_result,
                                posteam_score, defteam_score, score_differential,
                                posteam_score_post, defteam_score_post,
                                wp, def_wp, home_wp, away_wp, vegas_wp, vegas_home_wp,
                                shotgun, no_huddle, qb_dropback, qb_scramble,
                                drive_end_transition, old_game_id
                            ) VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                        """, (
                            play['play_id'],
                            play['game_id'],
                            play.get('home_team'),
                            play.get('away_team'),
                            play.get('season_type'),
                            play.get('week'),
                            play.get('drive'),
                            play.get('qtr'),
                            play.get('down'),
                            play.get('ydstogo'),
                            play.get('yardline_100'),
                            int(play.get('goal_to_go', 0)) if pd.notna(play.get('goal_to_go')) else 0,
                            play.get('game_date'),
                            play.get('quarter_seconds_remaining'),
                            play.get('half_seconds_remaining'),
                            play.get('game_seconds_remaining'),
                            play.get('game_half'),
                            play.get('posteam'),
                            play.get('defteam'),
                            play.get('posteam_type'),
                            play.get('side_of_field'),
                            play.get('play_type'),
                            play.get('desc'),
                            play.get('yards_gained'),
                            play.get('epa'),
                            play.get('wpa'),
                            int(play.get('success', 0)) if pd.notna(play.get('success')) else None,
                            play.get('air_epa'),
                            play.get('yac_epa'),
                            play.get('comp_air_epa'),
                            play.get('comp_yac_epa'),
                            play.get('qb_epa'),
                            play.get('pass_oe'),
                            play.get('cpoe'),
                            play.get('passer_player_id'),
                            play.get('passer_player_name'),
                            play.get('passing_yards'),
                            play.get('air_yards'),
                            play.get('yards_after_catch'),
                            play.get('xyac_epa'),
                            int(play.get('complete_pass', 0)) if pd.notna(play.get('complete_pass')) else None,
                            int(play.get('incomplete_pass', 0)) if pd.notna(play.get('incomplete_pass')) else None,
                            int(play.get('interception', 0)) if pd.notna(play.get('interception')) else None,
                            int(play.get('pass_touchdown', 0)) if pd.notna(play.get('pass_touchdown')) else None,
                            int(play.get('qb_hit', 0)) if pd.notna(play.get('qb_hit')) else None,
                            int(play.get('sack', 0)) if pd.notna(play.get('sack')) else None,
                            int(play.get('was_pressure', 0)) if pd.notna(play.get('was_pressure')) else None,
                            play.get('time_to_throw'),
                            play.get('time_to_pressure'),
                            play.get('rusher_player_id'),
                            play.get('rusher_player_name'),
                            play.get('rushing_yards'),
                            int(play.get('rush_touchdown', 0)) if pd.notna(play.get('rush_touchdown')) else None,
                            play.get('receiver_player_id'),
                            play.get('receiver_player_name'),
                            play.get('receiving_yards'),
                            play.get('field_goal_result'),
                            play.get('kick_distance'),
                            int(play.get('punt_blocked', 0)) if pd.notna(play.get('punt_blocked')) else None,
                            int(play.get('penalty', 0)) if pd.notna(play.get('penalty')) else None,
                            play.get('penalty_team'),
                            play.get('penalty_yards'),
                            int(play.get('fumble', 0)) if pd.notna(play.get('fumble')) else None,
                            int(play.get('fumble_lost', 0)) if pd.notna(play.get('fumble_lost')) else None,
                            int(play.get('third_down_converted', 0)) if pd.notna(play.get('third_down_converted')) else None,
                            int(play.get('third_down_failed', 0)) if pd.notna(play.get('third_down_failed')) else None,
                            int(play.get('fourth_down_converted', 0)) if pd.notna(play.get('fourth_down_converted')) else None,
                            int(play.get('fourth_down_failed', 0)) if pd.notna(play.get('fourth_down_failed')) else None,
                            play.get('td_team'),
                            play.get('td_player_id'),
                            play.get('td_player_name'),
                            play.get('extra_point_result'),
                            play.get('two_point_conv_result'),
                            play.get('posteam_score'),
                            play.get('defteam_score'),
                            play.get('score_differential'),
                            play.get('posteam_score_post'),
                            play.get('defteam_score_post'),
                            play.get('wp'),
                            play.get('def_wp'),
                            play.get('home_wp'),
                            play.get('away_wp'),
                            play.get('vegas_wp'),
                            play.get('vegas_home_wp'),
                            int(play.get('shotgun', 0)) if pd.notna(play.get('shotgun')) else None,
                            int(play.get('no_huddle', 0)) if pd.notna(play.get('no_huddle')) else None,
                            int(play.get('qb_dropback', 0)) if pd.notna(play.get('qb_dropback')) else None,
                            int(play.get('qb_scramble', 0)) if pd.notna(play.get('qb_scramble')) else None,
                            play.get('drive_end_transition'),
                            play.get('old_game_id')
                        ))

                    # Commit each batch
                    conn.commit()

                total_plays += len(pbp)
                logger.info(f"✅ Season {season}: {len(pbp):,} plays imported (Total: {total_plays:,})")

            self.stats['plays'] = total_plays
            logger.info(f"\n✅ TOTAL PLAYS IMPORTED: {total_plays:,}")

        except Exception as e:
            logger.error(f"❌ Error importing play-by-play: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_ngs_data(self):
        """Import all Next Gen Stats -> fact_ngs_* tables"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING NEXT GEN STATS ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            # 1. NGS Passing
            logger.info("\n1. NGS Passing...")
            ngs_pass = nfl.import_ngs_data(stat_type='passing', years=self.seasons)

            for _, row in tqdm(ngs_pass.iterrows(), total=len(ngs_pass), desc="NGS Passing"):
                conn.execute("""
                    INSERT OR REPLACE INTO fact_ngs_passing (
                        season, season_type, week, player_gsis_id, team_abbr,
                        avg_time_to_throw, avg_completed_air_yards, avg_intended_air_yards,
                        avg_air_yards_differential, aggressiveness, max_completed_air_distance,
                        avg_air_yards_to_sticks, attempts, completions, pass_yards,
                        pass_touchdowns, interceptions, passer_rating, completion_percentage,
                        expected_completion_percentage, completion_percentage_above_expectation,
                        avg_air_distance, max_air_distance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['season'], row['season_type'], row['week'],
                    row['player_gsis_id'], row['team_abbr'],
                    row.get('avg_time_to_throw'), row.get('avg_completed_air_yards'),
                    row.get('avg_intended_air_yards'), row.get('avg_air_yards_differential'),
                    row.get('aggressiveness'), row.get('max_completed_air_distance'),
                    row.get('avg_air_yards_to_sticks'), row.get('attempts'), row.get('completions'),
                    row.get('pass_yards'), row.get('pass_touchdowns'), row.get('interceptions'),
                    row.get('passer_rating'), row.get('completion_percentage'),
                    row.get('expected_completion_percentage'), row.get('completion_percentage_above_expectation'),
                    row.get('avg_air_distance'), row.get('max_air_distance')
                ))

            conn.commit()
            self.stats['ngs_passing'] = len(ngs_pass)
            logger.info(f"✅ NGS Passing: {self.stats['ngs_passing']:,} records")

            # 2. NGS Receiving
            logger.info("\n2. NGS Receiving...")
            ngs_rec = nfl.import_ngs_data(stat_type='receiving', years=self.seasons)

            for _, row in tqdm(ngs_rec.iterrows(), total=len(ngs_rec), desc="NGS Receiving"):
                conn.execute("""
                    INSERT OR REPLACE INTO fact_ngs_receiving (
                        season, season_type, week, player_gsis_id, team_abbr,
                        avg_cushion, avg_separation, avg_intended_air_yards,
                        percent_share_of_intended_air_yards, receptions, targets,
                        catch_percentage, yards, rec_touchdowns,
                        avg_yac, avg_expected_yac, avg_yac_above_expectation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['season'], row['season_type'], row['week'],
                    row['player_gsis_id'], row['team_abbr'],
                    row.get('avg_cushion'), row.get('avg_separation'),
                    row.get('avg_intended_air_yards'), row.get('percent_share_of_intended_air_yards'),
                    row.get('receptions'), row.get('targets'), row.get('catch_percentage'),
                    row.get('yards'), row.get('rec_touchdowns'),
                    row.get('avg_yac'), row.get('avg_expected_yac'), row.get('avg_yac_above_expectation')
                ))

            conn.commit()
            self.stats['ngs_receiving'] = len(ngs_rec)
            logger.info(f"✅ NGS Receiving: {self.stats['ngs_receiving']:,} records")

            # 3. NGS Rushing
            logger.info("\n3. NGS Rushing...")
            ngs_rush = nfl.import_ngs_data(stat_type='rushing', years=self.seasons)

            for _, row in tqdm(ngs_rush.iterrows(), total=len(ngs_rush), desc="NGS Rushing"):
                conn.execute("""
                    INSERT OR REPLACE INTO fact_ngs_rushing (
                        season, season_type, week, player_gsis_id, team_abbr,
                        efficiency, percent_attempts_gte_eight_defenders, avg_time_to_los,
                        rush_attempts, rush_yards, avg_rush_yards, rush_touchdowns,
                        expected_rush_yards, rush_yards_over_expected,
                        rush_yards_over_expected_per_att, rush_pct_over_expected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['season'], row['season_type'], row['week'],
                    row['player_gsis_id'], row['team_abbr'],
                    row.get('efficiency'), row.get('percent_attempts_gte_eight_defenders'),
                    row.get('avg_time_to_los'), row.get('rush_attempts'), row.get('rush_yards'),
                    row.get('avg_rush_yards'), row.get('rush_touchdowns'),
                    row.get('expected_rush_yards'), row.get('rush_yards_over_expected'),
                    row.get('rush_yards_over_expected_per_att'), row.get('rush_pct_over_expected')
                ))

            conn.commit()
            self.stats['ngs_rushing'] = len(ngs_rush)
            logger.info(f"✅ NGS Rushing: {self.stats['ngs_rushing']:,} records")

        except Exception as e:
            logger.error(f"❌ Error importing NGS data: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_injuries(self):
        """Import injury reports -> fact_injuries"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING INJURIES ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            logger.info("Fetching injury data...")
            injuries = nfl.import_injuries(self.seasons)

            logger.info(f"Importing {len(injuries):,} injury records...")

            for _, inj in tqdm(injuries.iterrows(), total=len(injuries), desc="Injuries"):
                # Convert date_modified to string if it exists
                date_modified = inj.get('date_modified')
                if pd.notna(date_modified):
                    date_modified = str(date_modified)
                else:
                    date_modified = None

                conn.execute("""
                    INSERT OR REPLACE INTO fact_injuries (
                        season, week, game_type, team, gsis_id, player_name, position,
                        report_primary_injury, report_secondary_injury, report_status,
                        practice_primary_injury, practice_secondary_injury, practice_status,
                        date_modified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    inj['season'], inj['week'], inj['game_type'], inj['team'],
                    inj.get('gsis_id'), inj['full_name'], inj.get('position'),
                    inj.get('report_primary_injury'), inj.get('report_secondary_injury'),
                    inj.get('report_status'),
                    inj.get('practice_primary_injury'), inj.get('practice_secondary_injury'),
                    inj.get('practice_status'), date_modified
                ))

            conn.commit()
            self.stats['injuries'] = len(injuries)
            logger.info(f"✅ Imported {self.stats['injuries']:,} injury records")

        except Exception as e:
            logger.error(f"❌ Error importing injuries: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_snap_counts(self):
        """Import snap counts -> fact_snap_counts"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING SNAP COUNTS ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            logger.info("Fetching snap count data...")
            snaps = nfl.import_snap_counts(self.seasons)

            logger.info(f"Importing {len(snaps):,} snap count records...")

            for _, snap in tqdm(snaps.iterrows(), total=len(snaps), desc="Snap counts"):
                conn.execute("""
                    INSERT OR REPLACE INTO fact_snap_counts (
                        game_id, season, game_type, week, player, pfr_player_id, position,
                        team, opponent, offense_snaps, offense_pct, defense_snaps, defense_pct,
                        st_snaps, st_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snap['game_id'], snap['season'], snap['game_type'], snap['week'],
                    snap['player'], snap.get('pfr_player_id'), snap.get('position'),
                    snap['team'], snap['opponent'],
                    snap.get('offense_snaps'), snap.get('offense_pct'),
                    snap.get('defense_snaps'), snap.get('defense_pct'),
                    snap.get('st_snaps'), snap.get('st_pct')
                ))

            conn.commit()
            self.stats['snap_counts'] = len(snaps)
            logger.info(f"✅ Imported {self.stats['snap_counts']:,} snap count records")

        except Exception as e:
            logger.error(f"❌ Error importing snap counts: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_rosters_and_players(self):
        """Import rosters and build player dimension -> dim_players + fact_weekly_rosters"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING ROSTERS & PLAYERS ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            # 1. Import weekly rosters
            logger.info("Fetching weekly roster data...")
            rosters = nfl.import_weekly_rosters(self.seasons)

            logger.info(f"Processing {len(rosters):,} roster entries...")

            players_added = set()

            for _, roster in tqdm(rosters.iterrows(), total=len(rosters), desc="Rosters"):
                player_id = roster.get('player_id')

                # Add to dim_players if not exists
                if player_id and player_id not in players_added:
                    conn.execute("""
                        INSERT OR IGNORE INTO dim_players (
                            player_id, player_name, first_name, last_name,
                            birth_date, height, weight, college, position,
                            espn_id, sportradar_id, yahoo_id, rotowire_id,
                            pff_id, pfr_id, sleeper_id,
                            entry_year, rookie_year, headshot_url
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id,
                        roster.get('player_name'),
                        roster.get('first_name'),
                        roster.get('last_name'),
                        roster.get('birth_date'),
                        roster.get('height'),
                        roster.get('weight'),
                        roster.get('college'),
                        roster.get('position'),
                        roster.get('espn_id'),
                        roster.get('sportradar_id'),
                        roster.get('yahoo_id'),
                        roster.get('rotowire_id'),
                        roster.get('pff_id'),
                        roster.get('pfr_id'),
                        roster.get('sleeper_id'),
                        roster.get('entry_year'),
                        roster.get('rookie_year'),
                        roster.get('headshot_url')
                    ))
                    players_added.add(player_id)

                # Add to fact_weekly_rosters
                conn.execute("""
                    INSERT OR REPLACE INTO fact_weekly_rosters (
                        season, week, team, player_id, position,
                        depth_chart_position, jersey_number, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    roster['season'],
                    roster['week'],
                    roster['team'],
                    player_id,
                    roster.get('position'),
                    roster.get('depth_chart_position'),
                    roster.get('jersey_number'),
                    roster.get('status')
                ))

            conn.commit()
            self.stats['players'] = len(players_added)
            self.stats['rosters'] = len(rosters)
            logger.info(f"✅ Imported {self.stats['players']:,} unique players")
            logger.info(f"✅ Imported {self.stats['rosters']:,} weekly roster entries")

        except Exception as e:
            logger.error(f"❌ Error importing rosters: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_depth_charts(self):
        """Import depth charts -> fact_depth_charts"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING DEPTH CHARTS ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            logger.info("Fetching depth chart data...")
            depth = nfl.import_depth_charts(self.seasons)

            logger.info(f"Importing {len(depth):,} depth chart records...")

            for _, d in tqdm(depth.iterrows(), total=len(depth), desc="Depth charts"):
                conn.execute("""
                    INSERT OR REPLACE INTO fact_depth_charts (
                        season, week, game_type, club_code, player_gsis_id,
                        position, depth_position, formation, depth_team,
                        player_name, football_name, jersey_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    d['season'], d['week'], d.get('game_type'), d['club_code'],
                    d.get('gsis_id'), d['position'], d.get('depth_position'),
                    d.get('formation'), d.get('depth_team'),
                    d.get('full_name'), d.get('football_name'), d.get('jersey_number')
                ))

            conn.commit()
            self.stats['depth_charts'] = len(depth)
            logger.info(f"✅ Imported {self.stats['depth_charts']:,} depth chart records")

        except Exception as e:
            logger.error(f"❌ Error importing depth charts: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def import_officials(self):
        """Import game officials -> fact_game_officials"""
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING OFFICIALS ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")

        conn = sqlite3.connect(self.db_path)

        try:
            logger.info("Fetching officials data...")
            officials = nfl.import_officials(self.seasons)

            logger.info(f"Importing {len(officials):,} official records...")

            officials_added = set()

            for _, off in tqdm(officials.iterrows(), total=len(officials), desc="Officials"):
                official_id = off['official_id']

                # Add to dim_officials if not exists
                if official_id not in officials_added:
                    conn.execute("""
                        INSERT OR IGNORE INTO dim_officials (official_id, name)
                        VALUES (?, ?)
                    """, (official_id, off['name']))
                    officials_added.add(official_id)

                # Add to fact_game_officials
                conn.execute("""
                    INSERT OR REPLACE INTO fact_game_officials (
                        game_id, season, official_id, official_name, official_position
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    off['game_id'], off['season'], official_id, off['name'], off.get('off_pos')
                ))

            conn.commit()
            self.stats['officials'] = len(officials)
            logger.info(f"✅ Imported {len(officials_added):,} unique officials")
            logger.info(f"✅ Imported {self.stats['officials']:,} game-official records")

        except Exception as e:
            logger.error(f"❌ Error importing officials: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def generate_import_summary(self):
        """Generate comprehensive import summary"""
        logger.info("\n" + "="*60)
        logger.info("IMPORT SUMMARY")
        logger.info("="*60)

        total_records = sum(self.stats.values())

        logger.info(f"\n📊 Total Records Imported: {total_records:,}")
        logger.info(f"\nBreakdown by source:")
        logger.info(f"  • Games: {self.stats['games']:,}")
        logger.info(f"  • Plays: {self.stats['plays']:,}")
        logger.info(f"  • NGS Passing: {self.stats['ngs_passing']:,}")
        logger.info(f"  • NGS Receiving: {self.stats['ngs_receiving']:,}")
        logger.info(f"  • NGS Rushing: {self.stats['ngs_rushing']:,}")
        logger.info(f"  • Injuries: {self.stats['injuries']:,}")
        logger.info(f"  • Snap Counts: {self.stats['snap_counts']:,}")
        logger.info(f"  • Players: {self.stats['players']:,}")
        logger.info(f"  • Weekly Rosters: {self.stats['rosters']:,}")
        logger.info(f"  • Depth Charts: {self.stats['depth_charts']:,}")
        logger.info(f"  • Officials: {self.stats['officials']:,}")

        # Database size
        db_size = self.db_path.stat().st_size
        logger.info(f"\n💾 Database Size: {db_size / 1024 / 1024:.2f} MB")

        # Save summary to JSON
        summary = {
            'import_date': datetime.now().isoformat(),
            'date_range': f"{self.start_year}-{self.end_year}",
            'total_records': total_records,
            'stats': self.stats,
            'database_size_mb': round(db_size / 1024 / 1024, 2)
        }

        summary_path = Path('logs') / 'bulk_import_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n📝 Summary saved to: {summary_path}")

    def run_full_import(self, skip_pbp=False):
        """Run full import of all data sources"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE NFL DATA IMPORT")
        logger.info(f"Date Range: {self.start_year}-{self.end_year}")
        logger.info("="*60)

        start_time = datetime.now()

        try:
            # 1. Games/Schedules
            self.import_schedules_and_games()

            # 2. Play-by-play (can be skipped if too slow)
            if not skip_pbp:
                self.import_play_by_play()
            else:
                logger.warning("⚠️  Skipping play-by-play import (--skip-pbp flag)")

            # 3. NGS data
            self.import_ngs_data()

            # 4. Injuries
            self.import_injuries()

            # 5. Snap counts
            self.import_snap_counts()

            # 6. Rosters & Players
            self.import_rosters_and_players()

            # 7. Depth charts
            self.import_depth_charts()

            # 8. Officials
            self.import_officials()

            # Summary
            self.generate_import_summary()

            elapsed = datetime.now() - start_time
            logger.info(f"\n⏱️  Total import time: {elapsed}")
            logger.info("\n✅ IMPORT COMPLETE!")

            return True

        except Exception as e:
            logger.error(f"\n❌ Import failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Import ALL NFL data into comprehensive database')
    parser.add_argument('--db', default='database/nfl_comprehensive.db', help='Path to database')
    parser.add_argument('--start-year', type=int, default=2016, help='Start year (default: 2016)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year (default: 2024)')
    parser.add_argument('--skip-pbp', action='store_true', help='Skip play-by-play import (faster)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be imported without importing')

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No data will be imported")
        logger.info(f"Database: {args.db}")
        logger.info(f"Date range: {args.start_year}-{args.end_year}")
        logger.info(f"Skip PBP: {args.skip_pbp}")
        logger.info(f"Estimated records: ~1,130,000")
        logger.info(f"Estimated time: 15-30 minutes (or ~5 min without PBP)")
        return

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    importer = ComprehensiveDataImporter(args.db, args.start_year, args.end_year)
    success = importer.run_full_import(skip_pbp=args.skip_pbp)

    if success:
        logger.info("\n✅ All data imported successfully!")
        exit(0)
    else:
        logger.error("\n❌ Import failed!")
        exit(1)


if __name__ == '__main__':
    main()
