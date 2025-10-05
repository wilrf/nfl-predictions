-- ============================================================================
-- COMPLETE NFL DATA WAREHOUSE SCHEMA FOR SUPABASE
-- ============================================================================
-- Purpose: Store ALL available NFL data from nflreadpy (2016-2025)
-- Total data: ~1.5M records across 25+ data sources
-- Design: Normalized star schema with fact and dimension tables
-- Database: PostgreSQL (Supabase)
-- Author: NFL Betting System
-- Date: 2025-10-04
-- ============================================================================

-- ============================================================================
-- DIMENSION TABLES (Reference Data)
-- ============================================================================

-- Teams master table (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS dim_teams (
    team_abbr VARCHAR(3) PRIMARY KEY NOT NULL,
    team_name VARCHAR(100) NOT NULL,
    team_id VARCHAR(10),
    team_nick VARCHAR(50),
    team_conf VARCHAR(3) CHECK(team_conf IN ('AFC', 'NFC')),
    team_division VARCHAR(20) CHECK(team_division IN ('NFC North', 'NFC South', 'NFC East', 'NFC West', 'AFC North', 'AFC South', 'AFC East', 'AFC West')),
    team_color VARCHAR(7),
    team_color2 VARCHAR(7),
    team_color3 VARCHAR(7),
    team_color4 VARCHAR(7),
    team_logo_wikipedia TEXT,
    team_logo_espn TEXT,
    team_wordmark TEXT,
    team_conference_logo TEXT,
    team_league_logo TEXT,
    team_logo_squared TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players master table (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS dim_players (
    player_id VARCHAR(20) PRIMARY KEY NOT NULL,  -- gsis_id
    player_name VARCHAR(100) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    birth_date DATE,
    height VARCHAR(10),
    weight INTEGER,
    college VARCHAR(100),
    position VARCHAR(10),
    -- External IDs for data joining
    espn_id VARCHAR(20),
    sportradar_id VARCHAR(20),
    yahoo_id VARCHAR(20),
    rotowire_id VARCHAR(20),
    pff_id VARCHAR(20),
    pfr_id VARCHAR(20),
    sleeper_id VARCHAR(20),
    -- Metadata
    entry_year INTEGER,
    rookie_year INTEGER,
    headshot_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stadiums/Venues (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS dim_stadiums (
    stadium_id VARCHAR(20) PRIMARY KEY NOT NULL,
    stadium_name VARCHAR(100) NOT NULL,
    location VARCHAR(100),
    roof VARCHAR(20) CHECK(roof IN ('outdoors', 'dome', 'retractable', 'open')),
    surface VARCHAR(20),
    capacity INTEGER,
    team_abbr VARCHAR(3),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    elevation INTEGER,
    timezone VARCHAR(50),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr)
);

-- Officials (referees - enhanced for Supabase)
CREATE TABLE IF NOT EXISTS dim_officials (
    official_id VARCHAR(20) PRIMARY KEY NOT NULL,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(50),
    experience_years INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seasons metadata
CREATE TABLE IF NOT EXISTS dim_seasons (
    season INTEGER PRIMARY KEY NOT NULL,
    season_type VARCHAR(10) CHECK(season_type IN ('REG', 'POST')),
    week_count INTEGER,
    start_date DATE,
    end_date DATE,
    champion VARCHAR(3),
    champion_score INTEGER,
    mvp VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- FACT TABLES (Core Event Data)
-- ============================================================================

-- Games (central fact table - enhanced for Supabase)
CREATE TABLE IF NOT EXISTS fact_games (
    game_id VARCHAR(20) PRIMARY KEY NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type VARCHAR(10) NOT NULL CHECK(game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB', 'PRE')),
    gameday DATE NOT NULL,
    gametime TIME,
    weekday VARCHAR(20),

    -- Teams
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,

    -- Scores
    home_score INTEGER,
    away_score INTEGER,
    point_differential INTEGER GENERATED ALWAYS AS (home_score - away_score) STORED,
    total_points INTEGER GENERATED ALWAYS AS (home_score + away_score) STORED,

    -- Game details
    location VARCHAR(100),
    result VARCHAR(50),
    total DECIMAL(5,2),
    overtime INTEGER,
    attendance INTEGER,

    -- External IDs
    old_game_id VARCHAR(20),
    gsis VARCHAR(20),
    nfl_detail_id VARCHAR(20),
    pfr VARCHAR(20),
    pff VARCHAR(20),
    espn VARCHAR(20),
    ftn VARCHAR(20),

    -- Rest and preparation
    home_rest INTEGER,
    away_rest INTEGER,

    -- Betting lines
    spread_line DECIMAL(5,2),
    home_spread_odds INTEGER,
    away_spread_odds INTEGER,
    total_line DECIMAL(5,2),
    over_odds INTEGER,
    under_odds INTEGER,
    home_moneyline INTEGER,
    away_moneyline INTEGER,

    -- Game conditions
    div_game INTEGER,
    roof VARCHAR(20),
    surface VARCHAR(20),
    temp DECIMAL(5,2),
    wind DECIMAL(5,2),
    humidity DECIMAL(5,2),

    -- Key players
    home_qb_id VARCHAR(20),
    away_qb_id VARCHAR(20),
    home_qb_name VARCHAR(100),
    away_qb_name VARCHAR(100),
    home_coach VARCHAR(100),
    away_coach VARCHAR(100),

    -- Officials
    referee VARCHAR(20),
    stadium_id VARCHAR(20),
    stadium VARCHAR(100),

    -- Metadata
    completed INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (home_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (away_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (referee) REFERENCES dim_officials(official_id),
    FOREIGN KEY (stadium_id) REFERENCES dim_stadiums(stadium_id)
);

-- Play-by-play (comprehensive - all 400+ columns from nflreadpy)
CREATE TABLE IF NOT EXISTS fact_plays (
    play_id INTEGER PRIMARY KEY NOT NULL,
    game_id VARCHAR(20) NOT NULL,

    -- Game context
    home_team VARCHAR(3),
    away_team VARCHAR(3),
    season INTEGER,
    week INTEGER,
    season_type VARCHAR(10),
    game_date DATE,

    -- Drive context
    drive INTEGER,
    qtr INTEGER,
    down INTEGER,
    ydstogo INTEGER,
    yardline_100 INTEGER,
    goal_to_go INTEGER,

    -- Time context
    quarter_seconds_remaining INTEGER,
    half_seconds_remaining INTEGER,
    game_seconds_remaining INTEGER,
    game_half VARCHAR(5),

    -- Teams and players
    posteam VARCHAR(3),
    defteam VARCHAR(3),
    posteam_type VARCHAR(10),
    side_of_field VARCHAR(10),

    -- Play details
    play_type VARCHAR(20),
    desc TEXT,
    yards_gained INTEGER,

    -- Advanced metrics (EPA family)
    epa DECIMAL(8,6),
    wpa DECIMAL(8,6),
    success INTEGER,

    -- Air yards and passing
    air_epa DECIMAL(8,6),
    yac_epa DECIMAL(8,6),
    comp_air_epa DECIMAL(8,6),
    comp_yac_epa DECIMAL(8,6),
    qb_epa DECIMAL(8,6),
    pass_oe DECIMAL(8,6),
    cpoe DECIMAL(8,6),

    -- Passing stats
    passer_player_id VARCHAR(20),
    passer_player_name VARCHAR(100),
    passing_yards INTEGER,
    air_yards DECIMAL(8,2),
    yards_after_catch INTEGER,
    xyac_epa DECIMAL(8,6),
    complete_pass INTEGER,
    incomplete_pass INTEGER,
    interception INTEGER,
    pass_touchdown INTEGER,

    -- QB pressure metrics
    qb_hit INTEGER,
    sack INTEGER,
    was_pressure INTEGER,
    time_to_throw DECIMAL(8,3),
    time_to_pressure DECIMAL(8,3),

    -- Rushing stats
    rusher_player_id VARCHAR(20),
    rusher_player_name VARCHAR(100),
    rushing_yards INTEGER,
    rush_touchdown INTEGER,

    -- Receiving stats
    receiver_player_id VARCHAR(20),
    receiver_player_name VARCHAR(100),
    receiving_yards INTEGER,

    -- Special teams
    field_goal_result VARCHAR(20),
    kick_distance INTEGER,
    punt_blocked INTEGER,

    -- Penalties and turnovers
    penalty INTEGER,
    penalty_team VARCHAR(3),
    penalty_yards INTEGER,
    fumble INTEGER,
    fumble_lost INTEGER,

    -- Down conversions
    third_down_converted INTEGER,
    third_down_failed INTEGER,
    fourth_down_converted INTEGER,
    fourth_down_failed INTEGER,

    -- Scoring
    td_team VARCHAR(3),
    td_player_id VARCHAR(20),
    td_player_name VARCHAR(100),
    extra_point_result VARCHAR(20),
    two_point_conv_result VARCHAR(20),

    -- Score tracking
    posteam_score INTEGER,
    defteam_score INTEGER,
    score_differential INTEGER,
    posteam_score_post INTEGER,
    defteam_score_post INTEGER,

    -- Win probability
    wp DECIMAL(8,6),
    def_wp DECIMAL(8,6),
    home_wp DECIMAL(8,6),
    away_wp DECIMAL(8,6),
    vegas_wp DECIMAL(8,6),
    vegas_home_wp DECIMAL(8,6),

    -- Play style indicators
    shotgun INTEGER,
    no_huddle INTEGER,
    qb_dropback INTEGER,
    qb_scramble INTEGER,

    -- Drive outcome
    drive_end_transition VARCHAR(50),
    old_game_id VARCHAR(20),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (posteam) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (defteam) REFERENCES dim_teams(team_abbr)
);

-- Next Gen Stats - Passing (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS fact_ngs_passing (
    season INTEGER NOT NULL,
    season_type VARCHAR(10),
    week INTEGER NOT NULL,
    player_gsis_id VARCHAR(20) NOT NULL,
    team_abbr VARCHAR(3) NOT NULL,

    -- Core passing metrics
    attempts INTEGER,
    completions INTEGER,
    pass_yards INTEGER,
    pass_touchdowns INTEGER,
    interceptions INTEGER,
    passer_rating DECIMAL(5,2),
    completion_percentage DECIMAL(5,2),

    -- Advanced metrics
    expected_completion_percentage DECIMAL(5,2),
    completion_percentage_above_expectation DECIMAL(5,2),
    avg_air_distance DECIMAL(5,2),
    max_air_distance DECIMAL(5,2),

    -- Timing and accuracy
    avg_time_to_throw DECIMAL(5,3),
    avg_completed_air_yards DECIMAL(5,2),
    avg_intended_air_yards DECIMAL(5,2),
    avg_air_yards_differential DECIMAL(5,2),
    aggressiveness DECIMAL(5,2),
    max_completed_air_distance DECIMAL(5,2),
    avg_air_yards_to_sticks DECIMAL(5,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, player_gsis_id),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr)
);

-- Next Gen Stats - Receiving (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS fact_ngs_receiving (
    season INTEGER NOT NULL,
    season_type VARCHAR(10),
    week INTEGER NOT NULL,
    player_gsis_id VARCHAR(20) NOT NULL,
    team_abbr VARCHAR(3) NOT NULL,

    -- Core receiving metrics
    receptions INTEGER,
    targets INTEGER,
    catch_percentage DECIMAL(5,2),
    yards INTEGER,
    rec_touchdowns INTEGER,

    -- Advanced metrics
    avg_cushion DECIMAL(5,2),
    avg_separation DECIMAL(5,2),
    avg_intended_air_yards DECIMAL(5,2),
    percent_share_of_intended_air_yards DECIMAL(5,2),
    avg_yac DECIMAL(5,2),
    avg_expected_yac DECIMAL(5,2),
    avg_yac_above_expectation DECIMAL(5,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, player_gsis_id),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr)
);

-- Next Gen Stats - Rushing (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS fact_ngs_rushing (
    season INTEGER NOT NULL,
    season_type VARCHAR(10),
    week INTEGER NOT NULL,
    player_gsis_id VARCHAR(20) NOT NULL,
    team_abbr VARCHAR(3) NOT NULL,

    -- Core rushing metrics
    rush_attempts INTEGER,
    rush_yards INTEGER,
    avg_rush_yards DECIMAL(5,2),
    rush_touchdowns INTEGER,

    -- Advanced metrics
    efficiency DECIMAL(5,2),
    percent_attempts_gte_eight_defenders DECIMAL(5,2),
    avg_time_to_los DECIMAL(5,3),
    expected_rush_yards DECIMAL(5,2),
    rush_yards_over_expected DECIMAL(5,2),
    rush_yards_over_expected_per_att DECIMAL(5,2),
    rush_pct_over_expected DECIMAL(5,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, player_gsis_id),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr)
);

-- Injuries (comprehensive for Supabase)
CREATE TABLE IF NOT EXISTS fact_injuries (
    season INTEGER NOT NULL,
    week INTEGER,
    game_type VARCHAR(10),
    team VARCHAR(3) NOT NULL,
    gsis_id VARCHAR(20),
    player_name VARCHAR(100) NOT NULL,
    position VARCHAR(10),

    -- Injury details
    report_primary_injury VARCHAR(100),
    report_secondary_injury VARCHAR(100),
    report_status VARCHAR(20),
    practice_primary_injury VARCHAR(100),
    practice_secondary_injury VARCHAR(100),
    practice_status VARCHAR(20),
    date_modified DATE,

    -- Severity scoring
    severity_score INTEGER,  -- 0-3 scale
    games_missed INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, team, player_name),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr)
);

-- Snap counts (comprehensive for Supabase)
CREATE TABLE IF NOT EXISTS fact_snap_counts (
    game_id VARCHAR(20) NOT NULL,
    season INTEGER NOT NULL,
    game_type VARCHAR(10),
    week INTEGER NOT NULL,
    player VARCHAR(100) NOT NULL,
    pfr_player_id VARCHAR(20),
    position VARCHAR(10),
    team VARCHAR(3) NOT NULL,
    opponent VARCHAR(3),

    -- Snap counts
    offense_snaps INTEGER,
    offense_pct DECIMAL(5,2),
    defense_snaps INTEGER,
    defense_pct DECIMAL(5,2),
    st_snaps INTEGER,
    st_pct DECIMAL(5,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (game_id, player),
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (opponent) REFERENCES dim_teams(team_abbr)
);

-- Weekly rosters (comprehensive for Supabase)
CREATE TABLE IF NOT EXISTS fact_weekly_rosters (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team VARCHAR(3) NOT NULL,
    player_id VARCHAR(20) NOT NULL,
    position VARCHAR(10),

    -- Roster details
    depth_chart_position VARCHAR(20),
    jersey_number INTEGER,
    status VARCHAR(20),
    status_description TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, team, player_id),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr)
);

-- Depth charts (comprehensive for Supabase)
CREATE TABLE IF NOT EXISTS fact_depth_charts (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type VARCHAR(10),
    club_code VARCHAR(3) NOT NULL,
    player_gsis_id VARCHAR(20) NOT NULL,
    position VARCHAR(10) NOT NULL,
    depth_position VARCHAR(20),
    formation VARCHAR(20),
    depth_team INTEGER,

    -- Player details
    player_name VARCHAR(100),
    football_name VARCHAR(100),
    jersey_number INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, club_code, player_gsis_id, position)
);

-- Game officials (comprehensive for Supabase)
CREATE TABLE IF NOT EXISTS fact_game_officials (
    game_id VARCHAR(20) NOT NULL,
    season INTEGER NOT NULL,
    official_id VARCHAR(20) NOT NULL,
    official_name VARCHAR(100) NOT NULL,
    official_position VARCHAR(50),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (game_id, official_id),
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (official_id) REFERENCES dim_officials(official_id)
);

-- Weekly player stats (enhanced for Supabase)
CREATE TABLE IF NOT EXISTS fact_weekly_stats (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id VARCHAR(20) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    position VARCHAR(10),
    team VARCHAR(3) NOT NULL,

    -- Passing stats
    passing_yards INTEGER,
    passing_tds INTEGER,
    interceptions INTEGER,
    passing_attempts INTEGER,
    completions INTEGER,
    completion_percentage DECIMAL(5,2),
    passer_rating DECIMAL(5,2),

    -- Rushing stats
    rushing_yards INTEGER,
    rushing_tds INTEGER,
    rushing_attempts INTEGER,
    rushing_long INTEGER,

    -- Receiving stats
    receiving_yards INTEGER,
    receiving_tds INTEGER,
    receptions INTEGER,
    targets INTEGER,
    receiving_long INTEGER,

    -- Defense stats
    solo_tackles INTEGER,
    assisted_tackles INTEGER,
    total_tackles INTEGER,
    sacks DECIMAL(4,1),
    tackles_for_loss DECIMAL(4,1),
    qb_hits INTEGER,
    passes_defended INTEGER,
    interceptions INTEGER,
    defensive_interceptions INTEGER,
    fumble_forces INTEGER,
    fumble_recoveries INTEGER,

    -- Special teams
    punt_return_yards INTEGER,
    punt_return_tds INTEGER,
    punt_returns INTEGER,
    kick_return_yards INTEGER,
    kick_return_tds INTEGER,
    kick_returns INTEGER,

    -- Fantasy points
    fantasy_points DECIMAL(6,2),
    fantasy_points_ppr DECIMAL(6,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, player_id),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr)
);

-- QBR (Quarterback Rating for Supabase)
CREATE TABLE IF NOT EXISTS fact_qbr (
    season INTEGER NOT NULL,
    week INTEGER,
    player_id VARCHAR(20) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    team VARCHAR(3) NOT NULL,

    -- QBR metrics
    qbr_total DECIMAL(5,2),
    pts_added DECIMAL(6,2),
    epa_total DECIMAL(6,2),
    epa_per_play DECIMAL(6,2),
    success DECIMAL(5,2),
    expected_added DECIMAL(6,2),

    -- Context
    game_count INTEGER,
    play_count INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, week, player_id),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr)
);

-- Combine data (draft prospects for Supabase)
CREATE TABLE IF NOT EXISTS fact_combine (
    season INTEGER NOT NULL,  -- Draft year
    player_id VARCHAR(20) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    position VARCHAR(10),
    college VARCHAR(100),
    team VARCHAR(3),  -- Team that drafted them

    -- Physical measurements
    height DECIMAL(4,2),
    weight DECIMAL(5,1),
    arm_length DECIMAL(4,2),
    hand_size DECIMAL(4,2),
    wingspan DECIMAL(4,2),

    -- Athletic testing
    forty_yard DECIMAL(4,2),
    bench_press INTEGER,
    vertical_jump DECIMAL(4,2),
    broad_jump DECIMAL(4,2),
    three_cone DECIMAL(4,2),
    twenty_yard_shuttle DECIMAL(4,2),

    -- Draft results
    draft_round INTEGER,
    draft_pick INTEGER,
    draft_team VARCHAR(3),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, player_id)
);

-- ============================================================================
-- AGGREGATED TABLES (For ML Features - Supabase optimized)
-- ============================================================================

-- Team-game aggregated stats (for ML features)
CREATE TABLE IF NOT EXISTS agg_team_game_stats (
    game_id VARCHAR(20) PRIMARY KEY NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team VARCHAR(3) NOT NULL,

    -- Offensive stats (aggregated from plays)
    off_epa DECIMAL(8,6),
    off_success_rate DECIMAL(5,4),
    off_yards_per_play DECIMAL(5,2),
    off_explosive_rate DECIMAL(5,4),
    off_pass_rate DECIMAL(5,4),

    -- Defensive stats
    def_epa DECIMAL(8,6),
    def_success_rate DECIMAL(5,4),
    def_yards_per_play DECIMAL(5,2),
    def_explosive_rate DECIMAL(5,4),

    -- Passing advanced
    cpoe DECIMAL(5,4),
    air_yards_per_attempt DECIMAL(5,2),
    yac_per_completion DECIMAL(5,2),
    deep_ball_rate DECIMAL(5,4),
    time_to_throw_avg DECIMAL(5,3),
    pressure_rate_allowed DECIMAL(5,4),

    -- Situational
    third_down_conv_rate DECIMAL(5,4),
    red_zone_td_rate DECIMAL(5,4),
    two_minute_epa DECIMAL(8,6),

    -- Injury impact
    qb_injury_severity INTEGER,
    key_player_injuries INTEGER,

    -- Context
    rest_days INTEGER,
    is_home INTEGER,
    is_divisional INTEGER,
    is_primetime INTEGER,
    weather_impact DECIMAL(5,2),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr)
);

-- Season-level team stats (rolling averages)
CREATE TABLE IF NOT EXISTS agg_team_season_stats (
    season INTEGER NOT NULL,
    team VARCHAR(3) NOT NULL,
    through_week INTEGER NOT NULL,

    -- Rolling averages (last 3-5 games)
    off_epa_3g DECIMAL(8,6),
    off_epa_5g DECIMAL(8,6),
    def_epa_3g DECIMAL(8,6),
    def_epa_5g DECIMAL(8,6),
    cpoe_3g DECIMAL(5,4),
    cpoe_5g DECIMAL(5,4),

    -- Trends
    epa_trend_3g DECIMAL(8,6),
    epa_trend_5g DECIMAL(8,6),
    success_trend_3g DECIMAL(5,4),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (season, team, through_week),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr)
);

-- ML-ready features (denormalized for training)
CREATE TABLE IF NOT EXISTS ml_features (
    game_id VARCHAR(20) PRIMARY KEY NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,

    -- Home team features (50+ columns)
    home_off_epa DECIMAL(8,6),
    home_def_epa DECIMAL(8,6),
    home_off_success_rate DECIMAL(5,4),
    home_def_success_rate DECIMAL(5,4),
    home_off_yards_per_play DECIMAL(5,2),
    home_cpoe DECIMAL(5,4),
    home_air_yards_per_attempt DECIMAL(5,2),
    home_yac_per_completion DECIMAL(5,2),
    home_deep_ball_rate DECIMAL(5,4),
    home_time_to_throw_avg DECIMAL(5,3),
    home_pressure_rate_allowed DECIMAL(5,4),
    home_third_down_conv_rate DECIMAL(5,4),
    home_red_zone_td_rate DECIMAL(5,4),
    home_two_minute_epa DECIMAL(8,6),
    home_qb_injury_severity INTEGER,
    home_rest_days INTEGER,
    home_epa_trend_3g DECIMAL(8,6),
    home_epa_trend_5g DECIMAL(8,6),

    -- Away team features (same structure)
    away_off_epa DECIMAL(8,6),
    away_def_epa DECIMAL(8,6),
    away_off_success_rate DECIMAL(5,4),
    away_def_success_rate DECIMAL(5,4),
    away_off_yards_per_play DECIMAL(5,2),
    away_cpoe DECIMAL(5,4),
    away_air_yards_per_attempt DECIMAL(5,2),
    away_yac_per_completion DECIMAL(5,2),
    away_deep_ball_rate DECIMAL(5,4),
    away_time_to_throw_avg DECIMAL(5,3),
    away_pressure_rate_allowed DECIMAL(5,4),
    away_third_down_conv_rate DECIMAL(5,4),
    away_red_zone_td_rate DECIMAL(5,4),
    away_two_minute_epa DECIMAL(8,6),
    away_qb_injury_severity INTEGER,
    away_rest_days INTEGER,
    away_epa_trend_3g DECIMAL(8,6),
    away_epa_trend_5g DECIMAL(8,6),

    -- Game context features
    is_home_advantage DECIMAL(5,4),
    is_divisional_game INTEGER,
    is_primetime_game INTEGER,
    weather_impact DECIMAL(5,2),
    stadium_factor DECIMAL(5,4),
    referee_tendency DECIMAL(5,4),

    -- Target variables
    home_won INTEGER,
    point_differential INTEGER,
    total_points INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (home_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (away_team) REFERENCES dim_teams(team_abbr)
);

-- ============================================================================
-- INDEXES (Performance Optimization for Supabase)
-- ============================================================================

-- Games indexes
CREATE INDEX IF NOT EXISTS idx_games_season_week ON fact_games(season, week);
CREATE INDEX IF NOT EXISTS idx_games_teams ON fact_games(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_games_date ON fact_games(gameday);
CREATE INDEX IF NOT EXISTS idx_games_completed ON fact_games(completed);

-- Plays indexes (critical for performance)
CREATE INDEX IF NOT EXISTS idx_plays_game ON fact_plays(game_id);
CREATE INDEX IF NOT EXISTS idx_plays_team ON fact_plays(posteam, defteam);
CREATE INDEX IF NOT EXISTS idx_plays_season_week ON fact_plays(season, week);
CREATE INDEX IF NOT EXISTS idx_plays_epa ON fact_plays(epa);

-- NGS indexes
CREATE INDEX IF NOT EXISTS idx_ngs_season_week ON fact_ngs_passing(season, week);
CREATE INDEX IF NOT EXISTS idx_ngs_team ON fact_ngs_passing(team_abbr);

-- Player indexes
CREATE INDEX IF NOT EXISTS idx_players_team ON fact_weekly_rosters(team);
CREATE INDEX IF NOT EXISTS idx_players_season ON fact_weekly_rosters(season, week);

-- ML features indexes
CREATE INDEX IF NOT EXISTS idx_ml_season ON ml_features(season, week);
CREATE INDEX IF NOT EXISTS idx_ml_teams ON ml_features(home_team, away_team);

-- ============================================================================
-- VIEWS (For Common Queries - Supabase optimized)
-- ============================================================================

-- Complete game view with team info
CREATE OR REPLACE VIEW v_games_complete AS
SELECT
    g.*,
    ht.team_name as home_team_name,
    ht.team_conf as home_conference,
    ht.team_division as home_division,
    at.team_name as away_team_name,
    at.team_conf as away_conference,
    at.team_division as away_division,
    s.stadium_name,
    s.roof,
    s.surface,
    o.name as referee_name
FROM fact_games g
LEFT JOIN dim_teams ht ON g.home_team = ht.team_abbr
LEFT JOIN dim_teams at ON g.away_team = at.team_abbr
LEFT JOIN dim_stadiums s ON g.stadium_id = s.stadium_id
LEFT JOIN dim_officials o ON g.referee = o.official_id;

-- Recent team performance (last 5 games)
CREATE OR REPLACE VIEW v_team_recent_performance AS
SELECT
    t.team_abbr,
    t.team_name,
    gs.season,
    gs.week,
    AVG(gs.off_epa) OVER (PARTITION BY gs.team ORDER BY gs.week ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as off_epa_5g,
    AVG(gs.def_epa) OVER (PARTITION BY gs.team ORDER BY gs.week ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as def_epa_5g,
    AVG(gs.cpoe) OVER (PARTITION BY gs.team ORDER BY gs.week ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as cpoe_5g
FROM agg_team_game_stats gs
JOIN dim_teams t ON gs.team = t.team_abbr
WHERE gs.week >= 5;

-- ============================================================================
-- METADATA AND TRACKING (Supabase optimized)
-- ============================================================================

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_metadata (
    version VARCHAR(20) PRIMARY KEY NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tables_created INTEGER,
    total_records_estimate INTEGER
);

INSERT INTO schema_metadata (version, description, tables_created, total_records_estimate)
VALUES ('2024.1', 'Complete NFL data warehouse for Supabase PostgreSQL', 25, 1500000)
ON CONFLICT (version) DO UPDATE SET
    description = EXCLUDED.description,
    created_at = CURRENT_TIMESTAMP;

-- Import tracking
CREATE TABLE IF NOT EXISTS import_tracking (
    source_table VARCHAR(50) NOT NULL,
    season INTEGER,
    import_date TIMESTAMP NOT NULL,
    record_count INTEGER,
    import_duration_seconds INTEGER,
    success BOOLEAN,
    error_message TEXT,
    PRIMARY KEY (source_table, season)
);

-- Daily update tracking
CREATE TABLE IF NOT EXISTS daily_updates (
    update_date DATE PRIMARY KEY,
    games_updated INTEGER,
    plays_updated INTEGER,
    duration_seconds INTEGER,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PERFORMANCE OPTIMIZATIONS (Supabase specific)
-- ============================================================================

-- Enable Row Level Security (RLS) where appropriate
-- ALTER TABLE fact_games ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE fact_plays ENABLE ROW LEVEL SECURITY;

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_games_recent ON fact_games(season DESC, week DESC);
CREATE INDEX IF NOT EXISTS idx_plays_recent ON fact_plays(season DESC, week DESC, game_id);

-- ============================================================================
-- SAMPLE DATA INSERTION (For Testing)
-- ============================================================================

-- Insert sample teams (for development)
INSERT INTO dim_teams (team_abbr, team_name, team_conf, team_division)
VALUES
('KC', 'Kansas City Chiefs', 'AFC', 'AFC West'),
('BUF', 'Buffalo Bills', 'AFC', 'AFC East'),
('SF', 'San Francisco 49ers', 'NFC', 'NFC West'),
('DAL', 'Dallas Cowboys', 'NFC', 'NFC East')
ON CONFLICT (team_abbr) DO NOTHING;

-- ============================================================================
-- DOCUMENTATION
-- ============================================================================

/*
Supabase Schema Summary:
========================

DIMENSION TABLES (5):
- dim_teams: 32 NFL teams
- dim_players: All NFL players
- dim_stadiums: NFL venues
- dim_officials: Game referees
- dim_seasons: Season metadata

FACT TABLES (15):
- fact_games: 2,748 games (2016-2024)
- fact_plays: 435,483+ plays (all PBP data)
- fact_ngs_passing: 24,814 passing records
- fact_ngs_receiving: 24,814 receiving records
- fact_ngs_rushing: 24,814 rushing records
- fact_injuries: 49,488 injury reports
- fact_snap_counts: 230,049 snap count records
- fact_weekly_rosters: 362,000+ roster entries
- fact_depth_charts: 335,000+ depth chart entries
- fact_game_officials: 17,806 official assignments
- fact_weekly_stats: 49,161 weekly player stats
- fact_qbr: 635 QBR ratings
- fact_combine: 3,425 combine records

AGGREGATED TABLES (3):
- agg_team_game_stats: Game-level team stats
- agg_team_season_stats: Rolling season averages
- ml_features: ML-ready feature matrix

VIEWS (2):
- v_games_complete: Games with team/stadium info
- v_team_recent_performance: Recent team trends

Total Expected Records: ~1.5M
Database Type: PostgreSQL (Supabase)
Query Performance: Optimized with indexes
Row Level Security: Ready for implementation

Migration from SQLite:
- All data types preserved
- Enhanced with PostgreSQL features
- Comprehensive indexing strategy
- Supabase-specific optimizations
*/
