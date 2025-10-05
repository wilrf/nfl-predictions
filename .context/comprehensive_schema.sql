-- ============================================================================
-- COMPREHENSIVE NFL DATA WAREHOUSE SCHEMA
-- ============================================================================
-- Purpose: Store ALL available NFL data from nfl_data_py (2016-2024)
-- Total data: ~1.13M records across 12 data sources
-- Design: Normalized star schema with fact and dimension tables
-- Database: SQLite (sufficient for ~500MB dataset)
-- ============================================================================

-- ============================================================================
-- DIMENSION TABLES (Reference Data)
-- ============================================================================

-- Teams master table
CREATE TABLE IF NOT EXISTS dim_teams (
    team_abbr TEXT PRIMARY KEY NOT NULL,
    team_name TEXT NOT NULL,
    team_id TEXT,
    team_nick TEXT,
    team_conf TEXT CHECK(team_conf IN ('AFC', 'NFC')),
    team_division TEXT CHECK(team_division IN ('NFC North', 'NFC South', 'NFC East', 'NFC West', 'AFC North', 'AFC South', 'AFC East', 'AFC West')),
    team_color TEXT,
    team_color2 TEXT,
    team_color3 TEXT,
    team_color4 TEXT,
    team_logo_wikipedia TEXT,
    team_logo_espn TEXT,
    team_wordmark TEXT,
    team_conference_logo TEXT,
    team_league_logo TEXT,
    team_logo_squared TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Players master table (unified from rosters)
CREATE TABLE IF NOT EXISTS dim_players (
    player_id TEXT PRIMARY KEY NOT NULL,  -- gsis_id
    player_name TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,
    birth_date TEXT,
    height TEXT,
    weight INTEGER,
    college TEXT,
    position TEXT,
    -- External IDs for data joining
    espn_id TEXT,
    sportradar_id TEXT,
    yahoo_id TEXT,
    rotowire_id TEXT,
    pff_id TEXT,
    pfr_id TEXT,
    sleeper_id TEXT,
    -- Metadata
    entry_year INTEGER,
    rookie_year INTEGER,
    headshot_url TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Stadiums/Venues
CREATE TABLE IF NOT EXISTS dim_stadiums (
    stadium_id TEXT PRIMARY KEY NOT NULL,
    stadium_name TEXT NOT NULL,
    location TEXT,
    roof TEXT CHECK(roof IN ('outdoors', 'dome', 'retractable', 'open')),
    surface TEXT,
    capacity INTEGER,
    team_abbr TEXT,
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr)
);

-- Officials (referees)
CREATE TABLE IF NOT EXISTS dim_officials (
    official_id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- FACT TABLES (Core Event Data)
-- ============================================================================

-- Games (central fact table) - ENHANCED from original schema
CREATE TABLE IF NOT EXISTS fact_games (
    game_id TEXT PRIMARY KEY NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT NOT NULL CHECK(game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB', 'PRE')),
    gameday TEXT NOT NULL,  -- Date as YYYY-MM-DD
    weekday TEXT,
    gametime TEXT,

    -- Teams
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,

    -- Scores
    home_score INTEGER,
    away_score INTEGER,
    result INTEGER,  -- Home team point differential
    total INTEGER,   -- Total points scored

    -- Stadium/Weather
    location TEXT,
    stadium_id TEXT,
    stadium TEXT,
    roof TEXT,
    surface TEXT,
    temp REAL,
    wind REAL,

    -- Game conditions
    overtime BOOLEAN DEFAULT 0,
    div_game BOOLEAN DEFAULT 0,  -- Divisional matchup

    -- Coaches & QBs
    home_coach TEXT,
    away_coach TEXT,
    home_qb_id TEXT,
    away_qb_id TEXT,
    home_qb_name TEXT,
    away_qb_name TEXT,

    -- Officials
    referee TEXT,

    -- Rest days
    home_rest INTEGER,
    away_rest INTEGER,

    -- Betting lines (from schedule data)
    spread_line REAL,
    home_spread_odds INTEGER,
    away_spread_odds INTEGER,
    total_line REAL,
    over_odds INTEGER,
    under_odds INTEGER,
    home_moneyline INTEGER,
    away_moneyline INTEGER,

    -- External IDs
    old_game_id TEXT,
    gsis TEXT,
    nfl_detail_id TEXT,
    pfr TEXT,
    pff TEXT,
    espn TEXT,
    ftn TEXT,

    -- Metadata
    completed BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (home_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (away_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (stadium_id) REFERENCES dim_stadiums(stadium_id),
    FOREIGN KEY (home_qb_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (away_qb_id) REFERENCES dim_players(player_id)
);

-- Play-by-play (THE BIG TABLE: ~432K rows × 396 columns)
-- We'll store the most critical columns, not all 396
CREATE TABLE IF NOT EXISTS fact_plays (
    play_id TEXT PRIMARY KEY NOT NULL,
    game_id TEXT NOT NULL,

    -- Game context
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    season_type TEXT,
    week INTEGER,

    -- Drive context
    drive INTEGER,
    qtr INTEGER,
    down INTEGER,
    ydstogo INTEGER,
    yardline_100 INTEGER,  -- Yards to opponent's endzone
    goal_to_go BOOLEAN,

    -- Time
    game_date TEXT,
    quarter_seconds_remaining INTEGER,
    half_seconds_remaining INTEGER,
    game_seconds_remaining INTEGER,
    game_half TEXT,

    -- Teams
    posteam TEXT,  -- Team with possession
    defteam TEXT,  -- Defensive team
    posteam_type TEXT CHECK(posteam_type IN ('home', 'away')),
    side_of_field TEXT,

    -- Play details
    play_type TEXT,  -- pass, run, punt, field_goal, kickoff, etc.
    desc TEXT,  -- Play description
    yards_gained INTEGER,

    -- ========== KEY METRICS (EPA, WPA, Success) ==========
    epa REAL,  -- Expected Points Added (TIER 1)
    wpa REAL,  -- Win Probability Added
    success BOOLEAN,  -- Successful play flag

    -- EPA breakdowns
    air_epa REAL,
    yac_epa REAL,
    comp_air_epa REAL,
    comp_yac_epa REAL,
    qb_epa REAL,

    -- Pass-specific EPA
    pass_oe REAL,  -- Passing over expectation
    cpoe REAL,  -- Completion % over expected (TIER 1)

    -- ========== PASSING METRICS ==========
    passer_player_id TEXT,
    passer_player_name TEXT,
    passing_yards REAL,
    air_yards REAL,
    yards_after_catch REAL,
    xyac_epa REAL,  -- Expected YAC EPA
    complete_pass BOOLEAN,
    incomplete_pass BOOLEAN,
    interception BOOLEAN,
    pass_touchdown BOOLEAN,

    -- Pass rush
    qb_hit BOOLEAN,
    sack BOOLEAN,
    was_pressure BOOLEAN,
    time_to_throw REAL,
    time_to_pressure REAL,

    -- ========== RUSHING METRICS ==========
    rusher_player_id TEXT,
    rusher_player_name TEXT,
    rushing_yards REAL,
    rush_touchdown BOOLEAN,

    -- ========== RECEIVING METRICS ==========
    receiver_player_id TEXT,
    receiver_player_name TEXT,
    receiving_yards REAL,

    -- ========== SPECIAL TEAMS ==========
    field_goal_result TEXT,
    kick_distance INTEGER,
    punt_blocked BOOLEAN,

    -- ========== PENALTIES & TURNOVERS ==========
    penalty BOOLEAN,
    penalty_team TEXT,
    penalty_yards INTEGER,
    fumble BOOLEAN,
    fumble_lost BOOLEAN,

    -- ========== SITUATIONAL ==========
    third_down_converted BOOLEAN,
    third_down_failed BOOLEAN,
    fourth_down_converted BOOLEAN,
    fourth_down_failed BOOLEAN,

    -- ========== SCORING ==========
    touchdown BOOLEAN,
    td_team TEXT,
    td_player_id TEXT,
    td_player_name TEXT,
    extra_point_result TEXT,
    two_point_conv_result TEXT,

    -- ========== SCORE TRACKING ==========
    posteam_score INTEGER,
    defteam_score INTEGER,
    score_differential INTEGER,
    posteam_score_post INTEGER,
    defteam_score_post INTEGER,

    -- ========== PROBABILITIES ==========
    wp REAL,  -- Win probability
    def_wp REAL,
    home_wp REAL,
    away_wp REAL,
    vegas_wp REAL,
    vegas_home_wp REAL,

    -- ========== FORMATION & PERSONNEL ==========
    shotgun BOOLEAN,
    no_huddle BOOLEAN,
    qb_dropback BOOLEAN,
    qb_scramble BOOLEAN,

    -- Drive result
    drive_end_transition TEXT,

    -- Metadata
    old_game_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (home_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (away_team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (passer_player_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (rusher_player_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (receiver_player_id) REFERENCES dim_players(player_id)
);

-- ============================================================================
-- PLAYER PERFORMANCE TABLES
-- ============================================================================

-- Next Gen Stats - Passing (weekly player stats)
CREATE TABLE IF NOT EXISTS fact_ngs_passing (
    ngs_passing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    week INTEGER NOT NULL,
    player_gsis_id TEXT NOT NULL,
    team_abbr TEXT,

    -- NGS Passing Metrics
    avg_time_to_throw REAL,
    avg_completed_air_yards REAL,
    avg_intended_air_yards REAL,
    avg_air_yards_differential REAL,
    aggressiveness REAL,  -- Deep ball rate
    max_completed_air_distance REAL,
    avg_air_yards_to_sticks REAL,

    -- Traditional stats
    attempts INTEGER,
    completions INTEGER,
    pass_yards REAL,
    pass_touchdowns INTEGER,
    interceptions INTEGER,
    passer_rating REAL,
    completion_percentage REAL,

    -- Expected vs Actual
    expected_completion_percentage REAL,
    completion_percentage_above_expectation REAL,  -- CPOE

    -- Air distance
    avg_air_distance REAL,
    max_air_distance REAL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_gsis_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr),
    UNIQUE(season, week, player_gsis_id)
);

-- Next Gen Stats - Receiving
CREATE TABLE IF NOT EXISTS fact_ngs_receiving (
    ngs_receiving_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    week INTEGER NOT NULL,
    player_gsis_id TEXT NOT NULL,
    team_abbr TEXT,

    -- NGS Receiving Metrics
    avg_cushion REAL,  -- Average DB cushion at snap
    avg_separation REAL,  -- Average separation at catch
    avg_intended_air_yards REAL,
    percent_share_of_intended_air_yards REAL,

    -- Traditional stats
    receptions INTEGER,
    targets INTEGER,
    catch_percentage REAL,
    yards REAL,
    rec_touchdowns INTEGER,

    -- YAC metrics
    avg_yac REAL,
    avg_expected_yac REAL,
    avg_yac_above_expectation REAL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_gsis_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr),
    UNIQUE(season, week, player_gsis_id)
);

-- Next Gen Stats - Rushing
CREATE TABLE IF NOT EXISTS fact_ngs_rushing (
    ngs_rushing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    week INTEGER NOT NULL,
    player_gsis_id TEXT NOT NULL,
    team_abbr TEXT,

    -- NGS Rushing Metrics
    efficiency REAL,
    percent_attempts_gte_eight_defenders REAL,  -- Stacked box %
    avg_time_to_los REAL,  -- Time to line of scrimmage

    -- Traditional stats
    rush_attempts INTEGER,
    rush_yards REAL,
    avg_rush_yards REAL,
    rush_touchdowns INTEGER,

    -- Expected vs Actual
    expected_rush_yards REAL,
    rush_yards_over_expected REAL,
    rush_yards_over_expected_per_att REAL,
    rush_pct_over_expected REAL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_gsis_id) REFERENCES dim_players(player_id),
    FOREIGN KEY (team_abbr) REFERENCES dim_teams(team_abbr),
    UNIQUE(season, week, player_gsis_id)
);

-- ============================================================================
-- ROSTER & PERSONNEL TABLES
-- ============================================================================

-- Weekly rosters (who was on the team each week)
CREATE TABLE IF NOT EXISTS fact_weekly_rosters (
    weekly_roster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,
    player_id TEXT NOT NULL,
    position TEXT,
    depth_chart_position TEXT,
    jersey_number INTEGER,
    status TEXT,  -- Active, Inactive, IR, etc.

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (player_id) REFERENCES dim_players(player_id),
    UNIQUE(season, week, team, player_id)
);

-- Depth charts (position depth by week)
CREATE TABLE IF NOT EXISTS fact_depth_charts (
    depth_chart_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT,
    club_code TEXT NOT NULL,
    player_gsis_id TEXT NOT NULL,

    position TEXT NOT NULL,  -- Listed position
    depth_position TEXT,  -- Depth chart position (e.g., LWR, RWR)
    formation TEXT,  -- Offense/Defense
    depth_team TEXT,  -- Which depth chart (offense/defense/special)

    player_name TEXT,
    football_name TEXT,
    jersey_number INTEGER,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (club_code) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (player_gsis_id) REFERENCES dim_players(player_id)
);

-- Snap counts (participation %)
CREATE TABLE IF NOT EXISTS fact_snap_counts (
    snap_count_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    game_type TEXT NOT NULL,
    week INTEGER NOT NULL,

    player TEXT NOT NULL,
    pfr_player_id TEXT,
    position TEXT,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,

    offense_snaps INTEGER,
    offense_pct REAL,
    defense_snaps INTEGER,
    defense_pct REAL,
    st_snaps INTEGER,
    st_pct REAL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (opponent) REFERENCES dim_teams(team_abbr),
    UNIQUE(game_id, player, team)
);

-- ============================================================================
-- INJURY & AVAILABILITY TABLES
-- ============================================================================

-- Injury reports (weekly injury status)
CREATE TABLE IF NOT EXISTS fact_injuries (
    injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT NOT NULL,
    team TEXT NOT NULL,

    gsis_id TEXT,  -- Player ID
    player_name TEXT NOT NULL,
    position TEXT,

    -- Injury details
    report_primary_injury TEXT,
    report_secondary_injury TEXT,
    report_status TEXT,  -- Out, Doubtful, Questionable, etc.

    -- Practice participation
    practice_primary_injury TEXT,
    practice_secondary_injury TEXT,
    practice_status TEXT,

    date_modified TEXT,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    FOREIGN KEY (gsis_id) REFERENCES dim_players(player_id)
);

-- ============================================================================
-- GAME OFFICIALS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_game_officials (
    game_official_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    official_id TEXT NOT NULL,
    official_name TEXT NOT NULL,
    official_position TEXT,  -- Referee, Umpire, etc.

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    FOREIGN KEY (official_id) REFERENCES dim_officials(official_id)
);

-- ============================================================================
-- BETTING OPERATIONS TABLES (from original schema)
-- ============================================================================

-- Odds snapshots from The Odds API
CREATE TABLE IF NOT EXISTS fact_odds_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    snapshot_type TEXT NOT NULL CHECK(snapshot_type IN ('opening', 'midweek', 'closing', 'current')),
    book TEXT NOT NULL,

    spread_home REAL NOT NULL,
    spread_away REAL NOT NULL,
    total_over REAL NOT NULL,
    total_under REAL NOT NULL,
    ml_home INTEGER NOT NULL,
    ml_away INTEGER NOT NULL,
    spread_odds_home INTEGER NOT NULL DEFAULT -110,
    spread_odds_away INTEGER NOT NULL DEFAULT -110,
    total_odds_over INTEGER NOT NULL DEFAULT -110,
    total_odds_under INTEGER NOT NULL DEFAULT -110,

    api_credits_used INTEGER NOT NULL DEFAULT 1,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id),
    UNIQUE(game_id, timestamp, book)
);

-- Our betting suggestions
CREATE TABLE IF NOT EXISTS fact_suggestions (
    suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    bet_type TEXT NOT NULL CHECK(bet_type IN ('spread', 'total', 'moneyline')),
    selection TEXT NOT NULL,
    line REAL NOT NULL,
    odds INTEGER NOT NULL,

    confidence REAL NOT NULL CHECK(confidence >= 50 AND confidence <= 90),
    margin REAL NOT NULL CHECK(margin >= 0 AND margin <= 30),
    edge REAL NOT NULL CHECK(edge >= 0.02),
    kelly_fraction REAL NOT NULL CHECK(kelly_fraction > 0 AND kelly_fraction <= 0.25),
    model_probability REAL NOT NULL CHECK(model_probability > 0 AND model_probability < 1),
    market_probability REAL NOT NULL CHECK(market_probability > 0 AND market_probability < 1),

    suggested_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    outcome TEXT CHECK(outcome IN ('win', 'loss', 'push', 'pending', 'void')),
    actual_result REAL,
    pnl REAL,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id)
);

-- CLV tracking
CREATE TABLE IF NOT EXISTS fact_clv_tracking (
    clv_id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,

    opening_line REAL NOT NULL,
    closing_line REAL NOT NULL,
    our_line REAL NOT NULL,
    clv_points REAL NOT NULL,
    clv_percentage REAL NOT NULL,
    beat_closing BOOLEAN NOT NULL,

    tracked_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (suggestion_id) REFERENCES fact_suggestions(suggestion_id),
    FOREIGN KEY (game_id) REFERENCES fact_games(game_id)
);

-- Model predictions
CREATE TABLE IF NOT EXISTS fact_model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    prediction_type TEXT NOT NULL CHECK(prediction_type IN ('spread', 'total', 'win_prob')),

    home_prediction REAL NOT NULL,
    away_prediction REAL NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),

    features_hash TEXT NOT NULL,
    features_json TEXT,  -- JSON of features used

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES fact_games(game_id)
);

-- ============================================================================
-- AGGREGATED FEATURE TABLES (for ML training)
-- ============================================================================

-- Team-level EPA stats (aggregated from play-by-play)
CREATE TABLE IF NOT EXISTS agg_team_epa_stats (
    team_epa_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,

    -- Games played
    games_played INTEGER NOT NULL,

    -- Offensive EPA
    off_plays INTEGER,
    off_epa_per_play REAL,
    off_epa_pass REAL,
    off_epa_run REAL,
    off_success_rate REAL,
    off_explosive_play_rate REAL,  -- EPA > 1.0

    -- Defensive EPA (lower is better)
    def_plays INTEGER,
    def_epa_per_play REAL,
    def_epa_pass REAL,
    def_epa_run REAL,
    def_success_rate REAL,

    -- Efficiency
    third_down_pct REAL,
    redzone_td_pct REAL,
    turnover_rate REAL,

    -- Situational
    epa_first_down REAL,
    epa_second_down REAL,
    epa_third_down REAL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    UNIQUE(season, week, team)
);

-- Team-level injury impact scores
CREATE TABLE IF NOT EXISTS agg_team_injury_scores (
    injury_score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,

    total_injuries INTEGER,
    key_injuries INTEGER,  -- QB, RB, WR, TE
    injury_severity_score REAL,  -- Weighted by position & status
    starter_injury_pct REAL,  -- % of usual starters injured

    qb_injured BOOLEAN,
    rb1_injured BOOLEAN,
    wr1_injured BOOLEAN,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    UNIQUE(season, week, team)
);

-- Rolling team performance (for time-series features)
CREATE TABLE IF NOT EXISTS agg_team_rolling_stats (
    rolling_stats_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,

    -- Rolling EPA (last N games)
    epa_last_3_games REAL,
    epa_last_5_games REAL,
    epa_ewma REAL,  -- Exponential weighted moving average

    -- Rolling success rates
    success_rate_last_3 REAL,
    success_rate_last_5 REAL,

    -- Form/momentum
    wins_last_3 INTEGER,
    wins_last_5 INTEGER,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team) REFERENCES dim_teams(team_abbr),
    UNIQUE(season, week, team)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Games indexes
CREATE INDEX IF NOT EXISTS idx_games_season_week ON fact_games(season, week);
CREATE INDEX IF NOT EXISTS idx_games_gameday ON fact_games(gameday);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON fact_games(home_team);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON fact_games(away_team);

-- Plays indexes (CRITICAL for 432K rows)
CREATE INDEX IF NOT EXISTS idx_plays_game_id ON fact_plays(game_id);
CREATE INDEX IF NOT EXISTS idx_plays_posteam ON fact_plays(posteam);
CREATE INDEX IF NOT EXISTS idx_plays_season_week ON fact_plays(week, season_type);
CREATE INDEX IF NOT EXISTS idx_plays_passer ON fact_plays(passer_player_id);
CREATE INDEX IF NOT EXISTS idx_plays_rusher ON fact_plays(rusher_player_id);
CREATE INDEX IF NOT EXISTS idx_plays_play_type ON fact_plays(play_type);

-- NGS indexes
CREATE INDEX IF NOT EXISTS idx_ngs_pass_player ON fact_ngs_passing(player_gsis_id, season, week);
CREATE INDEX IF NOT EXISTS idx_ngs_rec_player ON fact_ngs_receiving(player_gsis_id, season, week);
CREATE INDEX IF NOT EXISTS idx_ngs_rush_player ON fact_ngs_rushing(player_gsis_id, season, week);

-- Roster indexes
CREATE INDEX IF NOT EXISTS idx_roster_player ON fact_weekly_rosters(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_roster_team ON fact_weekly_rosters(team, season, week);

-- Injury indexes
CREATE INDEX IF NOT EXISTS idx_injuries_team ON fact_injuries(team, season, week);
CREATE INDEX IF NOT EXISTS idx_injuries_player ON fact_injuries(gsis_id, season, week);

-- Snap count indexes
CREATE INDEX IF NOT EXISTS idx_snaps_game ON fact_snap_counts(game_id);
CREATE INDEX IF NOT EXISTS idx_snaps_player ON fact_snap_counts(player, season, week);

-- Aggregation table indexes
CREATE INDEX IF NOT EXISTS idx_team_epa_lookup ON agg_team_epa_stats(season, week, team);
CREATE INDEX IF NOT EXISTS idx_injury_scores_lookup ON agg_team_injury_scores(season, week, team);
CREATE INDEX IF NOT EXISTS idx_rolling_stats_lookup ON agg_team_rolling_stats(season, week, team);

-- ============================================================================
-- VIEWS FOR QUICK ACCESS
-- ============================================================================

-- ML Training ready view (game-level features)
CREATE VIEW IF NOT EXISTS vw_ml_training_data AS
SELECT
    g.game_id,
    g.season,
    g.week,
    g.game_type,
    g.gameday,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.spread_line,
    g.total_line,

    -- Home team EPA
    h_epa.off_epa_per_play as home_off_epa,
    h_epa.def_epa_per_play as home_def_epa,
    h_epa.off_success_rate as home_off_success_rate,
    h_epa.third_down_pct as home_third_down_pct,
    h_epa.redzone_td_pct as home_redzone_td_pct,
    h_epa.games_played as home_games_played,

    -- Away team EPA
    a_epa.off_epa_per_play as away_off_epa,
    a_epa.def_epa_per_play as away_def_epa,
    a_epa.off_success_rate as away_off_success_rate,
    a_epa.third_down_pct as away_third_down_pct,
    a_epa.redzone_td_pct as away_redzone_td_pct,
    a_epa.games_played as away_games_played,

    -- Stadium
    g.stadium,
    g.roof,
    g.temp,
    g.wind,

    -- Derived features
    CASE WHEN g.roof = 'outdoors' THEN 1 ELSE 0 END as is_outdoor,
    CASE WHEN g.div_game = 1 THEN 1 ELSE 0 END as is_divisional,
    CASE WHEN g.game_type != 'REG' THEN 1 ELSE 0 END as is_playoff

FROM fact_games g
LEFT JOIN agg_team_epa_stats h_epa ON g.home_team = h_epa.team AND g.season = h_epa.season AND g.week = h_epa.week
LEFT JOIN agg_team_epa_stats a_epa ON g.away_team = a_epa.team AND g.season = a_epa.season AND g.week = a_epa.week
WHERE g.completed = 1;

-- Current week games with latest stats
CREATE VIEW IF NOT EXISTS vw_current_week_games AS
SELECT
    g.*,
    h_epa.off_epa_per_play as home_off_epa,
    a_epa.off_epa_per_play as away_off_epa
FROM fact_games g
LEFT JOIN agg_team_epa_stats h_epa ON g.home_team = h_epa.team AND g.season = h_epa.season AND (g.week - 1) = h_epa.week
LEFT JOIN agg_team_epa_stats a_epa ON g.away_team = a_epa.team AND g.season = a_epa.season AND (g.week - 1) = a_epa.week
WHERE g.completed = 0
ORDER BY g.gameday, g.gametime;

-- ============================================================================
-- METADATA TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_metadata (
    version TEXT PRIMARY KEY NOT NULL,
    description TEXT,
    total_tables INTEGER,
    total_records_estimate INTEGER,
    date_range_start INTEGER,  -- First season
    date_range_end INTEGER,    -- Last season
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial metadata
INSERT OR REPLACE INTO schema_metadata (version, description, total_tables, total_records_estimate, date_range_start, date_range_end)
VALUES (
    'v1.0.0',
    'Comprehensive NFL data warehouse - All nfl_data_py sources (2016-2024)',
    27,  -- Number of tables (excluding views)
    1130000,  -- ~1.13M total records
    2016,
    2024
);
