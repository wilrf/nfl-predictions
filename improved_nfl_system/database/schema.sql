-- NFL Betting Suggestion System Database Schema
-- FAIL FAST principle: All fields are required unless explicitly nullable

-- Core game information from nfl_data_py
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT NOT NULL CHECK(game_type IN ('REG', 'POST', 'PRE')),
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_time DATETIME NOT NULL,
    stadium TEXT NOT NULL,
    is_outdoor BOOLEAN NOT NULL DEFAULT 0,
    home_score INTEGER,  -- NULL until game complete
    away_score INTEGER,   -- NULL until game complete
    completed BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Odds snapshots from The Odds API
CREATE TABLE IF NOT EXISTS odds_snapshots (
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
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    UNIQUE(game_id, timestamp, book)
);

-- Our betting suggestions
CREATE TABLE IF NOT EXISTS suggestions (
    suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    bet_type TEXT NOT NULL CHECK(bet_type IN ('spread', 'total', 'moneyline')),
    selection TEXT NOT NULL,  -- e.g., 'home', 'away', 'over', 'under'
    line REAL NOT NULL,       -- The line at time of suggestion
    odds INTEGER NOT NULL,    -- American odds format
    confidence REAL NOT NULL CHECK(confidence >= 50 AND confidence <= 90),
    margin REAL NOT NULL CHECK(margin >= 0 AND margin <= 30),
    edge REAL NOT NULL CHECK(edge >= 0.02),  -- Minimum 2% edge
    kelly_fraction REAL NOT NULL CHECK(kelly_fraction > 0 AND kelly_fraction <= 0.25),
    model_probability REAL NOT NULL CHECK(model_probability > 0 AND model_probability < 1),
    market_probability REAL NOT NULL CHECK(market_probability > 0 AND market_probability < 1),
    suggested_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    outcome TEXT CHECK(outcome IN ('win', 'loss', 'push', 'pending', 'void')),
    actual_result REAL,  -- Actual game result for this bet
    pnl REAL,           -- Profit/loss if bet was placed
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Correlation warnings between suggestions
CREATE TABLE IF NOT EXISTS correlation_warnings (
    warning_id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id1 INTEGER NOT NULL,
    suggestion_id2 INTEGER NOT NULL,
    correlation_type TEXT NOT NULL,
    correlation_value REAL NOT NULL CHECK(correlation_value >= 0 AND correlation_value <= 1),
    warning_level TEXT NOT NULL CHECK(warning_level IN ('high', 'moderate', 'low')),
    warning_message TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (suggestion_id1) REFERENCES suggestions(suggestion_id),
    FOREIGN KEY (suggestion_id2) REFERENCES suggestions(suggestion_id),
    CHECK(suggestion_id1 < suggestion_id2)  -- Prevent duplicates
);

-- CLV tracking for our suggestions
CREATE TABLE IF NOT EXISTS clv_tracking (
    clv_id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    opening_line REAL NOT NULL,
    closing_line REAL NOT NULL,
    our_line REAL NOT NULL,      -- Line we got
    clv_points REAL NOT NULL,    -- Closing - Opening
    clv_percentage REAL NOT NULL,
    beat_closing BOOLEAN NOT NULL,
    tracked_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (suggestion_id) REFERENCES suggestions(suggestion_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Model predictions storage
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prediction_type TEXT NOT NULL CHECK(prediction_type IN ('spread', 'total', 'win_prob')),
    home_prediction REAL NOT NULL,
    away_prediction REAL NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    features_hash TEXT NOT NULL,  -- Hash of features used for reproducibility
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- API usage tracking for free tier management
CREATE TABLE IF NOT EXISTS api_usage (
    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_name TEXT NOT NULL DEFAULT 'the_odds_api',
    endpoint TEXT NOT NULL,
    credits_used INTEGER NOT NULL,
    remaining_credits INTEGER NOT NULL,
    request_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    response_code INTEGER NOT NULL,
    error_message TEXT,
    games_fetched INTEGER NOT NULL DEFAULT 0
);

-- Performance tracking
CREATE TABLE IF NOT EXISTS weekly_performance (
    week_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    total_suggestions INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    pushes INTEGER NOT NULL DEFAULT 0,
    win_rate REAL,
    roi REAL,
    avg_confidence REAL,
    avg_margin REAL,
    avg_clv REAL,
    total_pnl REAL,
    high_confidence_win_rate REAL,  -- 80+ confidence
    correlation_warnings_count INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, week)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_games_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_odds_game_time ON odds_snapshots(game_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_odds_snapshot_type ON odds_snapshots(snapshot_type);
CREATE INDEX IF NOT EXISTS idx_suggestions_game ON suggestions(game_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_confidence ON suggestions(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_suggestions_time ON suggestions(suggested_at);
CREATE INDEX IF NOT EXISTS idx_clv_suggestion ON clv_tracking(suggestion_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_time ON api_usage(request_time);
CREATE INDEX IF NOT EXISTS idx_model_predictions_game ON model_predictions(game_id);

-- Views for quick analysis
CREATE VIEW IF NOT EXISTS active_suggestions AS
SELECT
    s.*,
    g.home_team,
    g.away_team,
    g.game_time
FROM suggestions s
JOIN games g ON s.game_id = g.game_id
WHERE s.outcome = 'pending'
ORDER BY s.confidence DESC;

CREATE VIEW IF NOT EXISTS weekly_summary AS
SELECT
    g.season,
    g.week,
    COUNT(DISTINCT s.suggestion_id) as total_bets,
    AVG(s.confidence) as avg_confidence,
    AVG(s.margin) as avg_margin,
    AVG(s.edge) as avg_edge,
    SUM(CASE WHEN s.outcome = 'win' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN s.outcome = 'loss' THEN 1 ELSE 0 END) as losses
FROM suggestions s
JOIN games g ON s.game_id = g.game_id
GROUP BY g.season, g.week;