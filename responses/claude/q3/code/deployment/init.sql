-- Initialize NFL Betting Database

-- Create schema
CREATE SCHEMA IF NOT EXISTS nfl;

-- Set search path
SET search_path TO nfl, public;

-- Create tables
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20),
    probability_home_cover FLOAT,
    probability_over FLOAT,
    confidence FLOAT,
    kelly_fraction_home FLOAT,
    kelly_fraction_over FLOAT,
    recommended_bet_home FLOAT,
    recommended_bet_over FLOAT,
    expected_value_home FLOAT,
    expected_value_over FLOAT,
    features JSONB,
    metadata JSONB
);

CREATE INDEX idx_predictions_game_id ON predictions(game_id);
CREATE INDEX idx_predictions_time ON predictions(prediction_time);

CREATE TABLE IF NOT EXISTS game_results (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) UNIQUE NOT NULL,
    game_date DATE,
    home_team VARCHAR(10),
    away_team VARCHAR(10),
    home_score INTEGER,
    away_score INTEGER,
    spread_result VARCHAR(10),
    total_result VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_game_results_game_id ON game_results(game_id);
CREATE INDEX idx_game_results_date ON game_results(game_date);

CREATE TABLE IF NOT EXISTS betting_history (
    id SERIAL PRIMARY KEY,
    bet_id VARCHAR(50) UNIQUE,
    game_id VARCHAR(50) REFERENCES game_results(game_id),
    bet_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bet_type VARCHAR(20),
    stake DECIMAL(10, 2),
    odds DECIMAL(5, 3),
    probability FLOAT,
    result VARCHAR(10),
    profit DECIMAL(10, 2),
    bankroll_after DECIMAL(10, 2),
    model_version VARCHAR(20),
    metadata JSONB
);

CREATE INDEX idx_betting_history_game_id ON betting_history(game_id);
CREATE INDEX idx_betting_history_time ON betting_history(bet_time);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    evaluation_date DATE DEFAULT CURRENT_DATE,
    model_version VARCHAR(20),
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    log_loss FLOAT,
    calibration_error FLOAT,
    total_bets INTEGER,
    wins INTEGER,
    losses INTEGER,
    roi DECIMAL(5, 2),
    profit DECIMAL(10, 2),
    metadata JSONB
);

CREATE INDEX idx_model_performance_date ON model_performance(evaluation_date);

CREATE TABLE IF NOT EXISTS feature_drift (
    id SERIAL PRIMARY KEY,
    check_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_name VARCHAR(100),
    psi_score FLOAT,
    drift_status VARCHAR(20),
    baseline_mean FLOAT,
    current_mean FLOAT,
    baseline_std FLOAT,
    current_std FLOAT,
    metadata JSONB
);

CREATE INDEX idx_feature_drift_time ON feature_drift(check_time);
CREATE INDEX idx_feature_drift_feature ON feature_drift(feature_name);

-- Create views for analysis
CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(bet_time) as date,
    COUNT(*) as total_bets,
    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
    SUM(stake) as total_staked,
    SUM(profit) as total_profit,
    SUM(profit) / NULLIF(SUM(stake), 0) as roi,
    AVG(probability) as avg_confidence
FROM betting_history
GROUP BY DATE(bet_time)
ORDER BY date DESC;

CREATE OR REPLACE VIEW model_comparison AS
SELECT 
    model_version,
    AVG(accuracy) as avg_accuracy,
    AVG(auc_roc) as avg_auc,
    AVG(roi) as avg_roi,
    SUM(profit) as total_profit,
    COUNT(*) as n_evaluations
FROM model_performance
GROUP BY model_version
ORDER BY avg_roi DESC;

-- Create functions
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_game_results_updated_at
BEFORE UPDATE ON game_results
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON SCHEMA nfl TO nfl_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nfl TO nfl_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA nfl TO nfl_user;
