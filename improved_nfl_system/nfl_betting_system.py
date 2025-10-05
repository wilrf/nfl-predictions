"""
NFL Betting System V2.0
Complete rewrite addressing all statistical and architectural inefficiencies
Built on evidence-based practices from professional betting syndicates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BetOpportunity:
    """Represents a validated betting opportunity"""
    game_id: str
    bet_type: str
    selection: str
    market_odds: float
    fair_odds: float
    edge: float
    kelly_size: float
    confidence: float
    features: Dict[str, float]
    timestamp: datetime
    clv_potential: float  # Closing line value potential
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value of the bet"""
        win_prob = 1 / self.fair_odds
        return (win_prob * (self.market_odds - 1)) - (1 - win_prob)


class DataCache:
    """Redis-based caching system for expensive computations"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        self.ttl = ttl
    
    def get_or_compute(self, key: str, compute_func, *args, **kwargs):
        """Get from cache or compute and cache"""
        cached = self.redis_client.get(key)
        if cached:
            logger.info(f"Cache hit for {key}")
            return json.loads(cached)
        
        logger.info(f"Cache miss for {key}, computing...")
        result = compute_func(*args, **kwargs)
        self.redis_client.setex(key, self.ttl, json.dumps(result))
        return result
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)


class MarketEfficiencyAnalyzer:
    """Analyzes market efficiency and identifies inefficiencies"""
    
    def __init__(self):
        self.sharp_books = ['pinnacle', 'circa', 'bookmaker']
        self.soft_books = ['draftkings', 'fanduel', 'betmgm', 'caesars']
        
    def calculate_no_vig_probability(self, odds1: float, odds2: float) -> Tuple[float, float]:
        """Remove bookmaker vig to get true probabilities"""
        # Convert American odds to decimal if needed
        if odds1 < 0:
            odds1 = 1 + (100 / abs(odds1))
        elif odds1 > 0:
            odds1 = 1 + (odds1 / 100)
            
        if odds2 < 0:
            odds2 = 1 + (100 / abs(odds2))
        elif odds2 > 0:
            odds2 = 1 + (odds2 / 100)
        
        # Calculate implied probabilities
        prob1 = 1 / odds1
        prob2 = 1 / odds2
        
        # Remove vig using multiplicative method
        total_prob = prob1 + prob2
        vig = (total_prob - 1) / 2
        
        fair_prob1 = prob1 / (1 + vig)
        fair_prob2 = prob2 / (1 + vig)
        
        # Normalize to sum to 1
        fair_prob1 = fair_prob1 / (fair_prob1 + fair_prob2)
        fair_prob2 = fair_prob2 / (fair_prob1 + fair_prob2)
        
        return fair_prob1, fair_prob2
    
    def detect_steam_move(self, line_history: pd.DataFrame) -> bool:
        """Detect legitimate steam moves vs head fakes"""
        if len(line_history) < 3:
            return False
        
        # Look for coordinated movement across multiple sharp books
        sharp_movement = 0
        for book in self.sharp_books:
            if book in line_history.columns:
                recent_change = line_history[book].iloc[-1] - line_history[book].iloc[-3]
                if abs(recent_change) >= 1.0:  # 1 point move
                    sharp_movement += 1
        
        # Legitimate steam shows in 3+ sharp books
        return sharp_movement >= 3
    
    def calculate_clv_potential(self, current_line: float, predicted_close: float) -> float:
        """Estimate potential closing line value"""
        return predicted_close - current_line


class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.calibrator = None
        self.feature_importance = {}
        self.validation_metrics = {}
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train the model with proper validation"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities"""
        pass
    
    def calibrate_probabilities(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calibrate model probabilities using isotonic regression"""
        raw_probs = self.model.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, y_val)
        logger.info(f"Calibrated {self.model_type} model probabilities")
    
    def validate_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Calculate comprehensive validation metrics"""
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        
        probs = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'brier_score': brier_score_loss(y_test, probs),
            'log_loss': log_loss(y_test, probs),
            'auc_roc': roc_auc_score(y_test, probs),
            'calibration_error': self._calculate_calibration_error(y_test, probs)
        }
        
        self.validation_metrics = metrics
        return metrics
    
    def _calculate_calibration_error(self, y_true: pd.Series, y_pred: np.ndarray, n_bins=10) -> float:
        """Calculate expected calibration error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class SpreadModel(BaseModel):
    """XGBoost model for spread predictions with proper validation"""
    
    def __init__(self):
        super().__init__('spread')
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train with early stopping on validation set"""
        eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Calibrate probabilities
        self.calibrate_probabilities(X_val, y_val)
        
        # Store feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        logger.info(f"Trained spread model with {self.model.n_estimators} trees")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities"""
        raw_probs = self.model.predict_proba(X)
        if self.calibrator:
            calibrated = self.calibrator.transform(raw_probs[:, 1])
            return np.column_stack([1 - calibrated, calibrated])
        return raw_probs


class EnsembleModel:
    """Combines multiple models with weighted voting"""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble prediction"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred * weight)
        
        ensemble_prob = np.sum(predictions, axis=0)
        return np.column_stack([1 - ensemble_prob, ensemble_prob])
    
    def get_prediction_variance(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate prediction variance across models for uncertainty estimation"""
        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(X)[:, 1])
        
        return np.var(predictions, axis=0)


class FeatureEngineer:
    """Advanced feature engineering with market-derived features"""
    
    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor
        self.feature_stats = {}
        
    def create_features(self, game_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features = pd.DataFrame()
        
        # Temporal decay weighted averages
        features = self._add_decay_weighted_features(features, game_data)
        
        # Relative strength matchup features
        features = self._add_matchup_features(features, game_data)
        
        # Market-derived probability features
        features = self._add_market_features(features, market_data)
        
        # Situational features
        features = self._add_situational_features(features, game_data)
        
        # Advanced stats (EPA, DVOA, etc.)
        features = self._add_advanced_stats(features, game_data)
        
        return features
    
    def _add_decay_weighted_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal decay to recent performance"""
        for col in ['yards_per_play', 'points_per_game', 'third_down_pct']:
            if col in data.columns:
                weights = np.exp(-self.decay_factor * np.arange(len(data)))
                weights = weights / weights.sum()
                features[f'{col}_decay_weighted'] = np.average(data[col], weights=weights)
        
        return features
    
    def _add_matchup_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Create relative matchup features"""
        # Example: home offensive efficiency vs away defensive efficiency
        if 'home_off_efficiency' in data.columns and 'away_def_efficiency' in data.columns:
            features['matchup_advantage'] = data['home_off_efficiency'] - data['away_def_efficiency']
        
        return features
    
    def _add_market_features(self, features: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from betting market"""
        if not market_data.empty:
            # Line movement
            features['line_movement'] = market_data['current_line'] - market_data['opening_line']
            
            # Market consensus
            features['market_consensus'] = market_data['pinnacle_line'] if 'pinnacle_line' in market_data.columns else 0
            
            # Betting percentage vs money percentage divergence
            if 'bet_pct' in market_data.columns and 'money_pct' in market_data.columns:
                features['sharp_divergence'] = market_data['money_pct'] - market_data['bet_pct']
        
        return features
    
    def _add_situational_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add game situation features"""
        situation_cols = ['rest_days', 'travel_distance', 'division_game', 'prime_time']
        for col in situation_cols:
            if col in data.columns:
                features[col] = data[col]
        
        return features
    
    def _add_advanced_stats(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistics"""
        advanced_cols = ['epa_per_play', 'success_rate', 'dvoa_offense', 'dvoa_defense']
        for col in advanced_cols:
            if col in data.columns:
                features[col] = data[col]
        
        return features


class KellyCalculator:
    """Sophisticated Kelly Criterion implementation for correlated bets"""
    
    def __init__(self, max_kelly_fraction: float = 0.25):
        self.max_kelly_fraction = max_kelly_fraction
        
    def calculate_portfolio_kelly(self, opportunities: List[BetOpportunity], 
                                   correlation_matrix: np.ndarray,
                                   bankroll: float) -> Dict[str, float]:
        """Calculate Kelly sizes for portfolio of correlated bets"""
        n_bets = len(opportunities)
        
        if n_bets == 0:
            return {}
        
        # Extract edges and odds
        edges = np.array([opp.edge for opp in opportunities])
        odds = np.array([opp.market_odds - 1 for opp in opportunities])
        
        # Adjust for correlation
        if n_bets > 1 and correlation_matrix is not None:
            # Use Monte Carlo simulation for correlated Kelly
            sizes = self._monte_carlo_kelly(edges, odds, correlation_matrix)
        else:
            # Standard Kelly for independent bets
            sizes = self._standard_kelly(edges, odds)
        
        # Apply fractional Kelly and constraints
        sizes = sizes * self.max_kelly_fraction
        
        # Cap individual bet sizes at 5% of bankroll
        max_bet = 0.05 * bankroll
        sizes = np.minimum(sizes * bankroll, max_bet) / bankroll
        
        # Create output dictionary
        kelly_sizes = {}
        for i, opp in enumerate(opportunities):
            kelly_sizes[opp.game_id] = sizes[i]
        
        return kelly_sizes
    
    def _standard_kelly(self, edges: np.ndarray, odds: np.ndarray) -> np.ndarray:
        """Standard Kelly calculation for independent bets"""
        win_probs = (edges + 1) / (odds + 1)
        kelly_fractions = (win_probs * odds - (1 - win_probs)) / odds
        return np.maximum(kelly_fractions, 0)  # No negative bets
    
    def _monte_carlo_kelly(self, edges: np.ndarray, odds: np.ndarray, 
                           correlation_matrix: np.ndarray, n_sims: int = 10000) -> np.ndarray:
        """Monte Carlo simulation for correlated Kelly sizing"""
        n_bets = len(edges)
        
        # Generate correlated outcomes
        win_probs = (edges + 1) / (odds + 1)
        
        # Simulate correlated bet outcomes
        from scipy.stats import multivariate_normal
        mean = win_probs
        samples = multivariate_normal.rvs(mean=mean, cov=correlation_matrix, size=n_sims)
        
        # Convert to binary outcomes
        outcomes = samples > 0.5
        
        # Test different size combinations
        best_sizes = np.zeros(n_bets)
        best_growth = -np.inf
        
        # Grid search over reasonable Kelly fractions
        for trial in range(100):
            trial_sizes = np.random.uniform(0, 0.05, n_bets)
            
            # Calculate growth for these sizes
            growth = self._calculate_growth(outcomes, trial_sizes, odds)
            
            if growth > best_growth:
                best_growth = growth
                best_sizes = trial_sizes
        
        return best_sizes
    
    def _calculate_growth(self, outcomes: np.ndarray, sizes: np.ndarray, odds: np.ndarray) -> float:
        """Calculate expected log growth"""
        returns = np.where(outcomes, sizes * odds, -sizes)
        total_returns = np.sum(returns, axis=1)
        log_growth = np.mean(np.log(1 + total_returns))
        return log_growth


class RiskManager:
    """Enforces risk constraints and portfolio limits"""
    
    def __init__(self, config: Dict):
        self.max_bets_per_week = config.get('max_weekly_bets', 25)
        self.max_game_exposure = config.get('max_game_exposure', 0.15)
        self.correlation_limit = config.get('correlation_limit', 0.30)
        self.stop_loss_threshold = config.get('stop_loss', -0.08)
        self.current_exposure = {}
        
    def validate_portfolio(self, opportunities: List[BetOpportunity], 
                           existing_bets: List[BetOpportunity]) -> List[BetOpportunity]:
        """Apply risk constraints to bet selection"""
        validated = []
        game_exposures = {}
        
        # Sort by edge to prioritize best opportunities
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        
        for opp in opportunities:
            # Check bet count limit
            if len(validated) + len(existing_bets) >= self.max_bets_per_week:
                logger.warning(f"Reached max weekly bets ({self.max_bets_per_week})")
                break
            
            # Check game exposure
            game_id = opp.game_id.split('_')[0]  # Extract base game ID
            current_game_exposure = game_exposures.get(game_id, 0) + opp.kelly_size
            
            if current_game_exposure > self.max_game_exposure:
                logger.warning(f"Skipping bet due to game exposure limit: {game_id}")
                continue
            
            # Check correlation with existing bets
            if not self._check_correlation_limit(opp, validated):
                logger.warning(f"Skipping bet due to correlation limit")
                continue
            
            # Add to validated portfolio
            validated.append(opp)
            game_exposures[game_id] = current_game_exposure
        
        return validated
    
    def _check_correlation_limit(self, new_bet: BetOpportunity, 
                                 existing: List[BetOpportunity]) -> bool:
        """Check if new bet violates correlation limits"""
        # Simplified check - in practice would use actual correlation matrix
        same_game_bets = [b for b in existing if b.game_id.split('_')[0] == new_bet.game_id.split('_')[0]]
        return len(same_game_bets) < 2  # Max 2 bets per game
    
    def check_stop_loss(self, current_pnl: float, bankroll: float) -> bool:
        """Check if stop loss should trigger"""
        return (current_pnl / bankroll) < self.stop_loss_threshold


class CLVTracker:
    """Tracks and analyzes Closing Line Value performance"""
    
    def __init__(self):
        self.clv_history = []
        self.weekly_reports = []
        
    def record_bet(self, bet: BetOpportunity, opening_line: float, closing_line: float):
        """Record CLV for a bet"""
        clv = closing_line - opening_line
        
        record = {
            'timestamp': bet.timestamp,
            'game_id': bet.game_id,
            'bet_type': bet.bet_type,
            'opening_line': opening_line,
            'closing_line': closing_line,
            'clv': clv,
            'clv_pct': clv / abs(opening_line) if opening_line != 0 else 0,
            'edge': bet.edge,
            'result': None  # To be updated after game
        }
        
        self.clv_history.append(record)
        logger.info(f"CLV recorded: {clv:.2f} points ({record['clv_pct']:.1%})")
        
        return record
    
    def generate_weekly_report(self) -> Dict:
        """Generate comprehensive CLV report"""
        if not self.clv_history:
            return {}
        
        df = pd.DataFrame(self.clv_history)
        
        report = {
            'week': datetime.now().isocalendar()[1],
            'total_bets': len(df),
            'avg_clv': df['clv'].mean(),
            'avg_clv_pct': df['clv_pct'].mean(),
            'positive_clv_rate': (df['clv'] > 0).mean(),
            'clv_distribution': {
                'p25': df['clv_pct'].quantile(0.25),
                'p50': df['clv_pct'].quantile(0.50),
                'p75': df['clv_pct'].quantile(0.75)
            },
            'correlation_with_edge': df['clv'].corr(df['edge']),
            'by_bet_type': df.groupby('bet_type')['clv_pct'].agg(['mean', 'std']).to_dict()
        }
        
        self.weekly_reports.append(report)
        return report
    
    def get_clv_trend(self, window: int = 20) -> pd.Series:
        """Calculate rolling CLV trend"""
        if len(self.clv_history) < window:
            return pd.Series()
        
        df = pd.DataFrame(self.clv_history)
        return df['clv_pct'].rolling(window=window).mean()
    
    def validate_edge_calibration(self) -> Dict:
        """Check if predicted edges align with CLV"""
        df = pd.DataFrame(self.clv_history)
        
        # Group by edge buckets
        df['edge_bucket'] = pd.cut(df['edge'], bins=[0, 0.02, 0.03, 0.05, 1.0], 
                                   labels=['2-3%', '3-5%', '5%+', 'Invalid'])
        
        calibration = df.groupby('edge_bucket').agg({
            'clv_pct': ['mean', 'std', 'count']
        })
        
        return calibration.to_dict()


class PropsModel(BaseModel):
    """Specialized model for player props with player-level features"""
    
    def __init__(self):
        super().__init__('props')
        self.model = xgb.XGBRegressor(  # Regression for continuous props
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        )
        self.player_features = None
        
    def create_player_features(self, player_data: pd.DataFrame, 
                              game_context: pd.DataFrame) -> pd.DataFrame:
        """Create player-specific features for props"""
        features = pd.DataFrame()
        
        # Recent performance with decay
        features['avg_last3'] = player_data['stat'].iloc[-3:].mean()
        features['avg_last5'] = player_data['stat'].iloc[-5:].mean()
        features['trend'] = player_data['stat'].iloc[-3:].mean() - player_data['stat'].iloc[-6:-3].mean()
        
        # Usage metrics
        features['snap_pct_last3'] = player_data['snap_pct'].iloc[-3:].mean()
        features['target_share_last3'] = player_data['target_share'].iloc[-3:].mean()
        features['redzone_share_last3'] = player_data['rz_share'].iloc[-3:].mean()
        
        # Matchup specific
        features['vs_defense_rank'] = game_context['opp_def_rank_vs_position']
        features['pace_factor'] = game_context['expected_plays'] / 65  # Normalized to average
        
        # Game script projection
        features['expected_game_script'] = game_context['spread'] / 7  # Normalized spread
        features['passing_game_script'] = 1 if game_context['spread'] < -3 else 0
        
        # Injury/role adjustments
        features['teammates_out'] = game_context['key_teammates_injured']
        features['usage_boost_potential'] = features['teammates_out'] * 0.1
        
        return features
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train props model with player features"""
        # Store feature set for monitoring
        self.player_features = X.columns.tolist()
        
        # Train with early stopping
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        # For props, we need distribution not just point estimate
        self.fit_prediction_intervals(X_val, y_val)
        
        logger.info(f"Trained props model with {len(self.player_features)} features")
    
    def fit_prediction_intervals(self, X: pd.DataFrame, y: pd.Series):
        """Fit prediction intervals for props (critical for over/under)"""
        predictions = self.model.predict(X)
        residuals = y - predictions
        
        # Store residual distribution for interval estimation
        self.residual_std = residuals.std()
        self.residual_quantiles = {
            'p25': residuals.quantile(0.25),
            'p50': residuals.quantile(0.50),
            'p75': residuals.quantile(0.75)
        }
    
    def predict_proba(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """Predict probability of going over/under a line"""
        point_estimate = self.model.predict(X)
        
        # Use residual distribution to estimate probability
        from scipy.stats import norm
        
        # Probability of going over
        prob_over = 1 - norm.cdf(line, loc=point_estimate, scale=self.residual_std)
        prob_under = 1 - prob_over
        
        return np.column_stack([prob_under, prob_over])


class ModelVersionControl:
    """Track model versions and experiments"""
    
    def __init__(self, storage_path: str = 'model_artifacts/'):
        self.storage_path = storage_path
        self.experiment_log = []
        os.makedirs(storage_path, exist_ok=True)
        
    def save_model(self, model: BaseModel, metadata: Dict) -> str:
        """Save model with versioning"""
        import hashlib
        
        # Create version hash
        version_data = f"{datetime.now().isoformat()}_{json.dumps(metadata)}"
        version_hash = hashlib.md5(version_data.encode()).hexdigest()[:8]
        
        # Save model
        model_path = os.path.join(self.storage_path, f"{model.model_type}_{version_hash}.pkl")
        joblib.dump(model, model_path)
        
        # Log experiment
        experiment = {
            'version': version_hash,
            'timestamp': datetime.now().isoformat(),
            'model_type': model.model_type,
            'metadata': metadata,
            'validation_metrics': model.validation_metrics,
            'feature_importance': model.feature_importance,
            'path': model_path
        }
        
        self.experiment_log.append(experiment)
        
        # Save experiment log
        log_path = os.path.join(self.storage_path, 'experiments.json')
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
        
        logger.info(f"Saved model version {version_hash}")
        return version_hash
    
    def load_model(self, version: str) -> BaseModel:
        """Load specific model version"""
        for exp in self.experiment_log:
            if exp['version'] == version:
                return joblib.load(exp['path'])
        raise ValueError(f"Model version {version} not found")
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        exp1 = next(e for e in self.experiment_log if e['version'] == version1)
        exp2 = next(e for e in self.experiment_log if e['version'] == version2)
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metrics_diff': {
                metric: exp2['validation_metrics'].get(metric, 0) - exp1['validation_metrics'].get(metric, 0)
                for metric in exp1['validation_metrics']
            },
            'feature_changes': set(exp2['metadata'].get('features', [])) - set(exp1['metadata'].get('features', []))
        }
        
        return comparison


class NFLBettingSystem:
    """Main orchestration class for the complete betting system"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.cache = DataCache()
        self.market_analyzer = MarketEfficiencyAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.kelly_calculator = KellyCalculator()
        self.risk_manager = RiskManager(self.config['risk'])
        self.clv_tracker = CLVTracker()  # New CLV tracking
        self.model_version_control = ModelVersionControl()  # New versioning
        
        # Initialize models
        self.models = self._initialize_models()
        self.ensemble = EnsembleModel(list(self.models.values()))
        
        # Tracking
        self.bet_history = []
        self.performance_metrics = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_models(self) -> Dict[str, BaseModel]:
        """Initialize all prediction models"""
        models = {
            'spread': SpreadModel(),
            'props': PropsModel(),  # Added props model
            # Add other models as implemented
        }
        return models
    
    def run_weekly_analysis(self, week: int) -> List[BetOpportunity]:
        """Execute complete weekly betting analysis with CLV tracking"""
        logger.info(f"Running analysis for Week {week}")
        
        # 0. Data health check
        health_status = self._check_data_health(week)
        if not health_status['passed']:
            logger.error(f"Data health check failed: {health_status['issues']}")
            return []
        
        # 1. Fetch and cache data
        game_data = self._fetch_game_data(week)
        market_data = self._fetch_market_data(week)
        
        # Store opening lines for CLV tracking
        self._store_opening_lines(market_data)
        
        # 2. Engineer features
        features = self.feature_engineer.create_features(game_data, market_data)
        
        # 3. Generate predictions
        predictions = self.ensemble.predict_proba(features)
        variances = self.ensemble.get_prediction_variance(features)
        
        # 4. Identify opportunities
        opportunities = self._identify_opportunities(
            predictions, variances, market_data
        )
        
        # 5. Calculate Kelly sizes
        correlation_matrix = self._estimate_correlation_matrix(opportunities)
        kelly_sizes = self.kelly_calculator.calculate_portfolio_kelly(
            opportunities, correlation_matrix, self.config['bankroll']
        )
        
        # Update opportunities with Kelly sizes
        for opp in opportunities:
            opp.kelly_size = kelly_sizes.get(opp.game_id, 0)
        
        # 6. Apply risk management
        validated_bets = self.risk_manager.validate_portfolio(
            opportunities, self.bet_history
        )
        
        # 7. Track and log
        self._track_bets(validated_bets)
        self._log_performance()
        
        # 8. Generate CLV report
        clv_report = self.clv_tracker.generate_weekly_report()
        logger.info(f"Weekly CLV: {clv_report.get('avg_clv_pct', 0):.2%}")
        
        return validated_bets
    
    def _check_data_health(self, week: int) -> Dict:
        """Comprehensive data health check before betting"""
        issues = []
        checks = {
            'injury_freshness': False,
            'odds_freshness': False,
            'data_completeness': False,
            'timestamp_integrity': False
        }
        
        # Check injury data freshness
        injury_data = self._fetch_injury_data_with_timestamp(week)
        if not injury_data.empty:
            latest_update = injury_data['event_timestamp'].max()
            hours_old = (datetime.now() - latest_update).total_seconds() / 3600
            if hours_old > 24:
                issues.append(f"Injury data {hours_old:.1f} hours old")
            else:
                checks['injury_freshness'] = True
        else:
            issues.append("No injury data available")
        
        # Check odds freshness
        market_data = self._fetch_market_data(week)
        if not market_data.empty and 'timestamp' in market_data.columns:
            latest_odds = market_data['timestamp'].max()
            minutes_old = (datetime.now() - latest_odds).total_seconds() / 60
            if minutes_old > 30:
                issues.append(f"Odds data {minutes_old:.0f} minutes old")
            else:
                checks['odds_freshness'] = True
        else:
            issues.append("No odds data available")
        
        # Check data completeness
        required_fields = ['spread', 'total', 'moneyline']
        if not market_data.empty:
            missing_fields = [f for f in required_fields if f not in market_data.columns]
            if missing_fields:
                issues.append(f"Missing fields: {missing_fields}")
            else:
                checks['data_completeness'] = True
        
        # Check timestamp integrity
        if not market_data.empty and 'timestamp' in market_data.columns:
            future_timestamps = market_data[market_data['timestamp'] > datetime.now()]
            if not future_timestamps.empty:
                issues.append(f"Future timestamps detected: {len(future_timestamps)} entries")
            else:
                checks['timestamp_integrity'] = True
        
        passed = all(checks.values())
        
        return {
            'passed': passed,
            'checks': checks,
            'issues': issues,
            'timestamp': datetime.now()
        }
    
    def _fetch_injury_data_with_timestamp(self, week: int) -> pd.DataFrame:
        """Fetch injury data with proper timestamps"""
        # This would call the enhanced injury fetching from data_pipeline
        # Placeholder for integration
        return pd.DataFrame()
    
    def _store_opening_lines(self, market_data: pd.DataFrame):
        """Store opening lines for CLV calculation"""
        for _, row in market_data.iterrows():
            key = f"opening_{row['game_id']}_{row['bet_type']}"
            self.cache.redis_client.setex(key, 86400 * 7, json.dumps({
                'line': row['spread'],
                'total': row['total'],
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
            }))
    
    def track_closing_lines(self, week: int):
        """Track closing lines for CLV calculation"""
        closing_data = self._fetch_market_data(week)
        
        for bet in self.bet_history:
            if bet.timestamp.isocalendar()[1] == week:
                # Get opening line from cache
                key = f"opening_{bet.game_id}_{bet.bet_type}"
                opening_data = self.cache.redis_client.get(key)
                
                if opening_data:
                    opening = json.loads(opening_data)['line']
                    closing = closing_data[closing_data['game_id'] == bet.game_id]['spread'].iloc[0]
                    
                    # Record CLV
                    self.clv_tracker.record_bet(bet, opening, closing)
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report with CLV"""
        if not self.bet_history:
            return {}
        
        df = pd.DataFrame([{
            'game_id': b.game_id,
            'bet_type': b.bet_type,
            'edge': b.edge,
            'kelly_size': b.kelly_size,
            'clv_potential': b.clv_potential,
            'timestamp': b.timestamp
        } for b in self.bet_history])
        
        clv_df = pd.DataFrame(self.clv_tracker.clv_history)
        
        report = {
            'summary': {
                'total_bets': len(df),
                'avg_edge': df['edge'].mean(),
                'avg_kelly': df['kelly_size'].mean(),
                'by_type': df.groupby('bet_type')['edge'].agg(['count', 'mean']).to_dict()
            },
            'clv_analysis': {
                'avg_clv': clv_df['clv'].mean() if not clv_df.empty else 0,
                'positive_clv_rate': (clv_df['clv'] > 0).mean() if not clv_df.empty else 0,
                'clv_trend': self.clv_tracker.get_clv_trend().tolist()
            },
            'model_versions': {
                model_type: self.model_version_control.experiment_log[-1]['version'] 
                if self.model_version_control.experiment_log else 'unknown'
                for model_type in self.models.keys()
            },
            'data_health': self._check_data_health(datetime.now().isocalendar()[1])
        }
        
        return report
    
    def _fetch_game_data(self, week: int) -> pd.DataFrame:
        """Fetch game data with caching"""
        cache_key = f"game_data_week_{week}"
        
        def fetch():
            # Implement actual data fetching
            # This is placeholder
            return pd.DataFrame()
        
        return self.cache.get_or_compute(cache_key, fetch)
    
    def _fetch_market_data(self, week: int) -> pd.DataFrame:
        """Fetch market data from multiple sources"""
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for book in self.market_analyzer.sharp_books + self.market_analyzer.soft_books:
                future = executor.submit(self._fetch_book_odds, book, week)
                futures.append(future)
            
            market_data = pd.DataFrame()
            for future in as_completed(futures):
                book_data = future.result()
                market_data = pd.concat([market_data, book_data])
        
        return market_data
    
    def _fetch_book_odds(self, book: str, week: int) -> pd.DataFrame:
        """Fetch odds from specific book"""
        # Implement actual API call
        # This is placeholder
        return pd.DataFrame()
    
    def _identify_opportunities(self, predictions: np.ndarray, 
                               variances: np.ndarray,
                               market_data: pd.DataFrame) -> List[BetOpportunity]:
        """Identify positive EV betting opportunities"""
        opportunities = []
        
        for i, (pred, var) in enumerate(zip(predictions, variances)):
            # Only consider high-confidence predictions
            if var > 0.1:  # High variance = low confidence
                continue
            
            # Calculate edge vs market
            market_prob = market_data.iloc[i]['implied_probability']
            edge = pred[1] - market_prob
            
            # Minimum edge threshold (2%)
            if edge < 0.02:
                continue
            
            # Calculate CLV potential
            clv = self.market_analyzer.calculate_clv_potential(
                market_data.iloc[i]['current_line'],
                market_data.iloc[i]['predicted_close']
            )
            
            opp = BetOpportunity(
                game_id=market_data.iloc[i]['game_id'],
                bet_type='spread',
                selection=market_data.iloc[i]['team'],
                market_odds=market_data.iloc[i]['odds'],
                fair_odds=1/pred[1],
                edge=edge,
                kelly_size=0,  # Will be calculated
                confidence=pred[1],
                features={},
                timestamp=datetime.now(),
                clv_potential=clv
            )
            
            opportunities.append(opp)
        
        return opportunities
    
    def _estimate_correlation_matrix(self, opportunities: List[BetOpportunity]) -> np.ndarray:
        """Estimate correlation between betting opportunities"""
        n = len(opportunities)
        correlation = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Same game bets are highly correlated
                if opportunities[i].game_id.split('_')[0] == opportunities[j].game_id.split('_')[0]:
                    correlation[i, j] = correlation[j, i] = 0.7
                # Division games have some correlation
                elif 'division' in opportunities[i].features and 'division' in opportunities[j].features:
                    correlation[i, j] = correlation[j, i] = 0.3
        
        return correlation
    
    def _track_bets(self, bets: List[BetOpportunity]):
        """Track placed bets"""
        self.bet_history.extend(bets)
        logger.info(f"Tracked {len(bets)} new bets")
    
    def _log_performance(self):
        """Log system performance metrics"""
        if self.bet_history:
            recent_bets = self.bet_history[-100:]  # Last 100 bets
            
            # Calculate metrics
            win_rate = sum(1 for b in recent_bets if b.edge > 0) / len(recent_bets)
            avg_edge = np.mean([b.edge for b in recent_bets])
            avg_clv = np.mean([b.clv_potential for b in recent_bets])
            
            self.performance_metrics = {
                'win_rate': win_rate,
                'avg_edge': avg_edge,
                'avg_clv': avg_clv,
                'total_bets': len(self.bet_history)
            }
            
            logger.info(f"Performance: WR={win_rate:.2%}, Edge={avg_edge:.2%}, CLV={avg_clv:.2}")
    
    def backtest(self, start_week: int, end_week: int) -> pd.DataFrame:
        """Run walk-forward backtest"""
        results = []
        
        for week in range(start_week, end_week + 1):
            # Train on all data up to this week
            train_data = self._get_historical_data(end_week=week-1)
            
            # Validate on this week
            val_data = self._get_historical_data(week=week)
            
            # Train models
            for model in self.models.values():
                model.train(train_data['X'], train_data['y'], 
                          val_data['X'], val_data['y'])
            
            # Generate predictions
            opportunities = self.run_weekly_analysis(week)
            
            # Evaluate results
            week_results = self._evaluate_week(opportunities, val_data['actual_results'])
            results.append(week_results)
        
        return pd.DataFrame(results)
    
    def _get_historical_data(self, end_week: int = None, week: int = None) -> Dict:
        """Get historical data for training/validation"""
        # Implement historical data retrieval
        # This is placeholder
        return {'X': pd.DataFrame(), 'y': pd.Series(), 'actual_results': pd.DataFrame()}
    
    def _evaluate_week(self, opportunities: List[BetOpportunity], 
                      actual_results: pd.DataFrame) -> Dict:
        """Evaluate betting performance for a week"""
        # Implement evaluation logic
        # This is placeholder
        return {'week_roi': 0, 'win_rate': 0}


if __name__ == "__main__":
    # Example usage
    system = NFLBettingSystem('config/improved_config.json')
    
    # Run weekly analysis
    week_10_bets = system.run_weekly_analysis(week=10)
    
    print(f"Found {len(week_10_bets)} betting opportunities for Week 10")
    for bet in week_10_bets[:5]:
        print(f"  {bet.bet_type} on {bet.selection}: Edge={bet.edge:.2%}, Kelly={bet.kelly_size:.2%}")
    
    # Run backtest
    backtest_results = system.backtest(start_week=1, end_week=9)
    print(f"\nBacktest Results:")
    print(backtest_results.describe())
