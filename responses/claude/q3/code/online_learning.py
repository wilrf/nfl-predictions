"""
Online Learning Module for NFL Betting Model
Incremental learning using River library for real-time model updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# River imports for online learning
from river import compose
from river import linear_model
from river import ensemble
from river import preprocessing
from river import optim
from river import metrics as river_metrics
from river import tree
from river import neural_net as nn
from river import drift
from river import stats

# Additional imports
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning system"""
    # Model parameters
    learning_rate: float = 0.01
    l2_regularization: float = 0.001
    
    # Ensemble parameters
    n_models: int = 5
    model_types: List[str] = field(default_factory=lambda: ['sgd', 'tree', 'nn'])
    
    # Update parameters
    batch_size: int = 32
    update_frequency: str = 'weekly'  # 'daily', 'weekly', 'after_each_game'
    min_samples_for_update: int = 10
    
    # Drift detection parameters
    drift_threshold: float = 0.3
    warning_threshold: float = 0.2
    drift_window_size: int = 100
    
    # Performance tracking
    performance_window: int = 50
    metric_names: List[str] = field(default_factory=lambda: ['accuracy', 'log_loss', 'roi'])
    
    # Memory management
    max_memory_size: int = 10000  # Maximum samples to keep in memory
    replay_buffer_size: int = 1000  # Samples for experience replay
    
    # Adaptation parameters
    adaptive_learning: bool = True
    learning_rate_decay: float = 0.995
    concept_drift_reset: bool = True


class OnlineModel:
    """Base class for online learning models"""
    
    def __init__(self, model_type: str, config: OnlineLearningConfig):
        self.model_type = model_type
        self.config = config
        self.model = self._initialize_model(model_type)
        self.performance_history = deque(maxlen=config.performance_window)
        self.samples_seen = 0
        
    def _initialize_model(self, model_type: str):
        """Initialize the online learning model"""
        if model_type == 'sgd':
            return linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=self.config.learning_rate),
                l2=self.config.l2_regularization
            )
        elif model_type == 'tree':
            return tree.HoeffdingTreeClassifier(
                grace_period=50,
                split_confidence=1e-5,
                leaf_prediction='nb'
            )
        elif model_type == 'nn':
            return compose.Pipeline(
                preprocessing.StandardScaler(),
                nn.MLPClassifier(
                    hidden_dims=[50, 30],
                    activations=[nn.ReLU(), nn.ReLU()],
                    optimizer=optim.Adam(lr=self.config.learning_rate),
                    seed=42
                )
            )
        elif model_type == 'passive_aggressive':
            return linear_model.PAClassifier(
                C=1.0,
                mode=1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def update(self, x: Dict, y: int) -> float:
        """Update model with single sample"""
        # Make prediction before update
        y_pred = self.model.predict_proba_one(x)
        prob = y_pred.get(1, 0.5) if y_pred else 0.5
        
        # Update model
        self.model.learn_one(x, y)
        self.samples_seen += 1
        
        # Track performance
        loss = -np.log(prob if y == 1 else 1 - prob)
        self.performance_history.append({
            'loss': loss,
            'prediction': prob,
            'actual': y,
            'timestamp': datetime.now()
        })
        
        return loss
    
    def predict(self, x: Dict) -> float:
        """Make prediction"""
        y_pred = self.model.predict_proba_one(x)
        return y_pred.get(1, 0.5) if y_pred else 0.5
    
    def get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics"""
        if not self.performance_history:
            return {'accuracy': 0.5, 'log_loss': 1.0}
        
        recent = list(self.performance_history)
        predictions = [p['prediction'] for p in recent]
        actuals = [p['actual'] for p in recent]
        
        accuracy = np.mean([(p > 0.5) == a for p, a in zip(predictions, actuals)])
        log_loss = np.mean([p['loss'] for p in recent])
        
        return {
            'accuracy': accuracy,
            'log_loss': log_loss,
            'samples_seen': self.samples_seen
        }


class DriftDetector:
    """Detect concept drift in data stream"""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.detectors = {
            'adwin': drift.ADWIN(delta=0.002),
            'page_hinkley': drift.PageHinkley(min_instances=30, delta=0.005, threshold=50)
        }
        self.drift_history = []
        
    def update(self, error: float) -> Dict[str, bool]:
        """Update drift detectors with prediction error"""
        drift_detected = {}
        
        for name, detector in self.detectors.items():
            detector.update(error)
            
            if detector.drift_detected:
                drift_detected[name] = True
                self.drift_history.append({
                    'detector': name,
                    'timestamp': datetime.now(),
                    'samples_seen': len(detector)
                })
                logger.warning(f"Drift detected by {name} detector")
            else:
                drift_detected[name] = False
        
        return drift_detected
    
    def reset(self):
        """Reset drift detectors after handling drift"""
        for detector in self.detectors.values():
            detector.reset()
        logger.info("Drift detectors reset")
    
    def get_drift_score(self) -> float:
        """Get overall drift score (0-1)"""
        if not self.drift_history:
            return 0.0
        
        # Recent drift events
        recent_window = datetime.now() - timedelta(days=7)
        recent_drifts = [d for d in self.drift_history 
                        if d['timestamp'] > recent_window]
        
        # Calculate drift score based on frequency and recency
        drift_score = len(recent_drifts) / max(len(self.drift_history), 1)
        return min(drift_score, 1.0)


class OnlineEnsemble:
    """Ensemble of online learning models"""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.models = self._initialize_models()
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.drift_detector = DriftDetector(config)
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
    def _initialize_models(self) -> List[OnlineModel]:
        """Initialize ensemble of online models"""
        models = []
        for model_type in self.config.model_types:
            for i in range(self.config.n_models // len(self.config.model_types) + 1):
                models.append(OnlineModel(f"{model_type}_{i}", self.config))
        return models[:self.config.n_models]
    
    def update(self, x: Dict, y: int, weight: float = 1.0):
        """Update ensemble with new sample"""
        errors = []
        
        # Update each model
        for i, model in enumerate(self.models):
            error = model.update(x, y)
            errors.append(error)
        
        # Check for drift
        avg_error = np.mean(errors)
        drift_detected = self.drift_detector.update(avg_error)
        
        # Handle drift if detected
        if any(drift_detected.values()):
            self._handle_drift()
        
        # Update weights based on performance
        if self.config.adaptive_learning:
            self._update_weights()
        
        # Add to replay buffer
        self.replay_buffer.append((x, y, weight))
        
        return avg_error
    
    def predict(self, x: Dict) -> Tuple[float, float]:
        """Make ensemble prediction with uncertainty"""
        predictions = [model.predict(x) for model in self.models]
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=self.weights)
        
        # Uncertainty as standard deviation
        uncertainty = np.std(predictions)
        
        return ensemble_pred, uncertainty
    
    def batch_update(self, X: pd.DataFrame, y: pd.Series, 
                    weights: Optional[np.ndarray] = None):
        """Batch update for efficiency"""
        if weights is None:
            weights = np.ones(len(y))
        
        for idx, row in X.iterrows():
            x_dict = row.to_dict()
            self.update(x_dict, y.loc[idx], weights[X.index.get_loc(idx)])
    
    def _update_weights(self):
        """Update model weights based on recent performance"""
        performances = []
        for model in self.models:
            perf = model.get_recent_performance()
            # Use inverse log loss as performance measure
            performances.append(1 / (perf['log_loss'] + 1e-10))
        
        # Normalize weights
        performances = np.array(performances)
        self.weights = performances / performances.sum()
        
        # Apply minimum weight threshold
        min_weight = 1 / (len(self.models) * 10)
        self.weights = np.maximum(self.weights, min_weight)
        self.weights = self.weights / self.weights.sum()
    
    def _handle_drift(self):
        """Handle detected concept drift"""
        logger.info("Handling concept drift")
        
        if self.config.concept_drift_reset:
            # Reset worst performing models
            performances = [(model.get_recent_performance()['log_loss'], i) 
                          for i, model in enumerate(self.models)]
            performances.sort()
            
            # Reset bottom 30% of models
            n_reset = max(1, len(self.models) // 3)
            for _, idx in performances[-n_reset:]:
                model_type = self.models[idx].model_type.split('_')[0]
                self.models[idx] = OnlineModel(f"{model_type}_{idx}", self.config)
                logger.info(f"Reset model {idx} due to drift")
        
        # Retrain on replay buffer
        if len(self.replay_buffer) > 0:
            logger.info(f"Retraining on {len(self.replay_buffer)} samples from replay buffer")
            for x, y, weight in list(self.replay_buffer)[-100:]:  # Last 100 samples
                for model in self.models:
                    model.update(x, y)
        
        # Reset drift detector
        self.drift_detector.reset()
    
    def get_model_performances(self) -> pd.DataFrame:
        """Get performance metrics for all models"""
        performances = []
        for i, model in enumerate(self.models):
            perf = model.get_recent_performance()
            perf['model_id'] = i
            perf['model_type'] = model.model_type
            perf['weight'] = self.weights[i]
            performances.append(perf)
        
        return pd.DataFrame(performances)


class HybridOnlineLearning:
    """Hybrid system combining batch and online learning"""
    
    def __init__(self, base_model_path: Optional[str] = None, 
                 config: Optional[OnlineLearningConfig] = None):
        self.config = config or OnlineLearningConfig()
        self.online_ensemble = OnlineEnsemble(self.config)
        self.base_model = self._load_base_model(base_model_path) if base_model_path else None
        self.feature_stats = OnlineFeatureStats()
        self.performance_tracker = PerformanceTracker(self.config)
        self.last_update = datetime.now()
        self.updates_count = 0
        
    def _load_base_model(self, path: str):
        """Load pre-trained base model"""
        # This would load your main NFLBettingEnsemble model
        logger.info(f"Loading base model from {path}")
        # Placeholder for actual model loading
        return None
    
    def predict(self, X: pd.DataFrame, 
               use_hybrid: bool = True) -> pd.DataFrame:
        """Make predictions using hybrid approach"""
        results = pd.DataFrame(index=X.index)
        
        # Base model predictions (if available)
        if self.base_model and use_hybrid:
            base_predictions = self.base_model.predict_proba(X)
            results['base_prediction'] = base_predictions
        
        # Online model predictions
        online_predictions = []
        uncertainties = []
        
        for idx, row in X.iterrows():
            x_dict = row.to_dict()
            pred, unc = self.online_ensemble.predict(x_dict)
            online_predictions.append(pred)
            uncertainties.append(unc)
        
        results['online_prediction'] = online_predictions
        results['uncertainty'] = uncertainties
        
        # Hybrid prediction (weighted combination)
        if 'base_prediction' in results.columns:
            # Weight based on recency and drift
            drift_score = self.online_ensemble.drift_detector.get_drift_score()
            online_weight = min(0.5 + drift_score * 0.3, 0.8)
            
            results['hybrid_prediction'] = (
                results['base_prediction'] * (1 - online_weight) +
                results['online_prediction'] * online_weight
            )
        else:
            results['hybrid_prediction'] = results['online_prediction']
        
        return results
    
    def update(self, X: pd.DataFrame, y: pd.Series, 
              game_results: Optional[pd.DataFrame] = None):
        """Update models with new data"""
        logger.info(f"Updating with {len(X)} new samples")
        
        # Update feature statistics
        self.feature_stats.update(X)
        
        # Check for feature drift
        feature_drift = self.feature_stats.check_drift()
        if feature_drift:
            logger.warning(f"Feature drift detected in: {feature_drift}")
        
        # Update online ensemble
        self.online_ensemble.batch_update(X, y)
        
        # Track performance
        if game_results is not None:
            self.performance_tracker.update(game_results)
        
        self.updates_count += 1
        self.last_update = datetime.now()
        
        # Periodic base model retraining
        if self._should_retrain_base_model():
            self._trigger_base_model_retrain()
    
    def _should_retrain_base_model(self) -> bool:
        """Determine if base model needs retraining"""
        # Retrain weekly or after significant drift
        days_since_update = (datetime.now() - self.last_update).days
        drift_score = self.online_ensemble.drift_detector.get_drift_score()
        
        return (days_since_update >= 7 or 
                drift_score > self.config.drift_threshold or
                self.updates_count >= 100)
    
    def _trigger_base_model_retrain(self):
        """Trigger base model retraining (async in production)"""
        logger.info("Triggering base model retraining")
        # In production, this would trigger an async training job
        # Here we just log the event
        self.updates_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status and metrics"""
        return {
            'last_update': self.last_update.isoformat(),
            'updates_count': self.updates_count,
            'drift_score': self.online_ensemble.drift_detector.get_drift_score(),
            'model_performances': self.online_ensemble.get_model_performances().to_dict(),
            'feature_drift': self.feature_stats.get_drift_summary(),
            'performance_metrics': self.performance_tracker.get_summary()
        }
    
    def save_state(self, path: str):
        """Save online learning state"""
        state = {
            'config': self.config.__dict__,
            'weights': self.online_ensemble.weights.tolist(),
            'feature_stats': self.feature_stats.get_state(),
            'performance_history': self.performance_tracker.get_history(),
            'last_update': self.last_update.isoformat(),
            'updates_count': self.updates_count
        }
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'online_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save replay buffer
        if len(self.online_ensemble.replay_buffer) > 0:
            with open(path / 'replay_buffer.pkl', 'wb') as f:
                pickle.dump(list(self.online_ensemble.replay_buffer), f)
        
        logger.info(f"Online learning state saved to {path}")
    
    def load_state(self, path: str):
        """Load online learning state"""
        path = Path(path)
        
        with open(path / 'online_state.json', 'r') as f:
            state = json.load(f)
        
        self.online_ensemble.weights = np.array(state['weights'])
        self.last_update = datetime.fromisoformat(state['last_update'])
        self.updates_count = state['updates_count']
        
        # Load replay buffer if exists
        replay_buffer_path = path / 'replay_buffer.pkl'
        if replay_buffer_path.exists():
            with open(replay_buffer_path, 'rb') as f:
                replay_data = pickle.load(f)
                self.online_ensemble.replay_buffer = deque(replay_data, 
                                                          maxlen=self.config.replay_buffer_size)
        
        logger.info(f"Online learning state loaded from {path}")


class OnlineFeatureStats:
    """Track feature statistics for drift detection"""
    
    def __init__(self):
        self.stats = {}
        self.drift_thresholds = {}
        
    def update(self, X: pd.DataFrame):
        """Update feature statistics"""
        for col in X.columns:
            if col not in self.stats:
                self.stats[col] = {
                    'mean': stats.Mean(),
                    'var': stats.Var(),
                    'min': stats.Min(),
                    'max': stats.Max(),
                    'quantiles': stats.Quantile([0.25, 0.5, 0.75])
                }
            
            for val in X[col].values:
                for stat in self.stats[col].values():
                    stat.update(val)
    
    def check_drift(self, threshold: float = 0.2) -> List[str]:
        """Check for feature drift"""
        drifted_features = []
        
        for feature, feature_stats in self.stats.items():
            # Simple drift detection based on mean/variance shift
            # In production, use more sophisticated methods
            if feature in self.drift_thresholds:
                current_mean = feature_stats['mean'].get()
                baseline_mean = self.drift_thresholds[feature]['mean']
                
                if abs(current_mean - baseline_mean) / (baseline_mean + 1e-10) > threshold:
                    drifted_features.append(feature)
        
        return drifted_features
    
    def set_baseline(self):
        """Set current statistics as baseline for drift detection"""
        for feature, feature_stats in self.stats.items():
            self.drift_thresholds[feature] = {
                'mean': feature_stats['mean'].get(),
                'var': feature_stats['var'].get()
            }
    
    def get_state(self) -> Dict:
        """Get current state for serialization"""
        state = {}
        for feature, feature_stats in self.stats.items():
            state[feature] = {
                'mean': feature_stats['mean'].get(),
                'var': feature_stats['var'].get(),
                'min': feature_stats['min'].get(),
                'max': feature_stats['max'].get()
            }
        return state
    
    def get_drift_summary(self) -> Dict[str, float]:
        """Get drift scores for all features"""
        summary = {}
        for feature in self.stats.keys():
            if feature in self.drift_thresholds:
                current_mean = self.stats[feature]['mean'].get()
                baseline_mean = self.drift_thresholds[feature]['mean']
                drift_score = abs(current_mean - baseline_mean) / (baseline_mean + 1e-10)
                summary[feature] = drift_score
        return summary


class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.metrics = {name: river_metrics.LogLoss() for name in config.metric_names}
        self.history = deque(maxlen=1000)
        self.roi_tracker = ROITracker()
        
    def update(self, game_results: pd.DataFrame):
        """Update performance metrics with game results"""
        for _, row in game_results.iterrows():
            # Update metrics
            for metric_name, metric in self.metrics.items():
                if metric_name in row:
                    metric.update(row['actual'], row['prediction'])
            
            # Track betting performance
            if 'bet_amount' in row and 'payout' in row:
                self.roi_tracker.update(row['bet_amount'], row['payout'])
            
            # Store in history
            self.history.append({
                'timestamp': datetime.now(),
                'game_id': row.get('game_id'),
                'prediction': row['prediction'],
                'actual': row['actual'],
                'metrics': {name: metric.get() for name, metric in self.metrics.items()}
            })
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        summary = {name: metric.get() for name, metric in self.metrics.items()}
        summary.update(self.roi_tracker.get_summary())
        return summary
    
    def get_history(self) -> List[Dict]:
        """Get performance history"""
        return list(self.history)


class ROITracker:
    """Track return on investment for betting"""
    
    def __init__(self):
        self.total_wagered = 0
        self.total_returned = 0
        self.bets = deque(maxlen=1000)
        
    def update(self, bet_amount: float, payout: float):
        """Update with bet result"""
        self.total_wagered += bet_amount
        self.total_returned += payout
        
        self.bets.append({
            'amount': bet_amount,
            'payout': payout,
            'profit': payout - bet_amount,
            'timestamp': datetime.now()
        })
    
    def get_summary(self) -> Dict[str, float]:
        """Get ROI summary"""
        if self.total_wagered == 0:
            return {'roi': 0, 'total_profit': 0, 'win_rate': 0}
        
        roi = (self.total_returned - self.total_wagered) / self.total_wagered
        total_profit = self.total_returned - self.total_wagered
        win_rate = sum(1 for bet in self.bets if bet['profit'] > 0) / len(self.bets) if self.bets else 0
        
        return {
            'roi': roi,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'total_wagered': self.total_wagered,
            'total_returned': self.total_returned
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create configuration
    config = OnlineLearningConfig(
        learning_rate=0.01,
        n_models=3,
        model_types=['sgd', 'tree'],
        batch_size=10
    )
    
    # Initialize hybrid system
    hybrid_system = HybridOnlineLearning(config=config)
    
    # Generate synthetic data for demonstration
    n_samples = 100
    n_features = 30
    
    # Initial batch of data
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.choice([0, 1], n_samples))
    
    # Update online models
    hybrid_system.update(X_train, y_train)
    
    # Simulate streaming data
    print("Simulating online learning with streaming data...")
    for batch in range(5):
        # New batch of data
        X_new = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Make predictions
        predictions = hybrid_system.predict(X_new)
        print(f"\nBatch {batch + 1}:")
        print(f"Average prediction: {predictions['hybrid_prediction'].mean():.3f}")
        print(f"Average uncertainty: {predictions['uncertainty'].mean():.3f}")
        
        # Simulate actual results
        y_new = pd.Series(np.random.choice([0, 1], 20))
        
        # Update with new data
        hybrid_system.update(X_new, y_new)
        
        # Check drift
        drift_score = hybrid_system.online_ensemble.drift_detector.get_drift_score()
        print(f"Drift score: {drift_score:.3f}")
    
    # Get system status
    status = hybrid_system.get_status()
    print("\nSystem Status:")
    print(f"Updates count: {status['updates_count']}")
    print(f"Drift score: {status['drift_score']:.3f}")
    
    # Get model performances
    performances = hybrid_system.online_ensemble.get_model_performances()
    print("\nModel Performances:")
    print(performances[['model_type', 'accuracy', 'log_loss', 'weight']])
    
    # Save state
    hybrid_system.save_state('online_model_state')
    print("\nOnline learning state saved!")
    
    # Load state
    new_system = HybridOnlineLearning(config=config)
    new_system.load_state('online_model_state')
    print("Online learning state loaded successfully!")
