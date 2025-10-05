"""
NFL Two-Stage Ensemble Betting Model
Production-ready implementation with LightGBM, XGBoost, calibration, and Kelly optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import optuna
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the two-stage betting model"""
    # Base model parameters
    lgb_params: Dict[str, Any] = None
    xgb_params: Dict[str, Any] = None
    
    # Ensemble parameters
    ensemble_weights: List[float] = None
    use_calibration: bool = True
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # Conservative fractional Kelly
    max_bet_fraction: float = 0.03  # Max 3% of bankroll per bet
    min_edge_threshold: float = 0.02  # Minimum 2% edge required
    
    # Training parameters
    n_splits: int = 5
    early_stopping_rounds: int = 100
    random_state: int = 42
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state,
                'force_col_wise': True
            }
        
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'random_state': self.random_state,
                'use_label_encoder': False,
                'tree_method': 'hist'
            }
        
        if self.ensemble_weights is None:
            self.ensemble_weights = [0.6, 0.4]  # LightGBM, XGBoost


class BettingLoss:
    """Custom loss function optimized for betting with expected value"""
    
    @staticmethod
    def expected_value_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                           odds: np.ndarray) -> float:
        """
        Custom loss combining log loss with expected value optimization
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities
            odds: American odds converted to decimal
        """
        # Standard log loss component
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_loss_component = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        
        # Expected value component
        # EV = P(win) * profit - P(lose) * stake
        profit = odds - 1  # Profit per unit staked
        ev = y_pred * profit - (1 - y_pred)
        
        # Only penalize negative EV predictions
        ev_penalty = np.where(ev < 0, -ev, 0)
        ev_loss = np.mean(ev_penalty)
        
        # Combine losses (weighted)
        total_loss = 0.7 * log_loss_component + 0.3 * ev_loss
        
        return total_loss
    
    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray, 
                 odds: np.ndarray) -> np.ndarray:
        """Gradient for custom loss"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Log loss gradient
        log_grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
        
        # EV gradient (simplified)
        profit = odds - 1
        ev = y_pred * profit - (1 - y_pred)
        ev_grad = np.where(ev < 0, profit + 1, 0)
        
        return 0.7 * log_grad + 0.3 * ev_grad
    
    @staticmethod
    def hessian(y_true: np.ndarray, y_pred: np.ndarray, 
                odds: np.ndarray) -> np.ndarray:
        """Hessian for custom loss"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Log loss hessian
        log_hess = 1 / (y_pred * (1 - y_pred))
        
        # Simplified constant hessian for EV component
        ev_hess = np.ones_like(y_pred) * 0.1
        
        return 0.7 * log_hess + 0.3 * ev_hess


class KellyCalculator:
    """Kelly Criterion calculator for optimal bet sizing"""
    
    def __init__(self, fraction: float = 0.25, max_bet: float = 0.03):
        self.fraction = fraction
        self.max_bet = max_bet
    
    def calculate_stake(self, probability: float, odds: float, 
                       bankroll: float = 1.0) -> float:
        """
        Calculate optimal stake using fractional Kelly Criterion
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            bankroll: Current bankroll
        
        Returns:
            Optimal stake as fraction of bankroll
        """
        # Kelly formula: f = (p(b+1) - 1) / b
        # where f = fraction, p = probability, b = odds - 1
        b = odds - 1
        
        # Full Kelly
        if probability * odds > 1:  # Positive expected value
            kelly_full = (probability * odds - 1) / b
        else:
            return 0.0
        
        # Fractional Kelly for conservative approach
        kelly_fraction = kelly_full * self.fraction
        
        # Apply maximum bet constraint
        stake = min(kelly_fraction, self.max_bet)
        
        # Apply to bankroll
        return stake * bankroll
    
    def calculate_stakes_portfolio(self, probabilities: np.ndarray, 
                                  odds: np.ndarray, 
                                  bankroll: float = 1.0,
                                  correlation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate stakes for multiple bets considering correlation
        
        Args:
            probabilities: Array of win probabilities
            odds: Array of decimal odds
            bankroll: Current bankroll
            correlation_matrix: Correlation between bets
        
        Returns:
            Array of optimal stakes
        """
        n_bets = len(probabilities)
        stakes = np.zeros(n_bets)
        
        # Calculate individual Kelly stakes
        for i in range(n_bets):
            stakes[i] = self.calculate_stake(probabilities[i], odds[i], 1.0)
        
        # Adjust for correlation if provided
        if correlation_matrix is not None:
            # Reduce stakes based on average correlation
            avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices(n_bets, k=1)]))
            adjustment_factor = 1 - (avg_correlation * 0.5)  # Reduce by up to 50%
            stakes *= adjustment_factor
        
        # Normalize if total exceeds bankroll
        total_stake = np.sum(stakes)
        if total_stake > bankroll:
            stakes = stakes * (bankroll / total_stake) * 0.95  # 95% to leave buffer
        
        return stakes * bankroll


class NFLBettingEnsemble:
    """Two-stage ensemble model for NFL betting predictions"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.lgb_models: List[lgb.Booster] = []
        self.xgb_models: List[xgb.Booster] = []
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.kelly_calculator = KellyCalculator(
            fraction=self.config.kelly_fraction,
            max_bet=self.config.max_bet_fraction
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.performance_history: List[Dict] = []
        self.is_fitted = False
    
    def _create_custom_objective_lgb(self, odds: np.ndarray):
        """Create custom objective for LightGBM"""
        def custom_obj(y_true, y_pred):
            # Convert raw predictions to probabilities
            y_pred_prob = 1 / (1 + np.exp(-y_pred))
            
            # Calculate gradients and hessians
            grad = BettingLoss.gradient(y_true, y_pred_prob, odds)
            hess = BettingLoss.hessian(y_true, y_pred_prob, odds)
            
            # Transform back to raw scale
            grad = grad * y_pred_prob * (1 - y_pred_prob)
            hess = hess * (y_pred_prob * (1 - y_pred_prob)) ** 2
            
            return grad, hess
        
        return custom_obj
    
    def _create_custom_objective_xgb(self, odds: np.ndarray):
        """Create custom objective for XGBoost"""
        def custom_obj(y_pred, dtrain):
            y_true = dtrain.get_label()
            
            # Convert raw predictions to probabilities
            y_pred_prob = 1 / (1 + np.exp(-y_pred))
            
            # Calculate gradients and hessians
            grad = BettingLoss.gradient(y_true, y_pred_prob, odds)
            hess = BettingLoss.hessian(y_true, y_pred_prob, odds)
            
            # Transform back to raw scale
            grad = grad * y_pred_prob * (1 - y_pred_prob)
            hess = hess * (y_pred_prob * (1 - y_pred_prob)) ** 2
            
            return grad, hess
        
        return custom_obj
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            odds: Optional[pd.Series] = None,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'NFLBettingEnsemble':
        """
        Train the two-stage ensemble model
        
        Args:
            X: Training features
            y: Target labels
            odds: Betting odds for custom loss
            validation_data: Optional validation set
        """
        logger.info("Training NFL Betting Ensemble Model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Use default odds if not provided
        if odds is None:
            odds = pd.Series(np.ones(len(y)) * 2.0, index=y.index)  # Even odds
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        # Clear previous models
        self.lgb_models = []
        self.xgb_models = []
        
        fold_predictions_lgb = []
        fold_predictions_xgb = []
        fold_indices = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            logger.info(f"Training fold {fold + 1}/{self.config.n_splits}")
            
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            odds_train, odds_val = odds.iloc[train_idx], odds.iloc[val_idx]
            
            # Train LightGBM
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            lgb_model = lgb.train(
                self.config.lgb_params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds),
                    lgb.log_evaluation(0)
                ],
                fobj=self._create_custom_objective_lgb(odds_train.values) if self.config.lgb_params['objective'] == 'custom' else None
            )
            self.lgb_models.append(lgb_model)
            
            # Train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            xgb_model = xgb.train(
                self.config.xgb_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=False,
                obj=self._create_custom_objective_xgb(odds_train.values) if self.config.xgb_params['objective'] == 'custom' else None
            )
            self.xgb_models.append(xgb_model)
            
            # Store predictions for calibration
            lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
            xgb_pred = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration))
            
            fold_predictions_lgb.extend(lgb_pred)
            fold_predictions_xgb.extend(xgb_pred)
            fold_indices.extend(val_idx)
        
        # Train calibrators on out-of-fold predictions
        if self.config.use_calibration:
            logger.info("Training probability calibrators")
            
            # Reorder predictions to match original index
            sorted_indices = np.argsort(fold_indices)
            fold_predictions_lgb = np.array(fold_predictions_lgb)[sorted_indices]
            fold_predictions_xgb = np.array(fold_predictions_xgb)[sorted_indices]
            fold_labels = y.iloc[np.array(fold_indices)[sorted_indices]]
            
            # Train isotonic regression calibrators
            self.calibrators['lgb'] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators['lgb'].fit(fold_predictions_lgb, fold_labels)
            
            self.calibrators['xgb'] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators['xgb'].fit(fold_predictions_xgb, fold_labels)
            
            # Train ensemble calibrator
            ensemble_pred = (
                fold_predictions_lgb * self.config.ensemble_weights[0] + 
                fold_predictions_xgb * self.config.ensemble_weights[1]
            )
            self.calibrators['ensemble'] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators['ensemble'].fit(ensemble_pred, fold_labels)
        
        self.is_fitted = True
        logger.info("Model training completed")
        
        # Evaluate if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame, 
                      return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate probability predictions with optional uncertainty estimates
        
        Args:
            X: Features for prediction
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            Predicted probabilities and optionally uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Get predictions from all models
        lgb_predictions = []
        xgb_predictions = []
        
        for lgb_model in self.lgb_models:
            pred = lgb_model.predict(X_scaled, num_iteration=lgb_model.best_iteration)
            lgb_predictions.append(pred)
        
        for xgb_model in self.xgb_models:
            dtest = xgb.DMatrix(X_scaled)
            pred = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration))
            xgb_predictions.append(pred)
        
        # Average across folds
        lgb_pred_mean = np.mean(lgb_predictions, axis=0)
        xgb_pred_mean = np.mean(xgb_predictions, axis=0)
        
        # Apply calibration
        if self.config.use_calibration:
            lgb_pred_cal = self.calibrators['lgb'].transform(lgb_pred_mean)
            xgb_pred_cal = self.calibrators['xgb'].transform(xgb_pred_mean)
        else:
            lgb_pred_cal = lgb_pred_mean
            xgb_pred_cal = xgb_pred_mean
        
        # Ensemble predictions
        ensemble_pred = (
            lgb_pred_cal * self.config.ensemble_weights[0] + 
            xgb_pred_cal * self.config.ensemble_weights[1]
        )
        
        # Apply final calibration
        if self.config.use_calibration and 'ensemble' in self.calibrators:
            ensemble_pred = self.calibrators['ensemble'].transform(ensemble_pred)
        
        if return_uncertainty:
            # Calculate uncertainty as standard deviation across models
            all_predictions = np.array(lgb_predictions + xgb_predictions)
            uncertainty = np.std(all_predictions, axis=0)
            return ensemble_pred, uncertainty
        
        return ensemble_pred
    
    def predict_with_kelly(self, X: pd.DataFrame, odds: np.ndarray, 
                          bankroll: float = 10000) -> pd.DataFrame:
        """
        Generate predictions with Kelly Criterion betting recommendations
        
        Args:
            X: Features for prediction
            odds: Decimal odds for each game
            bankroll: Current bankroll
        
        Returns:
            DataFrame with predictions and betting recommendations
        """
        # Get probabilities and uncertainty
        probabilities, uncertainty = self.predict_proba(X, return_uncertainty=True)
        
        # Calculate Kelly stakes
        stakes = np.array([
            self.kelly_calculator.calculate_stake(prob, odd, 1.0)
            for prob, odd in zip(probabilities, odds)
        ])
        
        # Calculate expected value
        ev = probabilities * (odds - 1) - (1 - probabilities)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'probability': probabilities,
            'uncertainty': uncertainty,
            'odds': odds,
            'expected_value': ev,
            'kelly_fraction': stakes,
            'recommended_bet': stakes * bankroll,
            'confidence': 1 - uncertainty,  # Simple confidence measure
            'bet_decision': (ev > self.config.min_edge_threshold) & (stakes > 0)
        }, index=X.index)
        
        return results
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                odds: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True labels
            odds: Betting odds
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict_proba(X)
        
        metrics = {
            'log_loss': log_loss(y, predictions),
            'brier_score': brier_score_loss(y, predictions),
            'auc_roc': roc_auc_score(y, predictions),
            'accuracy': np.mean((predictions > 0.5) == y)
        }
        
        # Calculate calibration metrics
        calibration_error = self._calculate_calibration_error(y, predictions)
        metrics['expected_calibration_error'] = calibration_error
        
        # Calculate betting metrics if odds provided
        if odds is not None:
            ev = predictions * (odds - 1) - (1 - predictions)
            metrics['mean_expected_value'] = np.mean(ev)
            metrics['positive_ev_ratio'] = np.mean(ev > 0)
            
            # Simulate betting returns
            stakes = np.array([
                self.kelly_calculator.calculate_stake(prob, odd, 1.0)
                for prob, odd in zip(predictions, odds.values)
            ])
            
            returns = np.where(
                y == 1,
                stakes * (odds - 1),  # Win
                -stakes  # Loss
            )
            metrics['total_return'] = np.sum(returns)
            metrics['roi'] = np.sum(returns) / (np.sum(stakes) + 1e-10)
        
        return metrics
    
    def _calculate_calibration_error(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray, 
                                    n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)"""
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
    
    def update_weights_by_performance(self, recent_performance: List[Dict[str, float]]):
        """
        Update ensemble weights based on recent performance
        
        Args:
            recent_performance: List of recent performance metrics for each model
        """
        if len(recent_performance) < 3:
            logger.warning("Insufficient performance history for weight update")
            return
        
        # Calculate exponentially weighted average of recent performance
        weights = np.array([0.5, 0.3, 0.2])  # Recent to older
        
        lgb_scores = []
        xgb_scores = []
        
        for perf in recent_performance[-3:]:
            lgb_scores.append(perf.get('lgb_auc', 0.5))
            xgb_scores.append(perf.get('xgb_auc', 0.5))
        
        lgb_weighted_score = np.average(lgb_scores, weights=weights[:len(lgb_scores)])
        xgb_weighted_score = np.average(xgb_scores, weights=weights[:len(xgb_scores)])
        
        # Update weights proportionally
        total_score = lgb_weighted_score + xgb_weighted_score
        if total_score > 0:
            self.config.ensemble_weights = [
                lgb_weighted_score / total_score,
                xgb_weighted_score / total_score
            ]
            logger.info(f"Updated ensemble weights: LGB={self.config.ensemble_weights[0]:.3f}, "
                       f"XGB={self.config.ensemble_weights[1]:.3f}")
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                odds: Optional[pd.Series] = None,
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X: Training features
            y: Target labels
            odds: Betting odds
            n_trials: Number of optimization trials
        
        Returns:
            Best parameters found
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        def objective(trial):
            # Suggest hyperparameters
            lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
                'verbose': -1,
                'random_state': self.config.random_state
            }
            
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'random_state': self.config.random_state,
                'use_label_encoder': False
            }
            
            # Create temporary model with suggested parameters
            temp_config = ModelConfig(
                lgb_params=lgb_params,
                xgb_params=xgb_params,
                n_splits=3  # Fewer splits for faster optimization
            )
            
            temp_model = NFLBettingEnsemble(config=temp_config)
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                temp_model.fit(X_train, y_train, odds.iloc[train_idx] if odds is not None else None)
                
                # Evaluate on validation set
                val_metrics = temp_model.evaluate(X_val, y_val, 
                                                 odds.iloc[val_idx] if odds is not None else None)
                
                # Use ROI as primary metric if odds available, else use AUC
                if odds is not None and 'roi' in val_metrics:
                    scores.append(val_metrics['roi'])
                else:
                    scores.append(val_metrics['auc_roc'])
            
            return np.mean(scores)
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.config.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract best parameters
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        
        # Update model configuration with best parameters
        self.config.lgb_params.update({
            'num_leaves': best_params['lgb_num_leaves'],
            'learning_rate': best_params['lgb_learning_rate'],
            'feature_fraction': best_params['lgb_feature_fraction'],
            'bagging_fraction': best_params['lgb_bagging_fraction'],
            'bagging_freq': best_params['lgb_bagging_freq'],
            'min_child_samples': best_params['lgb_min_child_samples']
        })
        
        self.config.xgb_params.update({
            'max_depth': best_params['xgb_max_depth'],
            'learning_rate': best_params['xgb_learning_rate'],
            'subsample': best_params['xgb_subsample'],
            'colsample_bytree': best_params['xgb_colsample_bytree'],
            'min_child_weight': best_params['xgb_min_child_weight']
        })
        
        return best_params
    
    def save(self, path: Union[str, Path]):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(path / 'config.json', 'w') as f:
            config_dict = {
                'lgb_params': self.config.lgb_params,
                'xgb_params': self.config.xgb_params,
                'ensemble_weights': self.config.ensemble_weights,
                'kelly_fraction': self.config.kelly_fraction,
                'max_bet_fraction': self.config.max_bet_fraction,
                'min_edge_threshold': self.config.min_edge_threshold
            }
            json.dump(config_dict, f, indent=2)
        
        # Save models
        for i, model in enumerate(self.lgb_models):
            model.save_model(str(path / f'lgb_model_{i}.txt'))
        
        for i, model in enumerate(self.xgb_models):
            model.save_model(str(path / f'xgb_model_{i}.json'))
        
        # Save calibrators and scaler
        if self.calibrators:
            joblib.dump(self.calibrators, path / 'calibrators.pkl')
        joblib.dump(self.scaler, path / 'scaler.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'n_lgb_models': len(self.lgb_models),
            'n_xgb_models': len(self.xgb_models),
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NFLBettingEnsemble':
        """Load model from disk"""
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig(
            lgb_params=config_dict['lgb_params'],
            xgb_params=config_dict['xgb_params'],
            ensemble_weights=config_dict['ensemble_weights']
        )
        config.kelly_fraction = config_dict['kelly_fraction']
        config.max_bet_fraction = config_dict['max_bet_fraction']
        config.min_edge_threshold = config_dict['min_edge_threshold']
        
        # Create model instance
        model = cls(config=config)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        model.feature_names = metadata['feature_names']
        model.is_fitted = metadata['is_fitted']
        
        # Load models
        model.lgb_models = []
        for i in range(metadata['n_lgb_models']):
            lgb_model = lgb.Booster(model_file=str(path / f'lgb_model_{i}.txt'))
            model.lgb_models.append(lgb_model)
        
        model.xgb_models = []
        for i in range(metadata['n_xgb_models']):
            xgb_model = xgb.Booster()
            xgb_model.load_model(str(path / f'xgb_model_{i}.json'))
            model.xgb_models.append(xgb_model)
        
        # Load calibrators and scaler
        if (path / 'calibrators.pkl').exists():
            model.calibrators = joblib.dump(path / 'calibrators.pkl')
        model.scaler = joblib.load(path / 'scaler.pkl')
        
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 10000
    n_features = 30
    
    # Create synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create synthetic target with some correlation to features
    y = pd.Series((X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5) > 0).astype(int)
    
    # Create synthetic odds
    odds = pd.Series(np.random.uniform(1.5, 3.0, n_samples))
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    odds_train, odds_test = odds.iloc[:split_idx], odds.iloc[split_idx:]
    
    # Initialize and train model
    model = NFLBettingEnsemble()
    
    # Optimize hyperparameters (optional, takes time)
    # best_params = model.optimize_hyperparameters(X_train, y_train, odds_train, n_trials=20)
    
    # Train model
    model.fit(X_train, y_train, odds_train, validation_data=(X_test, y_test))
    
    # Make predictions with Kelly recommendations
    predictions = model.predict_with_kelly(X_test, odds_test.values, bankroll=10000)
    
    print("\nSample Predictions:")
    print(predictions.head(10))
    
    # Evaluate performance
    metrics = model.evaluate(X_test, y_test, odds_test)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    model.save('models/nfl_ensemble')
    
    # Load model
    loaded_model = NFLBettingEnsemble.load('models/nfl_ensemble')
    
    print("\nModel successfully saved and loaded!")
