#!/usr/bin/env python3
"""
Ensemble Predictor for Production Use
Combines Random Forest and XGBoost predictions with optimized weights
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Production-ready ensemble predictor combining Random Forest and XGBoost"""

    def __init__(self, models_dir: str = 'models/saved_models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.configs = {}
        self._load_models()
        self._load_configs()

    def _load_models(self):
        """Load all required models"""
        try:
            # XGBoost models
            with open(self.models_dir / 'spread_model.pkl', 'rb') as f:
                self.models['xgboost_spread'] = pickle.load(f)
            with open(self.models_dir / 'spread_calibrator.pkl', 'rb') as f:
                self.models['xgboost_spread_cal'] = pickle.load(f)
            
            with open(self.models_dir / 'total_model.pkl', 'rb') as f:
                self.models['xgboost_total'] = pickle.load(f)
            with open(self.models_dir / 'total_calibrator.pkl', 'rb') as f:
                self.models['xgboost_total_cal'] = pickle.load(f)

            # Random Forest models
            with open(self.models_dir / 'random_forest_spread_model.pkl', 'rb') as f:
                self.models['random_forest_spread'] = pickle.load(f)
            with open(self.models_dir / 'random_forest_spread_calibrator.pkl', 'rb') as f:
                self.models['random_forest_spread_cal'] = pickle.load(f)
            
            with open(self.models_dir / 'random_forest_total_model.pkl', 'rb') as f:
                self.models['random_forest_total'] = pickle.load(f)

            logger.info("✓ All models loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise

    def _load_configs(self):
        """Load ensemble configurations"""
        try:
            with open(self.models_dir / 'ensemble_spread_config.json', 'r') as f:
                self.configs['spread'] = json.load(f)
            
            with open(self.models_dir / 'ensemble_total_config.json', 'r') as f:
                self.configs['total'] = json.load(f)
            
            logger.info("✓ Ensemble configurations loaded")
            
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            raise

    def predict_spread(self, game_data: pd.DataFrame) -> Dict:
        """Predict spread outcome using ensemble model"""
        if 'spread' not in self.configs:
            raise ValueError("Spread ensemble configuration not loaded")

        config = self.configs['spread']
        weights = config['weights']
        feature_sets = config['feature_sets']

        # Prepare features for each model
        X_xgb = game_data[feature_sets['xgboost']]
        X_rf = game_data[feature_sets['random_forest']]

        # Get predictions from both models
        xgb_probs = self.models['xgboost_spread_cal'].transform(
            self.models['xgboost_spread'].predict_proba(X_xgb)[:, 1]
        )
        rf_probs = self.models['random_forest_spread_cal'].predict_proba(X_rf)[:, 1]

        # Combine with optimized weights
        ensemble_probs = weights['xgboost'] * xgb_probs + weights['random_forest'] * rf_probs
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        # Calculate confidence (distance from 0.5)
        confidence = np.abs(ensemble_probs - 0.5) * 2  # Scale to 0-1

        return {
            'prediction': int(ensemble_preds[0]),
            'probability': float(ensemble_probs[0]),
            'confidence': float(confidence[0]),
            'home_win_prob': float(ensemble_probs[0]),
            'away_win_prob': float(1 - ensemble_probs[0]),
            'model_breakdown': {
                'xgboost_prob': float(xgb_probs[0]),
                'random_forest_prob': float(rf_probs[0]),
                'xgboost_weight': weights['xgboost'],
                'random_forest_weight': weights['random_forest']
            }
        }

    def predict_total(self, game_data: pd.DataFrame) -> Dict:
        """Predict total points using ensemble model"""
        if 'total' not in self.configs:
            raise ValueError("Total ensemble configuration not loaded")

        config = self.configs['total']
        weights = config['weights']
        feature_sets = config['feature_sets']
        median_total = config['median_total']

        # Prepare features for each model
        X_xgb = game_data[feature_sets['xgboost']]
        X_rf = game_data[feature_sets['random_forest']]

        # Get XGBoost predictions (binary classification)
        xgb_probs = self.models['xgboost_total_cal'].transform(
            self.models['xgboost_total'].predict_proba(X_xgb)[:, 1]
        )

        # Get Random Forest predictions (regression)
        rf_preds = self.models['random_forest_total'].predict(X_rf)

        # Convert RF regression to probabilities
        # Use historical standard deviation for conversion
        rf_std = 13.25  # From training metrics
        from scipy.stats import norm
        rf_probs = 1 - norm.cdf(median_total, loc=rf_preds, scale=rf_std)

        # Combine with optimized weights
        ensemble_probs = weights['xgboost'] * xgb_probs + weights['random_forest'] * rf_probs
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        # Calculate confidence
        confidence = np.abs(ensemble_probs - 0.5) * 2

        return {
            'prediction': int(ensemble_preds[0]),  # 1 = over, 0 = under
            'probability': float(ensemble_probs[0]),
            'confidence': float(confidence[0]),
            'over_prob': float(ensemble_probs[0]),
            'under_prob': float(1 - ensemble_probs[0]),
            'predicted_total': float(rf_preds[0]),
            'median_threshold': median_total,
            'model_breakdown': {
                'xgboost_prob': float(xgb_probs[0]),
                'random_forest_prob': float(rf_probs[0]),
                'random_forest_prediction': float(rf_preds[0]),
                'xgboost_weight': weights['xgboost'],
                'random_forest_weight': weights['random_forest']
            }
        }

    def predict_game(self, game_data: pd.DataFrame) -> Dict:
        """Predict both spread and total for a game"""
        spread_result = self.predict_spread(game_data)
        total_result = self.predict_total(game_data)

        # Calculate overall confidence
        overall_confidence = (spread_result['confidence'] + total_result['confidence']) / 2

        return {
            'game_id': game_data.get('game_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'spread': spread_result,
            'total': total_result,
            'overall_confidence': float(overall_confidence),
            'recommendation': self._generate_recommendation(spread_result, total_result)
        }

    def _generate_recommendation(self, spread_result: Dict, total_result: Dict) -> Dict:
        """Generate betting recommendation based on confidence levels"""
        spread_conf = spread_result['confidence']
        total_conf = total_result['confidence']
        
        recommendations = []
        
        # Spread recommendation
        if spread_conf > 0.7:  # High confidence
            if spread_result['probability'] > 0.6:
                recommendations.append({
                    'type': 'spread',
                    'side': 'home' if spread_result['prediction'] == 1 else 'away',
                    'confidence': spread_conf,
                    'probability': spread_result['probability'],
                    'strength': 'strong'
                })
            elif spread_result['probability'] < 0.4:
                recommendations.append({
                    'type': 'spread',
                    'side': 'away' if spread_result['prediction'] == 1 else 'home',
                    'confidence': spread_conf,
                    'probability': 1 - spread_result['probability'],
                    'strength': 'strong'
                })
        
        # Total recommendation
        if total_conf > 0.7:  # High confidence
            if total_result['probability'] > 0.6:
                recommendations.append({
                    'type': 'total',
                    'side': 'over',
                    'confidence': total_conf,
                    'probability': total_result['probability'],
                    'strength': 'strong'
                })
            elif total_result['probability'] < 0.4:
                recommendations.append({
                    'type': 'total',
                    'side': 'under',
                    'confidence': total_conf,
                    'probability': 1 - total_result['probability'],
                    'strength': 'strong'
                })
        
        return {
            'recommendations': recommendations,
            'count': len(recommendations),
            'max_confidence': max(spread_conf, total_conf) if recommendations else 0
        }

    def predict_batch(self, games_data: pd.DataFrame) -> List[Dict]:
        """Predict multiple games at once"""
        results = []
        
        for idx, game in games_data.iterrows():
            try:
                result = self.predict_game(game.to_frame().T)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting game {idx}: {e}")
                continue
        
        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'models_loaded': list(self.models.keys()),
            'configurations': {
                'spread': {
                    'weights': self.configs['spread']['weights'],
                    'validation_accuracy': self.configs['spread']['validation_metrics']['accuracy'],
                    'test_accuracy': self.configs['spread']['test_metrics']['accuracy']
                },
                'total': {
                    'weights': self.configs['total']['weights'],
                    'validation_accuracy': self.configs['total']['validation_metrics']['accuracy'],
                    'test_accuracy': self.configs['total']['test_metrics']['accuracy']
                }
            },
            'feature_counts': {
                'xgboost': len(self.configs['spread']['feature_sets']['xgboost']),
                'random_forest': len(self.configs['spread']['feature_sets']['random_forest'])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = EnsemblePredictor()
    
    # Get model info
    info = predictor.get_model_info()
    print("Model Information:")
    print(json.dumps(info, indent=2))
    
    # Example prediction (would need actual game data)
    print("\nEnsemble predictor ready for production use!")
    print("Use predictor.predict_game(game_data) for single game predictions")
    print("Use predictor.predict_batch(games_data) for multiple games")
