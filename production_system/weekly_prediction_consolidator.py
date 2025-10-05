#!/usr/bin/env python3
"""
Weekly Prediction Consolidator
Consolidates predictions from multiple models into weekly predictions
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeeklyPredictionConsolidator:
    """Consolidate predictions from multiple models into weekly predictions"""
    
    def __init__(self):
        self.models_dir = Path('model_architecture/output')
        self.output_dir = Path('production_system/weekly_predictions')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load individual models
        self.individual_models = {}
        self.load_individual_models()
        
        logger.info("WeeklyPredictionConsolidator initialized")
    
    def load_individual_models(self):
        """Load individual models for comparison"""
        try:
            for model_name in ['xgboost', 'lightgbm', 'random_forest', 'neural_network']:
                model_file = self.models_dir / f'{model_name}_model.pkl'
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.individual_models[model_name] = pickle.load(f)
            
            logger.info(f"Loaded {len(self.individual_models)} individual models")
            
        except Exception as e:
            logger.error(f"Failed to load individual models: {e}")
    
    def generate_weekly_predictions(self, week_number: int, season: int = 2025):
        """Generate consolidated weekly predictions"""
        logger.info(f"Generating weekly predictions for Week {week_number}, Season {season}")
        
        # Sample NFL games for the week (simplified)
        weekly_games = self._get_weekly_games(week_number, season)
        
        predictions = []
        
        for game in weekly_games:
            # Get predictions from each model
            model_predictions = self._get_model_predictions(game)
            
            # Consolidate predictions
            consolidated = self._consolidate_predictions(game, model_predictions)
            predictions.append(consolidated)
        
        # Save weekly predictions
        self._save_weekly_predictions(week_number, season, predictions)
        
        return predictions
    
    def _get_weekly_games(self, week_number: int, season: int) -> List[Dict]:
        """Get sample NFL games for the week"""
        # Simplified game data - in production, this would come from NFL API
        sample_games = [
            {
                "home_team": "KC", "away_team": "BUF",
                "spread_line": -3.5, "total_line": 52.5,
                "temperature": 45, "wind_speed": 8, "humidity": 60
            },
            {
                "home_team": "DAL", "away_team": "PHI",
                "spread_line": -2.0, "total_line": 48.0,
                "temperature": 65, "wind_speed": 5, "humidity": 45
            },
            {
                "home_team": "SF", "away_team": "LAR",
                "spread_line": -4.0, "total_line": 45.5,
                "temperature": 72, "wind_speed": 3, "humidity": 55
            },
            {
                "home_team": "GB", "away_team": "MIN",
                "spread_line": -1.5, "total_line": 47.0,
                "temperature": 38, "wind_speed": 12, "humidity": 70
            },
            {
                "home_team": "MIA", "away_team": "NE",
                "spread_line": -6.0, "total_line": 43.0,
                "temperature": 78, "wind_speed": 7, "humidity": 80
            }
        ]
        
        return sample_games
    
    def _get_model_predictions(self, game: Dict) -> Dict[str, Dict]:
        """Get predictions from each individual model"""
        model_predictions = {}
        
        # Create feature vector (simplified)
        features = self._create_feature_vector(game)
        
        for model_name, model in self.individual_models.items():
            try:
                # Get prediction
                probabilities = model.predict_proba(features)
                home_prob = float(probabilities[0][1])
                away_prob = float(probabilities[0][0])
                
                # Determine prediction
                if home_prob > away_prob:
                    prediction = game['home_team']
                    confidence = abs(home_prob - 0.5) * 2
                else:
                    prediction = game['away_team']
                    confidence = abs(away_prob - 0.5) * 2
                
                model_predictions[model_name] = {
                    'home_probability': home_prob,
                    'away_probability': away_prob,
                    'prediction': prediction,
                    'confidence': confidence
                }
                
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
                model_predictions[model_name] = {
                    'home_probability': 0.5,
                    'away_probability': 0.5,
                    'prediction': 'TIE',
                    'confidence': 0.0
                }
        
        return model_predictions
    
    def _create_feature_vector(self, game: Dict) -> np.ndarray:
        """Create feature vector from game data"""
        # Initialize feature vector with zeros (46 features)
        features = np.zeros(46)
        
        # Map game data to features (simplified mapping)
        features[0] = game.get('spread_line', 0)
        features[1] = game.get('total_line', 0)
        features[2] = 1  # is_home
        features[3] = 5  # week_number (sample)
        features[4] = 0  # is_divisional
        features[5] = 0  # is_playoff
        features[6] = 1  # is_outdoor
        features[7] = 7  # rest_days_home
        features[8] = 7  # rest_days_away
        features[9] = 0  # rest_advantage
        features[10] = 5/18.0  # season_progress
        features[16] = game.get('temperature', 70)
        features[17] = game.get('wind_speed', 0)
        features[18] = game.get('humidity', 50)
        features[19] = 0  # precipitation
        
        return features.reshape(1, -1)
    
    def _consolidate_predictions(self, game: Dict, model_predictions: Dict) -> Dict:
        """Consolidate predictions from multiple models"""
        # Calculate weighted average probabilities
        home_probs = [pred['home_probability'] for pred in model_predictions.values()]
        away_probs = [pred['away_probability'] for pred in model_predictions.values()]
        confidences = [pred['confidence'] for pred in model_predictions.values()]
        
        # Simple average (could be weighted by model performance)
        avg_home_prob = np.mean(home_probs)
        avg_away_prob = np.mean(away_probs)
        avg_confidence = np.mean(confidences)
        
        # Determine consolidated prediction
        if avg_home_prob > avg_away_prob:
            consolidated_prediction = game['home_team']
        else:
            consolidated_prediction = game['away_team']
        
        # Calculate model agreement
        predictions = [pred['prediction'] for pred in model_predictions.values()]
        model_agreement = len(set(predictions)) == 1  # True if all models agree
        
        return {
            'game': game,
            'consolidated': {
                'home_probability': avg_home_prob,
                'away_probability': avg_away_prob,
                'prediction': consolidated_prediction,
                'confidence': avg_confidence,
                'model_agreement': model_agreement
            },
            'individual_models': model_predictions,
            'model_count': len(model_predictions)
        }
    
    def _save_weekly_predictions(self, week_number: int, season: int, predictions: List[Dict]):
        """Save weekly predictions to file"""
        timestamp = datetime.now().isoformat()
        
        weekly_data = {
            'week_number': week_number,
            'season': season,
            'timestamp': timestamp,
            'total_games': len(predictions),
            'predictions': predictions,
            'summary': self._generate_weekly_summary(predictions)
        }
        
        filename = f'week_{week_number}_season_{season}_predictions.json'
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(weekly_data, f, indent=2)
        
        logger.info(f"Weekly predictions saved to {filepath}")
    
    def _generate_weekly_summary(self, predictions: List[Dict]) -> Dict:
        """Generate summary statistics for the week"""
        total_games = len(predictions)
        model_agreements = sum(1 for pred in predictions if pred['consolidated']['model_agreement'])
        avg_confidence = np.mean([pred['consolidated']['confidence'] for pred in predictions])
        
        # Count predictions by team
        home_wins = sum(1 for pred in predictions if pred['consolidated']['prediction'] == pred['game']['home_team'])
        away_wins = total_games - home_wins
        
        return {
            'total_games': total_games,
            'model_agreement_rate': model_agreements / total_games if total_games > 0 else 0,
            'average_confidence': avg_confidence,
            'home_team_wins': home_wins,
            'away_team_wins': away_wins,
            'home_win_rate': home_wins / total_games if total_games > 0 else 0
        }
    
    def compare_model_performance(self, week_number: int, season: int = 2025):
        """Compare performance of individual models"""
        logger.info(f"Comparing model performance for Week {week_number}")
        
        # Load weekly predictions
        filename = f'week_{week_number}_season_{season}_predictions.json'
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            logger.error(f"Weekly predictions file not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            weekly_data = json.load(f)
        
        predictions = weekly_data['predictions']
        
        # Analyze each model's performance
        model_analysis = {}
        
        for model_name in self.individual_models.keys():
            model_predictions = []
            for pred in predictions:
                if model_name in pred['individual_models']:
                    model_predictions.append(pred['individual_models'][model_name])
            
            if model_predictions:
                avg_confidence = np.mean([pred['confidence'] for pred in model_predictions])
                predictions_made = len(model_predictions)
                
                model_analysis[model_name] = {
                    'predictions_made': predictions_made,
                    'average_confidence': avg_confidence,
                    'total_games': len(predictions)
                }
        
        return model_analysis

if __name__ == "__main__":
    consolidator = WeeklyPredictionConsolidator()
    
    # Generate predictions for Week 5
    predictions = consolidator.generate_weekly_predictions(week_number=5, season=2025)
    
    # Compare model performance
    model_comparison = consolidator.compare_model_performance(week_number=5, season=2025)
    
    logger.info("Weekly prediction consolidation complete")
    logger.info(f"Generated {len(predictions)} game predictions")
    
    if model_comparison:
        logger.info("Model Performance Comparison:")
        for model_name, stats in model_comparison.items():
            logger.info(f"  {model_name}: {stats['predictions_made']} predictions, avg confidence: {stats['average_confidence']:.3f}")
