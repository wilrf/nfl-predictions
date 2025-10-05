#!/usr/bin/env python
"""
Main script for NFL Betting Model System
Provides CLI interface for training, prediction, and system management
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio
import uvicorn

# Import our modules
from nfl_ensemble_model import NFLBettingEnsemble, ModelConfig
from feature_engineering import NFLFeatureEngineering
from online_learning import HybridOnlineLearning, OnlineLearningConfig
from monitoring import ModelMonitor, MonitoringConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLBettingSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.model = None
        self.feature_engineer = None
        self.online_learner = None
        self.monitor = None
        
    def train_model(self, train_data_path: str, test_data_path: str,
                   optimize: bool = False, save_path: str = "models/nfl_ensemble"):
        """Train the ensemble model"""
        logger.info("Starting model training...")
        
        # Load data
        logger.info(f"Loading training data from {train_data_path}")
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns 
                       if col not in ['target', 'game_id', 'actual']]
        X_train = train_df[feature_cols]
        y_train = train_df['target'] if 'target' in train_df else train_df['actual']
        
        X_test = test_df[feature_cols]
        y_test = test_df['target'] if 'target' in test_df else test_df['actual']
        
        # Extract odds if available
        odds_train = train_df['odds'] if 'odds' in train_df else None
        odds_test = test_df['odds'] if 'odds' in test_df else None
        
        # Initialize model
        config = ModelConfig()
        self.model = NFLBettingEnsemble(config)
        
        # Optimize hyperparameters if requested
        if optimize:
            logger.info("Optimizing hyperparameters...")
            best_params = self.model.optimize_hyperparameters(
                X_train, y_train, odds_train, n_trials=100
            )
            logger.info(f"Best parameters: {best_params}")
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train, odds_train, 
                      validation_data=(X_test, y_test))
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = self.model.evaluate(X_test, y_test, odds_test)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:.<30} {value:.4f}")
        print("="*50)
        
        # Save model
        logger.info(f"Saving model to {save_path}")
        self.model.save(save_path)
        
        return metrics
    
    def predict(self, data_path: str, model_path: str = "models/nfl_ensemble",
               output_path: str = None):
        """Generate predictions for new games"""
        logger.info("Loading model and data...")
        
        # Load model
        self.model = NFLBettingEnsemble.load(model_path)
        
        # Load data
        data_df = pd.read_csv(data_path)
        
        # Get features
        feature_cols = [col for col in data_df.columns 
                       if col not in ['game_id', 'target', 'actual']]
        X = data_df[feature_cols]
        
        # Get odds if available
        odds = data_df['odds'].values if 'odds' in data_df else np.ones(len(X)) * 1.91
        
        # Generate predictions
        logger.info("Generating predictions...")
        results = self.model.predict_with_kelly(X, odds, bankroll=10000)
        
        # Add game IDs if available
        if 'game_id' in data_df:
            results.index = data_df['game_id']
        
        print("\n" + "="*50)
        print("PREDICTIONS")
        print("="*50)
        print(results[['probability', 'expected_value', 'kelly_fraction', 
                      'recommended_bet', 'bet_decision']])
        print("="*50)
        
        # Save results
        if output_path:
            results.to_csv(output_path)
            logger.info(f"Predictions saved to {output_path}")
        
        return results
    
    def start_online_learning(self, model_path: str = "models/nfl_ensemble"):
        """Start online learning system"""
        logger.info("Initializing online learning system...")
        
        config = OnlineLearningConfig()
        self.online_learner = HybridOnlineLearning(
            base_model_path=model_path,
            config=config
        )
        
        logger.info("Online learning system started")
        logger.info("System will automatically update with new data")
        
        return self.online_learner
    
    def start_monitoring(self, baseline_data_path: str):
        """Start monitoring system"""
        logger.info("Initializing monitoring system...")
        
        # Load baseline data
        baseline_df = pd.read_csv(baseline_data_path)
        
        # Initialize monitor
        config = MonitoringConfig()
        self.monitor = ModelMonitor(config)
        
        # Set baseline
        feature_cols = [col for col in baseline_df.columns 
                       if col not in ['game_id', 'target', 'actual']]
        self.monitor.initialize(baseline_df[feature_cols])
        
        logger.info("Monitoring system started")
        logger.info("Drift detection and performance tracking enabled")
        
        return self.monitor
    
    def run_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        logger.info(f"Starting API server on {host}:{port}")
        
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=False,
            workers=1,
            log_level="info"
        )
    
    def feature_engineering(self, raw_data_path: str, output_path: str):
        """Run feature engineering pipeline"""
        logger.info("Starting feature engineering...")
        
        # Initialize feature engineer
        self.feature_engineer = NFLFeatureEngineering()
        
        # Load raw data (simplified for demo)
        raw_df = pd.read_csv(raw_data_path)
        
        # Transform features
        features = self.feature_engineer.transform(
            game_data=raw_df,
            team_stats=raw_df,  # In production, these would be separate
            betting_data=raw_df
        )
        
        # Save features
        features.to_csv(output_path)
        logger.info(f"Features saved to {output_path}")
        
        print(f"\nGenerated {len(features.columns)} features")
        print(f"Feature groups: {self.feature_engineer.get_feature_importance_groups().keys()}")
        
        return features
    
    def generate_report(self):
        """Generate comprehensive system report"""
        logger.info("Generating system report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": {},
            "online_learning": {},
            "monitoring": {}
        }
        
        # Model info
        if self.model:
            report["model"] = {
                "is_fitted": self.model.is_fitted,
                "n_features": len(self.model.feature_names),
                "ensemble_weights": self.model.config.ensemble_weights
            }
        
        # Online learning status
        if self.online_learner:
            report["online_learning"] = self.online_learner.get_status()
        
        # Monitoring status
        if self.monitor:
            report["monitoring"] = self.monitor.get_status()
        
        # Save report
        report_path = f"reports/system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        return report


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="NFL Betting Model System")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--train-data', required=True, help='Path to training data')
    train_parser.add_argument('--test-data', required=True, help='Path to test data')
    train_parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    train_parser.add_argument('--save-path', default='models/nfl_ensemble', help='Model save path')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--data', required=True, help='Path to prediction data')
    predict_parser.add_argument('--model', default='models/nfl_ensemble', help='Model path')
    predict_parser.add_argument('--output', help='Output path for predictions')
    
    # Feature engineering command
    feature_parser = subparsers.add_parser('features', help='Run feature engineering')
    feature_parser.add_argument('--input', required=True, help='Path to raw data')
    feature_parser.add_argument('--output', required=True, help='Output path for features')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host address')
    api_parser.add_argument('--port', default=8000, type=int, help='Port number')
    
    # Monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring')
    monitor_parser.add_argument('--baseline', required=True, help='Path to baseline data')
    
    # Online learning command
    online_parser = subparsers.add_parser('online', help='Start online learning')
    online_parser.add_argument('--model', default='models/nfl_ensemble', help='Base model path')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate system report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize system
    system = NFLBettingSystem()
    
    try:
        if args.command == 'train':
            system.train_model(
                args.train_data,
                args.test_data,
                args.optimize,
                args.save_path
            )
        
        elif args.command == 'predict':
            system.predict(args.data, args.model, args.output)
        
        elif args.command == 'features':
            system.feature_engineering(args.input, args.output)
        
        elif args.command == 'api':
            system.run_api_server(args.host, args.port)
        
        elif args.command == 'monitor':
            system.start_monitoring(args.baseline)
            print("Monitoring system running. Press Ctrl+C to stop.")
            try:
                while True:
                    asyncio.sleep(60)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        
        elif args.command == 'online':
            system.start_online_learning(args.model)
            print("Online learning system running. Press Ctrl+C to stop.")
            try:
                while True:
                    asyncio.sleep(60)
            except KeyboardInterrupt:
                print("\nOnline learning stopped.")
        
        elif args.command == 'report':
            report = system.generate_report()
            print("\nSystem Report Generated:")
            print(json.dumps(report, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
