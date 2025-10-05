"""
Feature Adapters for Validation Framework
Bridges existing NFL betting system with the validation framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureAdapter:
    """Base adapter class for converting system features to validation format"""

    def __init__(self, db_path: str = None):
        """Initialize adapter with database connection"""
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'database' / 'nfl_suggestions.db'

        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def prepare_baseline_features(self, season: int, week: int = None) -> pd.DataFrame:
        """
        Prepare baseline features (traditional stats) for validation

        Args:
            season: NFL season
            week: Optional specific week, otherwise full season

        Returns:
            DataFrame with baseline features
        """
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.stadium,
            g.is_outdoor
        FROM games g
        WHERE g.season = ?
        """
        params = [season]

        if week is not None:
            query += " AND g.week = ?"
            params.append(week)

        try:
            baseline_df = pd.read_sql_query(query, self.conn, params=params)

            # Add traditional stats if available
            # These would come from team stats table
            baseline_df['home_total_yards'] = np.random.normal(380, 60, len(baseline_df))
            baseline_df['away_total_yards'] = np.random.normal(360, 55, len(baseline_df))
            baseline_df['home_turnovers'] = np.random.poisson(1.5, len(baseline_df))
            baseline_df['away_turnovers'] = np.random.poisson(1.5, len(baseline_df))

            return baseline_df

        except Exception as e:
            logger.error(f"Error preparing baseline features: {e}")
            return pd.DataFrame()

    def prepare_epa_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare EPA-based features for validation

        Args:
            games_df: DataFrame with game information

        Returns:
            DataFrame with EPA features
        """
        epa_features = []

        for _, game in games_df.iterrows():
            # Extract EPA features from existing system
            # In production, these would come from actual calculations
            epa_data = {
                'game_id': game['game_id'],
                'home_off_epa': np.random.normal(0.05, 0.15),
                'home_def_epa': np.random.normal(-0.05, 0.15),
                'away_off_epa': np.random.normal(0.02, 0.15),
                'away_def_epa': np.random.normal(-0.02, 0.15),
                'home_epa_differential': np.random.normal(0, 0.3),
                'away_epa_differential': np.random.normal(0, 0.3),
                'epa_trend_home': np.random.normal(0, 0.1),
                'epa_trend_away': np.random.normal(0, 0.1)
            }
            epa_features.append(epa_data)

        return pd.DataFrame(epa_features)

    def prepare_injury_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare injury-based features for validation

        Args:
            games_df: DataFrame with game information

        Returns:
            DataFrame with injury features
        """
        injury_features = []

        for _, game in games_df.iterrows():
            # Extract injury features
            # In production, would fetch from injury reports
            injury_data = {
                'game_id': game['game_id'],
                'home_qb_injured': np.random.choice([0, 1], p=[0.9, 0.1]),
                'home_rb1_injured': np.random.choice([0, 1], p=[0.85, 0.15]),
                'home_wr1_injured': np.random.choice([0, 1], p=[0.85, 0.15]),
                'home_key_players_out': np.random.poisson(1.2),
                'away_qb_injured': np.random.choice([0, 1], p=[0.9, 0.1]),
                'away_rb1_injured': np.random.choice([0, 1], p=[0.85, 0.15]),
                'away_wr1_injured': np.random.choice([0, 1], p=[0.85, 0.15]),
                'away_key_players_out': np.random.poisson(1.2),
                'injury_impact_differential': np.random.normal(0, 2)
            }
            injury_features.append(injury_data)

        return pd.DataFrame(injury_features)

    def prepare_weather_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare weather-based features for validation

        Args:
            games_df: DataFrame with game information

        Returns:
            DataFrame with weather features
        """
        weather_features = []

        for _, game in games_df.iterrows():
            # Only relevant for outdoor games
            if game.get('is_outdoor', False):
                weather_data = {
                    'game_id': game['game_id'],
                    'temperature': np.random.normal(65, 20),
                    'wind_speed': np.random.exponential(5),
                    'precipitation': np.random.exponential(0.1),
                    'weather_impact_score': np.random.normal(0, 1)
                }
            else:
                # Dome/indoor conditions
                weather_data = {
                    'game_id': game['game_id'],
                    'temperature': 72,
                    'wind_speed': 0,
                    'precipitation': 0,
                    'weather_impact_score': 0
                }

            weather_features.append(weather_data)

        return pd.DataFrame(weather_features)

    def prepare_market_data(self, games_df: pd.DataFrame) -> Dict:
        """
        Prepare market data (odds, lines, outcomes) for validation

        Args:
            games_df: DataFrame with game information

        Returns:
            Dictionary with market data
        """
        market_lines = []
        outcomes = []
        predictions = []

        for _, game in games_df.iterrows():
            # Get odds from database if available
            odds_query = """
            SELECT spread_home, total_line
            FROM odds
            WHERE game_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """

            odds = self.conn.execute(odds_query, (game['game_id'],)).fetchone()

            if odds:
                market_lines.append(odds['spread_home'])
                # Simulate outcome (would use actual results in production)
                outcomes.append(np.random.choice([0, 1]))
                # Simulate prediction
                predictions.append(odds['spread_home'] + np.random.normal(0, 2))
            else:
                # Use default values if no odds available
                market_lines.append(np.random.normal(0, 3))
                outcomes.append(np.random.choice([0, 1]))
                predictions.append(np.random.normal(0, 3))

        return {
            'market_lines': np.array(market_lines),
            'outcomes': np.array(outcomes),
            'predictions': np.array(predictions)
        }

    def prepare_supplementary_data(self, games_df: pd.DataFrame) -> Dict:
        """
        Prepare supplementary data for interaction testing

        Args:
            games_df: DataFrame with game information

        Returns:
            Dictionary with supplementary data
        """
        weather_df = self.prepare_weather_features(games_df)
        injury_df = self.prepare_injury_features(games_df)

        # Referee data (would fetch from actual source)
        referee_df = pd.DataFrame({
            'game_id': games_df['game_id'],
            'avg_penalties_per_game': np.random.normal(12, 3, len(games_df)),
            'home_bias_score': np.random.normal(0, 0.5, len(games_df))
        })

        return {
            'weather': weather_df,
            'injury': injury_df,
            'referee': referee_df,
            'outcomes': pd.DataFrame({
                'game_id': games_df['game_id'],
                'total_penalties': np.random.poisson(12, len(games_df)),
                'total_score_variance': np.random.exponential(15, len(games_df))
            })
        }


class SystemIntegrationAdapter:
    """Adapter for integrating validation results back into the main system"""

    def __init__(self, validation_results: Dict):
        """
        Initialize with validation results

        Args:
            validation_results: Results from validation framework
        """
        self.results = validation_results
        self.validated_features = []
        self.feature_weights = {}

    def extract_validated_features(self) -> List[str]:
        """
        Extract list of features that passed validation

        Returns:
            List of validated feature names
        """
        validated = []

        # Check Phase 1 results
        if self.results.get('phase_1', {}).get('phase_1_success'):
            phase1 = self.results['phase_1']
            if phase1.get('importance_testing', {}).get('statistically_significant'):
                # Features passed statistical validation
                validated.extend(phase1.get('significant_features', []))

        # Check Phase 2 results
        if self.results.get('phase_2', {}).get('phase_2_success'):
            phase2 = self.results['phase_2']
            if phase2.get('market_efficiency', {}).get('exploitable'):
                # Features provide market edge
                validated.extend(phase2.get('exploitable_features', []))

        # Check Phase 3 results
        if self.results.get('phase_3', {}).get('phase_3_success'):
            phase3 = self.results['phase_3']
            # Only include temporally stable features
            validated.extend(phase3.get('reliable_features', []))

        # Return unique features that passed all phases
        self.validated_features = list(set(validated))
        return self.validated_features

    def get_feature_weights(self) -> Dict[str, float]:
        """
        Get feature weights based on validation results

        Returns:
            Dictionary of feature weights
        """
        weights = {}

        # Base weights from statistical importance
        if 'phase_1' in self.results:
            importance = self.results['phase_1'].get('importance_testing', {})
            base_weight = importance.get('effect_size', 1.0)

            for feature in self.validated_features:
                weights[feature] = base_weight

        # Adjust weights based on market efficiency
        if 'phase_2' in self.results:
            market = self.results['phase_2'].get('market_efficiency', {})
            roi_multiplier = 1 + market.get('actual_roi', 0)

            for feature in weights:
                weights[feature] *= roi_multiplier

        # Adjust weights based on temporal stability
        if 'phase_3' in self.results:
            rankings = self.results['phase_3'].get('feature_rankings', [])
            for rank_data in rankings:
                feature = rank_data['feature']
                if feature in weights:
                    weights[feature] *= rank_data['reliability_score']

        self.feature_weights = weights
        return weights

    def generate_model_config(self) -> Dict:
        """
        Generate model configuration based on validation

        Returns:
            Configuration dictionary for model training
        """
        config = {
            'features': self.validated_features,
            'feature_weights': self.get_feature_weights(),
            'hyperparameters': self._recommend_hyperparameters(),
            'validation_strategy': self._recommend_validation_strategy(),
            'deployment_config': self._get_deployment_config()
        }

        return config

    def _recommend_hyperparameters(self) -> Dict:
        """Recommend model hyperparameters based on validation"""
        # Base hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8
        }

        # Adjust based on feature count
        if len(self.validated_features) > 10:
            params['max_depth'] = 7
            params['n_estimators'] = 150

        # Adjust based on expected ROI
        if self.results.get('phase_2', {}).get('market_efficiency', {}).get('actual_roi', 0) > 0.03:
            params['learning_rate'] = 0.05  # More conservative for high-value features

        return params

    def _recommend_validation_strategy(self) -> Dict:
        """Recommend validation strategy for model training"""
        return {
            'method': 'TimeSeriesSplit',
            'n_splits': 5,
            'test_size': 0.2,
            'gap': 1  # 1 week gap between train and test
        }

    def _get_deployment_config(self) -> Dict:
        """Get deployment configuration"""
        phase4 = self.results.get('phase_4', {})
        roadmap = phase4.get('implementation_roadmap', {})

        return {
            'immediate_features': roadmap.get('phase_1_immediate', {}).get('features', []),
            'monitoring_required': roadmap.get('phase_1_immediate', {}).get('monitoring_level', 'standard'),
            'expected_roi': roadmap.get('phase_1_immediate', {}).get('total_expected_roi', 0),
            'risk_assessment': roadmap.get('risk_assessment', {}),
            'rollback_threshold': -0.02  # Rollback if ROI drops below -2%
        }


class ValidationRunner:
    """Convenience class to run validation with adapters"""

    def __init__(self, db_path: str = None):
        """Initialize validation runner"""
        self.adapter = FeatureAdapter(db_path)
        self.validation_results = None

    def run_epa_validation(self, season: int) -> Dict:
        """
        Run complete validation pipeline for EPA features

        Args:
            season: NFL season to validate

        Returns:
            Validation results
        """
        from .data_validation_framework import DataValidationFramework

        # Prepare data
        baseline = self.adapter.prepare_baseline_features(season)
        epa_features = self.adapter.prepare_epa_features(baseline)
        market_data = self.adapter.prepare_market_data(baseline)
        supplementary = self.adapter.prepare_supplementary_data(baseline)

        # Create target (would use actual outcomes in production)
        target = np.random.normal(0, 14, len(baseline))

        # Initialize validation framework
        framework = DataValidationFramework({
            'min_seasons_required': 1,  # For testing
            'min_sample_size': 50,
            'significance_level': 0.05,
            'roi_threshold': 0.02
        })

        # Run validation
        results = framework.run_complete_validation_pipeline(
            data_source='epa_metrics',
            baseline_features=baseline,
            new_features=epa_features,
            target=target,
            market_data=market_data,
            supplementary_data=supplementary
        )

        self.validation_results = results
        return results

    def get_integration_config(self) -> Dict:
        """Get configuration for system integration"""
        if not self.validation_results:
            raise ValueError("No validation results available. Run validation first.")

        integration = SystemIntegrationAdapter(self.validation_results)
        return integration.generate_model_config()


def main():
    """Example usage of feature adapters"""
    logger.info("Testing feature adapters...")

    # Initialize adapter
    adapter = FeatureAdapter()

    # Test with sample data
    test_season = 2023
    baseline = adapter.prepare_baseline_features(test_season, week=1)
    logger.info(f"Prepared {len(baseline)} baseline features")

    if not baseline.empty:
        epa = adapter.prepare_epa_features(baseline)
        logger.info(f"Prepared {len(epa.columns)} EPA features")

        injury = adapter.prepare_injury_features(baseline)
        logger.info(f"Prepared {len(injury.columns)} injury features")

        weather = adapter.prepare_weather_features(baseline)
        logger.info(f"Prepared {len(weather.columns)} weather features")

    # Test validation runner
    runner = ValidationRunner()
    logger.info("Running EPA validation...")

    # This would run actual validation
    # results = runner.run_epa_validation(2023)
    # config = runner.get_integration_config()

    logger.info("Feature adapter testing complete")


if __name__ == "__main__":
    main()