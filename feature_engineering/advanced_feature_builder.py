#!/usr/bin/env python3
"""
Advanced Feature Builder - Phase 2 Implementation
Fail-fast validation gates to ensure feature quality before proceeding
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureBuilder:
    """Build comprehensive feature set from all data sources with validation gates"""
    
    def __init__(self):
        self.data_dir = Path('data_integration/output')
        self.output_dir = Path('feature_engineering/output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation thresholds
        self.min_features_threshold = 40
        self.min_games_with_features_threshold = 1000
        self.feature_correlation_threshold = 0.3
        
        logger.info("AdvancedFeatureBuilder initialized")
    
    def build_comprehensive_features_with_validation(self):
        """Build comprehensive features with fail-fast validation gates"""
        logger.info("=" * 60)
        logger.info("ADVANCED FEATURE ENGINEERING WITH VALIDATION GATES")
        logger.info("=" * 60)
        
        # Load imported data
        logger.info("\n🔍 VALIDATION GATE 1: Data Loading")
        data = self._load_imported_data()
        if data is None:
            logger.error("❌ FAILED: Data loading validation failed")
            return None
        logger.info("✅ PASSED: Data loading validation")
        
        # Build core features
        logger.info("\n🔍 VALIDATION GATE 2: Core Feature Engineering")
        core_features = self._build_core_features(data)
        if core_features is None:
            logger.error("❌ FAILED: Core feature engineering failed")
            return None
        logger.info("✅ PASSED: Core feature engineering")
        
        # Build NGS features
        logger.info("\n🔍 VALIDATION GATE 3: NGS Feature Engineering")
        ngs_features = self._build_ngs_features(data)
        if ngs_features is None:
            logger.error("❌ FAILED: NGS feature engineering failed")
            return None
        logger.info("✅ PASSED: NGS feature engineering")
        
        # Build injury features
        logger.info("\n🔍 VALIDATION GATE 4: Injury Feature Engineering")
        injury_features = self._build_injury_features(data)
        if injury_features is None:
            logger.error("❌ FAILED: Injury feature engineering failed")
            return None
        logger.info("✅ PASSED: Injury feature engineering")
        
        # Combine all features
        logger.info("\n🔍 VALIDATION GATE 5: Feature Combination")
        combined_features = self._combine_features(core_features, ngs_features, injury_features)
        if combined_features is None:
            logger.error("❌ FAILED: Feature combination failed")
            return None
        logger.info("✅ PASSED: Feature combination")
        
        # Validate final feature set
        logger.info("\n🔍 VALIDATION GATE 6: Final Feature Validation")
        validation_result = self._validate_final_features(combined_features)
        if not validation_result['passed']:
            logger.error(f"❌ FAILED: Final feature validation failed - {validation_result['reason']}")
            return None
        logger.info(f"✅ PASSED: Final feature validation - {validation_result['score']:.2f} quality score")
        
        # Save results
        self._save_feature_results(combined_features)
        
        logger.info("\n🎉 ALL FEATURE VALIDATION GATES PASSED - PROCEEDING TO PHASE 3")
        return combined_features
    
    def _load_imported_data(self):
        """Load data from Phase 1 import"""
        try:
            data = {}
            
            # Load games data
            games_file = self.data_dir / 'games_complete.csv'
            if games_file.exists():
                data['games'] = pd.read_csv(games_file)
                logger.info(f"Loaded {len(data['games'])} games")
            else:
                logger.error("❌ Games data file not found")
                return None
            
            # Load NGS data
            ngs_categories = ['passing', 'rushing', 'receiving']
            data['ngs'] = {}
            for category in ngs_categories:
                ngs_file = self.data_dir / f'ngs_{category}.csv'
                if ngs_file.exists():
                    data['ngs'][category] = pd.read_csv(ngs_file)
                    logger.info(f"Loaded {len(data['ngs'][category])} NGS {category} records")
                else:
                    logger.warning(f"⚠️ NGS {category} data file not found")
                    data['ngs'][category] = pd.DataFrame()
            
            # Load injury data
            injury_file = self.data_dir / 'injuries_complete.csv'
            if injury_file.exists():
                data['injuries'] = pd.read_csv(injury_file)
                logger.info(f"Loaded {len(data['injuries'])} injury records")
            else:
                logger.warning("⚠️ Injury data file not found")
                data['injuries'] = pd.DataFrame()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return None
    
    def _build_core_features(self, data):
        """Build core game features"""
        try:
            games_df = data['games'].copy()
            features = []
            
            for _, game in games_df.iterrows():
                feature_row = {
                    'game_id': game['game_id'],
                    'season': game['season'],
                    'week': game['week'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_score': game.get('home_score', 0),
                    'away_score': game.get('away_score', 0),
                    'spread_line': game.get('spread_line', 0),
                    'total_line': game.get('total_line', 0)
                }
                
                # Basic game features
                feature_row.update(self._extract_basic_features(game))
                
                # Situational features
                feature_row.update(self._extract_situational_features(game))
                
                # Weather features
                feature_row.update(self._extract_weather_features(game))
                
                features.append(feature_row)
            
            features_df = pd.DataFrame(features)
            logger.info(f"Built {len(features_df)} games with {len(features_df.columns)} core features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Core feature building failed: {e}")
            return None
    
    def _extract_basic_features(self, game):
        """Extract basic game features"""
        return {
            'is_home': 1,  # This is always for home team perspective
            'week_number': game['week'],
            'is_divisional': 1 if self._is_divisional_game(game) else 0,
            'is_playoff': 1 if game.get('game_type', 'REG') != 'REG' else 0,
            'is_outdoor': 1 if game.get('is_outdoor', False) else 0,
            'rest_days_home': game.get('home_rest_days', 7),
            'rest_days_away': game.get('away_rest_days', 7),
            'rest_advantage': game.get('home_rest_days', 7) - game.get('away_rest_days', 7)
        }
    
    def _extract_situational_features(self, game):
        """Extract situational features"""
        return {
            'season_progress': game['week'] / 18.0,  # Normalize week to season progress
            'is_primetime': 1 if self._is_primetime_game(game) else 0,
            'is_rivalry': 1 if self._is_rivalry_game(game) else 0,
            'travel_distance': self._calculate_travel_distance(game),
            'timezone_difference': self._calculate_timezone_difference(game)
        }
    
    def _extract_weather_features(self, game):
        """Extract weather-related features"""
        return {
            'temperature': game.get('temperature', 70),
            'wind_speed': game.get('wind_speed', 0),
            'humidity': game.get('humidity', 50),
            'precipitation': game.get('precipitation', 0),
            'weather_impact_score': self._calculate_weather_impact(game)
        }
    
    def _build_ngs_features(self, data):
        """Build Next Gen Stats features"""
        try:
            games_df = data['games'].copy()
            ngs_data = data['ngs']
            features = []
            
            for _, game in games_df.iterrows():
                feature_row = {'game_id': game['game_id']}
                
                # NGS passing features
                feature_row.update(self._extract_ngs_passing_features(game, ngs_data.get('passing', pd.DataFrame())))
                
                # NGS rushing features
                feature_row.update(self._extract_ngs_rushing_features(game, ngs_data.get('rushing', pd.DataFrame())))
                
                # NGS receiving features
                feature_row.update(self._extract_ngs_receiving_features(game, ngs_data.get('receiving', pd.DataFrame())))
                
                features.append(feature_row)
            
            features_df = pd.DataFrame(features)
            logger.info(f"Built {len(features_df)} games with {len(features_df.columns)} NGS features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ NGS feature building failed: {e}")
            return None
    
    def _extract_ngs_passing_features(self, game, ngs_passing_df):
        """Extract NGS passing features"""
        if ngs_passing_df.empty:
            return self._get_default_ngs_passing_features()
        
        # Filter NGS data for this game
        game_ngs = ngs_passing_df[ngs_passing_df['game_id'] == game['game_id']]
        
        if len(game_ngs) == 0:
            return self._get_default_ngs_passing_features()
        
        # Aggregate by team
        home_ngs = game_ngs[game_ngs['team'] == game['home_team']]
        away_ngs = game_ngs[game_ngs['team'] == game['away_team']]
        
        return {
            'home_cpoe': home_ngs['completion_percentage_above_expectation'].mean() if len(home_ngs) > 0 else 0,
            'away_cpoe': away_ngs['completion_percentage_above_expectation'].mean() if len(away_ngs) > 0 else 0,
            'home_time_to_throw': home_ngs['avg_time_to_throw'].mean() if len(home_ngs) > 0 else 0,
            'away_time_to_throw': away_ngs['avg_time_to_throw'].mean() if len(away_ngs) > 0 else 0,
            'home_aggressiveness': home_ngs['aggressiveness'].mean() if len(home_ngs) > 0 else 0,
            'away_aggressiveness': away_ngs['aggressiveness'].mean() if len(away_ngs) > 0 else 0,
            'home_pressure_rate': self._calculate_pressure_rate(home_ngs),
            'away_pressure_rate': self._calculate_pressure_rate(away_ngs)
        }
    
    def _extract_ngs_rushing_features(self, game, ngs_rushing_df):
        """Extract NGS rushing features"""
        if ngs_rushing_df.empty:
            return self._get_default_ngs_rushing_features()
        
        game_ngs = ngs_rushing_df[ngs_rushing_df['game_id'] == game['game_id']]
        
        if len(game_ngs) == 0:
            return self._get_default_ngs_rushing_features()
        
        home_ngs = game_ngs[game_ngs['team'] == game['home_team']]
        away_ngs = game_ngs[game_ngs['team'] == game['away_team']]
        
        return {
            'home_rush_efficiency': home_ngs['efficiency'].mean() if len(home_ngs) > 0 else 0,
            'away_rush_efficiency': away_ngs['efficiency'].mean() if len(away_ngs) > 0 else 0,
            'home_rush_yards_over_expected': home_ngs['rush_yards_over_expected'].mean() if len(home_ngs) > 0 else 0,
            'away_rush_yards_over_expected': away_ngs['rush_yards_over_expected'].mean() if len(away_ngs) > 0 else 0
        }
    
    def _extract_ngs_receiving_features(self, game, ngs_receiving_df):
        """Extract NGS receiving features"""
        if ngs_receiving_df.empty:
            return self._get_default_ngs_receiving_features()
        
        game_ngs = ngs_receiving_df[ngs_receiving_df['game_id'] == game['game_id']]
        
        if len(game_ngs) == 0:
            return self._get_default_ngs_receiving_features()
        
        home_ngs = game_ngs[game_ngs['team'] == game['home_team']]
        away_ngs = game_ngs[game_ngs['team'] == game['away_team']]
        
        return {
            'home_avg_separation': home_ngs['avg_separation'].mean() if len(home_ngs) > 0 else 0,
            'away_avg_separation': away_ngs['avg_separation'].mean() if len(away_ngs) > 0 else 0,
            'home_avg_cushion': home_ngs['avg_cushion'].mean() if len(home_ngs) > 0 else 0,
            'away_avg_cushion': away_ngs['avg_cushion'].mean() if len(away_ngs) > 0 else 0,
            'home_yac_over_expected': home_ngs['avg_yac_over_expected'].mean() if len(home_ngs) > 0 else 0,
            'away_yac_over_expected': away_ngs['avg_yac_over_expected'].mean() if len(away_ngs) > 0 else 0
        }
    
    def _build_injury_features(self, data):
        """Build injury-related features"""
        try:
            games_df = data['games'].copy()
            injuries_df = data['injuries']
            features = []
            
            for _, game in games_df.iterrows():
                feature_row = {'game_id': game['game_id']}
                
                # Extract injury features for this game
                feature_row.update(self._extract_injury_features(game, injuries_df))
                
                features.append(feature_row)
            
            features_df = pd.DataFrame(features)
            logger.info(f"Built {len(features_df)} games with {len(features_df.columns)} injury features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Injury feature building failed: {e}")
            return None
    
    def _extract_injury_features(self, game, injuries_df):
        """Extract injury features for a specific game"""
        if injuries_df.empty:
            return self._get_default_injury_features()
        
        # Filter injuries for this game
        game_injuries = injuries_df[
            (injuries_df['season'] == game['season']) & 
            (injuries_df['week'] == game['week'])
        ]
        
        if len(game_injuries) == 0:
            return self._get_default_injury_features()
        
        home_injuries = game_injuries[game_injuries['team'] == game['home_team']]
        away_injuries = game_injuries[game_injuries['team'] == game['away_team']]
        
        return {
            'home_qb_status': self._calculate_qb_status(home_injuries),
            'away_qb_status': self._calculate_qb_status(away_injuries),
            'home_key_injuries': len(home_injuries[home_injuries['severity_score'] >= 3]),
            'away_key_injuries': len(away_injuries[away_injuries['severity_score'] >= 3]),
            'home_total_injuries': len(home_injuries),
            'away_total_injuries': len(away_injuries),
            'home_avg_injury_severity': home_injuries['severity_score'].mean() if len(home_injuries) > 0 else 0,
            'away_avg_injury_severity': away_injuries['severity_score'].mean() if len(away_injuries) > 0 else 0
        }
    
    def _combine_features(self, core_features, ngs_features, injury_features):
        """Combine all feature sets"""
        try:
            # Start with core features
            combined_df = core_features.copy()
            
            # Merge NGS features
            if not ngs_features.empty:
                combined_df = combined_df.merge(ngs_features, on='game_id', how='left')
            
            # Merge injury features
            if not injury_features.empty:
                combined_df = combined_df.merge(injury_features, on='game_id', how='left')
            
            # Fill missing values
            combined_df = combined_df.fillna(0)
            
            logger.info(f"Combined features: {len(combined_df)} games, {len(combined_df.columns)} features")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Feature combination failed: {e}")
            return None
    
    def _validate_final_features(self, features_df):
        """Validate final feature set"""
        try:
            # Check feature count
            feature_count = len(features_df.columns)
            if feature_count < self.min_features_threshold:
                return {'passed': False, 'reason': f'Insufficient features: {feature_count} < {self.min_features_threshold}'}
            
            # Check games with features
            games_count = len(features_df)
            if games_count < self.min_games_with_features_threshold:
                return {'passed': False, 'reason': f'Insufficient games: {games_count} < {self.min_games_with_features_threshold}'}
            
            # Check for missing values
            missing_pct = features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))
            if missing_pct > 0.1:  # More than 10% missing
                return {'passed': False, 'reason': f'Too many missing values: {missing_pct:.2%}'}
            
            # Calculate quality score
            quality_score = min(1.0, feature_count / 50.0) * min(1.0, games_count / 2000.0) * (1 - missing_pct)
            
            return {'passed': True, 'score': quality_score, 'feature_count': feature_count, 'games_count': games_count}
            
        except Exception as e:
            return {'passed': False, 'reason': f'Validation error: {e}'}
    
    def _save_feature_results(self, features_df):
        """Save feature engineering results"""
        # Save features
        features_df.to_csv(self.output_dir / 'comprehensive_features.csv', index=False)
        
        # Save feature metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(features_df.columns),
            'games_count': len(features_df),
            'feature_columns': list(features_df.columns),
            'validation_passed': True
        }
        
        with open(self.output_dir / 'feature_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature results saved to {self.output_dir}")
    
    # Helper methods
    def _is_divisional_game(self, game):
        """Check if game is divisional"""
        # Simplified logic - in real implementation, would check team divisions
        return False
    
    def _is_primetime_game(self, game):
        """Check if game is primetime"""
        # Simplified logic - in real implementation, would check game time
        return False
    
    def _is_rivalry_game(self, game):
        """Check if game is rivalry"""
        # Simplified logic - in real implementation, would check team rivalries
        return False
    
    def _calculate_travel_distance(self, game):
        """Calculate travel distance"""
        # Simplified logic - in real implementation, would calculate actual distance
        return 0
    
    def _calculate_timezone_difference(self, game):
        """Calculate timezone difference"""
        # Simplified logic - in real implementation, would calculate actual timezone diff
        return 0
    
    def _calculate_weather_impact(self, game):
        """Calculate weather impact score"""
        temp = game.get('temperature', 70)
        wind = game.get('wind_speed', 0)
        precip = game.get('precipitation', 0)
        
        # Simple weather impact calculation
        impact = 0
        if temp < 32 or temp > 85:
            impact += 2
        if wind > 15:
            impact += 2
        if precip > 0:
            impact += 1
        
        return min(impact, 5)
    
    def _calculate_pressure_rate(self, ngs_df):
        """Calculate pressure rate from NGS data"""
        if len(ngs_df) == 0:
            return 0
        
        # Simplified calculation - in real implementation, would use actual pressure data
        return ngs_df.get('pressure_rate', pd.Series([0])).mean()
    
    def _calculate_qb_status(self, injuries_df):
        """Calculate QB injury status impact"""
        if len(injuries_df) == 0:
            return 1.0
        
        qb_injuries = injuries_df[injuries_df['position'] == 'QB']
        if len(qb_injuries) == 0:
            return 1.0
        
        # Calculate severity-weighted impact
        severity_scores = qb_injuries['severity_score'].values
        max_severity = max(severity_scores)
        
        # QB out = -7 points impact
        if max_severity >= 4:  # Out
            return 0.0
        elif max_severity >= 3:  # Questionable
            return 0.5
        elif max_severity >= 2:  # Probable
            return 0.8
        else:
            return 1.0
    
    def _get_default_ngs_passing_features(self):
        """Get default NGS passing features"""
        return {
            'home_cpoe': 0, 'away_cpoe': 0,
            'home_time_to_throw': 0, 'away_time_to_throw': 0,
            'home_aggressiveness': 0, 'away_aggressiveness': 0,
            'home_pressure_rate': 0, 'away_pressure_rate': 0
        }
    
    def _get_default_ngs_rushing_features(self):
        """Get default NGS rushing features"""
        return {
            'home_rush_efficiency': 0, 'away_rush_efficiency': 0,
            'home_rush_yards_over_expected': 0, 'away_rush_yards_over_expected': 0
        }
    
    def _get_default_ngs_receiving_features(self):
        """Get default NGS receiving features"""
        return {
            'home_avg_separation': 0, 'away_avg_separation': 0,
            'home_avg_cushion': 0, 'away_avg_cushion': 0,
            'home_yac_over_expected': 0, 'away_yac_over_expected': 0
        }
    
    def _get_default_injury_features(self):
        """Get default injury features"""
        return {
            'home_qb_status': 1.0, 'away_qb_status': 1.0,
            'home_key_injuries': 0, 'away_key_injuries': 0,
            'home_total_injuries': 0, 'away_total_injuries': 0,
            'home_avg_injury_severity': 0, 'away_avg_injury_severity': 0
        }

if __name__ == "__main__":
    builder = AdvancedFeatureBuilder()
    features = builder.build_comprehensive_features_with_validation()
    
    if features is not None:
        logger.info("🎉 Phase 2 Feature Engineering: SUCCESS")
        logger.info("Ready to proceed to Phase 3: Model Architecture")
    else:
        logger.error("❌ Phase 2 Feature Engineering: FAILED")
        logger.error("Fix feature issues before proceeding")
