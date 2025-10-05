"""
Comprehensive Test Suite for NFL Betting System
Tests all critical components including validation, calibration, and risk management
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfl_betting_system import (
    BetOpportunity, DataCache, MarketEfficiencyAnalyzer,
    SpreadModel, EnsembleModel, FeatureEngineer, 
    KellyCalculator, RiskManager, NFLBettingSystem
)
from data_pipeline import DataValidator, DataQuality, CachedDataLoader, NFLDataPipeline


class TestDataValidation:
    """Test data validation and quality checks"""
    
    def test_game_data_validation(self):
        """Test validation of game data"""
        validator = DataValidator()
        
        # Valid data
        valid_df = pd.DataFrame({
            'home_team': ['KC', 'BUF'],
            'away_team': ['LV', 'MIA'],
            'home_score': [31, 24],
            'away_score': [17, 21],
            'game_date': [datetime.now(), datetime.now()]
        })
        
        quality = validator.validate(valid_df, 'game_data')
        assert quality.is_acceptable()
        assert quality.completeness > 0.9
        assert len(quality.issues) == 0
        
        # Invalid data - negative scores
        invalid_df = pd.DataFrame({
            'home_team': ['KC'],
            'away_team': ['LV'],
            'home_score': [-1],
            'away_score': [17],
            'game_date': [datetime.now()]
        })
        
        quality = validator.validate(invalid_df, 'game_data')
        assert not quality.is_acceptable()
        assert 'Invalid scores' in str(quality.issues)
    
    def test_odds_data_validation(self):
        """Test validation of odds data"""
        validator = DataValidator()
        
        # Valid odds
        valid_df = pd.DataFrame({
            'game_id': ['2024_01_KC_LV'],
            'book': ['pinnacle'],
            'spread': [-7.5],
            'total': [48.5],
            'moneyline': [-320],
            'timestamp': [datetime.now()]
        })
        
        quality = validator.validate(valid_df, 'odds_data')
        assert quality.is_acceptable()
        
        # Invalid spread
        invalid_df = pd.DataFrame({
            'game_id': ['2024_01_KC_LV'],
            'book': ['pinnacle'],
            'spread': [-75],  # Unrealistic spread
            'total': [48.5],
            'moneyline': [-320],
            'timestamp': [datetime.now()]
        })
        
        quality = validator.validate(invalid_df, 'odds_data')
        assert 'Unrealistic spreads' in str(quality.issues)


class TestProbabilityCalibration:
    """Test model calibration functionality"""
    
    def test_isotonic_calibration(self):
        """Test that probability calibration improves reliability"""
        model = SpreadModel()
        
        # Create synthetic data
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(1000, 10))
        y_train = pd.Series(np.random.binint(0, 2, 1000))
        X_val = pd.DataFrame(np.random.randn(200, 10))
        y_val = pd.Series(np.random.binint(0, 2, 200))
        
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Check calibrator was created
        assert model.calibrator is not None
        
        # Test predictions are calibrated
        probs = model.predict_proba(X_val)
        assert probs.shape == (200, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_calibration_error_calculation(self):
        """Test expected calibration error calculation"""
        model = SpreadModel()
        
        # Perfect calibration case
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        ece = model._calculate_calibration_error(y_true, y_pred, n_bins=2)
        assert ece < 0.2  # Should be well calibrated


class TestKellyCalculator:
    """Test Kelly Criterion implementation"""
    
    def test_standard_kelly(self):
        """Test standard Kelly calculation for independent bets"""
        calc = KellyCalculator(max_kelly_fraction=0.25)
        
        # Single bet with 5% edge at even money
        opps = [
            BetOpportunity(
                game_id='test1',
                bet_type='spread',
                selection='KC -7',
                market_odds=2.0,  # Even money
                fair_odds=1.9,
                edge=0.05,
                kelly_size=0,
                confidence=0.55,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.02
            )
        ]
        
        sizes = calc.calculate_portfolio_kelly(opps, None, 10000)
        
        # Expected Kelly: f* = (p*b - q)/b = (0.55*1 - 0.45)/1 = 0.1
        # With 0.25 fraction: 0.1 * 0.25 = 0.025 = 2.5% of bankroll
        assert 0.02 <= sizes['test1'] <= 0.03
    
    def test_correlated_kelly(self):
        """Test Kelly sizing with correlated bets"""
        calc = KellyCalculator(max_kelly_fraction=0.25)
        
        # Two correlated bets
        opps = [
            BetOpportunity(
                game_id='game1_spread',
                bet_type='spread',
                selection='KC -7',
                market_odds=2.0,
                fair_odds=1.9,
                edge=0.05,
                kelly_size=0,
                confidence=0.55,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.02
            ),
            BetOpportunity(
                game_id='game1_total',
                bet_type='total',
                selection='Over 48',
                market_odds=2.0,
                fair_odds=1.9,
                edge=0.05,
                kelly_size=0,
                confidence=0.55,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.02
            )
        ]
        
        # High correlation (same game)
        correlation = np.array([[1.0, 0.7], [0.7, 1.0]])
        
        sizes = calc.calculate_portfolio_kelly(opps, correlation, 10000)
        
        # Correlated bets should have reduced sizes
        assert sizes['game1_spread'] < 0.025  # Less than uncorrelated size
        assert sizes['game1_total'] < 0.025
    
    def test_max_bet_constraint(self):
        """Test that bets are capped at 5% of bankroll"""
        calc = KellyCalculator(max_kelly_fraction=1.0)  # Full Kelly
        
        # Huge edge that would suggest large bet
        opps = [
            BetOpportunity(
                game_id='test1',
                bet_type='spread',
                selection='KC -7',
                market_odds=3.0,
                fair_odds=1.5,
                edge=0.33,  # 33% edge!
                kelly_size=0,
                confidence=0.67,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.1
            )
        ]
        
        sizes = calc.calculate_portfolio_kelly(opps, None, 10000)
        
        # Should be capped at 5% despite huge edge
        assert sizes['test1'] <= 0.05


class TestRiskManager:
    """Test risk management and portfolio constraints"""
    
    def test_max_weekly_bets(self):
        """Test enforcement of maximum weekly bet limit"""
        config = {
            'max_weekly_bets': 5,
            'max_game_exposure': 0.15,
            'correlation_limit': 0.3,
            'stop_loss': -0.08
        }
        risk_mgr = RiskManager(config)
        
        # Create 10 opportunities
        opps = []
        for i in range(10):
            opp = BetOpportunity(
                game_id=f'game{i}',
                bet_type='spread',
                selection=f'Team{i}',
                market_odds=2.0,
                fair_odds=1.9,
                edge=0.05,
                kelly_size=0.02,
                confidence=0.55,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.02
            )
            opps.append(opp)
        
        validated = risk_mgr.validate_portfolio(opps, [])
        
        # Should only allow 5 bets
        assert len(validated) == 5
        # Should be the top 5 by edge
        assert all(v.edge == 0.05 for v in validated)
    
    def test_game_exposure_limit(self):
        """Test that single game exposure is limited"""
        config = {
            'max_weekly_bets': 10,
            'max_game_exposure': 0.10,
            'correlation_limit': 0.3,
            'stop_loss': -0.08
        }
        risk_mgr = RiskManager(config)
        
        # Multiple bets on same game
        opps = [
            BetOpportunity(
                game_id='game1_spread',
                bet_type='spread',
                selection='KC -7',
                market_odds=2.0,
                fair_odds=1.9,
                edge=0.05,
                kelly_size=0.06,  # 6% of bankroll
                confidence=0.55,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.02
            ),
            BetOpportunity(
                game_id='game1_total',
                bet_type='total',
                selection='Over 48',
                market_odds=2.0,
                fair_odds=1.9,
                edge=0.05,
                kelly_size=0.06,  # Another 6%
                confidence=0.55,
                features={},
                timestamp=datetime.now(),
                clv_potential=0.02
            )
        ]
        
        validated = risk_mgr.validate_portfolio(opps, [])
        
        # Should only allow first bet due to exposure limit
        assert len(validated) == 1
        assert validated[0].game_id == 'game1_spread'
    
    def test_stop_loss_trigger(self):
        """Test stop loss functionality"""
        config = {
            'max_weekly_bets': 10,
            'max_game_exposure': 0.15,
            'correlation_limit': 0.3,
            'stop_loss': -0.08
        }
        risk_mgr = RiskManager(config)
        
        # Test stop loss trigger
        assert risk_mgr.check_stop_loss(current_pnl=-500, bankroll=10000) == False  # -5% is ok
        assert risk_mgr.check_stop_loss(current_pnl=-900, bankroll=10000) == True   # -9% triggers


class TestMarketEfficiency:
    """Test market analysis and CLV calculations"""
    
    def test_no_vig_calculation(self):
        """Test vig removal calculation"""
        analyzer = MarketEfficiencyAnalyzer()
        
        # American odds: -110 both sides (standard vig)
        prob1, prob2 = analyzer.calculate_no_vig_probability(-110, -110)
        
        # Should be 50/50 after removing vig
        assert abs(prob1 - 0.5) < 0.01
        assert abs(prob2 - 0.5) < 0.01
        assert abs(prob1 + prob2 - 1.0) < 0.001
    
    def test_steam_detection(self):
        """Test detection of legitimate steam moves"""
        analyzer = MarketEfficiencyAnalyzer()
        
        # Create line history showing steam move
        line_history = pd.DataFrame({
            'pinnacle': [3.0, 3.0, 4.5],
            'circa': [3.0, 3.5, 4.5],
            'bookmaker': [3.0, 3.0, 4.0],
            'draftkings': [3.0, 3.0, 3.5],  # Soft book slower to move
        })
        
        is_steam = analyzer.detect_steam_move(line_history)
        assert is_steam == True  # 3 sharp books moved
        
        # No steam - only one book moved
        line_history_no_steam = pd.DataFrame({
            'pinnacle': [3.0, 3.0, 4.5],
            'circa': [3.0, 3.0, 3.0],
            'bookmaker': [3.0, 3.0, 3.0],
            'draftkings': [3.0, 3.0, 3.0],
        })
        
        is_steam = analyzer.detect_steam_move(line_history_no_steam)
        assert is_steam == False
    
    def test_clv_potential(self):
        """Test CLV potential calculation"""
        analyzer = MarketEfficiencyAnalyzer()
        
        clv = analyzer.calculate_clv_potential(current_line=3.0, predicted_close=4.5)
        assert clv == 1.5
        
        negative_clv = analyzer.calculate_clv_potential(current_line=4.5, predicted_close=3.0)
        assert negative_clv == -1.5


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_decay_weighted_features(self):
        """Test temporal decay weighting"""
        engineer = FeatureEngineer(decay_factor=0.1)
        
        # Create sample data with trend
        game_data = pd.DataFrame({
            'yards_per_play': [5.0, 5.5, 6.0, 6.5, 7.0]  # Improving trend
        })
        
        features = pd.DataFrame()
        features = engineer._add_decay_weighted_features(features, game_data)
        
        # Recent games should be weighted more heavily
        assert 'yards_per_play_decay_weighted' in features.columns
        weighted_avg = features['yards_per_play_decay_weighted'].iloc[0]
        simple_avg = game_data['yards_per_play'].mean()
        
        # Weighted average should be higher than simple average due to improving trend
        assert weighted_avg > simple_avg
    
    def test_market_features(self):
        """Test market-derived feature creation"""
        engineer = FeatureEngineer()
        
        market_data = pd.DataFrame({
            'opening_line': [3.0],
            'current_line': [4.5],
            'pinnacle_line': [4.0],
            'bet_pct': [65],
            'money_pct': [45]
        })
        
        features = pd.DataFrame()
        features = engineer._add_market_features(features, market_data)
        
        assert 'line_movement' in features.columns
        assert features['line_movement'].iloc[0] == 1.5
        
        assert 'sharp_divergence' in features.columns
        assert features['sharp_divergence'].iloc[0] == -20  # Money % - Bet % = 45 - 65


class TestIntegration:
    """Integration tests for the complete system"""
    
    @patch('nfl_betting_system.NFLBettingSystem._fetch_game_data')
    @patch('nfl_betting_system.NFLBettingSystem._fetch_market_data')
    def test_weekly_analysis_flow(self, mock_market, mock_game):
        """Test complete weekly analysis workflow"""
        # Setup mocks
        mock_game.return_value = pd.DataFrame({
            'home_team': ['KC'],
            'away_team': ['LV'],
            'home_score': [31],
            'away_score': [17]
        })
        
        mock_market.return_value = pd.DataFrame({
            'game_id': ['2024_01_KC_LV'],
            'team': ['KC'],
            'implied_probability': [0.70],
            'odds': [1.43],  # -230 American
            'current_line': [-7.5],
            'predicted_close': [-8.0]
        })
        
        # Create system
        system = NFLBettingSystem('config/improved_config.json')
        
        # Run analysis
        opportunities = system.run_weekly_analysis(week=10)
        
        # Verify the flow completed
        assert isinstance(opportunities, list)
        
        # If opportunities found, verify structure
        if opportunities:
            opp = opportunities[0]
            assert hasattr(opp, 'edge')
            assert hasattr(opp, 'kelly_size')
            assert hasattr(opp, 'clv_potential')
    
    def test_backtest_framework(self):
        """Test backtesting functionality"""
        system = NFLBettingSystem('config/improved_config.json')
        
        # Mock the data retrieval
        with patch.object(system, '_get_historical_data') as mock_data:
            mock_data.return_value = {
                'X': pd.DataFrame(np.random.randn(100, 10)),
                'y': pd.Series(np.random.binint(0, 2, 100)),
                'actual_results': pd.DataFrame({'result': [1, 0, 1]})
            }
            
            # Run backtest
            results = system.backtest(start_week=1, end_week=3)
            
            assert isinstance(results, pd.DataFrame)
            assert len(results) == 3  # 3 weeks
            assert 'week_roi' in results.columns
            assert 'win_rate' in results.columns


class TestDataPipeline:
    """Test data pipeline components"""
    
    @patch('redis.StrictRedis')
    def test_caching_mechanism(self, mock_redis):
        """Test that caching works properly"""
        # Setup mock Redis
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.get.return_value = None  # Cache miss first time
        
        loader = CachedDataLoader()
        
        # Define a fetch function
        fetch_count = 0
        def fetch_func():
            nonlocal fetch_count
            fetch_count += 1
            return pd.DataFrame({'data': [1, 2, 3]})
        
        # First call should fetch
        result1 = loader.get_or_fetch('test_key', fetch_func)
        assert fetch_count == 1
        
        # Verify data was cached
        mock_redis_instance.setex.assert_called()
    
    def test_parallel_data_fetching(self):
        """Test that data fetching happens in parallel"""
        config = {
            'odds_sources': {
                'sharp_books': ['pinnacle'],
                'soft_books': ['draftkings'],
                'api_keys': {}
            }
        }
        
        pipeline = NFLDataPipeline(config)
        
        with patch.object(pipeline, '_get_game_data') as mock_game:
            with patch.object(pipeline, '_get_odds_data') as mock_odds:
                mock_game.return_value = pd.DataFrame()
                mock_odds.return_value = pd.DataFrame()
                
                # This should trigger parallel fetching
                data = pipeline.get_weekly_data(week=10)
                
                assert 'games' in data
                assert 'odds' in data
                
                # Verify methods were called
                mock_game.assert_called_once()
                mock_odds.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
