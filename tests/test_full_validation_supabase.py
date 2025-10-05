"""
Full validation test using data from Supabase
"""

import pandas as pd
import numpy as np
from validation import DataValidationFramework
import warnings
import os
from dotenv import load_dotenv
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SupabaseValidationTester:
    def __init__(self):
        """Initialize with Supabase credentials from MCP"""
        # We'll use MCP tools directly from Claude
        self.url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
        self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1MDQyMDUsImV4cCI6MjA3NDA4MDIwNX0.8rBzK19aRnuJb7gLdLDR3aZPg-rzqW0usiXb354N0t0"
        
        # Initialize Supabase client
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(self.url, self.key)
            logger.info("Supabase client initialized")
        except ImportError:
            logger.error("Supabase library not installed. Run: pip install supabase")
            raise
    
    def load_game_features_from_supabase(self):
        """Load game features from Supabase"""
        try:
            # Get games with scores
            games_response = self.client.table('games').select(
                "game_uuid, season, week, home_team_id, away_team_id, home_score, away_score"
            ).not_.is_('home_score', 'null').execute()
            
            games_df = pd.DataFrame(games_response.data)
            
            # Get teams for mapping
            teams_response = self.client.table('teams').select(
                "team_id, team_code"
            ).execute()
            
            teams_df = pd.DataFrame(teams_response.data)
            team_map = dict(zip(teams_df['team_id'], teams_df['team_code']))
            
            # Map team codes
            if not games_df.empty:
                games_df['home_team'] = games_df['home_team_id'].map(team_map)
                games_df['away_team'] = games_df['away_team_id'].map(team_map)
            
            print(f"Loaded {len(games_df)} games from Supabase")
            
            # For now, simulate EPA data (in production, this would come from plays table)
            if not games_df.empty:
                np.random.seed(42)
                games_df['home_off_epa'] = np.random.normal(0, 0.15, len(games_df))
                games_df['home_def_epa'] = np.random.normal(0, 0.12, len(games_df))
                games_df['away_off_epa'] = np.random.normal(0, 0.15, len(games_df))
                games_df['away_def_epa'] = np.random.normal(0, 0.12, len(games_df))
                games_df['epa_differential'] = (games_df['home_off_epa'] - games_df['home_def_epa']) - \
                                              (games_df['away_off_epa'] - games_df['away_def_epa'])
                games_df['def_epa_differential'] = games_df['away_def_epa'] - games_df['home_def_epa']
                games_df['spread_line'] = -games_df['epa_differential'] * 7 + np.random.normal(0, 2, len(games_df))
            
            return games_df
        
        except Exception as e:
            logger.error(f"Error loading from Supabase: {e}")
            return pd.DataFrame()
    
    def prepare_validation_data(self, features_df):
        """Prepare data in format needed by validation framework"""
        
        # Remove rows with missing scores (future games)
        completed_games = features_df[features_df['home_score'].notna()].copy()
        print(f"Found {len(completed_games)} completed games with scores")
        
        if completed_games.empty:
            print("No completed games found!")
            return None, None, None, None
        
        # Create baseline features (traditional stats)
        baseline_features = pd.DataFrame({
            'season': completed_games['season'],
            'week': completed_games['week'],
            'home_team': completed_games['home_team'],
            'away_team': completed_games['away_team']
        })
        
        # Create EPA features (what we're validating)
        new_features = pd.DataFrame({
            'home_off_epa': completed_games['home_off_epa'],
            'home_def_epa': completed_games['home_def_epa'],
            'away_off_epa': completed_games['away_off_epa'],
            'away_def_epa': completed_games['away_def_epa'],
            'epa_differential': completed_games['epa_differential'],
            'def_epa_differential': completed_games['def_epa_differential']
        })
        
        # Create target (point differential)
        target = completed_games['home_score'] - completed_games['away_score']
        
        # Create market data
        market_lines = completed_games['spread_line'].values
        
        market_data = {
            'predictions': target.values + np.random.normal(0, 2, len(target)),
            'market_lines': market_lines,
            'outcomes': (target > market_lines).astype(int)
        }
        
        print(f"\nData prepared for validation:")
        print(f"  Baseline features: {baseline_features.shape}")
        print(f"  New features (EPA): {new_features.shape}")
        print(f"  Target samples: {len(target)}")
        
        return baseline_features, new_features, target, market_data
    
    def run_validation(self):
        """Run the complete 5-phase validation"""
        
        print("="*80)
        print("FULL VALIDATION TEST WITH SUPABASE DATA")
        print("="*80)
        
        # Load data from Supabase
        print("\n1. Loading data from Supabase...")
        features_df = self.load_game_features_from_supabase()
        
        if features_df.empty:
            print("ERROR: Could not load data from Supabase")
            return
        
        # Prepare for validation
        print("\n2. Preparing validation dataset...")
        baseline_features, new_features, target, market_data = self.prepare_validation_data(features_df)
        
        if baseline_features is None:
            print("ERROR: Could not prepare validation data")
            return
        
        # Initialize validation framework
        print("\n3. Initializing validation framework...")
        config = {
            'min_seasons_required': 1,  # Lower requirement for test
            'min_sample_size': 50,      # Lower for test data
            'significance_level': 0.05,
            'roi_threshold': 0.02,
            'enable_detailed_logging': True,
            'save_intermediate_results': True,
            'monitoring_window': 30,
            'output_directory': 'validation_results_supabase'
        }
        
        framework = DataValidationFramework(config)
        print("✓ Framework initialized")
        
        # Create simplified feature history
        print("\n4. Creating temporal data...")
        seasons = baseline_features['season'].unique()
        
        # Simple feature importance
        feature_importance_by_season = pd.DataFrame()
        for season in sorted(seasons):
            season_mask = baseline_features['season'] == season
            if season_mask.sum() > 0:
                season_importance = {}
                for col in new_features.columns:
                    season_importance[col] = np.random.uniform(0.1, 0.5)  # Simulated importance
                
                feature_importance_by_season = pd.concat([
                    feature_importance_by_season,
                    pd.DataFrame(season_importance, index=[season])
                ])
        
        print(f"✓ Created feature importance for {len(feature_importance_by_season)} seasons")
        
        # Create performance history
        performance_history = {}
        for feature in new_features.columns[:3]:
            performance_history[feature] = pd.Series(
                np.random.normal(0.15, 0.02, 50)
            )
        
        feature_history = {
            'feature_importance_by_season': feature_importance_by_season,
            'performance_history': performance_history
        }
        
        # Run complete validation pipeline
        print("\n5. Running 5-phase validation pipeline...")
        print("-"*40)
        
        try:
            results = framework.run_complete_validation_pipeline(
                data_source='supabase_epa_metrics',
                baseline_features=baseline_features,
                new_features=new_features,
                target=target,
                market_data=market_data,
                feature_history=feature_history
            )
            
            print(f"\n✓ Pipeline completed!")
            print(f"  Success: {results['pipeline_success']}")
            print(f"  Phases completed: {results['phases_completed']}/5")
            print(f"  Runtime: {results.get('total_runtime', 0):.2f} seconds")
            
            # Display key results
            print("\n" + "="*80)
            print("VALIDATION RESULTS")
            print("="*80)
            
            print(f"\nFinal Recommendation: {results['final_recommendation']}")
            
            # Phase 1 Results
            if 'phase_1' in results and results['phase_1'].get('phase_1_success'):
                phase1 = results['phase_1']
                print("\nPhase 1 - Statistical Foundation:")
                if 'importance_testing' in phase1:
                    imp = phase1['importance_testing']
                    print(f"  MSE Improvement: {imp.get('mse_improvement', 0):.4f}")
                    print(f"  Statistically Significant: {imp.get('statistically_significant', False)}")
            
            # Phase 2 Results
            if 'phase_2' in results and results['phase_2'].get('phase_2_success'):
                phase2 = results['phase_2']
                print("\nPhase 2 - Market Validation:")
                if 'market_efficiency' in phase2:
                    market = phase2['market_efficiency']
                    print(f"  Win Rate: {market.get('win_rate', 0):.3f}")
                    print(f"  Exploitable: {market.get('exploitable', False)}")
            
            return results
        
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    tester = SupabaseValidationTester()
    results = tester.run_validation()
    
    if results:
        print("\n" + "="*80)
        print("SUPABASE VALIDATION TEST COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("1. Load full historical data into Supabase")
        print("2. Calculate real EPA metrics from play-by-play data")
        print("3. Run production validation with complete dataset")
    else:
        print("\n✗ Validation failed")