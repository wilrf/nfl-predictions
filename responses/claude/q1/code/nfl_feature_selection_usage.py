"""
NFL Feature Selection - Usage Guide and Examples
================================================

This script demonstrates how to use the NFLFeatureSelector class with real data
and provides practical examples for production deployment.
"""

import pandas as pd
import numpy as np
from nfl_feature_selection import NFLFeatureSelector
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# EXAMPLE 1: Basic Usage with Your Data
# ==============================================================================

def basic_usage_example():
    """
    Basic example showing how to use the feature selector with your actual data.
    """
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Load your actual NFL data
    # df = pd.read_csv('your_nfl_data.csv')
    
    # For demonstration, we'll use sample data
    from nfl_feature_selection import create_sample_nfl_data
    df = create_sample_nfl_data(n_games=2500, n_features=500)
    
    # Initialize the feature selector
    selector = NFLFeatureSelector(
        target_col='covered_spread',    # Your target column
        n_top_features=25,              # Number of features to select
        correlation_threshold=0.90,     # Correlation threshold
        vif_threshold=10.0,            # VIF threshold
        random_state=42                # For reproducibility
    )
    
    # Fit the selector
    selector.fit(df)
    
    # Get selected features
    selected_features = selector.selected_features
    print(f"\nSelected {len(selected_features)} features")
    
    # Use selected features for your model
    X = df[selected_features]
    y = df['covered_spread']
    
    return selector, X, y


# ==============================================================================
# EXAMPLE 2: Custom Feature Engineering Pipeline
# ==============================================================================

def feature_engineering_pipeline():
    """
    Example showing how to integrate feature engineering before selection.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Feature Engineering + Selection")
    print("="*80)
    
    # Load data
    from nfl_feature_selection import create_sample_nfl_data
    df = create_sample_nfl_data(n_games=2500, n_features=300)
    
    # Add custom engineered features
    print("Adding engineered features...")
    
    # 1. Rolling averages (team momentum)
    for col in ['team_stat_0', 'team_stat_1', 'team_stat_2']:
        if col in df.columns:
            df[f'{col}_rolling_3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_rolling_7'] = df[col].rolling(window=7, min_periods=1).mean()
    
    # 2. Interaction features
    df['offense_defense_ratio'] = df['team_stat_0'] / (df['team_stat_1'] + 1)
    df['weather_impact'] = df['temperature'] * df['wind_speed'] / 100
    
    # 3. Polynomial features for key metrics
    df['epa_squared'] = df['epa_metric_0'] ** 2
    df['epa_cubed'] = df['epa_metric_0'] ** 3
    
    # 4. Lag features (previous game performance)
    for i in [1, 2, 3]:
        df[f'prev_game_{i}'] = df['team_stat_0'].shift(i)
    
    # 5. Categorical encodings
    if 'home_team' in df.columns:
        team_performance = df.groupby('home_team')['covered_spread'].mean()
        df['home_team_win_rate'] = df['home_team'].map(team_performance)
    
    print(f"Total features after engineering: {df.shape[1]}")
    
    # Run feature selection
    selector = NFLFeatureSelector(
        target_col='covered_spread',
        n_top_features=30,
        correlation_threshold=0.85,  # Stricter threshold
        random_state=42
    )
    
    selector.fit(df)
    
    return selector


# ==============================================================================
# EXAMPLE 3: Cross-Validation and Model Evaluation
# ==============================================================================

def cross_validation_example():
    """
    Example showing proper cross-validation with selected features.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Cross-Validation Pipeline")
    print("="*80)
    
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from xgboost import XGBClassifier
    from sklearn.metrics import make_scorer, roc_auc_score
    
    # Load and prepare data
    from nfl_feature_selection import create_sample_nfl_data
    df = create_sample_nfl_data(n_games=2500, n_features=500)
    
    # Add a date column for time series split (simulated)
    df['game_date'] = pd.date_range(start='2014-01-01', periods=len(df), freq='D')
    df = df.sort_values('game_date')
    
    # Feature selection
    selector = NFLFeatureSelector(
        target_col='covered_spread',
        n_top_features=25,
        random_state=42
    )
    
    selector.fit(df.drop(columns=['game_date']))
    
    # Prepare data with selected features
    X = df[selector.selected_features]
    y = df['covered_spread']
    
    # Time Series Cross-Validation (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train model with selected features
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    
    # Evaluate
    scores = cross_val_score(
        model, X, y, 
        cv=tscv, 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"Time Series CV Results:")
    print(f"  Mean AUC: {scores.mean():.4f}")
    print(f"  Std AUC:  {scores.std():.4f}")
    print(f"  All scores: {scores}")
    
    return scores


# ==============================================================================
# EXAMPLE 4: Feature Importance Analysis
# ==============================================================================

def analyze_feature_importance():
    """
    Deep dive into understanding feature importance and interactions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Feature Importance Analysis")
    print("="*80)
    
    from nfl_feature_selection import create_sample_nfl_data
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load data and run selection
    df = create_sample_nfl_data(n_games=2500, n_features=500)
    
    selector = NFLFeatureSelector(
        target_col='covered_spread',
        n_top_features=20,
        random_state=42
    )
    
    selector.fit(df)
    
    # 1. Analyze feature importance breakdown
    importance_df = selector.feature_importance_df
    
    print("\nTop 10 Features by Importance:")
    print("-" * 50)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30s} | Score: {row['combined_score']:.4f}")
    
    # 2. Analyze feature categories
    feature_categories = {
        'team_stats': [],
        'player_metrics': [],
        'epa_metrics': [],
        'weather': [],
        'other': []
    }
    
    for feature in selector.selected_features:
        if 'team_stat' in feature:
            feature_categories['team_stats'].append(feature)
        elif 'player_metric' in feature:
            feature_categories['player_metrics'].append(feature)
        elif 'epa' in feature:
            feature_categories['epa_metrics'].append(feature)
        elif feature in ['temperature', 'wind_speed', 'humidity']:
            feature_categories['weather'].append(feature)
        else:
            feature_categories['other'].append(feature)
    
    print("\nFeature Distribution by Category:")
    print("-" * 50)
    for category, features in feature_categories.items():
        print(f"{category:15s}: {len(features):3d} features")
    
    # 3. Interaction analysis
    if hasattr(selector, 'interaction_matrix') and selector.interaction_matrix is not None:
        print("\nTop Feature Interactions Detected:")
        print("-" * 50)
        # The interaction matrix was already printed during fitting
    
    return selector


# ==============================================================================
# EXAMPLE 5: Production Deployment
# ==============================================================================

class ProductionFeatureSelector:
    """
    Production-ready wrapper for feature selection in NFL betting models.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize with configuration file or default settings.
        """
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        self.selector = None
        self.selected_features = None
        self.feature_stats = {}
        
    @staticmethod
    def get_default_config():
        return {
            'n_top_features': 25,
            'correlation_threshold': 0.90,
            'vif_threshold': 10.0,
            'random_state': 42,
            'target_col': 'covered_spread',
            'validation_split': 0.2,
            'cv_folds': 5,
            'min_feature_importance': 0.001
        }
    
    def load_config(self, path: str):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the feature selector with production checks.
        """
        # Data quality checks
        self._validate_data(df)
        
        # Store statistics for monitoring
        self.feature_stats['n_samples'] = len(df)
        self.feature_stats['n_features_original'] = df.shape[1] - 1  # Exclude target
        self.feature_stats['target_distribution'] = df[self.config['target_col']].value_counts().to_dict()
        
        # Initialize and fit selector
        self.selector = NFLFeatureSelector(
            target_col=self.config['target_col'],
            n_top_features=self.config['n_top_features'],
            correlation_threshold=self.config['correlation_threshold'],
            vif_threshold=self.config['vif_threshold'],
            random_state=self.config['random_state']
        )
        
        self.selector.fit(df)
        self.selected_features = self.selector.selected_features
        
        # Store additional stats
        self.feature_stats['n_features_selected'] = len(self.selected_features)
        self.feature_stats['reduction_ratio'] = (
            1 - self.feature_stats['n_features_selected'] / self.feature_stats['n_features_original']
        )
        
        return self
    
    def _validate_data(self, df: pd.DataFrame):
        """
        Validate input data for production use.
        """
        # Check for target column
        if self.config['target_col'] not in df.columns:
            raise ValueError(f"Target column '{self.config['target_col']}' not found")
        
        # Check for minimum samples
        if len(df) < 100:
            raise ValueError(f"Insufficient data: {len(df)} samples (minimum: 100)")
        
        # Check target balance
        target_balance = df[self.config['target_col']].value_counts(normalize=True)
        if target_balance.min() < 0.1:
            print("WARNING: Severe class imbalance detected")
        
        # Check for constant features
        constant_features = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"WARNING: {len(constant_features)} constant features detected")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using selected features.
        """
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        # Ensure all selected features are present
        missing_features = set(self.selected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        return df[self.selected_features]
    
    def get_report(self) -> dict:
        """
        Generate comprehensive report for monitoring.
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': self.config,
            'statistics': self.feature_stats,
            'selected_features': self.selected_features,
            'feature_importance': self.selector.feature_importance_df.to_dict('records') 
                                  if hasattr(self.selector, 'feature_importance_df') else None
        }
        
        return report


def production_example():
    """
    Example of production deployment.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Production Deployment")
    print("="*80)
    
    # Initialize production selector
    prod_selector = ProductionFeatureSelector()
    
    # Load your data
    from nfl_feature_selection import create_sample_nfl_data
    train_df = create_sample_nfl_data(n_games=2000, n_features=500)
    test_df = create_sample_nfl_data(n_games=500, n_features=500)
    
    # Fit selector
    prod_selector.fit(train_df)
    
    # Transform test data
    X_test_selected = prod_selector.transform(test_df)
    
    # Get report
    report = prod_selector.get_report()
    
    print(f"\nProduction Feature Selection Report:")
    print(f"  Original features: {report['statistics']['n_features_original']}")
    print(f"  Selected features: {report['statistics']['n_features_selected']}")
    print(f"  Reduction ratio: {report['statistics']['reduction_ratio']:.1%}")
    
    return prod_selector


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("""
    NFL Feature Selection - Usage Examples
    =======================================
    
    This script demonstrates various usage patterns for the
    NFLFeatureSelector class in different scenarios.
    """)
    
    # Run examples
    print("\nRunning Example 1: Basic Usage")
    selector1, X, y = basic_usage_example()
    
    print("\nRunning Example 2: Feature Engineering Pipeline")
    selector2 = feature_engineering_pipeline()
    
    print("\nRunning Example 3: Cross-Validation")
    cv_scores = cross_validation_example()
    
    print("\nRunning Example 4: Feature Importance Analysis")
    selector4 = analyze_feature_importance()
    
    print("\nRunning Example 5: Production Deployment")
    prod_selector = production_example()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)
    
    print("""
    Next Steps:
    -----------
    1. Replace sample data with your actual NFL dataset
    2. Adjust parameters based on your specific needs
    3. Add custom feature engineering relevant to your strategy
    4. Integrate with your betting model pipeline
    5. Monitor feature importance drift over time
    6. Retrain periodically with new data
    """)
