"""
Simplified NFL Feature Selection Demo
=====================================
This demo shows the core concepts without requiring all dependencies.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_games=1000, n_features=100):
    """Create sample NFL betting data."""
    np.random.seed(42)
    
    # Create features
    X = np.random.randn(n_games, n_features)
    
    # Add some informative features
    X[:, 0] *= 2  # Team strength
    X[:, 1] *= 1.5  # Recent performance
    X[:, 2] *= 1.2  # Head-to-head
    
    # Create interactions
    X[:, 10] = X[:, 0] * X[:, 1]  # Interaction feature
    X[:, 11] = X[:, 0] ** 2  # Polynomial feature
    
    # Add correlated features (multicollinearity)
    X[:, 20] = X[:, 0] * 0.9 + np.random.randn(n_games) * 0.1
    X[:, 21] = X[:, 1] * 0.8 + np.random.randn(n_games) * 0.2
    
    # Create target with signal from important features
    signal = (0.5 * X[:, 0] + 
              0.3 * X[:, 1] + 
              0.2 * X[:, 10] + 
              np.random.randn(n_games) * 0.5)
    
    y = (signal > np.median(signal)).astype(int)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    feature_names[0] = 'team_strength'
    feature_names[1] = 'recent_performance'
    feature_names[2] = 'head_to_head'
    feature_names[10] = 'strength_x_performance'
    feature_names[11] = 'strength_squared'
    feature_names[20] = 'corr_strength'
    feature_names[21] = 'corr_performance'
    
    return pd.DataFrame(X, columns=feature_names), y

def calculate_correlation_groups(X, threshold=0.9):
    """Find groups of highly correlated features."""
    corr_matrix = X.corr().abs()
    
    # Find correlated pairs
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                correlated_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    return correlated_pairs

def feature_selection_pipeline(X, y, n_features=20):
    """Complete feature selection pipeline."""
    print("="*60)
    print("NFL FEATURE SELECTION PIPELINE")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Target distribution: {np.bincount(y) / len(y)}")
    
    # 1. Statistical Feature Selection (F-statistic)
    print("\n1. Statistical Feature Selection (ANOVA F-test)")
    print("-"*40)
    selector_f = SelectKBest(score_func=f_classif, k='all')
    selector_f.fit(X_train, y_train)
    f_scores = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)
    
    print("Top 10 features by F-score:")
    for i, row in f_scores.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['f_score']:8.2f}")
    
    # 2. Mutual Information
    print("\n2. Mutual Information Feature Selection")
    print("-"*40)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("Top 10 features by Mutual Information:")
    for i, row in mi_df.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['mi_score']:8.4f}")
    
    # 3. Random Forest Feature Importance
    print("\n3. Random Forest Feature Importance")
    print("-"*40)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 features by Random Forest:")
    for i, row in rf_importance.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:8.4f}")
    
    # 4. Combine scores (ensemble approach)
    print("\n4. Combined Feature Importance")
    print("-"*40)
    
    # Normalize scores
    f_scores['f_norm'] = f_scores['f_score'] / f_scores['f_score'].max()
    mi_df['mi_norm'] = mi_df['mi_score'] / mi_df['mi_score'].max()
    rf_importance['rf_norm'] = rf_importance['importance'] / rf_importance['importance'].max()
    
    # Merge all scores
    combined = f_scores[['feature', 'f_norm']].merge(
        mi_df[['feature', 'mi_norm']], on='feature'
    ).merge(
        rf_importance[['feature', 'rf_norm']], on='feature'
    )
    
    # Calculate ensemble score (weighted average)
    weights = {'f': 0.2, 'mi': 0.3, 'rf': 0.5}
    combined['ensemble_score'] = (
        weights['f'] * combined['f_norm'] +
        weights['mi'] * combined['mi_norm'] +
        weights['rf'] * combined['rf_norm']
    )
    
    combined = combined.sort_values('ensemble_score', ascending=False)
    
    print("Top 10 features by Ensemble Score:")
    for i, row in combined.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['ensemble_score']:8.4f}")
    
    # 5. Handle Multicollinearity
    print("\n5. Multicollinearity Detection")
    print("-"*40)
    corr_pairs = calculate_correlation_groups(X_train, threshold=0.9)
    
    if corr_pairs:
        print("Highly correlated feature pairs (>0.9):")
        for feat1, feat2, corr in corr_pairs[:5]:
            print(f"  {feat1:20s} <-> {feat2:20s}: {corr:.3f}")
        
        # Remove less important feature from correlated pairs
        features_to_remove = set()
        importance_dict = dict(zip(combined['feature'], combined['ensemble_score']))
        
        for feat1, feat2, _ in corr_pairs:
            if importance_dict.get(feat1, 0) > importance_dict.get(feat2, 0):
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        print(f"\nRemoving {len(features_to_remove)} correlated features")
        
        # Filter out correlated features
        combined_filtered = combined[~combined['feature'].isin(features_to_remove)]
    else:
        combined_filtered = combined
    
    # 6. Select final features
    print(f"\n6. Final Feature Selection (Top {n_features})")
    print("-"*40)
    selected_features = combined_filtered.head(n_features)['feature'].tolist()
    
    print(f"Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        score = combined_filtered[combined_filtered['feature'] == feat]['ensemble_score'].values[0]
        print(f"  {i:2d}. {feat:25s}: {score:.4f}")
    
    # 7. Evaluate with selected features
    print("\n7. Model Performance Evaluation")
    print("-"*40)
    
    # Train with all features
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    score_all = rf_all.score(X_test, y_test)
    
    # Train with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selected.fit(X_train_selected, y_train)
    score_selected = rf_selected.score(X_test_selected, y_test)
    
    print(f"Accuracy with all features ({X.shape[1]}): {score_all:.4f}")
    print(f"Accuracy with selected features ({len(selected_features)}): {score_selected:.4f}")
    print(f"Feature reduction: {(1 - len(selected_features)/X.shape[1]):.1%}")
    
    return selected_features, combined_filtered

def main():
    """Main execution."""
    print("""
    NFL Betting Model - Feature Selection Demo
    ===========================================
    
    This demo shows core feature selection concepts:
    - Multiple importance metrics (F-test, MI, RF)
    - Ensemble scoring approach
    - Multicollinearity detection
    - Performance evaluation
    """)
    
    # Create sample data
    print("\nCreating sample NFL betting data...")
    X, y = create_sample_data(n_games=1000, n_features=100)
    
    # Run feature selection
    selected_features, importance_df = feature_selection_pipeline(X, y, n_features=20)
    
    # Show feature categories in selected set
    print("\n" + "="*60)
    print("ANALYSIS OF SELECTED FEATURES")
    print("="*60)
    
    key_features = ['team_strength', 'recent_performance', 'strength_x_performance', 
                   'strength_squared', 'head_to_head']
    
    print("\nKey features captured:")
    for feat in key_features:
        if feat in selected_features:
            rank = selected_features.index(feat) + 1
            print(f"  ✓ {feat:25s} (rank: {rank})")
        else:
            print(f"  ✗ {feat:25s} (not selected)")
    
    print("\n" + "="*60)
    print("Feature selection complete!")
    print("="*60)
    
    return selected_features, importance_df

if __name__ == "__main__":
    selected_features, importance_df = main()
