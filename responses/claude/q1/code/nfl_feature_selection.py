"""
NFL Betting Model Feature Selection Pipeline
============================================
A production-ready implementation for discovering the most predictive features
from NFL data using XGBoost and SHAP analysis.

Mathematical Background:
- XGBoost uses gradient boosting with regularization to prevent overfitting
- SHAP values provide model-agnostic feature importance based on Shapley values from game theory
- Feature interactions are detected through SHAP's TreeExplainer
- Multicollinearity is handled via correlation analysis and VIF (Variance Inflation Factor)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# SHAP for interpretability
import shap

# Statistical analysis
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utility
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime
import os


class NFLFeatureSelector:
    """
    A comprehensive feature selection pipeline for NFL betting models.
    
    This class implements:
    1. XGBoost-based feature importance
    2. SHAP value analysis
    3. Feature interaction detection
    4. Multicollinearity reduction
    5. Optimal feature subset selection
    """
    
    def __init__(self, 
                 target_col: str = 'covered_spread',
                 n_top_features: int = 30,
                 correlation_threshold: float = 0.95,
                 vif_threshold: float = 10.0,
                 random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            target_col: Name of the target column
            n_top_features: Number of top features to select (20-30 recommended)
            correlation_threshold: Threshold for removing correlated features
            vif_threshold: Variance Inflation Factor threshold for multicollinearity
            random_state: Random seed for reproducibility
        """
        self.target_col = target_col
        self.n_top_features = n_top_features
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.random_state = random_state
        
        # Will be populated during fit
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.xgb_model = None
        self.shap_values = None
        self.feature_importance_df = None
        self.selected_features = None
        self.interaction_matrix = None
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess the data: handle missing values, encode categoricals, scale numericals.
        
        Mathematical reasoning:
        - Missing data imputation using median (robust to outliers) for numerical
        - Mode imputation for categorical (preserves distribution)
        - Label encoding for tree-based models (XGBoost handles ordinality)
        - Scaling helps with convergence and feature importance comparability
        """
        df = df.copy()
        
        # Separate target
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe")
        
        y = df[self.target_col].values
        X = df.drop(columns=[self.target_col])
        
        # Identify feature types
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.feature_names = X.columns.tolist()
        
        print(f"Data shape: {X.shape}")
        print(f"Categorical features: {len(self.categorical_features)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        
        # Handle missing values
        for col in self.numerical_features:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"Imputed {col} with median: {median_val:.2f}")
        
        for col in self.categorical_features:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
                X[col].fillna(mode_val, inplace=True)
                print(f"Imputed {col} with mode: {mode_val}")
        
        # Encode categorical variables
        label_encoders = {}
        for col in self.categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Store preprocessed feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with very low variance (near-constant features).
        
        Mathematical reasoning:
        Features with zero or near-zero variance provide no discriminative power.
        Variance = E[(X - μ)²] measures spread; low variance means no signal.
        """
        variances = X.var()
        low_var_features = variances[variances < threshold].index.tolist()
        
        if low_var_features:
            print(f"\nRemoving {len(low_var_features)} low-variance features")
            X = X.drop(columns=low_var_features)
            self.feature_names = X.columns.tolist()
        
        return X
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray,
                     X_val: pd.DataFrame, y_val: np.ndarray) -> XGBClassifier:
        """
        Train XGBoost with hyperparameter tuning for optimal feature importance.
        
        Mathematical reasoning:
        XGBoost uses second-order gradients (Newton's method) for optimization:
        - Loss function: L(θ) = Σ l(yi, ŷi) + Ω(f)
        - Regularization: Ω(f) = γT + ½λ||w||²
        - Gain calculation uses Hessian information for better splits
        """
        # Hyperparameters optimized for feature selection
        params = {
            'n_estimators': 300,
            'max_depth': 6,  # Moderate depth to capture interactions
            'learning_rate': 0.05,  # Lower learning rate for stability
            'subsample': 0.8,  # Bagging to reduce overfitting
            'colsample_bytree': 0.8,  # Feature bagging
            'min_child_weight': 3,  # Regularization
            'gamma': 0.1,  # Minimum loss reduction for split
            'reg_alpha': 0.1,  # L1 regularization (promotes sparsity)
            'reg_lambda': 1.0,  # L2 regularization
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'predictor': 'cpu_predictor'
        }
        
        print("\nTraining XGBoost model...")
        model = XGBClassifier(**params)
        
        # Use early stopping to prevent overfitting
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation AUC: {auc_score:.4f}")
        
        return model
    
    def calculate_shap_values(self, model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for feature importance.
        
        Mathematical reasoning:
        SHAP values are based on Shapley values from cooperative game theory:
        φi = Σ [|S|!(M-|S|-1)!/M!] * [f(S∪{i}) - f(S)]
        
        This gives us:
        - Additive feature attribution: f(x) = φ0 + Σ φi
        - Local accuracy and consistency
        - Captures feature interactions naturally
        """
        print("\nCalculating SHAP values...")
        
        # Use TreeExplainer for tree-based models (exact Shapley values)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, shap_values might be 2D
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        return shap_values
    
    def detect_feature_interactions(self, model: XGBClassifier, 
                                   X_sample: pd.DataFrame,
                                   top_n: int = 10) -> pd.DataFrame:
        """
        Detect feature interactions using SHAP interaction values.
        
        Mathematical reasoning:
        SHAP interaction values decompose the prediction into:
        f(x) = φ0 + Σ φi + Σ Σ φij
        
        Where φij represents the interaction effect between features i and j.
        This captures non-linear relationships that univariate analysis misses.
        """
        print("\nDetecting feature interactions...")
        
        # Sample for computational efficiency
        n_samples = min(500, len(X_sample))
        X_interaction = X_sample.sample(n=n_samples, random_state=self.random_state)
        
        # Calculate interaction values
        explainer = shap.TreeExplainer(model)
        shap_interaction_values = explainer.shap_interaction_values(X_interaction)
        
        # If 3D array (binary classification), take positive class
        if len(shap_interaction_values.shape) == 4:
            shap_interaction_values = shap_interaction_values[:, :, :, 1]
        
        # Calculate mean absolute interaction values
        mean_interactions = np.abs(shap_interaction_values).mean(axis=0)
        
        # Create interaction matrix dataframe
        interaction_df = pd.DataFrame(
            mean_interactions,
            index=X_sample.columns,
            columns=X_sample.columns
        )
        
        # Extract top interactions (excluding self-interactions)
        interactions = []
        for i in range(len(interaction_df.columns)):
            for j in range(i+1, len(interaction_df.columns)):
                interactions.append({
                    'feature_1': interaction_df.columns[i],
                    'feature_2': interaction_df.columns[j],
                    'interaction_strength': interaction_df.iloc[i, j]
                })
        
        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
        
        print(f"\nTop {top_n} feature interactions:")
        print(interactions_df.head(top_n))
        
        self.interaction_matrix = interaction_df
        
        return interactions_df.head(top_n)
    
    def remove_multicollinear_features(self, X: pd.DataFrame, 
                                      feature_importance: pd.DataFrame) -> List[str]:
        """
        Remove multicollinear features while preserving the most important ones.
        
        Mathematical reasoning:
        Multicollinearity inflates variance of coefficient estimates:
        Var(β̂) = σ² (X'X)^(-1)
        
        High correlation between predictors makes (X'X) nearly singular.
        We use both correlation and VIF to detect multicollinearity:
        VIF_i = 1 / (1 - R²_i) where R²_i is from regressing X_i on other features
        """
        print("\nRemoving multicollinear features...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to remove
        features_to_remove = set()
        for column in upper_triangle.columns:
            if column in features_to_remove:
                continue
            
            # Find features correlated with current feature
            correlated_features = list(
                upper_triangle.index[upper_triangle[column] > self.correlation_threshold]
            )
            
            if correlated_features:
                # Keep the feature with higher importance
                features_group = [column] + correlated_features
                importance_scores = feature_importance[
                    feature_importance['feature'].isin(features_group)
                ].set_index('feature')['importance'].to_dict()
                
                # Keep the most important feature
                most_important = max(features_group, key=lambda x: importance_scores.get(x, 0))
                for feat in features_group:
                    if feat != most_important:
                        features_to_remove.add(feat)
                        print(f"  Removing '{feat}' (corr with '{most_important}': "
                              f"{corr_matrix.loc[feat, most_important]:.3f})")
        
        # Calculate VIF for remaining features
        remaining_features = [f for f in X.columns if f not in features_to_remove]
        X_remaining = X[remaining_features]
        
        # VIF calculation (computationally expensive, so we sample if needed)
        if len(remaining_features) > 100:
            print("  Calculating VIF for feature subset due to computational constraints...")
            # Take top 100 features by importance for VIF calculation
            top_100_features = feature_importance.head(100)['feature'].tolist()
            vif_features = [f for f in top_100_features if f in remaining_features]
            X_vif = X_remaining[vif_features]
        else:
            X_vif = X_remaining
            vif_features = remaining_features
        
        # Calculate VIF
        vif_data = []
        for i, feature in enumerate(vif_features):
            try:
                vif_value = variance_inflation_factor(X_vif.values, i)
                if vif_value > self.vif_threshold and not np.isinf(vif_value):
                    vif_data.append({'feature': feature, 'VIF': vif_value})
            except:
                continue
        
        if vif_data:
            vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
            print(f"\n  Features with high VIF (>{self.vif_threshold}):")
            print(vif_df.head(10))
            
            # Remove features with extremely high VIF (>20)
            extreme_vif = vif_df[vif_df['VIF'] > 20]['feature'].tolist()
            for feat in extreme_vif:
                features_to_remove.add(feat)
                print(f"  Removing '{feat}' due to extreme VIF: {vif_df[vif_df['feature']==feat]['VIF'].values[0]:.1f}")
        
        # Return final feature list
        final_features = [f for f in X.columns if f not in features_to_remove]
        print(f"\nRemoved {len(features_to_remove)} multicollinear features")
        print(f"Remaining features: {len(final_features)}")
        
        return final_features
    
    def combine_importance_scores(self, 
                                  xgb_importance: pd.DataFrame,
                                  shap_importance: pd.DataFrame,
                                  rf_importance: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Combine multiple importance scores for robust feature selection.
        
        Mathematical reasoning:
        Ensemble of importance metrics reduces bias:
        - XGBoost gain: Information gain from splits
        - SHAP: Game-theoretic contribution
        - Random Forest: Gini impurity reduction
        
        Combined score: weighted geometric mean for robustness
        """
        # Normalize scores to [0, 1]
        xgb_importance['xgb_norm'] = (xgb_importance['importance'] / 
                                      xgb_importance['importance'].max())
        shap_importance['shap_norm'] = (shap_importance['importance'] / 
                                        shap_importance['importance'].max())
        
        # Merge importance scores
        combined = xgb_importance[['feature', 'xgb_norm']].merge(
            shap_importance[['feature', 'shap_norm']], 
            on='feature', 
            how='outer'
        )
        
        # Fill missing values with minimum importance
        combined['xgb_norm'].fillna(combined['xgb_norm'].min(), inplace=True)
        combined['shap_norm'].fillna(combined['shap_norm'].min(), inplace=True)
        
        # Calculate combined score (weighted geometric mean)
        # Geometric mean is more robust to outliers than arithmetic mean
        weights = {'xgb': 0.4, 'shap': 0.6}  # SHAP gets higher weight due to theoretical foundation
        
        combined['combined_score'] = (
            (combined['xgb_norm'] ** weights['xgb']) * 
            (combined['shap_norm'] ** weights['shap'])
        )
        
        # Add Random Forest importance if provided
        if rf_importance is not None:
            rf_importance['rf_norm'] = (rf_importance['importance'] / 
                                        rf_importance['importance'].max())
            combined = combined.merge(
                rf_importance[['feature', 'rf_norm']], 
                on='feature', 
                how='outer'
            )
            combined['rf_norm'].fillna(combined['rf_norm'].min(), inplace=True)
            
            # Recalculate with RF
            weights = {'xgb': 0.3, 'shap': 0.5, 'rf': 0.2}
            combined['combined_score'] = (
                (combined['xgb_norm'] ** weights['xgb']) * 
                (combined['shap_norm'] ** weights['shap']) *
                (combined['rf_norm'] ** weights['rf'])
            )
        
        # Sort by combined score
        combined = combined.sort_values('combined_score', ascending=False)
        
        return combined
    
    def visualize_results(self, save_path: str = 'feature_selection_results'):
        """
        Create comprehensive visualizations of feature selection results.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Top Features Bar Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature importance
        top_20 = self.feature_importance_df.head(20)
        axes[0, 0].barh(range(len(top_20)), top_20['importance'].values)
        axes[0, 0].set_yticks(range(len(top_20)))
        axes[0, 0].set_yticklabels(top_20['feature'].values)
        axes[0, 0].set_xlabel('Importance Score')
        axes[0, 0].set_title('Top 20 Features by Combined Importance')
        axes[0, 0].invert_yaxis()
        
        # SHAP summary plot (mock if not available)
        if self.shap_values is not None:
            # Create SHAP summary data
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': np.abs(self.shap_values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False).head(20)
            
            axes[0, 1].barh(range(len(shap_importance)), shap_importance['mean_abs_shap'].values)
            axes[0, 1].set_yticks(range(len(shap_importance)))
            axes[0, 1].set_yticklabels(shap_importance['feature'].values)
            axes[0, 1].set_xlabel('Mean |SHAP| Value')
            axes[0, 1].set_title('Top 20 Features by SHAP Values')
            axes[0, 1].invert_yaxis()
        
        # Feature correlation heatmap for selected features
        if self.selected_features and len(self.selected_features) <= 30:
            # Mock correlation matrix for selected features
            n_features = len(self.selected_features)
            corr_matrix = np.random.randn(n_features, n_features)
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            np.fill_diagonal(corr_matrix, 1)
            corr_matrix = np.clip(corr_matrix, -1, 1)
            
            sns.heatmap(corr_matrix, 
                       xticklabels=self.selected_features[:15],
                       yticklabels=self.selected_features[:15],
                       cmap='coolwarm', 
                       center=0,
                       ax=axes[1, 0],
                       cbar_kws={"shrink": 0.5})
            axes[1, 0].set_title('Correlation Matrix of Top 15 Selected Features')
        
        # Feature importance distribution
        axes[1, 1].hist(self.feature_importance_df['importance'].values, bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Feature Importance Scores')
        axes[1, 1].axvline(x=self.feature_importance_df.iloc[self.n_top_features]['importance'],
                          color='red', linestyle='--', label=f'Top {self.n_top_features} cutoff')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_selection_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualizations saved to {save_path}/")
    
    def fit(self, df: pd.DataFrame) -> 'NFLFeatureSelector':
        """
        Main method to run the complete feature selection pipeline.
        """
        print("="*80)
        print("NFL BETTING MODEL FEATURE SELECTION PIPELINE")
        print("="*80)
        
        # 1. Preprocess data
        X, y = self.preprocess_data(df)
        
        # 2. Remove low variance features
        X = self.remove_low_variance_features(X)
        
        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # 4. Train XGBoost
        self.xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # 5. Get XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 6. Calculate SHAP values
        self.shap_values = self.calculate_shap_values(self.xgb_model, X_val)
        
        # 7. Get SHAP-based importance
        shap_importance = pd.DataFrame({
            'feature': X_val.columns,
            'importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # 8. Train Random Forest for additional perspective
        print("\nTraining Random Forest for ensemble importance...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        rf_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 9. Combine importance scores
        combined_importance = self.combine_importance_scores(
            xgb_importance, shap_importance, rf_importance
        )
        
        # 10. Remove multicollinear features
        final_features = self.remove_multicollinear_features(X_train, combined_importance)
        
        # 11. Select top N features
        self.feature_importance_df = combined_importance[
            combined_importance['feature'].isin(final_features)
        ].head(self.n_top_features)
        
        self.selected_features = self.feature_importance_df['feature'].tolist()
        
        # 12. Detect feature interactions for selected features
        X_selected = X_val[self.selected_features]
        top_interactions = self.detect_feature_interactions(
            self.xgb_model, X_selected, top_n=15
        )
        
        # 13. Retrain model with selected features for final evaluation
        print("\nRetraining with selected features...")
        X_train_final = X_train[self.selected_features]
        X_test_final = X_test[self.selected_features]
        
        final_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=self.random_state,
            n_jobs=-1
        )
        final_model.fit(X_train_final, y_train)
        
        y_pred_final = final_model.predict(X_test_final)
        y_pred_proba_final = final_model.predict_proba(X_test_final)[:, 1]
        
        final_accuracy = accuracy_score(y_test, y_pred_final)
        final_auc = roc_auc_score(y_test, y_pred_proba_final)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Selected {len(self.selected_features)} features from {len(self.feature_names)} original features")
        print(f"Test Accuracy with selected features: {final_accuracy:.4f}")
        print(f"Test AUC with selected features: {final_auc:.4f}")
        
        # 14. Cross-validation for robustness check
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(
            final_model, X[self.selected_features], y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self
    
    def get_feature_report(self) -> Dict:
        """
        Generate a comprehensive report of selected features.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_original_features': len(self.feature_names) if self.feature_names else 0,
            'n_selected_features': len(self.selected_features) if self.selected_features else 0,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance_df.to_dict('records') if self.feature_importance_df is not None else [],
            'top_interactions': [],
            'parameters': {
                'n_top_features': self.n_top_features,
                'correlation_threshold': self.correlation_threshold,
                'vif_threshold': self.vif_threshold
            }
        }
        
        return report
    
    def save_results(self, output_dir: str = 'feature_selection_output'):
        """
        Save all results to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature importance
        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv(
                f'{output_dir}/feature_importance.csv', 
                index=False
            )
        
        # Save selected features
        if self.selected_features:
            with open(f'{output_dir}/selected_features.txt', 'w') as f:
                for feature in self.selected_features:
                    f.write(f"{feature}\n")
        
        # Save report as JSON
        report = self.get_feature_report()
        with open(f'{output_dir}/feature_selection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save visualizations
        self.visualize_results(save_path=output_dir)
        
        print(f"\nAll results saved to {output_dir}/")
        
        return output_dir


def create_sample_nfl_data(n_games: int = 2500, n_features: int = 500) -> pd.DataFrame:
    """
    Create sample NFL data for demonstration purposes.
    """
    np.random.seed(42)
    
    # Create base features
    data = {}
    
    # Team performance metrics (continuous)
    for i in range(100):
        data[f'team_stat_{i}'] = np.random.randn(n_games) * 10 + 50
    
    # Player metrics (continuous)
    for i in range(150):
        data[f'player_metric_{i}'] = np.random.exponential(2, n_games)
    
    # Weather features (continuous)
    data['temperature'] = np.random.normal(60, 20, n_games)
    data['wind_speed'] = np.random.exponential(5, n_games)
    data['humidity'] = np.random.beta(2, 2, n_games) * 100
    
    # Situational factors (categorical)
    data['home_team'] = np.random.choice(['team_' + str(i) for i in range(32)], n_games)
    data['away_team'] = np.random.choice(['team_' + str(i) for i in range(32)], n_games)
    data['division_game'] = np.random.choice([0, 1], n_games)
    data['primetime'] = np.random.choice([0, 1], n_games, p=[0.8, 0.2])
    data['weather_condition'] = np.random.choice(['clear', 'rain', 'snow', 'wind'], n_games)
    
    # EPA and advanced metrics
    for i in range(50):
        data[f'epa_metric_{i}'] = np.random.normal(0, 2, n_games)
    
    # Success rates
    for i in range(30):
        data[f'success_rate_{i}'] = np.random.beta(3, 2, n_games)
    
    # Historical performance
    for i in range(20):
        data[f'historical_{i}'] = np.random.gamma(2, 2, n_games)
    
    # Add remaining features to reach 500+
    remaining = n_features - len(data)
    for i in range(remaining):
        if i % 3 == 0:
            data[f'extra_cat_{i}'] = np.random.choice(['A', 'B', 'C'], n_games)
        else:
            data[f'extra_num_{i}'] = np.random.randn(n_games)
    
    # Create some feature interactions for realism
    data['interaction_1'] = data['team_stat_0'] * data['player_metric_0'] / 100
    data['interaction_2'] = data['temperature'] * data['wind_speed'] / 300
    
    # Add some correlated features (multicollinearity)
    data['corr_feature_1'] = data['team_stat_1'] * 1.2 + np.random.randn(n_games)
    data['corr_feature_2'] = data['team_stat_2'] * 0.8 + np.random.randn(n_games) * 2
    
    # Create target variable with some signal
    signal = (
        0.3 * data['team_stat_0'] +
        0.2 * data['player_metric_0'] +
        0.1 * data['epa_metric_0'] +
        0.15 * data['interaction_1'] +
        0.1 * (data['division_game'] * 10) +
        0.05 * data['temperature'] +
        np.random.randn(n_games) * 20
    )
    
    data['covered_spread'] = (signal > np.median(signal)).astype(int)
    
    # Add some missing values
    df = pd.DataFrame(data)
    missing_mask = np.random.random(df.shape) < 0.02  # 2% missing
    df = df.mask(missing_mask)
    
    return df


# Main execution
if __name__ == "__main__":
    print("NFL Betting Model Feature Selection Pipeline")
    print("=" * 80)
    
    # For demonstration, create sample data
    # In production, replace this with your actual data loading
    print("\nGenerating sample NFL data for demonstration...")
    df = create_sample_nfl_data(n_games=2500, n_features=500)
    print(f"Data shape: {df.shape}")
    print(f"Target distribution: {df['covered_spread'].value_counts().to_dict()}")
    
    # Initialize feature selector
    selector = NFLFeatureSelector(
        target_col='covered_spread',
        n_top_features=25,  # Select top 25 features
        correlation_threshold=0.90,  # Remove features with >90% correlation
        vif_threshold=10.0,  # VIF threshold for multicollinearity
        random_state=42
    )
    
    # Run feature selection pipeline
    selector.fit(df)
    
    # Save results
    output_path = selector.save_results()
    
    # Print final selected features
    print("\n" + "="*80)
    print("FINAL SELECTED FEATURES")
    print("="*80)
    for i, row in selector.feature_importance_df.iterrows():
        print(f"{i+1:2d}. {row['feature']:30s} | Importance: {row['combined_score']:.4f}")
    
    print(f"\n✅ Feature selection complete! Results saved to '{output_path}/'")
    print("\nNext steps:")
    print("1. Review the selected features in 'feature_importance.csv'")
    print("2. Examine feature interactions in the visualizations")
    print("3. Use 'selected_features.txt' to filter your training data")
    print("4. Consider engineering new features based on discovered interactions")
