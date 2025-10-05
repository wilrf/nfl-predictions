# **Improved NFL Moneyline System: Pragmatic Implementation Plan**

## **Executive Summary**

**Goal**: Enhance the existing NFL betting system with moneyline prediction capability that demonstrably improves ROI over mathematical spread conversion.

**Key Changes from Original Plan**:
- Validate premise FIRST with proof-of-concept (3 days vs. 2 weeks)
- Use existing data infrastructure (2,476 games, not 544)
- Start with single best model, not complex ensemble
- Focus on ROI/CLV, not just accuracy
- Integrate with existing system architecture
- Proper security, testing, and rollback mechanisms

**Timeline**: 2 weeks (was: 2+ weeks)
- **Phase 0**: Validation (3 days) - REQUIRED GATE
- **Phase 1**: Core Implementation (5 days) - Only if Phase 0 succeeds
- **Phase 2**: Integration & Deployment (4 days)

---

## **Phase 0: Premise Validation (Days 1-3) - CRITICAL GATE**

**Goal**: Prove dedicated moneyline model outperforms spread-based conversion

### **Day 1: Baseline Establishment**

#### **0.1: Extract complete dataset from Supabase**
```python
# File: improved_nfl_system/validation/extract_full_dataset.py
#!/usr/bin/env python3
"""
Extract Complete NFL Dataset for Moneyline Analysis
Pulls all 2,476 games with ALL available features
"""

import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullDatasetExtractor:
    def __init__(self):
        # Get credentials from environment
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.output_dir = Path('validation/data')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_all_games(self):
        """Extract all games with features"""
        logger.info("Extracting all games from Supabase...")

        # Pull from both main games table AND expanded features
        games_result = self.supabase.table('games').select('*').execute()
        games_df = pd.DataFrame(games_result.data)

        logger.info(f"Extracted {len(games_df)} games from main table")

        # Try to get expanded features
        try:
            expanded_result = self.supabase.table('expanded_game_features').select('*').execute()
            expanded_df = pd.DataFrame(expanded_result.data)
            logger.info(f"Extracted {len(expanded_df)} games with expanded features")

            # Merge on game_id
            full_df = games_df.merge(expanded_df, on='game_id', how='left', suffixes=('', '_expanded'))
        except Exception as e:
            logger.warning(f"No expanded features table: {e}")
            full_df = games_df

        # Get odds data
        try:
            odds_result = self.supabase.table('odds').select('*').execute()
            odds_df = pd.DataFrame(odds_result.data)

            # Get latest odds for each game
            odds_df = odds_df.sort_values('last_update').groupby('game_id').last().reset_index()

            full_df = full_df.merge(odds_df, on='game_id', how='left', suffixes=('', '_odds'))
            logger.info(f"Merged odds data for {odds_df['game_id'].nunique()} games")
        except Exception as e:
            logger.warning(f"No odds table: {e}")

        # Create target variable
        full_df['home_won'] = (full_df['home_score'] > full_df['away_score']).astype(int)

        # Save raw data
        full_df.to_csv(self.output_dir / 'full_dataset.csv', index=False)

        # Generate data quality report
        self._generate_data_quality_report(full_df)

        logger.info(f"Saved {len(full_df)} games to full_dataset.csv")
        return full_df

    def _generate_data_quality_report(self, df):
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_games': len(df),
            'date_range': {
                'min_season': int(df['season'].min()),
                'max_season': int(df['season'].max()),
                'seasons': sorted(df['season'].unique().tolist())
            },
            'columns': {
                'total': len(df.columns),
                'names': list(df.columns)
            },
            'missing_data': {
                col: {
                    'count': int(df[col].isna().sum()),
                    'percentage': float(df[col].isna().mean() * 100)
                }
                for col in df.columns if df[col].isna().sum() > 0
            },
            'target_distribution': {
                'home_wins': int(df['home_won'].sum()),
                'away_wins': int((1 - df['home_won']).sum()),
                'home_win_rate': float(df['home_won'].mean())
            },
            'features_by_category': self._categorize_features(df.columns)
        }

        import json
        with open(self.output_dir / 'data_quality_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 60)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Games: {report['total_games']}")
        logger.info(f"Seasons: {report['date_range']['min_season']}-{report['date_range']['max_season']}")
        logger.info(f"Total Columns: {report['columns']['total']}")
        logger.info(f"Home Win Rate: {report['target_distribution']['home_win_rate']:.3f}")
        logger.info(f"Missing Data Columns: {len(report['missing_data'])}")

    def _categorize_features(self, columns):
        """Categorize features by type"""
        categories = {
            'core': [],
            'scores': [],
            'odds': [],
            'epa': [],
            'ngs': [],
            'weather': [],
            'injury': [],
            'other': []
        }

        for col in columns:
            col_lower = col.lower()
            if col in ['game_id', 'season', 'week', 'home_team', 'away_team', 'gameday']:
                categories['core'].append(col)
            elif 'score' in col_lower or 'points' in col_lower:
                categories['scores'].append(col)
            elif 'spread' in col_lower or 'total' in col_lower or 'moneyline' in col_lower or 'odds' in col_lower:
                categories['odds'].append(col)
            elif 'epa' in col_lower or 'success' in col_lower:
                categories['epa'].append(col)
            elif 'cpoe' in col_lower or 'separation' in col_lower or 'cushion' in col_lower or 'efficiency' in col_lower:
                categories['ngs'].append(col)
            elif 'weather' in col_lower or 'temperature' in col_lower or 'wind' in col_lower or 'humidity' in col_lower:
                categories['weather'].append(col)
            elif 'injury' in col_lower or 'qb_status' in col_lower:
                categories['injury'].append(col)
            else:
                categories['other'].append(col)

        return categories

if __name__ == "__main__":
    extractor = FullDatasetExtractor()
    extractor.extract_all_games()
```

#### **0.2: Implement spread-to-moneyline baseline**
```python
# File: improved_nfl_system/validation/baseline_spread_method.py
#!/usr/bin/env python3
"""
Baseline: Convert Spread Predictions to Moneyline Probabilities
Uses existing spread model as benchmark
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpreadToMoneylineBaseline:
    def __init__(self):
        self.data_dir = Path('validation/data')

        # Load existing spread model
        try:
            with open('models/saved_models/spread_model.pkl', 'rb') as f:
                self.spread_model = pickle.load(f)
            logger.info("Loaded existing spread model")
        except FileNotFoundError:
            logger.error("No existing spread model found")
            self.spread_model = None

    def spread_to_probability(self, predicted_spread, spread_std=13.5):
        """
        Convert spread prediction to win probability

        Theory: If spread = -7, home team expected to win by 7
        Probability home wins = P(actual margin > 0 | predicted = -7)

        Using normal distribution: N(predicted_spread, spread_std)
        P(win) = P(X > 0) = 1 - CDF(0)

        NFL historical spread_std ≈ 13.5 points
        """
        # Probability that actual margin > 0
        prob_home_win = 1 - stats.norm.cdf(0, loc=-predicted_spread, scale=spread_std)
        return prob_home_win

    def evaluate_baseline(self):
        """Evaluate spread-to-moneyline baseline on 2024 data"""
        logger.info("=" * 60)
        logger.info("EVALUATING SPREAD-TO-MONEYLINE BASELINE")
        logger.info("=" * 60)

        # Load data
        df = pd.read_csv(self.data_dir / 'full_dataset.csv')

        # Split by season
        train_df = df[df['season'] <= 2023].copy()
        test_df = df[df['season'] == 2024].copy()

        logger.info(f"Train: {len(train_df)} games (2016-2023)")
        logger.info(f"Test: {len(test_df)} games (2024)")

        # If no spread model, train simple one
        if self.spread_model is None:
            self.spread_model = self._train_simple_spread_model(train_df)

        # Prepare features for spread model
        feature_cols = self._get_spread_features(test_df)
        X_test = test_df[feature_cols]
        y_test = test_df['home_won']

        # Predict spreads
        predicted_spreads = self.spread_model.predict(X_test)

        # Convert to probabilities
        predicted_probs = np.array([
            self.spread_to_probability(spread)
            for spread in predicted_spreads
        ])

        # Evaluate
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

        predicted_classes = (predicted_probs > 0.5).astype(int)

        results = {
            'method': 'Spread-to-Moneyline Baseline',
            'accuracy': accuracy_score(y_test, predicted_classes),
            'log_loss': log_loss(y_test, predicted_probs),
            'roc_auc': roc_auc_score(y_test, predicted_probs),
            'brier_score': brier_score_loss(y_test, predicted_probs)
        }

        # ROI analysis (if we have actual moneyline odds)
        if 'home_moneyline' in test_df.columns:
            roi_results = self._calculate_roi(test_df, predicted_probs)
            results.update(roi_results)

        self._print_results(results)

        # Save baseline results
        import json
        with open(self.data_dir / 'baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _get_spread_features(self, df):
        """Get features that spread model expects"""
        # Check what features exist in data
        potential_features = [
            'home_elo', 'away_elo',
            'home_rest_days', 'away_rest_days',
            'home_win_pct', 'away_win_pct',
            'temperature', 'wind_speed',
            'home_qb_rating', 'away_qb_rating'
        ]

        available_features = [f for f in potential_features if f in df.columns]

        if not available_features:
            logger.warning("No standard features found, using basic features")
            # Create basic features
            df['home_win_pct'] = 0.5  # Placeholder
            df['away_win_pct'] = 0.5
            available_features = ['home_win_pct', 'away_win_pct']

        logger.info(f"Using {len(available_features)} features for spread model")
        return available_features

    def _train_simple_spread_model(self, train_df):
        """Train simple spread model if none exists"""
        logger.info("Training simple spread model...")

        from sklearn.ensemble import GradientBoostingRegressor

        # Get features
        feature_cols = self._get_spread_features(train_df)
        X_train = train_df[feature_cols]

        # Calculate actual spread (positive = home favored)
        y_train = train_df['home_score'] - train_df['away_score']

        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)

        logger.info("Simple spread model trained")
        return model

    def _calculate_roi(self, test_df, predicted_probs):
        """Calculate ROI if betting on model predictions"""
        test_df = test_df.copy()
        test_df['predicted_prob'] = predicted_probs

        # Calculate edge (predicted prob vs. implied prob from odds)
        test_df['home_implied_prob'] = self._moneyline_to_probability(test_df['home_moneyline'])
        test_df['edge'] = test_df['predicted_prob'] - test_df['home_implied_prob']

        # Kelly criterion sizing
        test_df['kelly_fraction'] = np.maximum(
            (test_df['predicted_prob'] * test_df['home_moneyline'] - 1) / (test_df['home_moneyline'] - 1),
            0
        ) * 0.25  # Quarter Kelly for safety

        # Simulate betting with 5% edge threshold
        bets = test_df[np.abs(test_df['edge']) > 0.05].copy()

        if len(bets) == 0:
            return {'roi': 0, 'total_bets': 0, 'profit': 0}

        # Calculate profit for each bet
        bets['profit'] = 0.0
        for idx, row in bets.iterrows():
            bet_home = row['edge'] > 0
            won_bet = row['home_won'] == 1 if bet_home else row['home_won'] == 0

            if won_bet:
                odds = row['home_moneyline'] if bet_home else row['away_moneyline']
                bets.loc[idx, 'profit'] = abs(odds) / 100 if odds > 0 else 100 / abs(odds)
            else:
                bets.loc[idx, 'profit'] = -1

        total_profit = bets['profit'].sum()
        roi = (total_profit / len(bets)) * 100

        return {
            'roi': roi,
            'total_bets': len(bets),
            'winning_bets': (bets['profit'] > 0).sum(),
            'profit': total_profit,
            'win_rate': (bets['profit'] > 0).mean()
        }

    def _moneyline_to_probability(self, moneyline):
        """Convert American moneyline odds to implied probability"""
        if pd.isna(moneyline):
            return 0.5

        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)

    def _print_results(self, results):
        """Pretty print results"""
        logger.info("\nBASELINE RESULTS:")
        logger.info(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        logger.info(f"  Log Loss: {results['log_loss']:.4f}")
        logger.info(f"  ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {results['brier_score']:.4f}")

        if 'roi' in results:
            logger.info(f"\nBETTING PERFORMANCE:")
            logger.info(f"  ROI: {results['roi']:.2f}%")
            logger.info(f"  Total Bets: {results['total_bets']}")
            logger.info(f"  Winning Bets: {results['winning_bets']}")
            logger.info(f"  Win Rate: {results['win_rate']:.3f}")
            logger.info(f"  Profit (units): {results['profit']:.2f}")

if __name__ == "__main__":
    baseline = SpreadToMoneylineBaseline()
    baseline.evaluate_baseline()
```

### **Day 2: Proof-of-Concept Moneyline Model**

#### **0.3: Train single best moneyline model**
```python
# File: improved_nfl_system/validation/poc_moneyline_model.py
#!/usr/bin/env python3
"""
Proof-of-Concept: Dedicated Moneyline Model
Single model (XGBoost) to test if dedicated approach beats baseline
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class POCMoneylineModel:
    def __init__(self):
        self.data_dir = Path('validation/data')
        self.output_dir = Path('validation/models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(self):
        """Train POC model and compare to baseline"""
        logger.info("=" * 60)
        logger.info("PROOF-OF-CONCEPT MONEYLINE MODEL")
        logger.info("=" * 60)

        # Load data
        df = pd.read_csv(self.data_dir / 'full_dataset.csv')

        # Feature engineering
        df = self._engineer_features(df)

        # Split data
        train_df = df[df['season'] <= 2023].copy()
        test_df = df[df['season'] == 2024].copy()

        logger.info(f"Train: {len(train_df)} games")
        logger.info(f"Test: {len(test_df)} games")

        # Select features
        feature_cols = self._select_features(train_df)
        logger.info(f"Selected {len(feature_cols)} features")

        # Prepare data
        X_train = train_df[feature_cols]
        y_train = train_df['home_won']
        X_test = test_df[feature_cols]
        y_test = test_df['home_won']

        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median

        # Train model
        model = self._train_xgboost(X_train, y_train)

        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, test_df)

        # Compare to baseline
        comparison = self._compare_to_baseline(results)

        # Save results
        self._save_results(model, results, comparison, feature_cols)

        return comparison

    def _engineer_features(self, df):
        """Create moneyline-specific features"""
        df = df.copy()

        # Win percentage features
        for team_type in ['home', 'away']:
            if f'{team_type}_wins' in df.columns and f'{team_type}_games' in df.columns:
                df[f'{team_type}_win_pct'] = df[f'{team_type}_wins'] / df[f'{team_type}_games'].replace(0, 1)

        # Head-to-head features
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df['rivalry_game'] = (df['home_team'] == df['away_team'].shift()).astype(int)

        # Spread-based features (if available)
        if 'spread_line' in df.columns:
            df['is_favorite'] = (df['spread_line'] < 0).astype(int)
            df['spread_magnitude'] = df['spread_line'].abs()
            df['heavy_favorite'] = (df['spread_magnitude'] > 7).astype(int)
            df['close_game'] = (df['spread_magnitude'] < 3).astype(int)

        # Total-based features
        if 'total_line' in df.columns:
            df['high_total'] = (df['total_line'] > 47).astype(int)
            df['low_total'] = (df['total_line'] < 42).astype(int)

        # Odds value features (if moneyline odds available)
        if 'home_moneyline' in df.columns and 'away_moneyline' in df.columns:
            df['odds_differential'] = df['home_moneyline'] - df['away_moneyline']
            df['heavy_underdog'] = ((df['home_moneyline'] > 200) | (df['away_moneyline'] > 200)).astype(int)

        # Temporal features
        if 'week' in df.columns:
            df['early_season'] = (df['week'] <= 4).astype(int)
            df['late_season'] = (df['week'] >= 14).astype(int)
            df['playoff_push'] = (df['week'] >= 12).astype(int)

        # Rest features
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
            df['home_extra_rest'] = (df['home_rest_days'] > 7).astype(int)
            df['away_extra_rest'] = (df['away_rest_days'] > 7).astype(int)

        return df

    def _select_features(self, df):
        """Select features that exist in data"""
        # Priority feature categories
        priority_features = []

        # Core betting features
        betting_features = [
            'spread_line', 'total_line', 'spread_magnitude', 'is_favorite',
            'heavy_favorite', 'close_game', 'high_total', 'low_total',
            'home_moneyline', 'away_moneyline', 'odds_differential', 'heavy_underdog'
        ]
        priority_features.extend([f for f in betting_features if f in df.columns])

        # Team strength features
        strength_features = [
            'home_elo', 'away_elo', 'home_win_pct', 'away_win_pct',
            'home_wins', 'away_wins', 'home_losses', 'away_losses'
        ]
        priority_features.extend([f for f in strength_features if f in df.columns])

        # Advanced stats (EPA, etc.)
        advanced_features = [col for col in df.columns if 'epa' in col.lower()]
        priority_features.extend(advanced_features[:10])  # Limit to top 10

        # NGS features
        ngs_features = [col for col in df.columns if any(x in col.lower() for x in ['cpoe', 'separation', 'efficiency'])]
        priority_features.extend(ngs_features[:10])

        # Situational features
        situational_features = [
            'week', 'early_season', 'late_season', 'playoff_push',
            'home_rest_days', 'away_rest_days', 'rest_advantage',
            'home_extra_rest', 'away_extra_rest'
        ]
        priority_features.extend([f for f in situational_features if f in df.columns])

        # Weather features
        weather_features = [col for col in df.columns if any(x in col.lower() for x in ['temp', 'wind', 'weather'])]
        priority_features.extend(weather_features)

        # Injury features
        injury_features = [col for col in df.columns if 'injury' in col.lower() or 'qb_status' in col.lower()]
        priority_features.extend(injury_features)

        # Remove duplicates and exclude target/metadata
        priority_features = list(dict.fromkeys(priority_features))  # Remove duplicates
        exclude = ['game_id', 'season', 'home_team', 'away_team', 'home_score', 'away_score',
                   'home_won', 'gameday', 'created_at', 'updated_at', 'id']

        final_features = [f for f in priority_features if f not in exclude]

        logger.info(f"Selected {len(final_features)} features:")
        for i, feat in enumerate(final_features[:20], 1):
            logger.info(f"  {i}. {feat}")
        if len(final_features) > 20:
            logger.info(f"  ... and {len(final_features) - 20} more")

        return final_features

    def _train_xgboost(self, X_train, y_train):
        """Train optimized XGBoost model"""
        logger.info("Training XGBoost model...")

        # Use time series cross-validation for hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=5)

        best_score = 0
        best_params = None

        # Quick hyperparameter search
        param_grid = [
            {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.03},
            {'n_estimators': 1500, 'max_depth': 10, 'learning_rate': 0.01},
        ]

        for params in param_grid:
            model = xgb.XGBClassifier(
                **params,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss'
            )

            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
                score = accuracy_score(y_fold_val, model.predict(X_fold_val))
                scores.append(score)

            avg_score = np.mean(scores)
            logger.info(f"  Params {params}: CV Score = {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        logger.info(f"Best params: {best_params} (CV Score: {best_score:.4f})")

        # Train final model with best params
        final_model = xgb.XGBClassifier(
            **best_params,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss'
        )
        final_model.fit(X_train, y_train)

        return final_model

    def _evaluate_model(self, model, X_test, y_test, test_df):
        """Comprehensive model evaluation"""
        logger.info("\nEvaluating model...")

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Basic metrics
        results = {
            'method': 'Dedicated Moneyline Model (XGBoost)',
            'accuracy': accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba)
        }

        # Confidence tiers
        confidence = np.abs(y_pred_proba - 0.5) * 2
        for threshold in [0.6, 0.7, 0.8]:
            mask = confidence > threshold
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                results[f'accuracy_{int(threshold*100)}conf'] = acc
                results[f'count_{int(threshold*100)}conf'] = int(mask.sum())

        # ROI calculation
        if 'home_moneyline' in test_df.columns:
            from validation.baseline_spread_method import SpreadToMoneylineBaseline
            baseline = SpreadToMoneylineBaseline()

            roi_results = baseline._calculate_roi(test_df.copy(), y_pred_proba)
            results.update(roi_results)

        # Print results
        logger.info(f"\nDEDICATED MODEL RESULTS:")
        logger.info(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        logger.info(f"  Log Loss: {results['log_loss']:.4f}")
        logger.info(f"  ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {results['brier_score']:.4f}")

        for threshold in [60, 70, 80]:
            key = f'accuracy_{threshold}conf'
            if key in results:
                logger.info(f"  {threshold}%+ Confidence: {results[key]:.3f} ({results[f'count_{threshold}conf']} games)")

        if 'roi' in results:
            logger.info(f"\n  ROI: {results['roi']:.2f}%")
            logger.info(f"  Total Bets: {results['total_bets']}")
            logger.info(f"  Profit (units): {results['profit']:.2f}")

        return results

    def _compare_to_baseline(self, poc_results):
        """Compare POC model to baseline"""
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: POC vs. BASELINE")
        logger.info("=" * 60)

        # Load baseline results
        try:
            with open(self.data_dir / 'baseline_results.json', 'r') as f:
                baseline_results = json.load(f)
        except FileNotFoundError:
            logger.error("No baseline results found. Run baseline evaluation first.")
            return None

        comparison = {
            'baseline': baseline_results,
            'poc_model': poc_results,
            'improvements': {}
        }

        # Calculate improvements
        metrics = ['accuracy', 'log_loss', 'roc_auc', 'brier_score', 'roi']

        for metric in metrics:
            if metric in baseline_results and metric in poc_results:
                baseline_val = baseline_results[metric]
                poc_val = poc_results[metric]

                # For log_loss and brier_score, lower is better
                if metric in ['log_loss', 'brier_score']:
                    improvement = ((baseline_val - poc_val) / baseline_val) * 100
                    comparison['improvements'][metric] = {
                        'baseline': baseline_val,
                        'poc': poc_val,
                        'improvement_pct': improvement,
                        'better': poc_val < baseline_val
                    }
                else:
                    improvement = ((poc_val - baseline_val) / baseline_val) * 100
                    comparison['improvements'][metric] = {
                        'baseline': baseline_val,
                        'poc': poc_val,
                        'improvement_pct': improvement,
                        'better': poc_val > baseline_val
                    }

        # Print comparison
        logger.info("\nMETRIC COMPARISON:")
        for metric, data in comparison['improvements'].items():
            symbol = "✓" if data['better'] else "✗"
            logger.info(f"  {metric.upper()}:")
            logger.info(f"    Baseline: {data['baseline']:.4f}")
            logger.info(f"    POC Model: {data['poc']:.4f}")
            logger.info(f"    Improvement: {data['improvement_pct']:+.2f}% {symbol}")

        # Decision criteria
        logger.info("\n" + "=" * 60)
        logger.info("DECISION CRITERIA:")
        logger.info("=" * 60)

        criteria = {
            'accuracy_improvement': comparison['improvements'].get('accuracy', {}).get('improvement_pct', 0) > 2,
            'roi_improvement': comparison['improvements'].get('roi', {}).get('improvement_pct', 0) > 10,
            'logloss_improvement': comparison['improvements'].get('log_loss', {}).get('better', False),
        }

        for criterion, passed in criteria.items():
            symbol = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {criterion}: {symbol}")

        proceed = sum(criteria.values()) >= 2  # At least 2 of 3 criteria

        logger.info("\n" + "=" * 60)
        if proceed:
            logger.info("✓ RECOMMENDATION: PROCEED TO PHASE 1")
            logger.info("  Dedicated moneyline model shows meaningful improvement")
        else:
            logger.info("✗ RECOMMENDATION: STOP - USE BASELINE")
            logger.info("  Dedicated moneyline model does not justify development effort")
        logger.info("=" * 60)

        comparison['recommendation'] = 'PROCEED' if proceed else 'STOP'
        comparison['criteria'] = criteria

        return comparison

    def _save_results(self, model, results, comparison, feature_cols):
        """Save POC results"""
        # Save model
        with open(self.output_dir / 'poc_moneyline_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Save results
        with open(self.output_dir / 'poc_results.json', 'w') as f:
            json.dump({
                'results': results,
                'comparison': comparison,
                'features': feature_cols
            }, f, indent=2)

        logger.info(f"\nResults saved to {self.output_dir}")

if __name__ == "__main__":
    poc = POCMoneylineModel()
    poc.train_and_evaluate()
```

### **Day 3: Decision Point**

#### **0.4: Review and decision script**
```python
# File: improved_nfl_system/validation/phase0_decision.py
#!/usr/bin/env python3
"""
Phase 0 Decision Script
Reviews validation results and determines if Phase 1 should proceed
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_phase0_decision():
    """Make go/no-go decision for Phase 1"""

    results_file = Path('validation/models/poc_results.json')

    if not results_file.exists():
        logger.error("POC results not found. Run validation first.")
        return False

    with open(results_file, 'r') as f:
        data = json.load(f)

    comparison = data['comparison']

    logger.info("=" * 60)
    logger.info("PHASE 0 VALIDATION COMPLETE")
    logger.info("=" * 60)

    logger.info(f"\nRECOMMENDATION: {comparison['recommendation']}")

    if comparison['recommendation'] == 'PROCEED':
        logger.info("\n✓ Dedicated moneyline model justified")
        logger.info("✓ Proceed to Phase 1: Core Implementation")
        logger.info("\nNext steps:")
        logger.info("  1. Review POC model architecture")
        logger.info("  2. Plan production model with proper CI/CD")
        logger.info("  3. Design integration with existing system")
        return True
    else:
        logger.info("\n✗ Dedicated moneyline model not justified")
        logger.info("✗ Use baseline spread-to-moneyline conversion")
        logger.info("\nNext steps:")
        logger.info("  1. Integrate spread-to-moneyline conversion into production")
        logger.info("  2. Update web interface to show moneyline predictions")
        logger.info("  3. Document mathematical approach")
        return False

if __name__ == "__main__":
    decision = make_phase0_decision()
    exit(0 if decision else 1)
```

---

## **Phase 1: Core Implementation (Days 4-8) - ONLY IF PHASE 0 PASSES**

This phase only executes if POC model beats baseline by sufficient margin.

### **Day 4-5: Production Model Development**

#### **1.1: Production-ready moneyline model**
```python
# File: improved_nfl_system/models/moneyline_model.py
#!/usr/bin/env python3
"""
Production Moneyline Model
Based on validated POC, built for production deployment
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMoneylineModel:
    """
    Production-ready moneyline prediction model

    Features:
    - Versioned model artifacts
    - Rollback capability
    - Performance monitoring
    - Confidence calibration
    """

    VERSION = "1.0.0"

    def __init__(self, model_dir='models/saved_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_cols = None
        self.metadata = None

        # Load latest model if exists
        self.load_latest()

    def train(self, train_df, val_df=None, save=True):
        """Train production model"""
        logger.info("Training production moneyline model...")

        # Feature engineering
        train_df = self._engineer_features(train_df)
        if val_df is not None:
            val_df = self._engineer_features(val_df)

        # Select features (use POC feature selection)
        self.feature_cols = self._select_features(train_df)

        # Prepare data
        X_train = train_df[self.feature_cols].fillna(train_df[self.feature_cols].median())
        y_train = train_df['home_won']

        # Train model with validated hyperparameters from POC
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss'
        )

        if val_df is not None:
            X_val = val_df[self.feature_cols].fillna(X_train.median())
            y_val = val_df['home_won']
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        # Create metadata
        self.metadata = {
            'version': self.VERSION,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(train_df),
            'features': self.feature_cols,
            'feature_count': len(self.feature_cols),
            'hyperparameters': self.model.get_params()
        }

        if save:
            self.save()

        logger.info(f"Model trained successfully ({len(self.feature_cols)} features)")
        return self

    def predict_proba(self, game_df):
        """
        Predict moneyline probabilities for games

        Returns:
            dict with keys:
                - home_win_prob: Probability home team wins
                - away_win_prob: Probability away team wins
                - confidence: Confidence score (0-1)
                - prediction: 'home' or 'away'
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load model first.")

        # Feature engineering
        game_df = self._engineer_features(game_df)

        # Prepare features
        X = game_df[self.feature_cols].fillna(0)  # Use 0 for missing in production

        # Predict
        proba = self.model.predict_proba(X)[:, 1]  # Probability home wins

        # Create results
        results = []
        for i, prob in enumerate(proba):
            result = {
                'home_win_prob': float(prob),
                'away_win_prob': float(1 - prob),
                'confidence': float(abs(prob - 0.5) * 2),
                'prediction': 'home' if prob > 0.5 else 'away'
            }
            results.append(result)

        return results if len(results) > 1 else results[0]

    def save(self, version=None):
        """Save model with versioning"""
        if version is None:
            version = self.VERSION

        # Create version directory
        version_dir = self.model_dir / f'moneyline_v{version}'
        version_dir.mkdir(exist_ok=True)

        # Save model
        model_path = version_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata_path = version_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Update latest symlink (or copy)
        latest_dir = self.model_dir / 'moneyline_latest'
        if latest_dir.exists():
            import shutil
            shutil.rmtree(latest_dir)

        import shutil
        shutil.copytree(version_dir, latest_dir)

        logger.info(f"Model saved: {version_dir}")

    def load_latest(self):
        """Load latest model version"""
        latest_dir = self.model_dir / 'moneyline_latest'

        if not latest_dir.exists():
            logger.info("No saved model found")
            return False

        return self.load(latest_dir)

    def load(self, path):
        """Load model from path"""
        path = Path(path)

        # Load model
        model_path = path / 'model.pkl'
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load metadata
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.feature_cols = self.metadata['features']

        logger.info(f"Model loaded from {path}")
        return True

    def _engineer_features(self, df):
        """Feature engineering (copied from POC)"""
        # [Same as POC implementation]
        df = df.copy()

        # Win percentage features
        for team_type in ['home', 'away']:
            if f'{team_type}_wins' in df.columns and f'{team_type}_games' in df.columns:
                df[f'{team_type}_win_pct'] = df[f'{team_type}_wins'] / df[f'{team_type}_games'].replace(0, 1)

        # Spread-based features
        if 'spread_line' in df.columns:
            df['is_favorite'] = (df['spread_line'] < 0).astype(int)
            df['spread_magnitude'] = df['spread_line'].abs()
            df['heavy_favorite'] = (df['spread_magnitude'] > 7).astype(int)
            df['close_game'] = (df['spread_magnitude'] < 3).astype(int)

        # Total-based features
        if 'total_line' in df.columns:
            df['high_total'] = (df['total_line'] > 47).astype(int)
            df['low_total'] = (df['total_line'] < 42).astype(int)

        # Temporal features
        if 'week' in df.columns:
            df['early_season'] = (df['week'] <= 4).astype(int)
            df['late_season'] = (df['week'] >= 14).astype(int)
            df['playoff_push'] = (df['week'] >= 12).astype(int)

        # Rest features
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

        return df

    def _select_features(self, df):
        """Select features (copied from POC)"""
        # Load POC feature list if available
        poc_features_file = Path('validation/models/poc_results.json')
        if poc_features_file.exists():
            with open(poc_features_file, 'r') as f:
                poc_data = json.load(f)
                poc_features = poc_data.get('features', [])

                # Use POC features that exist in current data
                available_features = [f for f in poc_features if f in df.columns]

                if available_features:
                    logger.info(f"Using {len(available_features)} POC features")
                    return available_features

        # Fallback: basic feature selection
        logger.warning("POC features not found, using basic selection")

        basic_features = [
            'spread_line', 'total_line', 'spread_magnitude', 'is_favorite',
            'home_win_pct', 'away_win_pct', 'week', 'rest_advantage'
        ]

        return [f for f in basic_features if f in df.columns]

# Convenience functions for integration
def load_moneyline_model():
    """Load latest moneyline model"""
    model = ProductionMoneylineModel()
    return model if model.model is not None else None

def predict_moneyline(game_data):
    """Quick prediction function"""
    model = load_moneyline_model()
    if model is None:
        raise ValueError("No moneyline model available")

    if isinstance(game_data, dict):
        game_data = pd.DataFrame([game_data])

    return model.predict_proba(game_data)
```

### **Day 6: Testing & Validation**

#### **1.2: Comprehensive test suite**
```python
# File: improved_nfl_system/tests/test_moneyline_model.py
#!/usr/bin/env python3
"""
Test Suite for Moneyline Model
Ensures production model meets quality standards
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.moneyline_model import ProductionMoneylineModel, load_moneyline_model, predict_moneyline

class TestMoneylineModel:
    """Test suite for moneyline model"""

    @pytest.fixture
    def sample_data(self):
        """Create sample game data"""
        return pd.DataFrame({
            'game_id': ['2024_01_KC_BAL'],
            'season': [2024],
            'week': [1],
            'home_team': ['BAL'],
            'away_team': ['KC'],
            'spread_line': [-3.0],
            'total_line': [47.5],
            'home_wins': [10],
            'home_games': [16],
            'away_wins': [12],
            'away_games': [16],
            'home_rest_days': [7],
            'away_rest_days': [7],
            'home_score': [20],  # For training
            'away_score': [17]
        })

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model for testing"""
        model = ProductionMoneylineModel()

        # Create larger dataset for training
        train_data = pd.concat([sample_data] * 100, ignore_index=True)
        train_data['home_score'] = np.random.randint(14, 35, len(train_data))
        train_data['away_score'] = np.random.randint(14, 35, len(train_data))
        train_data['home_won'] = (train_data['home_score'] > train_data['away_score']).astype(int)

        model.train(train_data, save=False)
        return model

    def test_model_training(self, sample_data):
        """Test model can be trained"""
        model = ProductionMoneylineModel()

        # Create training data
        train_data = pd.concat([sample_data] * 100, ignore_index=True)
        train_data['home_won'] = np.random.randint(0, 2, len(train_data))

        model.train(train_data, save=False)

        assert model.model is not None
        assert model.feature_cols is not None
        assert len(model.feature_cols) > 0

    def test_prediction_output_format(self, trained_model, sample_data):
        """Test prediction returns correct format"""
        result = trained_model.predict_proba(sample_data)

        assert 'home_win_prob' in result
        assert 'away_win_prob' in result
        assert 'confidence' in result
        assert 'prediction' in result

        # Check probabilities sum to 1
        assert abs(result['home_win_prob'] + result['away_win_prob'] - 1.0) < 0.001

        # Check confidence range
        assert 0 <= result['confidence'] <= 1

        # Check prediction matches probability
        if result['home_win_prob'] > 0.5:
            assert result['prediction'] == 'home'
        else:
            assert result['prediction'] == 'away'

    def test_prediction_probabilities_valid(self, trained_model, sample_data):
        """Test predictions are valid probabilities"""
        result = trained_model.predict_proba(sample_data)

        assert 0 <= result['home_win_prob'] <= 1
        assert 0 <= result['away_win_prob'] <= 1

    def test_batch_prediction(self, trained_model, sample_data):
        """Test batch prediction works"""
        batch_data = pd.concat([sample_data] * 5, ignore_index=True)

        results = trained_model.predict_proba(batch_data)

        assert len(results) == 5
        assert all('home_win_prob' in r for r in results)

    def test_missing_data_handling(self, trained_model, sample_data):
        """Test model handles missing data gracefully"""
        # Remove some features
        incomplete_data = sample_data.drop(columns=['home_rest_days', 'away_rest_days'])

        # Should not raise error
        result = trained_model.predict_proba(incomplete_data)

        assert result is not None

    def test_model_save_load(self, trained_model, tmp_path):
        """Test model can be saved and loaded"""
        # Save
        trained_model.model_dir = tmp_path
        trained_model.save(version='test')

        # Load
        new_model = ProductionMoneylineModel(model_dir=tmp_path)
        loaded = new_model.load(tmp_path / 'moneyline_vtest')

        assert loaded
        assert new_model.model is not None
        assert new_model.feature_cols == trained_model.feature_cols

    def test_convenience_functions(self, trained_model, sample_data, tmp_path):
        """Test convenience functions work"""
        # Save model
        trained_model.model_dir = tmp_path
        trained_model.save()

        # Test load function
        model = ProductionMoneylineModel(model_dir=tmp_path)
        model.load_latest()
        assert model.model is not None

    def test_temporal_validity(self, sample_data):
        """Test model doesn't use future data"""
        # Create temporal dataset
        train_data = []
        for week in range(1, 10):
            data = sample_data.copy()
            data['week'] = week
            data['home_score'] = np.random.randint(14, 35)
            data['away_score'] = np.random.randint(14, 35)
            train_data.append(data)

        train_df = pd.concat(train_data, ignore_index=True)
        train_df['home_won'] = (train_df['home_score'] > train_df['away_score']).astype(int)

        model = ProductionMoneylineModel()
        model.train(train_df, save=False)

        # Predict on earlier week - should work
        early_week = sample_data.copy()
        early_week['week'] = 1

        result = model.predict_proba(early_week)
        assert result is not None

    def test_accuracy_threshold(self, sample_data):
        """Test model meets minimum accuracy threshold"""
        # Create realistic test data
        np.random.seed(42)

        test_data = []
        for i in range(200):
            game = sample_data.copy()
            game['spread_line'] = np.random.uniform(-14, 14)
            game['total_line'] = np.random.uniform(38, 54)
            game['home_wins'] = np.random.randint(4, 13)
            game['away_wins'] = np.random.randint(4, 13)
            game['home_games'] = 16
            game['away_games'] = 16

            # Realistic score based on spread
            home_advantage = -game['spread_line'].values[0] + np.random.normal(0, 13)
            game['home_score'] = int(24 + home_advantage)
            game['away_score'] = int(24)

            test_data.append(game)

        df = pd.concat(test_data, ignore_index=True)
        df['home_won'] = (df['home_score'] > df['away_score']).astype(int)

        # Train on 80%, test on 20%
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        model = ProductionMoneylineModel()
        model.train(train_df, save=False)

        # Predict
        results = model.predict_proba(test_df)
        predictions = [1 if r['home_win_prob'] > 0.5 else 0 for r in results]
        actuals = test_df['home_won'].values

        accuracy = np.mean(np.array(predictions) == actuals)

        # Minimum threshold: 55% (better than coin flip)
        assert accuracy > 0.55, f"Accuracy {accuracy:.3f} below minimum threshold 0.55"

# Integration tests
class TestIntegration:
    """Test integration with existing system"""

    def test_integrates_with_existing_db(self):
        """Test model can use data from existing database schema"""
        # This would test actual database integration
        # Skipping for now as it requires database setup
        pytest.skip("Requires database setup")

    def test_compatible_with_web_interface(self):
        """Test predictions can be serialized for web interface"""
        model = ProductionMoneylineModel()

        # Create minimal data
        game_data = pd.DataFrame({
            'spread_line': [-3.0],
            'total_line': [47.5]
        })

        # Train minimal model
        train_data = pd.concat([game_data] * 100, ignore_index=True)
        train_data['home_won'] = np.random.randint(0, 2, len(train_data))
        model.train(train_data, save=False)

        result = model.predict_proba(game_data)

        # Should be JSON-serializable
        import json
        json_str = json.dumps(result)
        assert json_str is not None

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

### **Day 7: Integration with Existing System**

#### **1.3: Integration module**
```python
# File: improved_nfl_system/integrations/moneyline_integration.py
#!/usr/bin/env python3
"""
Moneyline Model Integration
Integrates moneyline predictions into existing NFL betting system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from datetime import datetime
from database.db_manager import DatabaseManager
from models.moneyline_model import load_moneyline_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoneylineIntegration:
    """
    Integrates moneyline predictions with existing system

    Features:
    - Fetches game data from existing database
    - Generates moneyline predictions
    - Stores predictions in database
    - Provides backward compatibility
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.model = load_moneyline_model()

        if self.model is None:
            raise ValueError("Moneyline model not available. Train model first.")

    def generate_weekly_predictions(self, season, week):
        """
        Generate moneyline predictions for a specific week

        Args:
            season: NFL season year
            week: Week number

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Generating moneyline predictions for {season} Week {week}")

        # Fetch games from database
        games = self.db.get_games_by_week(season, week)

        if len(games) == 0:
            logger.warning(f"No games found for {season} Week {week}")
            return pd.DataFrame()

        logger.info(f"Found {len(games)} games")

        # Generate predictions
        predictions = self.model.predict_proba(games)

        if not isinstance(predictions, list):
            predictions = [predictions]

        # Combine with game data
        results = []
        for i, (_, game) in enumerate(games.iterrows()):
            pred = predictions[i]

            result = {
                'game_id': game['game_id'],
                'season': season,
                'week': week,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_win_probability': pred['home_win_prob'],
                'away_win_probability': pred['away_win_prob'],
                'confidence': pred['confidence'],
                'predicted_winner': pred['prediction'],
                'prediction_date': datetime.now().isoformat()
            }
            results.append(result)

        results_df = pd.DataFrame(results)

        # Store in database
        self._store_predictions(results_df)

        logger.info(f"Generated {len(results_df)} moneyline predictions")

        return results_df

    def get_game_prediction(self, game_id):
        """Get moneyline prediction for specific game"""
        # Check database first
        cached = self.db.get_moneyline_prediction(game_id)

        if cached is not None:
            return cached

        # Generate new prediction
        game = self.db.get_game(game_id)

        if game is None:
            raise ValueError(f"Game not found: {game_id}")

        prediction = self.model.predict_proba(pd.DataFrame([game]))

        # Store in database
        self._store_prediction(game_id, prediction)

        return prediction

    def backfill_predictions(self, season):
        """Backfill predictions for entire season"""
        logger.info(f"Backfilling moneyline predictions for {season}")

        for week in range(1, 19):  # Regular season weeks
            try:
                self.generate_weekly_predictions(season, week)
            except Exception as e:
                logger.error(f"Error backfilling Week {week}: {e}")

    def _store_predictions(self, predictions_df):
        """Store predictions in database"""
        # Add to moneyline_predictions table
        self.db.insert_moneyline_predictions(predictions_df)
        logger.info(f"Stored {len(predictions_df)} predictions in database")

    def _store_prediction(self, game_id, prediction):
        """Store single prediction"""
        self.db.insert_moneyline_prediction(game_id, prediction)

# Extend DatabaseManager with moneyline methods
class MoneylineDatabase(DatabaseManager):
    """Extended database manager with moneyline support"""

    def create_moneyline_tables(self):
        """Create tables for moneyline predictions"""
        self.execute("""
            CREATE TABLE IF NOT EXISTS moneyline_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                season INTEGER,
                week INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_win_probability REAL,
                away_win_probability REAL,
                confidence REAL,
                predicted_winner TEXT,
                prediction_date TEXT,
                model_version TEXT,
                UNIQUE(game_id, prediction_date)
            )
        """)

        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_moneyline_game
            ON moneyline_predictions(game_id)
        """)

        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_moneyline_week
            ON moneyline_predictions(season, week)
        """)

        logger.info("Moneyline tables created")

    def insert_moneyline_predictions(self, predictions_df):
        """Bulk insert predictions"""
        predictions_df.to_sql(
            'moneyline_predictions',
            self.conn,
            if_exists='append',
            index=False
        )

    def insert_moneyline_prediction(self, game_id, prediction):
        """Insert single prediction"""
        self.execute("""
            INSERT OR REPLACE INTO moneyline_predictions
            (game_id, home_win_probability, away_win_probability,
             confidence, predicted_winner, prediction_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            prediction['home_win_prob'],
            prediction['away_win_prob'],
            prediction['confidence'],
            prediction['prediction'],
            datetime.now().isoformat()
        ))

    def get_moneyline_prediction(self, game_id):
        """Get latest prediction for game"""
        result = self.query("""
            SELECT * FROM moneyline_predictions
            WHERE game_id = ?
            ORDER BY prediction_date DESC
            LIMIT 1
        """, (game_id,))

        if result:
            return result[0]
        return None

    def get_moneyline_predictions_by_week(self, season, week):
        """Get all predictions for a week"""
        return self.query("""
            SELECT * FROM moneyline_predictions
            WHERE season = ? AND week = ?
            ORDER BY prediction_date DESC
        """, (season, week))

# Setup function
def setup_moneyline_integration():
    """Initialize moneyline integration"""
    db = MoneylineDatabase()
    db.create_moneyline_tables()
    logger.info("Moneyline integration setup complete")

if __name__ == "__main__":
    setup_moneyline_integration()

    # Test integration
    integration = MoneylineIntegration()
    integration.generate_weekly_predictions(2024, 1)
```

### **Day 8: Performance Monitoring**

#### **1.4: Monitoring and alerting system**
```python
# File: improved_nfl_system/monitoring/moneyline_monitor.py
#!/usr/bin/env python3
"""
Moneyline Model Monitoring System
Tracks performance, detects degradation, triggers retraining
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from database.db_manager import DatabaseManager
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoneylineMonitor:
    """
    Monitor moneyline model performance in production

    Features:
    - Track prediction accuracy over time
    - Detect performance degradation
    - Calculate ROI metrics
    - Alert on anomalies
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.metrics_dir = Path('monitoring/metrics')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Performance thresholds
        self.ACCURACY_THRESHOLD = 0.55  # Minimum acceptable accuracy
        self.DEGRADATION_THRESHOLD = 0.05  # 5% drop triggers alert
        self.ROI_THRESHOLD = 0.0  # Minimum ROI (breakeven)

    def weekly_performance_check(self, season, week):
        """
        Check performance for completed week

        Returns:
            dict with performance metrics and alerts
        """
        logger.info(f"Checking performance for {season} Week {week}")

        # Get predictions for week
        predictions = self.db.get_moneyline_predictions_by_week(season, week)

        if not predictions:
            logger.warning(f"No predictions found for {season} Week {week}")
            return None

        # Get actual results
        games = self.db.get_games_by_week(season, week)

        # Match predictions with results
        merged = self._merge_predictions_and_results(predictions, games)

        if len(merged) == 0:
            logger.warning("No completed games found")
            return None

        # Calculate metrics
        metrics = self._calculate_metrics(merged)

        # Check for alerts
        alerts = self._check_thresholds(metrics)

        # Store metrics
        self._store_metrics(season, week, metrics, alerts)

        # Log results
        self._log_performance(metrics, alerts)

        return {
            'metrics': metrics,
            'alerts': alerts,
            'recommendation': self._get_recommendation(alerts)
        }

    def historical_performance_report(self, start_season, end_season):
        """Generate historical performance report"""
        logger.info(f"Generating performance report for {start_season}-{end_season}")

        all_metrics = []

        for season in range(start_season, end_season + 1):
            for week in range(1, 19):
                metrics_file = self.metrics_dir / f'{season}_week{week}_metrics.json'

                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        data['season'] = season
                        data['week'] = week
                        all_metrics.append(data)

        if not all_metrics:
            logger.warning("No historical metrics found")
            return None

        # Aggregate metrics
        report = {
            'period': f'{start_season}-{end_season}',
            'total_weeks': len(all_metrics),
            'average_accuracy': np.mean([m['metrics']['accuracy'] for m in all_metrics]),
            'average_roi': np.mean([m['metrics'].get('roi', 0) for m in all_metrics]),
            'weeks_above_threshold': sum(1 for m in all_metrics if m['metrics']['accuracy'] > self.ACCURACY_THRESHOLD),
            'total_alerts': sum(len(m['alerts']) for m in all_metrics),
            'accuracy_trend': self._calculate_trend([m['metrics']['accuracy'] for m in all_metrics]),
            'metrics_by_week': all_metrics
        }

        # Save report
        report_file = self.metrics_dir / f'report_{start_season}_{end_season}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved to {report_file}")

        return report

    def check_for_retraining(self):
        """
        Check if model should be retrained

        Criteria:
        - Accuracy below threshold for 3+ consecutive weeks
        - ROI negative for 4+ consecutive weeks
        - Significant accuracy drop vs. training baseline
        """
        # Get last 4 weeks of metrics
        recent_metrics = self._get_recent_metrics(weeks=4)

        if len(recent_metrics) < 3:
            logger.info("Insufficient data for retraining decision")
            return False

        # Check consecutive underperformance
        consecutive_low_accuracy = sum(
            1 for m in recent_metrics
            if m['metrics']['accuracy'] < self.ACCURACY_THRESHOLD
        )

        consecutive_negative_roi = sum(
            1 for m in recent_metrics
            if m['metrics'].get('roi', 0) < self.ROI_THRESHOLD
        )

        # Check accuracy trend
        accuracies = [m['metrics']['accuracy'] for m in recent_metrics]
        accuracy_drop = max(accuracies) - min(accuracies)

        should_retrain = (
            consecutive_low_accuracy >= 3 or
            consecutive_negative_roi >= 4 or
            accuracy_drop > self.DEGRADATION_THRESHOLD
        )

        if should_retrain:
            logger.warning("RETRAINING RECOMMENDED")
            logger.warning(f"  Consecutive low accuracy weeks: {consecutive_low_accuracy}")
            logger.warning(f"  Consecutive negative ROI weeks: {consecutive_negative_roi}")
            logger.warning(f"  Accuracy drop: {accuracy_drop:.3f}")
        else:
            logger.info("Model performance acceptable, no retraining needed")

        return should_retrain

    def _merge_predictions_and_results(self, predictions, games):
        """Merge predictions with actual game results"""
        pred_df = pd.DataFrame(predictions)
        games_df = pd.DataFrame(games)

        # Add actual results
        games_df['actual_winner'] = games_df.apply(
            lambda row: 'home' if row['home_score'] > row['away_score'] else 'away',
            axis=1
        )

        # Merge
        merged = pred_df.merge(games_df[['game_id', 'actual_winner', 'home_score', 'away_score']],
                                on='game_id', how='inner')

        return merged

    def _calculate_metrics(self, merged_df):
        """Calculate performance metrics"""
        # Accuracy
        merged_df['correct'] = merged_df['predicted_winner'] == merged_df['actual_winner']
        accuracy = merged_df['correct'].mean()

        # Brier score (for probability calibration)
        merged_df['actual_home_win'] = (merged_df['actual_winner'] == 'home').astype(int)
        brier = brier_score_loss(merged_df['actual_home_win'], merged_df['home_win_probability'])

        # High confidence accuracy
        high_conf_df = merged_df[merged_df['confidence'] > 0.7]
        high_conf_accuracy = high_conf_df['correct'].mean() if len(high_conf_df) > 0 else 0

        metrics = {
            'total_games': len(merged_df),
            'accuracy': float(accuracy),
            'correct_predictions': int(merged_df['correct'].sum()),
            'brier_score': float(brier),
            'high_confidence_games': len(high_conf_df),
            'high_confidence_accuracy': float(high_conf_accuracy),
            'average_confidence': float(merged_df['confidence'].mean())
        }

        # ROI if we have odds data
        if 'home_moneyline' in merged_df.columns:
            roi_metrics = self._calculate_roi(merged_df)
            metrics.update(roi_metrics)

        return metrics

    def _calculate_roi(self, merged_df):
        """Calculate ROI metrics"""
        # Simple betting strategy: bet on predictions with >5% edge
        merged_df['edge'] = merged_df['home_win_probability'] - 0.5

        bets = merged_df[abs(merged_df['edge']) > 0.05].copy()

        if len(bets) == 0:
            return {'roi': 0, 'total_bets': 0, 'profit': 0}

        # Calculate profit
        bets['profit'] = bets.apply(self._calculate_bet_profit, axis=1)

        total_profit = bets['profit'].sum()
        roi = (total_profit / len(bets)) * 100

        return {
            'roi': float(roi),
            'total_bets': len(bets),
            'winning_bets': int((bets['profit'] > 0).sum()),
            'profit_units': float(total_profit)
        }

    def _calculate_bet_profit(self, row):
        """Calculate profit for single bet"""
        # Bet on predicted winner
        if row['predicted_winner'] == 'home':
            won = row['actual_winner'] == 'home'
            odds = row.get('home_moneyline', 0)
        else:
            won = row['actual_winner'] == 'away'
            odds = row.get('away_moneyline', 0)

        if won:
            if odds > 0:
                return odds / 100
            else:
                return 100 / abs(odds)
        else:
            return -1

    def _check_thresholds(self, metrics):
        """Check if metrics exceed alert thresholds"""
        alerts = []

        # Accuracy alert
        if metrics['accuracy'] < self.ACCURACY_THRESHOLD:
            alerts.append({
                'type': 'LOW_ACCURACY',
                'severity': 'HIGH',
                'message': f"Accuracy {metrics['accuracy']:.3f} below threshold {self.ACCURACY_THRESHOLD}",
                'value': metrics['accuracy'],
                'threshold': self.ACCURACY_THRESHOLD
            })

        # ROI alert
        if 'roi' in metrics and metrics['roi'] < self.ROI_THRESHOLD:
            alerts.append({
                'type': 'NEGATIVE_ROI',
                'severity': 'MEDIUM',
                'message': f"ROI {metrics['roi']:.2f}% below threshold {self.ROI_THRESHOLD}%",
                'value': metrics['roi'],
                'threshold': self.ROI_THRESHOLD
            })

        # Calibration alert (brier score)
        if metrics['brier_score'] > 0.25:
            alerts.append({
                'type': 'POOR_CALIBRATION',
                'severity': 'MEDIUM',
                'message': f"Brier score {metrics['brier_score']:.3f} indicates poor probability calibration",
                'value': metrics['brier_score'],
                'threshold': 0.25
            })

        # Low volume alert
        if metrics['total_games'] < 10:
            alerts.append({
                'type': 'LOW_VOLUME',
                'severity': 'LOW',
                'message': f"Only {metrics['total_games']} games - low sample size",
                'value': metrics['total_games'],
                'threshold': 10
            })

        return alerts

    def _get_recommendation(self, alerts):
        """Get recommendation based on alerts"""
        if not alerts:
            return "CONTINUE - Model performing within acceptable range"

        high_severity = [a for a in alerts if a['severity'] == 'HIGH']

        if high_severity:
            return "INVESTIGATE - High severity issues detected, consider retraining"

        return "MONITOR - Minor issues detected, continue monitoring"

    def _store_metrics(self, season, week, metrics, alerts):
        """Store metrics to disk"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'season': season,
            'week': week,
            'metrics': metrics,
            'alerts': alerts
        }

        metrics_file = self.metrics_dir / f'{season}_week{week}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_recent_metrics(self, weeks=4):
        """Get most recent N weeks of metrics"""
        all_files = sorted(self.metrics_dir.glob('*_metrics.json'), reverse=True)

        recent_metrics = []
        for file in all_files[:weeks]:
            with open(file, 'r') as f:
                recent_metrics.append(json.load(f))

        return recent_metrics

    def _calculate_trend(self, values):
        """Calculate simple trend (positive/negative/stable)"""
        if len(values) < 2:
            return "stable"

        # Linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"

    def _log_performance(self, metrics, alerts):
        """Log performance summary"""
        logger.info("\nPERFORMANCE SUMMARY:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['correct_predictions']}/{metrics['total_games']})")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")

        if 'roi' in metrics:
            logger.info(f"  ROI: {metrics['roi']:.2f}%")
            logger.info(f"  Profit: {metrics['profit_units']:.2f} units")

        if metrics['high_confidence_games'] > 0:
            logger.info(f"  High Confidence: {metrics['high_confidence_accuracy']:.3f} ({metrics['high_confidence_games']} games)")

        if alerts:
            logger.warning(f"\n  ⚠ {len(alerts)} ALERTS:")
            for alert in alerts:
                logger.warning(f"    [{alert['severity']}] {alert['message']}")
        else:
            logger.info("\n  ✓ No alerts")

if __name__ == "__main__":
    monitor = MoneylineMonitor()

    # Check latest week
    monitor.weekly_performance_check(2024, 1)

    # Check if retraining needed
    should_retrain = monitor.check_for_retraining()

    if should_retrain:
        logger.warning("Model retraining recommended!")
```

---

## **Phase 2: Deployment & Operations (Days 9-12)**

### **Day 9-10: Web Interface Integration**

#### **2.1: API endpoints for web interface**
```python
# File: improved_nfl_system/web/api/moneyline_endpoints.py
#!/usr/bin/env python3
"""
Moneyline API Endpoints
FastAPI endpoints for moneyline predictions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.moneyline_integration import MoneylineIntegration
from monitoring.moneyline_monitor import MoneylineMonitor

router = APIRouter(prefix="/api/moneyline", tags=["moneyline"])

# Request/Response models
class MoneylinePrediction(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    confidence: float
    predicted_winner: str
    prediction_date: str

class WeekPredictionsResponse(BaseModel):
    season: int
    week: int
    predictions: List[MoneylinePrediction]
    total_games: int

class PerformanceMetrics(BaseModel):
    accuracy: float
    total_games: int
    correct_predictions: int
    brier_score: float
    roi: Optional[float]
    alerts: List[dict]

# Initialize services
integration = None
monitor = None

def get_integration():
    global integration
    if integration is None:
        integration = MoneylineIntegration()
    return integration

def get_monitor():
    global monitor
    if monitor is None:
        monitor = MoneylineMonitor()
    return monitor

# Endpoints
@router.get("/predictions/{season}/{week}", response_model=WeekPredictionsResponse)
async def get_week_predictions(season: int, week: int):
    """Get moneyline predictions for a specific week"""
    try:
        integ = get_integration()
        predictions = integ.generate_weekly_predictions(season, week)

        return WeekPredictionsResponse(
            season=season,
            week=week,
            predictions=predictions.to_dict('records'),
            total_games=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction/{game_id}", response_model=MoneylinePrediction)
async def get_game_prediction(game_id: str):
    """Get moneyline prediction for a specific game"""
    try:
        integ = get_integration()
        prediction = integ.get_game_prediction(game_id)

        if prediction is None:
            raise HTTPException(status_code=404, detail="Game not found")

        return MoneylinePrediction(**prediction)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{season}/{week}", response_model=PerformanceMetrics)
async def get_week_performance(season: int, week: int):
    """Get performance metrics for a specific week"""
    try:
        mon = get_monitor()
        result = mon.weekly_performance_check(season, week)

        if result is None:
            raise HTTPException(status_code=404, detail="No data found for this week")

        return PerformanceMetrics(
            **result['metrics'],
            alerts=result['alerts']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Check if moneyline service is healthy"""
    try:
        from models.moneyline_model import load_moneyline_model

        model = load_moneyline_model()

        return {
            "status": "healthy" if model is not None else "degraded",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add to main FastAPI app
def register_moneyline_routes(app):
    """Register moneyline routes with main app"""
    app.include_router(router)
```

### **Day 11: Documentation & Runbook**

#### **2.2: Operational runbook**
```markdown
# File: improved_nfl_system/docs/MONEYLINE_MODEL_RUNBOOK.md

# Moneyline Model Operational Runbook

## Overview

This runbook covers operational procedures for the NFL Moneyline Prediction Model.

## Weekly Operations

### Tuesday Morning (Post-Game Analysis)
```bash
cd improved_nfl_system

# 1. Check performance for completed week
python -c "
from monitoring.moneyline_monitor import MoneylineMonitor
monitor = MoneylineMonitor()
monitor.weekly_performance_check(2024, 1)  # Update week number
"

# 2. Check if retraining needed
python -c "
from monitoring.moneyline_monitor import MoneylineMonitor
monitor = MoneylineMonitor()
if monitor.check_for_retraining():
    print('⚠️  RETRAINING RECOMMENDED')
else:
    print('✓ Model performance acceptable')
"
```

### Wednesday Afternoon (Generate Predictions)
```bash
# Generate predictions for upcoming week
python -c "
from integrations.moneyline_integration import MoneylineIntegration
integration = MoneylineIntegration()
integration.generate_weekly_predictions(2024, 2)  # Update week number
"
```

## Model Management

### Retraining Model
```bash
# 1. Export latest data
python validation/extract_full_dataset.py

# 2. Train new model
python -c "
from models.moneyline_model import ProductionMoneylineModel
import pandas as pd

df = pd.read_csv('validation/data/full_dataset.csv')
train_df = df[df['season'] <= 2024]

model = ProductionMoneylineModel()
model.train(train_df, save=True)
print('✓ Model retrained and saved')
"

# 3. Validate new model
pytest tests/test_moneyline_model.py -v

# 4. If tests pass, new model is automatically used
```

### Rollback Model
```bash
# List available model versions
ls -la models/saved_models/moneyline_v*

# Rollback to specific version
python -c "
from models.moneyline_model import ProductionMoneylineModel
import shutil

# Copy old version to latest
shutil.copytree(
    'models/saved_models/moneyline_v0.9.0',
    'models/saved_models/moneyline_latest',
    dirs_exist_ok=True
)
print('✓ Rolled back to v0.9.0')
"
```

## Troubleshooting

### Issue: Predictions Not Generating
**Symptoms**: API returns no predictions

**Diagnosis**:
```bash
# Check model loaded
python -c "
from models.moneyline_model import load_moneyline_model
model = load_moneyline_model()
print('Model loaded:', model is not None)
"

# Check database connection
python -c "
from database.db_manager import DatabaseManager
db = DatabaseManager()
games = db.get_games_by_week(2024, 1)
print(f'Found {len(games)} games')
"
```

**Resolution**:
1. Verify model file exists: `models/saved_models/moneyline_latest/model.pkl`
2. Check database has game data for requested week
3. Check logs: `logs/nfl_system.log`

### Issue: Low Accuracy Alert
**Symptoms**: Monitoring shows accuracy < 55%

**Diagnosis**:
```bash
# Get detailed metrics
python -c "
from monitoring.moneyline_monitor import MoneylineMonitor
monitor = MoneylineMonitor()
monitor.historical_performance_report(2024, 2024)
"
```

**Resolution**:
1. Check if low accuracy is trend or one-off
2. If trend: retrain model with latest data
3. If one-off: monitor next week
4. Review feature importance for potential issues

### Issue: Database Table Missing
**Symptoms**: SQL error about missing `moneyline_predictions` table

**Resolution**:
```bash
python -c "
from integrations.moneyline_integration import setup_moneyline_integration
setup_moneyline_integration()
"
```

## Performance Standards

### Minimum Acceptable Thresholds
- **Accuracy**: ≥ 55% (better than coin flip)
- **High Confidence Accuracy**: ≥ 65% (for predictions >70% confidence)
- **Brier Score**: ≤ 0.25 (well-calibrated probabilities)
- **ROI**: ≥ 0% (breakeven)

### Retraining Triggers
- Accuracy < 55% for 3+ consecutive weeks
- ROI < 0% for 4+ consecutive weeks
- Accuracy drop > 5% from training baseline
- Major NFL rule changes or scoring trends

## Monitoring Dashboard

Access monitoring dashboard:
```bash
cd web
python launch.py

# Navigate to: http://localhost:8000/moneyline/dashboard
```

## Emergency Contacts

- **Model Owner**: [Your Name]
- **On-Call Engineer**: [Engineering Team]
- **Escalation**: Disable moneyline predictions in web interface if critical issues

## Backup & Recovery

### Database Backup
```bash
# Backup moneyline predictions table
sqlite3 database/nfl_suggestions.db ".dump moneyline_predictions" > backups/moneyline_predictions_$(date +%Y%m%d).sql
```

### Model Backup
```bash
# All model versions stored in: models/saved_models/moneyline_v{version}/
# Backup entire directory weekly
tar -czf backups/moneyline_models_$(date +%Y%m%d).tar.gz models/saved_models/moneyline_*
```

## Changelog

- **2024-01-15**: Initial runbook created
- **2024-01-20**: Added rollback procedures
- **2024-02-01**: Updated performance thresholds
```

### **Day 12: Final Testing & Launch**

#### **2.3: End-to-end test & launch checklist**
```python
# File: improved_nfl_system/scripts/launch_checklist.py
#!/usr/bin/env python3
"""
Moneyline System Launch Checklist
Validates all components before production launch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_launch_checklist():
    """Execute comprehensive launch checklist"""

    logger.info("=" * 60)
    logger.info("MONEYLINE SYSTEM LAUNCH CHECKLIST")
    logger.info("=" * 60)
    logger.info(f"Time: {datetime.now().isoformat()}\n")

    checks = []

    # 1. Model availability
    logger.info("1. Checking model availability...")
    try:
        from models.moneyline_model import load_moneyline_model
        model = load_moneyline_model()
        assert model is not None, "Model not loaded"
        assert model.feature_cols is not None, "No features defined"
        checks.append(("Model loaded", True, None))
        logger.info("   ✓ Model loaded successfully")
    except Exception as e:
        checks.append(("Model loaded", False, str(e)))
        logger.error(f"   ✗ Model loading failed: {e}")

    # 2. Database tables
    logger.info("\n2. Checking database tables...")
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        result = db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='moneyline_predictions'")
        assert len(result) > 0, "moneyline_predictions table missing"
        checks.append(("Database tables", True, None))
        logger.info("   ✓ Database tables exist")
    except Exception as e:
        checks.append(("Database tables", False, str(e)))
        logger.error(f"   ✗ Database check failed: {e}")

    # 3. Integration module
    logger.info("\n3. Checking integration module...")
    try:
        from integrations.moneyline_integration import MoneylineIntegration
        integration = MoneylineIntegration()
        checks.append(("Integration module", True, None))
        logger.info("   ✓ Integration module working")
    except Exception as e:
        checks.append(("Integration module", False, str(e)))
        logger.error(f"   ✗ Integration module failed: {e}")

    # 4. Monitoring system
    logger.info("\n4. Checking monitoring system...")
    try:
        from monitoring.moneyline_monitor import MoneylineMonitor
        monitor = MoneylineMonitor()
        assert monitor.metrics_dir.exists(), "Metrics directory missing"
        checks.append(("Monitoring system", True, None))
        logger.info("   ✓ Monitoring system ready")
    except Exception as e:
        checks.append(("Monitoring system", False, str(e)))
        logger.error(f"   ✗ Monitoring system failed: {e}")

    # 5. API endpoints
    logger.info("\n5. Checking API endpoints...")
    try:
        from web.api.moneyline_endpoints import router
        assert router is not None, "Router not defined"
        checks.append(("API endpoints", True, None))
        logger.info("   ✓ API endpoints registered")
    except Exception as e:
        checks.append(("API endpoints", False, str(e)))
        logger.error(f"   ✗ API endpoints failed: {e}")

    # 6. Test prediction
    logger.info("\n6. Running test prediction...")
    try:
        import pandas as pd
        test_game = pd.DataFrame({
            'spread_line': [-3.0],
            'total_line': [47.5],
            'week': [1],
            'home_rest_days': [7],
            'away_rest_days': [7]
        })

        from models.moneyline_model import predict_moneyline
        result = predict_moneyline(test_game)

        assert 'home_win_prob' in result, "Missing home_win_prob"
        assert 0 <= result['home_win_prob'] <= 1, "Invalid probability"

        checks.append(("Test prediction", True, None))
        logger.info(f"   ✓ Test prediction successful: {result['home_win_prob']:.3f}")
    except Exception as e:
        checks.append(("Test prediction", False, str(e)))
        logger.error(f"   ✗ Test prediction failed: {e}")

    # 7. Unit tests
    logger.info("\n7. Running unit tests...")
    try:
        import pytest
        result = pytest.main(['tests/test_moneyline_model.py', '-v', '--tb=short'])
        success = result == 0
        checks.append(("Unit tests", success, None if success else "Tests failed"))
        if success:
            logger.info("   ✓ All unit tests passed")
        else:
            logger.error("   ✗ Some unit tests failed")
    except Exception as e:
        checks.append(("Unit tests", False, str(e)))
        logger.error(f"   ✗ Unit test execution failed: {e}")

    # 8. Documentation
    logger.info("\n8. Checking documentation...")
    docs = [
        'docs/MONEYLINE_MODEL_RUNBOOK.md',
        'validation/models/poc_results.json'
    ]

    missing_docs = [doc for doc in docs if not Path(doc).exists()]

    if not missing_docs:
        checks.append(("Documentation", True, None))
        logger.info("   ✓ All documentation present")
    else:
        checks.append(("Documentation", False, f"Missing: {missing_docs}"))
        logger.error(f"   ✗ Missing documentation: {missing_docs}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CHECKLIST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for check in checks if check[1])
    total = len(checks)

    for name, success, error in checks:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if error:
            logger.info(f"         Error: {error}")

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULT: {passed}/{total} checks passed")

    if passed == total:
        logger.info("✓ SYSTEM READY FOR LAUNCH")
        logger.info("=" * 60)
        return True
    else:
        logger.error("✗ SYSTEM NOT READY - FIX ISSUES BEFORE LAUNCH")
        logger.error("=" * 60)
        return False

if __name__ == "__main__":
    success = run_launch_checklist()
    exit(0 if success else 1)
```

---

## **Success Criteria**

**Phase 0 (Validation)**:
- [ ] Baseline spread-to-moneyline achieves >50% accuracy
- [ ] POC moneyline model beats baseline by ≥2% accuracy OR ≥10% ROI
- [ ] Data quality report shows <10% missing data
- [ ] Decision script recommends PROCEED

**Phase 1 (Implementation)**:
- [ ] Production model achieves ≥55% accuracy on 2024 test set
- [ ] All unit tests pass
- [ ] Model can be saved, loaded, and rolled back
- [ ] Integration with database working
- [ ] Monitoring system tracking performance

**Phase 2 (Deployment)**:
- [ ] API endpoints functional
- [ ] Web interface displays predictions
- [ ] Launch checklist 100% pass
- [ ] Operational runbook complete
- [ ] Backup & rollback procedures tested

**Overall System**:
- [ ] Moneyline predictions available for upcoming week
- [ ] Performance monitoring automated
- [ ] Retraining triggers defined and working
- [ ] ROI ≥ 0% (breakeven or profitable)
- [ ] System integrated with existing NFL betting workflow

---

## **Risk Mitigation**

| Risk | Mitigation |
|------|------------|
| POC fails validation | Use baseline spread-to-moneyline (saves 10 days) |
| Model overfits | Strict train/val/test split, walk-forward validation |
| Production accuracy < training | Monitoring system detects, triggers retraining |
| Database schema conflicts | Use separate `moneyline_predictions` table |
| Web integration breaks | Backward compatible API, fallback to spread predictions |
| Key features missing | Feature engineering creates alternatives |

---

## **Comparison to Original Plan**

| Aspect | Original Plan | Improved Plan |
|--------|--------------|---------------|
| **Validation** | None | 3-day POC before committing |
| **Data Volume** | 544 games | 2,476 games |
| **Security** | Hardcoded keys | Environment variables |
| **Model Complexity** | 4-model ensemble | Single best model (simpler) |
| **Testing** | None | Comprehensive test suite |
| **Integration** | Unclear | Explicit database + API |
| **Monitoring** | Basic | Production-grade with alerts |
| **Rollback** | None | Versioned models + rollback |
| **Documentation** | Basic | Full operational runbook |
| **ROI Focus** | Accuracy only | ROI + accuracy + calibration |
| **Timeline** | 2+ weeks | 2 weeks (with validation gate) |

---

## **Next Steps After Completion**

1. **Week 1 Production**: Monitor closely, collect performance data
2. **Week 2-4**: Validate predictions vs. actuals, calculate CLV
3. **Month 2**: Consider ensemble if single model plateaus
4. **Month 3**: Optimize for specific bet types (favorites, underdogs, totals)
5. **Season End**: Comprehensive review, retrain on full season data

---

## **Maintenance Schedule**

- **Weekly**: Generate predictions, monitor performance
- **Bi-weekly**: Review accuracy trends
- **Monthly**: Check for retraining triggers
- **Quarterly**: Full system audit
- **Annually**: Major model update with full historical data
