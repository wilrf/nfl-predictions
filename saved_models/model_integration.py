"""
Model Integration Module
Wraps existing XGBoost models for predictions
Uses isotonic calibration for probability adjustment
FAIL FAST - No fallback models
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Custom exception for model operations"""
    pass


class NFLModelEnsemble:
    """
    Integrates existing XGBoost models
    REAL DATA ONLY - no synthetic features
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize model ensemble

        Args:
            models_dir: Directory containing saved models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / 'saved_models'

        self.models_dir = Path(models_dir)
        self.models = {}
        self.calibrators = {}

        # Load models - FAIL if not found
        self._load_models()

    def _load_models(self):
        """Load XGBoost models from disk - FAIL FAST if missing"""
        required_models = ['spread', 'total']  # Minimum required

        for model_name in required_models:
            model_path = self.models_dir / f'{model_name}_model.pkl'

            if not model_path.exists():
                raise ModelError(f"Required model not found: {model_path}")

            try:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name} model")
            except Exception as e:
                raise ModelError(f"Failed to load {model_name} model: {e}")

            # Load calibrator if available
            calibrator_path = self.models_dir / f'{model_name}_calibrator.pkl'
            if calibrator_path.exists():
                try:
                    with open(calibrator_path, 'rb') as f:
                        self.calibrators[model_name] = pickle.load(f)
                    logger.info(f"Loaded {model_name} calibrator")
                except:
                    # Calibrator is optional
                    logger.warning(f"Could not load calibrator for {model_name}")

    def predict_spread(self, features: pd.DataFrame) -> Dict:
        """
        Predict spread outcome

        Args:
            features: Feature dataframe with REAL game data

        Returns:
            Dict with predictions and probabilities
        """
        if features.empty:
            raise ModelError("Empty features dataframe")

        if 'spread' not in self.models:
            raise ModelError("Spread model not loaded")

        try:
            # Get raw predictions
            model = self.models['spread']
            raw_proba = model.predict_proba(features)

            # Apply calibration if available
            if 'spread' in self.calibrators:
                calibrated_proba = self.calibrators['spread'].transform(raw_proba[:, 1])
                home_win_prob = calibrated_proba[0]
            else:
                home_win_prob = raw_proba[0, 1]

            # Get predicted spread
            predicted_spread = model.predict(features)[0]

            return {
                'home_win_prob': float(home_win_prob),
                'away_win_prob': float(1 - home_win_prob),
                'predicted_spread': float(predicted_spread),
                'model_confidence': self._calculate_model_confidence(raw_proba[0])
            }

        except Exception as e:
            raise ModelError(f"Spread prediction failed: {e}")

    def predict_total(self, features: pd.DataFrame) -> Dict:
        """
        Predict total points outcome

        Args:
            features: Feature dataframe with REAL game data

        Returns:
            Dict with predictions and probabilities
        """
        if features.empty:
            raise ModelError("Empty features dataframe")

        if 'total' not in self.models:
            raise ModelError("Total model not loaded")

        try:
            # Get raw predictions
            model = self.models['total']
            raw_proba = model.predict_proba(features)

            # Apply calibration if available
            if 'total' in self.calibrators:
                calibrated_proba = self.calibrators['total'].transform(raw_proba[:, 1])
                over_prob = calibrated_proba[0]
            else:
                over_prob = raw_proba[0, 1]

            # Get predicted total
            predicted_total = model.predict(features)[0]

            return {
                'over_prob': float(over_prob),
                'under_prob': float(1 - over_prob),
                'predicted_total': float(predicted_total),
                'model_confidence': self._calculate_model_confidence(raw_proba[0])
            }

        except Exception as e:
            raise ModelError(f"Total prediction failed: {e}")

    def _calculate_model_confidence(self, probabilities: np.ndarray) -> float:
        """
        Calculate model's confidence in its prediction

        Args:
            probabilities: Probability array from model

        Returns:
            Confidence score (0-1)
        """
        # Higher probability difference = higher confidence
        max_prob = max(probabilities)
        min_prob = min(probabilities)
        prob_diff = max_prob - min_prob

        # Map to 0-1 scale
        # 50/50 = 0 confidence, 100/0 = 1 confidence
        confidence = prob_diff

        return round(confidence, 3)

    def create_calibrator(self, model_name: str, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Create isotonic calibrator for a model

        Args:
            model_name: Name of model to calibrate
            X_val: Validation features
            y_val: Validation targets
        """
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not loaded")

        try:
            model = self.models[model_name]

            # Get raw predictions
            raw_proba = model.predict_proba(X_val)[:, 1]

            # Fit isotonic calibrator
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(raw_proba, y_val)

            self.calibrators[model_name] = calibrator

            # Save calibrator
            calibrator_path = self.models_dir / f'{model_name}_calibrator.pkl'
            with open(calibrator_path, 'wb') as f:
                pickle.dump(calibrator, f)

            logger.info(f"Created calibrator for {model_name}")

        except Exception as e:
            raise ModelError(f"Failed to create calibrator: {e}")

    def check_calibration_quality(self, model_name: str,
                                 X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Check how well calibrated the model is

        Args:
            model_name: Model to check
            X_test: Test features
            y_test: Test targets

        Returns:
            Calibration error (lower is better)
        """
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not loaded")

        try:
            model = self.models[model_name]

            # Get predictions
            if model_name in self.calibrators:
                raw_proba = model.predict_proba(X_test)[:, 1]
                predictions = self.calibrators[model_name].transform(raw_proba)
            else:
                predictions = model.predict_proba(X_test)[:, 1]

            # Calculate calibration error
            # Group predictions into bins
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(predictions, bin_edges) - 1

            calibration_error = 0
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() > 0:
                    predicted_prob = predictions[mask].mean()
                    actual_prob = y_test[mask].mean()
                    calibration_error += abs(predicted_prob - actual_prob) * mask.sum()

            calibration_error /= len(predictions)

            return round(calibration_error, 4)

        except Exception as e:
            raise ModelError(f"Failed to check calibration: {e}")

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """
        Get feature importance from model

        Args:
            model_name: Model name

        Returns:
            Dict of feature importance scores
        """
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            feature_names = model.feature_names_ if hasattr(model, 'feature_names_') else None
            importances = model.feature_importances_

            if feature_names:
                return dict(zip(feature_names, importances))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        else:
            return {}

    def validate_features(self, features: pd.DataFrame, model_name: str) -> bool:
        """
        Validate that features match model expectations

        Args:
            features: Feature dataframe
            model_name: Model to validate against

        Returns:
            True if valid
        """
        if model_name not in self.models:
            raise ModelError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        # Check if model has expected feature names
        if hasattr(model, 'feature_names_'):
            expected_features = model.feature_names_
            actual_features = features.columns.tolist()

            missing = set(expected_features) - set(actual_features)
            if missing:
                raise ModelError(f"Missing required features: {missing}")

            extra = set(actual_features) - set(expected_features)
            if extra:
                logger.warning(f"Extra features will be ignored: {extra}")

        # Check for NaN values
        if features.isnull().any().any():
            raise ModelError("Features contain NaN values")

        # Check for infinite values
        if np.isinf(features.values).any():
            raise ModelError("Features contain infinite values")

        return True