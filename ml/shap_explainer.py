entering """
ts_agent.ml.shap_explainer
===========================
SHAP-based explainability components for ML segment prediction.

Provides SHAP analysis and feature importance calculations
extracted from predictor.py for better modularity.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ts_agent.domain.models import ShapFeature
from ts_agent.ml.feature_engineering import ALL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Lazy import SHAP to handle missing dependency gracefully
_SHAP_AVAILABLE = False
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None


class ShapExplainer:
    """
    SHAP-based explainer for segment prediction models.
    
    Provides feature importance and explanation capabilities
    with graceful fallback when SHAP is not available.
    """
    
    def __init__(self, pipeline: Pipeline, X_train: pd.DataFrame, max_evals: int = 100):
        """
        Initialize SHAP explainer.
        
        Args:
            pipeline: Trained sklearn pipeline
            X_train: Training data for background
            max_evals: Maximum evaluations for SHAP
        """
        self.pipeline = pipeline
        self.X_train = X_train
        self.max_evals = max_evals
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self) -> None:
        """Set up SHAP explainer with appropriate method."""
        if not _SHAP_AVAILABLE:
            logger.warning("SHAP not available - using fallback explainer")
            return
        
        try:
            # Use TreeExplainer for tree-based models
            classifier = self.pipeline.named_steps['classifier']
            
            # Handle calibrated classifiers
            if hasattr(classifier, 'base_estimator'):
                base_estimator = classifier.base_estimator
            else:
                base_estimator = classifier
            
            # Check if it's a tree-based model
            if hasattr(base_estimator, 'estimators_') or hasattr(base_estimator, 'tree_'):
                # Transform training data through preprocessing pipeline
                X_transformed = self.pipeline.named_steps['preprocessor'].transform(self.X_train)
                self.explainer = shap.TreeExplainer(base_estimator, X_transformed[:100])
                logger.info("Using SHAP TreeExplainer")
            else:
                # Use general explainer for other models
                background = self.X_train.sample(min(50, len(self.X_train)))
                self.explainer = shap.KernelExplainer(
                    self.pipeline.predict_proba, 
                    background,
                    max_evals=self.max_evals
                )
                logger.info("Using SHAP KernelExplainer")
                
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, X: pd.DataFrame, max_features: int = 10) -> list[ShapFeature]:
        """
        Get SHAP explanations for predictions.
        
        Args:
            X: Input features
            max_features: Maximum number of features to return
            
        Returns:
            List of ShapFeature objects sorted by importance
        """
        if not self.explainer or not _SHAP_AVAILABLE:
            return self._fallback_explanation(X, max_features)
        
        try:
            if hasattr(self.explainer, 'shap_values'):
                # TreeExplainer
                X_transformed = self.pipeline.named_steps['preprocessor'].transform(X)
                shap_values = self.explainer.shap_values(X_transformed)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    # Use the class with highest probability
                    proba = self.pipeline.predict_proba(X)[0]
                    best_class = proba.argmax()
                    shap_values = shap_values[best_class]
                
                # Get values for first sample
                if len(shap_values.shape) > 1:
                    shap_values = shap_values[0]
                    
            else:
                # KernelExplainer
                shap_values = self.explainer.shap_values(X)[0]
            
            # Convert to ShapFeature objects
            features = []
            feature_names = ALL_FEATURE_COLUMNS
            
            for i, (feature, value) in enumerate(zip(feature_names, shap_values)):
                if abs(value) > 1e-6:  # Filter near-zero values
                    features.append(ShapFeature(
                        f=feature,
                        v=float(value),
                        r=i + 1
                    ))
            
            # Sort by absolute importance
            features.sort(key=lambda x: abs(x.v), reverse=True)
            return features[:max_features]
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(X, max_features)
    
    def _fallback_explanation(self, X: pd.DataFrame, max_features: int) -> list[ShapFeature]:
        """
        Fallback explanation when SHAP is unavailable.
        
        Uses feature importance from the trained model.
        """
        try:
            from ts_agent.ml.pipeline import extract_feature_importance
            
            importance = extract_feature_importance(self.pipeline, ALL_FEATURE_COLUMNS)
            
            # Create pseudo-SHAP values
            features = []
            for i, (feature, imp) in enumerate(importance.items()):
                if imp > 0:
                    # Get actual feature value
                    feature_value = X[feature].iloc[0] if feature in X.columns else 0.0
                    
                    # Create pseudo SHAP value (importance * normalized feature value)
                    shap_value = imp * (feature_value / (feature_value + 1) if feature_value >= 0 else -1)
                    
                    features.append(ShapFeature(
                        f=feature,
                        v=float(shap_value),
                        r=i + 1
                    ))
            
            # Sort by absolute importance
            features.sort(key=lambda x: abs(x.v), reverse=True)
            return features[:max_features]
            
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
            return []
    
    def get_feature_importance_matrix(self, X_sample: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Get feature importance matrix for multiple samples/segments.
        
        Args:
            X_sample: Sample data representing different segments
            
        Returns:
            Dictionary mapping segment_id to feature importance
        """
        if not self.explainer or not _SHAP_AVAILABLE:
            return self._fallback_importance_matrix(X_sample)
        
        try:
            importance_matrix = {}
            
            for idx, (_, row) in enumerate(X_sample.iterrows()):
                X_single = pd.DataFrame([row])
                shap_features = self.explain_prediction(X_single)
                
                segment_key = f"segment_{idx:03d}"
                importance_matrix[segment_key] = {
                    feat.f: abs(feat.v) for feat in shap_features
                }
            
            return importance_matrix
            
        except Exception as e:
            logger.warning(f"SHAP importance matrix failed: {e}")
            return self._fallback_importance_matrix(X_sample)
    
    def _fallback_importance_matrix(self, X_sample: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Fallback importance matrix using model feature importance."""
        try:
            from ts_agent.ml.pipeline import extract_feature_importance
            
            base_importance = extract_feature_importance(self.pipeline, ALL_FEATURE_COLUMNS)
            
            # Create uniform importance for all segments
            importance_matrix = {}
            for idx in range(len(X_sample)):
                segment_key = f"segment_{idx:03d}"
                importance_matrix[segment_key] = base_importance.copy()
            
            return importance_matrix
            
        except Exception as e:
            logger.error(f"Fallback importance matrix failed: {e}")
            return {}


def create_explainer(pipeline: Pipeline, X_train: pd.DataFrame, max_evals: int = 100) -> ShapExplainer:
    """
    Factory function to create SHAP explainer.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_train: Training data
        max_evals: Maximum evaluations for SHAP
        
    Returns:
        ShapExplainer instance
    """
    return ShapExplainer(pipeline, X_train, max_evals)