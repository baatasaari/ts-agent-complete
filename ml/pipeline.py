"""
ts_agent.ml.pipeline
====================
ML pipeline components for segment prediction.

Contains preprocessing, model building, and training utilities
extracted from predictor.py for better modularity.
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from ts_agent.ml.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from ts_agent.domain.models import ModelAlgorithm


def build_preprocessor() -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.
    
    Returns:
        ColumnTransformer configured for segment prediction features
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(unknown_value=-1, encoded_missing_value=-1)),
    ])
    
    return ColumnTransformer([
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_model(algorithm: ModelAlgorithm, calibrated: bool = True) -> Pipeline:
    """
    Build ML model pipeline.
    
    Args:
        algorithm: Which algorithm to use
        calibrated: Whether to apply probability calibration
        
    Returns:
        Configured sklearn Pipeline
    """
    preprocessor = build_preprocessor()
    
    # Select base estimator
    if algorithm == ModelAlgorithm.LOGISTIC_REGRESSION:
        estimator = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
    elif algorithm == ModelAlgorithm.RANDOM_FOREST:
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
    elif algorithm == ModelAlgorithm.GRADIENT_BOOSTING:
        estimator = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Apply calibration if requested
    if calibrated:
        estimator = CalibratedClassifierCV(estimator, cv=3)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator),
    ])


def extract_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """
    Extract feature importance from trained pipeline.
    
    Args:
        pipeline: Trained sklearn pipeline
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    classifier = pipeline.named_steps['classifier']
    
    # Handle calibrated classifiers
    if hasattr(classifier, 'base_estimator'):
        base_estimator = classifier.base_estimator
    else:
        base_estimator = classifier
    
    # Extract importance based on model type
    if hasattr(base_estimator, 'feature_importances_'):
        importance = base_estimator.feature_importances_
    elif hasattr(base_estimator, 'coef_'):
        # For logistic regression, use absolute coefficient values
        importance = abs(base_estimator.coef_[0])
    else:
        # Fallback to uniform importance
        importance = [1.0] * len(feature_names)
    
    return dict(zip(feature_names, importance))


def validate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """
    Validate pipeline performance.
    
    Args:
        pipeline: Trained pipeline
        X: Feature matrix
        y: Target labels
        
    Returns:
        Dictionary of validation metrics
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Cross-validation scores
    cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    
    # Full dataset scores
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    
    metrics = {
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(set(y)),
    }
    
    return metrics