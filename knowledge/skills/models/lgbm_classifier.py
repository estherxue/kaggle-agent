"""LightGBM classifier configuration and utilities.

Recommended starting point for tabular classification.
"""

import lightgbm as lgb
import numpy as np
from typing import Dict, Any, Optional


class LGBMClassifierConfig:
    """Configuration for LightGBM classifier.

    Provides reasonable defaults for different dataset sizes.
    """

    # Small dataset (< 10K samples)
    SMALL = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Lower to prevent overfitting
        'max_depth': 5,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
    }

    # Medium dataset (10K - 100K samples)
    MEDIUM = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'verbose': -1,
    }

    # Large dataset (> 100K samples)
    LARGE = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': -1,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 10,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'verbose': -1,
    }

    @classmethod
    def get_params_for_dataset_size(cls, n_samples: int) -> Dict[str, Any]:
        """Get recommended params based on dataset size."""
        if n_samples < 10000:
            return cls.SMALL.copy()
        elif n_samples < 100000:
            return cls.MEDIUM.copy()
        else:
            return cls.LARGE.copy()


def create_lgbm_classifier(
    n_samples: int,
    class_weights: Optional[Dict[int, float]] = None,
    custom_params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMClassifier:
    """Create a configured LightGBM classifier.

    Parameters:
    -----------
    n_samples : int
        Number of training samples
    class_weights : dict or 'balanced'
        Class weights for imbalanced problems
    custom_params : dict
        Override any default parameters

    Returns:
    --------
    lgb.LGBMClassifier
        Configured classifier
    """
    params = LGBMClassifierConfig.get_params_for_dataset_size(n_samples)

    if custom_params:
        params.update(custom_params)

    classifier = lgb.LGBMClassifier(
        n_estimators=10000,  # Will use early stopping
        **params
    )

    if class_weights:
        classifier.set_params(class_weight=class_weights)

    return classifier


def train_with_early_stopping(
    model: lgb.LGBMClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
    early_stopping_rounds: int = 50,
    verbose: bool = False
) -> lgb.LGBMClassifier:
    """Train LightGBM model with early stopping.

    Parameters:
    -----------
    model : LGBMClassifier
        Model to train
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    early_stopping_rounds : int
        Patience for early stopping
    verbose : bool
        Print training progress

    Returns:
    --------
    Trained model
    """
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=verbose),
            lgb.log_evaluation(period=100 if verbose else 0)
        ]
    )

    if verbose:
        print(f"Best iteration: {model.best_iteration_}")
        print(f"Best score: {model.best_score_}")

    return model


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Sample data
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create model
    model = create_lgbm_classifier(
        n_samples=len(X_train),
        custom_params={'learning_rate': 0.05}
    )

    # Train
    model = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val,
        early_stopping_rounds=50,
        verbose=True
    )

    print(f"\nTraining complete. Used {model.best_iteration_} iterations.")
