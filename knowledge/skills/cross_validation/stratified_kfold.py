"""Stratified K-Fold cross-validation wrapper.

Simplified interface for common CV patterns with integrated early stopping.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Callable, Dict, List, Any


class StratifiedKFoldCV:
    """Stratified K-Fold CV with built-in early stopping support.

    Simplifies the CV loop boilerplate for LightGBM, XGBoost, etc.

    Example:
    --------
    >>> cv = StratifiedKFoldCV(n_splits=5, random_state=42)
    >>>
    >>> def make_model():
    ...     return lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05)
    >>>
    >>> results = cv.cross_validate(
    ...     model_factory=make_model,
    ...     X=X,
    ...     y=y,
    ...     fit_params={'early_stopping_rounds': 50}
    ... )
    >>> print(f"CV Score: {results['cv_score']:.4f}")
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def cross_validate(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        fit_params: Dict[str, Any] = None,
        predict_proba: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run cross-validation.

        Parameters:
        -----------
        model_factory : callable
            Function that returns a fresh model instance
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        fit_params : dict
            Additional parameters for model.fit()
        predict_proba : bool
            Use predict_proba for predictions (for classifiers)
        verbose : bool
            Print fold progress

        Returns:
        --------
        dict
            Results including OOF predictions, fold scores, and statistics
        """
        oof_preds = np.zeros(len(X))
        fold_scores = []
        models = []

        fit_params = fit_params or {}

        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            if verbose:
                print(f"Fold {fold + 1}/{self.n_splits}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create fresh model
            model = model_factory()

            # Fit model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train, **fit_params)

            # Predict
            if predict_proba and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)

            oof_preds[val_idx] = preds
            models.append(model)

            # Calculate fold score (will be filled by user based on metric)
            fold_scores.append({
                'fold': fold,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
            })

        # Compile results
        results = {
            'oof_predictions': oof_preds,
            'fold_scores': fold_scores,
            'fold_models': models,
            'cv_indices': list(self.skf.split(X, y)),
        }

        return results

    def cross_validate_with_metric(
        self,
        model_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        metric_fn: Callable,
        fit_params: Dict[str, Any] = None,
        predict_proba: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run cross-validation with custom metric.

        Parameters:
        -----------
        metric_fn : callable
            Function(y_true, y_pred) -> score
            Higher score must be better

        Returns:
        --------
        dict
            Results with detailed scores
        """
        oof_preds = np.zeros(len(X))
        fold_scores = []
        models = []

        fit_params = fit_params or {}

        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            if verbose:
                print(f"Fold {fold + 1}/{self.n_splits}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create fresh model
            model = model_factory()

            # Fit
            model.fit(X_train, y_train, **fit_params)

            # Predict
            if predict_proba and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)

            # Score
            score = metric_fn(y_val, preds)
            fold_scores.append(score)

            if verbose:
                print(f"  Score: {score:.4f}")

            oof_preds[val_idx] = preds
            models.append(model)

        # Overall score
        overall_score = metric_fn(y, oof_preds)

        results = {
            'oof_predictions': oof_preds,
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'overall_score': overall_score,
            'fold_models': models,
        }

        if verbose:
            print(f"\nCV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            print(f"OOF Score: {overall_score:.4f}")

        return results


def get_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> List[tuple]:
    """Get train/val indices for stratified k-fold.

    Simplified function for manual CV loops.

    Returns:
    --------
    list of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    return list(skf.split(X, y))


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
    y = pd.Series(y)

    # Define model factory
    def make_rf():
        return RandomForestClassifier(n_estimators=100, random_state=42)

    # Run CV
    cv = StratifiedKFoldCV(n_splits=5)
    results = cv.cross_validate_with_metric(
        model_factory=make_rf,
        X=X,
        y=y,
        metric_fn=accuracy_score,
        predict_proba=False,
    )
