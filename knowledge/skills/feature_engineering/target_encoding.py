"""Target encoding for categorical features.

Safe target encoder with smoothing and CV-aware fitting.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with smoothing to prevent overfitting.

    Parameters:
    -----------
    smoothing : float
        Smoothing factor (higher = more regularization)
    min_samples_leaf : int
        Minimum samples per category
    noise : float
        Amount of noise to add (for regularization)

    Usage:
    ------
    >>> encoder = TargetEncoder(smoothing=10.0)
    >>> X_train_encoded = encoder.fit_transform(X_train, y_train)
    >>> X_test_encoded = encoder.transform(X_test)
    """

    def __init__(self, smoothing=1.0, min_samples_leaf=1, noise=0.0):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise = noise
        self.encodings = {}
        self.global_mean = None

    def fit(self, X, y):
        """Fit the encoder.

        Parameters:
        -----------
        X : pd.DataFrame or pd.Series
            Categorical features
        y : pd.Series
            Target variable
        """
        # Handle Series input
        if isinstance(X, pd.Series):
            X = X.to_frame()

        self.global_mean = y.mean()

        for col in X.columns:
            # Calculate category statistics
            stats = y.groupby(X[col]).agg(['mean', 'count'])

            # Apply smoothing
            # Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smooth = (
                stats['count'] * stats['mean'] + self.smoothing * self.global_mean
            ) / (stats['count'] + self.smoothing)

            self.encodings[col] = smooth.to_dict()

        return self

    def transform(self, X):
        """Transform categorical features.

        Parameters:
        -----------
        X : pd.DataFrame or pd.Series
            Categorical features

        Returns:
        --------
        pd.DataFrame
            Encoded features
        """
        # Handle Series input
        if isinstance(X, pd.Series):
            X = X.to_frame()

        X_transformed = X.copy()

        for col in X.columns:
            # Map categories to encoded values
            # Unknown categories get global mean
            encoded = X[col].map(self.encodings.get(col, {}))

            # Add noise if specified
            if self.noise > 0:
                encoded = encoded + np.random.normal(0, self.noise, len(X))

            X_transformed[col] = encoded.fillna(self.global_mean)

        return X_transformed

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def leave_one_out_encode(df, col, target, smoothing=1.0):
    """Leave-one-out target encoding (no leakage within training data).

    Each row is encoded using the mean of all OTHER rows with the same category.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    col : str
        Column to encode
    target : str
        Target column name
    smoothing : float
        Smoothing factor

    Returns:
    --------
    pd.Series
        Encoded values
    """
    global_mean = df[target].mean()

    # Calculate sums and counts for each category
    category_stats = df.groupby(col)[target].agg(['sum', 'count'])

    # For each row, calculate leave-one-out mean
    # (total sum - this row's target) / (total count - 1)
    row_sums = df[col].map(category_stats['sum'])
    row_counts = df[col].map(category_stats['count'])

    loo_sums = row_sums - df[target]
    loo_counts = row_counts - 1

    # Apply smoothing
    encoded = (loo_sums + smoothing * global_mean) / (loo_counts + smoothing)

    return encoded.fillna(global_mean)


# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'B', 'C', 'C', 'A', 'B'],
        'target': [1, 0, 1, 1, 0, 0, 1, 0]
    })

    # Regular target encoding
    encoder = TargetEncoder(smoothing=1.0)
    encoded = encoder.fit_transform(df[['category']], df['target'])
    print("Target encoded:")
    print(encoded)

    # Leave-one-out encoding
    loo_encoded = leave_one_out_encode(df, 'category', 'target')
    print("\nLeave-one-out encoded:")
    print(loo_encoded)
