---
name: Target Encoding
applicable_types: [tabular]
---

## Applicable Conditions

Target encoding is effective when:
- High cardinality categorical features (>10 unique values)
- Tree-based models where OHE creates too many features
- Linear models where categorical relationships matter
- Dataset is large enough (>10K samples) to avoid overfitting

**Avoid when:**
- Low cardinality categories (< 5) - use OHE instead
- Very small datasets - high overfitting risk
- Time series with future target leakage risk
- Categories that appear in test but not train

## Usage

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with smoothing."""

    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings = {}
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = y.mean()

        for col in X.columns:
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            # Smoothing: weighted average between category mean and global mean
            smooth = (
                stats['count'] * stats['mean'] + self.smoothing * self.global_mean
            ) / (stats['count'] + self.smoothing)
            self.encodings[col] = smooth

        return self

    def transform(self, X):
        X_transformed = X.copy()

        for col in X.columns:
            # Map categories to encoded values
            # Unknown categories get global mean
            X_transformed[col] = X[col].map(
                self.encodings[col]
            ).fillna(self.global_mean)

        return X_transformed

# Usage example
encoder = TargetEncoder(smoothing=10.0)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)
```

## Validation

- competition: house-prices
  date: 2024-01-15
  cv_improvement: +0.002
  notes: Best for Neighborhood feature (25 categories)

- competition: porto-seguro
  date: 2024-02-20
  cv_improvement: +0.001
  notes: Minimal improvement, data already numeric

## Variants

### Leave-One-Out Encoding

Use during CV to prevent leakage:

```python
def leave_one_out_encode(df, col, target):
    """Encode each row using all other rows' target mean."""
    global_mean = target.mean()

    # Calculate leave-one-out means
    sums = df.groupby(col)[target].transform('sum') - target
    counts = df.groupby(col)[target].transform('count') - 1

    return (sums / counts).fillna(global_mean)
```

### Regularized per Fold

In CV, compute encoding using only training fold:

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    encoder = TargetEncoder()
    encoder.fit(X.iloc[train_idx], y.iloc[train_idx])
    X_val_encoded = encoder.transform(X.iloc[val_idx])
    # Use X_val_encoded for validation predictions
```

## Warnings

- **Data Leakage Risk**: Never fit on full dataset before CV
- **Unknown Categories**: Always handle categories in test but not train
- **Smoothing**: Higher smoothing for smaller datasets
