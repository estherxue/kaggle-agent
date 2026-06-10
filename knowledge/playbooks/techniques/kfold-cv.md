---
name: Stratified K-Fold Cross Validation
applicable_types: [tabular]
---

## Applicable Conditions

Use stratified k-fold CV when:
- Classification problems
- Imbalanced classes
- Need for stable CV scores
- Dataset is not time-dependent

**Don't use when:**
- Time series data (use time-based split)
- Data has group structure (use GroupKFold)
- Extreme class imbalance (< 1% minority)

## Usage

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

class StratifiedKFoldTrainer:
    """Generic stratified k-fold trainer."""

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def cross_validate(self, model_factory, X, y):
        """Run cross-validation.

        Args:
            model_factory: Function that returns a fresh model
            X: Features
            y: Target

        Returns:
            Dict with scores and OOF predictions
        """
        oof_preds = np.zeros(len(X))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            self.skf.split(X, y)
        ):
            print(f"Fold {fold + 1}/{self.n_splits}")

            # Split data
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model = model_factory()
            model.fit(X_tr, y_tr)

            # Predict
            preds = model.predict(X_val)
            oof_preds[val_idx] = preds

            # Score
            score = self._score(y_val, preds)
            fold_scores.append(score)
            print(f"  Score: {score:.4f}")

        # Overall score
        overall_score = self._score(y, oof_preds)

        return {
            'oof_preds': oof_preds,
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'overall_score': overall_score,
        }

    def _score(self, y_true, y_pred):
        """Calculate score - override for custom metrics."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)


# Usage with LightGBM
import lightgbm as lgb

def make_lgb_model():
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
    )

trainer = StratifiedKFoldTrainer(n_splits=5)
results = trainer.cross_validate(make_lgb_model, X, y)
print(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
```

## Validation

- competition: titanic
  date: 2024-03-01
  cv_improvement: N/A (baseline)
  notes: Standard for classification

- competition: santander
  date: 2024-04-10
  cv_improvement: +0.005
  notes: 10-fold gave more stable scores

## Tips

### Number of Folds

- **5 folds**: Standard, good balance of speed/stability
- **10 folds**: More stable, slower, better for small datasets
- **3 folds**: Faster, less stable, large datasets only

### Shuffle

- Always shuffle with `random_state` for reproducibility
- Exception: Time-dependent data

### Regression

For regression, use regular KFold:

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

### Early Stopping Integration

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.train(
        params,
        train_set=lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx]),
        valid_sets=[lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
```
